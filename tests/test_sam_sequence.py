"""Dependent stacking tasks through the real app, with no physical ROS clients."""
import asyncio
import threading

import pytest

from x5a_web_agent import app
from x5a_web_agent.session import Session
from x5a_web_agent.task_sequence import split_sequence


def pick(source, destination=None):
    prompt = f'{source} cube'
    if destination:
        prompt += f' | place: {destination} cube; prel: on'
    return dict(action='remote_grasp', prompt=prompt, object='cube', color=source)


class FakeBridge:
    busy = False
    dry_run = False

    def __init__(self):
        self.calls = []
        self.revision = 0
        self.current_prompt = ''
        self.fail_execute = False
        self.ready_ok = True

    def send_remote_grasp(self, prompt, **kwargs):
        self.calls.append(('plan', prompt, self.revision))
        self.current_prompt = prompt
        return dict(success=True, stage='SUCCESS', plan_valid=True)

    def send_remote_place(self, prompt, relation, **kwargs):
        return self.send_remote_grasp(f'place: {prompt}; prel: {relation}', **kwargs)

    def execute_latest(self):
        self.calls.append(('execute', self.current_prompt))
        placed = 'place:' in self.current_prompt
        if placed and not self.fail_execute:
            self.revision += 1
        return dict(success=not self.fail_execute, stage='FAILED' if self.fail_execute else 'SUCCESS',
                    message='PLACE failed' if self.fail_execute else '执行完成',
                    holding=True if self.fail_execute else not placed)

    def pose_ready(self):
        self.calls.append(('ready',))
        return dict(success=self.ready_ok, message='ready' if self.ready_ok else 'ready failed')

    def pose_home(self):
        self.calls.append(('home',))
        return dict(success=True)

    def release(self):
        self.calls.append(('release',))
        return dict(success=True, holding=False)

    def cancel(self):
        self.calls.append(('cancel',))
        return False


@pytest.fixture
def web(monkeypatch):
    monkeypatch.setattr(app, 'hub', app.ConnectionHub())
    monkeypatch.setattr(app, 'session', Session())
    monkeypatch.setattr(app, '_active_run', None)
    monkeypatch.setattr(app, '_command_number', 0)
    monkeypatch.setattr(app, 'bridge', FakeBridge())
    monkeypatch.setattr(app, '_needs_digital_human_handshake', lambda _: False)
    for name, value in [('X5A_GRASP_ROUTE', 'remote'), ('X5A_AUTO_EXECUTE', '1'),
                        ('X5A_CLOUD_PARSE', '1'), ('X5A_CARRY_TO_READY', '0')]:
        monkeypatch.setenv(name, value)
    return app


async def finish(web, tasks):
    accepted = await web.dispatch_parsed(dict(action='sequence', tasks=tasks))
    if accepted.get('ok'):
        for _ in range(1000):
            if web._active_run is None:
                break
            await asyncio.sleep(.001)
        assert web._active_run is None, 'fake task did not finish'
    return accepted


def test_exact_two_level_request_parses_each_clause_separately(web, monkeypatch):
    calls = []
    def cloud(text):
        calls.append(text)
        source, destination = ('white', 'orange') if '橙色' in text else ('red', 'white')
        return dict(object='cube', color=source, action='pick', sam_prompt=f'{source} cube',
                    place_prompt=f'{destination} cube', place_rel='on')
    monkeypatch.setattr(web, 'cloud_parse_text', cloud)
    parsed = asyncio.run(web._cloud_parse_command('把白色方块放到橙色方块上，然后把红色方块放到白色方块上'))
    assert parsed['action'] == 'sequence'
    assert len(calls) == 2 and all('然后' not in text for text in calls)
    assert [task['prompt'] for task in parsed['tasks']] == [pick('white', 'orange')['prompt'], pick('red', 'white')['prompt']]


@pytest.mark.parametrize('text,expected', [
    ('抓白色，然后放到橙色上，接着抓红色，再放到白色上', ['抓白色', '放到橙色上', '抓红色', '放到白色上']),
    ('先把白色放到橙色上，再把红色放到白色上', ['先把白色放到橙色上', '把红色放到白色上']),
    ('pick white then place on orange', ['pick white', 'place on orange']),
    ('抓白色和橙色之间的红色方块', None),
    ('pick an earthen cup', None),
])
def test_explicit_order_markers_preserve_targets(text, expected):
    assert split_sequence(text) == expected


@pytest.mark.parametrize('text', ['抓白色，然后', '然后'.join(['抓白色']*9)])
def test_incomplete_or_oversized_sequence_is_rejected(text):
    with pytest.raises(ValueError):
        split_sequence(text)


def test_invalid_second_clause_rejects_whole_command_before_motion(web, monkeypatch):
    monkeypatch.setattr(web, 'cloud_parse_text', lambda text:
        dict(object='cube', sam_prompt='white cube', action='pick') if '白色' in text else None)
    parsed = asyncio.run(web._cloud_parse_command('抓白色，然后做个无法理解的动作'))
    assert parsed['action'] == 'unknown' and '第 2 步' in parsed['message']
    assert web.bridge.calls == []


def test_two_placements_execute_in_order_after_observation_pose_and_scene_update(web):
    events = []
    web.hub.add_observer(events.append)
    result = asyncio.run(finish(web, [pick('white', 'orange'), pick('red', 'white')]))
    assert result['ok']
    assert [call[0] for call in web.bridge.calls] == ['plan', 'execute', 'ready', 'plan', 'execute']
    assert [call[2] for call in web.bridge.calls if call[0] == 'plan'] == [0, 1]
    assert web.hub.state['last_result']['steps_completed'] == 2
    assert web.hub.state['stage'] == 'SUCCESS' and not web.hub.state['busy']
    assert not web.session.holding
    # A successful first step must not temporarily accept another command.
    first = next(event for event in events if event.get('type') == 'result' and not event.get('sequence_summary'))
    assert first['success'] is True and first['step_index'] == 1
    assert all(event['busy'] for event in events if event.get('type') == 'state'
               and event.get('command_id') == 1 and event is not events[-1])


def test_failed_first_placement_stops_before_second_planning_and_preserves_holding(web):
    web.bridge.fail_execute = True
    asyncio.run(finish(web, [pick('white', 'orange'), pick('red', 'white')]))
    assert [call[0] for call in web.bridge.calls] == ['plan', 'execute']
    assert web.hub.state['stage'] == 'FAILED' and web.session.holding


def test_return_to_observation_failure_stops_before_second_scene_capture(web):
    web.bridge.ready_ok = False
    asyncio.run(finish(web, [pick('white', 'orange'), pick('red', 'white')]))
    assert [call[0] for call in web.bridge.calls] == ['plan', 'execute', 'ready']
    assert '返回观察位失败' in web.hub.state['last_result']['message']


def test_cancel_between_steps_remains_effective_while_bridge_is_idle(web):
    def observer(event):
        if event.get('type') == 'result' and event.get('step_index') == 1 and not event.get('sequence_summary'):
            assert web._request_cancel()['ok']
    web.hub.add_observer(observer)
    asyncio.run(finish(web, [pick('white', 'orange'), pick('red', 'white')]))
    assert [call[0] for call in web.bridge.calls] == ['plan', 'execute', 'cancel']
    assert web.hub.state['stage'] == 'CANCELED'


def test_cancel_after_planning_does_not_submit_execute(web):
    plan = web.bridge.send_remote_grasp
    def cancel_after_plan(*args, **kwargs):
        result = plan(*args, **kwargs)
        web._active_run.cancelled.set()
        return result
    web.bridge.send_remote_grasp = cancel_after_plan
    asyncio.run(finish(web, [pick('white', 'orange'), pick('red', 'white')]))
    assert [call[0] for call in web.bridge.calls] == ['plan']
    assert web.hub.state['stage'] == 'CANCELED'


def test_plan_only_mode_never_advances_to_dependent_second_step(web, monkeypatch):
    monkeypatch.setenv('X5A_AUTO_EXECUTE', '0')
    asyncio.run(finish(web, [pick('white', 'orange'), pick('red', 'white')]))
    assert [call[0] for call in web.bridge.calls] == ['plan']
    assert web.hub.state['stage'] == 'DRY_RUN'
    assert web.hub.state['last_result']['steps_completed'] == 0


def test_visualization_without_valid_plan_cannot_trigger_execution(web):
    plan = web.bridge.send_remote_grasp
    def no_plan(*args, **kwargs):
        return {**plan(*args, **kwargs), 'plan_valid': False}
    web.bridge.send_remote_grasp = no_plan
    asyncio.run(finish(web, [pick('white', 'orange'), pick('red', 'white')]))
    assert [call[0] for call in web.bridge.calls] == ['plan']
    assert web.hub.state['stage'] == 'FAILED'


def test_text_input_is_rejected_before_cloud_parsing_during_an_active_command(web, monkeypatch):
    monkeypatch.setattr(web, 'cloud_parse_text', lambda _: pytest.fail('busy input reached cloud parser'))
    web.hub.state['busy'] = True
    response = asyncio.run(web.handle_text_command('把白色方块放到橙色方块上'))
    assert not response['ok'] and web.bridge.calls == []


def test_consecutive_grasps_without_a_release_are_rejected_before_first_step(web):
    response = asyncio.run(finish(web, [pick('white'), pick('red')]))
    assert not response['ok'] and '仍在持物' in response['message']
    assert web.bridge.calls == []


def test_grasp_and_follow_up_place_sequence_uses_real_holding_state(web):
    tasks = [pick('white'), dict(action='remote_place', prompt='orange cube', relation='on'),
             pick('red'), dict(action='remote_place', prompt='white cube', relation='on')]
    assert asyncio.run(finish(web, tasks))['ok']
    assert len([call for call in web.bridge.calls if call[0] == 'execute']) == 4
    assert not web.session.holding


def test_follow_up_place_reuses_already_reached_observation_pose(web, monkeypatch):
    monkeypatch.setenv('X5A_CARRY_TO_READY', '1')
    tasks = [pick('white'), dict(action='remote_place', prompt='orange cube', relation='on')]
    assert asyncio.run(finish(web, tasks))['ok']
    assert [call[0] for call in web.bridge.calls] == ['plan', 'execute', 'ready', 'plan', 'execute']


def test_no_other_command_enters_during_post_grasp_ready_motion(web, monkeypatch):
    monkeypatch.setenv('X5A_CARRY_TO_READY', '1')
    entered, release = threading.Event(), threading.Event()
    def ready():
        entered.set()
        assert release.wait(2.)
        return dict(success=True)
    web.bridge.pose_ready = ready
    async def scenario():
        assert (await web.dispatch_parsed(pick('white')))['ok']
        try:
            for _ in range(1000):
                if entered.is_set():
                    break
                await asyncio.sleep(.001)
            assert entered.is_set() and web.hub.state['busy']
            rejected = await web.dispatch_parsed(dict(action='release'))
            assert not rejected['ok']
        finally:
            release.set()
        while web._active_run is not None:
            await asyncio.sleep(.001)
    asyncio.run(scenario())
    assert not any(call[0] == 'release' for call in web.bridge.calls)
