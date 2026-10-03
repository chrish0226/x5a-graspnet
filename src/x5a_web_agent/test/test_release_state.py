"""Release and failed-execution state regressions; ROS bridge is always fake."""
import asyncio
from types import SimpleNamespace
import pytest
from x5a_web_agent.session import Session
from x5a_web_agent.command_parser import follow_up_intent


@pytest.mark.parametrize('text', ['松开', '松开夹爪', '放下', '松手'])
def test_release_words_are_release_commands(text):
    assert follow_up_intent(text)['action'] == 'release'


def test_failure_after_grasp_remembers_holding_without_inventing_identity():
    session = Session()
    session.observe_executor({'success': False, 'holding': True}, prompt='new target')
    assert session.holding and session.held_label() == '物体'
    session.hold('orange cube')
    session.observe_executor({'success': False, 'holding': True}, prompt='wrong target')
    assert session.held_label() == 'orange cube'


def test_failure_after_release_clears_holding_but_unknown_feedback_preserves_it():
    session = Session()
    session.hold('orange cube')
    session.observe_executor({'success': False})
    assert session.holding
    session.observe_executor({'success': False, 'holding': False})
    assert not session.holding


@pytest.fixture
def web(monkeypatch):
    from x5a_web_agent import app
    monkeypatch.setattr(app, 'session', Session())
    monkeypatch.setattr(app, 'hub', app.ConnectionHub())
    return app


@pytest.mark.parametrize('holding', [False, True])
def test_release_reaches_executor_even_when_web_cache_is_empty(web, monkeypatch, holding):
    calls = []
    fake = SimpleNamespace(busy=False, release=lambda: calls.append('release') or {
        'success': True, 'holding': False, 'stage': 'SUCCESS'})
    monkeypatch.setattr(web, 'bridge', fake)
    if holding:
        web.session.hold('orange cube')
    async def scenario():
        response = await web.dispatch_parsed({'action': 'release'})
        assert response['ok']
        assert web.hub.state['last_result'] is None
        pending = asyncio.all_tasks()-{asyncio.current_task()}
        await asyncio.gather(*pending)
    asyncio.run(scenario())
    assert calls == ['release'] and not web.session.holding


def test_failed_grasp_execution_syncs_web_holding(web, monkeypatch):
    fake = SimpleNamespace(
        send_remote_grasp=lambda *_, **__: {'success': True},
        execute_latest=lambda: {'success': False, 'holding': True, 'stage': 'FAILED',
                                'message': 'PLACE failed'})
    monkeypatch.setattr(web, 'bridge', fake)
    monkeypatch.setattr(web, 'auto_execute_enabled', lambda: True)
    asyncio.run(web._run_remote_grasp('orange cube | place: box', {'object': 'cube'}))
    assert web.session.holding and web.hub.state['holding']
    assert web.hub.state['last_result']['success'] is False


def test_busy_release_is_not_sent_to_executor(web, monkeypatch):
    calls = []
    monkeypatch.setattr(web, 'bridge', SimpleNamespace(busy=False, release=lambda: calls.append('release')))
    web.hub.state.update(busy=True, stage='ACCEPTED')
    response = asyncio.run(web.dispatch_parsed({'action': 'release'}))
    assert not response['ok'] and calls == []


def test_failed_place_after_opening_updates_web_to_empty(web, monkeypatch):
    web.session.hold('orange cube')
    fake = SimpleNamespace(
        send_remote_place=lambda *_, **__: {'success': True},
        execute_latest=lambda: {'success': False, 'holding': False, 'stage': 'FAILED'})
    monkeypatch.setattr(web, 'bridge', fake)
    monkeypatch.setattr(web, 'auto_execute_enabled', lambda: True)
    asyncio.run(web._run_remote_place('box', {}))
    assert not web.session.holding and not web.hub.state['holding']


def test_failed_release_preserves_known_held_object(web, monkeypatch):
    web.session.hold('orange cube')
    monkeypatch.setattr(web, 'bridge', SimpleNamespace(release=lambda: {
        'success': False, 'holding': True, 'stage': 'FAILED'}))
    asyncio.run(web._run_release({}))
    assert web.session.held_label() == 'orange cube'
    assert web.hub.state['holding'] and web.hub.state['stage'] == 'FAILED'
