from types import SimpleNamespace as NS

import pytest
from action_msgs.msg import GoalStatus
from x5a_web_agent.ros_bridge import RosBridge


@pytest.mark.parametrize('status', [GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED])
def test_cancelled_or_aborted_action_cannot_look_successful(status):
    wrapped = NS(status=status, result=NS(success=True))
    goal = NS(accepted=True, get_result_async=lambda: NS(result=lambda: wrapped))
    fake = NS(_node=object(), _remote_client=NS(wait_for_server=lambda **_: True,
        send_goal_async=lambda *_, **__: NS(result=lambda: goal)), _wait=lambda *_: True)
    result = RosBridge._send_remote_locked(fake, 'white cube', None, 1., 20, True)
    assert not result['success'] and not result['plan_valid']


def test_timed_out_planning_is_cancelled_before_returning():
    cancelled = []
    goal = NS(accepted=True, get_result_async=lambda: object(),
              cancel_goal_async=lambda: cancelled.append(True))
    waits = iter([True, False])
    fake = NS(_node=object(), _remote_client=NS(wait_for_server=lambda **_: True,
        send_goal_async=lambda *_, **__: NS(result=lambda: goal)), _wait=lambda *_: next(waits))
    result = RosBridge._send_remote_locked(fake, 'white cube', None, 1., 20, True)
    assert not result['success'] and cancelled == [True]


@pytest.mark.parametrize('reason', ['目标有效深度不足：工作范围检查后少于 50 个点',
                                    'PLACE：完整下降路径检查未通过'])
def test_aborted_planning_preserves_the_real_failure_reason(reason):
    wrapped = NS(status=GoalStatus.STATUS_ABORTED, result=NS(success=False, error_message=reason))
    goal = NS(accepted=True, get_result_async=lambda: NS(result=lambda: wrapped))
    fake = NS(_node=object(), _remote_client=NS(wait_for_server=lambda **_: True,
        send_goal_async=lambda *_, **__: NS(result=lambda: goal)), _wait=lambda *_: True)
    result = RosBridge._send_remote_locked(fake, 'white cube', None, 1., 20, True)
    assert result['message'] == reason and not result['success'] and not result['plan_valid']
