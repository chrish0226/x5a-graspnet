import json
from types import SimpleNamespace as NS
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode


def test_configured_budget_is_not_reported_as_exhausted_and_stages_count_once(tmp_path):
    report = [dict(reason='geometric_plan', detail='GRASP: GRASP: complete straight approach required'),
              dict(reason='geometric_plan', detail='PREGRASP: collision-aware IK failed'),
              dict(reason='geometry_or_budget', detail='67 poses rejected; planning budget 30.0s',
                   budget_exhausted=False, elapsed_s=5.9)]
    node = NS(plan_budget_s=30.)
    message = RemoteGraspNode._planning_failure_message(node, report, None, tmp_path)
    saved = json.loads((tmp_path/'plan_failures.json').read_text())
    assert saved['attempts'] == 2
    assert saved['stage_counts'] == {'GRASP': 1, 'PREGRASP': 1}
    assert not saved['budget_exhausted'] and '预算已用尽' not in message
    report[-1]['budget_exhausted'] = True
    assert '预算已用尽' in RemoteGraspNode._planning_failure_message(node, report, None, tmp_path)
