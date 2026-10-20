from x5a_web_agent.terminal import TerminalConsole


def test_sequence_log_uses_steps_and_timings_without_raw_model_json(capsys, monkeypatch):
    monkeypatch.delenv('X5A_VERBOSE_TASK_LOGS', raising=False)
    console = TerminalConsole(None, show_help=False)
    tasks = [dict(action='remote_grasp', source_text='把白色方块放到橙色方块上'),
             dict(action='remote_grasp', source_text='把红色方块放到白色方块上')]
    console.on_payload(dict(type='parsed', result=dict(action='sequence', tasks=tasks)))
    console.on_payload(dict(type='step', command_id=3, step_index=2, step_total=2,
                            message='把红色方块放到白色方块上'))
    console.on_payload(dict(type='status', command_id=3, step_index=2, step_total=2,
                            stage='REMOTE_INFERENCE', stage_zh='SAM 分割与抓取推理', elapsed_s=8.2))
    console.on_payload(dict(type='status', command_id=3, step_index=2, step_total=2,
                            stage='REMOTE_INFERENCE', stage_zh='SAM 分割与抓取推理', elapsed_s=9.))
    console.on_payload(dict(type='status', stage='CANDIDATE_REJECT', stage_zh='候选被拒绝'))
    console.on_payload(dict(type='result', command_id=3, step_index=2, step_total=2,
                            success=True, message='放置轨迹执行完成', duration_s=6.5, planning_s=1.2,
                            holding=False))
    output = capsys.readouterr().out
    assert '连续任务，共 2 步' in output
    assert '1. 把白色方块放到橙色方块上' in output
    assert '[任务 3 · 2/2]' in output
    assert output.count('SAM 分割与抓取推理') == 1
    assert '规划 1.2s / 本步 6.5s' in output and '夹爪已释放' in output
    assert '"action"' not in output and '"prompt"' not in output
    assert '候选被拒绝' not in output


def test_verbose_logs_can_show_original_parser_data(capsys, monkeypatch):
    monkeypatch.setenv('X5A_VERBOSE_TASK_LOGS', '1')
    console = TerminalConsole(None, show_help=False)
    console.on_payload(dict(type='parsed', result=dict(action='remote_grasp', prompt='white cube')))
    assert '解析详情' in capsys.readouterr().out


def test_execution_stage_reaches_web_bridge_over_isolated_ros_transport():
    import time
    import rclpy
    from x5a_web_agent.ros_bridge import RosBridge
    from x5a_remote_grasp.remote_grasp_executor import RemoteGraspExecutor
    rclpy.init()
    executor = bridge = None
    stages = []
    try:
        executor = RemoteGraspExecutor()
        assert not executor.enabled
        bridge = RosBridge()
        bridge.set_stage_callback(stages.append)
        bridge.start()
        deadline = time.monotonic()+3.
        while 'EXEC_GRASP' not in stages and time.monotonic() < deadline:
            # Only publishes a status message, never an arm/gripper goal.
            executor._publish_execution_stage('GRASP')
            time.sleep(.03)
        assert 'EXEC_GRASP' in stages
    finally:
        if bridge is not None:
            bridge.shutdown()
        if executor is not None:
            executor.destroy_node()
        rclpy.try_shutdown()
