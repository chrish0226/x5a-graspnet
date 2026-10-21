"""Read-only startup decisions. Never kills an existing robot stack."""
import sys
import time

import rclpy

rclpy.init()
node = rclpy.create_node('wrist_workspace_preflight')
try:
    until = time.monotonic() + 2.0
    while time.monotonic() < until:
        rclpy.spin_once(node, timeout_sec=0.1)
    services = {name for name, _ in node.get_service_names_and_types()}
    conflicts = services & {'/x5a_remote_grasp/execute', '/x5a/pick_place/_action/send_goal',
                            '/wrist_pick/pick_place/_action/send_goal', '/wrist_pick/prepare'}
    if conflicts:
        raise RuntimeError('已有抓取任务服务运行，请先停止它的启动终端：' + ', '.join(sorted(conflicts)))
    arm = node.get_publishers_info_by_topic('/arm_status')
    commands = node.get_publishers_info_by_topic('/arm_cmd')
    if len(arm) > 1 or len(commands) > 1:
        raise RuntimeError('机械臂驱动或控制指令来源重复，请关闭重复驱动/手动控制窗口')
    if arm and '/arm/recover' not in services:
        raise RuntimeError('已有原厂驱动运行，新的反馈恢复逻辑尚未生效。请放稳机械臂、关闭旧驱动终端，再运行本工作区 scripts/start_driver.sh')
    if commands and commands[0].node_name != 'x5a_official_trajectory_adapter':
        raise RuntimeError('控制话题正在被手动或其它节点占用：' + commands[0].node_name)
    if commands and not arm:
        raise RuntimeError('已有轨迹适配器但没有机械臂反馈，请先恢复原驱动')
    if any('wrist_preview' in name for name, _ in node.get_node_names_and_namespaces()):
        raise RuntimeError('已有腕部定位节点，请先停止它的启动终端以避免重复坐标变换')
    camera_topics = ['/wrist_camera/color/image_raw', '/wrist_camera/depth/image_raw',
                     '/wrist_camera/color/camera_info']
    counts = [len(node.get_publishers_info_by_topic(t)) for t in camera_topics]
    if any(counts):
        # The measured HSV pixel thresholds and handeye parameters require HD
        # aligned images. Do not silently reuse an arbitrary camera profile.
        raise RuntimeError('腕部相机已被另一启动终端占用，请先停止该相机，再用本工作区启动高清配置')
    print(' '.join('true' if start else 'false' for start in (not arm, not commands, True)))
except Exception as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(2)
finally:
    node.destroy_node()
    rclpy.try_shutdown()
