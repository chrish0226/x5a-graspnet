from types import SimpleNamespace as NS
import numpy as np
import pytest

pytest.importorskip('rclpy')
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
from x5a_remote_grasp.geometry import rpy_rotation, matrix_to_quaternion
from x5a_remote_grasp.moveit_plan import PlanOnlyMoveIt


@pytest.mark.parametrize('mode', ['eye_in_hand', 'eye_to_hand'])
def test_camera_collision_follows_robot_link_and_preserves_orientation(mode):
    # Construct only the geometry owner, with no ROS node or command clients.
    node = object.__new__(RemoteGraspNode)
    node.camera_mount = mode
    node.camera_mount_link = 'tool0'
    node.camera_touch_links = ['link6', 'tool0']
    node.camera_frame = 'wrist_camera_link'
    node.base_frame = 'base_link'
    node.camera_center = np.array([.02, -.01, .03])
    node.camera_size = np.array([.1, .05, .05])
    node.require_camera = True
    node.require_table = False
    node.place_support_hull = None
    T = np.eye(4); T[:3, :3] = rpy_rotation(.2, -.4, .6); T[:3, 3] = [.03, .04, .05]
    calls = []
    node._lookup_matrix = lambda *args: (calls.append(args) or T)
    scene = node._safety_scene(None)
    parent = 'tool0' if mode == 'eye_in_hand' else 'base_link'
    assert calls[0][:2] == (parent, 'wrist_camera_link')
    attached = scene.robot_state.attached_collision_objects[0]
    if mode == 'eye_in_hand':
        assert attached.link_name == 'tool0'
        assert attached.touch_links == ['link6', 'tool0']
        camera = attached.object
        assert next(o for o in scene.world.collision_objects if o.id == camera.id).operation == CollisionObject.REMOVE
    else:
        assert attached.object.operation == CollisionObject.REMOVE
        camera = next(o for o in scene.world.collision_objects if o.id == 'remote_grasp_camera')
    assert scene.robot_state.is_diff
    assert camera.header.frame_id == parent
    pose = camera.primitive_poses[0]
    np.testing.assert_allclose([pose.position.x, pose.position.y, pose.position.z], (T @ np.r_[node.camera_center, 1])[:3])
    np.testing.assert_allclose([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], matrix_to_quaternion(T[:3, :3]))


@pytest.mark.parametrize('exists', [True, False])
def test_attached_camera_cleanup_is_idempotent(exists):
    class Future:
        def done(self): return True
        def result(self): return self.value
    class Client:
        def __init__(self, value): self.value = value
        def wait_for_service(self, **kwargs): return True
        def call_async(self, request):
            self.request = request
            f = Future(); f.value = self.value; return f
    p = object.__new__(PlanOnlyMoveIt); p.deadline = None
    present = PlanningScene()
    if exists: present.robot_state.attached_collision_objects = [AttachedCollisionObject(object=CollisionObject(id='remote_grasp_camera'))]
    p.scene_query_client = Client(NS(scene=present)); p.scene_client = Client(NS(success=True))
    scene = PlanningScene(); scene.is_diff = True; scene.robot_state.is_diff = True
    scene.robot_state.attached_collision_objects = [AttachedCollisionObject(object=CollisionObject(id='remote_grasp_camera', operation=CollisionObject.REMOVE))]
    assert p.apply_scene(scene)[0]
    assert len(p.scene_client.request.scene.robot_state.attached_collision_objects) == int(exists)
