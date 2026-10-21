"""The contact band must have a hard floor and preserve other collision rules."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
from moveit_msgs.msg import AllowedCollisionMatrix, AllowedCollisionEntry, CollisionObject, PlanningScene
from x5a_remote_grasp.scene_policy import (
    TABLE_ID, TABLE_LIMIT_ID, contact_limit_box, merge_collision_pairs)
from x5a_remote_grasp.remote_grasp_node import RemoteGraspNode
from x5a_remote_grasp.moveit_plan import PlanOnlyMoveIt


def test_contact_band_has_fixed_bottom_and_five_mm_lower_limit():
    center, size = contact_limit_box([.25, .3, -.026], [.52, .72, .04], .005)
    assert center[2] + size[2]/2 == pytest.approx(-.011)
    assert center[2] - size[2]/2 == pytest.approx(-.046)
    np.testing.assert_allclose(size[:2], [.52, .72])
    for tolerance in (-.001, .0051, float('nan')):
        with pytest.raises(ValueError):
            contact_limit_box([0, 0, 0], [1, 1, .04], tolerance)


def test_contact_pairs_preserve_self_collisions_and_do_not_allow_palm():
    names = ['link6', 'link7', 'link8', 'camera']
    matrix = AllowedCollisionMatrix(entry_names=names, entry_values=[
        AllowedCollisionEntry(enabled=row) for row in (
            [False, True, True, False], [True, False, True, False],
            [True, True, False, False], [False, False, False, False])])
    pairs = {(TABLE_ID, finger): True for finger in ('link7', 'link8')}
    pairs.update({(TABLE_LIMIT_ID, finger): False for finger in ('link7', 'link8')})
    merged = merge_collision_pairs(matrix, pairs)
    def allowed(a, b):
        return merged.entry_values[merged.entry_names.index(a)].enabled[merged.entry_names.index(b)]
    assert allowed('link7', 'link8') and allowed('link6', 'link7')
    assert allowed(TABLE_ID, 'link7') and allowed('link8', TABLE_ID)
    assert not allowed(TABLE_LIMIT_ID, 'link7')
    assert not allowed(TABLE_ID, 'link6') and not allowed(TABLE_ID, 'camera')
    assert matrix.entry_names == names  # caller's scene snapshot was not mutated
    reset = merge_collision_pairs(merged, {(TABLE_ID, 'link7'): False})
    assert not reset.entry_values[reset.entry_names.index(TABLE_ID)].enabled[reset.entry_names.index('link7')]


def test_current_scene_defaults_survive_matrix_expansion():
    matrix = AllowedCollisionMatrix(default_entry_names=['sensor'], default_entry_values=[True])
    merged = merge_collision_pairs(matrix, {('sensor', 'table'): False, ('finger', 'table'): True})
    idx = {name: i for i, name in enumerate(merged.entry_names)}
    assert merged.default_entry_names == ['sensor']
    assert merged.default_entry_values == [True]
    assert merged.entry_values[idx['sensor']].enabled[idx['finger']]
    assert not merged.entry_values[idx['sensor']].enabled[idx['table']]


@pytest.mark.parametrize('tolerance', [0.0, .005])
def test_scene_keeps_physical_table_and_revokes_contact_when_disabled(tolerance):
    node = object.__new__(RemoteGraspNode)
    node.base_frame = 'base_link'; node.require_table = True; node.require_camera = False
    node.table_center = np.array([.25, .3, -.026]); node.table_size = np.array([.52, .72, .04])
    node.table_contact_tolerance = tolerance; node.table_touch_links = ['link7', 'link8']
    node.place_support_hull = None
    requests = []
    node.planner = NS(apply_scene=lambda scene, **kw: (requests.append((scene, kw)) or (True, '')))
    assert node._apply_safety_scene(None)[0]
    scene, options = requests[0]
    table = next(o for o in scene.world.collision_objects if o.id == TABLE_ID)
    assert table.primitive_poses[0].position.z == -.026
    hard = next(o for o in scene.world.collision_objects if o.id == TABLE_LIMIT_ID)
    assert hard.operation == (CollisionObject.ADD if tolerance else CollisionObject.REMOVE)
    assert options['collision_pairs'][TABLE_ID, 'link7'] == bool(tolerance)
    assert not options['collision_pairs'][TABLE_LIMIT_ID, 'link7']


def test_scene_apply_preserves_original_acm_and_fails_if_snapshot_unavailable():
    class Future:
        def __init__(self, value): self.value = value
        def done(self): return True
        def result(self): return self.value
    class Client:
        available = True
        def __init__(self, value): self.value = value; self.requests = []
        def wait_for_service(self, **kw): return self.available
        def call_async(self, request): self.requests.append(request); return Future(self.value)
    current = PlanningScene(allowed_collision_matrix=AllowedCollisionMatrix(
        entry_names=['palm', 'finger'], entry_values=[
            AllowedCollisionEntry(enabled=[False, True]), AllowedCollisionEntry(enabled=[True, False])]))
    planner = object.__new__(PlanOnlyMoveIt); planner.deadline = None
    planner.scene_query_client = Client(NS(scene=current)); planner.scene_client = Client(NS(success=True))
    scene = PlanningScene(is_diff=True)
    assert planner.apply_scene(scene, {(TABLE_ID, 'finger'): True})[0]
    merged = planner.scene_client.requests[0].scene.allowed_collision_matrix
    assert merged.entry_values[0].enabled[1]
    assert not scene.allowed_collision_matrix.entry_names
    planner.scene_query_client.available = False
    assert not planner.apply_scene(scene, {(TABLE_ID, 'finger'): True})[0]
    assert len(planner.scene_client.requests) == 1
