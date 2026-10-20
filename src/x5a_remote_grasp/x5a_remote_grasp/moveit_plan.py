"""MoveIt collision-aware IK and planning-only helpers.

This module intentionally imports neither ExecuteTrajectory nor any controller
action. Every MoveGroup goal sets planning_options.plan_only to true.
"""
from __future__ import annotations

import copy
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PlanningScene,
    PositionConstraint,
    RobotState,
    RobotTrajectory,
)
from moveit_msgs.srv import (
    GetPlanningScene,
    ApplyPlanningScene,
    GetCartesianPath,
    GetPositionIK,
    GetPositionFK,
    GetStateValidity,
)
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from shape_msgs.msg import SolidPrimitive


MOVEIT_SUCCESS = 1
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
CARTESIAN_STAGES = frozenset({"GRASP", "PLACE"})
# 本链路规划在 /display_planned_path 上的专属标记（move_group 也往这个 topic
# 发它自己的规划，执行器靠它区分）。
DISPLAY_MODEL_ID = "X5A_REMOTE_GRASP"


@dataclass
class SequencePlan:
    stages: List[str]
    trajectories: List[RobotTrajectory]
    trajectory_start: RobotState


class PlanOnlyMoveIt:
    def __init__(
        self,
        node,
        callback_group: ReentrantCallbackGroup,
        group_name: str,
        tcp_frame: str,
        move_action: str,
        ik_service: str,
        validity_service: str,
        apply_scene_service: str,
        allowed_time: float,
        result_timeout: float,
        ik_timeout: float,
        attempts: int,
        velocity_scaling: float,
        acceleration_scaling: float,
        position_tolerance: float,
        orientation_tolerance: float,
        cartesian_service: str = "/compute_cartesian_path",
        cartesian_step: float = 0.01,
        min_cartesian_fraction: float = 0.95,
        jump_threshold: float = 0.0,
        parallel_ik_service: str = "",
        parallel_ik_workers: int = 4,
        parallel_ik_fallback_serial: bool = True,
        stage_scaling: Optional[dict] = None,
        pipeline_id: str = "ompl",
        planner_id: str = "RRTConnect",
    ) -> None:
        self.node = node
        self.deadline = None
        self.last_solution_seed: Optional[List[float]] = None
        self.sequence_cache_enabled = False
        self.grasp_branch_recovery_enabled = False
        self.reset_sequence_cache()
        self.group_name = group_name
        self.pipeline_id = str(pipeline_id)
        self.planner_id = str(planner_id)
        self.tcp_frame = tcp_frame
        self.allowed_time = float(allowed_time)
        self.result_timeout = float(result_timeout)
        self.ik_timeout = float(ik_timeout)
        self.attempts = int(attempts)
        self.velocity_scaling = float(velocity_scaling)
        self.acceleration_scaling = float(acceleration_scaling)
        self.stage_scaling = dict(self.STAGE_SCALING)
        if stage_scaling is not None:
            for stage, values in stage_scaling.items():
                pair = tuple(float(v) for v in values)
                if len(pair) != 2 or not all(np.isfinite(v) and 0.0 < v <= 1.0 for v in pair):
                    raise ValueError('stage speed/acceleration must be in (0, 1]')
                self.stage_scaling[stage.upper()] = pair
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.cartesian_step = float(cartesian_step)
        self.min_cartesian_fraction = float(min_cartesian_fraction)
        self.jump_threshold = float(jump_threshold)
        self.parallel_ik = None
        self.parallel_ik_fallback_serial = parallel_ik_fallback_serial
        if parallel_ik_service:
            from x5a_remote_grasp.parallel_ik import ParallelIKClient
            self.parallel_ik = ParallelIKClient(node, callback_group,
                parallel_ik_service, parallel_ik_workers, self._wait)
        self.move_client = ActionClient(
            node, MoveGroup, move_action, callback_group=callback_group
        )
        self.ik_client = node.create_client(
            GetPositionIK, ik_service, callback_group=callback_group
        )
        self.validity_client = node.create_client(
            GetStateValidity, validity_service, callback_group=callback_group
        )
        self.scene_client = node.create_client(
            ApplyPlanningScene, apply_scene_service, callback_group=callback_group
        )
        self.scene_query_client = node.create_client(GetPlanningScene, '/get_planning_scene', callback_group=callback_group)
        self.cartesian_client = node.create_client(
            GetCartesianPath, cartesian_service, callback_group=callback_group
        )
        self.fk_client = node.create_client(GetPositionFK, '/compute_fk', callback_group=callback_group)

    def _timeout(self, requested):
        if self.deadline is None:
            return requested
        return max(0.0, min(requested, self.deadline-time.monotonic()))

    def _wait(self, future, timeout):
        return self.wait_future(future, self._timeout(timeout))

    @staticmethod
    def wait_future(future, timeout: float) -> bool:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if future.done():
                return True
            time.sleep(0.01)
        return future.done()

    def ready(self, timeout: float = 2.0) -> Tuple[bool, str]:
        if self.parallel_ik is not None:
            if not self.parallel_ik.client.wait_for_service(timeout_sec=timeout) and not self.parallel_ik_fallback_serial:
                return False, 'independent IK service unavailable'
        if not self.fk_client.wait_for_service(timeout_sec=timeout):
            return False, "MoveIt /compute_fk is unavailable"
        if not self.ik_client.wait_for_service(timeout_sec=timeout):
            return False, "MoveIt /compute_ik is unavailable"
        if not self.validity_client.wait_for_service(timeout_sec=timeout):
            return False, "MoveIt /check_state_validity is unavailable"
        if not self.scene_client.wait_for_service(timeout_sec=timeout):
            return False, "MoveIt /apply_planning_scene is unavailable"
        if not self.move_client.wait_for_server(timeout_sec=timeout):
            return False, "MoveIt move_action is unavailable"
        if not self.cartesian_client.wait_for_service(timeout_sec=timeout):
            return False, "MoveIt /compute_cartesian_path is unavailable"
        return True, ""

    def apply_scene(self, scene: PlanningScene, collision_pairs=None) -> Tuple[bool, str]:
        self.reset_sequence_cache()
        # MoveIt reports failure when REMOVE names an object already absent.
        # Query first so cleanup remains safe across tasks and node restarts.
        from moveit_msgs.msg import CollisionObject, PlanningSceneComponents
        if (collision_pairs or any(o.operation == CollisionObject.REMOVE for o in scene.world.collision_objects)
                or any(a.object.operation == CollisionObject.REMOVE
                       for a in scene.robot_state.attached_collision_objects)):
            if not self.scene_query_client.wait_for_service(timeout_sec=self._timeout(1.0)):
                return False, 'planning scene query unavailable during obstacle cleanup'
            query = GetPlanningScene.Request()
            query.components.components = (PlanningSceneComponents.WORLD_OBJECT_NAMES
                                           | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
                                           | PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
            future = self.scene_query_client.call_async(query)
            if not self._wait(future, 2.0) or future.result() is None:
                return False, 'planning scene query failed during obstacle cleanup'
            present = {o.id for o in future.result().scene.world.collision_objects}
            scene = copy.deepcopy(scene)
            scene.world.collision_objects = [o for o in scene.world.collision_objects
                                             if o.operation != CollisionObject.REMOVE or o.id in present]
            present_attached = {a.object.id for a in future.result().scene.robot_state.attached_collision_objects}
            scene.robot_state.attached_collision_objects = [
                a for a in scene.robot_state.attached_collision_objects
                if a.object.operation != CollisionObject.REMOVE or a.object.id in present_attached]
            if collision_pairs:
                from x5a_remote_grasp.scene_policy import merge_collision_pairs
                scene.allowed_collision_matrix = merge_collision_pairs(
                    future.result().scene.allowed_collision_matrix, collision_pairs)
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.scene_client.call_async(request)
        if not self.wait_future(future, 3.0):
            return False, "planning scene apply timed out"
        response = future.result()
        if response is None or not response.success:
            return False, "planning scene apply failed"
        return True, ""

    @staticmethod
    def joints_to_state(joints: Sequence[float]) -> RobotState:
        state = RobotState()
        state.joint_state.name = list(ARM_JOINTS)
        state.joint_state.position = [float(v) for v in joints[:6]]
        # Only arm joints are supplied: preserve the observed finger joints.
        state.is_diff = True
        return state

    def collision_aware_ik(
        self,
        pose: PoseStamped,
        seed: Optional[RobotState] = None,
        extra_joint_seeds: Optional[Sequence[Sequence[float]]] = None,
    ) -> Tuple[Optional[RobotState], str]:
        attempts: List[Optional[RobotState]] = []
        if seed is not None:
            attempts.append(seed)
        for joints in extra_joint_seeds or []:
            if len(joints) >= 6:
                attempts.append(self.joints_to_state(joints))
        attempts.append(None)
        # Every seed must preserve the planned jaw opening, including serial
        # fallback. Otherwise PREGRASP can be solved with the live closed jaws
        # even though execution opens them before moving.
        if seed is not None:
            planned = dict(zip(seed.joint_state.name, seed.joint_state.position))
            fingers = {name: planned[name] for name in ('joint7', 'joint8') if name in planned}
            if fingers:
                adjusted = []
                for attempt in attempts:
                    state = copy.deepcopy(attempt) if attempt is not None else RobotState(is_diff=True)
                    values = dict(zip(state.joint_state.name, state.joint_state.position))
                    values.update(fingers)
                    state.joint_state.name, state.joint_state.position = list(values), list(values.values())
                    adjusted.append(state)
                attempts = adjusted
        last_reason = "collision-aware IK failed"
        seen = set()
        unique = []
        for attempt in attempts:
            key = (
                "current"
                if attempt is None
                else tuple(round(float(v), 4) for v in attempt.joint_state.position[:6])
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(attempt)
        backend = getattr(self, 'parallel_ik', None)
        if self._timeout(1.0) <= 0:
            return None, 'planning budget exhausted during IK'
        if backend is not None and backend.ready():
            for solution in backend.solutions(pose, unique, self.group_name,
                    self.tcp_frame, self.ik_timeout, self._timeout):
                if self._timeout(1.0) <= 0:
                    return None, 'planning budget exhausted during parallel IK'
                valid, reason = self.state_valid(solution)
                if valid:
                    return solution, ''
                last_reason = reason
            return None, 'parallel collision-aware IK: '+last_reason
        if backend is not None and not self.parallel_ik_fallback_serial:
            raise RuntimeError('independent IK service unavailable')
        for attempt in unique:
            if self._timeout(1.0) <= 0:
                return None, 'planning budget exhausted during IK'
            solution, reason = self._collision_aware_ik_once(pose, attempt)
            if solution is not None:
                return solution, ""
            last_reason = reason
        return None, last_reason

    def _collision_aware_ik_once(
        self, pose: PoseStamped, seed: Optional[RobotState]
    ) -> Tuple[Optional[RobotState], str]:
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.tcp_frame
        request.ik_request.pose_stamped = pose
        request.ik_request.avoid_collisions = True
        seconds = max(0.01, self._timeout(self.ik_timeout))
        request.ik_request.timeout.sec = int(seconds)
        request.ik_request.timeout.nanosec = int((seconds - int(seconds)) * 1.0e9)
        if seed is None:
            request.ik_request.robot_state.is_diff = True
        else:
            request.ik_request.robot_state = copy.deepcopy(seed)
        future = self.ik_client.call_async(request)
        if not self._wait(future, self.ik_timeout + 1.0):
            return None, "collision-aware IK timed out"
        response = future.result()
        if response is None:
            return None, "collision-aware IK returned no response"
        if response.error_code.val != MOVEIT_SUCCESS:
            return None, f"collision-aware IK failed ({response.error_code.val})"
        valid, reason = self.state_valid(response.solution)
        if not valid:
            return None, reason
        return response.solution, ""

    def state_valid(self, state: RobotState) -> Tuple[bool, str]:
        request = GetStateValidity.Request()
        request.robot_state = state
        request.group_name = self.group_name
        future = self.validity_client.call_async(request)
        if not self._wait(future, 2.0):
            return False, "state validity check timed out"
        response = future.result()
        if response is None:
            return False, "state validity returned no response"
        if not response.valid:
            contacts = len(response.contacts)
            return False, f"MoveIt state is in collision ({contacts} contacts)"
        return True, ""

    def validate_waypoint_ik(
        self,
        poses: Sequence[Tuple[str, PoseStamped]],
        extra_joint_seeds: Optional[Sequence[Sequence[float]]] = None,
    ) -> Tuple[bool, str]:
        seed = None
        for stage, pose in poses:
            if self._timeout(1.0) <= 0:
                return None, 'planning budget exhausted'
            seed, reason = self.collision_aware_ik(
                pose,
                seed,
                extra_joint_seeds=extra_joint_seeds if seed is None else None,
            )
            if seed is None and extra_joint_seeds:
                seed, reason = self.collision_aware_ik(
                    pose, None, extra_joint_seeds=extra_joint_seeds
                )
            if seed is None:
                return False, f"{stage}: {reason}"
        # 留下一整套解作为下一次规划的 IK 种子（跨 goal 复用，命中率远高于固定种子）
        self.last_solution_seed = (
            None if seed is None else [float(v) for v in seed.joint_state.position[:6]]
        )
        return True, ""

    # 每阶段速度/加速度缩放（GRASP/PLACE 接近段慢=轻接触；其余为工作速度档）
    STAGE_SCALING = {
        # PREGRASP/TRANSPORT 0.55：工作速度档（此前 0.25 为排查关节掉线临时降低；
        # 若关节掉线复发，跟踪误差会再次超限，届时先查 CAN/供电再调此值）
        "PREGRASP": (0.55, 0.30),
        "GRASP": (0.28, 0.16),
        "LIFT": (0.35, 0.20),
        "TRANSPORT": (0.55, 0.30),
        "PLACE": (0.28, 0.16),
        "RETREAT": (0.35, 0.20),
    }

    def reset_sequence_cache(self):
        """Only reuse work inside one immutable planning request/scene."""
        self._sequence_cache = {}
        self.sequence_cache_stats = dict(hits=0, negative_hits=0, stage_calls=0,
                                        approach_recoveries=0)

    @staticmethod
    def _pose_key(pose):
        p, q = pose.pose.position, pose.pose.orientation
        return (pose.header.frame_id, pose.header.stamp.sec, pose.header.stamp.nanosec,
                p.x, p.y, p.z, q.x, q.y, q.z, q.w)

    @staticmethod
    def reverse_trajectory(trajectory):
        result = copy.deepcopy(trajectory)
        original = trajectory.joint_trajectory.points
        if not original:
            raise ValueError('cannot reverse an empty trajectory')
        times = [p.time_from_start.sec*1_000_000_000+p.time_from_start.nanosec for p in original]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('trajectory times must increase')
        result.joint_trajectory.points = list(reversed(result.joint_trajectory.points))
        for point, stamp in zip(result.joint_trajectory.points, reversed(times)):
            elapsed = times[-1]-stamp
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(elapsed, 1_000_000_000)
            point.velocities = [-value for value in point.velocities]
        return result

    def _recover_approach(self, pregrasp, grasp, initial_state, previous_state):
        """Solve contact IK, propagate a straight approach backwards, then
        connect the current robot state to that compatible pregrasp branch.
        """
        if self._timeout(1.0) <= 0:
            return None
        contact, reason = self.collision_aware_ik(grasp, previous_state,
            extra_joint_seeds=getattr(self, 'default_joint_seeds', None))
        if contact is None:
            return None
        velocity, acceleration = getattr(self, 'stage_scaling', self.STAGE_SCALING)['GRASP']
        backwards, reason = self.plan_cartesian(pregrasp, contact, velocity, acceleration)
        if backwards is None or not self.verify_endpoint(backwards, contact, pregrasp)[0]:
            return None
        compatible = self.end_state(contact, backwards)
        velocity, acceleration = getattr(self, 'stage_scaling', self.STAGE_SCALING)['PREGRASP']
        connection, reason = self.plan_joints(compatible, initial_state, velocity, acceleration)
        if connection is None:
            return None
        approach = self.reverse_trajectory(backwards)
        if (not self.verify_endpoint(connection.planned_trajectory, connection.trajectory_start, pregrasp)[0]
                or not self.verify_endpoint(approach, compatible, grasp)[0]):
            return None
        # No discontinuity may be hidden between the new global connection
        # and the reversed approach, even if both individual FK checks pass.
        end = connection.planned_trajectory.joint_trajectory.points[-1].positions
        first = approach.joint_trajectory.points[0].positions
        if (connection.planned_trajectory.joint_trajectory.joint_names != approach.joint_trajectory.joint_names
                or len(end) != len(first) or max(abs(a-b) for a, b in zip(end, first)) > 1e-4):
            return None
        return connection.planned_trajectory, connection.trajectory_start, approach, compatible

    def plan_sequence(
        self, poses: Sequence[Tuple[str, PoseStamped]]
    ) -> Tuple[Optional[SequencePlan], str]:
        start_state: Optional[RobotState] = None
        opening = getattr(self, 'expected_pregrasp_open_position', None)
        if opening is not None:
            start_state = RobotState(is_diff=True)
            start_state.joint_state.name = ['joint7', 'joint8']
            start_state.joint_state.position = [float(opening), float(opening)]
        trajectory_start: Optional[RobotState] = None
        trajectories: List[RobotTrajectory] = []
        stages: List[str] = []
        cache = self._sequence_cache if getattr(self, 'sequence_cache_enabled', False) else None
        stats = getattr(self, 'sequence_cache_stats', {})
        context = tuple(getattr(self, name, None) for name in (
            'expected_pregrasp_open_position', 'expected_grasp_finger_position', 'expected_place_open_position',
            'pipeline_id', 'planner_id'))
        seeds = tuple(tuple(float(v) for v in seed) for seed in getattr(self, 'default_joint_seeds', []) or [])
        keys = [(context, seeds)]
        for stage, pose in poses:
            if self._timeout(1.0) <= 0:
                return None, f'{stage}: planning budget exhausted'
            velocity, acceleration = getattr(self, 'stage_scaling', self.STAGE_SCALING).get(
                stage.upper(), (self.velocity_scaling, self.acceleration_scaling)
            )
            if cache is not None:
                clearance = getattr(self, 'retreat_clearance_pose', None) if stage.upper() == 'RETREAT' else None
                keys.append((stage, self._pose_key(pose), velocity, acceleration,
                             None if clearance is None else self._pose_key(clearance)))
                key = tuple(keys)
                cached = cache.get(key)
                if cached is not None:
                    stats['hits'] += 1
                    if isinstance(cached, str):
                        stats['negative_hits'] += 1
                        return None, cached
                    trajectory, used_start, start_state = copy.deepcopy(cached)
                    if trajectory_start is None:
                        trajectory_start = copy.deepcopy(used_start)
                    trajectories.append(trajectory)
                    stages.append(stage)
                    self.last_solution_seed = self._arm_joints(start_state)
                    continue
            stats['stage_calls'] = stats.get('stage_calls', 0)+1
            trajectory, used_start, reason = self._plan_stage(
                stage,
                pose,
                start_state,
                velocity,
                acceleration,
            )
            if (trajectory is None and stage.upper() == 'GRASP' and stages == ['PREGRASP']
                    and getattr(self, 'grasp_branch_recovery_enabled', False)):
                recovered = self._recover_approach(poses[0][1], pose, trajectory_start, start_state)
                if recovered is not None:
                    connection, initial, trajectory, used_start = recovered
                    trajectories[0], trajectory_start = connection, copy.deepcopy(initial)
                    stats['approach_recoveries'] = stats.get('approach_recoveries', 0)+1
                    if cache is not None:
                        cache[tuple(keys[:-1])] = copy.deepcopy((connection, initial, used_start))
            if trajectory is None or used_start is None:
                if cache is not None and len(cache) < 1024 and not any(word in reason.lower() for word in ('timeout', 'timed out', 'budget')):
                    cache[key] = f'{stage}: {reason}'
                return None, f"{stage}: {reason}"
            verified, reason = self.verify_endpoint(trajectory, used_start, pose)
            if not verified:
                return None, f'{stage}: {reason}'
            if not trajectory.joint_trajectory.points:
                return None, f"{stage}: MoveIt returned an empty trajectory"
            if trajectory_start is None:
                trajectory_start = copy.deepcopy(used_start)
            trajectories.append(trajectory)
            stages.append(stage)
            start_state = self.end_state(used_start, trajectory)
            finger_position = (getattr(self, 'expected_grasp_finger_position', None) if stage.upper() == 'GRASP'
                               else getattr(self, 'expected_place_open_position', None) if stage.upper() == 'PLACE'
                               else None)
            if finger_position is not None:
                names = list(start_state.joint_state.name)
                positions = list(start_state.joint_state.position)
                for name in ('joint7', 'joint8'):
                    value = float(finger_position)
                    if name in names:
                        positions[names.index(name)] = value
                    else:
                        names.append(name)
                        positions.append(value)
                start_state.joint_state.name = names
                start_state.joint_state.position = positions
            self.last_solution_seed = self._arm_joints(start_state)
            if cache is not None and len(cache) < 1024:
                cache[key] = copy.deepcopy((trajectory, used_start, start_state))
        assert trajectory_start is not None
        return SequencePlan(stages, trajectories, trajectory_start), ""

    def _plan_stage(
        self,
        stage: str,
        pose: PoseStamped,
        start_state: Optional[RobotState],
        velocity: float,
        acceleration: float,
    ) -> Tuple[Optional[RobotTrajectory], Optional[RobotState], str]:
        name = stage.upper()
        if name == 'RETREAT' and start_state is not None and getattr(self, 'retreat_clearance_pose', None) is not None:
            return self._plan_retreat_with_clearance(pose, start_state, velocity, acceleration)
        if name in CARTESIAN_STAGES and start_state is not None:
            trajectory, reason = self.plan_cartesian(
                pose, start_state, velocity, acceleration
            )
            if trajectory is not None:
                return trajectory, start_state, ""
            self.node.get_logger().warn(
                f"{stage} cartesian failed ({reason}); rejecting incomplete approach"
            )
            return None, None, f'{stage}: complete straight approach required: {reason}'
        extra = None if start_state is not None and name != 'PREGRASP' else getattr(
            self, "default_joint_seeds", None
        )
        ik_state, ik_reason = self.collision_aware_ik(
            pose, start_state, extra_joint_seeds=extra
        )
        if ik_state is not None:
            result, reason = self.plan_joints(ik_state, start_state, velocity, acceleration)
            if result is not None:
                used_start = result.trajectory_start
                return result.planned_trajectory, used_start, ""
            self.node.get_logger().warn(
                f"{stage} joint-space plan failed ({reason}); falling back to pose goal"
            )
        elif ik_reason:
            self.node.get_logger().warn(f"{stage} IK failed ({ik_reason})")
            return None, None, ik_reason
        result, reason = self.plan_pose(pose, start_state, velocity, acceleration)
        if result is None:
            return None, None, reason
        return result.planned_trajectory, result.trajectory_start, ""

    def _plan_retreat_with_clearance(self, pose, start_state, velocity, acceleration):
        clearance = self.retreat_clearance_pose
        first, reason = self.plan_cartesian(clearance, start_state, velocity, acceleration)
        if first is None:
            return None, None, 'RETREAT vertical clearance failed: '+reason
        valid, reason = self.verify_endpoint(first, start_state, clearance)
        if not valid:
            return None, None, reason
        middle = self.end_state(start_state, first)
        second, _, reason = self._plan_stage('RETREAT_TRANSIT', pose, middle, velocity, acceleration)
        if second is None:
            return None, None, reason
        return self.join_trajectories(first, second), start_state, ''

    @staticmethod
    def join_trajectories(first, second):
        a, b = first.joint_trajectory, second.joint_trajectory
        if not a.points or not b.points or list(a.joint_names) != list(b.joint_names):
            raise ValueError('cannot join retreat trajectories with missing/mismatched joints')
        if max(abs(x-y) for x, y in zip(a.points[-1].positions, b.points[0].positions)) > .001:
            raise ValueError('retreat trajectory junction is discontinuous')
        result = copy.deepcopy(first)
        end = a.points[-1].time_from_start
        offset_ns = end.sec*1_000_000_000+end.nanosec
        for original in b.points:
            point = copy.deepcopy(original)
            stamp = point.time_from_start
            delta_ns = stamp.sec*1_000_000_000+stamp.nanosec
            if delta_ns <= 0:
                continue
            stamp.sec, stamp.nanosec = divmod(offset_ns+delta_ns, 1_000_000_000)
            result.joint_trajectory.points.append(point)
        return result

    def plan_cartesian(
        self,
        pose: PoseStamped,
        start_state: RobotState,
        velocity_scaling: Optional[float] = None,
        acceleration_scaling: Optional[float] = None,
    ) -> Tuple[Optional[RobotTrajectory], str]:
        request = GetCartesianPath.Request()
        request.header = pose.header
        request.start_state = copy.deepcopy(start_state)
        request.group_name = self.group_name
        request.link_name = self.tcp_frame
        request.max_step = max(0.001, self.cartesian_step)
        request.jump_threshold = max(0.0, self.jump_threshold)
        request.avoid_collisions = True
        # GetCartesianPath.Request 没有速度/加速度缩放字段（Humble 的 moveit_msgs
        # 已移除），move_group 一律按满速做时间参数化，GRASP/PLACE 的 STAGE_SCALING
        # 会被静默忽略：下探段峰值关节速度超出适配器 max_joint_speed_rad_s 后整条
        # 轨迹被判 INVALID_GOAL 丢弃，臂停在原地不下降也不开爪。这里改用时间拉伸，
        # 把阶段缩放补回来（几何路径不变，只放慢时序）。
        velocity = self.velocity_scaling if velocity_scaling is None else velocity_scaling
        acceleration = (
            self.acceleration_scaling
            if acceleration_scaling is None
            else acceleration_scaling
        )
        waypoint = Pose()
        waypoint.position = pose.pose.position
        waypoint.orientation = pose.pose.orientation
        request.waypoints = [waypoint]
        future = self.cartesian_client.call_async(request)
        if not self._wait(future, 30.0):
            return None, "cartesian path timed out"
        response = future.result()
        if response is None:
            return None, "cartesian path returned no response"
        fraction = float(response.fraction)
        if response.error_code.val != MOVEIT_SUCCESS:
            return None, f'cartesian path failed ({response.error_code.val})'
        if fraction < max(0.9999, self.min_cartesian_fraction):
            return None, f"cartesian fraction {fraction:.2f} below {self.min_cartesian_fraction:.2f}"
        if not response.solution.joint_trajectory.points:
            return None, "cartesian path returned an empty trajectory"
        self._apply_stage_scaling(response.solution, velocity, acceleration)
        return response.solution, ""

    def _apply_stage_scaling(
        self, trajectory: RobotTrajectory, velocity: float, acceleration: float
    ) -> None:
        """按阶段缩放放慢笛卡尔轨迹（时间 × 1/velocity），并记录峰值关节速度。

        速度场除以拉伸系数、加速度场除以系数平方，保持 FJT 轨迹自洽。
        """
        points = trajectory.joint_trajectory.points
        if len(points) < 2:
            return
        factor = max(1.0 / min(1.0, max(0.001, velocity)),
                     1.0 / np.sqrt(min(1.0, max(0.001, acceleration))))
        accel_factor = factor * factor
        for point in points:
            stamp = point.time_from_start
            total = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            total *= factor
            nanos = int(round(total*1e9))
            stamp.sec, stamp.nanosec = divmod(nanos, 1000000000)
            if len(point.velocities) == len(point.positions):
                point.velocities = [float(v) / factor for v in point.velocities]
            if len(point.accelerations) == len(point.positions):
                point.accelerations = [
                    float(a) / accel_factor for a in point.accelerations
                ]
        speeds = []
        for first, second in zip(points, points[1:]):
            span = (
                (float(second.time_from_start.sec) + float(second.time_from_start.nanosec) * 1e-9)
                - (float(first.time_from_start.sec) + float(first.time_from_start.nanosec) * 1e-9)
            )
            if span <= 1e-9:
                continue
            speeds.extend(
                abs(b - a) / span
                for a, b in zip(first.positions, second.positions)
            )
        if speeds:
            self.node.get_logger().info(
                f"cartesian stage scaled x{factor:.2f} "
                f"(velocity={velocity:.2f} acceleration={acceleration:.2f}), "
                f"peak_joint_speed={max(speeds):.3f} rad/s, "
                f"duration={float(points[-1].time_from_start.sec) + float(points[-1].time_from_start.nanosec) * 1e-9:.2f}s"
            )

    def plan_joints(
        self,
        goal_state: RobotState,
        start_state: Optional[RobotState],
        velocity_scaling: float,
        acceleration_scaling: float,
    ):
        joints = self._arm_joints(goal_state)
        if joints is None:
            return None, "IK solution is missing joint1..joint6"
        if not self.move_client.wait_for_server(timeout_sec=self._timeout(3.0)):
            return None, "MoveGroup action server is unavailable; check planning.move_group_action"
        goal = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = self.group_name
        request.pipeline_id = getattr(self, 'pipeline_id', 'ompl')
        request.planner_id = getattr(self, 'planner_id', 'RRTConnect')
        request.num_planning_attempts = max(1, self.attempts)
        request.allowed_planning_time = max(0.01, self._timeout(self.allowed_time))
        request.max_velocity_scaling_factor = min(1.0, max(0.001, velocity_scaling))
        request.max_acceleration_scaling_factor = min(
            1.0, max(0.001, acceleration_scaling)
        )
        if start_state is None:
            request.start_state.is_diff = True
        else:
            request.start_state = copy.deepcopy(start_state)
        constraint = Constraints()
        for name, value in zip(ARM_JOINTS, joints):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = float(value)
            joint.tolerance_above = 0.003
            joint.tolerance_below = 0.003
            joint.weight = 1.0
            constraint.joint_constraints.append(joint)
        request.goal_constraints = [constraint]
        goal.request = request
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True
        send_future = self.move_client.send_goal_async(goal)
        if not self._wait(send_future, 5.0):
            return None, "MoveGroup joint goal timed out"
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return None, "MoveGroup rejected joint-space goal"
        result_future = handle.get_result_async()
        if not self._wait(result_future, self.result_timeout):
            handle.cancel_goal_async()
            return None, "MoveGroup joint planning timed out"
        wrapped = result_future.result()
        if wrapped is None or wrapped.result.error_code.val != MOVEIT_SUCCESS:
            code = 0 if wrapped is None else wrapped.result.error_code.val
            return None, f"MoveGroup joint planning failed ({code})"
        return wrapped.result, ""

    @staticmethod
    def _arm_joints(state: RobotState) -> Optional[List[float]]:
        names = list(state.joint_state.name)
        positions = list(state.joint_state.position)
        try:
            return [float(positions[names.index(name)]) for name in ARM_JOINTS]
        except ValueError:
            return None

    def plan_pose(
        self, pose: PoseStamped, start_state: Optional[RobotState],
        velocity_scaling: Optional[float] = None,
        acceleration_scaling: Optional[float] = None,
    ):
        goal = MoveGroup.Goal()
        request = MotionPlanRequest()
        request.group_name = self.group_name
        request.pipeline_id = getattr(self, 'pipeline_id', 'ompl')
        request.planner_id = getattr(self, 'planner_id', 'RRTConnect')
        request.num_planning_attempts = max(1, self.attempts)
        request.allowed_planning_time = max(0.01, self._timeout(self.allowed_time))
        velocity = self.velocity_scaling if velocity_scaling is None else velocity_scaling
        acceleration = (
            self.acceleration_scaling if acceleration_scaling is None else acceleration_scaling
        )
        request.max_velocity_scaling_factor = min(1.0, max(0.001, velocity))
        request.max_acceleration_scaling_factor = min(1.0, max(0.001, acceleration))
        if start_state is None:
            request.start_state.is_diff = True
        else:
            request.start_state = copy.deepcopy(start_state)
        constraint = Constraints()
        position = PositionConstraint()
        position.header = pose.header
        position.link_name = self.tcp_frame
        position.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [max(0.001, self.position_tolerance)]
        position.constraint_region.primitives = [sphere]
        position.constraint_region.primitive_poses = [pose.pose]
        orientation = OrientationConstraint()
        orientation.header = pose.header
        orientation.link_name = self.tcp_frame
        orientation.orientation = pose.pose.orientation
        orientation.absolute_x_axis_tolerance = self.orientation_tolerance
        orientation.absolute_y_axis_tolerance = self.orientation_tolerance
        orientation.absolute_z_axis_tolerance = self.orientation_tolerance
        orientation.weight = 1.0
        constraint.position_constraints = [position]
        constraint.orientation_constraints = [orientation]
        request.goal_constraints = [constraint]
        goal.request = request
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True
        send_future = self.move_client.send_goal_async(goal)
        if not self._wait(send_future, 5.0):
            return None, "MoveGroup goal timed out"
        handle = send_future.result()
        if handle is None or not handle.accepted:
            return None, "MoveGroup rejected planning-only goal"
        result_future = handle.get_result_async()
        if not self._wait(result_future, self.result_timeout):
            handle.cancel_goal_async()
            return None, "MoveGroup planning timed out"
        wrapped = result_future.result()
        if wrapped is None or wrapped.result.error_code.val != MOVEIT_SUCCESS:
            code = 0 if wrapped is None else wrapped.result.error_code.val
            return None, f"MoveGroup planning failed ({code})"
        return wrapped.result, ""

    def verify_endpoint(self, trajectory, start, pose):
        """Reject trajectories that terminate short of the requested TCP pose."""
        if not trajectory.joint_trajectory.points:
            return False, 'empty endpoint trajectory'
        if not self.fk_client.wait_for_service(timeout_sec=self._timeout(0.5)):
            return False, 'endpoint FK service unavailable'
        request = GetPositionFK.Request()
        request.header = pose.header
        request.fk_link_names = [self.tcp_frame]
        request.robot_state = self.end_state(start, trajectory)
        future = self.fk_client.call_async(request)
        if not self._wait(future, 2.0):
            return False, 'endpoint FK timed out'
        result = future.result()
        if result is None or result.error_code.val != MOVEIT_SUCCESS or not result.pose_stamped:
            return False, 'endpoint FK failed'
        actual = result.pose_stamped[0].pose
        target = pose.pose
        error = np.linalg.norm([actual.position.x-target.position.x,
                               actual.position.y-target.position.y,
                               actual.position.z-target.position.z])
        a = np.array([actual.orientation.x, actual.orientation.y, actual.orientation.z, actual.orientation.w])
        b = np.array([target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w])
        angle = 2*np.arccos(np.clip(abs(np.dot(a, b))/(np.linalg.norm(a)*np.linalg.norm(b)), 0, 1))
        if not np.isfinite(error+angle) or error > self.position_tolerance or angle > self.orientation_tolerance:
            return False, f'TCP endpoint error {error*1000:.1f}mm/{np.degrees(angle):.1f}deg'
        return True, ''

    @staticmethod
    def end_state(start: RobotState, trajectory: RobotTrajectory) -> RobotState:
        output = copy.deepcopy(start)
        names = list(trajectory.joint_trajectory.joint_names)
        positions = list(trajectory.joint_trajectory.points[-1].positions)
        state_names = list(output.joint_state.name)
        state_positions = list(output.joint_state.position)
        for name, value in zip(names, positions):
            if name in state_names:
                state_positions[state_names.index(name)] = float(value)
            else:
                state_names.append(name)
                state_positions.append(float(value))
        output.joint_state.name = state_names
        output.joint_state.position = state_positions
        output.is_diff = False
        return output

    @staticmethod
    def display_message(plan: SequencePlan, model_id: str = DISPLAY_MODEL_ID) -> DisplayTrajectory:
        """发布给执行器的规划。

        model_id 用作本链路专属标记：/display_planned_path 上 move_group 也会
        发它自己的规划（通常只有 1 段），执行器据此只认本节点的规划。
        段名不放在消息里（RobotTrajectory 没有名字字段），执行器按段数映射；
        段数由本文件的 poses 列表决定，见 execution 侧 STAGE_TABLE。
        """
        message = DisplayTrajectory()
        message.model_id = model_id
        message.trajectory_start = plan.trajectory_start
        message.trajectory = list(plan.trajectories)
        return message
