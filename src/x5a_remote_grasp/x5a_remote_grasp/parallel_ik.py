"""Client for independent, snapshot-bound IK workers; no execution actions."""
import copy
import hashlib
import math

from moveit_msgs.msg import PositionIKRequest
from moveit_msgs.srv import GetPlanningScene
from rcl_interfaces.srv import GetParameters
from x5a_task_interfaces.srv import ParallelIK


class ParallelIKClient:
    def __init__(self, node, callback_group, service, workers, wait):
        self.client = node.create_client(ParallelIK, service, callback_group=callback_group)
        self.scene = node.create_client(GetPlanningScene, '/get_planning_scene', callback_group=callback_group)
        self.parameters = node.create_client(GetParameters, '/move_group/get_parameters', callback_group=callback_group)
        self.workers = max(1, min(4, int(workers)))
        self.wait = wait
        self.model_digest = None

    def ready(self):
        return self.client.service_is_ready()

    def _snapshot(self, remaining):
        if not self.scene.wait_for_service(timeout_sec=remaining(2.0)):
            raise RuntimeError('parallel IK: planning scene service unavailable')
        if self.model_digest is None and not self.parameters.wait_for_service(timeout_sec=remaining(2.0)):
            raise RuntimeError('parallel IK: active model parameter service unavailable')
        scene_request = GetPlanningScene.Request()
        scene_request.components.components = 1023  # state, attached bodies, world, ACM, transforms, padding
        scene_future = self.scene.call_async(scene_request)
        model_future = None
        if self.model_digest is None:
            request = GetParameters.Request()
            request.names = ['robot_description', 'robot_description_semantic']
            model_future = self.parameters.call_async(request)
        if not self.wait(scene_future, 2.0) or scene_future.result() is None:
            raise RuntimeError('parallel IK: authoritative scene snapshot unavailable')
        if model_future is not None:
            if not self.wait(model_future, 2.0) or model_future.result() is None:
                raise RuntimeError('parallel IK: active robot model parameters unavailable')
            values = model_future.result().values
            if len(values) != 2 or any(v.type != 4 or not v.string_value for v in values):
                raise RuntimeError('parallel IK: active URDF/SRDF missing')
            self.model_digest = hashlib.sha256((values[0].string_value+'\0'+values[1].string_value).encode()).hexdigest()
        scene = scene_future.result().scene
        if scene.is_diff or scene.robot_state.is_diff or not scene.robot_state.joint_state.name:
            raise RuntimeError('parallel IK: incomplete scene snapshot')
        return scene

    def solutions(self, pose, attempts, group, tcp, timeout, remaining):
        """Probe the preferred seed first; parallelize fallback seeds only.

        Both batches use ONE complete snapshot, including the observed fingers.
        Keep results in seed preference order, not worker completion order.
        The caller must recheck each solution against the live MoveIt scene.
        """
        scene = self._snapshot(remaining)
        # Planned LIFT/RETREAT can have different fingers than the live arm:
        # GRASP closes them and PLACE opens them. All fallback seeds must use
        # those planned fingers, including the otherwise-current-state seed.
        if attempts and attempts[0] is not None:
            planned = dict(zip(attempts[0].joint_state.name, attempts[0].joint_state.position))
            if any(name in planned for name in ('joint7', 'joint8')):
                scene = copy.deepcopy(scene)
                names = list(scene.robot_state.joint_state.name)
                positions = list(scene.robot_state.joint_state.position)
                for name in ('joint7', 'joint8'):
                    if name in planned:
                        if name not in names:
                            raise RuntimeError('parallel IK: snapshot missing observed finger state')
                        positions[names.index(name)] = planned[name]
                scene.robot_state.joint_state.position = positions
        for seeds, workers in ((attempts[:1], 1), (attempts[1:], self.workers)):
            if not seeds:
                continue
            budget = remaining(timeout*math.ceil(len(seeds)/workers)+0.3)
            if budget <= 0:
                return
            batch = ParallelIK.Request()
            batch.scene = scene
            batch.model_digest = self.model_digest
            batch.workers = workers
            batch.total_budget_s = min(10.0, budget)
            for seed in seeds:
                request = PositionIKRequest()
                request.group_name, request.ik_link_name = group, tcp
                request.pose_stamped = pose
                request.avoid_collisions = True
                seconds = min(timeout, budget)
                request.timeout.sec = int(seconds)
                request.timeout.nanosec = int((seconds-int(seconds))*1e9)
                if seed is None:
                    request.robot_state.is_diff = True
                else:
                    request.robot_state = copy.deepcopy(seed)
                batch.requests.append(request)
            future = self.client.call_async(batch)
            if not self.wait(future, budget+0.5) or future.result() is None:
                if remaining(1.0) <= 0:
                    return  # Caller exhausted its whole planning budget.
                raise RuntimeError('parallel IK batch timed out')
            result = future.result()
            if (not result.success or result.model_digest != self.model_digest
                    or len(result.error_codes) != len(seeds) or len(result.solutions) != len(seeds)):
                raise RuntimeError('parallel IK batch contract failed: '+result.message)
            for code, solution in zip(result.error_codes, result.solutions):
                if code.val == 1:
                    yield solution
                elif code.val not in (-31, -6):
                    raise RuntimeError(f'parallel IK rejected request/state (code {code.val})')
