"""Choose an observed table buffer for this task, clear from the top, and reobserve each layer."""
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import yaml
from std_msgs.msg import String
from x5a_vision.clearance import read_buffer, buffer_policy
from x5a_pick_place.pick_place_node import quat_to_yaw
from x5a_pick_place.stacking import tcp_for_object_center


class ClearingMixin:
    def _resolve_blocker(self, color):
        relations = self._clearing_details().get('blocked_by', {})
        memory = getattr(self, 'scene_memory', None)
        snapshot = self.vision_snapshot() if memory is not None else None
        def remembered(lower):
            return (memory.candidate(lower, snapshot, self.base_frame, self.observation_side)
                    if memory is not None else None)
        blocker = relations.get(color) or remembered(color)
        seen = {color}
        for _ in range(2):
            if blocker is None:
                return None
            if blocker in seen:
                return None
            seen.add(blocker)
            higher = remembered(blocker) or relations.get(blocker)
            if higher is None:
                return blocker
            blocker = higher
        return None

    def _inside_buffer_table(self, xy):
        # Keep the complete observed 8 cm clearance patch on the modeled table.
        if (abs(xy[0]-self.table_x)+.04 >= self.table_sx*.5
                or abs(xy[1]-self.table_y)+.04 >= self.table_sy*.5):
            return False
        return True

    def _request_buffer(self, xy):
        self._buffer_precheck = None
        self._buffer_selection = {'xy_m': list(xy), 'base_frame': self.base_frame,
                                  'observation_side': self.observation_side, 'token': uuid.uuid4().hex}
        self.buffer_request_pub.publish(String(data=json.dumps(self._buffer_selection)))
        return time.monotonic()

    def _buffer_reachable(self, xy, source_top, blocker, held=False):
        if held:
            orientations = [list(self._held_grasp_rpy)]
        else:
            pose = self.cube_poses.get(blocker)
            yaw = quat_to_yaw(pose.pose.orientation) if pose else None
            orientations = self.grasp_orientation_candidates(source_top[0]+self.gx, source_top[1]+self.gy, yaw)[:2]
        for orientation in orientations:
            if self._cancelled() or self.servo_readiness_fault():
                return False
            if held:
                center = (xy[0], xy[1], self.table_z+.5*self.table_sz+.5*self.obj_sz+self.stack_clearance)
                tcp = tcp_for_object_center(center, self._held_object_offset, orientation)
            else:
                tcp = (xy[0]+self.gx, xy[1]+self.gy,
                       self.table_z+.5*self.table_sz+self.obj_sz+self.gz+self.stack_clearance)
            # IK only: these checks never submit an arm trajectory. The existing
            # planner still verifies the complete transit/descent during execution.
            reachable = True
            for height in (0., self.pre_place_h):
                joints = self.ik_joints_for_pose(self.pose_rpy(tcp[0], tcp[1], tcp[2]+height, orientation),
                                                collision_aware_only=True)
                if joints is None or self.servo_bounds_fault(joints):
                    reachable = False
                    break
            if reachable:
                return True
        return False

    def _select_buffer(self, blocker, source_top, held=False):
        started = time.monotonic()
        stats = {'blocker': blocker, 'candidates': 0, 'outside_table': 0,
                 'unreachable': 0, 'vision_rejected': 0, 'proof_changed': 0}
        def finish(xy, reason):
            self._buffer_diagnostics = {**stats, 'selected_xy_m': list(xy) if xy else None,
                                        'reason': reason, 'elapsed_s': round(time.monotonic()-started, 3)}
            logger = getattr(self, 'log_stage', None)
            if logger:
                logger('BUFFER_SELECTION', **self._buffer_diagnostics)
            return xy, reason
        if not held:
            mode, _ = buffer_policy(self.buffer_file)
            if mode == 'auto':
                candidates = self._clearing_details().get('buffer_candidates', {}).get(blocker, [])
                self._buffer_options = [tuple(c['xy_m']) for c in candidates]
            else:
                self._buffer_options = [read_buffer(self.buffer_file, self.base_frame, self.observation_side)]
        options = list(self._buffer_options)
        stats['candidates'] = len(options)
        if held and self._buffer_selection:
            chosen = tuple(self._buffer_selection['xy_m'])
            options = [chosen]+[xy for xy in options if xy != chosen]
        reason = '当前目标旁边没有完整可见且空闲的暂放位置，请调整摆放或观察方向'
        deadline = time.monotonic()+30.0
        for xy in options:
            if time.monotonic() >= deadline:
                return finish(None, '暂放位选择超时，未找到经过验证的可用空位')
            if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
                continue
            if self._cancelled() or self.servo_readiness_fault():
                return finish(None, self.servo_readiness_fault() or '任务已取消')
            if (not self._inside_buffer_table(xy)
                    or math.dist(xy, source_top[:2]) < max(self.obj_sx, self.obj_sy)+.04):
                stats['outside_table'] += 1
                continue
            after = self._request_buffer(xy)
            # Vision runs in its own ROS process. Solve IK while the visual
            # verifier accumulates new exposures; neither operation moves the arm.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='buffer_ik') as workers:
                reachable = workers.submit(self._buffer_reachable, xy, source_top, blocker, held)
                reason = self._await_buffer(xy, after)
                if reason:
                    stats['vision_rejected'] += 1
                    continue
                if not reachable.result():
                    stats['unreachable'] += 1
                    reason = '候选空位的放置姿态不可达，未找到可用暂放点'
                    continue
            # A slow IK response must not turn an old visual proof into a lease.
            reason = self._check_buffer(xy, after)
            if reason:
                stats['vision_rejected'] += 1
                continue
            if not self._freeze_buffer_precheck(xy):
                stats['proof_changed'] += 1
                reason = '暂放位在预检查完成时发生变化，请重新检查'
                continue
            return finish(xy, '')
        return finish(None, reason)

    def _freeze_buffer_precheck(self, xy):
        data = self._clearing_details()
        selected = data.get('selected_buffer', {})
        if (selected.get('empty') is not True or not self._buffer_selection
                or selected.get('token') != self._buffer_selection['token']
                or tuple(selected.get('xy_m', ())) != xy):
            return False
        self._buffer_precheck = {**self._buffer_selection, 'xy_m': list(xy),
            'verified_at': time.monotonic(), 'stamp_ns': data.get('stamp_ns', 0),
            'invalid_reason': ''}
        self._record_buffer_change(self._detection_details)
        return not self._buffer_precheck['invalid_reason']

    def _record_buffer_change(self, details):
        proof = self._buffer_precheck
        if not proof:
            return
        clearing = details.get('clearing', {})
        selected = clearing.get('selected_buffer', {})
        if (selected.get('token') == proof['token'] and clearing.get('stamp_ns', 0) > proof['stamp_ns']
                and selected.get('empty') is False and '占用' in selected.get('reason', '')):
            proof['invalid_reason'] = '抓取期间检测到暂放位被占用'

    def _direct_buffer_after_lift(self, xy, source_top, blocker):
        proof = self._buffer_precheck
        if (not proof or not self._buffer_selection or proof['token'] != self._buffer_selection['token']
                or tuple(proof['xy_m']) != xy or proof['base_frame'] != self.base_frame
                or proof['observation_side'] != self.observation_side):
            return '本次暂放位预检查已失效'
        if self._cancelled() or self.servo_readiness_fault():
            return self.servo_readiness_fault() or '任务已取消'
        if time.monotonic()-proof['verified_at'] > self.buffer_precheck_ttl:
            return '暂放位预检查已超时，请重新观察'
        if proof['invalid_reason']:
            return proof['invalid_reason']
        # The actual held offset exists only after closing and attaching. Solve
        # against it at the lifted pose, without returning the wrist camera home.
        if not self._buffer_reachable(xy, source_top, blocker, held=True):
            return '实际夹持姿态下暂放点不可达'
        if self._cancelled() or self.servo_readiness_fault():
            return self.servo_readiness_fault() or '任务已取消'
        if proof['invalid_reason']:
            return proof['invalid_reason']
        if time.monotonic()-proof['verified_at'] > self.buffer_precheck_ttl:
            return '暂放位预检查已超时，请重新观察'
        return ''

    def _clearing_details(self, after=0.0):
        data = self._detection_details.get('clearing', {})
        stamp = data.get('stamp_ns', 0)
        if not isinstance(stamp, int):
            return {}
        image_age = (self.get_clock().now().nanoseconds-stamp)*1e-9
        if (self._details_arrival <= after or time.monotonic()-self._details_arrival > 2.0
                or not -.05 <= image_age <= 2.0):
            return {}
        return data

    def _check_buffer(self, xy, after=0.0):
        data = self._clearing_details(after).get('selected_buffer', {})
        if (data.get('base_frame') != self.base_frame
                or data.get('observation_side') != self.observation_side
                or tuple(data.get('xy_m', ())) != xy
                or not self._buffer_selection or data.get('token') != self._buffer_selection['token']):
            return '暂放位视觉检查尚未就绪，请等相机更新后重试'
        if data.get('empty') is not True:
            return data.get('reason', '暂放位有占用或深度不足')
        return ''

    def _await_buffer(self, xy, after):
        deadline = time.monotonic()+self.vision_wait_timeout
        reason = '暂放位视觉检查尚未就绪'
        frames, previous_stamp = 0, None
        while time.monotonic() < deadline:
            if self._cancelled() or self.servo_readiness_fault():
                return self.servo_readiness_fault() or '任务已取消'
            reason = self._check_buffer(xy, after)
            if not reason:
                stamp = self._clearing_details(after).get('stamp_ns')
                if stamp != previous_stamp:
                    frames += 1
                    previous_stamp = stamp
                if frames >= 3:
                    return ''
            else:
                frames = 0
            time.sleep(.05)
        return reason or '暂放位尚未获得三帧连续的新深度确认'

    def clear_target_if_needed(self, color):
        """Own one ROS goal; remove at most two upper cubes, from the top down."""
        return self._clear_target_layers(color, remaining=2, cleared=())

    def _clear_target_layers(self, color, remaining, cleared):
        blocker = self._resolve_blocker(color)
        memory = getattr(self, 'scene_memory', None)
        remembered = memory is not None and memory.has_lower(color)
        if self.vision_snapshot()[color]['ready'] and blocker is None and not remembered:
            return True, ''
        if blocker is None and (self.vision_snapshot()[color]['status'] == 'no_top_face' or remembered):
            deadline = time.monotonic()+self.vision_wait_timeout
            while time.monotonic() < deadline:
                if self._cancelled() or self.servo_readiness_fault():
                    return False, self.servo_readiness_fault() or '任务已取消'
                blocker = self._resolve_blocker(color)
                if blocker is not None:
                    break
                if (self.vision_snapshot()[color]['ready']
                        and not (memory is not None and memory.has_lower(color))):
                    return True, ''
                time.sleep(.05)
            if blocker is None:
                return False, '目标顶面被遮挡，但未确认上层方块的颜色和稳定位置，未开始移开'
        if blocker is None:
            # Ordinary missing/stale detection keeps the existing wait/retry path.
            return True, ''
        if blocker not in self.allowed_colors or blocker == color:
            return False, '上层方块关系不明确，未开始移开'
        if remaining <= 0 or blocker in cleared:
            return False, '三层清障范围内未确认目标已露出，停止后续抓取'
        if self.plan_only:
            return False, '自动移开需要实际重新观察；仅规划模式不执行连续抓取'
        if self._holding:
            return False, '夹爪中已有方块，请先完成放置'
        snapshot = self.vision_snapshot()[blocker]
        if not snapshot['ready'] or snapshot.get('xyz') is None:
            return False, '上层方块未稳定定位，未开始移开'
        source_top = tuple(snapshot['xyz'])
        self.report_stage('SELECTING_BUFFER')
        try:
            xy, fault = self._select_buffer(blocker, source_top)
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            return False, str(exc)
        if xy is None:
            return False, fault+'；未开始移开'
        current_source = self.vision_snapshot()[blocker]
        if (not current_source['ready'] or math.dist(current_source['xyz'], source_top) > .01
                or self._resolve_blocker(color) != blocker):
            return False, '选择暂放位期间叠放关系或上层位置发生变化，请重新发出抓取指令'
        if self._cancelled() or self.servo_readiness_fault():
            return False, self.servo_readiness_fault() or '任务已取消'
        self.report_stage('CLEARING_BLOCKER', target_color=color, blocker=blocker)
        ok, message = self.execute_pick_place(blocker, pick_only=True, hold_at_lift=True)
        if not ok:
            return False, '移开上层方块时停止：'+message
        if not self.placement_ready() or self._held_color != blocker:
            return False, '未确认上层方块的夹持状态，停止后续抓取'
        fault = self._direct_buffer_after_lift(xy, source_top, blocker)
        if fault:
            return False, fault+'；上层方块夹持状态保留'
        self.report_stage('BUFFER_PLACEMENT')
        center = (xy[0], xy[1], self.table_z+.5*self.table_sz+.5*self.obj_sz+self.stack_clearance)
        orientation = list(self._held_grasp_rpy)
        tcp = tcp_for_object_center(center, self._held_object_offset, orientation)
        ok, message = self.execute_place_motion(tcp, [orientation], center)
        if not ok:
            return False, '上层方块暂放时停止：'+message
        if self._holding or not self._wait_observation():
            return False, '已完成暂放，但未确认释放或返回观察位；停止后续抓取'
        observed_after = time.monotonic()
        self.report_stage('REOBSERVE_UNCOVERED')
        deadline = observed_after+self.vision_wait_timeout
        while time.monotonic() < deadline:
            if self._cancelled() or self.servo_readiness_fault():
                return False, self.servo_readiness_fault() or '任务已取消'
            item = self.vision_snapshot()[color]
            arrival = self.cube_pose_arrivals.get(color, 0.0)
            pose = self.cube_poses.get(color)
            exposure = (pose.header.stamp.sec*1_000_000_000+pose.header.stamp.nanosec) if pose else 0
            details = self._clearing_details(observed_after)
            next_blocker = self._resolve_blocker(color) if details else None
            still_covered = ((memory is not None and memory.has_lower(color))
                             or (next_blocker is not None and next_blocker not in (*cleared, blocker)))
            if (item['ready'] and not still_covered and arrival > observed_after
                    and details and exposure >= details['stamp_ns']):
                xyz = item['xyz']
                if (math.dist(xyz[:2], source_top[:2]) > .015
                        or abs(xyz[2]-(source_top[2]-self.obj_sz-self.stack_clearance)) > .012):
                    return False, '上层方块已移开，但目标位置与原叠放位置不一致，停止抓取'
                return True, ''
            # The requested bottom can remain hidden after removing the third
            # layer. Require a new exposure of the next blocker before recursion.
            if next_blocker in self.allowed_colors and next_blocker not in (*cleared, blocker):
                next_item = self.vision_snapshot()[next_blocker]
                next_pose = self.cube_poses.get(next_blocker)
                next_stamp = (next_pose.header.stamp.sec*1_000_000_000+next_pose.header.stamp.nanosec) if next_pose else 0
                if (next_item['ready'] and next_item.get('xyz') is not None
                        and self.cube_pose_arrivals.get(next_blocker, 0.) > observed_after
                        and next_stamp >= details['stamp_ns']):
                    xyz = next_item['xyz']
                    if (math.dist(xyz[:2], source_top[:2]) > .015
                            or abs(xyz[2]-(source_top[2]-self.obj_sz-self.stack_clearance)) > .012):
                        return False, '下一层方块的位置或高度与原叠放关系不一致，停止清障'
                    return self._clear_target_layers(color, remaining-1, (*cleared, blocker))
            time.sleep(.05)
        return False, '上层方块已移到暂放位，但目标未获得新的稳定顶面定位；请检查后重试'
