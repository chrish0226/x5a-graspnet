#!/usr/bin/env python3
"""Bounded passive CAN/SDK recorder; never publishes commands or calls services."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import select
import socket
import struct
import time

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600:
        parser.error('duration must be within 1 hour')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Initialize DDS before opening CAN so startup cannot fill the socket queue.
    rclpy.init()
    node = rclpy.create_node('motor_link_recorder_readonly')
    can = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    can.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    can.setsockopt(socket.SOL_SOCKET, 35, 1)  # Linux SO_TIMESTAMPNS
    can.setsockopt(socket.SOL_SOCKET, 40, 1)  # Linux SO_RXQ_OVFL
    can.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_ERR_FILTER, struct.pack('=I', 0x1fffffff))
    can.bind(('can1',))
    can.setblocking(False)
    events = deque(maxlen=300)
    counts, max_gaps, previous = Counter(), {}, {}
    health, previous_health = None, None
    drops = 0
    started = time.monotonic()

    def received(msg):
        nonlocal health, previous_health
        health = json.loads(msg.data)
        state = tuple(health.get(k) for k in ('online', 'protected', 'ready', 'protection_epoch'))
        if state != previous_health:
            events.append({'kind': 'sdk_health_transition', 'wall_time': time.time(),
                           'elapsed_s': time.monotonic() - started, 'health': health,
                           'latest_kernel_feedback_time': dict(previous)})
            previous_health = state

    sub = node.create_subscription(String, '/arm/health', received,
        QoSProfile(depth=20, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def save(finished=False):
        report = {'started_at': time.time() - (time.monotonic() - started),
                  'duration_s': time.monotonic() - started, 'finished': finished,
                  'motion_sent': False, 'health': health, 'socket_drops': drops,
                  'socket_buffer_bytes': can.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
                  'feedback': {k: {'frames': v, 'kernel_max_gap_ms': max_gaps.get(k, 0) * 1000}
                               for k, v in counts.items()}, 'events': list(events)}
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temp.replace(args.output)

    try:
        save_at = started
        while time.monotonic() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=0)
            if select.select([can], [], [], .005)[0]:
                for _ in range(128):
                    try:
                        raw, ancillary, flags, _ = can.recvmsg(16, 128)
                    except BlockingIOError:
                        break
                    cid, length, payload = struct.unpack('=IB3x8s', raw)
                    if flags & socket.MSG_DONTROUTE:
                        continue  # Ignore local command echo.
                    kernel_time = None
                    for level, kind, data in ancillary:
                        if level == socket.SOL_SOCKET and kind == 35:
                            sec, ns = struct.unpack('@ll', data)
                            kernel_time = sec + ns / 1e9
                        elif level == socket.SOL_SOCKET and kind == 40:
                            count = struct.unpack('=I', data)[0]
                            if count != drops:
                                events.append({'kind': 'recorder_queue_overflow',
                                               'wall_time': time.time(), 'dropped_frames': count - drops})
                            drops = count
                    if cid & 0x20000000:
                        events.append({'kind': 'can_error_frame', 'wall_time': kernel_time,
                                       'id': hex(cid), 'payload': payload[:length].hex()})
                        continue
                    if cid not in (0, 1, 2, 4) or length != 8 or kernel_time is None:
                        continue
                    motor = cid if cid else payload[0] & 15
                    if motor not in (1, 2, 4, 5, 6, 7, 8):
                        continue
                    key = str(motor)
                    if key in previous:
                        gap = kernel_time - previous[key]
                        max_gaps[key] = max(max_gaps.get(key, 0), gap)
                        if gap > .05:
                            events.append({'kind': 'kernel_feedback_gap', 'motor_id': motor,
                                           'wall_time': kernel_time, 'gap_ms': gap * 1000})
                    previous[key] = kernel_time
                    counts[key] += 1
            if time.monotonic() >= save_at:
                save()
                save_at = time.monotonic() + 5
    finally:
        save(finished=True)
        can.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
