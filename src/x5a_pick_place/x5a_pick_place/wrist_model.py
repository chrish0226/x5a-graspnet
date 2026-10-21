"""Reuse the calibrated preview geometry for an isolated MoveIt model."""
import math
import xml.etree.ElementTree as ET

import numpy as np

from x5a_handeye.robot_model import require_matching_model


PREFIX = 'wrist_preview/'


def prefixed_models(urdf, srdf, expected, tool_camera):
    require_matching_model(urdf, expected)
    robot = ET.fromstring(urdf)
    semantic = ET.fromstring(srdf)
    links = {e.get('name') for e in robot.findall('link')}
    for e in robot.iter():
        if e.tag == 'link' and 'name' in e.attrib:
            e.set('name', PREFIX + e.get('name'))
        for attr in ('link',):
            if e.get(attr) in links:
                e.set(attr, PREFIX + e.get(attr))
    for e in semantic.iter():
        for attr in ('base_link', 'tip_link', 'parent_link', 'child_link', 'link1', 'link2', 'name'):
            if e.get(attr) in links:
                e.set(attr, PREFIX + e.get(attr))
        if e.tag == 'virtual_joint':
            e.set('parent_frame', PREFIX + e.get('parent_frame'))
    # The camera is already included in link6 mass. This is collision geometry
    # only. Bounds come from all Gemini 345Lg SDK CAD meshes transformed into
    # the color optical frame, enlarged by 2mm on each side. A box centred on
    # the lens is incorrect: the body centre is 8mm to its right.
    T = np.asarray(tool_camera, dtype=float).copy()
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('invalid camera transform')
    T[:3, 3] += T[:3, :3] @ np.array([0.008, 0., -0.0174101052])
    R = T[:3, :3]
    rpy = (math.atan2(R[2, 1], R[2, 2]),
           math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])),
           math.atan2(R[1, 0], R[0, 0]))
    payload_name = PREFIX + 'camera_payload'
    payload = ET.SubElement(robot, 'link', name=payload_name)
    for kind in ('collision', 'visual'):
        obj = ET.SubElement(payload, kind)
        ET.SubElement(obj, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(ET.SubElement(obj, 'geometry'), 'box', size='0.128 0.033 0.04795001')
    joint = ET.SubElement(robot, 'joint', name='wrist_camera_payload_joint', type='fixed')
    ET.SubElement(joint, 'parent', link=PREFIX + 'tool0')
    ET.SubElement(joint, 'child', link=payload_name)
    ET.SubElement(joint, 'origin', xyz=' '.join(map(str, T[:3, 3])), rpy=' '.join(map(str, rpy)))
    for link in ('link6', 'tool0'):
        ET.SubElement(semantic, 'disable_collisions', link1=payload_name,
                      link2=PREFIX + link, reason='Adjacent')
    return ET.tostring(robot, encoding='unicode'), ET.tostring(semantic, encoding='unicode')
