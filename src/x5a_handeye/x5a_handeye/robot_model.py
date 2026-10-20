"""Export/verify an isolated kinematic model; never change encoder feedback.

An axis offset is represented as R_origin @ R_axis(offset), so the exported
URDF at raw q agrees with nominal FK at q + offset. The SDK gravity URDF is
deliberately outside this module.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from x5a_handeye.x5a_fk import (
    JOINT_ORIGIN_XYZ, JOINT_ORIGIN_RPY, JOINT_AXIS, LINK6_TO_TOOL0_XYZ,
    _axis_matrix, _rpy_matrix, fk_base_tool0,
)

CHAIN_NAMES = [f'joint{i}' for i in range(1, 7)] + ['link6_tool0_joint']


def valid_offsets(offsets):
    q = np.asarray(offsets, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('need six finite axis offsets in radians')
    if q[0] != 0 or q[5] != 0:
        raise ValueError('joint1/joint6 offsets are not identifiable separately from base/camera transforms')
    if np.max(np.abs(q)) > math.radians(5):
        raise ValueError('candidate offsets exceed the 5 degree diagnostic bound')
    return q


def corrected_samples(samples, offsets):
    offsets = valid_offsets(offsets)
    out = []
    for s in samples:
        q = np.asarray(s.get('joints'), dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('raw six-joint feedback is required to recompute calibration FK')
        out.append({**s, 'T_base_tool': fk_base_tool0(q + offsets)})
    return out


def model_signature(xml):
    root = ET.fromstring(xml)
    result = {}
    for name in CHAIN_NAMES:
        matches = root.findall(f"joint[@name='{name}']")
        if len(matches) != 1:
            raise ValueError(f'robot_description needs exactly one {name}')
        joint = matches[0]
        origin, axis = joint.find('origin'), joint.find('axis')
        xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
        rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
        a = [float(v) for v in axis.get('xyz').split()] if axis is not None else [0., 0., 0.]
        if any(len(v) != 3 or not np.isfinite(v).all() for v in (xyz, rpy, a)):
            raise ValueError(f'invalid origin/axis in {name}')
        result[name] = {
            'parent': joint.find('parent').get('link'),
            'child': joint.find('child').get('link'),
            'type': joint.get('type'), 'xyz': xyz,
            'rotation': _rpy_matrix(*rpy).tolist(), 'axis': a,
        }
    return result


def require_matching_model(xml, expected):
    actual = model_signature(xml)
    if set(expected) != set(CHAIN_NAMES):
        raise ValueError('incomplete calibrated robot model signature')
    for name in CHAIN_NAMES:
        for key in ('parent', 'child', 'type'):
            if actual[name][key] != expected[name][key]:
                raise ValueError(f'{name} {key} differs from the calibrated robot model')
        for key in ('xyz', 'rotation', 'axis'):
            if not np.allclose(actual[name][key], expected[name][key], atol=1e-9, rtol=0):
                raise ValueError(f'{name} {key} differs from the calibrated robot model')


def nominal_signature():
    out = {}
    for i, (xyz, rpy, axis) in enumerate(zip(JOINT_ORIGIN_XYZ, JOINT_ORIGIN_RPY, JOINT_AXIS), 1):
        out[f'joint{i}'] = dict(parent='base_link' if i == 1 else f'link{i-1}',
                               child=f'link{i}', type='revolute', xyz=list(xyz),
                               rotation=_rpy_matrix(*rpy).tolist(), axis=list(axis))
    out['link6_tool0_joint'] = dict(parent='link6', child='tool0', type='fixed',
                                  xyz=list(LINK6_TO_TOOL0_XYZ),
                                  rotation=np.eye(3).tolist(), axis=[0., 0., 0.])
    return out


def export_candidate_urdf(xml, offsets):
    offsets = valid_offsets(offsets)
    require_matching_model(xml, nominal_signature())  # prevent applying twice
    root = ET.fromstring(xml)
    for i, angle in enumerate(offsets, 1):
        if angle == 0:
            continue
        origin = root.find(f"joint[@name='joint{i}']/origin")
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
        R = _rpy_matrix(*rpy) @ _axis_matrix(JOINT_AXIS[i-1], angle)
        # URDF RPY = Rz(yaw) Ry(pitch) Rx(roll); this is far from gimbal lock.
        updated = (math.atan2(R[2, 1], R[2, 2]),
                   math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])),
                   math.atan2(R[1, 0], R[0, 0]))
        origin.set('rpy', ' '.join(format(v, '.17g') for v in updated))
    return ET.tostring(root, encoding='unicode', xml_declaration=True)
