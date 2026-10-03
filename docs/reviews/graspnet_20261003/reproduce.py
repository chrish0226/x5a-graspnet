"""Read-only, CPU-only checks of archived GraspNet code and RGB-D captures.

No ROS clients, robot commands, network calls, or model loading.
Run: /usr/bin/python3 docs/reviews/graspnet_20261003/reproduce.py
"""
import ast
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


HERE = Path(__file__).resolve().parent
CAPTURES = Path('/home/quella/.local/share/x5a_remote_grasp/captures')


def read_json(path):
    return json.loads(path.read_text())


def aperture_function():
    tree = ast.parse((HERE / 'grasp_server.snapshot.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_required_openings')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<server-function>', 'exec'), namespace)
    return namespace['_required_openings']


def check_aperture():
    # Object is 60 mm wide but displaced +40 mm relative to the grasp origin.
    # At this fixed origin, symmetric jaws need 140 mm, not the 60 mm span.
    cloud = np.zeros((100, 3))
    cloud[:, 1] = np.linspace(0.01, 0.07, len(cloud))
    row = np.zeros(17)
    row[4:13] = np.eye(3).reshape(-1)
    measured = aperture_function()(row[None, :], cloud)[0]
    needed = 2 * np.max(np.abs(cloud[:, 1]))
    assert np.isclose(measured, 0.06) and np.isclose(needed, 0.14)
    return {'reported_span_mm': measured * 1000, 'fixed_center_required_mm': needed * 1000}


def check_nms():
    # Mirrors server NMS: no rotation is examined.
    positions = np.array([[0, 0, 0], [0, 0, 0]])
    rotations = [np.eye(3), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])]
    keep = []
    for i in (0, 1):
        if all(np.linalg.norm(positions[i] - positions[j]) >= 0.02 for j in keep):
            keep.append(i)
    angle = np.degrees(np.arccos((np.trace(rotations[0].T @ rotations[1]) - 1) / 2))
    assert keep == [0] and np.isclose(angle, 90)
    return {'input_count': 2, 'kept_count': len(keep), 'rotation_difference_deg': angle}


def check_band(transform):
    # All points lie on a horizontal plane. The server's compressed camera-Z
    # endpoints should not imply object thickness, but they do.
    x, y = np.meshgrid(np.linspace(0.16, 0.24, 100), np.linspace(0.24, 0.36, 100))
    base = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.035)])
    camera = (base - transform[:3, 3]) @ transform[:3, :3]
    endpoints = np.repeat(camera.mean(axis=0)[None, :], 2, axis=0)
    endpoints[:, 2] = np.percentile(camera[:, 2], [5, 95])
    reconstructed = endpoints @ transform[:3, :3].T + transform[:3, 3]
    apparent = abs(reconstructed[1, 2] - reconstructed[0, 2])
    assert apparent > 0.015
    return {'actual_height_range_mm': float(np.ptp(base[:, 2]) * 1000),
            'compressed_band_range_mm': float(apparent * 1000)}


def capture_metrics(record):
    directory = CAPTURES / record['capture']
    metadata = read_json(directory / 'camera_info.json')
    response = read_json(directory / 'grasp_response.json')
    validation = read_json(directory / 'tf_validation.json')
    best = max(record['objects'], key=lambda o: o['score'])
    mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(best['mask_png_b64'])))) > 127
    depth = cv2.imread(str(directory / 'depth.png'), cv2.IMREAD_UNCHANGED).astype(float)
    depth *= metadata['depth_scale_m_per_unit']
    v, u = np.indices(depth.shape)
    cloud = np.stack([(u - metadata['cx']) * depth / metadata['fx'],
                     (v - metadata['cy']) * depth / metadata['fy'], depth], axis=-1)
    valid = (depth >= 0.1) & (depth <= 1.5)
    raw = mask & valid
    dilated = binary_dilation(mask, iterations=8) & valid
    assert int(dilated.sum()) == response['num_points'], 'SAM replay differs from archived inference'
    transform = np.asarray(validation['transform_base_camera'])
    base = cloud @ transform[:3, :3].T + transform[:3, 3]
    outside = dilated & ~raw
    distance_pixels = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    candidates = []
    for grasp in response['grasps']:
        t = np.asarray(grasp['translation_m'])
        px = int(round(t[0] / t[2] * metadata['fx'] + metadata['cx']))
        py = int(round(t[1] / t[2] * metadata['fy'] + metadata['cy']))
        candidates.append({'score': grasp['score'], 'depth_m': grasp['depth_m'],
                           'inside_raw_mask': bool(mask[py, px]),
                           'mask_distance_px': float(distance_pixels[py, px])})
    out = {'capture': record['capture'], 'sam_score': best['score'],
           'raw_points': int(raw.sum()), 'inference_points': int(dilated.sum()),
           'outside_raw_mask_points': int(outside.sum()),
           'outside_fraction': float(outside.sum() / dilated.sum()),
           'outside_base_z_p5_p50_p95_mm': np.percentile(base[outside, 2], [5, 50, 95]).tolist(),
           'candidates': candidates,
           'actual_raw_base_z_p5_p95_mm': (np.percentile(base[raw, 2], [5, 95]) * 1000).tolist(),
           'compressed_band_mm': (np.asarray(validation['object_center_band_base']) * 1000).tolist()}
    out['outside_base_z_p5_p50_p95_mm'] = [value * 1000 for value in out['outside_base_z_p5_p50_p95_mm']]
    attempts = read_json(directory / 'ik_poses.json')
    last = attempts[-1]
    out['last_attempt_kind'] = last['kind']
    if last['label'].startswith('#'):
        index = int(last['label'][1:])
        original = np.asarray(validation['candidates'][index]['transform_base_grasp'])[:3, :3]
        xq, yq, zq, wq = last['poses']['GRASP'][3:]
        executed = np.array([
            [1-2*(yq*yq+zq*zq), 2*(xq*yq-zq*wq), 2*(xq*zq+yq*wq)],
            [2*(xq*yq+zq*wq), 1-2*(xq*xq+zq*zq), 2*(yq*zq-xq*wq)],
            [2*(xq*zq-yq*wq), 2*(yq*zq+xq*wq), 1-2*(xq*xq+yq*yq)]])
        # Closing axis is unoriented for a symmetric parallel jaw.
        dot = abs(float(original[:, 1] @ executed[:, 1]))
        out['last_attempt_closing_axis_change_deg'] = float(np.degrees(np.arccos(np.clip(dot, -1, 1))))
    return out, transform


def main():
    rows = read_json(HERE / 'sam_replay.json')
    captures = []
    for row in rows:
        summary, transform = capture_metrics(row)
        captures.append(summary)
    result = {'aperture_counterexample': check_aperture(), 'nms_counterexample': check_nms(),
              'height_band_counterexample': check_band(transform), 'captures': captures}
    (HERE / 'evidence.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
