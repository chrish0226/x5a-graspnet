"""GraspNet grasp-pose service (port 127.0.0.1:8301).

Two HTTP contracts:

POST /grasp      - simple contract: files `image`/`depth`, multiplier depth_scale,
                   returns flat lists. Used for quick manual tests.

POST /v1/grasp   - contract expected by the ROS `x5a_remote_grasp` package:
                   files `rgb`/`depth`, divisor depth_scale (e.g. 1000 for mm),
                   fields prompt/fx/fy/cx/cy/min_depth/max_depth/threshold/
                   mask_dilation_px/num_point/top_k/collision_thresh/voxel_size/
                   target_distance_thresh/max_gripper_width.
                   Returns {convention, frame, grasps:[{score, width_m, height_m,
                   depth_m, translation_m, rotation_matrix, transform_camera_grasp}]}
                   in the camera optical frame (x right, y down, z forward).
"""
import base64
import io
import hashlib
import logging
import json
import math
import os
import sys
import threading
import time
import urllib.request

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "dataset"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

from graspnet import GraspNet, pred_decode  # noqa: E402
from collision_detector import ModelFreeCollisionDetector  # noqa: E402
from data_utils import CameraInfo, create_point_cloud_from_depth_image  # noqa: E402
from grasp_quality import nms_indices, opening_measurement  # noqa: E402

CHECKPOINT = os.path.join(ROOT_DIR, "checkpoint-rs.tar")  # zip-format torch.save, name misleading
SAM_URL = os.environ.get("SAM_URL", "http://127.0.0.1:8300/segment_text")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="GraspNet service", version="1.1")
_net = None
_lock = threading.Lock()
_model_lock = threading.Lock()
LOGGER = logging.getLogger('graspnet.quality')


def _warmup():
    def load():
        try:
            get_net()
        except Exception:
            pass
    threading.Thread(target=load, daemon=True).start()


@app.on_event("startup")
def _startup():
    _warmup()


def get_net():
    with _model_lock:
        return _load_net()


def _load_net():
    global _net
    if _net is None:
        net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                       cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                       is_training=False)
        net.to(DEVICE)
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        _net = net
    return _net


@app.get("/grasp_health")
def grasp_health():
    return {"status": "ok", "model_loaded": _net is not None, "device": DEVICE}


def _read_image(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _read_depth_raw(data: bytes) -> np.ndarray:
    try:
        return np.load(io.BytesIO(data)).astype(np.float32)
    except ValueError:
        return np.array(Image.open(io.BytesIO(data))).astype(np.float32)


def _decode_mask(b64: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))))


def rgb_png_bytes(image: np.ndarray) -> bytes:
    bio = io.BytesIO()
    Image.fromarray(image).save(bio, format="PNG")
    return bio.getvalue()


def _sam_call(image: np.ndarray, prompt: str, threshold: float):
    """Call the SAM service, return the raw instance list."""
    buf = rgb_png_bytes(image)
    boundary = "----gspboundary"
    parts = []
    fields = [("file", ("frame.png", buf, "image/png")),
              ("prompt", (None, prompt, None)),
              ("threshold", (None, str(threshold), None)),
              ("viz", (None, "false", None))]
    for name, (filename, value, mime) in fields:
        head = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
        if filename is not None:
            head += f'; filename="{filename}"\r\nContent-Type: {mime}'
        parts.append((head + "\r\n\r\n").encode() + (value if isinstance(value, bytes) else value.encode()) + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(SAM_URL, data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("objects", [])


def _sam_mask(image: np.ndarray, prompt: str, threshold: float, color: str | None,
              reference_prompt: str | None = None, relation: str | None = None):
    """Segment the target; with a reference_prompt, do two-stage relational
    grounding: SAM has no reliable exclusion semantics for "X next to Y"
    (it latches onto the reference), so the relation is resolved geometrically:
    on/inside -> target center inside the reference box; next_to -> nearest
    non-reference instance."""
    objs = _sam_call(image, prompt, threshold)
    if not objs:
        raise HTTPException(404, f"SAM found no object for prompt '{prompt}'")

    if reference_prompt:
        ref_objs = _sam_call(image, reference_prompt, threshold)
        if not ref_objs:
            raise HTTPException(404, f"SAM found no reference for '{reference_prompt}'")
        ref = max(ref_objs, key=lambda o: o["score"])
        rx1, ry1, rx2, ry2 = ref["box_xyxy"]

        def center(o):
            x1, y1, x2, y2 = o["box_xyxy"]
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        rel = (relation or "next_to").strip().lower()
        if rel in {"on", "above", "inside", "in", "on_top"}:
            cands = [
                o for o in objs
                if rx1 + 8 <= center(o)[0] <= rx2 - 8 and ry1 + 8 <= center(o)[1] <= ry2 - 8
            ]
            if not cands:
                raise HTTPException(404, f"no '{prompt}' on the {reference_prompt}")
            cands.sort(key=lambda o: -o["score"])
            best = cands[0]
            tag = "reference_on"
        else:
            margin = 12.0
            ref_c = center(ref)
            cands = []
            for o in objs:
                c = center(o)
                if rx1 - margin <= c[0] <= rx2 + margin and ry1 - margin <= c[1] <= ry2 + margin:
                    continue  # 参照物自身
                cands.append((math.hypot(c[0] - ref_c[0], c[1] - ref_c[1]), o))
            if not cands:
                raise HTTPException(404, f"no '{prompt}' found near the {reference_prompt}")
            cands.sort(key=lambda t: t[0])
            best = cands[0][1]
            tag = "reference"
        mask = (_decode_mask(best["mask_png_b64"]) > 127).astype(np.uint8)
        return mask, {"selected_by": tag, "reference": reference_prompt, "relation": rel,
                      "box_xyxy": best["box_xyxy"], "sam_score": best["score"]}

    if color:
        import colorsys
        rgb_names = {"red": (255, 30, 30), "green": (30, 220, 60), "blue": (40, 60, 255),
                     "orange": (255, 140, 0), "yellow": (250, 230, 30), "white": (240, 240, 240),
                     "black": (25, 25, 25), "purple": (140, 40, 200), "gray": (128, 128, 128)}
        r, g, b = rgb_names.get(color.lower(), (255, 255, 255))
        want_h, want_s, want_v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        scored = []
        for o in objs:
            m = (_decode_mask(o["mask_png_b64"]) > 127).astype(np.uint8)
            if m.sum() == 0:
                continue
            mean = image[m > 0].reshape(-1, 3).mean(0) / 255.0
            h, s, v = colorsys.rgb_to_hsv(*mean)
            if want_s < 0.2:  # white/black/gray: match value, ignore hue
                dist = abs(v - want_v) + max(0.0, s - 0.3)
            else:
                hue_dist = min(abs(h - want_h), 1.0 - abs(h - want_h))
                dist = hue_dist + max(0.0, 0.25 - s) + 0.3 * abs(v - want_v)
            scored.append((dist, o["score"], m))
        if not scored:
            raise HTTPException(404, "no instance matched the color filter")
        scored.sort(key=lambda x: (x[0], -x[1]))
        return scored[0][2], {"selected_by": "color", "sam_score": scored[0][1]}
    best = max(objs, key=lambda o: o["score"])
    return (_decode_mask(best["mask_png_b64"]) > 127).astype(np.uint8), \
        {"selected_by": "score", "sam_score": best["score"]}


class _GraspGroupLite:
    """Duck-typed stand-in for graspnetAPI.GraspGroup (only what collision
    detector accesses: translations, rotation_matrices, heights, depths, widths)."""

    def __init__(self, arr: np.ndarray):
        self.grasp_group_array = arr
        self.translations = arr[:, 13:16]
        self.rotation_matrices = arr[:, 4:13].reshape(-1, 3, 3)
        self.heights = arr[:, 2]
        self.depths = arr[:, 3]
        self.widths = arr[:, 1]


def _run_graspnet(rgb, depth_m, mask, fx, fy, cx, cy,
                  num_point=20000, top_k=10, collision_thresh=0.0, voxel_size=0.01,
                  max_gripper_width=0.088, min_depth=0.05, max_depth=2.0,
                  mask_collision=None, target_distance_thresh=0.006):
    """mask -> filtered cloud -> GraspNet -> filtered + NMS'd 17-col rows."""
    m = mask.astype(bool)
    cloud = create_point_cloud_from_depth_image(
        depth_m, CameraInfo(rgb.shape[1], rgb.shape[0], fx, fy, cx, cy, 1.0), organized=False)
    color_flat = (rgb.reshape(-1, 3) / 255.0).astype(np.float32)
    mask_flat = m.reshape(-1) & (depth_m.reshape(-1) >= min_depth) & (depth_m.reshape(-1) <= max_depth)
    cloud_masked = cloud[mask_flat]
    m_col = mask if mask_collision is None else mask_collision
    col_flat = np.asarray(m_col).reshape(-1).astype(bool) & (depth_m.reshape(-1) >= min_depth) & (depth_m.reshape(-1) <= max_depth)
    target_cloud = cloud[col_flat]
    # Collision checking needs the scene, not only the segmented target.
    valid = np.isfinite(depth_m.reshape(-1)) & (depth_m.reshape(-1) >= min_depth) & (depth_m.reshape(-1) <= max_depth)
    cloud_collision = cloud[valid]
    if len(target_cloud) < 30:
        raise HTTPException(422, 'target has too few valid depth points')
    low, high = np.min(target_cloud, axis=0)-0.18, np.max(target_cloud, axis=0)+0.18
    cloud_collision = cloud_collision[np.all((cloud_collision >= low) & (cloud_collision <= high), axis=1)]
    object_band = None
    if len(target_cloud) >= 30:
        zc = target_cloud[:, 2].astype(np.float64)
        z_lo, z_hi = np.percentile(zc, [5, 95])
        cxm, cym = float(np.mean(target_cloud[:, 0])), float(np.mean(target_cloud[:, 1]))
        object_band = [[round(cxm, 5), round(cym, 5), round(float(z_lo), 5)],
                       [round(cxm, 5), round(cym, 5), round(float(z_hi), 5)]]
    color_masked = color_flat[mask_flat]
    if len(cloud_masked) < 50:
        raise HTTPException(422, f"too few valid points in mask ({int(mask_flat.sum())})")
    if not 2048 <= num_point <= 50000 or not 1 <= top_k <= 64:
        raise HTTPException(400, 'num_point/top_k outside supported bounds')
    seed = int.from_bytes(hashlib.blake2b(cloud_masked.tobytes(), digest_size=4).digest(), 'little')
    rng = np.random.default_rng(seed)
    if len(cloud_masked) >= num_point:
        idxs = rng.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs = np.concatenate([np.arange(len(cloud_masked)),
                               rng.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)])
    cloud_sampled = torch.from_numpy(cloud_masked[idxs][np.newaxis].astype(np.float32)).to(DEVICE)
    net = get_net()
    with torch.no_grad(), _lock:
        torch.manual_seed(seed)
        end_points = net({"point_clouds": cloud_sampled, "cloud_colors": color_masked[idxs]})
        preds = pred_decode(end_points)
    arr = preds[0].detach().cpu().numpy()
    arr = arr[np.all(np.isfinite(arr), axis=1) & (arr[:, 0] > 0) & (arr[:, 3] > 0)]
    raw_count = len(arr)
    # The predictor may consume context, but a returned seed must belong to the
    # ORIGINAL target. Contact-GraspNet uses the same separation of concerns.
    t = arr[:, 13:16]
    px = np.rint(t[:, 0]/t[:, 2]*fx+cx).astype(int)
    py = np.rint(t[:, 1]/t[:, 2]*fy+cy).astype(int)
    inside = (px >= 0) & (px < rgb.shape[1]) & (py >= 0) & (py < rgb.shape[0])
    belongs = np.zeros(len(arr), dtype=bool)
    belongs[inside] = np.asarray(m_col, dtype=bool)[py[inside], px[inside]]
    from scipy.spatial import cKDTree
    distance, _ = cKDTree(target_cloud).query(t)
    belongs &= distance <= max(0.001, float(target_distance_thresh))
    arr = arr[belongs]
    target_count = len(arr)
    # Do geometry filtering BEFORE top-k/NMS, preserving viable directions.
    required = _required_openings(arr, target_cloud)
    keep = [i for i, opening in enumerate(required)
            if opening is not None and 0.003 < opening <= max_gripper_width-0.006]
    arr = arr[keep]
    if len(arr):
        # Width is an opening, not a measured object size. Use target geometry;
        # robot-side actual-jaw validation remains the final authority.
        arr[:, 1] = [min(max_gripper_width, required[i]+0.006) for i in keep]
    if collision_thresh > 0:
        try:
            mfcd = ModelFreeCollisionDetector(cloud_collision, voxel_size=voxel_size)
            target_detector = ModelFreeCollisionDetector(target_cloud, voxel_size=voxel_size)
            checked = []
            for start in range(0, len(arr), 32):
                batch = arr[start:start+32]
                opened = batch.copy()
                opened[:, 1] = max_gripper_width
                collision = mfcd.detect(_GraspGroupLite(opened), approach_dist=0.05, collision_thresh=collision_thresh)
                _, empty = target_detector.detect(_GraspGroupLite(batch), return_empty_grasp=True, empty_thresh=0.01)
                checked.append(batch[~collision & ~empty])
            arr = np.concatenate(checked) if checked else arr
        except Exception as exc:
            LOGGER.exception('scene collision validation failed')
            raise HTTPException(500, 'collision validation failed; no unchecked grasps returned') from exc
    keep = nms_indices(arr, top_k)
    LOGGER.info('quality raw=%d target=%d valid=%d returned=%d seed=%d', raw_count, target_count, len(arr), len(keep), seed)
    return arr[keep], int(mask_flat.sum()), _required_openings(arr[keep], target_cloud), object_band, seed


def _required_openings(rows, cloud, approach_span=0.05, min_points=30):
    """Opening centered on each grasp, within its actual finger volume.

    A narrow off-center object still needs twice the furthest jaw-axis extent.
    Insufficient geometry returns None, which is rejected before NMS/top-k.
    """
    out = []
    for row in rows:
        try:
            R = row[4:13].reshape(3, 3).astype(float)
            t = row[13:16].astype(float)
            measured = opening_measurement(cloud, R, t, float(row[3]), 0.06,
                                           float(row[2]), min_points)
            if measured is None:
                out.append(None)
                continue
            out.append(round(measured['required_opening_m'], 5))
        except Exception:
            out.append(None)
    return out


def _rows_to_simple(rows):
    out = []
    for row in rows:
        r = row[4:13].reshape(3, 3)
        out.append({
            "score": round(float(row[0]), 4),
            "width": round(float(row[1]), 4),
            "depth": round(float(row[3]), 4),
            "translation": [round(float(v), 5) for v in row[13:16]],
            "rotation": [round(float(v), 6) for v in row[4:13]],
            "approach": [round(float(v), 5) for v in r[:, 0]],
        })
    return out


def _rows_to_v1(rows, frame_name, required=None):
    grasps = []
    for row_i, row in enumerate(rows):
        R = row[4:13].reshape(3, 3).astype(float)
        t = row[13:16].astype(float)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        grasps.append({
            "score": round(float(row[0]), 5),
            "width_m": round(float(row[1]), 5),
            "required_opening_m": (required[row_i] if required and row_i < len(required) else None),
            "height_m": round(float(row[2]), 5),
            "depth_m": round(float(row[3]), 5),
            "translation_m": [round(float(v), 6) for v in t],
            "rotation_matrix": [[round(float(v), 6) for v in r] for r in R],
            "transform_camera_grasp": [[round(float(v), 6) for v in r] for r in T],
        })
    return {
        "convention": {"x": "right", "y": "down", "z": "forward", "units": "metres",
                       "approach_axis": "grasp_x", "frame": frame_name},
        "frame": frame_name,
        "grasps": grasps,
    }


def _erode_or_dilate(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if not -32 <= dilate_px <= 32:
        raise HTTPException(400, 'mask dilation outside supported bounds')
    try:
        from scipy.ndimage import binary_dilation, binary_erosion
        if dilate_px > 0:
            return binary_dilation(mask, iterations=dilate_px).astype(np.uint8)
        if dilate_px < 0:
            return binary_erosion(mask, iterations=-dilate_px).astype(np.uint8)
    except Exception as exc:
        raise HTTPException(500, 'mask morphology failed') from exc
    return mask


async def _shared_grasp(image: UploadFile, depth: UploadFile, prompt, threshold,
                        fx, fy, cx, cy, depth_mul, color, mask_file,
                        min_depth, max_depth, dilate_px, num_point, top_k,
                        collision_thresh, voxel_size, max_gripper_width,
                        reference_prompt: str | None = None, relation: str | None = None,
                        target_distance_thresh=0.006):
    t0 = time.time()
    numeric = [fx, fy, cx, cy, depth_mul, min_depth, max_depth,
               collision_thresh, voxel_size, max_gripper_width, target_distance_thresh]
    if (not np.all(np.isfinite(numeric)) or min(fx, fy, depth_mul) <= 0
            or not 0 < min_depth < max_depth or not 0 < voxel_size <= 0.03
            or not 0.006 < max_gripper_width <= 0.12
            or not 0 < target_distance_thresh <= 0.03 or collision_thresh < 0):
        raise HTTPException(400, 'invalid camera/depth/gripper parameters')
    rgb = _read_image(await image.read())
    depth_m = _read_depth_raw(await depth.read()) * depth_mul
    if depth_m.shape[:2] != rgb.shape[:2]:
        raise HTTPException(400, f"depth {depth_m.shape[:2]} and image {rgb.shape[:2]} resolutions differ")
    if mask_file is not None:
        mask = (_decode_mask(base64.b64encode(await mask_file.read()).decode()) > 0).astype(np.uint8)
        instance = {"selected_by": "client_mask"}
    else:
        if not prompt:
            raise HTTPException(400, "need either `mask` or `prompt`")
        mask, instance = _sam_mask(rgb, prompt, threshold, color, reference_prompt, relation)
    if mask.shape != depth_m.shape:
        raise HTTPException(400, 'target mask and depth resolutions differ')
    mask_raw = np.array(mask, copy=True)
    mask = _erode_or_dilate(mask, dilate_px)
    rows, npts, required, band, seed = _run_graspnet(rgb, depth_m, mask, fx, fy, cx, cy, num_point, top_k,
                                               collision_thresh, voxel_size, max_gripper_width, min_depth, max_depth,
                                               mask_collision=mask_raw, target_distance_thresh=target_distance_thresh)
    target = (mask_raw > 0) & np.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)
    mask_b64 = base64.b64encode(rgb_png_bytes((target.astype(np.uint8)*255))).decode()
    return rows, npts, required, band, instance, round(time.time() - t0, 3), mask_b64, seed


@app.post("/grasp")
async def grasp(image: UploadFile = File(...), depth: UploadFile = File(...),
                prompt: str = Form(""), threshold: float = Form(0.5),
                fx: float = Form(...), fy: float = Form(...),
                cx: float = Form(...), cy: float = Form(...),
                depth_scale: float = Form(0.001), color: str | None = Form(None),
                mask: UploadFile | None = File(None),
                collision_thresh: float = Form(0.0)):
    rows, npts, required, band, instance, latency, mask_b64, seed = await _shared_grasp(
        image, depth, prompt, threshold, fx, fy, cx, cy, depth_scale, color, mask,
        0.05, 2.0, -4, 20000, 10, collision_thresh, 0.01, 0.088)
    return {"grasps": _rows_to_simple(rows), "num_points": npts,
            "instance": instance, "latency_s": latency, "frame": "camera"}


@app.post("/v1/locate")
async def locate(rgb: UploadFile = File(...), depth: UploadFile = File(...),
                 prompt: str = Form(...),
                 fx: float = Form(...), fy: float = Form(...),
                 cx: float = Form(...), cy: float = Form(...),
                 depth_scale: float = Form(1000.0),
                 threshold: float = Form(0.45), mask_dilation_px: int = Form(0),
                 min_depth: float = Form(0.10), max_depth: float = Form(1.50)):
    """只定位不抓取：返回目标的 3D 中心与顶面高度（相机系），供放置流程用。
    prompt 支持 "| ref: <参照>; rel: <on/next_to>" 关系编码。"""
    rgb_img = _read_image(await rgb.read())
    if (not np.all(np.isfinite([fx, fy, cx, cy, depth_scale, min_depth, max_depth]))
            or min(fx, fy, depth_scale) <= 0 or not 0 < min_depth < max_depth):
        raise HTTPException(400, 'invalid camera/depth parameters')
    depth_m = _read_depth_raw(await depth.read()) / depth_scale
    if depth_m.shape[:2] != rgb_img.shape[:2]:
        raise HTTPException(400, "depth 和 image 分辨率不一致")
    reference_prompt, relation = None, None
    if " | ref: " in prompt:
        prompt, ref_part = prompt.split(" | ref: ", 1)
        prompt = prompt.strip()
        reference_prompt, relation = ref_part, None
        if "; rel: " in ref_part:
            reference_prompt, relation = ref_part.split("; rel: ", 1)
            relation = relation.strip() or None
        reference_prompt = reference_prompt.strip()
    mask, instance = _sam_mask(rgb_img, prompt, threshold, None, reference_prompt, relation)
    # Localization geometry always comes from the original instance mask.
    cloud = create_point_cloud_from_depth_image(
        depth_m, CameraInfo(rgb_img.shape[1], rgb_img.shape[0], fx, fy, cx, cy, 1.0), organized=False)
    sel = mask.reshape(-1) > 0
    sel &= (depth_m.reshape(-1) >= min_depth) & (depth_m.reshape(-1) <= max_depth)
    pts = cloud[sel]
    if len(pts) < 30:
        raise HTTPException(422, f"目标点数过少 ({int(sel.sum())})")
    center = np.median(pts, axis=0)
    top_z = np.percentile(pts[:, 2], 95)
    top = center.copy()
    top[2] = top_z
    return {"instance": instance, "num_points": int(sel.sum()),
            "center_cam": [round(float(v), 4) for v in center],
            "top_cam": [round(float(v), 4) for v in top],
            "extent_z": round(float(np.percentile(pts[:, 2], 95) - np.percentile(pts[:, 2], 5)), 4),
            "target_mask_png_b64": base64.b64encode(rgb_png_bytes((sel.reshape(depth_m.shape).astype(np.uint8)*255))).decode(),
            "quality_schema": 2}

@app.post("/v1/grasp")
async def grasp_v1(rgb: UploadFile = File(...), depth: UploadFile = File(...),
                   prompt: str = Form(...),
                   fx: float = Form(...), fy: float = Form(...),
                   cx: float = Form(...), cy: float = Form(...),
                   depth_scale: float = Form(1000.0),
                   min_depth: float = Form(0.10), max_depth: float = Form(1.50),
                   threshold: float = Form(0.45), mask_dilation_px: int = Form(2),
                   num_point: int = Form(20000), top_k: int = Form(20),
                   collision_thresh: float = Form(0.01), voxel_size: float = Form(0.01),
                   target_distance_thresh: float = Form(0.006),
                   max_gripper_width: float = Form(0.088),
                   mask: UploadFile | None = File(None)):
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise HTTPException(400, 'invalid depth scale')
    # 关系指代编码在 prompt 里，按 " | " 段解析：
    #   "cube | ref: red cube; rel: next_to"                        → 抓取参照定位
    #   "orange cube | place: white box; prel: on"                  → 放置意图：
    #     reference 属于放置目标，抓取定位按分数选择（目标在哪就抓哪）
    reference_prompt = None
    relation = None
    has_place = " | place: " in prompt
    parts = [p.strip() for p in prompt.split(" | ") if p.strip()]
    if len(parts) > 1 or (parts and parts[0] != prompt.strip()):
        prompt = parts[0]
    for seg in parts[1:]:
        if seg.startswith("ref: "):
            body = seg[5:].strip()
            if "; rel: " in body:
                reference_prompt, relation = [s.strip() for s in body.split("; rel: ", 1)]
            else:
                reference_prompt = body or None
        # place: 段由 /v1/grasp 的 place 参数链路处理，这里忽略
    if has_place:
        reference_prompt = None
        relation = None
    try:
        rows, npts, required, band, instance, latency, mask_b64, seed = await _shared_grasp(
            rgb, depth, prompt, threshold, fx, fy, cx, cy, 1.0 / depth_scale, None, mask,
            min_depth, max_depth, mask_dilation_px, num_point, top_k,
            collision_thresh, voxel_size, max_gripper_width, reference_prompt, relation, target_distance_thresh)
    except HTTPException as exc:
        # 兼容 Qwen 字段串扰：把放置目标误填进 reference 时（"放到X上面"），
        # 抓取目标的参照定位必然 404，此时退化为按分数选择抓取目标。
        if not (has_place and reference_prompt and exc.status_code == 404):
            raise
        rows, npts, required, band, instance, latency, mask_b64, seed = await _shared_grasp(
            rgb, depth, prompt, threshold, fx, fy, cx, cy, 1.0 / depth_scale, None, mask,
            min_depth, max_depth, mask_dilation_px, num_point, top_k,
            collision_thresh, voxel_size, max_gripper_width, None, None, target_distance_thresh)
        instance = dict(instance or {})
        instance["selected_by"] = "score_fallback"
        instance["note"] = "reference 是放置目标，已忽略"
    out = _rows_to_v1(rows, "camera_color_optical_frame", required)
    out["object_band_cam"] = band
    out.update({"num_points": npts, "instance": instance, "latency_s": latency,
                "target_distance_thresh_applied": True,
                "target_mask_png_b64": mask_b64, "quality_schema": 2,
                "sampling_seed": seed, "collision_checked": collision_thresh > 0})
    return out
