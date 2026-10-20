#!/usr/bin/env python3
"""Eye-to-hand and eye-in-hand calibration using OpenCV.

Eye-in-hand: camera mounted on the tool, board fixed in the workspace.
Input T_base_tool directly and T_camera_board; output T_tool_camera.
The closed chain T_base_tool * T_tool_camera * T_camera_board is constant.

OpenCV calibrateHandEye (eye-to-hand):
  - Board rigidly mounted on end-effector; camera fixed in workspace.
  - Input R_gripper2base / t_gripper2base  :=  inv(T_base_tool)   # ^{g}T_b
  - Input R_target2cam  / t_target2cam      :=  T_camera_board     # ^{c}T_t
  - Output R_cam2gripper / t_cam2gripper   :=  T_base_camera      # ^{b}T_c
    (with the inverted-pose trick, gripper slot is the base)

Closed chain residual for every sample i:
  T_base_tool_i * T_tool_board  =  T_base_camera * T_camera_board_i

References:
  OpenCV calib3d::calibrateHandEye docs (Tsai/Park/Horaud/Andreff/Daniilidis)
  TUM CAMPAR hand-eye notes
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

from x5a_handeye.transforms import (
    R_to_quat,
    inv_T,
    pose_to_T,
    quat_to_R,
    se3_errors,
    T_to_pose,
)


def as_T(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        xyz = value.get("xyz") or value.get("position")
        quat = value.get("quat_xyzw") or value.get("orientation") or value.get("quat")
        if xyz is None or quat is None:
            raise ValueError(f"pose dict missing xyz/quat: keys={list(value.keys())}")
        return pose_to_T(xyz, quat)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"unsupported transform shape {arr.shape}")
    return arr


def normalize_samples(raw: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    for s in raw:
        if s.get("T_base_tool") is None or s.get("T_camera_board") is None:
            continue
        try:
            item = {
                "index": s.get("index", len(out)),
                "joints": s.get("joints"),
                "T_base_tool": as_T(s["T_base_tool"]),
                "T_camera_board": as_T(s["T_camera_board"]),
                "quality": s.get("quality"),
            }
            out.append(item)
        except Exception as exc:
            print(f"skip sample index={s.get('index')}: {exc}")
    return out


def average_T(Ts: Sequence[np.ndarray]) -> np.ndarray:
    t_mean = np.mean([T[:3, 3] for T in Ts], axis=0)
    quats: List[np.ndarray] = []
    for T in Ts:
        q = np.array(R_to_quat(T[:3, :3]), dtype=float)
        if quats and float(np.dot(quats[0], q)) < 0:
            q = -q
        quats.append(q)
    q_mean = np.mean(np.asarray(quats), axis=0)
    q_mean /= np.linalg.norm(q_mean)
    if q_mean[3] < 0:
        q_mean = -q_mean
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*q_mean.tolist())
    T[:3, 3] = t_mean
    return T


def estimate_T_tool_board(samples: Sequence[dict], T_base_camera: np.ndarray) -> np.ndarray:
    """T_tool_board = inv(T_base_tool) * T_base_camera * T_camera_board."""
    Ts = []
    for s in samples:
        Ts.append(inv_T(s["T_base_tool"]) @ T_base_camera @ s["T_camera_board"])
    return average_T(Ts)


def evaluate(samples: Sequence[dict], T_base_camera: np.ndarray, T_tool_board: np.ndarray,
             calibration_type: str = 'eye_to_hand') -> dict:
    terr, rerr = [], []
    for s in samples:
        # left: robot chain, right: vision chain (both express board in base)
        if calibration_type == 'eye_in_hand':
            T_base_board_robot = T_tool_board  # constant T_base_board
            T_base_board_vision = s['T_base_tool'] @ T_base_camera @ s['T_camera_board']
        else:
            T_base_board_robot = s["T_base_tool"] @ T_tool_board
            T_base_board_vision = T_base_camera @ s["T_camera_board"]
        te, re = se3_errors(T_base_board_robot, T_base_board_vision)
        terr.append(te)
        rerr.append(re)
    terr = np.asarray(terr, dtype=float)
    rerr = np.asarray(rerr, dtype=float)
    return {
        "translation_mean_m": float(np.mean(terr)),
        "translation_median_m": float(np.median(terr)),
        "translation_max_m": float(np.max(terr)),
        "rotation_mean_deg": float(np.mean(rerr)),
        "rotation_median_deg": float(np.median(rerr)),
        "rotation_max_deg": float(np.max(rerr)),
        "per_sample_t": terr.tolist(),
        "per_sample_r": rerr.tolist(),
    }


def pack_opencv_eye_to_hand(samples: Sequence[dict]):
    """OpenCV eye-to-hand inputs: inverted robot pose + target2cam."""
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for s in samples:
        T_gripper_base = inv_T(s["T_base_tool"])  # ^{g}T_b
        T_target_cam = s["T_camera_board"]  # ^{c}T_t
        R_g2b.append(T_gripper_base[:3, :3])
        t_g2b.append(T_gripper_base[:3, 3].reshape(3, 1))
        R_t2c.append(T_target_cam[:3, :3])
        t_t2c.append(T_target_cam[:3, 3].reshape(3, 1))
    return R_g2b, t_g2b, R_t2c, t_t2c


def solve_opencv_handeye(samples: Sequence[dict], method_flag,
                         calibration_type: str = 'eye_to_hand') -> Tuple[np.ndarray, np.ndarray]:
    if calibration_type not in ('eye_to_hand', 'eye_in_hand'):
        raise ValueError(f'unsupported calibration type {calibration_type}')
    if calibration_type == 'eye_in_hand':
        R_g2b = [s['T_base_tool'][:3, :3] for s in samples]
        t_g2b = [s['T_base_tool'][:3, 3].reshape(3, 1) for s in samples]
        R_t2c = [s['T_camera_board'][:3, :3] for s in samples]
        t_t2c = [s['T_camera_board'][:3, 3].reshape(3, 1) for s in samples]
    else:
        R_g2b, t_g2b, R_t2c, t_t2c = pack_opencv_eye_to_hand(samples)
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=method_flag
    )
    # Official eye-to-hand trick: output slot is ^{b}T_c
    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = R_cam2base
    T_base_camera[:3, 3] = np.asarray(t_cam2base, dtype=float).reshape(3)
    if not np.all(np.isfinite(T_base_camera)):
        raise ValueError('solver returned a non-finite transform')
    if (not np.allclose(R_cam2base.T @ R_cam2base, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R_cam2base), 1.0, atol=1e-5)):
        raise ValueError('solver returned an invalid rotation')
    if calibration_type == 'eye_in_hand':
        # The OpenCV output is T_tool_camera; the board is fixed in base.
        T_tool_board = average_T([s['T_base_tool'] @ T_base_camera @ s['T_camera_board']
                                 for s in samples])
    else:
        T_tool_board = estimate_T_tool_board(samples, T_base_camera)
    return T_base_camera, T_tool_board


def motion_diversity_report(samples: Sequence[dict]) -> dict:
    """Rotation-axis diversity check (OpenCV recommends non-parallel motions)."""
    if len(samples) < 3:
        return {"ok": False, "n_motions": 0}
    axes = []
    angles = []
    for i in range(len(samples) - 1):
        dT = inv_T(samples[i]["T_base_tool"]) @ samples[i + 1]["T_base_tool"]
        rvec, _ = cv2.Rodrigues(dT[:3, :3])
        ang = float(np.linalg.norm(rvec))
        if ang < np.deg2rad(5.0):
            continue
        axis = (rvec.reshape(3) / ang).tolist()
        axes.append(axis)
        angles.append(np.rad2deg(ang))
    if len(axes) < 2:
        return {"ok": False, "n_motions": len(axes), "angles_deg": angles}
    # max |dot| between unit axes should be clearly < 1 (not all parallel)
    max_abs_dot = 0.0
    min_abs_dot = 1.0
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            dot = abs(float(np.dot(axes[i], axes[j])))
            max_abs_dot = max(max_abs_dot, dot)
            min_abs_dot = min(min_abs_dot, dot)
    return {
        "ok": min_abs_dot <= 0.95,
        "n_motions": len(axes),
        "angles_deg_mean": float(np.mean(angles)),
        "angles_deg_max": float(np.max(angles)),
        "max_abs_axis_dot": float(max_abs_dot),
        "min_abs_axis_dot": float(min_abs_dot),
        "parallel_warning": bool(min_abs_dot > 0.95),
    }


def reject_outliers(
    samples: List[dict], T_bc: np.ndarray, T_tb: np.ndarray, max_reject: int,
    calibration_type: str = 'eye_to_hand'
) -> List[dict]:
    err = evaluate(samples, T_bc, T_tb, calibration_type)
    per_t = np.asarray(err["per_sample_t"], dtype=float)
    if len(samples) <= 12 or max_reject <= 0:
        return samples
    thr = float(
        np.median(per_t)
        + 2.5 * (np.percentile(per_t, 75) - np.percentile(per_t, 25) + 1e-9)
    )
    keep = [s for s, t in zip(samples, per_t) if t <= thr]
    n_reject = len(samples) - len(keep)
    if 0 < n_reject <= max_reject and len(keep) >= 12:
        print(f"outlier reject: {n_reject} samples thr={thr*1000:.2f} mm")
        return keep
    return samples


def compare_methods(samples: Sequence[dict], calibration_type: str = 'eye_to_hand') -> List[tuple]:
    methods = []
    for name in [
        "CALIB_HAND_EYE_TSAI",
        "CALIB_HAND_EYE_PARK",
        "CALIB_HAND_EYE_HORAUD",
        "CALIB_HAND_EYE_ANDREFF",
        "CALIB_HAND_EYE_DANIILIDIS",
    ]:
        if hasattr(cv2, name):
            methods.append((name.replace("CALIB_HAND_EYE_", ""), getattr(cv2, name)))

    results = []
    for name, flag in methods:
        try:
            T_bc, T_tb = solve_opencv_handeye(samples, flag, calibration_type)
            err = evaluate(samples, T_bc, T_tb, calibration_type)
            results.append((err["translation_mean_m"], name, T_bc, T_tb, err))
            print(
                f"{name:10s}  t_mean={err['translation_mean_m']*1000:7.2f} mm  "
                f"t_med={err['translation_median_m']*1000:7.2f} mm  "
                f"t_max={err['translation_max_m']*1000:7.2f} mm  "
                f"r_mean={err['rotation_mean_deg']:6.2f} deg"
            )
        except Exception as exc:
            print(f"{name:10s}  FAIL {exc}")
    results.sort(key=lambda x: x[0])
    return results


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="OpenCV eye-to-hand / eye-in-hand solver")
    ap.add_argument('--calibration-type', choices=['eye_to_hand', 'eye_in_hand'],
                    help='Default: sample metadata, or eye_to_hand for legacy samples')
    ap.add_argument('--camera-frame', help='Optical frame used by the board detector')
    ap.add_argument('--camera-root-frame', default='camera_link')
    ap.add_argument(
        "--samples",
        required=True,
    )
    ap.add_argument(
        "--output",
        help='Default: a separate mode-specific result in this package config directory',
    )
    ap.add_argument("--max-reject", type=int, default=5)
    ap.add_argument("--holdout", type=int, help="random hold-out count (default: 6 for eye-in-hand, 0 for legacy)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument('--pass-deg', type=float, default=2.0,
                    help='mean rotation residual threshold for PASS (degrees)')
    ap.add_argument(
        "--pass-mm",
        type=float,
        default=10.0,
        help="mean translation residual threshold for PASS (mm)",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.samples).read_text())
    metadata = data if isinstance(data, dict) else {}
    recorded_type = metadata.get('calibration_type')
    calibration_type = args.calibration_type or recorded_type or 'eye_to_hand'
    if calibration_type not in ('eye_to_hand', 'eye_in_hand'):
        raise SystemExit(f'unsupported calibration type {calibration_type}')
    if recorded_type and args.calibration_type and recorded_type != args.calibration_type:
        raise SystemExit('sample calibration type differs from the requested mode; collect a new dataset')
    if args.holdout is None:
        args.holdout = 6 if calibration_type == 'eye_in_hand' else 0
    samples = normalize_samples(metadata.get('samples', []) if metadata else data)
    frames = {s.get('source', {}).get('board_frame_id') for s in metadata.get('samples', [])}
    frames.discard(None)
    frames.discard('')
    if len(frames) > 1:
        raise SystemExit('samples contain multiple camera optical frames')
    recorded_frame = metadata.get('camera_frame') or (next(iter(frames)) if frames else None)
    if frames and recorded_frame not in frames:
        raise SystemExit('camera_frame metadata differs from the sampled optical frame')
    if args.camera_frame and recorded_frame and args.camera_frame != recorded_frame:
        raise SystemExit('requested camera optical frame differs from the recorded frame')
    camera_frame = args.camera_frame or recorded_frame or 'camera_color_optical_frame'
    output = args.output or str(Path(__file__).resolve().parents[1]/'config'/
        ('handeye_result_eye_in_hand.yaml' if calibration_type == 'eye_in_hand' else 'handeye_result.yaml'))
    print(f"loaded {len(samples)} valid samples from {args.samples}")
    if len(samples) < 10:
        raise SystemExit(f"need >= 10 samples, got {len(samples)}")
    if args.holdout < 0 or (args.holdout > 0 and len(samples) - args.holdout < 12):
        raise SystemExit('holdout requires at least 12 remaining training samples')
    if args.pass_mm <= 0 or args.pass_deg <= 0:
        raise SystemExit('validation thresholds must be positive')

    div = motion_diversity_report(samples)
    print("motion diversity:", div)
    if div.get("parallel_warning"):
        print("WARNING: rotation axes nearly parallel — add more orientation variety")

    rng = np.random.default_rng(args.seed)
    holdout: List[dict] = []
    train = list(samples)
    if args.holdout > 0 and len(samples) - args.holdout >= 12:
        idx = np.arange(len(samples))
        rng.shuffle(idx)
        hold_set = set(idx[: args.holdout].tolist())
        train = [samples[i] for i in range(len(samples)) if i not in hold_set]
        holdout = [samples[i] for i in range(len(samples)) if i in hold_set]
        print(f"split train={len(train)} holdout={len(holdout)}")

    print(f"--- OpenCV {calibration_type} solvers (train) ---")
    results = compare_methods(train, calibration_type)
    if not results:
        raise SystemExit("all solvers failed")

    # outlier rejection + re-solve
    _, _, T_bc0, T_tb0, _ = results[0]
    used = reject_outliers(train, T_bc0, T_tb0, args.max_reject, calibration_type)
    if len(used) != len(train):
        print("--- re-solve after outlier reject ---")
        results = compare_methods(used, calibration_type)
        if not results:
            raise SystemExit("all solvers failed after reject")

    tmean, name, T_bc, T_tb, err = results[0]
    div = motion_diversity_report(used)
    hold_err = evaluate(holdout, T_bc, T_tb, calibration_type) if holdout else None
    all_err = evaluate(samples, T_bc, T_tb, calibration_type)

    xyz, quat = T_to_pose(T_bc)
    tb_xyz, tb_quat = T_to_pose(T_tb)

    pass_train = (err["translation_mean_m"] * 1000.0 < args.pass_mm
                  and err['rotation_mean_deg'] < args.pass_deg)
    pass_hold = True
    if hold_err is not None:
        pass_hold = (hold_err["translation_mean_m"] * 1000.0 < args.pass_mm * 1.5
                     and hold_err['rotation_mean_deg'] < args.pass_deg * 1.5)
    # require enough inliers on full set
    all_t = np.asarray(all_err["per_sample_t"])
    n_good = int(np.sum(all_t < 0.015))
    pass_coverage = n_good >= max(10, int(0.6 * len(samples)))
    verdict = "PASS" if (pass_train and pass_hold and pass_coverage and div.get('ok')) else "FAIL"

    out = {
        "calibration_type": calibration_type,
        "method_convention": f"opencv_calibrateHandEye_{calibration_type}",
        "parent_frame": metadata.get('tool_frame', 'tool0') if calibration_type == 'eye_in_hand'
                        else metadata.get('base_frame', 'base_link'),
        "camera_frame": camera_frame,
        "camera_root_frame": args.camera_root_frame,
        "tool_frame": metadata.get("tool_frame", "tool0"),
        "base_frame": metadata.get('base_frame', 'base_link'),
        "board_frame": "calibration_board",
        "translation": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
        "rotation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
        ('T_tool_camera' if calibration_type == 'eye_in_hand' else 'T_base_camera'): T_bc.tolist(),
        ('T_base_board' if calibration_type == 'eye_in_hand' else 'T_tool_board'): T_tb.tolist(),
        ('base_board_translation' if calibration_type == 'eye_in_hand' else 'tool_board_translation'):
            {"x": tb_xyz[0], "y": tb_xyz[1], "z": tb_xyz[2]},
        ('base_board_rotation' if calibration_type == 'eye_in_hand' else 'tool_board_rotation'): {
            "x": tb_quat[0],
            "y": tb_quat[1],
            "z": tb_quat[2],
            "w": tb_quat[3],
        },
        "solver": name,
        "samples": {
            "total": len(samples),
            "train": len(train),
            "used": len(used),
            "holdout": len(holdout),
            "source": str(args.samples),
            "collection": metadata.get("collection", ""),
            "used_indices": [s["index"] for s in used],
        },
        "motion_diversity": div,
        "error": {
            "translation_mean_m": err["translation_mean_m"],
            "translation_median_m": err["translation_median_m"],
            "translation_max_m": err["translation_max_m"],
            "rotation_mean_deg": err["rotation_mean_deg"],
            "rotation_max_deg": err["rotation_max_deg"],
        },
        "holdout_error": None
        if hold_err is None
        else {
            "translation_mean_m": hold_err["translation_mean_m"],
            "translation_median_m": hold_err["translation_median_m"],
            "translation_max_m": hold_err["translation_max_m"],
            "rotation_mean_deg": hold_err["rotation_mean_deg"],
            "rotation_max_deg": hold_err["rotation_max_deg"],
        },
        "all_samples_error": {
            "translation_mean_m": all_err["translation_mean_m"],
            "translation_median_m": all_err["translation_median_m"],
            "translation_max_m": all_err["translation_max_m"],
            "rotation_mean_deg": all_err["rotation_mean_deg"],
            "n_under_15mm": n_good,
        },
        "solver_comparison": [
            {
                "solver": r[1],
                "translation_mean_m": r[4]["translation_mean_m"],
                "rotation_mean_deg": r[4]["rotation_mean_deg"],
            }
            for r in results
        ],
        "verdict": verdict,
        "pass_criteria_mm": args.pass_mm,
        'pass_criteria_deg': args.pass_deg,
        'observation_geometry': metadata.get('observation_geometry'),
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(yaml.safe_dump(out, sort_keys=False))

    print("SELECTED", name)
    print('T_tool_camera' if calibration_type == 'eye_in_hand' else 'T_base_camera',
          'xyz', [round(v, 6) for v in xyz], 'quat', [round(v, 6) for v in quat])
    print('T_base_board' if calibration_type == 'eye_in_hand' else 'T_tool_board',
          'xyz', [round(v, 6) for v in tb_xyz], 'quat', [round(v, 6) for v in tb_quat])
    print(
        f"train  mean={err['translation_mean_m']*1000:.2f} mm  "
        f"max={err['translation_max_m']*1000:.2f} mm  "
        f"r_mean={err['rotation_mean_deg']:.2f} deg"
    )
    if hold_err is not None:
        print(
            f"holdout mean={hold_err['translation_mean_m']*1000:.2f} mm  "
            f"max={hold_err['translation_max_m']*1000:.2f} mm"
        )
    print(
        f"all    mean={all_err['translation_mean_m']*1000:.2f} mm  "
        f"n<15mm={n_good}/{len(samples)}"
    )
    print(calibration_type.upper()+" CALIBRATION:", verdict)
    print("saved", output)
    if verdict == 'FAIL':
        raise SystemExit(2)


if __name__ == "__main__":
    main()
