"""Measured IK reachability table for the X5A arm.

`scripts/probe_reachability.py` sweeps /compute_ik over a workspace grid and
writes this format; the plan node reads it back to order its pitch/yaw ladder
(so the first attempt is the one most likely to solve) and to explain an
unreachable target in words the operator can act on.

The table is an accelerator and a wording source only: the node still validates
every waypoint with real IK and planning. It goes stale when the tool mount or
the calibrated scene changes, so re-run the probe after touching either.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TABLE_VERSION = 1

# 经验规律（实测自 /compute_ik，见 memory）：竖直工具 pitch1.45 只能到 z≈0.10，
# 1.15 到 0.20，0.90 到 0.25+。没有实测表或表格未覆盖该点时按此选起始 pitch。
EMPIRICAL_PITCH_Z = ((0.10, 1.45), (0.20, 1.15))


def cell_key(x: float, y: float, z: float, digits: int = 3) -> str:
    return f"{round(x, digits):.{digits}f},{round(y, digits):.{digits}f},{round(z, digits):.{digits}f}"


def parse_cell_key(key: str) -> Optional[Tuple[float, float, float]]:
    parts = str(key).split(",")
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def empirical_pitch_order(pitches: Sequence[float], z: float) -> List[float]:
    """Order the pitch ladder so the pitch most likely to solve comes first."""
    preferred = EMPIRICAL_PITCH_Z[-1][1]
    for limit, pitch in EMPIRICAL_PITCH_Z:
        if z <= limit:
            preferred = pitch
            break
    return sorted(pitches, key=lambda pitch: (abs(pitch - preferred), -pitch))


def _entry_ok(row: object, yaw_rel_deg: Optional[int] = None) -> bool:
    if not isinstance(row, dict):
        return False
    if yaw_rel_deg is None:
        return any(isinstance(entry, dict) and bool(entry.get("ok")) for entry in row.values())
    entry = row.get(str(int(yaw_rel_deg)))
    return isinstance(entry, dict) and bool(entry.get("ok"))


@dataclass
class ReachabilityTable:
    """Loaded table plus lookup helpers. An empty table declines every lookup."""

    results: Dict[str, Dict[str, Dict[str, dict]]] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)
    path: Optional[Path] = None
    _coords: Optional[List[Tuple[Tuple[float, float, float], str]]] = field(
        default=None, repr=False
    )

    @classmethod
    def load(cls, path: Path) -> "ReachabilityTable":
        try:
            payload = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return cls()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
            return cls()
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        return cls(results=payload["results"], meta=meta, path=Path(path))

    def __bool__(self) -> bool:
        return bool(self.results)

    def coordinates(self) -> List[Tuple[Tuple[float, float, float], str]]:
        """(coordinate, cell key) pairs, parsed once and cached."""
        if self._coords is None:
            pairs = []
            for key in self.results:
                coord = parse_cell_key(key)
                if coord is not None:
                    pairs.append((coord, key))
            self._coords = pairs
        return self._coords

    def nearest_cell(
        self, x: float, y: float, z: float, max_distance: float = 0.06
    ) -> Optional[Dict[str, Dict[str, dict]]]:
        """Nearest measured cell within max_distance, or None when uncovered."""
        best_distance = None
        best_key = None
        for coord, key in self.coordinates():
            distance = math.dist((x, y, z), coord)
            if best_distance is None or distance < best_distance:
                best_distance, best_key = distance, key
        if best_distance is None or best_distance > max_distance:
            return None
        cell = self.results.get(best_key or "")
        return cell if isinstance(cell, dict) else None

    def measured(self, x: float, y: float, z: float) -> List[float]:
        """Pitches measured solvable at (x, y, z); empty when nothing is."""
        cell = self.nearest_cell(x, y, z)
        if cell is None:
            return []
        return [float(pitch) for pitch, row in cell.items() if _entry_ok(row)]

    def order_pitches(
        self, pitches: Sequence[float], x: float, y: float, z: float
    ) -> List[float]:
        """Measured-solvable pitches first, the rest by the empirical z rule."""
        base = list(pitches)
        empirical = empirical_pitch_order(base, z)
        if not self:
            return empirical
        solvable = set(self.measured(x, y, z))
        if not solvable:
            return empirical
        return sorted(base, key=lambda pitch: (0 if pitch in solvable else 1, empirical.index(pitch)))

    def order_yaws(
        self, yaw_rels: Sequence[int], x: float, y: float, z: float, pitch: float
    ) -> List[int]:
        """Yaws measured solvable at this pitch first, original order otherwise."""
        if not self:
            return list(yaw_rels)
        cell = self.nearest_cell(x, y, z)
        if cell is None:
            return list(yaw_rels)
        row = cell.get(f"{pitch:.2f}")
        if not isinstance(row, dict):
            return list(yaw_rels)
        return sorted(
            yaw_rels,
            key=lambda rel: (
                0 if _entry_ok(row, rel) else 1,
                list(yaw_rels).index(rel),
            ),
        )

    def max_z_for_pitch(self, pitch: float) -> Optional[float]:
        summary = self.meta.get("summary")
        if isinstance(summary, dict):
            per_pitch = summary.get("pitch_max_z")
            if isinstance(per_pitch, dict):
                value = per_pitch.get(f"{pitch:.2f}")
                if isinstance(value, (int, float)):
                    return float(value)
        heights = [
            coord[2]
            for coord, key in self.coordinates()
            if _entry_ok((self.results.get(key) or {}).get(f"{pitch:.2f}"))
        ]
        return max(heights) if heights else None

    def height_cap(self) -> Optional[float]:
        caps = [self.max_z_for_pitch(pitch) for pitch in (0.90, 1.15, 1.45)]
        values = [cap for cap in caps if cap is not None]
        return max(values) if values else None

    def pitch_capabilities(self) -> Dict[str, float]:
        """{pitch: 实测最高可达 z}，用于启动日志和「够不到」的措辞。"""
        caps: Dict[str, float] = {}
        for coord, key in self.coordinates():
            cell = self.results.get(key)
            if not isinstance(cell, dict):
                continue
            for pitch, row in cell.items():
                if _entry_ok(row):
                    caps[pitch] = max(caps.get(pitch, 0.0), coord[2])
        return dict(sorted(caps.items()))

    def verdict(self, x: float, y: float, z: float) -> str:
        """One line the operator can act on when a target cannot be planned."""
        position = f"目标位于 ({x:.3f}, {y:.3f}, {z:.3f})"
        if not self:
            return f"{position}，无实测可达性数据，规划失败"
        cap = self.height_cap()
        cap_text = f"，实测整体最高可达 z≈{cap:.2f} m" if cap is not None else ""
        if self.nearest_cell(x, y, z) is None:
            return f"{position} 附近没有实测数据（网格未覆盖{cap_text}），规划失败"
        if self.measured(x, y, z):
            return f"{position}，实测可达"
        return (
            f"{position} 实测不可达{cap_text}；"
            "把物体往机械臂正前方挪近一些，或降低目标高度"
        )


def build_summary(results: Dict[str, Dict[str, Dict[str, dict]]]) -> Dict[str, object]:
    """Aggregate max reachable height per pitch plus a radius profile."""
    pitch_max_z: Dict[str, float] = {}
    totals: Dict[str, int] = {}
    hits: Dict[str, int] = {}
    radius_max_z: Dict[float, float] = {}
    for key, cell in results.items():
        coord = parse_cell_key(key)
        if coord is None or not isinstance(cell, dict):
            continue
        x, y, z = coord
        radius = round(math.hypot(x, y), 3)
        for pitch, row in cell.items():
            totals[pitch] = totals.get(pitch, 0) + 1
            if _entry_ok(row):
                hits[pitch] = hits.get(pitch, 0) + 1
                pitch_max_z[pitch] = max(pitch_max_z.get(pitch, 0.0), z)
                radius_max_z[radius] = max(radius_max_z.get(radius, 0.0), z)
    return {
        "pitch_max_z": dict(sorted(pitch_max_z.items())),
        "feasible_ratio_by_pitch": {
            pitch: round(hits.get(pitch, 0) / totals[pitch], 3) for pitch in sorted(totals)
        },
        "radius_max_z": [
            {"r": radius, "z_max": radius_max_z[radius]} for radius in sorted(radius_max_z)
        ],
    }
