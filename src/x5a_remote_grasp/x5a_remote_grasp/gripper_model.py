"""CAD-backed stock X5A jaw model, shared by robot and inference service.

The existing tool0/hand-eye frame never moves. Mesh bodies use their URDF
origins at zero jaw opening; the two joint translations are applied explicitly.
Contact is limited to the flat inner faces, not the taper or the rack.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def model_digest(data):
    payload = {key: value for key, value in data.items() if key != 'model_id'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def gravity_aligned_rotation(up_camera):
    """Rotate a camera cloud into a right-handed Z-up frame without translation."""
    up = np.asarray(up_camera, dtype=float)
    if (up.shape != (3,) or not np.isfinite(up).all()
            or not np.isclose(np.linalg.norm(up), 1., atol=.01)):
        raise ValueError('unit world-up vector in camera frame required')
    up = up/np.linalg.norm(up)
    horizontal = np.array([1., 0., 0.]) if abs(up[0]) <= .95 else np.array([0., 1., 0.])
    horizontal -= (horizontal @ up)*up
    horizontal /= np.linalg.norm(horizontal)
    return np.stack((horizontal, np.cross(up, horizontal), up))


@dataclass(frozen=True)
class MeshBody:
    name: str
    vertices: np.ndarray
    planes: np.ndarray
    jaw_axis: np.ndarray

    def __post_init__(self):
        for name in ('vertices', 'planes', 'jaw_axis'):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.isfinite(value).all():
                raise ValueError('nonfinite CAD body')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3
                or self.planes.ndim != 2 or self.planes.shape[1] != 4
                or self.jaw_axis.shape != (3,)):
            raise ValueError('invalid CAD body shape')
        if (len(self.vertices) < 4 or len(self.planes) < 4
                or not np.allclose(np.linalg.norm(self.planes[:, :3], axis=1), 1)
                or np.max(self.vertices @ self.planes[:, :3].T + self.planes[:, 3]) > 1e-7):
            raise ValueError('invalid CAD convex envelope')

    def swept_planes(self, approach):
        if approach <= 0:
            return self.planes
        # Exact convex sweep of the conservative CAD convex envelope.
        from scipy.spatial import ConvexHull
        shifted = self.vertices - np.array([approach, 0., 0.])
        equations = ConvexHull(np.vstack((self.vertices, shifted))).equations
        return np.unique(np.round(equations, 12), axis=0)

    def inside(self, points, half_gap, *, approach=0., closing_to=None):
        """Convex hull of CAD body swept along approach OR jaw closing axis."""
        translation = self.jaw_axis * half_gap
        lo, hi = self.vertices.min(0)+translation, self.vertices.max(0)+translation
        lo[0] -= approach
        if closing_to is not None:
            end = self.jaw_axis * closing_to
            lo = np.minimum(lo, self.vertices.min(0)+end)
            hi = np.maximum(hi, self.vertices.max(0)+end)
        eligible = np.all((points >= lo-1e-9) & (points <= hi+1e-9), axis=1)
        indices = np.flatnonzero(eligible)
        result = np.zeros(len(points), bool)
        if not len(indices):
            return result
        planes = self.swept_planes(approach)
        if closing_to is not None and np.any(self.jaw_axis):
            # Minkowski segment: clipping the translated body by only its
            # original planes is not generally exact. Build its actual sweep.
            from scipy.spatial import ConvexHull
            planes = np.unique(np.round(ConvexHull(np.vstack((
                self.vertices+translation, self.vertices+self.jaw_axis*closing_to
            ))).equations, 12), axis=0)
            shifted = points[indices]
        else:
            shifted = points[indices]-translation
        # Chunk planes to avoid allocating large clouds x dense CAD matrices.
        keep = np.ones(len(indices), bool)
        for start in range(0, len(planes), 32):
            p = planes[start:start+32]
            keep &= np.all(shifted @ p[:, :3].T+p[:, 3] <= 1e-9, axis=1)
            if not np.any(keep):
                break
        result[indices[keep]] = True
        return result


class GripperModel:
    def __init__(self, data):
        if data.get('schema') != 1 or data.get('model_id') != model_digest(data):
            raise ValueError('gripper model schema/hash mismatch')
        self.data = data
        self.model_id = data['model_id']
        self.tip_offset = float(data['contact']['tip_x_m'])
        self.pad_length = float(data['contact']['length_m'])
        self.pad_height = float(data['contact']['height_m'])
        self.max_width = float(data['max_width_m'])
        # A successful empirical insertion reference is not a physical tip.
        self.native_reference = float(data['native_grasp_reference_offset_m'])
        self.contact_validation = data.get('contact_validation', {})
        if (data['tcp_frame'] != 'wrist_preview/tool0'
                or not .01 < self.tip_offset < .10
                or not .005 < self.pad_length <= self.tip_offset
                or not .004 < self.pad_height < .04
                or not .006 < self.max_width <= .12
                or not 0 < self.native_reference <= self.tip_offset):
            raise ValueError('invalid stock-jaw contact geometry')
        self.bodies = tuple(MeshBody(row['name'], row['vertices_tool0_m'],
            row['planes_tool0'], row['jaw_axis']) for row in data['bodies'])
        if {b.name for b in self.bodies} != {'link6', 'link7', 'link8'}:
            raise ValueError('missing gripper CAD bodies')
        # Prepared once per immutable model, shared by parallel geometry checks.
        self._sweeps = {}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def geometry(self):
        try:
            from .grasp_quality import GripperGeometry
        except ImportError:
            from grasp_quality import GripperGeometry
        return GripperGeometry(tip_offset=self.tip_offset,
            finger_length=float(self.data['finger_length_m']),
            finger_thickness=float(self.data['finger_thickness_m']),
            finger_height=self.pad_height, pad_length=self.pad_length,
            max_width=self.max_width, opening_margin=.006, cad_model=self)

    def validate_sources(self, root):
        root = Path(root)
        for source, digest in self.data['source_hashes'].items():
            if hashlib.sha256((root/source).read_bytes()).hexdigest() != digest:
                raise ValueError('gripper CAD/URDF changed; regenerate model: '+source)
        for row in self.data['bodies']:
            path = root / row['collision_mesh']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['collision_mesh_sha256']:
                raise ValueError('generated gripper collision mesh changed')

    def apply_to_urdf(self, xml, root):
        """Install the SAME convex envelopes into MoveIt; preserve every joint."""
        import xml.etree.ElementTree as ET
        self.validate_sources(root)
        robot = ET.fromstring(xml)
        for row in self.data['bodies']:
            link = robot.find("link[@name='%s']" % row['name'])
            if link is None:
                raise ValueError('gripper link missing from robot description')
            geometry = link.find('collision/geometry')
            for child in list(geometry):
                geometry.remove(child)
            # resource_retriever requires a URI; a bare absolute path silently
            # leaves MoveIt's link without collision geometry.
            ET.SubElement(geometry, 'mesh', filename=(Path(root)/row['collision_mesh']).resolve().as_uri())
        return ET.tostring(robot, encoding='unicode')

    def _collision_masks(self, points, opening, approach, close_to):
        open_mask = np.zeros(len(points), bool)
        closing_mask = np.zeros(len(points), bool)
        for body in self.bodies:
            # Cache a sweep model rather than recomputing its hull per candidate.
            key = (body.name, float(approach))
            swept = self._sweeps.get(key)
            if swept is None:
                vertices = np.vstack((body.vertices, body.vertices-[approach, 0, 0])) if approach else body.vertices
                swept = MeshBody(body.name, vertices, body.swept_planes(approach), body.jaw_axis)
                self._sweeps[key] = swept
            open_mask |= swept.inside(points, opening/2)
            if np.count_nonzero(open_mask) >= 3:
                return open_mask, closing_mask
            if np.any(body.jaw_axis):
                closing_mask |= body.inside(points, opening/2, closing_to=close_to/2)
        return open_mask, closing_mask

    def contact_centers(self, target, center):
        """Observed contact patches for a thin elongated tool, never depth fill.

        The whole-mask centre of pliers often lies in the gap between handles.
        Every returned patch still goes through the same contact/CAD/IK checks.
        """
        target = np.asarray(target)
        if np.ptp(np.percentile(target[:, 2], [5, 95])) > .035:
            return []
        values, axes = np.linalg.eigh(np.cov(target[:, :2].T))
        if values[-1] < 2*values[0]:
            return []
        along = target[:, :2] @ axes[:, -1]
        if np.ptp(np.percentile(along, [2, 98])) < .07:
            return []
        across = target[:, :2] @ axes[:, 0]
        result = []
        for fraction in (.2, .4, .6, .8):
            indices = np.flatnonzero(abs(along-np.quantile(along, fraction)) < .006)
            indices = indices[np.argsort(across[indices])]
            for group in np.split(indices, np.flatnonzero(np.diff(across[indices]) > .008)+1):
                if len(group) < 30:
                    continue
                point = np.array(center, copy=True)
                point[:2] = np.median(target[group, :2], axis=0)
                if np.linalg.norm(point-center) < .012 or any(np.linalg.norm(point-p) < .012 for p in result):
                    continue
                result.append(point)
        return result[:8]

    @staticmethod
    def thin_tool_axis(target):
        target = np.asarray(target)
        if np.ptp(np.percentile(target[:, 2], [5, 95])) >= .035:
            return None
        values, axes = np.linalg.eigh(np.cov(target[:, :2].T))
        along = target[:, :2] @ axes[:, -1]
        return (axes[:, -1] if values[-1] > 2*values[0]
            and np.ptp(np.percentile(along, [2, 98])) > .07 else None)

    def contact_measurement(self, target, tool, min_points=12):
        """Measure ONLY the inset physical contact patch, without a noise halo."""
        if self.thin_tool_axis(target) is None:
            try:
                from .grasp_quality import opening_measurement
            except ImportError:
                from grasp_quality import opening_measurement
            return opening_measurement(target, tool[:3, :3], tool[:3, 3],
                self.tip_offset, self.pad_length, self.pad_height, min_points)
        v = self.contact_validation
        local = (np.asarray(target)-tool[:3, 3]) @ tool[:3, :3]
        selected = ((local[:, 0] >= self.tip_offset-self.pad_length+v['rear_inset_m'])
            & (local[:, 0] <= self.tip_offset-v['tip_inset_m'])
            & (abs(local[:, 2]) <= self.pad_height/2-v['height_inset_m']))
        points = local[selected]
        if len(points) < min_points:
            return None
        low, high = np.percentile(points[:, 1], [2, 98])
        return dict(required_opening_m=float(2*max(abs(low), abs(high))),
            span_m=float(high-low), center_offset_m=float((low+high)/2),
            contact_points=len(points),
            contact_insertion_p10_m=float(np.percentile(self.tip_offset-points[:, 0], 10)))

    def contact_quality(self, target, tool, min_points=12):
        """Necessary pinch geometry; this does not claim force closure/friction."""
        from scipy.spatial import cKDTree
        target, tool = np.asarray(target), np.asarray(tool)
        measure = self.contact_measurement(target, tool, min_points)
        if measure is None:
            return dict(valid=False, reason='insufficient_inset_contact')
        if measure['span_m'] < .003:
            return dict(measure, valid=False, reason='insufficient_closing_span')
        axis = self.thin_tool_axis(target)
        if axis is None:
            return dict(measure, valid=True, reason='ok')
        if abs(measure['center_offset_m']) > .002:
            return dict(measure, valid=False, reason='unbalanced_contact')
        v = self.contact_validation
        # A long, thin tool must be pinched ACROSS its shaft/handles. Counting
        # points at two distant ends along a handle is not a usable jaw angle.
        closing = tool[:3, 1]
        horizontal = np.linalg.norm(closing[:2])
        angle = float(np.degrees(np.arccos(np.clip(
            abs(closing[:2] @ axis)/max(horizontal, 1e-12), 0, 1))))
        measure['closing_to_tool_axis_deg'] = angle
        if angle < v['thin_tool_min_closing_angle_deg']:
            return dict(measure, valid=False, reason='closing_along_tool')
        if abs(closing[2]) > np.sin(np.radians(v['thin_tool_max_vertical_closing_deg'])):
            return dict(measure, valid=False, reason='closing_toward_table')
        local = (target-tool[:3, 3]) @ tool[:3, :3]
        core = ((local[:, 0] >= self.tip_offset-self.pad_length+v['rear_inset_m'])
            & (local[:, 0] <= self.tip_offset-v['tip_inset_m'])
            & (abs(local[:, 2]) <= self.pad_height/2-v['height_inset_m']))
        points = local[core]
        low, high = np.percentile(points[:, 1], [2, 98])
        left = points[points[:, 1] <= low+.002][:, [0, 2]]
        right = points[points[:, 1] >= high-.002][:, [0, 2]]
        if min(len(left), len(right)) < min_points:
            return dict(measure, valid=False, reason='one_sided_contact')
        distance, _ = cKDTree(left).query(right)
        paired = right[distance <= v['pair_distance_m']]
        cells = np.unique(np.floor(paired/.002).astype(np.int64), axis=0)
        measure['paired_contact_cells'] = len(cells)
        if len(cells) < v['min_pair_voxels']:
            return dict(measure, valid=False, reason='nonopposing_contact')
        return dict(measure, valid=True, reason='ok')

    def assess(self, target, scene, tool, geometry, approach):
        try:
            from .grasp_quality import opening_measurement, PreparedGraspScene, voxel_cloud
        except ImportError:
            from grasp_quality import opening_measurement, PreparedGraspScene, voxel_cloud
        r, t = tool[:3, :3], tool[:3, 3]
        # Contact uses only the flat 15mm front jaw face and the height shared
        # by both faces across that entire interval; never the sloped backing.
        measure = self.contact_quality(target, tool, geometry.min_contact_points)
        if not measure['valid']:
            return dict(measure, geometry_model_id=self.model_id)
        if measure['required_opening_m']+geometry.opening_margin > self.max_width:
            return dict(measure, valid=False, reason='opening_or_offcenter', geometry_model_id=self.model_id)
        points = scene.points if isinstance(scene, PreparedGraspScene) else voxel_cloud(scene, .003)
        local = (points-t) @ r
        opened, closing = self._collision_masks(local, self.max_width, approach,
            measure['required_opening_m'])
        collisions = int(opened.sum())
        # Only intended target contact at the true flat face may touch during
        # closure. The taper, rack, palm and neighbouring objects stay obstacles.
        from scipy.spatial import cKDTree
        pad = ((local[:, 0] >= self.tip_offset-self.pad_length-1e-9)
            & (local[:, 0] <= self.tip_offset+geometry.noise_margin)
            & (np.abs(local[:, 2]) <= self.pad_height/2+geometry.noise_margin)
            & (np.abs(local[:, 1]) <= measure['required_opening_m']/2+geometry.noise_margin))
        indices = np.flatnonzero(closing & pad)
        if len(indices):
            distance, _ = cKDTree(target).query(points[indices])
            closing[indices[distance <= .002]] = False
        closing_collisions = int(closing.sum())
        reason = ('scene_collision' if collisions >= geometry.min_collision_voxels else
                  'closing_collision' if closing_collisions >= geometry.min_collision_voxels else 'ok')
        return dict(measure, valid=reason == 'ok', reason=reason,
            collision_voxels=collisions, closing_collision_voxels=closing_collisions,
            open_width_m=measure['required_opening_m']+geometry.opening_margin,
            geometry_model_id=self.model_id)
