"""Off-centre grasp, tilted release and small-support regressions."""
import numpy as np
import pytest
from x5a_remote_grasp.geometry import rpy_rotation
from x5a_remote_grasp.place_geometry import (
    object_in_tool, placement_tool, support_surface, supported_center, preserves_upright,
    support_prism,
)


def box_cloud():
    x, y, z = np.meshgrid(np.linspace(-0.015, 0.015, 16),
                         np.linspace(-0.015, 0.015, 16), np.linspace(0, 0.03, 16))
    return np.c_[x.ravel(), y.ravel(), z.ravel()]+[0.12, 0.25, 0.006]


@pytest.mark.parametrize('pitch,yaw', [(1.45, 1.27), (0.9, 0.6), (1.1, -1.2)])
def test_bottom_and_object_center_align_even_after_tool_rotation(pitch, yaw):
    grasp = np.eye(4)
    grasp[:3, :3] = rpy_rotation(0, 1.45, 1.1)
    grasp[:3, 3] = [0.128, 0.24, 0.046]  # off-centre tool, not object centre
    local = object_in_tool(box_cloud(), grasp)
    R = rpy_rotation(0, pitch, yaw)
    surface = [0.10, 0.33, 0.036]
    placed = placement_tool(local, R, surface)
    world = local @ R.T+placed[:3, 3]
    lo, hi = np.percentile(world, [2, 98], axis=0)
    np.testing.assert_allclose((lo[:2]+hi[:2])/2, surface[:2], atol=1e-12)
    assert lo[2] == pytest.approx(0.038)


def test_top_face_center_ignores_asymmetric_side_cloud_and_rejects_edge_offsets():
    x, y = np.meshgrid(np.linspace(-0.015, 0.015, 30), np.linspace(-0.015, 0.015, 30))
    top = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.036)]
    angle = np.pi/4
    R = rpy_rotation(0, 0, angle)
    top = top @ R.T+[0.10, 0.33, 0]
    side = np.tile(top[:30], (3, 1)); side[:, 2] = 0.020
    surface, hull = support_surface(np.vstack([top, side]))
    np.testing.assert_allclose(surface, [0.10, 0.33, 0.036], atol=0.0005)
    assert supported_center(surface[:2], hull)
    assert not supported_center(surface[:2]+[0.03, 0], hull)
    assert not supported_center(surface[:2]+[0.018, 0.018], hull)


def test_stacking_rejects_pitch_ladder_that_tilts_the_object():
    grasp = rpy_rotation(0, 1.45, 1.1)
    assert preserves_upright(grasp, rpy_rotation(0, 0, 0.4) @ grasp)
    assert not preserves_upright(grasp, rpy_rotation(0, 0.9, 1.1))


def test_missing_held_object_geometry_fails_closed():
    with pytest.raises(ValueError):
        object_in_tool(np.empty((0, 3)), np.eye(4))


def test_support_collision_mesh_is_closed_and_inflated():
    hull = np.array([[-.015, -.015], [.015, -.015], [.015, .015], [-.015, .015]])
    vertices, triangles = support_prism(hull, .006, .036)
    np.testing.assert_allclose(vertices[:4, :2], hull*(17/15))
    from collections import Counter
    edges = Counter(tuple(sorted((t[i], t[(i+1)%3]))) for t in triangles for i in range(3))
    assert set(edges.values()) == {2}  # Every edge has exactly two incident faces.
    assert vertices[:, 2].min() == .006 and vertices[:, 2].max() == .036


def test_cad_tip_is_not_used_as_the_hardware_working_grasp_reference():
    import struct
    import xml.etree.ElementTree as ET
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[3]
    robot = ET.parse(root/'src/X5A/urdf/X5A_moveit.urdf').getroot()
    tool_x = float(robot.find("joint[@name='link6_tool0_joint']/origin").attrib['xyz'].split()[0])
    expected = []
    for link, joint in [('link7', 'joint7'), ('link8', 'joint8')]:
        offset = float(robot.find(f"joint[@name='{joint}']/origin").attrib['xyz'].split()[0])
        raw = (root/f'src/X5A/meshes/{link}.STL').read_bytes()
        count = struct.unpack('<I', raw[80:84])[0]
        points = np.frombuffer(raw[84:], dtype=np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=count)['v'].reshape(-1, 3)
        expected.append(float(points[:, 0].max())+offset-tool_x)
    config = yaml.safe_load((root/'src/x5a_remote_grasp/config/remote_grasp.yaml').read_text())['/**']['ros__parameters']['grasp']
    # CAD tip geometry and the experimentally working grasp reference are
    # distinct. Substituting 44.8mm caused the observed empty top-face grasp.
    assert expected[0] == pytest.approx(.0448, abs=1e-7)
    assert expected[1] == pytest.approx(.0448, abs=1e-7)
    assert config['tip_offset_m'] == .015
    assert config['transform_grasp_tool_translation'][0] == -config['tip_offset_m']
