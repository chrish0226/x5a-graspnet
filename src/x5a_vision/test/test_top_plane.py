import numpy as np

from x5a_vision.top_plane import fit_horizontal_top, height_mask, plane_rectangle_center, rgbd_dimensions_match


def test_transient_native_depth_cannot_be_indexed_as_aligned_hd_depth():
    assert rgbd_dimensions_match((1080,1920,3), (1080,1920), 1920,1080)
    assert not rgbd_dimensions_match((1080,1920,3), (480,640), 1920,1080)
    assert not rgbd_dimensions_match((1080,1920,3), (1080,1920), 640,480)


def test_top_plane_survives_more_side_pixels_than_top_pixels():
    rng = np.random.default_rng(12)
    center = np.array([.4, .2, .033])
    top = center+rng.uniform([-.015, -.015, -.0003], [.015, .015, .0003], (600, 3))
    side = center+rng.uniform([.015, -.015, -.03], [.015, .015, 0.], (1400, 3))
    outliers = rng.uniform([.38, .18, -.1], [.42, .22, .1], (80, 3))
    selected = fit_horizontal_top(np.r_[top, side, outliers])
    assert selected is not None
    actual_center, _ = plane_rectangle_center(selected)
    np.testing.assert_allclose(actual_center, center, atol=.002)
    assert selected[:, 2].mean() > .031


def test_side_without_top_is_rejected_and_large_paper_plane_is_rejected():
    rng = np.random.default_rng(17)
    side = rng.uniform([.4, .19, 0.], [.4, .22, .03], (600, 3))
    assert fit_horizontal_top(side) is None
    paper = rng.uniform([.1, .1, .003], [.4, .4, .003], (600, 3))
    assert fit_horizontal_top(paper) is None
    assert fit_horizontal_top(np.full((100, 3), np.nan)) is None


def test_height_gate_uses_base_height_instead_of_camera_depth():
    # Optical Z points along base X; base height comes from the image row.
    T = np.eye(4)
    T[:3, :3] = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
    K = np.array([[100., 0., 0.], [0., 100., 0.], [0., 0., 1.]])
    depth = np.full((3, 3), .5)
    gate = height_mask(depth, K, T, .004, .011, .2, .8)
    np.testing.assert_array_equal(gate, [[False]*3, [True]*3, [True]*3])
    depth[1, 1] = np.nan
    assert not height_mask(depth, K, T, .004, .011, .2, .8)[1, 1]


def test_two_layer_cube_top_height_gate():
    from types import SimpleNamespace
    from x5a_vision.cube_detector import CubeDetector
    detector = SimpleNamespace(table_z=-.006, object_height=.03, table_tol=.015, max_cube_layers=2)
    assert CubeDetector.valid_cube_top(detector, .024)
    assert CubeDetector.valid_cube_top(detector, .054)
    assert not CubeDetector.valid_cube_top(detector, .085)
