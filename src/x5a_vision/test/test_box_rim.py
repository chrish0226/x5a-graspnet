import numpy as np
from x5a_vision.box_rim import locate_rim, rim_candidates


def test_rim_center_rejects_side_depth_and_wrong_box_size():
    K = np.array([[1000., 0., 200.], [0., 1000., 200.], [0., 0., 1.]])
    depth = np.zeros((400, 400))
    depth[100:301, 50:351] = .44  # side/interior plane at the wrong height
    depth[100:104, 50:351] = .5
    depth[297:301, 50:351] = .5
    depth[100:301, 50:54] = .5
    depth[100:301, 347:351] = .5
    mask = depth > 0
    rim = locate_rim(depth, mask, K, np.eye(4), .5, .005, (.14, .09))
    assert rim is not None
    np.testing.assert_allclose(rim['center'], [0., 0., .5], atol=.001)
    np.testing.assert_allclose(rim['size'], [.10, .15], atol=.001)
    assert locate_rim(depth, mask, K, np.eye(4), .5, .005, (.20, .15)) is None
    assert locate_rim(depth, mask, K, np.eye(4), .6, .005, (.14, .09)) is None
    assert locate_rim(np.zeros_like(depth), mask, K, np.eye(4), .5, .005, (.14, .09)) is None


def rim_scene():
    K = np.array([[1000.,0.,200.],[0.,1000.,200.],[0.,0.,1.]])
    depth = np.full((400,400),.44)
    dark = np.zeros(depth.shape,bool)
    dark[100:105,50:351] = True
    dark[296:301,50:351] = True
    dark[100:301,50:55] = True
    dark[100:301,346:351] = True
    depth[dark] = .5
    return depth,dark,K


def test_box_rim_survives_bright_occupied_interior_and_small_depth_holes():
    depth,dark,K = rim_scene()
    # This contour has tiny actual dark area: old area/rectangularity filtering
    # rejected it even though the measured box rim provides its full geometry.
    depth[100:105,180:184] = 0
    rims = rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09))
    assert len(rims) == 1
    np.testing.assert_allclose(rims[0][0]['center'],[0.,0.,.5],atol=.001)


def test_missing_rim_side_cannot_supply_a_box_center():
    depth,dark,K = rim_scene()
    dark[:,346:] = False
    assert rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09)) == []


def test_black_shadow_at_table_height_is_not_a_box():
    depth,dark,K = rim_scene()
    depth[dark] = .44
    assert rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09)) == []


def test_relaxed_box_support_allows_one_missing_side_but_not_two():
    depth,dark,K = rim_scene()
    dark[:,346:] = False
    options=dict(min_supported_sides=3,size_tolerance=.025,edge_support_fraction=.2)
    rims=rim_candidates(depth,dark,K,np.eye(4),.5,.008,(.14,.09),**options)
    assert len(rims)==1
    np.testing.assert_allclose(rims[0][0]['center'],[0.,0.,.5],atol=.003)
    dark[296:,:]=False
    assert rim_candidates(depth,dark,K,np.eye(4),.5,.008,(.14,.09),**options)==[]


def test_adaptive_rim_recovers_raised_box_without_fixed_height_xy_projection():
    from x5a_vision.box_rim import adaptive_rim_candidates
    depth,dark,K = rim_scene()
    depth[dark] = .512
    assert rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09)) == []
    result = adaptive_rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09))
    assert len(result) == 1
    np.testing.assert_allclose(result[0][0]['center'], [0.,0.,.512], atol=.001)
    np.testing.assert_allclose(result[0][0]['size'], [.1024,.1536], atol=.001)


def test_adaptive_rim_measures_mild_tilt_and_rejects_wrong_geometry():
    from x5a_vision.box_rim import adaptive_rim_candidates
    depth,dark,K = rim_scene()
    ys,xs = np.indices(depth.shape)
    depth[dark] = (.5/(1-.1*(xs-200)/1000-.05*(ys-200)/1000))[dark]
    result = adaptive_rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09))
    assert len(result) == 1
    assert result[0][0]['tilt_deg'] > 5
    assert abs(result[0][0]['center'][2]-.5) < .002
    assert adaptive_rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.20,.15)) == []


def test_adaptive_rim_does_not_accept_two_edges_missing_depth_or_table_shadow():
    from x5a_vision.box_rim import adaptive_rim_candidates
    depth,dark,K = rim_scene()
    dark[:,346:] = False
    dark[296:,:] = False
    assert adaptive_rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09)) == []
    depth,dark,K = rim_scene()
    assert adaptive_rim_candidates(np.zeros_like(depth),dark,K,np.eye(4),.5,.005,(.14,.09)) == []
    assert adaptive_rim_candidates(np.full_like(depth,np.nan),dark,K,np.eye(4),.5,.005,(.14,.09)) == []
    depth[dark] = .44
    assert adaptive_rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09)) == []
    depth[dark] = .54
    assert adaptive_rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09)) == []


def test_adaptive_rim_survives_holes_and_inconsistent_side_depth():
    from x5a_vision.box_rim import adaptive_rim_candidates
    depth,dark,K = rim_scene()
    depth[100:105,170:185] = 0
    depth[160:168,50:55] = .46
    depth[dark & (depth == .5)] += .009
    result = adaptive_rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09))
    assert len(result) == 1
    np.testing.assert_allclose(result[0][0]['center'], [0.,0.,.509], atol=.002)


def test_gray_rim_is_detected_with_colored_blocks_inside():
    from x5a_vision.box_rim import adaptive_rim_candidates, rim_color_mask
    depth, rim, K = rim_scene()
    hsv = np.zeros((*depth.shape,3), np.uint8)
    hsv[:, :, 2] = 210  # Bright background/table/interior is not rim support.
    hsv[rim] = [0,35,145]  # Lit gray plastic rejected by old V<=85.
    hsv[130:190,100:160] = [0,230,220]  # red cube
    hsv[210:270,240:300] = [0,15,220]  # white cube
    dark = hsv[:,:,2] <= 85
    assert adaptive_rim_candidates(depth,dark,K,np.eye(4),.5,.005,(.14,.09)) == []
    support = rim_color_mask(hsv,dark)
    result = adaptive_rim_candidates(depth,support,K,np.eye(4),.5,.005,(.14,.09))
    assert len(result) == 1
    np.testing.assert_allclose(result[0][0]['center'],[0.,0.,.5],atol=.002)
    assert not support[150,120] and not support[240,270]


def test_gray_patch_cannot_replace_missing_depth_or_box_geometry():
    from x5a_vision.box_rim import adaptive_rim_candidates, rim_color_mask
    depth,rim,K = rim_scene()
    hsv=np.zeros((*depth.shape,3),np.uint8);hsv[:,:,2]=145
    support=rim_color_mask(hsv,np.zeros(depth.shape,bool))
    assert adaptive_rim_candidates(np.zeros_like(depth),support,K,np.eye(4),.5,.005,(.14,.09)) == []
    small=np.zeros(depth.shape);small[150:210,150:210]=.5
    assert adaptive_rim_candidates(small,support,K,np.eye(4),.5,.005,(.14,.09)) == []


def test_rim_search_window_requires_fresh_depth_and_recovers_with_occupied_interior():
    from x5a_vision.box_rim import adaptive_rim_candidates
    depth,mask,K=rim_scene()
    initial=adaptive_rim_candidates(depth,mask,K,np.eye(4),.5,.005,(.14,.09))
    assert len(initial)==1
    hint=initial[0][0]
    # Gray cube top close to rim height supplies more pixels than the thin rim.
    depth[140:260,100:300]=.491
    mask[140:260,100:300]=True
    found=adaptive_rim_candidates(depth,mask,K,np.eye(4),.5,.005,(.14,.09),hint=hint)
    assert len(found)==1
    np.testing.assert_allclose(found[0][0]['center'],[0.,0.,.5],atol=.003)
    assert adaptive_rim_candidates(np.zeros_like(depth),mask,K,np.eye(4),.5,.005,(.14,.09),hint=hint)==[]
    depth[mask]=.44
    assert adaptive_rim_candidates(depth,mask,K,np.eye(4),.5,.005,(.14,.09),hint=hint)==[]
