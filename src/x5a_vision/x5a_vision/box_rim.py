"""Locate a measured horizontal box rim; reject missing or inconsistent depth."""
import math
import cv2
import numpy as np
from x5a_vision.top_plane import deproject


def rim_color_mask(hsv, dark_mask, gray_max_s=100, gray_max_v=165):
    """Include lit gray plastic rims without requiring a dark empty interior.

    This is only color support: measured rim height, plane fit, physical size,
    three supported sides and an opposite pair must still pass afterward.
    """
    if not 0 <= gray_max_s <= 150 or not 85 <= gray_max_v <= 200:
        raise ValueError('invalid gray rim color bounds')
    gray = (hsv[:, :, 1] <= gray_max_s) & (hsv[:, :, 2] <= gray_max_v)
    return np.asarray(dark_mask, bool) | gray


def locate_rim(depth, mask, K, T, z, tolerance, inner_size, wall=0.005,
               min_supported_sides=4, size_tolerance=.018, edge_support_fraction=.35):
    ys, xs = np.nonzero(mask & np.isfinite(depth) & (depth > .15) & (depth < 1.))
    if len(xs) < 100:
        return None
    rays = deproject(xs, ys, np.ones(len(xs)), K) @ T[:3, :3].T
    points = T[:3, 3] + rays * depth[ys, xs, None]
    keep = (np.abs(points[:, 2]-z) <= tolerance) & (np.abs(rays[:, 2]) > 1e-6)
    if np.count_nonzero(keep) < 100:
        return None
    # Depth selects the actual rim. Intersect its pixels with the measured
    # height to avoid shifting the XY rectangle with depth noise.
    rays = rays[keep]
    scales = (z-T[2, 3])/rays[:, 2]
    if np.any(scales <= 0):
        return None
    plane = T[:3, 3] + rays*scales[:, None]
    return _rim_rectangle(plane, inner_size, wall, min_supported_sides,
                          size_tolerance, edge_support_fraction,
                          normal=np.array([0., 0., 1.]), offset=z)


def _rim_rectangle(plane, inner_size, wall, min_supported_sides,
                   size_tolerance, edge_support_fraction, normal, offset):
    """Validate physical size and edge support on an already measured plane."""
    if len(plane) < 100:
        return None
    depth_points = len(plane)
    # A few flying depth pixels must not expand the fitted physical rectangle.
    low, high = np.percentile(plane[:, :2], [.5, 99.5], axis=0)
    plane = plane[np.all((plane[:, :2] >= low) & (plane[:, :2] <= high), axis=1)]
    rect = cv2.minAreaRect(plane[:, :2].astype(np.float32))
    (x,y), (sx,sy), angle = rect
    measured = sorted((sx,sy))
    expected = sorted(v+2*wall for v in inner_size)
    if any(abs(a-b) > size_tolerance for a,b in zip(measured,expected)):
        return None
    # A partial rim or an interior patch cannot reliably locate all four edges.
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    local = (plane[:, :2]-[x,y]) @ np.array([[c,-s],[s,c]])
    supported = []
    for axis, extent in enumerate((sx,sy)):
        for sign in (-1,1):
            edge = local[sign*local[:,axis] >= extent*.5-.008]
            other_extent = sy if axis == 0 else sx
            edge = edge[np.abs(edge[:,1-axis]) <= other_extent*.35]
            supported.append(len(edge) >= 10 and np.ptp(edge[:,1-axis]) >= other_extent*edge_support_fraction)
    if sum(supported) < min_supported_sides or not ((supported[0] and supported[1]) or (supported[2] and supported[3])):
        return None
    yaw = math.radians(angle + (90. if sy > sx else 0.))
    corners_xy = cv2.boxPoints(rect)
    corners_z = (offset-corners_xy @ normal[:2])/normal[2]
    center_z = (offset-np.dot([x,y], normal[:2]))/normal[2]
    corners = np.c_[corners_xy, corners_z]
    return dict(center=np.array([x,y,center_z]), yaw=yaw, corners=corners,
                size=measured, depth_points=depth_points,
                tilt_deg=math.degrees(math.acos(float(np.clip(normal[2], -1., 1.)))))


def rim_candidates(depth, dark_mask, K, T, z, tolerance, inner_size, **rim_options):
    """Find rims using measured height; interiors may be bright or occupied."""
    from x5a_vision.top_plane import height_mask
    near_rim = height_mask(depth, K, T, z-tolerance, z+tolerance, .15, 1.)
    support = (near_rim & dark_mask).astype(np.uint8)
    # Close small holes for component association, fit only original valid pixels.
    grouped = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        component = np.zeros(depth.shape, np.uint8)
        cv2.drawContours(component, [contour], -1, 1, -1)
        rim = locate_rim(depth, support.astype(bool) & component.astype(bool), K, T,
                         z, tolerance, inner_size, **rim_options)
        if rim is not None:
            results.append((rim, contour))
    return results


def adaptive_rim_candidates(depth, dark_mask, K, T, z, tolerance, inner_size,
                            height_range=.015, plane_tolerance=.003, max_tilt_deg=8.,
                            wall=.005, min_supported_sides=3, size_tolerance=.025,
                            edge_support_fraction=.2, hint=None):
    """Fit the actual near-horizontal rim instead of intersecting a fixed Z.

    The nominal height bounds the search; only current, valid depth is used.
    Sides, holes and flying pixels are excluded by a robust plane fit, followed
    by the same size and three/four-side validation as the fixed-height path.
    """
    from x5a_vision.top_plane import height_mask
    if (not 0 < height_range <= .02 or not .001 <= plane_tolerance <= .005
            or not 0 < max_tilt_deg <= 10):
        raise ValueError('invalid adaptive rim plane parameters')
    near = height_mask(depth, K, T, z-height_range-tolerance,
                       z+height_range+tolerance, .15, 1.)
    support = near & dark_mask
    grouped = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((9,9),np.uint8))
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    minimum_normal_z = math.cos(math.radians(max_tilt_deg))
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        # Crop each component to avoid allocating HD masks for small fragments.
        bx, by, bw, bh = cv2.boundingRect(contour)
        component = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(component, [contour-[bx, by]], -1, 1, -1)
        ys, xs = np.nonzero(component.astype(bool) & support[by:by+bh, bx:bx+bw])
        ys, xs = ys+by, xs+bx
        if len(xs) < 100:
            continue
        points = deproject(xs, ys, depth[ys,xs], K) @ T[:3,:3].T + T[:3,3]
        if hint is not None:
            # A current depth-supported contour provides a search window, not
            # a replacement pose. Keep only points near its four edges so an
            # interior cube face cannot win the plane fit by pixel count.
            c,s = math.cos(hint['yaw']),math.sin(hint['yaw'])
            local=(points[:,:2]-np.asarray(hint['center'])[:2]) @ np.array([[c,-s],[s,c]])
            half=np.asarray(sorted(hint['size'],reverse=True))*.5
            band=.012
            keep=(np.all(np.abs(local)<=half+band,axis=1)
                  & np.any(np.abs(np.abs(local)-half)<=band,axis=1))
            points=points[keep]
            if len(points)<100:
                continue
        # An upper box rim must span both dimensions; narrow strips cannot
        # establish its center even if a plane fits their depth perfectly.
        spans = np.ptp(points[:,:2], axis=0)
        if min(spans) < .025 or max(spans) > .35:
            continue
        sample = points[np.linspace(0, len(points)-1, min(2500, len(points)), dtype=int)]
        rng = np.random.default_rng(1)
        trials = sample[np.array([rng.choice(len(sample), 3, replace=False) for _ in range(160)])]
        anchors = trials[:, 0]
        normals = np.cross(trials[:, 1]-anchors, trials[:, 2]-anchors)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths >= 1e-8
        normals[valid] /= lengths[valid, None]
        normals[normals[:, 2] < 0] *= -1
        valid &= normals[:, 2] >= minimum_normal_z
        if not np.any(valid):
            continue
        normals, anchors = normals[valid], anchors[valid]
        offsets = np.sum(normals*anchors, axis=1)
        counts = np.count_nonzero(np.abs(sample @ normals.T-offsets) <= plane_tolerance, axis=0)
        winner = int(np.argmax(counts))
        if counts[winner] < 100:
            continue
        normal, offset = normals[winner], offsets[winner]
        # Refine only inliers. Recheck tilt after SVD: refinement must not turn
        # a horizontal hypothesis into a side-plane or a depth discontinuity.
        for _ in range(2):
            selected = np.abs(points @ normal-offset) <= plane_tolerance
            inliers = points[selected]
            if len(inliers) < 100:
                break
            center = inliers.mean(axis=0)
            _, _, vt = np.linalg.svd(inliers-center, full_matrices=False)
            normal = vt[-1]
            if normal[2] < 0:
                normal = -normal
            offset = float(np.dot(normal, center))
        if normal[2] < minimum_normal_z:
            continue
        selected = np.abs(points @ normal-offset) <= plane_tolerance
        if np.count_nonzero(selected) < 100:
            continue
        # Keep measured XY. Forcing oblique rays onto even a slightly biased
        # plane magnifies Z noise into several millimetres of XY displacement.
        plane = points[selected]
        rim = _rim_rectangle(plane, inner_size, wall, min_supported_sides,
                             size_tolerance, edge_support_fraction, normal, offset)
        if rim is None or abs(rim['center'][2]-z) > height_range:
            continue
        if hint is not None and np.max(np.abs(np.asarray(rim['size'])-np.asarray(sorted(hint['size'])))) > .008:
            # Reject clipping by an old search window after a box moves. The
            # caller retries full-frame detection using current depth.
            continue
        rim['plane_residual_m'] = float(np.sqrt(np.mean((points[selected] @ normal-offset)**2)))
        results.append((rim, contour))
    return results
