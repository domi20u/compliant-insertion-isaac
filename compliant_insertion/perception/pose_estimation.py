"""RGB-D back-projection and primitive fitting for 6-DoF object poses.

This stage consumes ``RawDetection`` (2D box + mask + class label) and
produces ``DetectedObject`` (6-DoF pose + measured geometric extents).
No learned model here — just classical geometry.

Why classical and not FoundationPose or MegaPose? Two reasons:

  1. Pegs and holes are perfectly described by two-parameter primitives
     (cylinder: axis + radius; circle: plane + radius). RANSAC on the
     masked point cloud gets ~mm-level accuracy in <50 ms per object,
     with no per-category training. FoundationPose is the right tool
     when the object has rich geometry (a bracket, a connector); for
     cylinders it is overkill and brings in a CAD-mesh requirement.

  2. The PRIMITIVE PARAMETERS are exactly what the cross-validation
     stage needs: "does the chosen peg's measured diameter fit the
     chosen hole's measured diameter?" A pose-only output would have
     forced us to keep the masks around and re-measure them, which is
     more pipeline state for no extra signal.

Coordinate convention
---------------------
Everything in this module is in CAMERA frame internally; the final
``pose_w`` field of DetectedObject is in WORLD frame, transformed by
the ``T_world_cam`` matrix at the very end. This avoids accumulating
floating-point error from many cam↔world round-trips during fitting.

The cylinder axis is encoded in the quaternion as: object-local +Z =
the cylinder's long axis (matching scene_cfg.PEG_LENGTH-aligned
geometry). For holes, object-local +Z = hole axis pointing OUT of the
socket (i.e. opposite the insertion direction).
"""
from __future__ import annotations

import numpy as np

from .detection import RawDetection
from .interfaces import CameraIntrinsics, DetectedObject


# ─── RGB-D back-projection ───────────────────────────────────────────────────
def back_project(depth: np.ndarray, mask: np.ndarray,
                 K: CameraIntrinsics,
                 depth_max: float = 2.0) -> np.ndarray:
    """Return [M, 3] camera-frame points for the pixels under ``mask``.

    Pixels with zero or beyond-range depth are dropped. The convention
    matches Isaac Lab's distance_to_image_plane sensor: depth is the
    Z-component (NOT the slant distance), so the back-projection is
    the standard pinhole:

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth[v, u]
    """
    if depth.ndim == 3 and depth.shape[-1] == 1:    # tolerate [H, W, 1]
        depth = depth[..., 0]
    H, W = depth.shape[:2]
    vs, us = np.where(mask)
    if vs.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth[vs, us].astype(np.float32)
    valid = (z > 1e-3) & (z < depth_max)
    vs, us, z = vs[valid], us[valid], z[valid]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    x = (us.astype(np.float32) - K.cx) * z / K.fx
    y = (vs.astype(np.float32) - K.cy) * z / K.fy
    return np.stack([x, y, z], axis=-1)


# ─── Cylinder fitting (for pegs) ─────────────────────────────────────────────
def fit_cylinder_ransac(
    points: np.ndarray,
    n_iters: int = 200,
    inlier_thresh: float = 0.002,
    expected_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float] | None:
    """RANSAC fit of a cylinder to a 3D point cloud.

    For tabletop pegs we have a strong prior: the cylinder axis is
    very nearly world-+Z (modulo small tilts). If ``expected_axis`` is
    given, we skip the costly random-pair axis estimation and just
    fit the radius and centerline position to that axis — five orders
    of magnitude faster and more robust against noisy masks.

    Returns ``(center, axis, radius, height, inlier_ratio)`` in
    CAMERA frame, or None if the fit is hopeless.

      - ``center`` is the geometric midpoint of the visible cylinder
        portion along the axis. NOT necessarily the object's centroid
        — for a partially occluded peg this is the midpoint of what we
        SAW. The runner uses (center + 0.5*height*axis) for the top
        endpoint, which is also where the gripper grasps.
      - ``axis`` is a unit vector along the cylinder long axis.
      - ``radius`` and ``height`` are measured (NOT nominal).
    """
    if points.shape[0] < 30:
        return None

    rng = np.random.default_rng(0)

    if expected_axis is not None:
        # Prior path: axis is given, only fit radius + axial extent.
        axis = expected_axis / (np.linalg.norm(expected_axis) + 1e-9)
        return _fit_cylinder_given_axis(points, axis, inlier_thresh)

    best = None
    n = points.shape[0]
    for _ in range(n_iters):
        i, j = rng.choice(n, size=2, replace=False)
        axis = points[j] - points[i]
        norm = np.linalg.norm(axis)
        if norm < 1e-3:
            continue
        axis = axis / norm
        fit = _fit_cylinder_given_axis(points, axis, inlier_thresh)
        if fit is None:
            continue
        _, _, _, _, inlier_ratio = fit
        if best is None or inlier_ratio > best[-1]:
            best = fit

    return best


def _fit_cylinder_given_axis(points, axis, inlier_thresh):
    """Inner loop: given an axis direction, fit the cylinder it best supports."""
    # Project points to the plane perpendicular to ``axis``. The cylinder
    # cross-section is a circle in that plane.
    centroid = points.mean(axis=0)
    rel = points - centroid
    axial = rel @ axis                                 # [N]
    perp = rel - np.outer(axial, axis)                 # [N, 3] in-plane
    # 2D least-squares circle fit in the plane (Kasa method).
    # Build a 2D basis on the plane.
    e1 = perp[0] / (np.linalg.norm(perp[0]) + 1e-9) if len(perp) else np.array([1, 0, 0])
    if abs(e1 @ axis) > 0.99:                          # degenerate
        e1 = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = e1 - (e1 @ axis) * axis
        e1 = e1 / (np.linalg.norm(e1) + 1e-9)
    e2 = np.cross(axis, e1)

    u = perp @ e1
    v = perp @ e2
    A = np.stack([2 * u, 2 * v, np.ones_like(u)], axis=-1)
    b = u ** 2 + v ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cu, cv, c3 = sol
    radius = float(np.sqrt(max(cu ** 2 + cv ** 2 + c3, 1e-9)))

    # Residual distance to cylinder surface (in-plane radial - radius).
    radial = np.sqrt((u - cu) ** 2 + (v - cv) ** 2)
    residual = np.abs(radial - radius)
    inliers = residual < inlier_thresh
    inlier_ratio = float(inliers.mean())
    if inlier_ratio < 0.2:
        return None

    # Center in 3D: shift centroid by (cu, cv) along (e1, e2).
    center = centroid + cu * e1 + cv * e2
    # Height = axial extent of inliers.
    if inliers.any():
        ax_in = axial[inliers]
        height = float(ax_in.max() - ax_in.min())
    else:
        height = float(axial.max() - axial.min())

    return center.astype(np.float32), axis.astype(np.float32), radius, height, inlier_ratio


# ─── Circle-on-plane fitting (for holes) ─────────────────────────────────────
def fit_hole_on_plane(
    points: np.ndarray,
    expected_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Fit a circle on a plane to a hole mask's point cloud.

    Two-step procedure:

      1. Fit a plane to the rim points (the mask covers the rim ring +
         possibly some of the surrounding fixture). With a world-+Z
         prior the plane normal is essentially given.
      2. Project to the plane and fit a circle (Kasa) to recover the
         hole center and radius.

    Returns ``(center_cam, normal_cam, radius, inlier_ratio)``. The
    normal points "out of the socket" toward the camera by convention.
    """
    if points.shape[0] < 20:
        return None

    if expected_normal is not None:
        normal = expected_normal / (np.linalg.norm(expected_normal) + 1e-9)
    else:
        # PCA: smallest principal component ≈ plane normal.
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        # Ensure it points toward the camera (origin is at the camera).
        centroid = points.mean(axis=0)
        if normal @ (-centroid) < 0:
            normal = -normal

    # Project to plane.
    centroid = points.mean(axis=0)
    rel = points - centroid
    proj = rel - np.outer(rel @ normal, normal)

    # Pick a 2D basis on the plane.
    e1 = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ normal) * normal
    e1 = e1 / (np.linalg.norm(e1) + 1e-9)
    e2 = np.cross(normal, e1)
    u = proj @ e1
    v = proj @ e2

    A = np.stack([2 * u, 2 * v, np.ones_like(u)], axis=-1)
    b = u ** 2 + v ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cu, cv, c3 = sol
    radius = float(np.sqrt(max(cu ** 2 + cv ** 2 + c3, 1e-9)))

    radial = np.sqrt((u - cu) ** 2 + (v - cv) ** 2)
    # Inliers = points close to the rim circle.
    residual = np.abs(radial - radius)
    inlier_ratio = float((residual < 0.003).mean())

    center = centroid + cu * e1 + cv * e2
    return center.astype(np.float32), normal.astype(np.float32), radius, inlier_ratio


# ─── Pose composition helpers ────────────────────────────────────────────────
def axis_to_quat_wxyz(axis: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) such that the object-local +Z aligns with ``axis``.

    The remaining roll is unconstrained for axially symmetric objects;
    we pick the one with minimal rotation from identity (shortest arc
    between world +Z and ``axis``).
    """
    z = np.array([0.0, 0.0, 1.0])
    a = axis / (np.linalg.norm(axis) + 1e-9)
    dot = float(np.clip(z @ a, -1.0, 1.0))
    if dot > 0.99999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if dot < -0.99999:
        # 180° about world X.
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    rot_axis = np.cross(z, a)
    rot_axis /= np.linalg.norm(rot_axis) + 1e-9
    angle = np.arccos(dot)
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), rot_axis[0] * s, rot_axis[1] * s, rot_axis[2] * s],
                    dtype=np.float32)


def transform_pose_to_world(center_cam: np.ndarray, axis_cam: np.ndarray,
                            T_world_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lift a (center, axis) pair from camera frame to world frame.

    Position is a homogeneous transform; axis is a pure rotation.
    """
    center_h = np.append(center_cam, 1.0)
    center_w = (T_world_cam @ center_h)[:3]
    axis_w = T_world_cam[:3, :3] @ axis_cam
    return center_w.astype(np.float32), axis_w.astype(np.float32)


# ─── Top-level pose estimation ───────────────────────────────────────────────
def estimate_poses(
    detections: list[RawDetection],
    rgb: np.ndarray,
    depth: np.ndarray,
    K: CameraIntrinsics,
    T_world_cam: np.ndarray,
    z_axis_prior: bool = True,
) -> list[DetectedObject]:
    """Run the full pose-estimation stage on a list of detections.

    ``z_axis_prior=True`` uses the very strong tabletop prior that
    pegs are vertical and hole axes are vertical. For an arm-mounted
    or oblique camera you should set this to False to let the RANSAC
    discover the axis from the points.
    """
    # World +Z, expressed in camera frame.
    world_z_cam = (np.linalg.inv(T_world_cam)[:3, :3] @ np.array([0.0, 0.0, 1.0])).astype(np.float32)
    objects: list[DetectedObject] = []

    peg_idx = 0
    hole_idx = 0
    for det in detections:
        points_cam = back_project(depth, det.mask, K)
        if points_cam.shape[0] < 30:
            continue

        if det.canonical_label == "peg":
            fit = fit_cylinder_ransac(
                points_cam,
                expected_axis=world_z_cam if z_axis_prior else None,
            )
            if fit is None:
                continue
            center_cam, axis_cam, radius, height, inlier_ratio = fit
            center_w, axis_w = transform_pose_to_world(center_cam, axis_cam, T_world_cam)
            quat_wxyz = axis_to_quat_wxyz(axis_w)
            pose_w = np.concatenate([center_w, quat_wxyz]).astype(np.float32)
            objects.append(DetectedObject(
                object_id=f"peg_{peg_idx}",
                label="peg",
                pose_w=pose_w,
                diameter_m=2.0 * radius,
                height_m=height,
                confidence=float(det.score) * inlier_ratio,
                bbox_xyxy=det.bbox_xyxy,
                mask=det.mask,
                primitive_inlier_ratio=inlier_ratio,
                color_rgb=_mean_color(rgb, det.mask),
            ))
            peg_idx += 1

        elif det.canonical_label == "hole":
            fit = fit_hole_on_plane(
                points_cam,
                expected_normal=world_z_cam if z_axis_prior else None,
            )
            if fit is None:
                continue
            center_cam, normal_cam, radius, inlier_ratio = fit
            center_w, normal_w = transform_pose_to_world(center_cam, normal_cam, T_world_cam)
            quat_wxyz = axis_to_quat_wxyz(normal_w)
            pose_w = np.concatenate([center_w, quat_wxyz]).astype(np.float32)
            objects.append(DetectedObject(
                object_id=f"hole_{hole_idx}",
                label="hole",
                pose_w=pose_w,
                diameter_m=2.0 * radius,
                height_m=None,
                confidence=float(det.score) * inlier_ratio,
                bbox_xyxy=det.bbox_xyxy,
                mask=det.mask,
                primitive_inlier_ratio=inlier_ratio,
                # A hole's detection mask covers the dark VOID, whose color is
                # shadow — not the colored socket block. Sample a ring just
                # outside the void (on the block top) so the measured color is
                # the block's identity color, which is what the peg matches to.
                color_rgb=_ring_color(rgb, det.mask),
            ))
            hole_idx += 1

    return objects


def _mean_color(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Mean RGB color inside the mask, as a hint for the VLM."""
    if mask.sum() == 0:
        return (0.5, 0.5, 0.5)
    pixels = rgb[mask] / 255.0
    m = pixels.mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Binary dilation by 4-connectivity, ``iterations`` times (no SciPy dep)."""
    out = mask.copy()
    for _ in range(iterations):
        nxt = out.copy()
        nxt[1:, :] |= out[:-1, :]
        nxt[:-1, :] |= out[1:, :]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:, :-1] |= out[:, 1:]
        out = nxt
    return out


def _ring_color(rgb: np.ndarray, mask: np.ndarray,
                grow: int = 6) -> tuple[float, float, float]:
    """Mean RGB of a ring just OUTSIDE ``mask`` (the block top around a hole).

    Dilates the void mask and samples the annulus ``dilated & ~mask`` so the
    color reflects the colored socket block rather than the dark hole
    interior. Falls back to the in-mask mean if the ring comes out empty.
    """
    if mask is None or mask.sum() == 0:
        return (0.5, 0.5, 0.5)
    ring = _dilate(mask, grow) & ~mask
    if not ring.any():
        return _mean_color(rgb, mask)
    m = (rgb[ring] / 255.0).mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))
