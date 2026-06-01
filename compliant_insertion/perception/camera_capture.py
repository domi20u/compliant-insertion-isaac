"""Capture an RGB+depth frame from an Isaac Lab Camera sensor.

Thin adapter between Isaac Lab's Camera sensor data layout (torch tensors
on GPU, batched, with USD-convention quaternions) and what the perception
package wants (numpy arrays, a CameraIntrinsics struct, and a 4x4
world←camera transform).

Camera-convention note: ``back_project`` in perception.pose_estimation
uses the pinhole / ROS convention (+X right, +Y down, +Z forward into the
scene). We therefore read ``camera.data.quat_w_ros`` — Isaac Lab's
ROS-convention orientation accessor — and use it directly as the rotation
of ``T_world_cam``, with NO axis flips. This holds regardless of the
``convention=`` set on the CameraCfg (that only affects the camera's USD
pose, not which accessor we read). Do not reintroduce manual Y/Z flips:
an earlier version read ``quat_w_world`` and flipped, which produced a
bogus extrinsic (every table point projected behind the camera).
"""
from __future__ import annotations

import numpy as np
import torch

from compliant_insertion.perception import CameraIntrinsics


def capture_frame(
    camera, env_id: int = 0, *,
    K_override: np.ndarray | None = None,
    T_world_cam_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics, np.ndarray]:
    """Return (rgb, depth, intrinsics, T_world_cam).

    rgb is [H, W, 3] uint8, depth is [H, W] float32 in meters, intrinsics
    is a CameraIntrinsics struct, T_world_cam is a 4x4 numpy SE(3) matrix
    that maps camera-frame points to world-frame points.

    ``K_override`` / ``T_world_cam_override`` let the caller supply analytic
    intrinsics / extrinsics for a STATIC camera instead of reading
    ``camera.data``. This is required here because Isaac Lab leaves the
    workbench camera's pose buffers unpopulated at perception time (pos_w =
    origin, quat = NaN/zero), which corrupts the back-projection. The RGB and
    depth outputs ARE valid, so we still read those from the sensor.
    """
    rgb_t = camera.data.output["rgb"][env_id]
    depth_t = camera.data.output["distance_to_image_plane"][env_id]
    rgb = rgb_t.detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    
    print(f"[debug-perception] captured frame: rgb={rgb.shape} {rgb.dtype}, depth={depth_t.shape} {depth_t.dtype}, saving it...")
    from PIL import Image
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        # Camera sensors sometimes hand back float images in [0, 1].
        if arr.max() <= 1.0 + 1e-3:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:        # drop alpha
        arr = arr[..., :3]
    Image.fromarray(arr, mode="RGB").save("debug_rgb.png")
    print(f"[debug-perception] image saved to debug_rgb.png")

    depth = depth_t.detach().cpu().numpy().astype(np.float32)
    # Isaac Lab's distance_to_image_plane comes back as [H, W, 1]; drop the
    # trailing singleton so depth is the [H, W] this function documents and
    # that pose_estimation.back_project assumes (it indexes depth[vs, us] with
    # 1-D pixel arrays — a [H, W, 1] depth yields [N, 1] and breaks the
    # subsequent boolean masking).
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    if K_override is not None:
        K = np.asarray(K_override, dtype=np.float64)
    else:
        K = camera.data.intrinsic_matrices[env_id].detach().cpu().numpy()
    intrinsics = CameraIntrinsics.from_K(K, width=rgb.shape[1], height=rgb.shape[0])

    if T_world_cam_override is not None:
        # Static camera: caller supplied the analytic ROS-convention extrinsic.
        # See the docstring for why camera.data.pos_w / quat are unusable here.
        T_world_cam = np.asarray(T_world_cam_override, dtype=np.float64)
        return rgb, depth, intrinsics, T_world_cam

    # Fallback: read the camera pose from the sensor (ROS pinhole convention,
    # +X right / +Y down / +Z forward — no axis flips). Only valid once the
    # camera's pose buffers are populated.
    pos_w = camera.data.pos_w[env_id].detach().cpu().numpy()
    quat_w = camera.data.quat_w_ros[env_id].detach().cpu().numpy()
    R = _quat_to_mat(quat_w)

    T_world_cam = np.eye(4, dtype=np.float64)
    T_world_cam[:3, :3] = R
    T_world_cam[:3, 3] = pos_w
    return rgb, depth, intrinsics, T_world_cam


def _quat_to_mat(q):
    """Isaac Lab quaternion (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
