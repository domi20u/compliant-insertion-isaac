"""Cluttered-scene diffusion-policy data collection: insertion-only DMP rollouts.

Counterpart to ``collect_demos.py`` (single-peg) and ``cluttered_run_e2e.py``
(cluttered runner). Generates an imitation dataset for a **goal-conditioned
insertion skill** in the cluttered tabletop scene, then exports it to a
RoboMimic-ish HDF5 that ``hdf5_to_lerobot.py`` turns into a LeRobotDataset for
Diffusion Policy / SmolVLA / π0.

Scope of v1 (deliberately narrow — see the conversation that motivated it):

  - **Insertion only.** Each episode runs the full HOME → DMP-to-grasp → GRASP
    → DMP-to-insert procedure, but ONLY the grasp→insert (peg-in-hand) segment
    is recorded. At deployment the policy's initial state is "post-grasp, peg in
    hand", so the approach/grasp transients are out-of-distribution and would
    teach states the policy never sees. The pick is the means, not the subject.

  - **10 mm-clearance socket only.** The benchmark's accuracy-vs-tolerance sweep
    {10, 5, 1} mm is collected later, once a residual policy can seat the tight
    ones. For now the target is always the red peg (peg_0) → red socket (hole_0,
    10 mm). The other pegs/sockets remain as visual distractors. The success
    filter (drops non-seated episodes) is the guarantee that what lands in the
    dataset is clean expert data.

  - **TCP-space DMP**, matching ``cluttered_run_e2e.py``: the DMP integrates a
    HAND-frame trajectory whose per-step output is fed straight to the OSC. The
    recorded action is therefore the OSC TCP pose target — the same action space
    a Diffusion Policy / VLA will command, keeping the comparison apples-to-apples.

  - **Ground-truth poses.** No perception in the loop — grasp/insert goals come
    from the (randomized) spec geometry. This is a clean expert; perception
    noise belongs in the deployment-time evaluation, not the demos.

Per-episode variation
---------------------
  - **Target peg XY** (peg_0): uniform over a reachable peg-side box.
  - **Target socket XY** (hole_0): uniform over a reachable socket-side box —
    the 5 kinematic prims are teleported together via
    ``socket_component_offsets`` (this is the axis that was previously fixed and
    is the main reason the policy can generalize the relative pick→place vector).
  - A minimum peg↔socket separation keeps the transport meaningful and bounds
    the geometry-derived lift below.
  - **Distractor pegs/sockets** stay at their spec positions (consistent
    clutter). Randomizing them too is a later, color-matching concern.
  - **DMP lift task-param** is DERIVED per episode from the transport distance so
    the transit apex clears the socket rim — fixing the collision that motivated
    this whole exercise (param1=0.05 was too low). See ``safe_task_params``.

Output: one HDF5 (same RoboMimic-ish layout as ``collect_demos.py``) plus extra
per-episode goal attrs for goal conditioning::

    data/demo_N/
      obs/{wrist_image, external_image, ee_pose_b, ee_vel_b, joint_pos, gripper_pos}
      actions                      [T, 7]   OSC TCP pose target (pos + quat), base frame
      dones, rewards
      @attrs: success, task_params, target_peg_id, target_hole_id,
              peg_init_pos_w, hole_xy, hole_top_w, goal_tcp_w,
              goal_pose_b (TCP insertion target, base frame, for conditioning),
              task (language string for VLAs), lateral_err_final, depth_frac_final, length

Usage::

    /path/to/IsaacLab/isaaclab.sh -p scripts/data/collect_cluttered_demos.py \\
        --config configs/insertion_traj.yaml \\
        --out data/cluttered_insertion_demos.hdf5 \\
        --num-episodes 200

Then convert to LeRobot (in an env with ``lerobot`` installed, plain python)::

    python scripts/data/hdf5_to_lerobot.py \\
        --hdf5 data/cluttered_insertion_demos.hdf5 \\
        --repo-id local/cluttered_insertion --root data/lerobot/cluttered_insertion
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

# ─── Step 1: launch Isaac Sim BEFORE any isaaclab.* imports. ─────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect cluttered insertion-only DMP demos for DP/VLA training.")
parser.add_argument("--config", type=Path, required=True,
                    help="YAML config (same schema as cluttered_run_e2e.py).")
parser.add_argument("--out", type=Path, required=True,
                    help="HDF5 output path. Parent dir is created if absent.")
parser.add_argument("--num-episodes", type=int, default=100)
parser.add_argument("--keep-failures", action="store_true",
                    help="Save unsuccessful episodes too (success attr=False).")
parser.add_argument("--playback-speed", type=float, default=10.0,
                    help="DMP integration speedup. Higher = faster robot motion "
                         "(and shorter, coarser recordings). Drop to ~5 if the "
                         "OSC tracking lag hurts insertion alignment.")
parser.add_argument("--home-height", type=float, default=0.30,
                    help="World Z (m) of the TCP 'home' the robot servos to "
                         "after reset, above the workspace center. Lower = "
                         "start closer to the table (shorter first transit).")
parser.add_argument("--cam-height", type=int, default=128)
parser.add_argument("--cam-width", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)

# --- Workspace randomization (one shared, reachable box for ALL objects). ---
# A single box (not separate peg/socket bands) so the relative peg→socket
# direction varies over the full 360°, not just one axis. All 3 pegs and all 3
# sockets are scattered here with pairwise min-separations; the target pair is
# peg_0 / hole_0.
parser.add_argument("--workspace", type=float, nargs=4,
                    default=(0.35, 0.55, -0.28, 0.28),
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="Shared XY box for all pegs + sockets.")
parser.add_argument("--min-transport", type=float, default=0.15,
                    help="Min horizontal peg_0↔hole_0 distance (m). Keeps the "
                         "transport meaningful and bounds the derived lift.")
parser.add_argument("--corridor-clear", type=float, default=0.05,
                    help="Min XY clearance a distractor must keep from the "
                         "peg_0→hole_0 transit line, so the carried peg's arc "
                         "isn't obstructed near its low endpoints. 0 disables.")
parser.add_argument("--max-place-tries", type=int, default=400)

# --- DMP task-param derivation (lift that clears the socket rim). ---
# param1 ≈ lift_height / transport_distance, param2 ≈ fraction of the path at
# which the apex is reached. The mapping from param1 to the realized apex
# height is approximate (it depends on the demo shape + rotodilatation), so the
# SUCCESS FILTER is the real guarantee. Run a small batch first and check the
# printed success rate; nudge --clear-margin / --p1-* if it's low.
parser.add_argument("--clear-margin", type=float, default=0.05,
                    help="Desired transit clearance (m) of the peg tip above "
                         "the socket rim top. Larger = safer, slower arc.")
parser.add_argument("--p1-min", type=float, default=0.12)
parser.add_argument("--p1-max", type=float, default=0.45)
parser.add_argument("--p1-jitter", type=float, default=0.03,
                    help="± uniform jitter on the derived lift fraction, for "
                         "harmless trajectory-shape multimodality.")
parser.add_argument("--p2-center", type=float, default=0.20)
parser.add_argument("--p2-jitter", type=float, default=0.04)
parser.add_argument("--task-params", type=float, nargs=2, default=None,
                    metavar=("P1", "P2"),
                    help="Override: force a FIXED task-param pair every episode "
                         "(bypasses the geometry-derived lift). Use for "
                         "debugging a known-good pair.")

# --- Insertion stop position. ---
# Where the DMP's (trained) steep descent stops, as a fraction of HOLE_DEPTH
# below the rim. The DMP does the descent itself — we only set the goal. Keep it
# SHALLOW: the residual policy seats the rest later, and a shallow stop keeps the
# Panda fingertips (≈tip_z + 0.039 m) well above the rim top, so the TCP never
# collides with the socket. Full seating (≈1.0) drives the fingers into the rim.
parser.add_argument("--insert-depth-frac", type=float, default=0.3,
                    help="Peg-tip stop depth as a fraction of HOLE_DEPTH below "
                         "the rim (0.0 = rim, 1.0 = bottom). Default 0.3: a "
                         "clean partial insertion, fingers well clear of the rim.")
parser.add_argument("--success-depth-frac", type=float, default=0.1,
                    help="Min depth_frac (tip below rim / rim height) for a demo "
                         "to count as a successful insertion. Low by design — "
                         "the DMP only needs to get the peg ALIGNED and STARTED "
                         "into the hole; the residual policy does full seating.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras must be enabled at AppLauncher time for the Camera sensors to produce
# rgb tensors (same reason as cluttered_run_e2e.py).
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Step 2: everything else can be imported now. ───────────────────────────
import h5py
import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import (
    OperationalSpaceController, OperationalSpaceControllerCfg,
)
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv,
    subtract_frame_transforms,
)

from compliant_insertion.env.cluttered_scene_cfg import (
    ClutteredInsertionSceneCfg,
    PEG_SPECS, SOCKET_SPECS,
    HOLE_DEPTH_M, SOCKET_BLOCK_SIZE_XY,
    TABLE_TOP_Z, PEG_LENGTH,
    GRIPPER_DOWN_QUAT,
    attach_pegs_and_sockets,
    hole_center_w, socket_top_z,
    socket_component_offsets, peg_spec, socket_spec,
    finger_grip_target,
    hand_pos_for_grasp, hand_pos_for_peg_tip,
    insertion_success,
    _camera_lookat_quat,
)

# ─── Constants (mirror cluttered_run_e2e.py). ────────────────────────────────
DMP_TIME_EXTENSION_FACTOR = 1.05
GRIP_HOLD_TICKS = 60
HOME_MOVE_TICKS = 120     # servo from default joints to the (lower) home TCP
RELEASE_OPEN_TICKS = 40   # open the fingers, let the peg drop into the hole
RETRACT_TICKS = 60        # lift the empty gripper clear, let the peg settle
SCENE_HZ = 100.0   # 1 / sim_dt (sim_dt = 0.01) — control + record rate.

TARGET_PEG_ID = "peg_0"
TARGET_HOLE_ID = "hole_0"      # 10 mm clearance (red)
TASK_STRING = "insert the red peg into the red socket"


# ─── Scene: cluttered scene + wrist cam + external cam. ──────────────────────
@configclass
class ClutteredDataGenSceneCfg(ClutteredInsertionSceneCfg):
    """Cluttered scene + a wrist cam (on panda_hand) + a fixed external cam.

    The inherited 640×480 ``workbench_camera`` (for perception) is throttled in
    ``main`` so it doesn't re-render every tick during collection.
    """

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=args_cli.cam_height,
        width=args_cli.cam_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, horizontal_aperture=20.955,
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Behind the hand origin along the gripper -Z, looking out the
            # fingers toward the peg tip / socket. Identity rot already looks
            # along panda_hand +Z (ROS cam +Z = forward).
            pos=(0.0, 0.0, -0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="ros",
        ),
    )

    external_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        update_period=0.0,
        height=args_cli.cam_height,
        width=args_cli.cam_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 6.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Third-person over-the-shoulder view framing the whole worktable
            # (peg side + socket side). Look-at built from the same helper the
            # workbench cam uses, OpenGL convention.
            pos=(1.05, 0.0, 0.65),
            rot=_camera_lookat_quat((1.05, 0.0, 0.65), (0.45, 0.0, 0.03)),
            convention="opengl",
        ),
    )


# ─── Per-episode in-memory recorder (same as collect_demos.py). ──────────────
class EpisodeRecorder:
    """Buffer per-step arrays; flush to one HDF5 group at episode end."""

    def __init__(self) -> None:
        self._lists: dict[str, list[np.ndarray]] = {}

    def append(self, **named_arrays: np.ndarray) -> None:
        for k, v in named_arrays.items():
            self._lists.setdefault(k, []).append(np.asarray(v))

    @property
    def length(self) -> int:
        any_key = next(iter(self._lists), None)
        return 0 if any_key is None else len(self._lists[any_key])

    def flush_to_group(self, group: h5py.Group, attrs: dict) -> None:
        T = self.length
        if T == 0:
            return
        obs_grp = group.create_group("obs")
        for key, vlist in self._lists.items():
            arr = np.stack(vlist, axis=0)
            target, name = (
                (obs_grp, key[len("obs_"):]) if key.startswith("obs_")
                else (group, key)
            )
            ds_kwargs: dict = {}
            if "image" in name:
                ds_kwargs.update(
                    compression="gzip", compression_opts=1,
                    chunks=(1, arr.shape[1], arr.shape[2], arr.shape[3]),
                )
            target.create_dataset(name, data=arr, **ds_kwargs)
        dones = np.zeros(T, dtype=bool)
        dones[-1] = True
        group.create_dataset("dones", data=dones)
        group.create_dataset("rewards", data=np.zeros(T, dtype=np.float32))
        for k, v in attrs.items():
            group.attrs[k] = v
        group.attrs["length"] = T


# ─── Config / DMP loading (same pattern as the other runners). ───────────────
REQUIRED_KEYS = ("methods_dmp", "model_path", "demo_path",
                 "start", "goal", "n_basis", "ins_offset")


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"[gen] config {path} missing keys: {missing}")
    repo_root = Path(__file__).resolve().parents[2]
    for key in ("methods_dmp", "model_path", "demo_path"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else (repo_root / p).resolve()
    cfg.setdefault("device", "cuda")
    cfg.setdefault("seed", 0)
    cfg.setdefault("rescale", None)
    return cfg


def _bootstrap_paths(methods_dmp: Path) -> None:
    methods_dmp = methods_dmp.resolve()
    if not (methods_dmp / "core" / "dmp_wrapper.py").exists():
        raise SystemExit(f"[gen] {methods_dmp} doesn't look like the dmp methods dir.")
    sys.path.insert(0, str(methods_dmp))


def load_nn_policy(model_path: Path, device):
    nn_policy = torch.jit.load(str(model_path), map_location=device)
    nn_policy.eval()
    return nn_policy


def build_primed(cfg: dict, task_params, device, nn_policy):
    from configs.configs import DMPConfig          # noqa: E402
    from core.dmp_wrapper import DMPWrapper         # noqa: E402
    from core.dmp_policy import build_primed_dmp    # noqa: E402
    return build_primed_dmp(
        DMPWrapper=DMPWrapper, DMPConfig=DMPConfig, nn_policy=nn_policy,
        demo_path=str(cfg["demo_path"]), n_basis=cfg["n_basis"],
        task_params=tuple(task_params), ins_offset=tuple(cfg["ins_offset"]),
        start=np.asarray(cfg["start"], dtype=np.float32),
        goal=np.asarray(cfg["goal"], dtype=np.float32),
        rescale=cfg.get("rescale"), device=device,
    )


# ─── State reader (mirrors cluttered_run_e2e.update_states). ─────────────────
def update_states(scene, robot, ee_frame_idx, arm_joint_ids):
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[
        :, arm_joint_ids, :][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]

    jacobian_b = jacobian_w.clone()
    R_b = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(R_b, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(R_b, jacobian_b[:, 3:, :])

    root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]
    rel_vel_w = ee_vel_w - robot.data.root_vel_w
    ee_lin_b = quat_apply_inverse(root_quat_w, rel_vel_w[:, 0:3])
    ee_ang_b = quat_apply_inverse(root_quat_w, rel_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_b, ee_ang_b], dim=-1)

    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]
    return (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
            root_pose_w, joint_pos, joint_vel)


# ─── Workspace sampling. ─────────────────────────────────────────────────────
def _far_enough(xy, others, min_dists) -> bool:
    for o, md in zip(others, min_dists):
        if np.hypot(xy[0] - o[0], xy[1] - o[1]) < md:
            return False
    return True


def _point_seg_dist(p, a, b) -> float:
    """XY distance from point ``p`` to the segment ``a``→``b``."""
    p = np.asarray(p, float); a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _spec_layout():
    """Fallback: every object at its declared spec position."""
    return ({s["id"]: np.asarray(peg_spec(s["id"])["xy"], np.float32) for s in PEG_SPECS},
            {s["id"]: np.asarray(socket_spec(s["id"])["xy"], np.float32) for s in SOCKET_SPECS})


def sample_layout(rng):
    """Scatter ALL pegs + ALL sockets in the shared workspace.

    Returns ``(peg_xy: dict[id→xy], socket_xy: dict[id→xy])``. Sockets are
    placed first (larger footprint), then pegs, each rejection-sampled against
    everything already placed with the appropriate min-separation. A min
    transport distance is enforced for the TARGET pair (peg_0 / hole_0). Falls
    back to spec positions if placement fails.

    Randomizing every object (not just the target pair) is what gives the
    distractor sockets motion AND the full-360° relative peg→socket direction.
    """
    x0, x1, y0, y1 = args_cli.workspace
    socket_sep = SOCKET_BLOCK_SIZE_XY + 0.04       # socket↔socket (footprints)
    # peg center vs socket footprint: must clear the OPEN gripper (fingertips
    # reach ±0.04 from the peg axis and descend into the rim's z-band during the
    # grasp), else a neighboring socket's rim gets clipped on the way down.
    peg_socket = SOCKET_BLOCK_SIZE_XY / 2 + 0.05   # 0.05 + 0.05 = 0.10
    peg_sep = 0.05                                 # peg↔peg

    def _try_place(existing, min_dists):
        for _ in range(args_cli.max_place_tries):
            xy = (rng.uniform(x0, x1), rng.uniform(y0, y1))
            if _far_enough(xy, existing, min_dists):
                return xy
        return None

    for _ in range(args_cli.max_place_tries):
        socket_xy: dict = {}
        ok = True
        for s in SOCKET_SPECS:
            placed = _try_place(list(socket_xy.values()),
                                [socket_sep] * len(socket_xy))
            if placed is None:
                ok = False
                break
            socket_xy[s["id"]] = placed
        if not ok:
            continue

        peg_xy: dict = {}
        sockets = list(socket_xy.values())
        for s in PEG_SPECS:
            others = sockets + list(peg_xy.values())
            mins = [peg_socket] * len(sockets) + [peg_sep] * len(peg_xy)
            placed = _try_place(others, mins)
            if placed is None:
                ok = False
                break
            peg_xy[s["id"]] = placed
        if not ok:
            continue

        tp, th = peg_xy[TARGET_PEG_ID], socket_xy[TARGET_HOLE_ID]
        if np.hypot(tp[0] - th[0], tp[1] - th[1]) < args_cli.min_transport:
            continue

        # Keep distractors off the peg_0→hole_0 transit line (footprint-aware
        # for sockets), so the carried peg's arc isn't obstructed.
        if args_cli.corridor_clear > 0:
            clear = True
            for pid, xy in peg_xy.items():
                if pid != TARGET_PEG_ID and _point_seg_dist(xy, tp, th) < args_cli.corridor_clear:
                    clear = False
                    break
            for hid, xy in socket_xy.items():
                if hid != TARGET_HOLE_ID and _point_seg_dist(
                        xy, tp, th) < args_cli.corridor_clear + SOCKET_BLOCK_SIZE_XY / 2:
                    clear = False
                    break
            if not clear:
                continue

        return ({k: np.asarray(v, np.float32) for k, v in peg_xy.items()},
                {k: np.asarray(v, np.float32) for k, v in socket_xy.items()})

    print("[gen] WARN: placement rejection failed, using spec positions")
    return _spec_layout()


def safe_task_params(grasp_tcp_w, insert_tcp_w, rng):
    """Derive a collision-safe DMP lift task-param pair for this geometry.

    param1 (lift fraction) is sized so the transit apex rises ~``clear_margin``
    above the socket rim regardless of the (randomized) transport distance —
    this is the fix for the original collision (param1=0.05 under-cleared). The
    mapping to realized apex height is approximate; success-filtering is the
    guarantee. ``--task-params`` forces a fixed pair instead.
    """
    if args_cli.task_params is not None:
        return tuple(float(v) for v in args_cli.task_params)
    D = float(np.linalg.norm(np.asarray(grasp_tcp_w)[:2] - np.asarray(insert_tcp_w)[:2]))
    D = max(D, 1e-3)
    target_lift = (socket_top_z(TARGET_HOLE_ID) + args_cli.clear_margin) - TABLE_TOP_Z
    p1 = np.clip(target_lift / D, args_cli.p1_min, args_cli.p1_max)
    p1 = float(np.clip(p1 + rng.uniform(-args_cli.p1_jitter, args_cli.p1_jitter),
                       args_cli.p1_min, args_cli.p1_max))
    p2 = float(np.clip(args_cli.p2_center + rng.uniform(-args_cli.p2_jitter, args_cli.p2_jitter),
                       0.05, 0.5))
    return (p1, p2)


# ─── Scene reset for one episode. ────────────────────────────────────────────
def reset_episode(scene, robot, sim_dt, device, peg_xy, socket_xy):
    """Reset robot to default joints, place all pegs, teleport all sockets.

    ``peg_xy`` / ``socket_xy`` are id→xy dicts from ``sample_layout``.
    """
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.set_joint_effort_target(
        torch.zeros(scene.num_envs, robot.num_joints, device=device))
    robot.write_data_to_sim()
    robot.reset()
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    zero_vel = torch.zeros(scene.num_envs, 6, device=device)

    # Pegs: every peg to its sampled xy, standing upright.
    for pid, xy in peg_xy.items():
        pos = torch.tensor([xy[0], xy[1], TABLE_TOP_Z + PEG_LENGTH / 2],
                           device=device, dtype=torch.float32).unsqueeze(0)
        scene[pid].write_root_pose_to_sim(torch.cat([pos, ident], dim=-1))
        scene[pid].write_root_velocity_to_sim(zero_vel)

    # Sockets: teleport each socket's 5 kinematic prims to its sampled xy.
    for hid, xy in socket_xy.items():
        for key, dx, dy, zc in socket_component_offsets(hid):
            pos = torch.tensor([xy[0] + dx, xy[1] + dy, zc],
                               device=device, dtype=torch.float32).unsqueeze(0)
            scene[key].write_root_pose_to_sim(torch.cat([pos, ident], dim=-1))
            scene[key].write_root_velocity_to_sim(zero_vel)

    scene.update(sim_dt)
    robot.update(sim_dt)


def peg_insertion_tip_w(peg_pos_w, peg_quat_w):
    """World position of the peg's LOWER end — the end that enters the hole.

    The dynamically-grasped peg keeps its upright spawn orientation (local +Z =
    world UP), so its insertion end is the local -Z end. ``peg_tip_from_body``
    returns the local +Z (TOP) end, which is why the depth check read the peg
    1 cm ABOVE the rim (depth_frac ≈ -0.25) for fully-seated pegs. We take the
    lower-z of the two ends, so it's also robust to a tilted peg.
    """
    half = quat_apply(
        peg_quat_w,
        torch.tensor([0.0, 0.0, PEG_LENGTH / 2], device=peg_pos_w.device,
                     dtype=peg_pos_w.dtype).expand_as(peg_pos_w))
    end_a = peg_pos_w + half
    end_b = peg_pos_w - half
    return torch.where(end_a[..., 2:3] <= end_b[..., 2:3], end_a, end_b)


def _record_step(recorder, cameras, ee_pose_b, ee_vel_b, joint_pos,
                 robot, finger_ids, ee_target_pose_b):
    """Append one (obs, action) sample. obs reflects the START of this tick;
    the action is the OSC target just computed for it — correctly paired."""
    wrist_cam, external_cam = cameras
    wrist_rgb = wrist_cam.data.output["rgb"][0].cpu().numpy()
    external_rgb = external_cam.data.output["rgb"][0].cpu().numpy()
    if wrist_rgb.shape[-1] == 4:
        wrist_rgb = wrist_rgb[..., :3]
    if external_rgb.shape[-1] == 4:
        external_rgb = external_rgb[..., :3]
    recorder.append(
        obs_wrist_image=wrist_rgb.astype(np.uint8),
        obs_external_image=external_rgb.astype(np.uint8),
        obs_ee_pose_b=ee_pose_b[0].cpu().numpy().astype(np.float32),
        obs_ee_vel_b=ee_vel_b[0].cpu().numpy().astype(np.float32),
        obs_joint_pos=joint_pos[0].cpu().numpy().astype(np.float32),
        obs_gripper_pos=robot.data.joint_pos[0, finger_ids].cpu().numpy().astype(np.float32),
        actions=ee_target_pose_b[0].cpu().numpy().astype(np.float32),
    )


# ─── One DMP TCP segment (optionally recorded). ──────────────────────────────
def run_tcp_segment(*, dmp, weights, tau, start_tcp_w, goal_tcp_w,
                    sim, scene, robot, osc, command, ee_target_pose_b, down_quat_b,
                    finger_cmd, arm_joint_ids, finger_ids, ee_frame_idx,
                    joint_centers, sim_dt, recorder=None, cameras=None):
    """Integrate one TCP-space DMP segment. Records (obs, action) per tick when
    ``recorder`` is given (insert segment only)."""
    dmp.x_0 = torch.tensor(start_tcp_w, device=sim.device, dtype=torch.float32)
    dmp.x_goal = torch.tensor(goal_tcp_w, device=sim.device, dtype=torch.float32)
    dmp.reset_step(weights)
    quat_world = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    dmp_time = 0.0
    while simulation_app.is_running() and dmp_time < DMP_TIME_EXTENSION_FACTOR * tau:
        step_dt = sim_dt * args_cli.playback_speed
        tcp_target_w, _ = dmp.step(step_dt)
        tcp_target_w = tcp_target_w.to(sim.device)
        if tcp_target_w.dim() == 1:
            tcp_target_w = tcp_target_w.unsqueeze(0)
        dmp_time += step_dt

        (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
         root_pose_w, joint_pos, joint_vel) = update_states(
            scene, robot, ee_frame_idx, arm_joint_ids)

        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            tcp_target_w.expand(scene.num_envs, -1), quat_world)
        ee_target_pose_b[:, 0:3] = target_pos_b
        command[:, 0:3] = target_pos_b
        command[:, 3:7] = quat_world

        osc.set_command(command=command, current_ee_pose_b=ee_pose_b,
                        current_task_frame_pose_b=None)
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b, current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=torch.zeros(scene.num_envs, 3, device=sim.device),
            mass_matrix=mass_matrix, gravity=gravity,
            current_joint_pos=joint_pos, current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers)
        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids)
        robot.write_data_to_sim()

        if recorder is not None:
            _record_step(recorder, cameras, ee_pose_b, ee_vel_b, joint_pos,
                         robot, finger_ids, ee_target_pose_b)

        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)


def hold_tcp(*, target_tcp_pos_w, n_ticks, finger_cmd, sim, scene, robot, osc,
             command, ee_target_pose_b, down_quat_b, arm_joint_ids, finger_ids,
             ee_frame_idx, joint_centers, sim_dt):
    """Servo the TCP to a fixed world position for n_ticks (settle / grasp)."""
    if not torch.is_tensor(target_tcp_pos_w):
        target_tcp_pos_w = torch.tensor(target_tcp_pos_w, device=sim.device,
                                        dtype=torch.float32)
    target_world = target_tcp_pos_w.unsqueeze(0).expand(scene.num_envs, -1)
    quat_world = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)
    for _ in range(n_ticks):
        if not simulation_app.is_running():
            return
        (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
         root_pose_w, joint_pos, joint_vel) = update_states(
            scene, robot, ee_frame_idx, arm_joint_ids)
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_world, quat_world)
        ee_target_pose_b[:, 0:3] = target_pos_b
        command[:, 0:3] = target_pos_b
        command[:, 3:7] = quat_world
        osc.set_command(command=command, current_ee_pose_b=ee_pose_b,
                        current_task_frame_pose_b=None)
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b, current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=torch.zeros(scene.num_envs, 3, device=sim.device),
            mass_matrix=mass_matrix, gravity=gravity,
            current_joint_pos=joint_pos, current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers)
        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)


# ─── One episode. ────────────────────────────────────────────────────────────
def run_one_episode(sim, scene, robot, osc, cameras, nn_policy, cfg,
                    ee_frame_idx, arm_joint_ids, finger_ids, joint_centers,
                    down_quat_b, sim_dt, rng):
    device = sim.device
    peg_xy, socket_xy = sample_layout(rng)
    peg0_xy, hole0_xy = peg_xy[TARGET_PEG_ID], socket_xy[TARGET_HOLE_ID]

    grip_target = finger_grip_target(scene.num_envs, len(finger_ids), device)
    open_target = torch.full((scene.num_envs, len(finger_ids)), 0.04, device=device)

    command = torch.zeros(scene.num_envs, osc.action_dim, device=device)
    command[:, 3:7] = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=device)
    ee_target_pose_b[:, 3:7] = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    osc.reset()
    reset_episode(scene, robot, sim_dt, device, peg_xy, socket_xy)

    # --- Move to a home TCP above the workspace center, gripper open. ---
    # Servo to a chosen Cartesian home (closer to the table than the default
    # ready pose) rather than reading wherever the default joints land. This
    # shortens the first transit and starts the robot near the work area. Using
    # the OSC to reach a reachable Cartesian point avoids guessing joint angles.
    wx0, wx1, wy0, wy1 = args_cli.workspace
    home_target = torch.tensor(
        [(wx0 + wx1) / 2.0, (wy0 + wy1) / 2.0, args_cli.home_height],
        device=device, dtype=torch.float32)
    hold_tcp(target_tcp_pos_w=home_target,
             n_ticks=HOME_MOVE_TICKS, finger_cmd=open_target, sim=sim, scene=scene,
             robot=robot, osc=osc, command=command, ee_target_pose_b=ee_target_pose_b,
             down_quat_b=down_quat_b, arm_joint_ids=arm_joint_ids,
             finger_ids=finger_ids, ee_frame_idx=ee_frame_idx,
             joint_centers=joint_centers, sim_dt=sim_dt)
    home_tcp_w = robot.data.body_pos_w[:, ee_frame_idx, :][0].detach().cpu().numpy()

    # --- TCP goals from (randomized) geometry. ---
    peg_center_w = torch.tensor(
        [peg0_xy[0], peg0_xy[1], TABLE_TOP_Z + PEG_LENGTH / 2], device=device)
    grasp_tcp_w = hand_pos_for_grasp(
        peg_center_w.unsqueeze(0), down_quat_b.unsqueeze(0))[0].detach().cpu().numpy()

    # Insertion goal = peg tip at insert_depth_frac of the hole depth below the
    # rim, centered on the (teleported) hole. The DMP's own trained steep
    # descent carries the peg in and stops here — we do NOT add a separate
    # descent. Keep this SHALLOW (partial insertion): the residual policy seats
    # the rest later, and a shallow stop keeps the gripper well clear of the rim
    # (fingertips at tip_z+0.039; rim top at socket_top_z), so the TCP never
    # collides with the socket.
    rim_z = socket_top_z(TARGET_HOLE_ID)
    insert_tip = torch.tensor(
        [hole0_xy[0], hole0_xy[1], rim_z - args_cli.insert_depth_frac * HOLE_DEPTH_M],
        device=device)
    insertion_tcp_w = hand_pos_for_peg_tip(
        insert_tip.unsqueeze(0), down_quat_b.unsqueeze(0))[0].detach().cpu().numpy()

    # --- Per-episode DMP weights, sized for a collision-safe lift. ---
    task_params = safe_task_params(grasp_tcp_w, insertion_tcp_w, rng)
    primed = build_primed(cfg, task_params, device, nn_policy)
    dmp, weights, tau = primed.dmp, primed.weights, primed.tau

    # --- Goal pose in base frame (constant; recorded for conditioning). ---
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    goal_pos_b, goal_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w,
        torch.tensor(insertion_tcp_w, device=device).unsqueeze(0),
        down_quat_b.unsqueeze(0))
    goal_pose_b = torch.cat([goal_pos_b, goal_quat_b], dim=-1)[0].cpu().numpy().astype(np.float32)

    # --- Segment 1: DMP home → grasp (NOT recorded), gripper open. ---
    run_tcp_segment(dmp=dmp, weights=weights, tau=tau,
                    start_tcp_w=home_tcp_w, goal_tcp_w=grasp_tcp_w,
                    sim=sim, scene=scene, robot=robot, osc=osc, command=command,
                    ee_target_pose_b=ee_target_pose_b, down_quat_b=down_quat_b,
                    finger_cmd=open_target, arm_joint_ids=arm_joint_ids,
                    finger_ids=finger_ids, ee_frame_idx=ee_frame_idx,
                    joint_centers=joint_centers, sim_dt=sim_dt, recorder=None)

    # --- Grasp: hold at the canonical grasp TCP, close gripper. ---
    hold_tcp(target_tcp_pos_w=torch.tensor(grasp_tcp_w, device=device, dtype=torch.float32),
             n_ticks=GRIP_HOLD_TICKS, finger_cmd=grip_target, sim=sim, scene=scene,
             robot=robot, osc=osc, command=command, ee_target_pose_b=ee_target_pose_b,
             down_quat_b=down_quat_b, arm_joint_ids=arm_joint_ids, finger_ids=finger_ids,
             ee_frame_idx=ee_frame_idx, joint_centers=joint_centers, sim_dt=sim_dt)

    # --- Segment 2: DMP grasp → insert (RECORDED), gripper closed. ---
    # Single segment — the DMP's trained trajectory does the steep descent into
    # the hole and stops at the goal (same as cluttered_run_e2e.py, which seats
    # cleanly). No hand-rolled descent.
    post_grasp_tcp_w = robot.data.body_pos_w[:, ee_frame_idx, :][0].detach().cpu().numpy()
    recorder = EpisodeRecorder()
    run_tcp_segment(dmp=dmp, weights=weights, tau=tau,
                    start_tcp_w=post_grasp_tcp_w, goal_tcp_w=insertion_tcp_w,
                    sim=sim, scene=scene, robot=robot, osc=osc, command=command,
                    ee_target_pose_b=ee_target_pose_b, down_quat_b=down_quat_b,
                    finger_cmd=grip_target, arm_joint_ids=arm_joint_ids,
                    finger_ids=finger_ids, ee_frame_idx=ee_frame_idx,
                    joint_centers=joint_centers, sim_dt=sim_dt,
                    recorder=recorder, cameras=cameras)

    # --- RELEASE: open the gripper, then lift it clear, and let the peg settle.
    # We score AFTER release, not while gripped: a correctly-aligned peg drops
    # to the bottom of the hole under gravity (depth_frac → high), while a
    # misaligned one tips out — so this is the true "is it in the hole?" test
    # and avoids false negatives from the shallow gripped stop position. Not
    # recorded (the gripper isn't part of the TCP action space).
    peg_obj = scene[TARGET_PEG_ID]
    hold_tcp(target_tcp_pos_w=torch.tensor(insertion_tcp_w, device=device, dtype=torch.float32),
             n_ticks=RELEASE_OPEN_TICKS, finger_cmd=open_target, sim=sim, scene=scene,
             robot=robot, osc=osc, command=command, ee_target_pose_b=ee_target_pose_b,
             down_quat_b=down_quat_b, arm_joint_ids=arm_joint_ids, finger_ids=finger_ids,
             ee_frame_idx=ee_frame_idx, joint_centers=joint_centers, sim_dt=sim_dt)
    retract_tcp = insertion_tcp_w.copy()
    retract_tcp[2] += 0.12
    hold_tcp(target_tcp_pos_w=torch.tensor(retract_tcp, device=device, dtype=torch.float32),
             n_ticks=RETRACT_TICKS, finger_cmd=open_target, sim=sim, scene=scene,
             robot=robot, osc=osc, command=command, ee_target_pose_b=ee_target_pose_b,
             down_quat_b=down_quat_b, arm_joint_ids=arm_joint_ids, finger_ids=finger_ids,
             ee_frame_idx=ee_frame_idx, joint_centers=joint_centers, sim_dt=sim_dt)

    # --- Score the (released, settled) peg on its own body pose. ---
    # hole_center_w()/socket_top_z() are spec-relative in XY, so override the
    # hole-bottom XY to the TELEPORTED socket; its Z is spec-correct (the rim
    # heights don't move). Lateral error is then measured against the real hole.
    tip_w = peg_insertion_tip_w(peg_obj.data.root_pos_w, peg_obj.data.root_quat_w)
    hole_bottom = (float(hole0_xy[0]), float(hole0_xy[1]),
                   float(hole_center_w(TARGET_HOLE_ID)[2]))
    succ, lat, depth = insertion_success(
        tip_w, hole_center_w=hole_bottom,
        hole_radius=float(socket_spec(TARGET_HOLE_ID)["hole_radius"]),
        rim_top_z=socket_top_z(TARGET_HOLE_ID),
        depth_threshold=args_cli.success_depth_frac)

    meta = {
        "task_params": np.asarray(task_params, np.float32),
        "target_peg_id": TARGET_PEG_ID,
        "target_hole_id": TARGET_HOLE_ID,
        "peg_init_pos_w": np.asarray([peg0_xy[0], peg0_xy[1], TABLE_TOP_Z + PEG_LENGTH / 2], np.float32),
        "hole_xy": np.asarray(hole0_xy, np.float32),
        "hole_top_w": np.asarray([hole0_xy[0], hole0_xy[1], socket_top_z(TARGET_HOLE_ID)], np.float32),
        "goal_tcp_w": np.asarray(insertion_tcp_w, np.float32),
        "goal_pose_b": goal_pose_b,
        "task": TASK_STRING,
        "lateral_err_final": np.float32(lat[0].item()),
        "depth_frac_final": np.float32(depth[0].item()),
    }
    return recorder, bool(succ[0].item()), meta


# ─── Main. ───────────────────────────────────────────────────────────────────
def main() -> None:
    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args_cli.config)
    _bootstrap_paths(cfg["methods_dmp"])
    torch.manual_seed(args_cli.seed)
    rng = np.random.default_rng(args_cli.seed)
    device = torch.device(args_cli.device)
    print(f"[gen] config={args_cli.config} out={args_cli.out} "
          f"episodes={args_cli.num_episodes} device={device}")

    nn_policy = load_nn_policy(cfg["model_path"], device)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 1.0], [0.5, 0.0, 0.4])
    scene_cfg = ClutteredDataGenSceneCfg(num_envs=1, env_spacing=2.0)
    # The inherited perception camera is unused here — throttle it so it doesn't
    # re-render every tick (we only read wrist_cam + external_cam).
    scene_cfg.workbench_camera.update_period = 1.0e9
    attach_pegs_and_sockets(scene_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[gen] scene up.")

    robot: Articulation = scene["robot"]
    wrist_cam: Camera = scene["wrist_cam"]
    external_cam: Camera = scene["external_cam"]
    finger_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    ee_frame_idx = robot.find_bodies("panda_hand")[0][0]
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"], impedance_mode="fixed",
        motion_stiffness_task=[2000.0, 2000.0, 2000.0, 400.0, 400.0, 400.0],
        motion_damping_ratio_task=1.0, inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False, gravity_compensation=True,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1], nullspace_control="position",
        nullspace_stiffness=5.0, nullspace_damping_ratio=1.0)
    osc = OperationalSpaceController(osc_cfg, num_envs=1, device=sim.device)

    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)
    joint_centers = robot.data.default_joint_pos[:, arm_joint_ids].clone()
    down_quat_b = torch.tensor(GRIPPER_DOWN_QUAT, device=device)

    with h5py.File(args_cli.out, "w") as h5:
        data_grp = h5.create_group("data")
        h5.attrs["created_at"] = dt.datetime.utcnow().isoformat() + "Z"
        h5.attrs["config_path"] = str(args_cli.config)
        h5.attrs["playback_speed"] = args_cli.playback_speed
        h5.attrs["control_hz"] = SCENE_HZ
        h5.attrs["down_quat_wxyz"] = np.asarray(GRIPPER_DOWN_QUAT, np.float32)
        h5.attrs["action_layout"] = "pos3+quat4 (base frame, OSC TCP target)"
        h5.attrs["obs_image_keys"] = np.asarray(["wrist_image", "external_image"], dtype="S")
        h5.attrs["cam_height"] = args_cli.cam_height
        h5.attrs["cam_width"] = args_cli.cam_width
        h5.attrs["target_peg_id"] = TARGET_PEG_ID
        h5.attrs["target_hole_id"] = TARGET_HOLE_ID
        h5.attrs["task"] = TASK_STRING
        h5.attrs["seed"] = args_cli.seed

        n_kept = 0
        n_attempted = 0
        for ep in range(args_cli.num_episodes):
            if not simulation_app.is_running():
                break
            recorder, success, meta = run_one_episode(
                sim, scene, robot, osc, (wrist_cam, external_cam), nn_policy,
                cfg, ee_frame_idx, arm_joint_ids, finger_ids, joint_centers,
                down_quat_b, sim_dt, rng)
            n_attempted += 1
            print(f"[gen] ep {ep:04d} success={int(success)} "
                  f"lateral={meta['lateral_err_final']*1000:.1f}mm "
                  f"depth={meta['depth_frac_final']:+.2f} "
                  f"tp={meta['task_params'].tolist()} T={recorder.length} kept={n_kept}")

            if not success and not args_cli.keep_failures:
                continue
            grp = data_grp.create_group(f"demo_{n_kept}")
            recorder.flush_to_group(grp, attrs={**meta, "success": bool(success)})
            n_kept += 1

        h5.attrs["num_episodes_attempted"] = n_attempted
        h5.attrs["num_episodes_kept"] = n_kept
        rate = (n_kept / n_attempted) if n_attempted else 0.0
        print(f"[gen] DONE: kept {n_kept}/{n_attempted} "
              f"(success rate {rate*100:.0f}%) → {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
