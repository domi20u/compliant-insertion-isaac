"""Initial diffusion-policy data collection: DMP rollouts with wrist + external cams.

Counterpart to ``run_dmp_in_sim_e2e.py``. Same phase machine
(APPROACH → DESCEND → GRASP → DMP), but the recording window is sliced to the
DMP phase only: approach/grasp are out-of-distribution for a policy whose
initial state at deployment is "post-grasp, peg in hand", so including those
states teaches transients the policy should never see.

Per-episode variation (v1 — minimal):
  - task_params: U(low, high) per axis. Drives the NN → DMP-weights map and
    therefore the trajectory shape, which is the main axis of diversity.
  - peg start position: small XY jitter around ``PEG_PLACE_POS``. Because the
    DMP is rotodilatation-rescaled to (x_0, x_goal), shifting the peg shifts
    the relative pick-place geometry, giving spatial coverage for free.

Held fixed for v1 (extend later):
  - socket position (would require moving the 4 rim cuboids each reset).
  - gripper orientation (constant ``GRIPPER_DOWN_QUAT`` throughout).
  - lighting and external-cam viewpoint (no domain randomization yet).
  - corrective / perturbed demos (would help DP recover from off-nominal
    states; left for v2 — at that point inject jitter into the DMP target
    stream and let the OSC's stiffness pull the peg back).

Output: one HDF5 file in RoboMimic-ish layout. One group per kept episode:

    data/demo_N/
      obs/
        wrist_image       [T, H, W, 3] uint8
        external_image    [T, H, W, 3] uint8
        ee_pose_b         [T, 7] float32   pos + quat (wxyz), base frame
        ee_vel_b          [T, 6] float32   lin + ang, base frame
        joint_pos         [T, 7] float32
        gripper_pos       [T, 2] float32
      actions             [T, 7] float32   OSC pose target THIS step (pos + quat)
      rewards             [T]    float32   zeros (recompute offline if wanted)
      dones               [T]    bool      only last step is True
      @attrs:
        success           bool
        task_params       [2] float32
        peg_init_pos_w    [3] float32
        socket_pos_w      [3] float32
        lateral_err_final float32
        depth_frac_final  float32
        length            int

Action layout note: 7-DoF action (3 pos + 4 quat) is recorded even though
orientation is constant. This keeps the schema generalizable to variable-
orientation tasks; a DP cloned on this dataset will simply learn the quat
component as fixed. Drop columns 3–7 in the dataloader if you want a 3-DoF
position-only policy.

Cameras:
  - wrist_cam: child of ``panda_hand``, looking along the gripper +Z (out the
    fingertips, toward whatever the peg points at). Best view for fine
    alignment at the socket.
  - external_cam: fixed world pose, oblique front-right view. Global context.

Camera offsets are hand-tuned approximations — replace with a programmatic
look-at via ``Camera.set_world_poses`` after init if you want exact framing.

Usage::

    /path/to/IsaacLab/isaaclab.sh -p scripts/data/collect_demos.py \\
        --config configs/insertion_traj.yaml \\
        --out data/insertion_demos.hdf5 \\
        --num-episodes 200
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

# ─── Step 1: launch Isaac Sim BEFORE any isaaclab.* imports. ─────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect DMP-driven demos for diffusion-policy training.")
parser.add_argument("--config", type=Path, required=True,
                    help="YAML config (same schema as run_dmp_in_sim_e2e.py).")
parser.add_argument("--out", type=Path, required=True,
                    help="HDF5 output path. Parent dir is created if absent.")
parser.add_argument("--num-episodes", type=int, default=100)
parser.add_argument("--task-param-low", type=float, nargs=2, default=(0.02, 0.10),
                    help="Lower bound per axis for task_params sampling.")
parser.add_argument("--task-param-high", type=float, nargs=2, default=(0.08, 0.30),
                    help="Upper bound per axis for task_params sampling.")
parser.add_argument("--peg-jitter-xy", type=float, default=0.03,
                    help="Half-extent (m) of uniform XY jitter on peg start.")
parser.add_argument("--playback-speed", type=float, default=5.0)
parser.add_argument("--cam-height", type=int, default=128)
parser.add_argument("--cam-width", type=int, default=128)
parser.add_argument("--keep-failures", action="store_true",
                    help="Save unsuccessful episodes too (success attr=False).")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
    matrix_from_quat, quat_apply_inverse, quat_inv,
    subtract_frame_transforms,
)

from compliant_insertion.env.scene_cfg import (
    InsertionSceneCfg,
    SOCKET_X, SOCKET_Y, SOCKET_TOP_Z,
    PEG_LENGTH, PEG_PLACE_POS,
    GRIPPER_DOWN_QUAT,
    insertion_success,
    peg_tip_from_body,
    hand_pos_for_grasp, hand_pos_for_peg_tip,
    finger_grip_target,
)


# ─── Phase geometry (mirrors run_dmp_in_sim_e2e.py). ─────────────────────────
APPROACH_CLEARANCE = 0.12
DMP_TIME_EXTENSION_FACTOR = 1.05
SEAT_TIP_Z = SOCKET_TOP_Z


# ─── Scene: insertion scene + cameras. ───────────────────────────────────────
@configclass
class DataGenSceneCfg(InsertionSceneCfg):
    """``InsertionSceneCfg`` + wrist cam on panda_hand + fixed external cam.

    update_period=0.0 → cameras update every render tick, so per-step
    observations are fresh. For multi-env scaling later, swap to
    ``TiledCameraCfg`` (batched GPU readback). For num_envs=1 ``Camera`` is
    simpler and integrates cleanly with ``scene.update(dt)``.
    """

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=args_cli.cam_height,
        width=args_cli.cam_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,                 # wider than default for close-in view
            horizontal_aperture=20.955,
            clipping_range=(0.01, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Behind the hand origin along the gripper's -Z (away from the
            # fingers), so the FOV encompasses the peg tip + whatever the peg
            # points at (socket during DMP). +Z of the parent panda_hand is
            # the gripper-forward axis; ROS convention has cam +Z = forward,
            # so identity rotation already looks along panda_hand +Z.
            pos=(0.0, 0.0, -0.05),
            rot=(1.0, 0.0, 0.0, 0.0),  # identity (wxyz)
            convention="ros",
        ),
    )

    external_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        update_period=0.0,
        height=args_cli.cam_height,
        width=args_cli.cam_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Oblique front-right view. Hand-tuned to frame both the peg pick
            # area and the socket; TUNE VISUALLY for your exact placement —
            # quaternions written from scratch are rarely right first try.
            pos=(1.2, -0.6, 0.55),
            rot=(0.5, -0.2, 0.4, 0.75),
            convention="ros",
        ),
    )


# ─── Per-episode in-memory recorder. ─────────────────────────────────────────
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
                # gzip-1: fast, ~3–5× compression on RGB; chunk = one frame so
                # random-access reads (which a DataLoader does heavily) only
                # decompress what's actually needed.
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


# ─── DMP construction. ───────────────────────────────────────────────────────
REQUIRED_KEYS = (
    "methods_dmp", "model_path", "demo_path",
    "start", "goal", "n_basis", "ins_offset",
)


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
        raise SystemExit(
            f"[gen] {methods_dmp} doesn't look like the dmp methods dir."
        )
    sys.path.insert(0, str(methods_dmp))


def load_nn_policy(model_path: Path, device):
    """Load the NN → DMP-weights policy ONCE per session (cache across episodes)."""
    nn_policy = torch.jit.load(str(model_path), map_location=device)
    nn_policy.eval()
    return nn_policy


def build_primed(cfg: dict, task_params, device, nn_policy):
    """Prime a DMP for these task_params, reusing the already-loaded NN.

    The demo imitation + rotodilatation registration still runs each call —
    cheap enough at the per-episode rate. The expensive part (loading the
    JIT model) is amortized in ``load_nn_policy``.
    """
    from configs.configs import DMPConfig          # noqa: E402
    from core.dmp_wrapper import DMPWrapper        # noqa: E402
    from core.dmp_policy import build_primed_dmp   # noqa: E402

    return build_primed_dmp(
        DMPWrapper=DMPWrapper, DMPConfig=DMPConfig,
        nn_policy=nn_policy,
        demo_path=str(cfg["demo_path"]),
        n_basis=cfg["n_basis"],
        task_params=tuple(task_params),
        ins_offset=tuple(cfg["ins_offset"]),
        start=np.asarray(cfg["start"], dtype=np.float32),
        goal=np.asarray(cfg["goal"], dtype=np.float32),
        rescale=cfg.get("rescale"),
        device=device,
    )


# ─── State reader (mirrors run_dmp_in_sim_e2e.py). ──────────────────────────
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
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]
    root_vel_w = robot.data.root_vel_w
    rel_vel_w = ee_vel_w - root_vel_w
    ee_lin_b = quat_apply_inverse(root_quat_w, rel_vel_w[:, 0:3])
    ee_ang_b = quat_apply_inverse(root_quat_w, rel_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_b, ee_ang_b], dim=-1)

    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
        root_pose_w, joint_pos, joint_vel,
    )


# ─── One episode: reset → approach → grasp → DMP (recorded). ─────────────────
def run_one_episode(
    sim, scene, robot, peg, osc, cameras,
    primed, ee_frame_idx, arm_joint_ids, finger_ids,
    joint_centers, peg_init_pos_w, sim_dt,
) -> tuple[EpisodeRecorder, bool, dict]:
    """Run one full episode. Records ONLY during the DMP phase."""
    wrist_cam, external_cam = cameras
    device = sim.device

    APPROACH_STEPS, DESCEND_STEPS, GRASP_STEPS = 120, 80, 40
    PHASE_APPROACH_END = APPROACH_STEPS
    PHASE_DESCEND_END = APPROACH_STEPS + DESCEND_STEPS
    PHASE_GRASP_END = PHASE_DESCEND_END + GRASP_STEPS

    # --- Reset robot + peg. ---
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    robot.set_joint_effort_target(
        torch.zeros(scene.num_envs, robot.num_joints, device=device))
    robot.write_data_to_sim()
    robot.reset()
    robot.update(sim_dt)

    peg_init_t = peg_init_pos_w.unsqueeze(0)  # [1, 3]
    peg_reset_pose = torch.cat([
        peg_init_t,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
    ], dim=-1)
    peg.write_root_pose_to_sim(peg_reset_pose)
    peg.write_root_velocity_to_sim(torch.zeros(scene.num_envs, 6, device=device))

    # Per-episode poses derived from THIS peg position.
    down_quat_b = torch.tensor(GRIPPER_DOWN_QUAT, device=device)
    grasp_hand_pos = hand_pos_for_grasp(peg_init_t, down_quat_b.unsqueeze(0))[0]
    approach_hand_pos = grasp_hand_pos.clone()
    approach_hand_pos[2] += APPROACH_CLEARANCE

    # DMP endpoints in peg-tip world frame. x_0 will be re-set to the ACTUAL
    # tip pose when the DMP phase starts (handles small grasp-induced drift).
    pick_tip_w = peg_init_t[0] + torch.tensor([0.0, 0.0, PEG_LENGTH / 2], device=device)
    seat_tip_w = torch.tensor([SOCKET_X, SOCKET_Y, SEAT_TIP_Z], device=device)
    primed.dmp.x_0 = pick_tip_w.clone()
    primed.dmp.x_goal = seat_tip_w.clone()
    tau = primed.tau

    # Reusable OSC tensors.
    command = torch.zeros(scene.num_envs, osc.action_dim, device=device)
    command[:, 3:7] = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=device)
    ee_target_pose_b[:, 3:7] = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    grip_target = finger_grip_target(scene.num_envs, len(finger_ids), device)
    open_target = torch.full((scene.num_envs, len(finger_ids)), 0.04, device=device)

    recorder = EpisodeRecorder()
    osc.reset()

    dmp_active = False
    dmp_time = 0.0
    last_target_xyz = approach_hand_pos.clone()
    cycle_step = 0
    max_dmp_steps = int(
        DMP_TIME_EXTENSION_FACTOR * tau / (sim_dt * args_cli.playback_speed)
    ) + 5

    success_final = False
    final_lateral = float("nan")
    final_depth_frac = float("nan")

    while True:
        # --- Phase machine: pick THIS step's target + finger command. ---
        if cycle_step < PHASE_APPROACH_END:
            phase, target_xyz, finger_cmd = "APPROACH", approach_hand_pos, open_target
        elif cycle_step < PHASE_DESCEND_END:
            phase, target_xyz, finger_cmd = "DESCEND", grasp_hand_pos, open_target
        elif cycle_step < PHASE_GRASP_END:
            phase, target_xyz, finger_cmd = "GRASP", grasp_hand_pos, grip_target
        else:
            phase = "DMP"
            finger_cmd = grip_target
            if not dmp_active:
                actual_tip = peg_tip_from_body(
                    peg.data.root_pos_w, peg.data.root_quat_w
                )[0]
                primed.dmp.x_0 = actual_tip.clone()
                primed.dmp.reset_step(primed.weights)
                dmp_active = True
                dmp_time = 0.0
            if dmp_time < DMP_TIME_EXTENSION_FACTOR * tau:
                step_dt = sim_dt * args_cli.playback_speed
                tip_pos, _ = primed.dmp.step(step_dt)
                last_target_xyz = hand_pos_for_peg_tip(
                    tip_pos.to(device), down_quat_b.unsqueeze(0)
                )[0]
                dmp_time += step_dt
            target_xyz = last_target_xyz

        # --- Read state, transform target to base frame, command OSC. ---
        (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
         root_pose_w, joint_pos, joint_vel) = update_states(
            scene, robot, ee_frame_idx, arm_joint_ids)

        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_xyz.unsqueeze(0).expand(scene.num_envs, -1),
            down_quat_b.unsqueeze(0).expand(scene.num_envs, -1),
        )
        ee_target_pose_b[:, 0:3] = target_pos_b
        command[:, 0:3] = ee_target_pose_b[:, 0:3]

        osc.set_command(
            command=command,
            current_ee_pose_b=ee_pose_b,
            current_task_frame_pose_b=None,
        )
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=torch.zeros(scene.num_envs, 3, device=device),
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers,
        )
        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids)
        robot.write_data_to_sim()

        # --- RECORD during DMP only. ee_pose_b / cam_data here reflect the
        #     state at the START of this tick — the action computed above
        #     is "what to do GIVEN this observation", so (obs, action) is
        #     correctly paired. ---
        if phase == "DMP":
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

        # --- Advance sim, evaluate end-of-episode conditions. ---
        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)
        cycle_step += 1

        if phase == "DMP" and dmp_time >= DMP_TIME_EXTENSION_FACTOR * tau:
            tip_w = peg_tip_from_body(peg.data.root_pos_w, peg.data.root_quat_w)
            succ, lat, depth = insertion_success(tip_w)
            success_final = bool(succ[0])
            final_lateral = float(lat[0])
            final_depth_frac = float(depth[0])
            break
        if cycle_step > PHASE_GRASP_END + max_dmp_steps + 50:
            break  # safety timeout — DMP somehow didn't complete

    return recorder, success_final, {
        "lateral_err_final": final_lateral,
        "depth_frac_final": final_depth_frac,
        "dmp_steps_recorded": recorder.length,
    }


# ─── Main: spin up sim once, loop episodes, write HDF5. ─────────────────────
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
    scene_cfg = DataGenSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[gen] scene up.")

    robot: Articulation = scene["robot"]
    peg: RigidObject = scene["peg"]
    wrist_cam: Camera = scene["wrist_cam"]
    external_cam: Camera = scene["external_cam"]

    finger_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    ee_frame_idx = robot.find_bodies("panda_hand")[0][0]
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]

    # OSC: same gains as the (now-stable) run_dmp_in_sim_e2e.py.
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        motion_stiffness_task=[2000.0, 2000.0, 2000.0, 400.0, 400.0, 400.0],
        motion_damping_ratio_task=1.0,
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=True,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
        nullspace_control="position",
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=1, device=sim.device)

    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)
    joint_centers = robot.data.default_joint_pos[:, arm_joint_ids].clone()

    nominal_peg = torch.tensor(PEG_PLACE_POS, device=device)
    socket_pos_w = np.array([SOCKET_X, SOCKET_Y, SOCKET_TOP_Z], dtype=np.float32)
    tp_low = np.asarray(args_cli.task_param_low, dtype=np.float32)
    tp_high = np.asarray(args_cli.task_param_high, dtype=np.float32)

    with h5py.File(args_cli.out, "w") as h5:
        data_grp = h5.create_group("data")
        h5.attrs["created_at"] = dt.datetime.utcnow().isoformat() + "Z"
        h5.attrs["config_path"] = str(args_cli.config)
        h5.attrs["playback_speed"] = args_cli.playback_speed
        h5.attrs["down_quat_wxyz"] = np.asarray(GRIPPER_DOWN_QUAT, dtype=np.float32)
        h5.attrs["action_layout"] = "pos3+quat4 (base frame, OSC target)"
        h5.attrs["cam_height"] = args_cli.cam_height
        h5.attrs["cam_width"] = args_cli.cam_width
        h5.attrs["socket_pos_w"] = socket_pos_w
        h5.attrs["task_param_low"] = tp_low
        h5.attrs["task_param_high"] = tp_high
        h5.attrs["peg_jitter_xy"] = args_cli.peg_jitter_xy
        h5.attrs["seed"] = args_cli.seed

        n_kept = 0
        for ep in range(args_cli.num_episodes):
            tp = rng.uniform(tp_low, tp_high).astype(np.float32)
            peg_dxy = rng.uniform(
                -args_cli.peg_jitter_xy, args_cli.peg_jitter_xy, size=2
            )
            peg_init = nominal_peg + torch.tensor(
                [float(peg_dxy[0]), float(peg_dxy[1]), 0.0], device=device,
            )

            primed = build_primed(cfg, tuple(tp.tolist()), device, nn_policy)
            recorder, success, ep_meta = run_one_episode(
                sim, scene, robot, peg, osc,
                (wrist_cam, external_cam),
                primed, ee_frame_idx, arm_joint_ids, finger_ids,
                joint_centers, peg_init, sim_dt,
            )

            print(f"[gen] ep {ep:04d} success={int(success)} "
                  f"lateral={ep_meta['lateral_err_final']*1000:.1f}mm "
                  f"depth={ep_meta['depth_frac_final']:+.2f} "
                  f"T={ep_meta['dmp_steps_recorded']} "
                  f"tp={tp.tolist()} kept={n_kept}")

            if not success and not args_cli.keep_failures:
                continue

            grp = data_grp.create_group(f"demo_{n_kept}")
            recorder.flush_to_group(grp, attrs={
                "success": bool(success),
                "task_params": tp,
                "peg_init_pos_w": peg_init.cpu().numpy().astype(np.float32),
                "socket_pos_w": socket_pos_w,
                "lateral_err_final": np.float32(ep_meta["lateral_err_final"]),
                "depth_frac_final": np.float32(ep_meta["depth_frac_final"]),
            })
            n_kept += 1

        h5.attrs["num_episodes_attempted"] = args_cli.num_episodes
        h5.attrs["num_episodes_kept"] = n_kept
        print(f"[gen] DONE: kept {n_kept}/{args_cli.num_episodes} → {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
