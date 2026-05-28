"""End-to-end: Franka executes a policy-primed DMP *step-wise* under OSC.

This replaces the two-pass workflow (generate a .npy trajectory, then replay
it) with a single pass: the DMP is integrated one step per sim tick, inside
the control loop, so the waypoint stream is produced on demand. No
intermediate trajectory file is written or read.

Execution model (per reset cycle):

  1. Steps 0 .. ``--rollout-start-step``: the arm holds a static hover pose
     above the socket. This lets you confirm OSC is healthy before motion.
  2. At ``--rollout-start-step``: the DMP integrator is (re)initialised with
     ``reset_step`` and then advanced by ``sim_dt * playback_speed`` each
     tick via ``step``. The returned position IS the OSC target — there is
     no interpolation step, because the DMP is integrated natively at the
     sim rate.
  3. Once the integrated DMP time exceeds ``tau`` (the demo's duration,
     divided by playback speed in wall-clock terms), we stop stepping and
     hold the final commanded pose. The DMP asymptotes to the goal, so the
     last ``step`` output already sits at the goal; holding it is just
     skipping further integration to save a little compute.
  4. On periodic reset (every ``--reset-period`` steps), joints return to
     default and the DMP integrator is re-initialised for the next cycle.

The DMP itself, its imitation demo, the trained NN policy, and the
rotodilatation endpoints all come from the same YAML config used by
``gen_insertion_trajectory.py``. The start/goal in that config define the
runtime endpoints; rotodilatation (if enabled) rotates+scales the demo's
shape onto them.

Usage::

    /path/to/IsaacLab/isaaclab.sh -p scripts/sim/run_dmp_in_sim_e2e.py \
        --config configs/insertion_traj.yaml \
        --task-params 0.05 0.20
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ─── Step 1: launch Isaac Sim BEFORE any isaaclab.* imports. ─────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="DMP-in-sim end-to-end: step-wise DMP execution under OSC.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs. Keep at 1 for this script.")
parser.add_argument("--config", type=Path, required=True,
                    help="YAML config (same schema as gen_insertion_trajectory.py): "
                         "paths, start/goal, n_basis, ins_offset, rescale, ...")
parser.add_argument("--task-params", type=float, nargs=2, required=True,
                    metavar=("P1", "P2"),
                    help="The two insertion task parameters fed to the policy.")
parser.add_argument("--rollout-start-step", type=int, default=200,
                    help="Sim step within each reset cycle at which DMP "
                         "stepping begins. Before this, the arm holds the "
                         "static hover pose.")
parser.add_argument("--reset-period", type=int, default=800,
                    help="Sim steps between joint resets. Make this larger "
                         "than rollout_start_step + tau/sim_dt/playback_speed "
                         "so the motion finishes before the cycle restarts.")
parser.add_argument("--playback-speed", type=float, default=1.0,
                    help="Multiplier on how fast the DMP clock advances per "
                         "sim tick. 1.0 runs the DMP at its native duration; "
                         "2.0 finishes in half the wall time. Implemented by "
                         "stepping the integrator with dt = sim_dt * speed, "
                         "so the command stream stays smooth at the sim rate "
                         "regardless of speed (no waypoint staircase).")
parser.add_argument("--hold-after-tau", action="store_true", default=True,
                    help="Stop integrating once DMP time exceeds tau and hold "
                         "the last command. On by default; the DMP has "
                         "converged to the goal by then.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Step 2: everything else can now be imported. ────────────────────────────
import sys

import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

from compliant_insertion.env.scene_cfg import (
    InsertionSceneCfg,
    SOCKET_X, SOCKET_Y, SOCKET_TOP_Z,
    PEG_LENGTH,
    peg_pose_from_hand,
    peg_tip_pose_from_hand,
    insertion_success,
)


# ─── Static-hold target (pre-rollout hover). ─────────────────────────────────
HOVER_OFFSET = 0.10
HAND_TARGET_Z = SOCKET_TOP_Z + HOVER_OFFSET + PEG_LENGTH
STATIC_TARGET_POS = (SOCKET_X, SOCKET_Y, HAND_TARGET_Z)


# ─── Config + DMP construction ───────────────────────────────────────────────
REQUIRED_KEYS = (
    "methods_dmp", "model_path", "demo_path",
    "start", "goal", "n_basis", "ins_offset",
)


def load_config(path: Path) -> dict:
    """Load + validate the YAML config (mirrors gen_insertion_trajectory.py).

    Note: ``out_path`` is NOT required here — this script never writes a
    trajectory file. Everything else matches so a single config drives both
    the offline generator and this end-to-end runner.
    """
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"[run] config {path} is missing keys: {missing}")

    repo_root = Path(__file__).resolve().parents[2]
    for key in ("methods_dmp", "model_path", "demo_path"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else (repo_root / p).resolve()

    cfg.setdefault("device", "cuda")
    cfg.setdefault("seed", 0)
    cfg.setdefault("rescale", None)

    offs = cfg["ins_offset"]
    if not (isinstance(offs, (list, tuple)) and len(offs) == 2):
        raise SystemExit(
            f"[run] ins_offset must be a 2-element list, got {offs!r}")
    if cfg["rescale"] not in (None, "rotodilatation", "rotodilatation_xy"):
        raise SystemExit(
            f"[run] rescale must be None, 'rotodilatation', or "
            f"'rotodilatation_xy' (got {cfg['rescale']!r}).")
    return cfg


def _bootstrap_paths(methods_dmp: Path) -> None:
    methods_dmp = methods_dmp.resolve()
    if not (methods_dmp / "core" / "dmp_wrapper.py").exists():
        raise SystemExit(
            f"[run] {methods_dmp} doesn't look like the dmp methods dir "
            f"(no core/dmp_wrapper.py). Check `methods_dmp` in the config.")
    sys.path.insert(0, str(methods_dmp))


def build_dmp_from_config(cfg: dict, task_params, device: torch.device):
    """Load the policy and build a policy-primed DMP ready for step-wise use.

    Returns a ``PrimedDMP`` (see core.dmp_policy). The wrapper inside it has
    already imitated the demo, had rotodilatation endpoints registered, and
    had the policy's y/z forcing weights spliced in.
    """
    # Deferred imports — only valid after _bootstrap_paths.
    from configs.configs import DMPConfig          # noqa: E402
    from core.dmp_wrapper import DMPWrapper        # noqa: E402
    from core.dmp_policy import build_primed_dmp    # noqa: E402

    nn_policy = torch.jit.load(str(cfg["model_path"]), map_location=device)
    nn_policy.eval()
    print(f"[run] loaded NN policy from {cfg['model_path']}")

    primed = build_primed_dmp(
        DMPWrapper=DMPWrapper, DMPConfig=DMPConfig,
        nn_policy=nn_policy,
        demo_path=str(cfg["demo_path"]),
        n_basis=cfg["n_basis"],
        task_params=tuple(task_params),
        ins_offset=tuple(cfg["ins_offset"]),
        start=np.asarray(cfg["start"], dtype=np.float32),
        goal=np.asarray(cfg["goal"], dtype=np.float32),
        rescale=cfg["rescale"],
        device=device,
    )
    print(f"[run] DMP primed: tau={primed.tau:.3f}s, "
          f"demo_x0={primed.demo_x0.tolist()} demo_xg={primed.demo_xg.tolist()}")
    print(f"[run] runtime start={cfg['start']} goal={cfg['goal']} "
          f"rescale={cfg['rescale']!r}")
    print(f"[run] nn_input={primed.nn_input.tolist()}")
    return primed


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene,
                  primed):
    """Runs the simulation loop with step-wise DMP execution."""

    robot: Articulation = scene["robot"]
    peg: RigidObject = scene["peg"]
    ee_frame_name = "panda_hand"
    arm_joint_names = ["panda_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    dmp = primed.dmp
    dmp_weights = primed.weights
    tau = primed.tau

    # ----- Build the OSC -----
    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        impedance_mode="fixed",
        motion_stiffness_task=[200.0, 200.0, 200.0, 30.0, 30.0, 30.0],
        motion_damping_ratio_task=1.0,
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=False,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1],
        nullspace_control="position",
        nullspace_stiffness=5.0,
        nullspace_damping_ratio=1.0,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # ----- Markers -----
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    sim_dt = sim.get_physics_dt()
    robot.update(dt=sim_dt)

    # ----- Nullspace target -----
    joint_centers = torch.mean(
        robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1
    )

    # ----- Read the home-pose hand quaternion (target orientation). -----
    (
        _, _, _, ee_pose_b, _, _, _, _, _,
    ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)
    home_quat_b = ee_pose_b[0, 3:7].clone()
    print(f"[info] home ee pose (base frame): "
          f"pos={ee_pose_b[0, 0:3].tolist()}, quat(wxyz)={home_quat_b.tolist()}")

    # ----- Sanity check: does the DMP start near the static hover pose? -----
    dmp_start = dmp.x_0.detach().cpu().numpy()
    start_err = float(np.linalg.norm(
        dmp_start - np.asarray(STATIC_TARGET_POS, dtype=dmp_start.dtype)))
    if start_err > 0.05:
        print(f"[warn] DMP start is {start_err*100:.1f} cm from the static-hold "
              f"target — expect a jerk at step {args_cli.rollout_start_step}. "
              f"Set the config 'start' to match the hover position.")
    # How many sim ticks the motion will take at this playback speed.
    est_steps = int(tau / (sim_dt * args_cli.playback_speed))
    print(f"[info] step-wise mode: hold static until step "
          f"{args_cli.rollout_start_step}, then step DMP for ~{est_steps} ticks "
          f"(tau={tau:.2f}s, speed={args_cli.playback_speed}), then hold goal.")

    # ----- Reusable command tensor. command[:, :3] is mutated each step to
    # the current DMP waypoint; orientation stays at home_quat_b. -----
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:, 3:7] = home_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_b[:, 3:7] = home_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)

    # ----- Step-wise DMP execution state. -----
    dmp_time = 0.0            # integrated DMP time (s) since rollout start
    dmp_active = False        # has reset_step been called this cycle?
    last_target_xyz = torch.tensor(STATIC_TARGET_POS, device=sim.device)
    cycle_step = 0            # steps since last reset

    count = 0
    while simulation_app.is_running():
        is_reset_step = (count % args_cli.reset_period) == 0

        if is_reset_step:
            # ----- Periodic reset -----
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)
            robot.write_data_to_sim()
            robot.reset()
            robot.update(sim_dt)
            cycle_step = 0
            dmp_time = 0.0
            dmp_active = False
            last_target_xyz = torch.tensor(STATIC_TARGET_POS, device=sim.device)
            print(f"[run] reset at step {count}")

        # ----- Compute the current target position. -----
        if cycle_step >= args_cli.rollout_start_step:
            if not dmp_active:
                # First tick of the rollout phase: (re)initialise the DMP
                # integrator from the configured start/goal + policy weights.
                dmp.reset_step(dmp_weights)
                dmp_active = True
                dmp_time = 0.0
                print(f"[run] DMP step-wise rollout started at step {count}")

            if (not args_cli.hold_after_tau) or (dmp_time < tau):
                # Advance the DMP by one (speed-scaled) tick. The returned
                # position is the OSC target directly — no interpolation.
                step_dt = sim_dt * args_cli.playback_speed
                pos, vel = dmp.step(step_dt)        # pos: [1, 3]
                # DMP is built on the sim device in main(); the .to() is a
                # cheap no-op guard in case a future caller changes that.
                last_target_xyz = pos[0].to(sim.device)
                dmp_time += step_dt
            # else: hold last_target_xyz (DMP has converged to the goal).

            target_xyz = last_target_xyz
        else:
            # Static hover before the rollout phase.
            target_xyz = torch.tensor(STATIC_TARGET_POS, device=sim.device)

        ee_target_pose_b[:, 0:3] = target_xyz.unsqueeze(0).expand(scene.num_envs, -1)
        command[:, 0:3] = ee_target_pose_b[:, 0:3]

        # ----- Read states. -----
        (
            jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
            root_pose_w, ee_pose_w, joint_pos, joint_vel,
        ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)

        tip_pos_w = peg_tip_pose_from_hand(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        success, lateral_err, depth_frac = insertion_success(tip_pos_w)

        if count % 50 == 0:
            print(f"[insert] tip_z={tip_pos_w[0, 2]:.3f} "
                  f"lateral={lateral_err[0]*1000:.1f}mm "
                  f"depth_frac={depth_frac[0]:+.2f} "
                  f"success={bool(success[0])}")

        # On reset steps, re-prime the OSC with the current EE pose.
        if is_reset_step:
            osc.reset()
        osc.set_command(
            command=command,
            current_ee_pose_b=ee_pose_b,
            current_task_frame_pose_b=None,
        )

        ee_force_b = torch.zeros(scene.num_envs, 3, device=sim.device)
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers,
        )

        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.write_data_to_sim()

        # ----- Markers. -----
        ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7],
        )
        ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

        if count % 100 == 0:
            mode = ("DMP" if cycle_step >= args_cli.rollout_start_step else "hold")
            print(f"[debug] step {count} mode={mode} dmp_t={dmp_time:.2f}/{tau:.2f} "
                  f"ee_pos_b={ee_pose_b[0, 0:3].tolist()} "
                  f"target_pos_b={ee_target_pose_b[0, 0:3].tolist()}")

        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)
        count += 1
        cycle_step += 1


def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
):
    """Read everything the OSC needs from physx + articulation buffers."""
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[
        :, arm_joint_ids, :
    ][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]

    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    ee_pose_w = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]
    root_vel_w = robot.data.root_vel_w
    relative_vel_w = ee_vel_w - root_vel_w
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]

    return (
        jacobian_b,
        mass_matrix,
        gravity,
        ee_pose_b,
        ee_vel_b,
        root_pose_w,
        ee_pose_w,
        joint_pos,
        joint_vel,
    )


def main():
    # ----- Build the DMP first (cheap), so config errors surface before we
    # stand up the whole scene. The DMP runs on the SAME device as the sim
    # so the per-step target tensor needs no cross-device copies. -----
    cfg = load_config(args_cli.config)
    _bootstrap_paths(cfg["methods_dmp"])
    torch.manual_seed(cfg["seed"])
    dmp_device = torch.device(args_cli.device)
    print(f"[run] config={args_cli.config} dmp_device={dmp_device} seed={cfg['seed']}")
    primed = build_dmp_from_config(cfg, args_cli.task_params, dmp_device)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 1.0], [0.5, 0.0, 0.4])

    scene_cfg = InsertionSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete.")
    run_simulator(sim, scene, primed)


if __name__ == "__main__":
    main()
    simulation_app.close()
