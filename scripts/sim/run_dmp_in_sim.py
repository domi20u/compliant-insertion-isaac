"""Pass-2: Franka tracks a pre-generated DMP trajectory with OSC.

Hybrid execution model:

  1. From step 0 to ``--rollout-start-step`` (default 200), the arm holds
     a static hover pose above the socket — same as pass-1. This lets you
     visually confirm OSC is healthy before the trajectory kicks in.
  2. At ``--rollout-start-step``, playback of the DMP trajectory begins.
     A "DMP clock" advances by ``sim_dt`` each loop iteration; the current
     target waypoint is the trajectory sample whose timestamp is closest
     to (but not past) that clock. If the trajectory's native ``times``
     array runs out, we clamp to the final waypoint (== hold final pose).
  3. On periodic reset (every ``--reset-period`` sim steps), joints go
     back to default, the DMP clock resets to zero, and the cycle repeats.

If ``--trajectory-file`` is not given, the script falls back to pure
pass-1 behaviour: static hold forever.

Generate the trajectory first with ``scripts/sim/gen_insertion_trajectory.py``::

    python scripts/sim/gen_insertion_trajectory.py \
        --start 0.4 0.0 0.55 --goal 0.4 0.30 0.55 \
        --out assets/trajectories/insertion_sim_rollout.npy

Then run::

    /path/to/IsaacLab/isaaclab.sh -p scripts/sim/run_dmp_in_sim.py \
        --trajectory-file assets/trajectories/insertion_sim_rollout.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ─── Step 1: launch Isaac Sim BEFORE any isaaclab.* imports. ─────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DMP-in-sim pass-2: trajectory rollout.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs. Keep at 1 for pass-2.")
parser.add_argument("--trajectory-file", type=Path, default=None,
                    help=".npy file with shape [T, 3] (xyz in base frame). "
                         "If a .npz with the same stem and a 'times' array "
                         "exists alongside, its timestamps are used for "
                         "playback; otherwise we assume uniform spacing "
                         "at sim_dt. Omit for static-hold-only behaviour.")
parser.add_argument("--rollout-start-step", type=int, default=200,
                    help="Sim step at which DMP playback begins within each "
                         "reset cycle. Before this, arm holds the static "
                         "hover pose.")
parser.add_argument("--reset-period", type=int, default=800,
                    help="Sim steps between joint resets. Should be larger "
                         "than rollout_start_step + trajectory duration so "
                         "the arm gets to finish the motion before resetting.")
parser.add_argument("--playback-speed", type=float, default=1.0,
                    help="Multiplier on the DMP clock rate. 1.0 plays the "
                         "trajectory at its native duration; 2.0 plays it "
                         "in half the time. Implemented by scaling how fast "
                         "the DMP clock advances per sim step, with linear "
                         "interpolation between waypoints — so the command "
                         "stream stays smooth at sim_dt regardless of the "
                         "DMP's native sampling density. Generate the DMP "
                         "at high resolution (small tau or large n_steps in "
                         "the wrapper) and use this flag to speed it up at "
                         "playback time; don't try to speed it up by "
                         "down-sampling at generation, which produces "
                         "staircase commands.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Step 2: everything else can now be imported. ────────────────────────────
import numpy as np
import torch

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


# ─── Static-hold target (pass-1 fallback). ───────────────────────────────────
HOVER_OFFSET = 0.10
HAND_TARGET_Z = SOCKET_TOP_Z + HOVER_OFFSET + PEG_LENGTH
STATIC_TARGET_POS = (SOCKET_X, SOCKET_Y, HAND_TARGET_Z)


# ─── Trajectory loading. ─────────────────────────────────────────────────────
def load_trajectory(path: Path | None, sim_dt: float, device: torch.device):
    """Load a [T, 3] base-frame trajectory and matching timestamps.

    Returns (positions [T, 3] tensor, times [T] tensor, total_duration_s)
    or (None, None, 0.0) if path is None.

    Resolution order for timestamps: a sibling .npz with a 'times' field
    wins; otherwise we synthesise uniform t = i * sim_dt. The .npz lookup
    matches the metadata bundle saved by gen_insertion_trajectory.py.
    """
    if path is None:
        return None, None, 0.0

    pos = np.load(path).astype(np.float32)
    pos = np.squeeze(pos)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"trajectory file {path} has shape {pos.shape} after "
                         f"squeeze, expected [T, 3]")

    sibling_npz = path.with_suffix(".npz")
    if sibling_npz.exists():
        meta = np.load(sibling_npz)
        if "times" in meta.files:
            times = np.asarray(meta["times"], dtype=np.float32).squeeze()
            # If the wrapper stored a batched [N, T] times array, squeeze to
            # [T] when N=1; otherwise take the first batch row.
            if times.ndim > 1:
                print(f"[traj] times has shape {times.shape}, taking first row")
                times = times[0]
            times = np.ascontiguousarray(times, dtype=np.float32)
            print(f"[traj] using timestamps from {sibling_npz.name} "
                  f"(shape {times.shape})")
        else:
            times = np.arange(pos.shape[0], dtype=np.float32) * sim_dt
            print(f"[traj] .npz has no 'times' field — synth uniform at sim_dt={sim_dt}")
    else:
        times = np.arange(pos.shape[0], dtype=np.float32) * sim_dt
        print(f"[traj] no .npz alongside — synth uniform at sim_dt={sim_dt}")

    if times.shape[0] != pos.shape[0]:
        raise ValueError(f"times has {times.shape[0]} samples, pos has "
                         f"{pos.shape[0]} — shapes must match.")

    pos_t = torch.tensor(pos, device=device)
    times_t = torch.tensor(times, device=device)
    duration = float(times[-1] - times[0])
    print(f"[traj] loaded {pos.shape[0]} samples, "
          f"duration={duration:.3f}s, "
          f"start={pos[0].tolist()}, end={pos[-1].tolist()}")
    return pos_t, times_t, duration


def sample_trajectory(pos: torch.Tensor, times: torch.Tensor, t: float) -> torch.Tensor:
    """Return the trajectory waypoint at clock time ``t`` via linear interp.

    The waypoint stream we hand to OSC must be smooth at the *sim* rate, not
    just at the DMP's native sampling rate. A zero-order hold (pick the most
    recent waypoint and stick with it) produces a staircase command: the arm
    races to each DMP sample, then idles waiting for the clock to advance to
    the next one. The faster the playback, the more visible the staircase.

    Linear interpolation between adjacent DMP samples gives a continuous
    target that the OSC tracks smoothly regardless of how the DMP itself
    was sampled. For OSC tracking purposes the linear approximation is well
    within the controller's bandwidth — the position error and velocity
    feedforward stay sensible.
    """
    if t <= float(times[0]):
        return pos[0]
    if t >= float(times[-1]):
        return pos[-1]
    # Find the index of the largest timestamp <= t (the "left" waypoint).
    t_tensor = torch.tensor(t, device=times.device, dtype=times.dtype)
    idx_right = int(torch.searchsorted(times, t_tensor).item())
    idx_left = max(0, idx_right - 1)
    idx_right = min(idx_right, pos.shape[0] - 1)

    t_left = float(times[idx_left])
    t_right = float(times[idx_right])
    if t_right == t_left:
        return pos[idx_left]

    alpha = (t - t_left) / (t_right - t_left)
    return (1.0 - alpha) * pos[idx_left] + alpha * pos[idx_right]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    robot: Articulation = scene["robot"]
    peg: RigidObject = scene["peg"]
    ee_frame_name = "panda_hand"
    arm_joint_names = ["panda_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    

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

    # ----- Load the DMP trajectory (or None for static-only mode). -----
    traj_pos, traj_times, traj_duration = load_trajectory(
        args_cli.trajectory_file, sim_dt, torch.device(sim.device))
    has_trajectory = traj_pos is not None

    if has_trajectory:
        # Sanity check: does the trajectory start near the static hover
        # position? Big mismatch means a discontinuity at the transition.
        start_err = float(torch.norm(
            traj_pos[0] - torch.tensor(STATIC_TARGET_POS, device=sim.device)))
        if start_err > 0.05:
            print(f"[warn] trajectory start is {start_err*100:.1f} cm from the "
                  f"static-hold target — expect a jerk at step "
                  f"{args_cli.rollout_start_step}. Consider rerunning the "
                  f"generator with --start matching the hover position.")
        print(f"[info] hybrid mode: hold static until step "
              f"{args_cli.rollout_start_step}, then play DMP for "
              f"{traj_duration:.2f}s, then hold final waypoint.")
    else:
        print("[info] no --trajectory-file given — static-hold-only mode.")

    # ----- Reusable command tensor. We'll mutate command[:, :3] each step
    # to point at the current waypoint; orientation stays at home_quat_b. -----
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:, 3:7] = home_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_b[:, 3:7] = home_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)

    # ----- State for the DMP clock. -----
    dmp_clock = 0.0       # seconds since the rollout started within this cycle
    cycle_step = 0        # steps since last reset

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
            #hand_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
            #hand_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
            #peg_pos_w, peg_quat_w = peg_pose_from_hand(hand_pos_w, hand_quat_w)
            #peg_pose_w = torch.cat([peg_pos_w, peg_quat_w], dim=-1)
            #peg.write_root_pose_to_sim(peg_pose_w)
            cycle_step = 0
            dmp_clock = 0.0
            print(f"[run] reset at step {count}")

        # ----- Compute the current target position. -----
        if has_trajectory and cycle_step >= args_cli.rollout_start_step:
            # DMP playback mode. The clock advances by sim_dt * speed so
            # that speed=2 finishes the trajectory in half the wall time.
            # Smoothness at sim rate is preserved by sample_trajectory's
            # linear interpolation, independent of the DMP's native rate.
            target_xyz = sample_trajectory(traj_pos, traj_times, dmp_clock)
            dmp_clock += sim_dt * args_cli.playback_speed
        else:
            # Static hold (pre-rollout, or no trajectory at all).
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
        # On reset steps, re-prime the OSC with the current EE pose. Other
        # steps just stream the new command — OSC.set_command is cheap.
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

        # ----- Markers. World-frame target each step so it tracks the
        # streaming DMP waypoint. -----
        ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7],
        )
        ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

        if count % 100 == 0:
            mode = ("DMP" if (has_trajectory and cycle_step >= args_cli.rollout_start_step)
                    else "hold")
            print(f"[debug] step {count} mode={mode} "
                  f"ee_pos_b={ee_pose_b[0, 0:3].tolist()} "
                  f"target_pos_b={ee_target_pose_b[0, 0:3].tolist()}")

        sim.step(render=True)
        robot.update(sim_dt)
        #hand_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
        #hand_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
        #peg_pos_w, peg_quat_w = peg_pose_from_hand(hand_pos_w, hand_quat_w)
        #peg_pose_w = torch.cat([peg_pos_w, peg_quat_w], dim=-1)
        #peg.write_root_pose_to_sim(peg_pose_w)
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
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 1.0], [0.5, 0.0, 0.4])

    scene_cfg = InsertionSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
