"""Pass-1: Franka holds a static pose above the socket using OSC.

Structured to mirror the official tutorial at
``scripts/tutorials/05_controllers/run_osc.py``. The only deltas vs that
tutorial:

  - scene is the insertion scene (socket + peg + Franka on a stand), defined
    in :mod:`compliant_insertion.env.scene_cfg`
  - OSC runs in pose-only mode (no wrench, no variable kp). The tilted-wall
    contact-force example is the wrong shape for a peg-in-hole pass-1 sanity
    check; we just want the EE to hold a pose.
  - one target instead of a cycling set: hover above the socket entrance,
    keeping the home-pose orientation.

Run from the repo root::

    /path/to/IsaacLab/isaaclab.sh -p scripts/sim/run_dmp_in_sim.py

What you should see: the Franka holds its gripper directly above the blue
socket. Every 500 sim steps, joints reset to default; the arm re-acquires
the target. If the arm jitters, lower ``motion_stiffness_task``. If it sags,
verify ``disable_gravity=True`` made it into the spawned robot.
"""
from __future__ import annotations

import argparse

# ─── Step 1: launch Isaac Sim BEFORE any isaaclab.* imports. ─────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DMP-in-sim pass-1: static hold.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs. Keep at 1 for pass-1.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Step 2: everything else can now be imported. ────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
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
)


# ─── Target definition. ──────────────────────────────────────────────────────
# Hover the peg TIP 10 cm above the socket entrance. OSC controls the hand
# frame, so the hand has to sit (PEG_LENGTH + HOVER_OFFSET) above the socket
# top with the peg pointing down. Orientation is filled in at runtime from
# the measured home-pose hand quaternion — see the comment in run_simulator.
HOVER_OFFSET = 0.10
HAND_TARGET_Z = SOCKET_TOP_Z + HOVER_OFFSET + PEG_LENGTH
STATIC_TARGET_POS = (SOCKET_X, SOCKET_Y, HAND_TARGET_Z)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    # ----- Scene entities and identifiers -----
    robot: Articulation = scene["robot"]

    # We control the hand (not the fingertip) because the peg will be rigidly
    # attached below the hand frame in pass-2. The tutorial uses
    # ``panda_leftfinger`` for the same reason that it doesn't matter much
    # there; here the choice has downstream consequences.
    ee_frame_name = "panda_hand"
    arm_joint_names = ["panda_joint.*"]
    ee_frame_idx = robot.find_bodies(ee_frame_name)[0][0]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]

    # ----- Build the OSC -----
    # Pose-only, fixed kp, full inertial decoupling, nullspace pulled toward
    # the center of the joint limits. Gravity compensation is OFF because the
    # scene config disables gravity on the robot (mirrors the tutorial).
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

    # ----- Sim dt -----
    sim_dt = sim.get_physics_dt()

    # ----- Prime buffers with one update before the first read. The
    # tutorial does the same: the first physx Jacobian/mass query returns
    # stale data otherwise.
    robot.update(dt=sim_dt)

    # ----- Nullspace target: center of the joint position limits. The
    # tutorial uses this and it works well as a "stay near home" prior.
    joint_centers = torch.mean(
        robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1
    )

    # ----- Read the home-pose hand quaternion. We want the target
    # orientation to MATCH the home pose, so OSC only has to translate.
    # If we hardcoded e.g. (0.707, -0.707, 0, 0), small mismatches with
    # the actual panda_hand local frame would cause the arm to rotate
    # immediately and the whole demo to look broken.
    (
        _, _, _, ee_pose_b, _, _, _, joint_pos, joint_vel,
    ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)
    home_quat_b = ee_pose_b[0, 3:7].clone()
    print(f"[info] home ee pose (base frame): "
          f"pos={ee_pose_b[0, 0:3].tolist()}, quat(wxyz)={home_quat_b.tolist()}")

    # ----- Build the absolute pose target (one fixed goal). -----
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_b[:, 0:3] = torch.tensor(STATIC_TARGET_POS, device=sim.device)
    ee_target_pose_b[:, 3:7] = home_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    # `command` is what we hand to OSC.set_command. For pose_abs with no
    # custom task frame, this is simply the target pose in the base frame.
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:, :7] = ee_target_pose_b

    # Zero-effort buffer used at reset to clear actuator state.
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)

    count = 0
    while simulation_app.is_running():

        if count % 500 == 0:
            # ----- Periodic reset: joints -> default, controller -> fresh -----
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)
            robot.write_data_to_sim()
            robot.reset()
            robot.update(sim_dt)

            # At reset, the Jacobians lag the joint state by one step. The
            # tutorial re-reads the EE pose specifically to feed set_command
            # with a current value. We do the same. We also bind ee_pose_w
            # here because the marker code below the if/else uses it on
            # every iteration — including step 0, when only this branch ran.
            (
                _, _, _, ee_pose_b, _, root_pose_w, ee_pose_w, joint_pos, joint_vel,
            ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)

            # Refresh the world-frame target for the marker.
            ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7],
            )
            ee_target_pose_w = torch.cat([ee_target_pos_w, ee_target_quat_w], dim=-1)

            # Hand the command to OSC. No task-frame transform needed because
            # we're already in the base frame and using it as the task frame.
            osc.reset()
            osc.set_command(
                command=command,
                current_ee_pose_b=ee_pose_b,
                current_task_frame_pose_b=None,
            )

            print(f"[run_dmp_in_sim] reset at step {count}, "
                  f"target_pos={STATIC_TARGET_POS}")

        else:
            # ----- Normal step: read states, compute torques, apply. -----
            (
                jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
                root_pose_w, ee_pose_w, joint_pos, joint_vel,
            ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)

            # ee_force_b is unused (no wrench target) but the API still wants it.
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

        # ----- Markers -----
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ee_target_pose_w[:, 0:3], ee_target_pose_w[:, 3:7])

        # ----- Periodic debug print -----
        if count % 100 == 0:
            print(f"[debug] step {count} "
                  f"ee_pos_b={ee_pose_b[0, 0:3].tolist()} "
                  f"target_pos_b={ee_target_pose_b[0, 0:3].tolist()}")

        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)
        count += 1


def update_states(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    ee_frame_idx: int,
    arm_joint_ids: list[int],
):
    """Read everything the OSC needs from physx + articulation buffers.

    Identical in structure to the tutorial's ``update_states`` minus the
    contact-sensor block. Returns nine tensors instead of ten.
    """
    # --- Dynamics quantities, restricted to arm DoFs. ---
    # The tutorial uses arm-only indices for the Jacobian and mass matrix
    # and it's correct: PhysX builds the operational-space quantities for
    # exactly the joints you ask for. The previous custom wrapper used the
    # full joint set and then sliced the torque vector, which is equivalent
    # only when arm and fingers are dynamically decoupled — which Franka's
    # are NOT (the finger drives have non-trivial inertia coupling through
    # the hand). Stick with arm-only.
    ee_jacobi_idx = ee_frame_idx - 1
    jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mass_matrix = robot.root_physx_view.get_generalized_mass_matrices()[
        :, arm_joint_ids, :
    ][:, :, arm_joint_ids]
    gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]

    # --- World-frame Jacobian -> base-frame Jacobian. ---
    # If the robot base is not aligned with the world (here it sits at
    # origin and rotated identity, so technically a no-op — but on a moving
    # base, or any robot mounted at an angle, this rotation is essential).
    jacobian_b = jacobian_w.clone()
    root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:, :])

    # --- EE pose in world and base frames. ---
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

    # --- EE velocity in the base frame. ---
    ee_vel_w = robot.data.body_vel_w[:, ee_frame_idx, :]
    root_vel_w = robot.data.root_vel_w
    relative_vel_w = ee_vel_w - root_vel_w
    ee_lin_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
    ee_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

    # --- Joint state. ---
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
    """Main function."""
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
