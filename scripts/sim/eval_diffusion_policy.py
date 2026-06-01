"""Closed-loop sim test of a trained diffusion policy (cluttered insertion).

Run by the USER via isaaclab.sh (the agent can't render). Per episode:

  reset + randomize layout → servo-grasp peg_0 → DIFFUSION-POLICY insertion
  (closed loop) → release → measure if the peg is in hole_0.

The grasp uses simple servo moves (no DMP) since the policy's job starts
post-grasp. The policy then drives the base-frame TCP position each control
tick; the down-quat orientation is constant. Goal conditioning is the
ground-truth insertion TCP (same as training).

Requires torchvision in the Isaac python (for the ResNet18 image encoder):
    /path/to/IsaacLab/isaaclab.sh -p -m pip install torchvision

Usage::

    /path/to/IsaacLab/isaaclab.sh -p scripts/sim/eval_diffusion_policy.py \
        --checkpoint assets/dp_models/cluttered_dp.pt --num-episodes 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Closed-loop DP insertion eval.")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-episodes", type=int, default=20)
parser.add_argument("--workspace", type=float, nargs=4,
                    default=(0.35, 0.55, -0.28, 0.28))
parser.add_argument("--min-transport", type=float, default=0.15)
parser.add_argument("--corridor-clear", type=float, default=0.05)
parser.add_argument("--max-place-tries", type=int, default=400)
parser.add_argument("--home-height", type=float, default=0.30)
parser.add_argument("--insert-depth-frac", type=float, default=0.3)
parser.add_argument("--success-depth-frac", type=float, default=0.1)
parser.add_argument("--max-policy-steps", type=int, default=220)
parser.add_argument("--no-temporal-ensemble", action="store_true",
                    help="Disable temporal ensembling (use raw 8-step chunks).")
parser.add_argument("--ddim-steps", type=int, default=8,
                    help="DDIM denoising steps per inference (temporal-ensemble mode).")
parser.add_argument("--smooth-beta", type=float, default=0.5,
                    help="EMA low-pass on the commanded TCP target (0=off, "
                         "higher=smoother but more lag). The main jerk fix.")
parser.add_argument("--max-step-m", type=float, default=0.0,
                    help="Clamp the per-tick change of the commanded TCP target "
                         "(m). 0 = off. Try ~0.01 if the peg still gets thrown.")
parser.add_argument("--cam-height", type=int, default=128)
parser.add_argument("--cam-width", type=int, default=128)
parser.add_argument("--seed", type=int, default=123)
AppLauncher.add_app_launcher_args(parser)  # provides --device
args_cli = parser.parse_args()
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv,
    subtract_frame_transforms,
)

from compliant_insertion.env.cluttered_scene_cfg import (
    ClutteredInsertionSceneCfg, PEG_SPECS, SOCKET_SPECS,
    HOLE_DEPTH_M, SOCKET_BLOCK_SIZE_XY, TABLE_TOP_Z, PEG_LENGTH,
    GRIPPER_DOWN_QUAT, attach_pegs_and_sockets,
    hole_center_w, socket_top_z, socket_component_offsets, peg_spec, socket_spec,
    finger_grip_target, hand_pos_for_grasp, hand_pos_for_peg_tip,
    insertion_success, _camera_lookat_quat,
)
from compliant_insertion.policy.inference import DiffusionPolicyRunner

TARGET_PEG_ID, TARGET_HOLE_ID = "peg_0", "hole_0"
GRIP_HOLD_TICKS, HOME_MOVE_TICKS = 60, 120
APPROACH_TICKS, DESCEND_TICKS = 80, 60
RELEASE_OPEN_TICKS, RETRACT_TICKS = 40, 60
APPROACH_CLEAR = 0.12


@configclass
class EvalSceneCfg(ClutteredInsertionSceneCfg):
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam", update_period=0.0,
        height=args_cli.cam_height, width=args_cli.cam_width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955,
                                         clipping_range=(0.01, 5.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -0.05), rot=(1.0, 0.0, 0.0, 0.0),
                                   convention="ros"))
    external_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam", update_period=0.0,
        height=args_cli.cam_height, width=args_cli.cam_width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 6.0)),
        offset=CameraCfg.OffsetCfg(pos=(1.05, 0.0, 0.65),
                                   rot=_camera_lookat_quat((1.05, 0.0, 0.65), (0.45, 0.0, 0.03)),
                                   convention="opengl"))


def update_states(scene, robot, ee_frame_idx, arm_joint_ids):
    ee_jacobi_idx = ee_frame_idx - 1
    jac_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
    mm = robot.root_physx_view.get_generalized_mass_matrices()[:, arm_joint_ids, :][:, :, arm_joint_ids]
    grav = robot.root_physx_view.get_gravity_compensation_forces()[:, arm_joint_ids]
    jac_b = jac_w.clone()
    R_b = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jac_b[:, :3, :] = torch.bmm(R_b, jac_b[:, :3, :])
    jac_b[:, 3:, :] = torch.bmm(R_b, jac_b[:, 3:, :])
    root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
    ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
    rel = robot.data.body_vel_w[:, ee_frame_idx, :] - robot.data.root_vel_w
    ee_vel_b = torch.cat([quat_apply_inverse(root_quat_w, rel[:, :3]),
                          quat_apply_inverse(root_quat_w, rel[:, 3:6])], dim=-1)
    root_pose_w = torch.cat([root_pos_w, root_quat_w], dim=-1)
    return (jac_b, mm, grav, ee_pose_b, ee_vel_b, root_pose_w,
            robot.data.joint_pos[:, arm_joint_ids], robot.data.joint_vel[:, arm_joint_ids])


def osc_step(target_pos_b, finger_cmd, ctx):
    """One control tick servoing the TCP to base-frame target_pos_b."""
    (sim, scene, robot, osc, command, down_quat_b, arm_ids, finger_ids,
     ee_idx, joint_centers, sim_dt) = ctx
    (jac_b, mm, grav, ee_pose_b, ee_vel_b, root_pose_w, jpos, jvel) = update_states(
        scene, robot, ee_idx, arm_ids)
    quat_world = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)
    command[:, 0:3] = target_pos_b
    command[:, 3:7] = quat_world
    osc.set_command(command=command, current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=None)
    eff = osc.compute(jacobian_b=jac_b, current_ee_pose_b=ee_pose_b, current_ee_vel_b=ee_vel_b,
                      current_ee_force_b=torch.zeros(scene.num_envs, 3, device=sim.device),
                      mass_matrix=mm, gravity=grav, current_joint_pos=jpos, current_joint_vel=jvel,
                      nullspace_joint_pos_target=joint_centers)
    robot.set_joint_effort_target(eff, joint_ids=arm_ids)
    robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids)
    robot.write_data_to_sim()
    sim.step(render=True)
    robot.update(sim_dt)
    scene.update(sim_dt)
    return ee_pose_b, ee_vel_b


def servo_to(target_w, n_ticks, finger_cmd, ctx):
    sim, scene, robot, osc, command, down_quat_b, arm_ids, finger_ids, ee_idx, jc, sim_dt = ctx
    tw = torch.as_tensor(target_w, device=sim.device, dtype=torch.float32)
    tw = tw.unsqueeze(0).expand(scene.num_envs, -1)
    quat_world = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)
    for _ in range(n_ticks):
        if not simulation_app.is_running():
            return
        _, _, _, ee_pose_b, _, root_pose_w, _, _ = update_states(scene, robot, ee_idx, arm_ids)
        target_pos_b, _ = subtract_frame_transforms(root_pose_w[:, :3], root_pose_w[:, 3:7], tw, quat_world)
        osc_step(target_pos_b, finger_cmd, ctx)


def peg_insertion_tip_w(pos_w, quat_w):
    half = quat_apply(quat_w, torch.tensor([0.0, 0.0, PEG_LENGTH / 2], device=pos_w.device,
                                           dtype=pos_w.dtype).expand_as(pos_w))
    a, b = pos_w + half, pos_w - half
    return torch.where(a[..., 2:3] <= b[..., 2:3], a, b)


# ─── Layout sampling (mirrors the collector). ────────────────────────────────
def _far(xy, others, mind):
    return all(np.hypot(xy[0] - o[0], xy[1] - o[1]) >= m for o, m in zip(others, mind))


def _pseg(p, a, b):
    p, a, b = map(lambda v: np.asarray(v, float), (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def sample_layout(rng):
    x0, x1, y0, y1 = args_cli.workspace
    s_sep, ps, pp = SOCKET_BLOCK_SIZE_XY + 0.04, SOCKET_BLOCK_SIZE_XY / 2 + 0.05, 0.05

    def place(existing, mind):
        for _ in range(args_cli.max_place_tries):
            xy = (rng.uniform(x0, x1), rng.uniform(y0, y1))
            if _far(xy, existing, mind):
                return xy
        return None

    for _ in range(args_cli.max_place_tries):
        sx, ok = {}, True
        for s in SOCKET_SPECS:
            p = place(list(sx.values()), [s_sep] * len(sx))
            if p is None: ok = False; break
            sx[s["id"]] = p
        if not ok: continue
        px, socks = {}, list(sx.values())
        for s in PEG_SPECS:
            p = place(socks + list(px.values()), [ps] * len(socks) + [pp] * len(px))
            if p is None: ok = False; break
            px[s["id"]] = p
        if not ok: continue
        tp, th = px[TARGET_PEG_ID], sx[TARGET_HOLE_ID]
        if np.hypot(tp[0] - th[0], tp[1] - th[1]) < args_cli.min_transport: continue
        if args_cli.corridor_clear > 0:
            if not all(_pseg(px[k], tp, th) >= args_cli.corridor_clear
                       for k in px if k != TARGET_PEG_ID): continue
            if not all(_pseg(sx[k], tp, th) >= args_cli.corridor_clear + SOCKET_BLOCK_SIZE_XY / 2
                       for k in sx if k != TARGET_HOLE_ID): continue
        return ({k: np.asarray(v, np.float32) for k, v in px.items()},
                {k: np.asarray(v, np.float32) for k, v in sx.items()})
    return ({s["id"]: np.asarray(peg_spec(s["id"])["xy"], np.float32) for s in PEG_SPECS},
            {s["id"]: np.asarray(socket_spec(s["id"])["xy"], np.float32) for s in SOCKET_SPECS})


def reset_episode(scene, robot, sim_dt, device, peg_xy, socket_xy):
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                   robot.data.default_joint_vel.clone())
    robot.set_joint_effort_target(torch.zeros(scene.num_envs, robot.num_joints, device=device))
    robot.write_data_to_sim(); robot.reset()
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    zv = torch.zeros(scene.num_envs, 6, device=device)
    for pid, xy in peg_xy.items():
        pos = torch.tensor([xy[0], xy[1], TABLE_TOP_Z + PEG_LENGTH / 2], device=device).unsqueeze(0)
        scene[pid].write_root_pose_to_sim(torch.cat([pos, ident], -1))
        scene[pid].write_root_velocity_to_sim(zv)
    for hid, xy in socket_xy.items():
        for key, dx, dy, zc in socket_component_offsets(hid):
            pos = torch.tensor([xy[0] + dx, xy[1] + dy, zc], device=device).unsqueeze(0)
            scene[key].write_root_pose_to_sim(torch.cat([pos, ident], -1))
            scene[key].write_root_velocity_to_sim(zv)
    scene.update(sim_dt); robot.update(sim_dt)


def main():
    device = torch.device(args_cli.device)
    rng = np.random.default_rng(args_cli.seed)
    runner = DiffusionPolicyRunner(
        args_cli.checkpoint, device=device,
        temporal_ensemble=not args_cli.no_temporal_ensemble,
        ddim_steps=args_cli.ddim_steps, smooth_beta=args_cli.smooth_beta,
        max_step_m=args_cli.max_step_m)
    print(f"[eval] loaded policy from {args_cli.checkpoint} "
          f"(temporal_ensemble={not args_cli.no_temporal_ensemble})")

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 1.0], [0.5, 0.0, 0.4])
    scene_cfg = EvalSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.workbench_camera.update_period = 1.0e9
    attach_pegs_and_sockets(scene_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    wrist_cam: Camera = scene["wrist_cam"]
    external_cam: Camera = scene["external_cam"]
    finger_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    arm_ids = robot.find_joints(["panda_joint.*"])[0]
    ee_idx = robot.find_bodies("panda_hand")[0][0]

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"], impedance_mode="fixed",
        motion_stiffness_task=[2000.0, 2000.0, 2000.0, 400.0, 400.0, 400.0],
        motion_damping_ratio_task=1.0, inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False, gravity_compensation=True,
        motion_control_axes_task=[1, 1, 1, 1, 1, 1], nullspace_control="position",
        nullspace_stiffness=5.0, nullspace_damping_ratio=1.0)
    osc = OperationalSpaceController(osc_cfg, num_envs=1, device=device)

    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)
    jc = robot.data.default_joint_pos[:, arm_ids].clone()
    down_quat_b = torch.tensor(GRIPPER_DOWN_QUAT, device=device)
    command = torch.zeros(scene.num_envs, osc.action_dim, device=device)
    grip = finger_grip_target(scene.num_envs, len(finger_ids), device)
    openg = torch.full((scene.num_envs, len(finger_ids)), 0.04, device=device)
    ctx = (sim, scene, robot, osc, command, down_quat_b, arm_ids, finger_ids, ee_idx, jc, sim_dt)

    n_succ = 0
    for ep in range(args_cli.num_episodes):
        if not simulation_app.is_running():
            break
        peg_xy, socket_xy = sample_layout(rng)
        peg0, hole0 = peg_xy[TARGET_PEG_ID], socket_xy[TARGET_HOLE_ID]
        osc.reset(); reset_episode(scene, robot, sim_dt, device, peg_xy, socket_xy)

        wx0, wx1, wy0, wy1 = args_cli.workspace
        servo_to([(wx0 + wx1) / 2, (wy0 + wy1) / 2, args_cli.home_height],
                 HOME_MOVE_TICKS, openg, ctx)

        peg_center = torch.tensor([peg0[0], peg0[1], TABLE_TOP_Z + PEG_LENGTH / 2], device=device)
        grasp_tcp = hand_pos_for_grasp(peg_center.unsqueeze(0), down_quat_b.unsqueeze(0))[0].cpu().numpy()
        approach = grasp_tcp.copy(); approach[2] += APPROACH_CLEAR
        servo_to(approach, APPROACH_TICKS, openg, ctx)
        servo_to(grasp_tcp, DESCEND_TICKS, openg, ctx)
        servo_to(grasp_tcp, GRIP_HOLD_TICKS, grip, ctx)

        # Goal conditioning (base-frame insertion TCP), same as training.
        rim_z = socket_top_z(TARGET_HOLE_ID)
        insert_tip = torch.tensor([hole0[0], hole0[1], rim_z - args_cli.insert_depth_frac * HOLE_DEPTH_M],
                                  device=device)
        insertion_tcp_w = hand_pos_for_peg_tip(insert_tip.unsqueeze(0), down_quat_b.unsqueeze(0))[0]
        rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
        gpos_b, gquat_b = subtract_frame_transforms(rp, rq, insertion_tcp_w.unsqueeze(0), down_quat_b.unsqueeze(0))
        goal_b = torch.cat([gpos_b, gquat_b], -1)[0].cpu().numpy().astype(np.float32)

        # ── Closed-loop diffusion-policy insertion. ──
        runner.reset()
        for _ in range(args_cli.max_policy_steps):
            if not simulation_app.is_running():
                break
            (_, _, _, ee_pose_b, ee_vel_b, _, _, _) = update_states(scene, robot, ee_idx, arm_ids)
            gripper = robot.data.joint_pos[:, finger_ids][0]
            lowdim = np.concatenate([ee_pose_b[0].cpu().numpy(), ee_vel_b[0].cpu().numpy(),
                                     gripper.cpu().numpy(), goal_b]).astype(np.float32)
            wrist = wrist_cam.data.output["rgb"][0]
            ext = external_cam.data.output["rgb"][0]
            act_pos_b = runner.act(wrist, ext, lowdim)        # [3] base-frame target
            target_pos_b = torch.tensor(act_pos_b, device=device).unsqueeze(0)
            osc_step(target_pos_b, grip, ctx)

        # ── Release + retract + score on the settled peg. ──
        servo_to(insertion_tcp_w.cpu().numpy(), RELEASE_OPEN_TICKS, openg, ctx)
        retract = insertion_tcp_w.cpu().numpy().copy(); retract[2] += 0.12
        servo_to(retract, RETRACT_TICKS, openg, ctx)

        peg_obj = scene[TARGET_PEG_ID]
        tip = peg_insertion_tip_w(peg_obj.data.root_pos_w, peg_obj.data.root_quat_w)
        hb = (float(hole0[0]), float(hole0[1]), float(hole_center_w(TARGET_HOLE_ID)[2]))
        succ, lat, depth = insertion_success(
            tip, hole_center_w=hb, hole_radius=float(socket_spec(TARGET_HOLE_ID)["hole_radius"]),
            rim_top_z=socket_top_z(TARGET_HOLE_ID), depth_threshold=args_cli.success_depth_frac)
        ok = bool(succ[0].item())
        n_succ += ok
        print(f"[eval] ep {ep:03d} success={int(ok)} lateral={lat[0].item()*1000:.1f}mm "
              f"depth={depth[0].item():+.2f} running={n_succ}/{ep+1}")

    print(f"[eval] DONE: {n_succ}/{args_cli.num_episodes} = "
          f"{n_succ/max(1,args_cli.num_episodes)*100:.0f}% success")


if __name__ == "__main__":
    main()
    simulation_app.close()
