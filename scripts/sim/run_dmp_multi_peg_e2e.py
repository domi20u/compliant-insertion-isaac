"""End-to-end FOUR-peg / FOUR-hole insertion under OSC with step-wise DMPs.

This is the multi-task generalisation of ``run_dmp_in_sim_e2e.py``. A SINGLE
DMP is kept alive for the whole run; each leg only swaps the DMP's ``x_0``,
``x_goal`` and forcing ``weights`` (the weights are selected by a task-param
pair, exactly the way the endpoints are swapped). The sequence is:

    1. First approach (task params 0.3, 0.8): TCP travels from the start pose
       to peg 0, where the gripper grasps it.
    2. Transport (task params 0.3, 0.5): TCP carries the grasped peg from the
       peg location to the matching hole, then the insertion is attempted.
    3. Release the grasp, then move DIRECTLY to the next peg
       (task params 0.3, 0.1) and repeat from the grasp.

This repeats until all four pegs have been attempted. The four holes have
different clearances (10 mm, 5 mm, 1 mm, 0.1 mm); peg i goes into hole i.

Key differences from the single-peg script
-------------------------------------------
* ONE shared DMP, weights swapped per leg via ``reset_step(weights)`` — the
  same call pattern the single-peg runner used, just with the weight tensor
  chosen per situation.
* The DMP is run in **TCP-world** coordinates and the TCP (not the peg tip) is
  what the OSC tracks. Only a single constant offset converts TCP <-> hand.
* Start/goal of every DMP come from the **ground-truth** poses read live from
  the sim: the peg's actual body pose (for the grasp goal) and the hole's known
  centre (for the insertion goal). This fixes the original endpoint bug, where
  the DMP start sat above the real EE and the goal sat below the socket:
    - DMP start  = the **current TCP** position when the phase begins, so the
      trajectory starts exactly where the end-effector already is.
    - Grasp goal = TCP that places the closed fingers at the peg's grip point
      (``hand_pos_for_grasp`` + the hand->TCP offset), i.e. a real reachable
      grasp, not a tip-space point.
    - Insertion goal = TCP computed via ``tcp_goal_for_hole_seat`` so the peg
      TIP rests at the rim top rather than being driven through the block.
* Per-task params are fed to the policy that primes each DMP, so the three
  legs use the three (P1, P2) pairs requested.
* Plotting is limited to the in-sim markers/trail. No matplotlib figure and no
  unused CLI knobs.

Usage::

    /path/to/IsaacLab/isaaclab.sh -p scripts/sim/run_dmp_multi_peg_e2e.py \
        --config configs/insertion_traj.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ─── Step 1: launch Isaac Sim BEFORE any isaaclab.* imports. ─────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Four-peg/four-hole step-wise DMP execution under OSC (TCP control).")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs. Keep at 1 for this script.")
parser.add_argument("--config", type=Path, required=True,
                    help="YAML config (same schema as gen_insertion_trajectory.py).")
parser.add_argument("--playback-speed", type=float, default=1.0,
                    help="Multiplier on how fast the DMP clock advances per "
                         "sim tick. 1.0 runs each DMP at its native duration.")
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
from isaaclab.assets import Articulation
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

from compliant_insertion.env.scene_multi_peg_cfg import (
    InsertionSceneCfg,
    NUM_TASKS,
    SOCKET_TOP_Z,
    PEG_LENGTH,
    GRIPPER_DOWN_QUAT,
    PEG_PLACE_Z,
    HOLE_CENTERS_XY, HOLE_CENTERS_TOP, HOLE_CLEARANCES,
    insertion_success,
    peg_tip_from_body,
    hand_pos_for_grasp,
    hand_pos_for_tcp, tcp_from_hand, tcp_goal_for_hole_seat,
    finger_grip_target,
)


# ─── Phase geometry. ─────────────────────────────────────────────────────────
APPROACH_CLEARANCE = 0.12     # hover height above a grasp/seat point
DMP_TIME_EXTENSION_FACTOR = 1.05  # DMPs run until their internal clock reaches tau * this factor. We may want

# DMP task-param pairs. The forcing weights produced by each pair are what we
# swap into the single shared DMP per leg.
#   - First approach of the whole run (home -> peg 0): (0.3, 0.8).
#   - Transport peg -> hole: (0.3, 0.5).
#   - After releasing, moving DIRECTLY to the next peg: (0.3, 0.1).
TASK_PARAMS_FIRST_PEG = (0.3, 0.8)   # DMP A: home -> first peg
TASK_PARAMS_TO_HOLE = (0.3, 0.5)     # DMP B: peg  -> hole
TASK_PARAMS_NEXT_PEG = (0.3, 0.1)    # DMP C: released hole -> next peg


# ─── Config + DMP construction (unchanged plumbing). ─────────────────────────
REQUIRED_KEYS = (
    "methods_dmp", "model_path", "demo_path",
    "start", "goal", "n_basis", "ins_offset",
)


def load_config(path: Path) -> dict:
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
        raise SystemExit(f"[run] ins_offset must be a 2-element list, got {offs!r}")
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


def build_primed_for_params(cfg, nn_policy, task_params, device):
    """Build one policy-primed DMP for a given (P1, P2) task-param pair.

    Endpoints registered here are placeholders — the runner overrides x_0 and
    x_goal each leg from live ground-truth scene geometry.
    """
    from configs.configs import DMPConfig          # noqa: E402
    from core.dmp_wrapper import DMPWrapper        # noqa: E402
    from core.dmp_policy import build_primed_dmp    # noqa: E402

    return build_primed_dmp(
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


def build_dmp_and_weights(cfg, task_params_list, device):
    """Build ONE DMP and harvest the policy forcing weights per task-param pair.

    Mirrors the single-peg script, which kept a single ``primed.dmp`` and drove
    it with ``reset_step(weights)`` + ``step(dt)``. Here we want the same single
    DMP but several behaviours, so we prime it once per task-param pair only to
    extract each pair's weight tensor (and tau). At runtime the runner reuses
    ONE inner DMP object and just swaps in the right weights — exactly the way
    x_0 / x_goal are swapped — via ``reset_step``.

    Returns:
        dmp:              the shared inner DMPWrapper to step every tick.
        weights_by_params: {(p1, p2): weights tensor [1, D, K]}.
        tau_by_params:     {(p1, p2): float tau}.
    """
    nn_policy = torch.jit.load(str(cfg["model_path"]), map_location=device)
    nn_policy.eval()
    print(f"[run] loaded NN policy from {cfg['model_path']}")

    weights_by_params = {}
    tau_by_params = {}
    shared_dmp = None
    for tp in task_params_list:
        primed = build_primed_for_params(cfg, nn_policy, tp, device)
        key = tuple(tp)
        weights_by_params[key] = primed.weights
        tau_by_params[key] = primed.tau
        if shared_dmp is None:
            shared_dmp = primed.dmp      # keep one inner DMP; reuse for all legs
        print(f"[run] harvested weights for task_params={tp} tau={primed.tau:.3f}s")
    return shared_dmp, weights_by_params, tau_by_params


# ─── Marker helpers. ─────────────────────────────────────────────────────────
def _sphere_marker(prim_path, color, radius):
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids):
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

    return (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
            root_pose_w, ee_pose_w, joint_pos, joint_vel)


def run_simulator(sim, scene, dmp, weights_by_params, tau_by_params):
    """Sequential four-task pick-and-insert loop with step-wise DMPs.

    A SINGLE inner ``dmp`` is reused for every leg. Per leg we swap its
    ``x_0``, ``x_goal`` and forcing ``weights`` (the latter selected by the
    leg's task-param pair) and re-initialise the integrator with
    ``reset_step(weights)`` — same call pattern as the single-peg script.
    """
    robot: Articulation = scene["robot"]
    pegs = [scene[f"peg{i}"] for i in range(NUM_TASKS)]

    finger_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    ee_frame_idx = robot.find_bodies("panda_hand")[0][0]
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]

    # ----- OSC -----
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

    # ----- Markers: current EE frame, OSC goal frame, and DMP spheres. -----
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    start_marker = _sphere_marker("/Visuals/dmp_start", (0.1, 0.8, 0.1), 0.012)
    dmp_goal_marker = _sphere_marker("/Visuals/dmp_goal", (0.9, 0.1, 0.1), 0.012)
    trail_marker = _sphere_marker("/Visuals/dmp_trail", (0.2, 0.4, 0.9), 0.005)
    trail_pts: list[list[float]] = []
    TRAIL_MAX = 600
    TRAIL_EVERY = 10

    sim_dt = sim.get_physics_dt()
    robot.update(dt=sim_dt)

    joint_centers = torch.mean(
        robot.data.soft_joint_pos_limits[:, arm_joint_ids, :], dim=-1
    )

    down_quat_b = torch.tensor(GRIPPER_DOWN_QUAT, device=sim.device)

    grip_target = finger_grip_target(scene.num_envs, len(finger_ids), sim.device)
    open_target = torch.full((scene.num_envs, len(finger_ids)), 0.04, device=sim.device)

    # Reusable command / target tensors (orientation fixed down throughout).
    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:, 3:7] = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_b[:, 3:7] = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)

    # ----- Helpers that read GROUND-TRUTH geometry from the live sim. -----
    def current_tcp_w():
        """World TCP position right now (from the live hand pose)."""
        ee_pos_w = robot.data.body_pos_w[:, ee_frame_idx]
        ee_quat_w = robot.data.body_quat_w[:, ee_frame_idx]
        return tcp_from_hand(ee_pos_w, ee_quat_w)[0]      # [3]

    def grasp_tcp_goal_w(peg):
        """TCP goal that grips the peg at its ground-truth body pose.

        Read the peg's actual centre, compute the hand pose that grasps it,
        then convert that hand pose to the TCP it implies. Result is a real,
        reachable TCP target — no tip-space mismatch.
        """
        peg_center_w = peg.data.root_pos_w               # [1, 3]
        hand_pos = hand_pos_for_grasp(
            peg_center_w, down_quat_b.unsqueeze(0))       # [1, 3]
        return tcp_from_hand(hand_pos, down_quat_b.unsqueeze(0))[0]   # [3]

    def hover_above(tcp_xyz):
        """A hover TCP point straight above a given TCP target."""
        out = tcp_xyz.clone()
        out[2] += APPROACH_CLEARANCE
        return out

    # ───────────────────────────────────────────────────────────────────────
    # Phase machine.
    #
    # For each task i we run, in order:
    #   APPROACH_PEG   : DMP A  -> hover above peg i                 (open)
    #                    weights: (0.3,0.8) for the first peg, else (0.3,0.1)
    #   DESCEND_PEG    : straight down hover -> grasp TCP            (open)
    #   GRASP          : hold, close fingers                         (grip)
    #   LIFT           : straight up grasp -> hover above peg        (grip)
    #   TO_HOLE        : DMP B  hover above peg -> hover above hole  (grip)
    #   INSERT         : DMP B-seat hover above hole -> seat TCP     (grip)
    #   RELEASE        : hold at seat, open fingers                  (open)
    # then start DMP A for peg i+1 directly with weights (0.3,0.1). After the
    # last peg's RELEASE, hold the seat pose (DONE).
    # ───────────────────────────────────────────────────────────────────────
    #INITIALIZE = "INITIALIZE"
    APPROACH_DMP = "APPROACH_PEG"
    DESCEND = "DESCEND_PEG"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TO_HOLE_DMP = "TO_HOLE"
    INSERT = "INSERT"
    RELEASE = "RELEASE"
    DONE = "DONE"

    # Step budgets for the non-DMP (straight-line / settle) phases.
    INITIALIZE_STEPS = 120
    DESCEND_STEPS = 80
    GRASP_STEPS = 20
    LIFT_STEPS = 80
    RELEASE_STEPS = 80

    task_idx = 0
    phase = APPROACH_DMP

    # DMP runtime state for the current leg.
    cur_tau = tau_by_params[tuple(TASK_PARAMS_FIRST_PEG)]
    dmp_time = 0.0
    phase_step = 0
    last_target_tcp = current_tcp_w().clone()

    def start_dmp(task_params, start_tcp_w, goal_tcp_w):
        """Re-init the SHARED DMP for a leg: swap weights, x_0, x_goal.

        Keeps one DMP object alive across all legs and only changes what makes
        a leg different — its endpoints and (via the task-param pair) its
        forcing weights — then resets the integrator. Same call pattern as the
        single-peg runner's ``dmp.reset_step(weights)``.
        """
        nonlocal dmp_time, last_target_tcp, cur_tau
        key = tuple(task_params)
        weights = weights_by_params[key]
        cur_tau = tau_by_params[key]
        dmp.x_0 = start_tcp_w.clone()
        dmp.x_goal = goal_tcp_w.clone()
        dmp.reset_step(weights)            # load this leg's weights + reset clock
        dmp_time = 0.0
        last_target_tcp = start_tcp_w.clone()
        # Place the static start/goal spheres for this leg.
        start_marker.visualize(translations=start_tcp_w.unsqueeze(0))
        dmp_goal_marker.visualize(translations=goal_tcp_w.unsqueeze(0))
        trail_pts.clear()
        print(f"[dmp] task={task_idx} params={task_params} "
              f"x0={start_tcp_w.tolist()} goal={goal_tcp_w.tolist()} "
              f"tau={cur_tau:.2f}")

    def step_active_dmp():
        """Advance the shared DMP one tick; return its TCP target [3]."""
        nonlocal dmp_time, last_target_tcp
        if dmp_time < DMP_TIME_EXTENSION_FACTOR * cur_tau:
            step_dt = sim_dt * args_cli.playback_speed
            tcp_pos, _ = dmp.step(step_dt)         # [1, 3] TCP target
            last_target_tcp = tcp_pos.to(sim.device)[0]
            dmp_time += step_dt
            print(f"[dmp] stepping with dt={step_dt:.3f}s time={dmp_time:.3f}s, cur_tau={cur_tau:.3f}s")
        return last_target_tcp

    def dmp_finished():
        return dmp_time >= DMP_TIME_EXTENSION_FACTOR * cur_tau

    # Prime the first leg: home -> hover above peg 0 with weights (0.3, 0.8).
    sim.step(render=False)
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_effort_target(zero_joint_efforts)
    robot.write_data_to_sim()
    robot.reset()
    robot.update(sim_dt)
    scene.update(sim_dt)
    peg0_grasp = grasp_tcp_goal_w(pegs[0])
    start_dmp(TASK_PARAMS_FIRST_PEG, current_tcp_w(), grasp_tcp_goal_w(pegs[0]))
    is_reset_step = True #start with one reset

    count = 0
    while simulation_app.is_running():
        # ----- Select TCP target + finger command for the current phase. -----
        if phase == APPROACH_DMP:
            finger_cmd = open_target
            target_tcp_w = step_active_dmp()
            if dmp_finished():
                phase, phase_step = GRASP, 0


        elif phase == GRASP:
            finger_cmd = grip_target
            target_tcp_w = grasp_tcp_goal_w(pegs[task_idx])
            print(f'target_tcp_w: {target_tcp_w}')
            phase_step += 1
            if phase_step >= GRASP_STEPS:
                # Continue DMP B into the seating descent: hover -> seat TCP.
                hole_seat = torch.tensor(
                    tcp_goal_for_hole_seat(task_idx), device=sim.device)
                start_dmp(TASK_PARAMS_TO_HOLE, current_tcp_w(), hole_seat)
                phase, phase_step = TO_HOLE_DMP, 0


        elif phase == TO_HOLE_DMP:
            finger_cmd = grip_target
            target_tcp_w = step_active_dmp()
            if dmp_finished():
                phase, phase_step = RELEASE, 0

        elif phase == RELEASE:
            finger_cmd = open_target               # let go of the peg
            target_tcp_w = last_target_tcp         # hold at the seat TCP
            phase_step += 1
            if phase_step >= RELEASE_STEPS:
                task_idx += 1
                if task_idx >= NUM_TASKS:
                    phase = DONE
                    print("[run] all four insertions attempted.")
                else:
                    # Move DIRECTLY to the next peg with weights (0.3, 0.1):
                    # DMP A from the released seat pose to a hover above peg i+1.
                    nxt = grasp_tcp_goal_w(pegs[task_idx])
                    start_dmp(TASK_PARAMS_NEXT_PEG, current_tcp_w(),
                              nxt)
                    phase, phase_step = APPROACH_DMP, 0

        else:  # DONE — hold the last seat pose.
            finger_cmd = open_target
            target_tcp_w = last_target_tcp

        # ----- Read states. -----
        (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
         root_pose_w, ee_pose_w, joint_pos, joint_vel) = update_states(
            sim, scene, robot, ee_frame_idx, arm_joint_ids)

        # Convert the TCP-world target to a HAND target, then to base frame.
        hand_target_w = hand_pos_for_tcp(
            target_tcp_w.unsqueeze(0), down_quat_b.unsqueeze(0))   # [1, 3]
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            hand_target_w.expand(scene.num_envs, -1),
            down_quat_b.unsqueeze(0).expand(scene.num_envs, -1),
        )
        ee_target_pose_b[:, 0:3] = target_pos_b
        command[:, 0:3] = ee_target_pose_b[:, 0:3]

        # ----- Success readout against the CURRENT task's hole. -----
        cur_hole = min(task_idx, NUM_TASKS - 1)
        tip_pos_w = peg_tip_from_body(
            pegs[cur_hole].data.root_pos_w, pegs[cur_hole].data.root_quat_w)
        success, lateral_err, depth_frac = insertion_success(tip_pos_w, cur_hole)

        # ----- Live DMP trail (TCP path). -----
        if phase in (APPROACH_DMP, TO_HOLE_DMP, INSERT) and (
                count % TRAIL_EVERY == 0):
            trail_pts.append(current_tcp_w().detach().cpu().tolist())
            if len(trail_pts) > TRAIL_MAX:
                trail_pts.pop(0)
        if trail_pts:
            trail_marker.visualize(
                translations=torch.tensor(trail_pts, device=sim.device))

        if count % 50 == 0:
            print(f"[task {cur_hole}|{phase}] clr={HOLE_CLEARANCES[cur_hole]*1000:.1f}mm "
                  f"tip_z={tip_pos_w[0, 2]:.3f} "
                  f"lateral={lateral_err[0]*1000:.1f}mm "
                  f"depth_frac={depth_frac[0]:+.2f} success={bool(success[0])}")

        # ----- OSC compute + write. -----
        if is_reset_step:
            osc.reset()
            is_reset_step = False

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
        robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids)
        robot.write_data_to_sim()

        # ----- Marker frames (current EE + OSC goal). -----
        ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7],
        )
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ee_target_pos_w, ee_target_quat_w)

        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)
        count += 1


def main():
    cfg = load_config(args_cli.config)
    _bootstrap_paths(cfg["methods_dmp"])
    torch.manual_seed(cfg["seed"])
    dmp_device = torch.device(args_cli.device)
    print(f"[run] config={args_cli.config} dmp_device={dmp_device} seed={cfg['seed']}")

    # Build ONE DMP and harvest the forcing weights for each task-param pair
    # used across the legs. At runtime the single DMP is reused; only its
    # weights (and x_0/x_goal) are swapped per leg.
    dmp, weights_by_params, tau_by_params = build_dmp_and_weights(
        cfg,
        [TASK_PARAMS_FIRST_PEG, TASK_PARAMS_TO_HOLE, TASK_PARAMS_NEXT_PEG],
        dmp_device,
    )

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.4, 1.4, 1.1], [0.45, 0.0, 0.4])

    scene_cfg = InsertionSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete.")
    run_simulator(sim, scene, dmp, weights_by_params, tau_by_params)


if __name__ == "__main__":
    main()
    simulation_app.close()
