"""End-to-end runner: cluttered scene, perceived peg→hole, TCP-controlled DMP.

Motion procedure (per reset cycle, after perception):

  HOME → DMP-TO-GRASP → GRASP → DMP-TO-INSERT → RELEASE

The DMP operates directly in TOOL-CENTER-POINT (TCP) space — that is,
its ``x_0`` / ``x_goal`` are positions of the Panda's hand frame in the
world, and its per-step output is a hand-frame position that we feed
straight to the operational-space controller. This is a change from
the earlier runner, which had the DMP output peg-tip positions and
converted hand pose at every tick via ``hand_pos_for_peg_tip``. TCP
control is cleaner because:

  - The same DMP describes a HAND trajectory for both segments,
    regardless of whether the gripper is empty (home → grasp) or
    holding the peg (grasp → insertion). The peg geometry only enters
    the goal computation, never the inner integration loop.

  - The OSC and the DMP share a frame, so there's no per-step
    coordinate gymnastics in the hot loop.

  - When we eventually compare against Diffusion Policy and residual
    RL, those policies will also command TCP poses (or joint actions
    that resolve to TCP). Keeping the DMP in the same space keeps the
    comparison apples-to-apples.

The peg/tip math from ``scene_cfg`` is still used — but ONCE per
segment, to compute the TCP GOAL that places the peg correctly:

    grasp_tcp_w     = hand_pos_for_grasp(peg_center_w_perceived, down_quat)
    insertion_tcp_w = hand_pos_for_peg_tip(hole_seat_w_perceived, down_quat)

After that, the DMP integrates a TCP trajectory between those endpoints
using the policy-primed forcing weights. Rotodilatation (per the demo's
configured ``rescale``) rotates and rescales the demo shape onto each
segment's endpoints, so the two segments inherit the demo's curvature
without retraining.

Per-episode artifacts:
  - reports/cluttered/episode_NNNN.json   — perception record,
                                            per-segment trail, final
                                            lateral/depth error,
                                            overall success.
  - reports/cluttered/som_TIMESTAMP.png   — Set-of-Mark annotated
                                            image fed to the VLM
                                            (when --vlm-backend != ground_truth).

Usage::

    /path/to/IsaacLab/isaaclab.sh -p scripts/sim/cluttered_run_e2e.py \\
        --config configs/insertion_traj.yaml \\
        --task-params 0.05 0.20 \\
        --vlm-backend ollama \\
        --ollama-model qwen2.5vl:7b \\
        --report-dir reports/cluttered_v1
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ─── Step 1: launch Isaac Sim BEFORE any isaaclab.* imports. ─────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Cluttered-scene perception + TCP-controlled DMP runner.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--config", type=Path, required=True,
                    help="YAML config (same schema as the single-peg runner).")
parser.add_argument("--task-params", type=float, nargs=2, required=True,
                    metavar=("P1", "P2"))
parser.add_argument("--vlm-backend",
                    choices=["ollama", "mock", "ground_truth"],
                    default="ground_truth")
parser.add_argument("--owlv2-score-threshold", type=float, default=0.25,
                    help="OWLv2 confidence floor (only used by --detector owlv2). "
                         "Lower than the standalone-calibrated 0.30 because the "
                         "top-crop already removes the robot-base false "
                         "positives the higher floor was guarding against; the "
                         "headroom buys back recall on faint pegs.")
parser.add_argument("--owlv2-crop-top-frac", type=float, default=0.18,
                    help="Fraction of image HEIGHT cropped off the TOP before "
                         "OWLv2 detection, to hide the robot base/mount that "
                         "draws spurious 'hole' boxes. Coords are remapped back "
                         "to the full frame. Set 0.0 to disable.")
parser.add_argument("--ollama-model", default="qwen2.5vl:7b")
parser.add_argument("--ollama-host", default="http://localhost:11434")
parser.add_argument("--report-dir", type=Path, default=Path("reports/cluttered"))
parser.add_argument("--playback-speed", type=float, default=5.0,
                    help="DMP integration speedup factor. Default 5.0 — the "
                         "raw demo plays back slowly relative to the OSC's "
                         "tracking bandwidth, so we accelerate the DMP clock "
                         "to keep wall-clock-per-episode reasonable. Drop to "
                         "1.0 when debugging tracking error.")
parser.add_argument("--max-cycles", type=int, default=3)
parser.add_argument("--detector",
                    choices=["ground_truth", "grounding_dino", "owlv2",
                             "locate_anything"],
                    default="ground_truth",
                    help="Which detector backend feeds the VLM matcher. "
                         "`ground_truth` is the oracle-projection stub (no "
                         "model weights, used for DMP-side development). "
                         "`grounding_dino` is GroundingDINO-base + SAM 2. "
                         "`owlv2` is OWLv2-base + SAM 2 (small) — single "
                         "forward pass, ~0.6 GB, the light/fast default for a "
                         "16 GB card sharing the GPU with Isaac Sim. "
                         "`locate_anything` is NVIDIA's LocateAnything-3B + "
                         "SAM 2 — heavier (~7.8 GB BF16) but stronger on "
                         "small/cluttered objects where GroundingDINO's "
                         "lexical prior misses. Only relevant when "
                         "--vlm-backend != ground_truth.")
parser.add_argument("--insert-to-bottom", action="store_true",
                    help="Target the hole BOTTOM (full seating). Default is "
                         "hole TOP — the DMP drives the tip just to the rim "
                         "and the residual layer would push deeper.")
parser.add_argument("--retract-after-release", action="store_true",
                    help="Lift TCP after releasing the peg, useful for "
                         "visualization. Strictly outside the requested "
                         "5-step procedure.")
parser.add_argument("--debug-perception", action="store_true",
                    help="Verbose dump of the perception pipeline: prints "
                         "camera intrinsics/extrinsics, per-detection poses "
                         "and confidences, VLM assignment + rejection "
                         "reasons, and saves intermediate images (raw RGB, "
                         "depth viz + .npy, SOM annotated frame) plus a "
                         "perception.json under --debug-dir / "
                         "episode_NNNN/. No-op for --vlm-backend=ground_truth "
                         "beyond the printout, since there are no images.")
parser.add_argument("--debug-dir", type=Path, default=None,
                    help="Where perception debug artifacts go. Defaults to "
                         "<report-dir>/debug/. Only used when "
                         "--debug-perception is set.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras must be enabled at AppLauncher time for Isaac Lab's Camera sensor
# to produce rgb / depth tensors — otherwise camera.data.output is empty
# and the perception layer's first frame capture returns garbage. We default
# this ON because every backend except --vlm-backend=ground_truth needs the
# workbench camera. AppLauncher.add_app_launcher_args registers
# --enable_cameras as a store_true flag, so we flip the parsed namespace
# before instantiating the launcher (a user can still pass
# --enable_cameras=False on the CLI to override).
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

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
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

from compliant_insertion.env.cluttered_scene_cfg import (
    ClutteredInsertionSceneCfg,
    PEG_SPECS, SOCKET_SPECS,
    HOLE_DEPTH_M,
    attach_pegs_and_sockets,
    peg_place_pos_w, hole_top_w, hole_center_w, socket_top_z,
    PEG_LENGTH,
    GRIPPER_DOWN_QUAT,
    finger_grip_target,
    hand_pos_for_grasp, hand_pos_for_peg_tip,
    peg_tip_from_body,
    insertion_success,
    workbench_camera_K, workbench_camera_T_world_cam,
)
from compliant_insertion.perception import (
    GroundTruthPerception,
    MockVLMBackend,
    OllamaQwenBackend,
    PerceptionOutput,
    perceive_scene,
)
from compliant_insertion.perception.camera_capture import capture_frame
from compliant_insertion.perception.detection import GroundTruthDetector

# ─── Constants ───────────────────────────────────────────────────────────────
# Wall-time DMP extension factor: the DMP's nominal duration is ``tau``,
# but we let it integrate slightly past that to give the convergence
# tail enough time to settle near the goal.
DMP_TIME_EXTENSION_FACTOR = 1.05

# How many sim ticks to hold the TCP fixed while opening/closing the gripper.
# Long enough that the finger pads finish their motion and friction stabilises;
# short enough that the wall clock per episode stays reasonable.
GRIP_HOLD_TICKS = 60
RELEASE_HOLD_TICKS = 60
SETTLE_TICKS = 30        # after reset, before reading the home TCP


# ─── Per-episode report (saved to disk for the post-hoc dashboard) ───────────
@dataclass
class PerceptionRecord:
    objects: dict
    assignment: dict[str, str]
    confidence: dict[str, float]
    validated_pairs: list[tuple[str, str]]
    rejected_pairs: list[tuple[str, str, str]]
    vlm_latency_s: float
    unfilled_holes: list[str]
    ungrasped_pegs: list[str]


@dataclass
class SegmentRecord:
    """Per-DMP-segment record. Two of these per episode.

    Coordinates are TCP (hand-frame) positions in the world, NOT peg-tip
    positions, because that's what the DMP integrates here.
    """

    segment_name: str
    start_tcp_w: list[float]
    goal_tcp_w: list[float]
    tau_s: float
    n_ticks: int
    tcp_trail_w: list[list[float]] = field(default_factory=list)
    final_tcp_w: list[float] = field(default_factory=list)
    final_tcp_err_m: float = float("nan")
    # The PEG-side state at segment end, for sanity checking the goal math.
    final_peg_tip_w: list[float] = field(default_factory=list)


@dataclass
class EpisodeReport:
    cycle_index: int
    chosen_peg: str
    chosen_hole: str
    perception: PerceptionRecord | None
    home_tcp_w: list[float]
    segments: list[SegmentRecord]
    final_lateral_err_m: float
    final_depth_frac: float
    overall_success: bool
    wall_time_s: float


# ─── Config / DMP loading ────────────────────────────────────────────────────
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
    return cfg


def _bootstrap_paths(methods_dmp: Path) -> None:
    methods_dmp = methods_dmp.resolve()
    if not (methods_dmp / "core" / "dmp_wrapper.py").exists():
        raise SystemExit(
            f"[run] {methods_dmp} doesn't look like the dmp methods dir.")
    sys.path.insert(0, str(methods_dmp))


def build_dmp_from_config(cfg, task_params, device):
    from configs.configs import DMPConfig          # noqa: E402
    from core.dmp_wrapper import DMPWrapper        # noqa: E402
    from core.dmp_policy import build_primed_dmp    # noqa: E402

    nn_policy = torch.jit.load(str(cfg["model_path"]), map_location=device)
    nn_policy.eval()
    primed = build_primed_dmp(
        DMPWrapper=DMPWrapper, DMPConfig=DMPConfig,
        nn_policy=nn_policy,
        demo_path=str(cfg["demo_path"]),
        n_basis=cfg["n_basis"],
        task_params=tuple(task_params),
        ins_offset=tuple(cfg["ins_offset"]),
        # Demo start/goal from the YAML are kept as the policy's prior; the
        # runtime per-segment endpoints are written into x_0/x_goal below.
        start=np.asarray(cfg["start"], dtype=np.float32),
        goal=np.asarray(cfg["goal"], dtype=np.float32),
        rescale=cfg["rescale"],
        device=device,
    )
    print(f"[run] DMP primed: tau={primed.tau:.3f}s")
    return primed


# ─── Perception backend factory ──────────────────────────────────────────────
def build_perception_backend(args, scene, debug_dir: Path | None):
    """Return (callable, label_for_report) for the chosen backend.

    The returned callable has signature ``fn(cycle_index: int) -> PerceptionOutput``.
    When ``debug_dir`` is not None (i.e. --debug-perception was set), the
    callable also stashes the raw capture on a ``last_raw_capture`` attribute
    so the caller can dump intermediate images, and SOM frames are written
    into ``debug_dir / episode_NNNN/som.png`` instead of a timestamped name
    in the report dir.
    """
    debug_on = debug_dir is not None

    if args.vlm_backend == "ground_truth":
        peg_specs_oracle = [
            (s["id"], peg_place_pos_w(s["id"]), s["diameter"], PEG_LENGTH, s["color"])
            for s in PEG_SPECS
        ]
        hole_specs_oracle = [
            (s["id"], hole_top_w(s["id"]), 2 * s["hole_radius"], s["color"])
            for s in SOCKET_SPECS
        ]
        gt = GroundTruthPerception.from_scene(peg_specs_oracle, hole_specs_oracle)

        def _run_perception_gt(cycle_index: int) -> PerceptionOutput:
            # GT backend doesn't touch the camera, so there's no raw capture
            # to stash. The debug printer still has full access to perc.
            _run_perception_gt.last_raw_capture = None
            return gt()

        _run_perception_gt.last_raw_capture = None
        return _run_perception_gt, "ground_truth"

    camera = scene["workbench_camera"]
    # Static-camera intrinsics / extrinsics, computed analytically from config.
    # camera.data.pos_w / quat are unpopulated (origin / NaN) at perception
    # time for this fixed-offset camera, so we cannot read the pose from the
    # sensor — see camera_capture.capture_frame.
    cam_K = workbench_camera_K()
    cam_T_world_cam = workbench_camera_T_world_cam()
    if args.detector == "grounding_dino":
        from compliant_insertion.perception.detection import GroundedSAM2Detector
        detector = GroundedSAM2Detector()
    elif args.detector == "owlv2":
        from compliant_insertion.perception.detection import OWLv2SAM2Detector
        detector = OWLv2SAM2Detector(
            score_threshold=args.owlv2_score_threshold,
            crop_top_frac=args.owlv2_crop_top_frac,
        )
    elif args.detector == "locate_anything":
        from compliant_insertion.perception.detection import (
            LocateAnythingSAM2Detector,
        )
        detector = LocateAnythingSAM2Detector()
    else:                                # "ground_truth"
        detector = None        # built lazily once we have the camera pose

    if args.vlm_backend == "ollama":
        vlm_backend = OllamaQwenBackend(model=args.ollama_model, host=args.ollama_host)
        backend_label = f"ollama:{args.ollama_model}"
    else:
        vlm_backend = MockVLMBackend()
        backend_label = "mock"

    def _run_perception(cycle_index: int) -> PerceptionOutput:
        rgb, depth, K, T_world_cam = capture_frame(
            camera, K_override=cam_K, T_world_cam_override=cam_T_world_cam)
        nonlocal detector
        if detector is None:
            oracle_dets = (
                [("peg", peg_place_pos_w(s["id"]), s["diameter"], PEG_LENGTH)
                 for s in PEG_SPECS]
                + [("hole", hole_top_w(s["id"]), 2 * s["hole_radius"], None)
                   for s in SOCKET_SPECS]
            )
            detector = GroundTruthDetector(oracle_dets, K, T_world_cam)
        # Route SOM into the per-episode debug subdir when debug is on; the
        # downstream dumper expects it there. Otherwise keep the legacy
        # timestamped name in report_dir.
        if debug_on:
            ep_dir = debug_dir / f"episode_{cycle_index:04d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            som_path = ep_dir / "som.png"
        else:
            som_path = args.report_dir / f"som_{int(time.time() * 1000)}.png"
            som_path.parent.mkdir(parents=True, exist_ok=True)
        # Stash so the debug printer can dump rgb/depth/K/T_world_cam
        # without re-capturing (and without coupling its signature to
        # the camera object).
        _run_perception.last_raw_capture = (rgb, depth, K, T_world_cam, som_path)
        # Expose the (now-instantiated, in the GT case) detector so the
        # debug dumper can pull last_raw_responses off LocateAnything for
        # prompt iteration. Set on every call because the GT detector is
        # built lazily on the first call.
        _run_perception.last_detector = detector
        return perceive_scene(
            rgb=rgb, depth=depth, K=K, T_world_cam=T_world_cam,
            detector=detector, vlm_backend=vlm_backend,
            som_image_out=str(som_path),
        )

    _run_perception.last_raw_capture = None
    return _run_perception, backend_label


# ─── State reader (used by every control phase) ──────────────────────────────
def update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids):
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


# ─── Pose-hold primitive (gripper grasp + release + post-reset settle) ───────
def hold_tcp(
    *, target_tcp_pos_w, n_ticks, finger_cmd,
    sim, scene, robot, osc, command, ee_target_pose_b, down_quat_b,
    arm_joint_ids, finger_ids, ee_frame_idx, joint_centers, sim_dt,
):
    """Servo TCP to ``target_tcp_pos_w`` (world frame) for n_ticks ticks.

    Used for the static phases: settle after reset, close gripper at the
    grasp pose, open gripper at the insertion pose. The arm orientation
    is held at ``down_quat_b`` throughout.
    """
    if not torch.is_tensor(target_tcp_pos_w):
        target_tcp_pos_w = torch.tensor(target_tcp_pos_w, device=sim.device,
                                        dtype=torch.float32)
    target_world = target_tcp_pos_w.unsqueeze(0).expand(scene.num_envs, -1)
    quat_world = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)
    for _ in range(n_ticks):
        if not simulation_app.is_running():
            return
        (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
         root_pose_w, _, joint_pos, joint_vel
         ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_world, quat_world,
        )
        ee_target_pose_b[:, 0:3] = target_pos_b
        command[:, 0:3] = target_pos_b
        command[:, 3:7] = quat_world

        osc.set_command(command=command, current_ee_pose_b=ee_pose_b,
                        current_task_frame_pose_b=None)
        ee_force_b = torch.zeros(scene.num_envs, 3, device=sim.device)
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b, current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_matrix, gravity=gravity,
            current_joint_pos=joint_pos, current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers,
        )
        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)


# ─── DMP segment runner (TCP-space) ──────────────────────────────────────────
def run_dmp_tcp_segment(
    *,
    segment_name,
    dmp, dmp_weights, tau,
    start_tcp_w, goal_tcp_w,
    sim, scene, robot, peg_for_trail,
    osc, command, ee_target_pose_b,
    down_quat_b,
    finger_cmd,
    arm_joint_ids, finger_ids, ee_frame_idx,
    joint_centers,
    playback_speed, sim_dt,
    trail_marker, trail_pts,
    record: SegmentRecord,
):
    """Integrate one DMP segment from ``start_tcp_w`` → ``goal_tcp_w``.

    The DMP integrates in TCP (hand) space — its output IS the OSC
    position command at every tick. This is the change from peg-tip
    DMP integration: no per-step ``hand_pos_for_peg_tip`` conversion.

    ``finger_cmd`` is held constant for the duration of the segment
    (closed during grasp→insertion, open during home→grasp), so each
    segment is a fixed-gripper-state TCP trajectory.

    Returns when the DMP's internal clock exceeds ``DMP_TIME_EXTENSION_FACTOR
    * tau``. We trust the DMP's convergence properties — no early-exit
    on goal-reached — because (a) the goal is the DMP's attractor so it
    asymptotes naturally, (b) early-exit on tolerance would couple the
    two segments to a per-task tuning parameter and we want them
    comparable.
    """
    dmp.x_0 = torch.tensor(start_tcp_w, device=sim.device, dtype=torch.float32)
    dmp.x_goal = torch.tensor(goal_tcp_w, device=sim.device, dtype=torch.float32)
    dmp.reset_step(dmp_weights)

    record.start_tcp_w = list(map(float, start_tcp_w))
    record.goal_tcp_w = list(map(float, goal_tcp_w))
    record.tau_s = float(tau)

    quat_world = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)
    dmp_time = 0.0
    n_ticks = 0
    TRAIL_EVERY = 10
    print(f"[seg] {segment_name}: TCP "
          f"{[round(float(v),3) for v in start_tcp_w]} -> "
          f"{[round(float(v),3) for v in goal_tcp_w]} tau={tau:.2f}s")

    while simulation_app.is_running():
        if dmp_time >= DMP_TIME_EXTENSION_FACTOR * tau:
            break

        step_dt = sim_dt * playback_speed
        # DMP step: output is now a TCP position in world frame.
        tcp_target_w, _ = dmp.step(step_dt)
        tcp_target_w = tcp_target_w.to(sim.device)        # [1, 3] or [3]
        if tcp_target_w.dim() == 1:
            tcp_target_w = tcp_target_w.unsqueeze(0)
        dmp_time += step_dt

        (jacobian_b, mass_matrix, gravity, ee_pose_b, ee_vel_b,
         root_pose_w, ee_pose_w, joint_pos, joint_vel,
         ) = update_states(sim, scene, robot, ee_frame_idx, arm_joint_ids)

        # World TCP target → base frame for OSC.
        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            tcp_target_w.expand(scene.num_envs, -1),
            quat_world,
        )
        ee_target_pose_b[:, 0:3] = target_pos_b
        command[:, 0:3] = target_pos_b
        command[:, 3:7] = quat_world

        osc.set_command(command=command, current_ee_pose_b=ee_pose_b,
                        current_task_frame_pose_b=None)
        ee_force_b = torch.zeros(scene.num_envs, 3, device=sim.device)
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b, current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_matrix, gravity=gravity,
            current_joint_pos=joint_pos, current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers,
        )
        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.set_joint_position_target(finger_cmd, joint_ids=finger_ids)
        robot.write_data_to_sim()

        # Trail: visualize the ACTUAL TCP, not the commanded one (so we see
        # tracking error in the marker stream).
        actual_tcp_w = robot.data.body_pos_w[:, ee_frame_idx, :]
        if n_ticks % TRAIL_EVERY == 0:
            trail_pts.append(actual_tcp_w[0].detach().cpu().tolist())
            record.tcp_trail_w.append(actual_tcp_w[0].detach().cpu().tolist())
        if trail_pts:
            trail_marker.visualize(
                translations=torch.tensor(trail_pts, device=sim.device)
            )

        sim.step(render=True)
        robot.update(sim_dt)
        scene.update(sim_dt)
        n_ticks += 1

    # Wrap up: record final TCP + tracking error to goal, plus current peg tip.
    record.n_ticks = n_ticks
    final_tcp_w = robot.data.body_pos_w[:, ee_frame_idx, :][0].detach().cpu().numpy()
    record.final_tcp_w = final_tcp_w.tolist()
    record.final_tcp_err_m = float(np.linalg.norm(final_tcp_w - np.asarray(goal_tcp_w)))
    if peg_for_trail is not None:
        peg_tip = peg_tip_from_body(
            peg_for_trail.data.root_pos_w, peg_for_trail.data.root_quat_w
        )[0].detach().cpu().tolist()
        record.final_peg_tip_w = peg_tip


# ─── Scene reset (home position) ─────────────────────────────────────────────
def reset_to_home(scene, robot, sim, sim_dt):
    """Step 1 of the motion procedure: reset to default joints.

    Hard kinematic reset to the configured default joint pose. After this
    + a brief settle, the TCP is at the "home" position (which is just
    wherever forward-kinematics puts the hand frame given the default
    joints — we don't impose an explicit Cartesian home target).

    Pegs are warped to their declared spec positions, with zero velocity.
    That doesn't depend on the chosen peg here: in this procedure we let
    the robot navigate to whichever peg perception picked, so every peg
    is left at its spec location.
    """
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_effort_target(
        torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)
    )
    robot.write_data_to_sim()
    robot.reset()

    for spec in PEG_SPECS:
        peg = scene[spec["id"]]
        place = peg_place_pos_w(spec["id"])
        peg_reset = torch.cat([
            torch.tensor(place, device=sim.device).unsqueeze(0),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device),
        ], dim=-1)
        peg.write_root_pose_to_sim(peg_reset)
        peg.write_root_velocity_to_sim(
            torch.zeros(scene.num_envs, 6, device=sim.device))

    scene.update(sim_dt)
    robot.update(sim_dt)


def serialize_perception(p: PerceptionOutput) -> PerceptionRecord:
    return PerceptionRecord(
        objects={
            oid: {
                "label": o.label,
                "pose_w": o.pose_w.tolist(),
                "diameter_m": o.diameter_m,
                "height_m": o.height_m,
                "confidence": o.confidence,
                "inlier_ratio": o.primitive_inlier_ratio,
            }
            for oid, o in p.objects.items()
        },
        assignment=p.matching.assignment,
        confidence=p.matching.confidence,
        validated_pairs=p.validated_pairs,
        rejected_pairs=p.rejected_pairs,
        vlm_latency_s=p.matching.vlm_latency_s,
        unfilled_holes=p.matching.unfilled_holes,
        ungrasped_pegs=p.matching.ungrasped_pegs,
    )


# ─── Perception debug dump (only used when --debug-perception is set) ────────
def _save_depth_visualization(depth: np.ndarray, out_path: Path) -> None:
    """Save a normalized grayscale PNG of the depth map.

    NaNs and non-positive samples are masked out and rendered as black so
    invalid returns are visually distinguishable from "very close" surfaces.
    No colormap — keeps the dependency surface small (PIL only). For a
    pretty preview, ``np.load(depth.npy)`` and use matplotlib downstream.
    """
    from PIL import Image
    d = np.asarray(depth, dtype=np.float32).copy()
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    mask = np.isfinite(d) & (d > 0)
    if mask.any():
        d_min = float(d[mask].min())
        d_max = float(d[mask].max())
        scale = max(d_max - d_min, 1e-6)
        viz = ((d - d_min) / scale * 255.0).clip(0, 255).astype(np.uint8)
        viz[~mask] = 0
    else:
        viz = np.zeros_like(d, dtype=np.uint8)
    Image.fromarray(viz, mode="L").save(out_path)


def _save_rgb(rgb: np.ndarray, out_path: Path) -> None:
    """Save an RGB capture, tolerating either uint8 or float [0,1] input."""
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
    Image.fromarray(arr, mode="RGB").save(out_path)


def _k_matrix_3x3(K):
    """Return a JSON-friendly 3x3 intrinsic matrix from a CameraIntrinsics or array.

    ``capture_frame`` hands perception a ``CameraIntrinsics`` dataclass (not a
    bare matrix), so ``np.asarray(K)`` yields a 0-d object array that neither
    serializes nor iterates. Rebuild the matrix from its fields instead.
    """
    if hasattr(K, "fx"):
        return [[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]]
    return np.asarray(K).tolist()


def dump_perception_debug(
    perc: PerceptionOutput,
    perception_fn,
    cycle_index: int,
    debug_dir: Path,
    backend_label: str,
) -> None:
    """Print + persist everything we know about this perception step.

    Layout::

        <debug_dir>/episode_NNNN/
            rgb.png            (raw camera RGB; non-GT backends only)
            depth.npy          (raw float depth, HxW)
            depth_viz.png      (normalized grayscale preview)
            camera.json        (K + T_world_cam)
            som.png            (set-of-mark annotated frame, already written
                                by the perception backend)
            perception.json    (full serialized PerceptionOutput)

    Always-on terminal printout: per-detection poses, dimensions,
    confidences, primitive inlier ratios; VLM assignment with confidences;
    rejected pairs with reasons; latency.
    """
    ep_dir = debug_dir / f"episode_{cycle_index:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    raw = getattr(perception_fn, "last_raw_capture", None)
    # ── Save intermediate images + camera info (only when we have a capture).
    # GT backend has none — we still dump perception.json for it.
    if raw is not None:
        rgb, depth, K, T_world_cam, som_path = raw
        try:
            _save_rgb(rgb, ep_dir / "rgb.png")
        except Exception as exc:
            print(f"[debug-perception] WARN: failed to save rgb.png: {exc}")
        try:
            np.save(ep_dir / "depth.npy", np.asarray(depth, dtype=np.float32))
            _save_depth_visualization(depth, ep_dir / "depth_viz.png")
        except Exception as exc:
            print(f"[debug-perception] WARN: failed to save depth: {exc}")
        try:
            T = np.asarray(T_world_cam)
            cam_meta = {
                "K": _k_matrix_3x3(K),
                "T_world_cam": T.tolist(),
                "cam_pos_w": T[:3, 3].tolist(),
                "rgb_shape": list(np.asarray(rgb).shape),
                "depth_shape": list(np.asarray(depth).shape),
            }
            with (ep_dir / "camera.json").open("w") as f:
                json.dump(cam_meta, f, indent=2)
        except Exception as exc:
            print(f"[debug-perception] WARN: failed to save camera.json: {exc}")
        # SOM was written by the perception backend straight into ep_dir
        # already (see build_perception_backend); just sanity-note it.
        som_note = f"som.png ({'present' if Path(som_path).exists() else 'MISSING'})"
    else:
        som_note = "none (ground_truth backend has no camera capture)"

    # ── Persist a serialized JSON of the perception output. ─────────────────
    try:
        rec = serialize_perception(perc)
        with (ep_dir / "perception.json").open("w") as f:
            json.dump(asdict(rec), f, indent=2, default=str)
    except Exception as exc:
        print(f"[debug-perception] WARN: failed to save perception.json: {exc}")

    # ── Terminal dump. ──────────────────────────────────────────────────────
    print(f"[debug-perception] ===== episode {cycle_index} "
          f"(backend={backend_label}) =====")
    if raw is not None:
        print(f"[debug-perception] camera K:")
        for row in _k_matrix_3x3(K):
            print(f"[debug-perception]   {[round(float(v), 3) for v in row]}")
        T = np.asarray(T_world_cam)
        print(f"[debug-perception] cam pos_w = "
              f"{[round(float(v), 4) for v in T[:3, 3]]}")
        print(f"[debug-perception] T_world_cam (row-major 4x4):")
        for row in T:
            print(f"[debug-perception]   {[round(float(v), 4) for v in row]}")
        print(f"[debug-perception] rgb shape = {list(np.asarray(rgb).shape)}, "
              f"depth shape = {list(np.asarray(depth).shape)}, "
              f"depth range = [{float(np.nanmin(depth)):.3f}, "
              f"{float(np.nanmax(depth)):.3f}] m")

    print(f"[debug-perception] detected objects ({len(perc.objects)}):")
    for oid, o in perc.objects.items():
        pose = np.asarray(o.pose_w).tolist()
        pose_str = "[" + ", ".join(f"{v:+.4f}" for v in pose) + "]"
        h_str = "n/a" if o.height_m is None else f"{o.height_m:.4f}m"
        print(f"[debug-perception]   {oid:>10s}  label={o.label:<5s}  "
              f"pose_w={pose_str}  d={o.diameter_m:.4f}m  h={h_str}  "
              f"conf={o.confidence:.3f}  inlier={o.primitive_inlier_ratio:.3f}")

    print(f"[debug-perception] VLM matching (latency "
          f"{perc.matching.vlm_latency_s:.3f}s):")
    if perc.matching.assignment:
        for peg, hole in perc.matching.assignment.items():
            conf = perc.matching.confidence.get(peg, float("nan"))
            print(f"[debug-perception]   {peg} -> {hole}  conf={conf:.3f}")
    else:
        print(f"[debug-perception]   (empty assignment)")

    print(f"[debug-perception] validated pairs ({len(perc.validated_pairs)}): "
          f"{perc.validated_pairs}")
    if perc.rejected_pairs:
        print(f"[debug-perception] rejected pairs ({len(perc.rejected_pairs)}):")
        for p, h, reason in perc.rejected_pairs:
            print(f"[debug-perception]   {p} -> {h}: {reason}")
    else:
        print(f"[debug-perception] rejected pairs: none")
    print(f"[debug-perception] unfilled holes: {perc.matching.unfilled_holes}")
    print(f"[debug-perception] ungrasped pegs: {perc.matching.ungrasped_pegs}")

    # ── If the active detector is LocateAnything, surface its raw text
    # responses for prompt iteration. We reach into perception_fn's
    # closure via a known attribute pattern; missing => silently skip.
    detector_obj = getattr(perception_fn, "last_detector", None)
    raw_responses = getattr(detector_obj, "last_raw_responses", None)
    if raw_responses:
        print(f"[debug-perception] detector raw responses "
              f"(class -> model output):")
        for cls, ans in raw_responses.items():
            # Truncate to keep log readable; full text is in perception.json
            # if the perception layer chose to persist it.
            preview = ans.replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:237] + "..."
            print(f"[debug-perception]   {cls:>6s}: {preview}")

    print(f"[debug-perception] artifacts -> {ep_dir}/  "
          f"(perception.json, {som_note}"
          + (", rgb.png, depth.npy, depth_viz.png, camera.json"
             if raw is not None else "")
          + ")")


# ─── Main loop ───────────────────────────────────────────────────────────────
def run_simulator(sim, scene, primed, perception_fn, backend_label, debug_dir):
    robot: Articulation = scene["robot"]
    finger_ids, _ = robot.find_joints(["panda_finger_joint.*"])
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]
    ee_frame_idx = robot.find_bodies("panda_hand")[0][0]
    dmp = primed.dmp
    dmp_weights = primed.weights
    tau = primed.tau

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
        nullspace_stiffness=5.0,
        nullspace_damping_ratio=1.0,
    )
    osc = OperationalSpaceController(osc_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers (visualization only).
    fm_cfg = FRAME_MARKER_CFG.copy()
    fm_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_marker = VisualizationMarkers(fm_cfg.replace(prim_path="/Visuals/ee_current"))

    def _sphere(prim, color, r):
        cfg = VisualizationMarkersCfg(
            prim_path=prim,
            markers={"sphere": sim_utils.SphereCfg(
                radius=r,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )},
        )
        return VisualizationMarkers(cfg)
    home_marker = _sphere("/Visuals/dmp_home_tcp", (0.5, 0.5, 0.5), 0.012)
    grasp_marker = _sphere("/Visuals/dmp_grasp_tcp", (0.1, 0.8, 0.1), 0.012)
    insert_marker = _sphere("/Visuals/dmp_insert_tcp", (0.9, 0.1, 0.1), 0.012)
    trail_marker = _sphere("/Visuals/dmp_trail", (0.2, 0.4, 0.9), 0.005)

    sim_dt = sim.get_physics_dt()
    robot.update(dt=sim_dt)
    joint_centers = robot.data.default_joint_pos[:, arm_joint_ids].clone()
    down_quat_b = torch.tensor(GRIPPER_DOWN_QUAT, device=sim.device)
    grip_target = finger_grip_target(scene.num_envs, len(finger_ids), sim.device)
    open_target = torch.full((scene.num_envs, len(finger_ids)), 0.04, device=sim.device)

    command = torch.zeros(scene.num_envs, osc.action_dim, device=sim.device)
    command[:, 3:7] = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)
    ee_target_pose_b = torch.zeros(scene.num_envs, 7, device=sim.device)
    ee_target_pose_b[:, 3:7] = down_quat_b.unsqueeze(0).expand(scene.num_envs, -1)

    args_cli.report_dir.mkdir(parents=True, exist_ok=True)

    cycle_index = 0
    while simulation_app.is_running():
        if args_cli.max_cycles and cycle_index >= args_cli.max_cycles:
            print(f"[run] reached max-cycles={args_cli.max_cycles}, exiting")
            break

        cycle_t0 = time.perf_counter()
        print(f"\n[run] ===== cycle {cycle_index} (backend={backend_label}) =====")

        # ── Step 0: perception (one shot at episode start). ─────────────────
        try:
            perc: PerceptionOutput = perception_fn(cycle_index)
            chosen_peg, chosen_hole = perc.primary_pair()
            print(f"[run] perception picked {chosen_peg} -> {chosen_hole} "
                  f"(VLM latency {perc.matching.vlm_latency_s:.2f}s)")
            if perc.rejected_pairs:
                for p, h, reason in perc.rejected_pairs:
                    print(f"        rejected {p}->{h}: {reason}")
            perception_record = serialize_perception(perc)
            if debug_dir is not None:
                # Best-effort: a failure in the debug dumper must not abort
                # the episode. Surface the exception, then carry on.
                try:
                    dump_perception_debug(
                        perc, perception_fn, cycle_index, debug_dir,
                        backend_label,
                    )
                except Exception as exc:
                    print(f"[debug-perception] dump failed: {exc}")
        except Exception as exc:
            print(f"[run] PERCEPTION FAILED: {exc}")
            cycle_index += 1
            continue

        # Use PERCEIVED poses (not oracle PEG_SPECS) for grasp/insertion targets.
        # This is the whole point of having a perception layer: the DMP
        # endpoints depend on what perception measured, with its noise budget.
        peg_pose_w_perceived = perc.peg_pose(chosen_peg)         # [7]
        hole_pose_w_perceived = perc.hole_pose(chosen_hole)      # [7]
        peg_center_w = peg_pose_w_perceived[:3].astype(np.float32)
        hole_top_pos_w = hole_pose_w_perceived[:3].astype(np.float32)
        # If --insert-to-bottom, drive the tip past the rim to the seat;
        # otherwise stop at the rim and leave depth to a future residual.
        if args_cli.insert_to_bottom:
            insertion_tip_target_w = hole_top_pos_w + np.array(
                [0.0, 0.0, -HOLE_DEPTH_M], dtype=np.float32)
        else:
            insertion_tip_target_w = hole_top_pos_w

        # ── Step 1: HOME (reset to default joints). ─────────────────────────
        print(f"[run] step 1: HOME (default joints)")
        reset_to_home(scene, robot, sim, sim_dt)
        # Brief settle so the robot's joint state is stationary before we
        # read the home TCP. Open gripper is the natural pre-grasp state.
        hold_tcp(
            target_tcp_pos_w=robot.data.body_pos_w[:, ee_frame_idx, :][0].detach(),
            n_ticks=SETTLE_TICKS, finger_cmd=open_target,
            sim=sim, scene=scene, robot=robot, osc=osc,
            command=command, ee_target_pose_b=ee_target_pose_b,
            down_quat_b=down_quat_b,
            arm_joint_ids=arm_joint_ids, finger_ids=finger_ids,
            ee_frame_idx=ee_frame_idx, joint_centers=joint_centers, sim_dt=sim_dt,
        )
        home_tcp_w = robot.data.body_pos_w[:, ee_frame_idx, :][0].detach().cpu().numpy()
        print(f"[run]   home TCP = {home_tcp_w.tolist()}")

        # Compute the two TCP goals for this episode (peg-aware geometry).
        grasp_tcp_w = hand_pos_for_grasp(
            torch.tensor(peg_center_w, device=sim.device).unsqueeze(0),
            down_quat_b.unsqueeze(0),
        )[0].detach().cpu().numpy()
        insertion_tcp_w = hand_pos_for_peg_tip(
            torch.tensor(insertion_tip_target_w, device=sim.device).unsqueeze(0),
            down_quat_b.unsqueeze(0),
        )[0].detach().cpu().numpy()
        print(f"[run]   grasp TCP  = {grasp_tcp_w.tolist()}")
        print(f"[run]   insert TCP = {insertion_tcp_w.tolist()}")

        home_marker.visualize(translations=torch.tensor([home_tcp_w], device=sim.device))
        grasp_marker.visualize(translations=torch.tensor([grasp_tcp_w], device=sim.device))
        insert_marker.visualize(translations=torch.tensor([insertion_tcp_w], device=sim.device))

        # ── Step 2: DMP-to-GRASP. Gripper OPEN throughout. ──────────────────
        print(f"[run] step 2: DMP home -> grasp")
        trail_pts: list[list[float]] = []
        seg_home_grasp = SegmentRecord(
            segment_name="home_to_grasp",
            start_tcp_w=[], goal_tcp_w=[], tau_s=0.0, n_ticks=0,
        )
        run_dmp_tcp_segment(
            segment_name="home_to_grasp",
            dmp=dmp, dmp_weights=dmp_weights, tau=tau,
            start_tcp_w=home_tcp_w, goal_tcp_w=grasp_tcp_w,
            sim=sim, scene=scene, robot=robot,
            peg_for_trail=None,             # gripper empty, no peg yet
            osc=osc, command=command, ee_target_pose_b=ee_target_pose_b,
            down_quat_b=down_quat_b,
            finger_cmd=open_target,
            arm_joint_ids=arm_joint_ids, finger_ids=finger_ids,
            ee_frame_idx=ee_frame_idx, joint_centers=joint_centers,
            playback_speed=args_cli.playback_speed, sim_dt=sim_dt,
            trail_marker=trail_marker, trail_pts=trail_pts,
            record=seg_home_grasp,
        )

        # ── Step 3: GRASP. Hold TCP, close gripper. ─────────────────────────
        print(f"[run] step 3: GRASP (close gripper)")
        # Servo to the COMMANDED grasp TCP (not whatever the DMP left us at —
        # the DMP may have settled with a small tracking error; we close at
        # the canonical grasp pose so finger-pad geometry is correct).
        hold_tcp(
            target_tcp_pos_w=torch.tensor(grasp_tcp_w, device=sim.device,
                                          dtype=torch.float32),
            n_ticks=GRIP_HOLD_TICKS, finger_cmd=grip_target,
            sim=sim, scene=scene, robot=robot, osc=osc,
            command=command, ee_target_pose_b=ee_target_pose_b,
            down_quat_b=down_quat_b,
            arm_joint_ids=arm_joint_ids, finger_ids=finger_ids,
            ee_frame_idx=ee_frame_idx, joint_centers=joint_centers, sim_dt=sim_dt,
        )

        # ── Step 4: DMP-to-INSERTION. Gripper CLOSED throughout. ────────────
        print(f"[run] step 4: DMP grasp -> insertion")
        peg_obj = scene[chosen_peg]
        # Re-read the actual TCP post-grasp (catches any small shift while
        # pads contacted the peg).
        post_grasp_tcp_w = robot.data.body_pos_w[:, ee_frame_idx, :][0].detach().cpu().numpy()
        seg_grasp_insert = SegmentRecord(
            segment_name="grasp_to_insertion",
            start_tcp_w=[], goal_tcp_w=[], tau_s=0.0, n_ticks=0,
        )
        run_dmp_tcp_segment(
            segment_name="grasp_to_insertion",
            dmp=dmp, dmp_weights=dmp_weights, tau=tau,
            start_tcp_w=post_grasp_tcp_w, goal_tcp_w=insertion_tcp_w,
            sim=sim, scene=scene, robot=robot,
            peg_for_trail=peg_obj,
            osc=osc, command=command, ee_target_pose_b=ee_target_pose_b,
            down_quat_b=down_quat_b,
            finger_cmd=grip_target,
            arm_joint_ids=arm_joint_ids, finger_ids=finger_ids,
            ee_frame_idx=ee_frame_idx, joint_centers=joint_centers,
            playback_speed=args_cli.playback_speed, sim_dt=sim_dt,
            trail_marker=trail_marker, trail_pts=trail_pts,
            record=seg_grasp_insert,
        )

        # Score the insertion (uses peg tip, not TCP, since that's the
        # geometrically meaningful thing — the TCP can be on-target while
        # a bad grasp leaves the peg cocked).
        peg_tip_w_now = peg_tip_from_body(peg_obj.data.root_pos_w, peg_obj.data.root_quat_w)
        success, lat, depth_frac = insertion_success(
            peg_tip_w_now,
            hole_center_w=hole_center_w(chosen_hole),
            rim_top_z=socket_top_z(chosen_hole),
        )

        # ── Step 5: RELEASE. Hold TCP at insertion, open gripper. ───────────
        print(f"[run] step 5: RELEASE (open gripper)")
        hold_tcp(
            target_tcp_pos_w=torch.tensor(insertion_tcp_w, device=sim.device,
                                          dtype=torch.float32),
            n_ticks=RELEASE_HOLD_TICKS, finger_cmd=open_target,
            sim=sim, scene=scene, robot=robot, osc=osc,
            command=command, ee_target_pose_b=ee_target_pose_b,
            down_quat_b=down_quat_b,
            arm_joint_ids=arm_joint_ids, finger_ids=finger_ids,
            ee_frame_idx=ee_frame_idx, joint_centers=joint_centers, sim_dt=sim_dt,
        )

        # Optional retract (outside the requested procedure, useful for viz).
        if args_cli.retract_after_release:
            retract_tcp = insertion_tcp_w.copy()
            retract_tcp[2] += 0.15
            hold_tcp(
                target_tcp_pos_w=torch.tensor(retract_tcp, device=sim.device,
                                              dtype=torch.float32),
                n_ticks=80, finger_cmd=open_target,
                sim=sim, scene=scene, robot=robot, osc=osc,
                command=command, ee_target_pose_b=ee_target_pose_b,
                down_quat_b=down_quat_b,
                arm_joint_ids=arm_joint_ids, finger_ids=finger_ids,
                ee_frame_idx=ee_frame_idx, joint_centers=joint_centers, sim_dt=sim_dt,
            )

        # ── Dump report. ────────────────────────────────────────────────────
        report = EpisodeReport(
            cycle_index=cycle_index,
            chosen_peg=chosen_peg,
            chosen_hole=chosen_hole,
            perception=perception_record,
            home_tcp_w=home_tcp_w.tolist(),
            segments=[seg_home_grasp, seg_grasp_insert],
            final_lateral_err_m=float(lat[0]),
            final_depth_frac=float(depth_frac[0]),
            overall_success=bool(success[0]),
            wall_time_s=time.perf_counter() - cycle_t0,
        )
        out = args_cli.report_dir / f"episode_{cycle_index:04d}.json"
        with out.open("w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"[run] cycle {cycle_index} done: success={report.overall_success}, "
              f"lateral={lat[0]*1000:.1f}mm, depth_frac={depth_frac[0]:+.2f}, "
              f"seg1_tcp_err={seg_home_grasp.final_tcp_err_m*1000:.1f}mm, "
              f"seg2_tcp_err={seg_grasp_insert.final_tcp_err_m*1000:.1f}mm "
              f"→ {out}")

        cycle_index += 1


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    cfg = load_config(args_cli.config)
    _bootstrap_paths(cfg["methods_dmp"])
    torch.manual_seed(cfg["seed"])
    dmp_device = torch.device(args_cli.device)
    primed = build_dmp_from_config(cfg, args_cli.task_params, dmp_device)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.2, 1.2, 1.0], [0.5, 0.0, 0.4])

    scene_cfg = ClutteredInsertionSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    attach_pegs_and_sockets(scene_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete.")

    # Debug dir: only materialised when --debug-perception is set. Default
    # to <report_dir>/debug/ so artifacts colocate with episode_*.json.
    if args_cli.debug_perception:
        debug_dir = (args_cli.debug_dir
                     if args_cli.debug_dir is not None
                     else args_cli.report_dir / "debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] perception debug ON -> {debug_dir}")
    else:
        debug_dir = None

    perception_fn, backend_label = build_perception_backend(
        args_cli, scene, debug_dir)
    run_simulator(sim, scene, primed, perception_fn, backend_label, debug_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
