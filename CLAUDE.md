# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Benchmarking DMP, Diffusion Policy, and residual RL for contact-rich peg-in-hole insertion in [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). A Franka Panda grasps a cylindrical peg from a table and inserts it into a socket, controlled by a policy-primed DMP over an OSC (Operational Space Controller).

## Installation

```bash
# Install the main package (editable)
pip install -e .

# Install the DMP fork (required for methods/dmp/)
pip install git+https://github.com/domi20u/MP_PyTorch.git
```

The real perception backends pull in heavy optional deps that are **lazy-imported**
on first use, so nothing below is needed unless you pass a non-`ground_truth`
backend to the cluttered runner:
- `--detector-backend grounding_dino` → `transformers`, `sam2` (GroundingDINO + SAM 2)
- `--vlm-backend ollama` → a running Ollama server + `ollama pull qwen2.5vl:7b`

## Running scripts

**All Isaac Lab / Isaac Sim scripts must be launched via `isaaclab.sh -p`** — the AppLauncher must be constructed before any `isaaclab.*` imports, so these scripts cannot be run with plain `python`.

```bash
# Generate a DMP trajectory offline (no sim, outputs .npy + .npz)
python scripts/sim/gen_insertion_trajectory.py \
    --config configs/insertion_traj.yaml --task-params 0.05 0.20

# Single peg: end-to-end DMP execution in sim
/path/to/IsaacLab/isaaclab.sh -p scripts/sim/run_dmp_in_sim_e2e.py \
    --config configs/insertion_traj.yaml --task-params 0.05 0.20

# Four-peg / four-hole sequential insertions
/path/to/IsaacLab/isaaclab.sh -p scripts/sim/run_dmp_multi_peg_e2e.py \
    --config configs/insertion_traj.yaml

# Cluttered scene (VLM-based perception + TCP-controlled DMP)
# Two independent backend knobs: --detector-backend selects the box/mask source,
# --vlm-backend selects the peg→hole matcher. Both default to ground_truth (an
# oracle that bypasses the real models, so the runner works with no extra deps).
/path/to/IsaacLab/isaaclab.sh -p scripts/sim/cluttered_run_e2e.py \
    --config configs/insertion_traj.yaml --task-params 0.05 0.20 \
    --detector-backend ground_truth \  # or: grounding_dino, locate_anything
    --vlm-backend ground_truth         # or: ollama, mock

# Collect Diffusion Policy demos (DMP rollouts → one RoboMimic-ish HDF5)
# Only the DMP phase is recorded; approach/grasp are sliced out as OOD.
/path/to/IsaacLab/isaaclab.sh -p scripts/data/collect_demos.py \
    --config configs/insertion_traj.yaml --out data/insertion_demos.hdf5 \
    --num-episodes 100

# Cluttered, insertion-only, goal-conditioned demos (10 mm socket only).
# Randomizes target peg + socket XY; records only the peg-in-hand insert
# segment; per-episode collision-safe lift; success-filtered.
/path/to/IsaacLab/isaaclab.sh -p scripts/data/collect_cluttered_demos.py \
    --config configs/insertion_traj.yaml \
    --out data/cluttered_insertion_demos.hdf5 --num-episodes 200

# Convert any collected HDF5 → LeRobotDataset (plain python, needs `lerobot`;
# NO isaaclab.sh). Feeds LeRobot Diffusion Policy / SmolVLA / π0.
python scripts/data/hdf5_to_lerobot.py \
    --hdf5 data/cluttered_insertion_demos.hdf5 \
    --repo-id local/cluttered_insertion \
    --root data/lerobot/cluttered_insertion

# Train the self-contained diffusion policy on the HDF5 (plain python, GPU;
# torch+torchvision only — no lerobot/diffusers). Frozen ResNet18 image feats +
# 1-D conv U-Net + DDPM. Saves head weights + norm stats + cfg to one .pt.
python scripts/train/train_diffusion_policy.py \
    --hdf5 data/cluttered_insertion_demos.hdf5 \
    --out assets/dp_models/cluttered_dp.pt --epochs 200

# Closed-loop sim test of the trained policy (run by the USER via isaaclab.sh;
# servo-grasps peg_0, then the DP drives the insertion). Needs torchvision in
# the Isaac python: isaaclab.sh -p -m pip install torchvision
/path/to/IsaacLab/isaaclab.sh -p scripts/sim/eval_diffusion_policy.py \
    --checkpoint assets/dp_models/cluttered_dp.pt --num-episodes 20

# DMP offline training (PI²) — standalone, no isaaclab.sh needed
python methods/dmp/scripts/generate_data.py --scenario insertion \
    --output-dir results/ins_v0 --n-runs 10
python methods/dmp/scripts/train_policy.py  --scenario insertion \
    --data-dir results/ins_v0 --model-name model --n-basis 15
```

There are no automated tests defined yet.

## Architecture

### Package layout

- `compliant_insertion/` — installable Python package (`pip install -e .`)
  - `env/` — Isaac Lab scene configs + all geometry constants and frame-math helpers
  - `perception/` — one-shot VLM perception pipeline (interfaces, detection, pose estimation, VLM matching, validation)
  - `trajectory/`, `eval/` — stubs for future use
- `methods/dmp/` — **not** a proper Python package; bootstrapped onto `sys.path` at runtime via `_bootstrap_paths()` in each runner script. Contains the DMP wrapper, PI² optimizer, and NN policy.
- `scripts/sim/` — Isaac Lab runner scripts (must use `isaaclab.sh -p`)
- `scripts/data/` — demo collection for Diffusion Policy training
- `assets/` — trained NN policy (`.pth`), demo trajectory, DMP rollouts
- `configs/` — YAML config shared by all runners

### OSC control pattern

All runners use Isaac Lab's `OperationalSpaceController` (`pose_abs` target). **Critical setup required in every scene config:**

1. Arm actuator stiffness and damping must be zeroed at the config level — OSC provides the joint torques directly; leaving PD gains in place causes the actuator to fight OSC.
2. Robot gravity is disabled (`disable_gravity=True`) and `gravity_compensation=True` is set in the OSC config. Toggle both together.

Finger joints are driven separately with `set_joint_position_target` (position control, not effort).

### Frame conventions

- Robot quaternions use `(w, x, y, z)` convention throughout.
- `GRIPPER_DOWN_QUAT = (0, 1, 0, 0)` — panda_hand +Z pointing world −Z (straight down); used as the fixed orientation throughout all phases.
- Peg tip is `PEG_TIP_OFFSET_Z` along the hand +Z axis from the hand origin. Convert a desired peg-tip world position to a hand target with `hand_pos_for_peg_tip(tip_pos_w, hand_quat_w)`.
- Multi-peg and cluttered runners use **TCP-space** DMPs (controlling the fingertip point directly, `PANDA_FINGERTIP_OFFSET_Z` along hand +Z) rather than peg-tip space. Single-peg runner uses peg-tip space.

### Hole geometry

Isaac Sim has no boolean subtraction on primitives. Every socket "hole" is four cuboid rim walls arranged around an empty cylindrical void. The peg descends into the gap between the walls. `HOLE_CLEARANCE` (single-peg) or `HOLE_CLEARANCES` (multi-peg) is the sole tolerance knob — change it and restart the sim.

### DMP pipeline

1. A demo trajectory (`assets/trajectories/traj_demo.csv`) is imitated by a `DMPWrapper`.
2. A TorchScript NN policy (`assets/dmp_models/insertion_model_2p.pth`) maps two task parameters to per-DOF forcing-term weights.
3. The primed DMP is integrated one step per sim tick (`dmp.step(dt)`) inside the control loop — no pre-computed trajectory file is used at runtime.
4. Rotodilatation (`rescale: rotodilatation_xy` in config) rescales the demo's shape onto the runtime `(x_0, x_goal)` endpoints so a single trained policy covers different pick/place geometries without retraining.

In multi-task runners a **single DMP object is reused** across legs: only `x_0`, `x_goal`, and the forcing weights (selected by task-param pair) are swapped via `dmp.reset_step(weights)` between segments.

### Cluttered-scene perception pipeline

`compliant_insertion/perception/interfaces.py` defines the data contracts: `DetectedObject`, `MatchingResult`, `PerceptionOutput`. The pipeline stages are:

1. **Detection** (`detection.py`) — open-vocabulary boxes (GroundingDINO) + masks (SAM 2)
2. **Pose estimation** (`pose_estimation.py`) — RGB-D back-projection + cylinder primitive fitting
3. **VLM matching** (`som_prompting.py`) — Set-of-Mark annotated image → structured Claude/Ollama call → `MatchingResult`
4. **Validation** (`validation.py`) — geometric cross-check that the assigned peg physically fits the assigned hole

The runner calls `perceive(rgb, depth, K, T_cam_world)` **once per episode** and hands the resulting `PerceptionOutput` to the DMP planner. `--vlm-backend ground_truth` bypasses all perception and reads poses directly from the sim.

### Config file (`configs/insertion_traj.yaml`)

Shared by every runner. Key fields: `methods_dmp` (path to `methods/dmp/`), `model_path`, `demo_path`, `start`/`goal` (placeholder endpoints — overridden at runtime from live scene geometry), `n_basis`, `rescale`, `ins_offset`. All paths are resolved relative to repo root if not absolute.
