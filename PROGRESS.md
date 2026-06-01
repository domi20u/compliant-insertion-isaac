# Progress Log

A running log of work sessions on the cluttered peg-in-hole insertion experiment.
Newest entries on top.

---

## 2026-06-01 — Diffusion policy: train + test on 142 demos

### Goal
Train and test a diffusion policy on the 142-episode cluttered-insertion dataset
(`data/cluttered_insertion_demos.hdf5`).

### What was built (self-contained — no lerobot/diffusers, to avoid disturbing
the env's torch 2.7+cu128 for the Blackwell 5070 Ti)
- `compliant_insertion/policy/diffusion_policy.py` — CNN diffusion policy (Chi et
  al. style): frozen pretrained ResNet18 image embeddings + 1-D conv U-Net with
  FiLM + inline DDPM (cosine, ε-pred). Action = 3-D base-frame TCP position
  (down-quat constant); goal-conditioned on the insertion TCP pose.
- `compliant_insertion/policy/inference.py` — `DiffusionPolicyRunner`:
  receding-horizon deployment (n_obs_steps=2, pred_horizon=16, exec 8).
- `scripts/train/train_diffusion_policy.py` — loads HDF5, precomputes frozen
  image feats (≈3 s), episode-level 128/14 split, DDPM training, EMA, periodic
  held-out eval (sampled action-chunk L2 in mm). Saves head+stats+cfg to one .pt.
- `scripts/sim/eval_diffusion_policy.py` — closed-loop sim test (USER runs via
  isaaclab.sh): servo-grasp peg_0, then the DP drives the insertion, release,
  measure. Needs torchvision in the Isaac python.

### Result (training, 200 epochs, ~10 min on the 5070 Ti)
Held-out action-prediction L2 fell 24.7 → 13.4 → 11.8 → 10.7 → ~10.1 mm and
plateaued ~10 mm. Train loss ~0.002. Offline metric is encouraging but the real
test is the **closed-loop sim success rate** (user must run — agent can't render).

### Notes / next
- 10 mm action L2 vs the 10 mm hole clearance (5 mm radial) is borderline
  open-loop; receding-horizon + goal conditioning should tighten it closed-loop.
- This is a state+vision DP; if the closed-loop rate is low, options: predict
  delta actions, raise epochs, or add the residual policy for final seating.

### First closed-loop sim run (user) + smoothing fixes
Runs; **1/N successful** (pipeline + learned skill confirmed). But motion was
jerky → stiff OSC lurched on the noisy/discontinuous setpoints → friction grasp
slipped → peg drifted off-distribution → big misses/collisions. Added to the
deployment runner (`inference.py`), no retrain:
- **Temporal ensembling** (ACT-style): re-sample every tick via fast DDIM
  (`sample_ddim`), execute a near-uniform recency-weighted average of all
  overlapping chunk predictions. (Bug fixed: initial ens weight made the newest
  prediction dominate ~1800×, defeating the averaging.)
- **EMA low-pass** on the commanded TCP target (`--smooth-beta`, default 0.5) —
  the main jerk fix; + optional per-tick step clamp (`--max-step-m`).
Offline proxy: per-tick setpoint jump 16 mm → 3.3 mm (~5×). Precision is still
bounded by the model's ~10 mm action noise → if success stays low, retrain with
delta actions / denser data (collect at lower --playback-speed) and/or add the
residual policy for final seating.

---

## 2026-06-01 — Cluttered insertion-only demo collector + LeRobot export

### Goal
Generate goal-conditioned **insertion-only** demos in the cluttered scene for a
Diffusion Policy / flow-matching VLA (SmolVLA, π0), and a path into LeRobot's
dataset format. v1 scope: **10 mm-clearance socket only** (red peg_0 → red
hole_0); tighter clearances come later once a residual policy can seat them.

### Design decisions
- **Insertion only**: full HOME→grasp→insert procedure runs, but only the
  grasp→insert (peg-in-hand) segment is recorded — matches the deployment
  assumption and keeps approach transients out of the dataset.
- **TCP-space DMP** (like `cluttered_run_e2e`), so the recorded action = OSC TCP
  pose target — same action space the DP/VLA will command.
- **Ground-truth poses** (no perception in the loop); perception noise belongs
  in eval, not the expert demos.
- **Socket randomization enabled**: converted the socket fixtures
  (base + 4 rim walls) from static `AssetBaseCfg` to **kinematic
  `RigidObjectCfg`** so they can be teleported per-reset via
  `socket_component_offsets()`. Collision behavior unchanged (kinematic ⇒
  immovable by contact); `cluttered_run_e2e` (never moves them) is unaffected.
- **Collision-safe lift**: DMP `param1` is derived per episode from the
  transport distance so the transit apex clears the rim — fixes the original
  collision (param1=0.05 under-cleared). `--task-params` forces a fixed pair;
  the **success filter** is the real guarantee of clean data.

### Changes made
- `env/cluttered_scene_cfg.py`: sockets → kinematic `RigidObjectCfg`; new
  `_socket_layout()` (single source of truth for the 5 prims) +
  `socket_component_offsets()` (runtime teleport helper).
- `scripts/data/collect_cluttered_demos.py` (new): cluttered, socket+peg
  randomizing, insertion-only recorder. Wrist cam (on `panda_hand`) + external
  third-person cam at 128². Records obs {wrist/external image, ee_pose_b,
  ee_vel_b, joint_pos, gripper_pos}, action {OSC TCP pose target}, and
  per-episode goal attrs (`goal_pose_b`, `goal_tcp_w`, `task` string). Prints a
  final success-rate summary for calibrating the lift.
- `scripts/data/hdf5_to_lerobot.py` (new): standalone (plain-python, no
  isaaclab) HDF5 → `LeRobotDataset` converter. Maps to observation.images.
  {wrist,external}, observation.state (15), observation.environment_state
  (7-dim goal), action (7), per-episode language `task`. Read-path unit-tested
  against a stub dataset; tolerant of both lerobot import paths + task-API
  variants.

### Fixes after first in-sim run
First run: pegs + orange socket varied, but distractor sockets stayed put,
insertion was always same-direction, the EE collided with the socket, and
success was reported 0% despite visibly-good insertions. Root causes + fixes:
- **Distractors fixed + single-axis transport** → `sample_layout` now scatters
  ALL 3 pegs + ALL 3 sockets in one shared `--workspace` box (was separate
  peg/socket y-bands). Gives distractor-socket motion AND full-360° relative
  peg→socket direction. `reset_episode` teleports every socket now.
- **EE-rim collision during insertion + 0% success (same root cause)**: I was
  driving the goal too DEEP (full seating, `--insert-depth-frac` 1.0→0.75).
  With the Panda fingertip offset, a deep goal brings the fingers to the rim top
  (z=0.05). **Key correction from the user**: the DMP is already trained for the
  steep descent and seats cleanly when it just STOPS at the right goal (as
  `cluttered_run_e2e` shows) — do NOT add a hand-rolled vertical descent. So the
  fix is purely the stop position: target a SHALLOW partial insertion
  (`--insert-depth-frac` default **0.3**), which keeps the fingertips ~3 cm
  above the rim → no collision, and matches the division of labor (the residual
  policy does full seating later). `--success-depth-frac` (default 0.1) scores a
  demo as successful once the peg is ALIGNED and STARTED into the hole, not
  fully seated. (An earlier pre-insert-waypoint + straight-line descent
  decomposition was tried and reverted per the user — the DMP owns the descent.)
- Success check now uses the socket's actual `hole_radius` (0.013), not the
  single-peg default (0.018), and the teleported socket XY.
- **Placement**: bumped peg↔socket separation to 0.10 m (open gripper clears
  neighboring rims on the grasp descent) and added a `--corridor-clear`
  keep-out so distractors stay off the peg_0→hole_0 transit line. The
  whole-layout retry loop makes spec-fallback essentially never trigger.

### Third pass (collisions gone; scoring + speed + start pose)
No more collisions, but success still read 0% while the peg was visibly inserted
(scored while gripped at the shallow stop, so depth/lateral were marginal).
- **Score after RELEASE**, not while gripped: the recorded DMP insert segment
  ends at the shallow stop, then (un-recorded) the gripper opens and lifts clear
  so the aligned peg drops to the hole bottom under gravity — the true "is it in
  the hole?" test. Fixes the false 0%.
- **Speed**: `--playback-speed` default 5→**10** (faster robot motion; shorter,
  coarser recordings — drop back to 5 if OSC lag hurts alignment).
- **Start closer to the table**: after reset, servo to a Cartesian home above
  the workspace center at `--home-height` (default **0.30 m**) instead of the
  default ready pose. Uses the OSC to reach a reachable point — no blind
  joint-angle guessing.

### Fourth pass — wrong peg end in the success check
4/5 visibly-successful but reported success=0; every seated peg logged
`depth=-0.25` (i.e. 1 cm ABOVE the rim). Cause: the dynamically-grasped peg
keeps its upright SPAWN orientation (local +Z = world UP), so
`peg_tip_from_body` (local +Z) returns the peg's TOP, not the bottom that enters
the hole. For a seated peg (bottom 0.01, top 0.06, rim 0.05) that's
depth_frac=(0.05-0.06)/0.04 = -0.25 — exactly the logged value. Added
`peg_insertion_tip_w` (lower-z end of the peg) and score on that → seated pegs
read depth_frac≈1.0; the real miss (ep0, lateral 44.5 mm) still fails on lateral.
Note: the insertion GOAL was always correct (`hand_pos_for_peg_tip` targets the
peg bottom); only the post-hoc scorer used the wrong end.

### Next steps
- [ ] Re-run a batch; expect ≈4/5 success now, scored on the seated peg bottom.
      Tune `--insert-depth-frac` / `--home-height` / `--playback-speed` to taste.
- [ ] Install `lerobot` in the training env; convert; train DP + SmolVLA.
- [ ] Later: collect 5 mm / 1 mm demos once the residual policy exists.

---

## 2026-06-01 — OWLv2 detection, color-matching redesign, camera-extrinsic fix

### Goal
Get the cluttered-scene pipeline (`scripts/sim/cluttered_run_e2e.py`) running
end-to-end with a GPU-friendly perception stack on a 16 GB RTX 5070 Ti, and
reframe the experiment around **insertion accuracy** rather than size
discrimination.

### Outcome (current state)
**Pipeline runs end-to-end without errors.** Detection → pose estimation →
color matching → validation → DMP grasp → DMP insert all execute. The **grasp
is successful**; the **insertion is not yet** — the peg collides with the
socket block. Remaining work is **DMP / OSC control accuracy**, which is the
actual subject of the benchmark.

### Design decisions
- **Accuracy benchmark, not size discrimination.** All pegs share one diameter
  (`PEG_DIAMETER = 2*PEG_RADIUS = 16 mm`); per-peg **color** is the identity.
  The high-level planner assigns peg→socket **by color**.
- **Sockets** are tinted their intended peg's color (whole block) and given a
  **swept clearance {10, 5, 1} mm** (red/green/blue), giving an
  accuracy-vs-tolerance signal. `hole_radius = (PEG_DIAMETER + clearance)/2`.
- Matching key is color; diameter is no longer measured for matching (too noisy
  at this object scale). See memory `cluttered-color-matching-design`.

### Changes made
- **OWLv2 detector** (`perception/detection.py`): new `OWLv2SAM2Detector`
  (OWLv2-base + SAM2-hiera-small), single forward pass, ~2.3 GB VRAM, ~0.34 s
  warm — replaces the ~7.8 GB LocateAnything-3B that OOM'd. `--detector owlv2`,
  with `--owlv2-score-threshold` (0.25) and `--owlv2-crop-top-frac` (0.18, crops
  the robot base out of frame to kill spurious "hole" detections). `min_score`
  param added to `filter_detection`.
- **Color matching**: uniform pegs + colored sockets in
  `env/cluttered_scene_cfg.py`; `MockVLMBackend` and the Ollama prompt match by
  color; `validation.py` checks color consistency (`COLOR_MATCH_MAX_DIST = 0.20`)
  instead of the diameter-fit gate; `GroundTruthPerception` matches by color.
- **Hole color from the block, not the void** (`pose_estimation.py`): holes
  sample a ring just outside the void mask (`_ring_color`) — the dark hole
  interior was giving every hole the same brown color.
- **Camera extrinsic fix** (the big one): `camera.data.pos_w/quat` are
  unpopulated (origin / NaN) at perception time for the fixed-offset camera,
  which corrupted back-projection (perceived z ≈ −0.9, then NaN/`DLASCL`). Now
  computed analytically from config: `workbench_camera_K()` /
  `workbench_camera_T_world_cam()` (ROS pinhole convention), passed to
  `capture_frame(..., K_override=, T_world_cam_override=)`. Verified on a real
  frame: all objects land on the table at their spec XY.
- **Bug fixes**: depth squeezed to `[H,W]` in `camera_capture` + `back_project`
  (was `[H,W,1]` → indexing crash); debug-dump `camera.json`/K-print no longer
  crash on `CameraIntrinsics`.
- **Tooling**: `scripts/data/viz_detections.py` — sim-free detector overlay on a
  saved frame. Removed stray `=4.57` file.

### How to run
```bash
../../../IsaacLab/isaaclab.sh -p scripts/sim/cluttered_run_e2e.py \
    --config configs/insertion_traj.yaml --task-params 0.05 0.20 \
    --detector owlv2 --vlm-backend mock --debug-perception --max-cycles 1
```
Note: Isaac Lab must be run by the user in their own terminal (it does not
render from the agent shell). Perception logic is verified standalone on saved
frames (`reports/cluttered/debug/episode_0000/`: `rgb.png`, `depth.npy`).

### Next steps
- [ ] **Insertion accuracy** — peg collides with the socket block. Investigate
      DMP/OSC tracking of the insert segment, the insertion goal Z (rim vs seat,
      `--insert-to-bottom`), and approach alignment over the hole.
- [ ] Re-check the **1 mm-clearance (blue)** socket is the hardest; expect
      misses there first — that's the tolerance signal, not a bug.
- [ ] Optional robustness: world-space workspace filter (reject detections off
      the table) as a more general alternative to the top-crop.
- [ ] Optional: move the VLM matcher off-GPU (Claude API backend) so the real
      `--vlm-backend ollama` 7B doesn't contend with Isaac Sim for VRAM.
