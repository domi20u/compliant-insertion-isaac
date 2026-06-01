"""Convert a cluttered-insertion HDF5 dataset into a LeRobotDataset.

Reads the RoboMimic-ish HDF5 written by ``collect_cluttered_demos.py`` and
emits a ``LeRobotDataset`` ready for LeRobot's Diffusion Policy, SmolVLA, or π0
training. Kept SEPARATE from the sim collector on purpose: ``lerobot`` pulls in
a heavy dependency stack, and (unlike the collector) this runs as PLAIN python
in whatever env has ``lerobot`` installed — no ``isaaclab.sh`` needed.

Feature mapping
---------------
  observation.images.wrist        [H, W, 3] uint8   ← obs/wrist_image
  observation.images.external     [H, W, 3] uint8   ← obs/external_image
  observation.state               [15] float32      ← ee_pose_b(7) + ee_vel_b(6)
                                                       + gripper_pos(2)
  observation.environment_state    [7] float32      ← goal_pose_b (insertion TCP
                                                       target, base frame) —
                                                       broadcast across the episode
                                                       so a Diffusion Policy can
                                                       condition on the goal.
  action                           [7] float32      ← actions (OSC TCP pose target)
  task (language)                  str              ← per-episode "task" attr,
                                                       e.g. for SmolVLA / π0.

The 7-dim ee_pose / action carry the constant down-quaternion; drop columns
3–7 in the dataloader if you want a position-only (3-DoF) policy.

Usage::

    python scripts/data/hdf5_to_lerobot.py \\
        --hdf5 data/cluttered_insertion_demos.hdf5 \\
        --repo-id local/cluttered_insertion \\
        --root data/lerobot/cluttered_insertion

Notes
-----
* Targets the current ``LeRobotDataset`` v2 API (``create`` → ``add_frame`` →
  ``save_episode``). Both the new (post-restructure) and legacy import paths and
  the per-frame / per-episode ``task`` calling conventions are handled
  defensively, since LeRobot's API has shifted across releases.
* ``--video`` encodes camera streams as MP4 (smaller, needs ffmpeg); the default
  stores PNG frames (simpler, fast random access for a dataloader).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


# ─── LeRobot import (tolerate both module layouts). ──────────────────────────
def _import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except Exception:
        pass
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except Exception as exc:
        raise SystemExit(
            "[lerobot] could not import LeRobotDataset. Install lerobot "
            "(`pip install lerobot`) in this environment.\n"
            f"  underlying error: {exc}")


def _decode_attr(v):
    """h5py returns bytes for string attrs — decode to str."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def build_features(H: int, W: int, video: bool) -> dict:
    img_dtype = "video" if video else "image"
    img_feat = {"dtype": img_dtype, "shape": (H, W, 3),
                "names": ["height", "width", "channel"]}
    return {
        "observation.images.wrist": dict(img_feat),
        "observation.images.external": dict(img_feat),
        "observation.state": {
            "dtype": "float32", "shape": (15,),
            "names": ["ee_pos_x", "ee_pos_y", "ee_pos_z",
                      "ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z",
                      "ee_vlin_x", "ee_vlin_y", "ee_vlin_z",
                      "ee_vang_x", "ee_vang_y", "ee_vang_z",
                      "finger_0", "finger_1"]},
        "observation.environment_state": {
            "dtype": "float32", "shape": (7,),
            "names": ["goal_pos_x", "goal_pos_y", "goal_pos_z",
                      "goal_quat_w", "goal_quat_x", "goal_quat_y", "goal_quat_z"]},
        "action": {
            "dtype": "float32", "shape": (7,),
            "names": ["pos_x", "pos_y", "pos_z",
                      "quat_w", "quat_x", "quat_y", "quat_z"]},
    }


def _add_frame(dataset, frame: dict, task: str):
    """add_frame with task — API differs across lerobot versions."""
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        frame = {**frame, "task": task}
        dataset.add_frame(frame)


def _save_episode(dataset, task: str):
    try:
        dataset.save_episode()
    except TypeError:
        dataset.save_episode(task=task)


def main() -> None:
    ap = argparse.ArgumentParser(description="HDF5 → LeRobotDataset converter.")
    ap.add_argument("--hdf5", type=Path, required=True)
    ap.add_argument("--repo-id", required=True,
                    help="LeRobot repo id, e.g. local/cluttered_insertion.")
    ap.add_argument("--root", type=Path, default=None,
                    help="Local dataset root (default: lerobot's cache).")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override fps (default: control_hz attr from the HDF5).")
    ap.add_argument("--video", action="store_true",
                    help="Encode images as MP4 (needs ffmpeg). Default: PNG frames.")
    ap.add_argument("--robot-type", default="panda")
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    LeRobotDataset = _import_lerobot_dataset()

    with h5py.File(args.hdf5, "r") as h5:
        data = h5["data"]
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        if not demo_keys:
            raise SystemExit(f"[lerobot] no demos in {args.hdf5}")
        if args.max_episodes is not None:
            demo_keys = demo_keys[:args.max_episodes]

        fps = args.fps or float(_decode_attr(h5.attrs.get("control_hz", 100.0)))
        sample = data[demo_keys[0]]["obs"]["wrist_image"]
        H, W = int(sample.shape[1]), int(sample.shape[2])
        print(f"[lerobot] {len(demo_keys)} demos, image {H}x{W}, fps={fps}, "
              f"video={args.video}")

        features = build_features(H, W, args.video)
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=int(round(fps)),
            features=features,
            root=str(args.root) if args.root else None,
            robot_type=args.robot_type,
            use_videos=args.video,
        )

        for dk in demo_keys:
            grp = data[dk]
            obs = grp["obs"]
            T = int(grp.attrs["length"])
            task = _decode_attr(grp.attrs.get("task", "insert the peg into the socket"))

            wrist = obs["wrist_image"][:]            # [T,H,W,3] uint8
            external = obs["external_image"][:]
            ee_pose = obs["ee_pose_b"][:].astype(np.float32)      # [T,7]
            ee_vel = obs["ee_vel_b"][:].astype(np.float32)       # [T,6]
            gripper = obs["gripper_pos"][:].astype(np.float32)   # [T,2]
            actions = grp["actions"][:].astype(np.float32)       # [T,7]
            goal = np.asarray(grp.attrs["goal_pose_b"], np.float32)  # [7]

            state = np.concatenate([ee_pose, ee_vel, gripper], axis=1)  # [T,15]
            goal_b = np.broadcast_to(goal, (T, 7)).astype(np.float32)

            for t in range(T):
                _add_frame(dataset, {
                    "observation.images.wrist": wrist[t],
                    "observation.images.external": external[t],
                    "observation.state": state[t],
                    "observation.environment_state": goal_b[t],
                    "action": actions[t],
                }, task=task)
            _save_episode(dataset, task=task)
            print(f"[lerobot]   {dk}: {T} frames")

    # Some versions consolidate stats/metadata lazily; call if present.
    if hasattr(dataset, "consolidate"):
        try:
            dataset.consolidate()
        except Exception as exc:
            print(f"[lerobot] consolidate() skipped: {exc}")

    out = args.root if args.root else "(lerobot cache)"
    print(f"[lerobot] DONE → {out}  (repo_id={args.repo_id})")


if __name__ == "__main__":
    main()
