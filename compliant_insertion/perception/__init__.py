"""One-shot perception layer for cluttered peg-in-hole scenes.

Public API
----------
The DMP runner only needs three names from this package::

    from compliant_insertion.perception import perceive_scene, PerceptionOutput
    from compliant_insertion.perception import OllamaQwenBackend

    output = perceive_scene(
        rgb=rgb, depth=depth,
        K=intrinsics, T_world_cam=cam_pose,
        vlm_backend=OllamaQwenBackend(model="qwen2.5vl:7b"),
    )
    peg_id, hole_id = output.primary_pair()
    peg_pose_w  = output.peg_pose(peg_id)
    hole_pose_w = output.hole_pose(hole_id)

Backends
--------
Three backends ship for use during different phases of development:

  - ``GroundTruthPerception``  — reads object poses directly from the
    Isaac Lab scene. Use during DMP-side development so the pipeline
    runs without model weights, an API key, or any VLM at all.

  - ``MockVLMBackend``  (in som_prompting) — pick-by-diameter stub.
    Lets you exercise detection + pose-estimation + validation + DMP
    end-to-end without standing up Ollama. Not a serious matcher.

  - ``OllamaQwenBackend``  — the real thing. Calls Qwen2.5-VL via the
    local Ollama HTTP API, with JSON-schema-constrained output.

The detection + pose stages are backend-agnostic; only the VLM call
swaps. ``GroundTruthPerception`` is its own top-level path that
skips the detection stage entirely and is provided as a convenience.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .detection import Detector, GroundedSAM2Detector, GroundTruthDetector
from .interfaces import (
    CameraIntrinsics,
    DetectedObject,
    MatchingResult,
    PerceptionOutput,
)
from .pose_estimation import estimate_poses
from .som_prompting import (
    MockVLMBackend,
    OllamaQwenBackend,
    VLMBackend,
    match_pegs_to_holes,
)
from .validation import validate

__all__ = [
    # Data classes
    "CameraIntrinsics", "DetectedObject", "MatchingResult", "PerceptionOutput",
    # Backends
    "Detector", "GroundedSAM2Detector", "GroundTruthDetector",
    "VLMBackend", "OllamaQwenBackend", "MockVLMBackend",
    # Top-level entry points
    "perceive_scene", "GroundTruthPerception",
]


# ─── Top-level entry point ───────────────────────────────────────────────────
def perceive_scene(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: CameraIntrinsics,
    T_world_cam: np.ndarray,
    *,
    detector: Detector,
    vlm_backend: VLMBackend,
    som_image_out: str | None = None,
) -> PerceptionOutput:
    """Run the full pipeline: detect → pose → SoM-match → validate.

    Args:
        rgb: [H, W, 3] uint8 image from the workbench camera.
        depth: [H, W] float32 depth in meters (Z-component, not slant).
        K: camera intrinsics.
        T_world_cam: 4x4 SE(3), camera pose in world frame.
        detector: GroundingDINO+SAM2 or a stub Detector implementation.
        vlm_backend: OllamaQwenBackend or MockVLMBackend.
        som_image_out: optional path to dump the SoM-annotated image
            (useful for the post-hoc dashboard).
    """
    # Stage 1: detection (off-the-shelf).
    detections = detector(rgb)
    if not detections:
        raise RuntimeError(
            "[perception] detector returned no objects. "
            "Either the scene is empty, the camera is mis-aimed, or "
            "the prompt is wrong for this asset palette."
        )

    # Stage 2: 6-DoF pose estimation from RGB-D (classical geometry).
    objects_list = estimate_poses(detections, rgb, depth, K, T_world_cam)
    if not objects_list:
        raise RuntimeError(
            "[perception] all detections failed primitive fitting. "
            "Check depth map units (must be meters) and the world-+Z "
            "prior (set z_axis_prior=False for oblique cameras)."
        )
    objects = {obj.object_id: obj for obj in objects_list}

    # Stage 3: SoM-prompted VLM matching (the one structured call).
    matching = match_pegs_to_holes(
        rgb, objects_list, vlm_backend, som_image_out=som_image_out,
    )

    # Stage 4: defensive validation + top-K fallback.
    output = validate(objects, matching)
    output.som_image_path = som_image_out
    return output


# ─── Ground-truth perception (skip detection + VLM entirely) ─────────────────
def _rgb_dist(a, b) -> float:
    """Euclidean RGB distance between two colors."""
    return float(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5)


@dataclass
class _OracleObject:
    object_id: str
    label: str
    pose_w: np.ndarray
    diameter_m: float
    height_m: float | None
    color_rgb: tuple[float, float, float] | None = None


class GroundTruthPerception:
    """Skip detection + VLM and read straight from the simulator.

    Used during DMP-side development to keep the runner working when
    Ollama / GroundingDINO / SAM 2 aren't available. The validation
    stage still runs, so the same downstream contract is exercised.

    Build by calling ``GroundTruthPerception.from_scene(scene)`` after
    Isaac Lab finishes spawning, then call its ``__call__`` like any
    perceive function.
    """

    def __init__(self, oracle_objects: list[_OracleObject]):
        self.oracle_objects = oracle_objects

    @classmethod
    def from_scene(
        cls,
        peg_specs: list[tuple],
        hole_specs: list[tuple],
    ) -> GroundTruthPerception:
        """Build from oracle tuples.

        Peg specs are ``(id, pos_w, diameter, height[, color])`` and hole
        specs ``(id, pos_w, diameter[, color])``. ``color`` is optional but
        recommended — the default matcher pairs peg→hole by color (the
        benchmark uses uniform-diameter pegs, so size can't discriminate).
        """
        oracles: list[_OracleObject] = []
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for spec in peg_specs:
            pid, pos_w, dia, height = spec[0], spec[1], spec[2], spec[3]
            color = spec[4] if len(spec) > 4 else None
            pose = np.concatenate([pos_w.astype(np.float32), identity])
            oracles.append(_OracleObject(pid, "peg", pose, dia, height, color))
        for spec in hole_specs:
            hid, pos_w, dia = spec[0], spec[1], spec[2]
            color = spec[3] if len(spec) > 3 else None
            pose = np.concatenate([pos_w.astype(np.float32), identity])
            oracles.append(_OracleObject(hid, "hole", pose, dia, None, color))
        return cls(oracles)

    def __call__(self, intended_pair: tuple[str, str] | None = None) -> PerceptionOutput:
        """Return a PerceptionOutput with one validated pair.

        ``intended_pair`` lets you script which pair the "VLM" picks
        for testing — useful for stress-testing the validator with
        deliberately wrong matches. If None, pick the tightest fit.
        """
        objects: dict[str, DetectedObject] = {}
        for o in self.oracle_objects:
            objects[o.object_id] = DetectedObject(
                object_id=o.object_id,
                label=o.label,                                      # type: ignore[arg-type]
                pose_w=o.pose_w,
                diameter_m=o.diameter_m,
                height_m=o.height_m,
                confidence=1.0,
                bbox_xyxy=(0, 0, 1, 1),
                mask=None,
                primitive_inlier_ratio=1.0,
                color_rgb=o.color_rgb,
            )

        pegs = [o.object_id for o in self.oracle_objects if o.label == "peg"]
        holes = [o.object_id for o in self.oracle_objects if o.label == "hole"]

        if intended_pair is not None:
            assignment = {intended_pair[0]: intended_pair[1]}
        else:
            # Match every peg to the nearest-color hole (unique). Falls back to
            # the tightest size fit only when colors are unavailable.
            assignment = {}
            used: set[str] = set()
            for p in pegs:
                free = [h for h in holes if h not in used]
                if not free:
                    break
                pc = objects[p].color_rgb
                if pc is not None and all(objects[h].color_rgb is not None for h in free):
                    h = min(free, key=lambda h: _rgb_dist(objects[h].color_rgb, pc))
                else:
                    h = min(free, key=lambda h: abs(
                        objects[h].diameter_m - objects[p].diameter_m))
                assignment[p] = h
                used.add(h)

        matching = MatchingResult(
            assignment=assignment,
            confidence={p: 1.0 for p in assignment},
            top_k={p: [] for p in assignment},
            unfilled_holes=[h for h in holes if h not in assignment.values()],
            ungrasped_pegs=[p for p in pegs if p not in assignment],
            raw_vlm_response="ground_truth",
            vlm_latency_s=0.0,
        )
        return validate(objects, matching)
