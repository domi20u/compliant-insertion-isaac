"""Interface contracts for the one-shot perception layer.

This module defines the data classes that flow between the four sub-stages
of the cluttered-scene perception pipeline. The contract is intentionally
narrow so that each sub-stage can be replaced independently (e.g. swap
GroundingDINO for YOLO-World, swap Claude for Gemini, swap RGB-D
back-projection for FoundationPose) without touching the DMP layer.

Layer boundary
--------------
Everything in this module is consumed by the DMP runner via a SINGLE
function call at episode start::

    perception_output = perceive(rgb, depth, K, T_cam_world)
    peg_pose_w  = perception_output.objects[peg_id].pose_w
    hole_pose_w = perception_output.objects[hole_id].pose_w
    # ... hand these to the DMP as the two task endpoints, done.

There is no in-loop call. The VLM is a one-shot scene parser; if you want
closed-loop visual servoing, that is a different (and much harder)
project. We deliberately exclude VLA-style end-to-end models here — they
collapse perception, matching, and motion into a black box that defeats
the controlled comparison between DMP / diffusion-policy / residual-RL
that the project is built around.

Coordinate conventions
----------------------
- ``pose_w`` is a 7-vector ``(x, y, z, qw, qx, qy, qz)`` in the simulator's
  world frame. The DMP runner converts to the robot base frame via
  ``subtract_frame_transforms`` exactly as in the single-object case.
- For pegs the orientation encodes the cylindrical axis (peg local +Z =
  cylinder long axis), matching ``scene_cfg.PEG_LENGTH``-aligned geometry.
- For holes the orientation encodes the hole axis (hole local +Z points
  OUT of the socket, i.e. opposite the insertion direction).
- ``diameter_m`` for a peg is its measured (NOT nominal) outer diameter,
  produced by primitive fitting. The geometric cross-validation step
  uses this to verify that the VLM-chosen peg actually fits the
  VLM-chosen hole.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ─── Single detected object ──────────────────────────────────────────────────
@dataclass
class DetectedObject:
    """One object recovered by detection + pose estimation.

    Carries everything downstream stages need: the geometric primitive
    parameters (for fit-based validation), the 2D image evidence (for
    Set-of-Mark annotation), and the 6-DoF world pose (for the DMP).

    ``label`` is the open-vocabulary text class produced by GroundingDINO;
    we normalize it to one of {"peg", "hole"} after detection.

    ``confidence`` is the *combined* score:
        detection_score * mask_quality * primitive_fit_inlier_ratio

    so it reflects "are we sure this object exists AND that we have a
    clean pose for it". The matching stage uses this to break ties when
    the VLM is ambiguous.
    """

    object_id: str                              # "peg_0", "hole_2", ...
    label: Literal["peg", "hole"]
    pose_w: np.ndarray                          # [7] (xyz + wxyz quat)
    diameter_m: float                           # measured outer diameter
    height_m: float | None                      # peg length; None for holes
    confidence: float                           # ∈ [0, 1]
    bbox_xyxy: tuple[int, int, int, int]        # for SoM annotation
    mask: np.ndarray | None = None              # [H, W] bool, for SoM
    primitive_inlier_ratio: float = 0.0         # how well the cylinder fit
    color_rgb: tuple[float, float, float] | None = None  # for VLM hint


# ─── VLM matching output ─────────────────────────────────────────────────────
@dataclass
class MatchingResult:
    """Output of the single structured VLM call.

    ``assignment`` is the headline result: ``{peg_id: hole_id}``. This is
    what the DMP layer ultimately consumes (alongside the per-object
    poses). Everything else is for the validation stage and for debugging
    when the VLM gets it wrong.

    ``top_k`` carries the VLM's runner-up matches so the validation
    stage can fall back when the first choice fails geometric
    cross-validation. Without this, a single hallucination kills the
    episode; with it, we degrade gracefully.

    ``unfilled_holes`` / ``ungrasped_pegs`` capture the VLM's explicit
    statement about which objects are NOT participating. This is an
    existence check: if the VLM claims to match a peg_id we never
    detected, we catch it here rather than crashing in the DMP.
    """

    assignment: dict[str, str]                                  # {peg: hole}
    confidence: dict[str, float]                                # per-pair conf
    top_k: dict[str, list[tuple[str, float, str]]]              # {peg: [(hole, conf, reason)]}
    unfilled_holes: list[str] = field(default_factory=list)
    ungrasped_pegs: list[str] = field(default_factory=list)
    raw_vlm_response: str = ""
    vlm_latency_s: float = 0.0


# ─── Final perception output (DMP runner consumes this) ──────────────────────
@dataclass
class PerceptionOutput:
    """Single struct the DMP runner takes at episode start.

    After validation, ``validated_pairs`` is the authoritative list of
    (peg_id, hole_id) tuples the runner should execute. It may be:

      - shorter than ``matching.assignment`` if some pairs failed
        cross-validation (e.g. measured peg diameter > measured hole
        diameter — i.e. it physically won't fit)
      - reordered if the runner is configured to execute by
        confidence rather than VLM order

    ``rejected_pairs`` carries (peg, hole, reason) for every dropped
    pair so post-hoc analysis can attribute failures to perception
    vs. matching vs. control.
    """

    objects: dict[str, DetectedObject]
    matching: MatchingResult
    validated_pairs: list[tuple[str, str]]
    rejected_pairs: list[tuple[str, str, str]] = field(default_factory=list)

    # Episode-level provenance (saved to disk for the post-hoc dashboard).
    rgb_path: str | None = None
    depth_path: str | None = None
    som_image_path: str | None = None

    # ----- Convenience accessors used by the DMP runner -----
    def peg_pose(self, peg_id: str) -> np.ndarray:
        return self.objects[peg_id].pose_w

    def hole_pose(self, hole_id: str) -> np.ndarray:
        return self.objects[hole_id].pose_w

    def primary_pair(self) -> tuple[str, str]:
        """The first validated (peg, hole) pair, or raise if none.

        For the current runner we execute a single insertion per
        episode; this returns the pair to use. When you extend to
        multi-insertion episodes, iterate over ``validated_pairs``.
        """
        if not self.validated_pairs:
            raise RuntimeError(
                "[perception] no validated peg→hole pairs survived "
                "cross-validation. Rejection log:\n"
                + "\n".join(f"  {p}→{h}: {r}" for p, h, r in self.rejected_pairs)
            )
        return self.validated_pairs[0]


# ─── Camera intrinsics (passed through to pose estimation) ───────────────────
@dataclass
class CameraIntrinsics:
    """Pinhole camera model used by RGB-D back-projection.

    Isaac Lab cameras expose this as the K matrix; we unpack the four
    numbers we actually need so the math reads cleanly.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_K(cls, K: np.ndarray, width: int, height: int) -> CameraIntrinsics:
        return cls(
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            width=width, height=height,
        )
