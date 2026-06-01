"""Defensive validation of the VLM's peg→hole matching.

This stage sits BETWEEN the VLM call and the DMP layer. It enforces
the invariants that the VLM is not trusted to satisfy on its own:

  1. EXISTENCE — every peg_id and hole_id named by the VLM was
     actually detected by the perception pipeline. (VLMs hallucinate
     missing labels; the matching call is the most common point of
     failure.)

  2. ROLE — the VLM didn't swap peg/hole labels (asking it to map
     "P1 → H2" sometimes produces "H1 → P2" on smaller models).

  3. COLOR — the chosen peg and hole agree in color. The benchmark uses
     uniform-diameter pegs and tints each socket with its intended peg's
     color, so the assignment key is color, not size. We reject a pair
     whose measured peg / hole colors disagree beyond a tolerance
     (catches the matcher pairing a red peg with a green socket).

  4. UNIQUENESS — no two pegs are assigned to the same hole.

The validator does NOT silently fix problems. When the primary
assignment fails, it walks the VLM's top_k alternatives in order and
returns the first one that passes; if none pass, the pair is
rejected with a recorded reason. This makes failure modes legible to
the post-hoc dashboard.
"""
from __future__ import annotations

from .interfaces import DetectedObject, MatchingResult, PerceptionOutput


# ─── Color-match tolerance ───────────────────────────────────────────────────
# The assignment key is COLOR (pegs are uniform-diameter; each socket is
# tinted its intended peg's color). A pair is accepted iff the measured peg
# and hole colors agree within COLOR_MATCH_MAX_DIST — Euclidean distance in
# RGB ∈ [0, 1]^3. Loosen it if lighting / shading pulls measured colors apart;
# tighten it if distinct peg colors are close enough to be confused. We do
# NOT gate on diameter anymore: at this object scale the measured diameter is
# too noisy to be a reliable discriminator, and size is intentionally constant.
#
# Calibrated on a real frame (peg mask color vs hole block-ring color): correct
# same-color pairs sit at ~0.04-0.08 RGB distance, cross-color pairs at
# ~0.24-0.30, so 0.20 separates them with margin. Relies on holes sampling the
# block ring (pose_estimation._ring_color), not the dark void.
COLOR_MATCH_MAX_DIST = 0.20


# ─── Per-pair validation ─────────────────────────────────────────────────────
def _validate_pair(
    peg_id: str, hole_id: str,
    objects: dict[str, DetectedObject],
) -> tuple[bool, str]:
    """Return (passed, reason).

    Reason is empty on success, a short diagnostic string on failure.
    """
    if peg_id not in objects:
        return False, f"peg {peg_id} not in detected objects"
    if hole_id not in objects:
        return False, f"hole {hole_id} not in detected objects"
    peg = objects[peg_id]
    hole = objects[hole_id]
    if peg.label != "peg":
        return False, f"{peg_id} is labeled {peg.label}, not peg"
    if hole.label != "hole":
        return False, f"{hole_id} is labeled {hole.label}, not hole"

    # Color-consistency: each socket is tinted its intended peg's color, so a
    # correct assignment pairs same-colored peg + hole.
    dist = _color_distance(peg.color_rgb, hole.color_rgb)
    if dist is None:
        return False, f"missing color on {peg_id} or {hole_id} (perception failure)"
    if dist > COLOR_MATCH_MAX_DIST:
        return False, (
            f"color mismatch: peg {_fmt_color(peg.color_rgb)} vs hole "
            f"{_fmt_color(hole.color_rgb)} (dist {dist:.2f} > "
            f"{COLOR_MATCH_MAX_DIST:.2f})"
        )

    # Sanity: poses must be finite.
    if not _poses_finite(peg, hole):
        return False, "non-finite pose component (perception failure)"

    return True, ""


def _color_distance(a, b) -> float | None:
    """Euclidean RGB distance, or None if either color is missing."""
    import numpy as np
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def _fmt_color(c) -> str:
    return "?" if c is None else "(" + ", ".join(f"{v:.2f}" for v in c) + ")"


def _poses_finite(*objs: DetectedObject) -> bool:
    import numpy as np
    return all(bool(np.all(np.isfinite(o.pose_w))) for o in objs)


# ─── Top-level validation: walks top_k alternatives on failure ───────────────
def validate(
    objects: dict[str, DetectedObject],
    matching: MatchingResult,
) -> PerceptionOutput:
    """Walk the matching, validate each pair, fall back to alternatives.

    Pegs are processed in order of VLM confidence (highest first).
    For each peg:

      1. Try the primary assignment. If it passes and the target hole
         isn't already used, accept it.
      2. Otherwise walk top_k alternatives in order, accepting the
         first that passes and isn't already used.
      3. If nothing passes, record (peg, primary_hole, reason) in
         rejected_pairs and continue with the next peg.

    The output is a PerceptionOutput with validated_pairs ready for the
    DMP runner to execute.
    """
    validated: list[tuple[str, str]] = []
    rejected: list[tuple[str, str, str]] = []
    used_holes: set[str] = set()

    # Sort by confidence descending so the highest-confidence peg gets
    # first pick when alternatives overlap.
    peg_order = sorted(
        matching.assignment.keys(),
        key=lambda p: matching.confidence.get(p, 0.0),
        reverse=True,
    )

    for peg_id in peg_order:
        primary_hole = matching.assignment[peg_id]
        candidates: list[tuple[str, str]] = [(primary_hole, "primary")]
        for alt_hole, _, _ in matching.top_k.get(peg_id, []):
            candidates.append((alt_hole, "alternative"))

        accepted = False
        last_reason = "no candidates"
        for hole_id, source in candidates:
            if hole_id in used_holes:
                last_reason = f"hole {hole_id} already assigned"
                continue
            ok, reason = _validate_pair(peg_id, hole_id, objects)
            if ok:
                validated.append((peg_id, hole_id))
                used_holes.add(hole_id)
                accepted = True
                # If we fell back, note that this peg's primary failed.
                if source == "alternative":
                    rejected.append((peg_id, primary_hole,
                                     f"primary failed; using alternative {hole_id}"))
                break
            last_reason = reason

        if not accepted:
            rejected.append((peg_id, primary_hole, last_reason))

    return PerceptionOutput(
        objects=objects,
        matching=matching,
        validated_pairs=validated,
        rejected_pairs=rejected,
    )
