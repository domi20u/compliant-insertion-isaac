"""Set-of-Mark prompting and structured VLM call for peg→hole matching.

This module is the ORIGINAL contribution of the perception layer. The
upstream detector/segmenter and the downstream VLM (local Qwen2.5-VL
served by Ollama) are both off-the-shelf; the custom work is:

  1. Set-of-Mark annotation — overlay numbered labels on the detected
     objects so the VLM can refer to them by integer ID rather than
     "the leftmost red one". This is what makes a single structured
     call sufficient: no follow-up clarifications, no iterative
     pointing, no chain-of-thought tax.

  2. Structured-output enforcement — the VLM is prompted to fill a
     fixed JSON schema using Ollama's ``format`` parameter (JSON
     schema-constrained decoding, available from Ollama 0.5+), so
     the downstream validation stage parses a known shape rather
     than trying to extract assignments from prose.

  3. Defensive prompt design — explicit instructions to (a) state
     UNMATCHED objects rather than forcing assignments, (b) provide
     top-K alternatives per peg, (c) include a one-sentence reason
     per assignment so post-hoc analysis can debug what the VLM
     was looking at.

We are NOT using the VLM to estimate poses or do geometric reasoning.
Frontier VLMs cannot reliably measure millimeters from an image, and
asking them to do so just hides perception errors inside a confident
prose answer. Pose comes from depth + RANSAC; the VLM only does the
combinatorial assignment problem on top.

Why local Ollama / Qwen2.5-VL rather than a hosted API:
  - keeps the whole stack on the RTX 5070 Ti workstation, no
    network round-trip per episode
  - reproducible: same model weights, same outputs, no silent
    model-version churn
  - lines up with the existing Ollama job-scouting + paper-summarizer
    tooling on the same machine
  - Qwen2.5-VL 7B fits comfortably alongside Isaac Sim on a 16 GB
    GPU; bump to 32B if you off-load Isaac Sim to a second GPU.
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import Protocol

import numpy as np

from .interfaces import DetectedObject, MatchingResult


# ─── Set-of-Mark annotation ──────────────────────────────────────────────────
def annotate_set_of_mark(
    rgb: np.ndarray,
    objects: list[DetectedObject],
    label_radius_px: int = 16,
    mask_alpha: float = 0.35,
) -> tuple[np.ndarray, dict[str, int]]:
    """Overlay numbered markers + mask tints on the RGB image.

    Returns (annotated_image, id_to_mark_number).

    Pegs are numbered with a "P" prefix (P1, P2, ...) and holes with
    an "H" prefix (H1, H2, ...). This is more legible than a single
    integer space and lets the VLM keep peg-vs-hole straight when it
    reasons about the assignment.

    The mask tint helps the VLM ground its description to the exact
    pixels we measured. Without it, on cluttered scenes the VLM sometimes
    refers to an adjacent object that looks similar — the tint forces
    the conversation onto OUR detections, not the raw scene.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:                              # pragma: no cover
        raise RuntimeError(
            "[som] PIL is required for Set-of-Mark annotation. "
            "Install with: pip install pillow"
        ) from e

    img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    id_to_mark: dict[str, int] = {}
    peg_n = 0
    hole_n = 0
    for obj in objects:
        if obj.label == "peg":
            peg_n += 1
            mark = f"P{peg_n}"
            tint = (255, 200, 80, int(255 * mask_alpha))   # warm for pegs
        else:
            hole_n += 1
            mark = f"H{hole_n}"
            tint = (80, 180, 255, int(255 * mask_alpha))   # cool for holes
        id_to_mark[obj.object_id] = mark                   # type: ignore[assignment]

        # Tint the mask.
        if obj.mask is not None:
            tint_layer = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
            tint_layer[obj.mask] = tint
            overlay.alpha_composite(Image.fromarray(tint_layer))
            draw = ImageDraw.Draw(overlay)

        # Place the label at the bbox center (more stable than mask centroid
        # for thin pegs whose centroid drifts off the visible silhouette).
        x0, y0, x1, y1 = obj.bbox_xyxy
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        # White disc with black outline for legibility on any background.
        draw.ellipse(
            (cx - label_radius_px, cy - label_radius_px,
             cx + label_radius_px, cy + label_radius_px),
            fill=(255, 255, 255, 255),
            outline=(0, 0, 0, 255),
            width=2,
        )
        # Text centered.
        tb = draw.textbbox((0, 0), mark, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.text((cx - tw / 2, cy - th / 2 - 2), mark,
                  fill=(0, 0, 0, 255), font=font)

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(composited), id_to_mark


# ─── VLM backend protocol ────────────────────────────────────────────────────
class VLMBackend(Protocol):
    """A frontier VLM that can take an image + JSON schema + return a dict."""

    def call(
        self,
        image_rgb: np.ndarray,
        prompt: str,
        schema: dict,
    ) -> tuple[dict, str, float]:
        """Return (parsed_json, raw_text, latency_seconds)."""
        ...


# ─── Ollama / Qwen2.5-VL backend ─────────────────────────────────────────────
class OllamaQwenBackend:
    """Local Qwen2.5-VL served by Ollama, with JSON-schema-constrained output.

    Talks to the Ollama HTTP API at ``host`` (default localhost:11434).
    Ollama exposes a ``format`` parameter that takes either the string
    ``"json"`` or a full JSON schema; with a schema, the model is
    constrained to produce structurally valid output. This replaces the
    tool-use mechanism we'd use against a hosted frontier API and gives
    us the same downstream parse path.

    Model tags worth knowing:
      - ``qwen2.5vl:7b``   — default, ~9 GB VRAM, good for SoM matching
      - ``qwen2.5vl:32b``  — better at fine color/material disambiguation,
                              ~22 GB VRAM, slower
      - ``qwen2.5vl:72b``  — overkill for this task, but useful as the
                              "frontier sanity check" baseline in the
                              paper's ablation table.

    Pull the model once with ``ollama pull qwen2.5vl:7b`` before first run.
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        host: str = "http://localhost:11434",
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        num_ctx: int = 8192,
    ):
        try:
            import requests                               # noqa: F401
        except ImportError as e:                          # pragma: no cover
            raise RuntimeError(
                "[som] requests is required for OllamaQwenBackend. "
                "Install with: pip install requests"
            ) from e
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        # Temperature 0 is critical for reproducibility — the matching
        # call must be deterministic given the same scene, so the
        # rejection log attributes failures to the matcher / validator,
        # not to sampling jitter.
        self.temperature = temperature
        self.num_ctx = num_ctx

    def call(self, image_rgb, prompt, schema):
        from PIL import Image
        import requests

        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt,
                # Ollama accepts a list of base64 images on the message.
                "images": [b64],
            }],
            "format": schema,                # JSON-schema-constrained decoding
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }

        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=self.timeout_s,
        )
        latency = time.perf_counter() - t0
        if resp.status_code != 200:
            raise RuntimeError(
                f"[som] Ollama returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        body = resp.json()
        raw_text = body.get("message", {}).get("content", "")
        if not raw_text:
            raise RuntimeError(f"[som] Ollama returned empty content: {body}")

        # With format=schema, Ollama guarantees the output is a JSON document
        # matching the schema — but we still parse defensively in case the
        # local Ollama is older and the schema constraint was a no-op.
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"[som] Qwen output is not valid JSON. "
                f"Is your Ollama ≥ 0.5 (schema-constrained format)? "
                f"Raw: {raw_text[:500]}"
            ) from e

        return parsed, raw_text, latency


# ─── A deterministic mock backend for testing ────────────────────────────────
class MockVLMBackend:
    """Pick the closest-COLOR peg / hole pair, deterministically.

    Stand-in for the real Claude/Qwen call during integration testing.
    Lets the runner exercise the full validation + DMP path without an
    API key or GPU VLM. Matching is by color because the benchmark uses
    uniform-diameter pegs and tints each socket its intended peg's color
    (see env.cluttered_scene_cfg). NOT a serious matcher — it ignores
    spatial layout, material, and everything else a real VLM would use.
    """

    def call(self, image_rgb, prompt, schema):
        # Extract object summaries from the prompt (we know the format).
        objects = _parse_object_summary_from_prompt(prompt)
        pegs = [o for o in objects if o["label"].startswith("P")]
        holes = [o for o in objects if o["label"].startswith("H")]

        assignments = []
        used_holes: set[str] = set()
        for peg in pegs:
            candidates = sorted(
                [h for h in holes if h["label"] not in used_holes],
                key=lambda h: _color_dist(h.get("color"), peg.get("color")),
            )
            if not candidates:
                continue
            best = candidates[0]
            alternatives = [
                {"hole_label": c["label"],
                 "confidence": max(0.0, 1.0 - 0.05 * i)}
                for i, c in enumerate(candidates[1:3])
            ]
            assignments.append({
                "peg_label": peg["label"],
                "hole_label": best["label"],
                "confidence": 0.9,
                "reason": "mock backend: closest color match",
                "alternatives": alternatives,
            })
            used_holes.add(best["label"])

        result = {
            "assignments": assignments,
            "unfilled_holes": [h["label"] for h in holes if h["label"] not in used_holes],
            "ungrasped_pegs": [],
        }
        return result, json.dumps(result), 0.0


def _color_dist(a, b) -> float:
    """Euclidean RGB distance; large sentinel when either color is missing."""
    if a is None or b is None:
        return 1e9
    return float(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5)


def _parse_object_summary_from_prompt(prompt: str) -> list[dict]:
    """Extract object table from the prompt (used by MockVLMBackend only)."""
    import re
    objects = []
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("-") and ("diameter" in line):
            # "- P1: diameter=8.2mm, ..., color=(0.85, 0.30, 0.20)" → parse loosely
            try:
                label = line.split(":")[0].strip("- ").strip()
            except (ValueError, IndexError):
                continue
            obj = {"label": label}
            try:
                obj["diameter_mm"] = float(line.split("diameter=")[1].split("mm")[0])
            except (ValueError, IndexError):
                pass
            m = re.search(r"color=\(([^)]*)\)", line)
            if m:
                try:
                    obj["color"] = tuple(float(v) for v in m.group(1).split(","))
                except ValueError:
                    pass
            objects.append(obj)
    return objects


# ─── Prompt + schema ─────────────────────────────────────────────────────────
MATCHING_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "description": "One entry per peg that should be inserted.",
            "items": {
                "type": "object",
                "properties": {
                    "peg_label": {"type": "string"},
                    "hole_label": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                    "alternatives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "hole_label": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["hole_label", "confidence"],
                        },
                    },
                },
                "required": ["peg_label", "hole_label", "confidence", "reason"],
            },
        },
        "unfilled_holes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Holes that no peg should fill (state explicitly).",
        },
        "ungrasped_pegs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Pegs that are not part of any insertion.",
        },
    },
    "required": ["assignments", "unfilled_holes", "ungrasped_pegs"],
}


PROMPT_TEMPLATE = """You are given a top-down view of a robot workbench with several pegs
(warm tint, labeled P1, P2, ...) and several insertion holes (cool tint,
labeled H1, H2, ...).

Detected objects (these labels are the ONLY ones you may use):

{object_table}

Task: produce a peg-to-hole assignment. Every peg is the SAME size, so
size is NOT a cue. Each socket has been painted the COLOR of the peg it
is meant to receive. Use these rules in order:

1. Match each peg to the hole whose COLOR matches the peg's color
   (e.g. the red peg goes into the red socket). Use the color values in
   the table and the tints in the image together.
2. If two holes are similarly colored, use spatial layout and any other
   visual cue to break the tie.
3. If a peg has no plausible hole, list it under "ungrasped_pegs".
   If a hole has no plausible peg, list it under "unfilled_holes".
   Do NOT force assignments.
4. For each assignment, provide up to two alternative holes ranked by
   plausibility, with their own confidence. The robot's validation
   step will use these if your top pick fails its checks.

Output ONLY through the assign_pegs_to_holes tool — no prose."""


def build_prompt(objects: list[DetectedObject], id_to_mark: dict[str, int]) -> str:
    """Format the object table that goes into the VLM prompt."""
    rows = []
    for obj in objects:
        mark = id_to_mark[obj.object_id]
        color_str = (
            f"({obj.color_rgb[0]:.2f}, {obj.color_rgb[1]:.2f}, {obj.color_rgb[2]:.2f})"
            if obj.color_rgb else "?"
        )
        diameter_mm = obj.diameter_m * 1000.0
        height_str = (f", height={obj.height_m * 1000:.1f}mm"
                      if obj.height_m is not None else "")
        rows.append(
            f"  - {mark}: diameter={diameter_mm:.1f}mm{height_str}, color={color_str}"
        )
    return PROMPT_TEMPLATE.format(object_table="\n".join(rows))


# ─── Top-level matching call ─────────────────────────────────────────────────
def match_pegs_to_holes(
    rgb: np.ndarray,
    objects: list[DetectedObject],
    backend: VLMBackend,
    som_image_out: str | None = None,
) -> MatchingResult:
    """Run the SoM-prompted VLM call and parse the structured response.

    This is the SINGLE call we make per episode. No iterative dialogue,
    no follow-up clarifications — those would multiply latency and
    cost while opening the door to inconsistency between turns.
    """
    annotated, id_to_mark = annotate_set_of_mark(rgb, objects)
    if som_image_out is not None:
        try:
            from PIL import Image
            Image.fromarray(annotated).save(som_image_out)
        except Exception:                                  # pragma: no cover
            pass

    prompt = build_prompt(objects, id_to_mark)
    parsed, raw, latency = backend.call(annotated, prompt, MATCHING_SCHEMA)

    # Invert id_to_mark for parsing the response.
    mark_to_id = {v: k for k, v in id_to_mark.items()}

    assignment: dict[str, str] = {}
    confidence: dict[str, float] = {}
    top_k: dict[str, list[tuple[str, float, str]]] = {}

    for entry in parsed.get("assignments", []):
        peg_mark = entry.get("peg_label")
        hole_mark = entry.get("hole_label")
        # Existence checks — the VLM may hallucinate marks. Drop those.
        if peg_mark not in mark_to_id or hole_mark not in mark_to_id:
            continue
        peg_id = mark_to_id[peg_mark]
        hole_id = mark_to_id[hole_mark]
        if not (peg_id.startswith("peg") and hole_id.startswith("hole")):
            continue                                       # role-swap hallucination
        assignment[peg_id] = hole_id
        confidence[peg_id] = float(entry.get("confidence", 0.0))
        alts: list[tuple[str, float, str]] = []
        for alt in entry.get("alternatives", []) or []:
            alt_mark = alt.get("hole_label")
            if alt_mark in mark_to_id and mark_to_id[alt_mark].startswith("hole"):
                alts.append((
                    mark_to_id[alt_mark],
                    float(alt.get("confidence", 0.0)),
                    entry.get("reason", ""),
                ))
        top_k[peg_id] = alts

    # Unmatched lists, also routed through mark_to_id with existence checks.
    unfilled = [mark_to_id[m] for m in parsed.get("unfilled_holes", [])
                if m in mark_to_id and mark_to_id[m].startswith("hole")]
    ungrasped = [mark_to_id[m] for m in parsed.get("ungrasped_pegs", [])
                 if m in mark_to_id and mark_to_id[m].startswith("peg")]

    return MatchingResult(
        assignment=assignment,
        confidence=confidence,
        top_k=top_k,
        unfilled_holes=unfilled,
        ungrasped_pegs=ungrasped,
        raw_vlm_response=raw,
        vlm_latency_s=latency,
    )
