"""Open-vocabulary detection + segmentation via GroundingDINO + SAM 2.

This is the OFF-THE-SHELF half of the pipeline. We do not reinvent
detection or segmentation. The contributions here are:

  1. The text prompts that turn an open-vocabulary detector into a
     domain-specific "pegs and holes" detector without retraining. We
     use disjunctive prompts ("metal peg . cylindrical peg . wooden
     peg" etc.) because GroundingDINO's recall is sensitive to lexical
     framing.
  2. A normalization step that maps the detector's free-form text label
     onto our two canonical classes {"peg", "hole"}.
  3. A quality filter that drops detections whose score, area, or
     aspect ratio fall outside reasonable bounds (catches the common
     GroundingDINO failure mode of latching onto the whole table).

The actual model loading is lazy and behind a flag, because Dominik
will most often run with the GroundTruthDetector backend during
development (no 8 GB of weights to manage) and only flip to the real
detector for end-to-end VLM evaluation.

Dependencies (only when ``use_models=True``):
    pip install transformers torch  # for GroundingDINO
    pip install sam2                 # SAM 2 with checkpoint downloads
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# ─── Detection-stage data structures ─────────────────────────────────────────
@dataclass
class RawDetection:
    """One pre-pose detection: 2D mask + text label.

    This is the output of detection alone — no 3D pose yet. The pose
    estimation stage consumes a list of these.
    """

    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray                # [H, W] bool
    score: float                    # detector confidence
    raw_label: str                  # detector's raw text output ("metal peg")
    canonical_label: str            # "peg" | "hole" | "unknown"


class Detector(Protocol):
    """A pluggable detector. Anything implementing this is a valid backend."""

    def __call__(self, rgb: np.ndarray) -> list[RawDetection]: ...


# ─── Prompt engineering: the only "custom" part of detection ─────────────────
# GroundingDINO conditions on a single text string with phrases separated by
# periods. Recall is sensitive to lexical framing, so we use disjunctive
# prompts that span color / material / shape descriptors. Empirically this
# nearly doubles recall on cluttered tabletop scenes versus "peg . hole".
#
# Keep the two prompts separate and run two detection passes: combining
# everything into one prompt makes GroundingDINO produce mixed labels that
# the canonicalization step has to disambiguate by box geometry, which is
# fragile.
PEG_PROMPT = (
    "metal peg . cylindrical peg . wooden peg . plastic peg . "
    "small cylinder . vertical rod . upright pin . dowel"
)
HOLE_PROMPT = (
    "hole . socket . round hole . circular opening . insertion hole . "
    "drilled hole . fixture with a hole"
)

# Filters that catch GroundingDINO's most common failure: latching onto
# the table or the robot arm rather than the small objects on top of it.
MIN_SCORE = 0.25                # detector confidence floor
MIN_AREA_FRAC = 1e-4            # min mask area as a fraction of image
MAX_AREA_FRAC = 0.10            # max mask area; bigger = probably the table
MAX_ASPECT = 6.0                # pegs are vertical, but very-tall ASPECT > 6
                                # almost always means the detector grabbed
                                # a vertical surface or the robot column


def canonicalize_label(raw_label: str) -> str:
    """Map a free-form detector label onto {"peg", "hole", "unknown"}.

    Pure string heuristic — it does not need to be smart, because the
    real disambiguation happens at the VLM stage via Set-of-Mark
    prompting. The point here is just to route the detection into the
    correct downstream primitive-fitter (cylinder vs. circle).
    """
    s = raw_label.lower()
    if any(k in s for k in ("hole", "socket", "opening", "drilled", "fixture")):
        return "hole"
    if any(k in s for k in ("peg", "cylinder", "rod", "pin", "dowel")):
        return "peg"
    return "unknown"


def filter_detection(
    det: RawDetection, image_area: int, min_score: float = MIN_SCORE,
) -> bool:
    """Return True iff this detection passes the sanity filter.

    ``min_score`` is overridable because different detectors put their
    confidence on different scales — GroundingDINO sits around 0.25+,
    whereas OWLv2's open-vocabulary logits run noticeably lower, so a
    shared 0.25 floor would silently drop every OWLv2 box.
    """
    if det.score < min_score:
        return False
    area = int(det.mask.sum())
    if area < MIN_AREA_FRAC * image_area:
        return False
    if area > MAX_AREA_FRAC * image_area:
        return False
    x0, y0, x1, y1 = det.bbox_xyxy
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    if max(w, h) / min(w, h) > MAX_ASPECT:
        return False
    if det.canonical_label == "unknown":
        return False
    return True


# ─── GroundingDINO + SAM 2 backend ───────────────────────────────────────────
class GroundedSAM2Detector:
    """Production backend: GroundingDINO for boxes, SAM 2 for masks.

    Lazily instantiates the models on first call so importing this
    module is cheap (matters because the runner imports it before
    Isaac Sim starts).
    """

    def __init__(self, device: str = "cuda",
                 gdino_model: str = "IDEA-Research/grounding-dino-base",
                 sam2_checkpoint: str = "facebook/sam2-hiera-large"):
        self.device = device
        self.gdino_model = gdino_model
        self.sam2_checkpoint = sam2_checkpoint
        self._gdino_processor = None
        self._gdino = None
        self._sam2 = None

    def _ensure_loaded(self) -> None:
        if self._gdino is not None:
            return
        # Deferred imports so the rest of the pipeline runs without
        # transformers / sam2 installed.
        from transformers import (
            AutoProcessor, AutoModelForZeroShotObjectDetection,
        )
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._gdino_processor = AutoProcessor.from_pretrained(self.gdino_model)
        self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.gdino_model
        ).to(self.device)
        self._sam2 = SAM2ImagePredictor.from_pretrained(self.sam2_checkpoint)

    def __call__(self, rgb: np.ndarray) -> list[RawDetection]:
        self._ensure_loaded()
        H, W = rgb.shape[:2]
        image_area = H * W
        detections: list[RawDetection] = []

        for prompt in (PEG_PROMPT, HOLE_PROMPT):
            boxes, scores, labels = self._run_gdino(rgb, prompt)
            if len(boxes) == 0:
                continue
            masks = self._run_sam2(rgb, boxes)
            for box, score, label, mask in zip(boxes, scores, labels, masks):
                det = RawDetection(
                    bbox_xyxy=tuple(int(x) for x in box),     # type: ignore[arg-type]
                    mask=mask.astype(bool),
                    score=float(score),
                    raw_label=label,
                    canonical_label=canonicalize_label(label),
                )
                if filter_detection(det, image_area):
                    detections.append(det)

        return detections

    def _run_gdino(self, rgb, prompt):
        import torch
        inputs = self._gdino_processor(
            images=rgb, text=prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._gdino(**inputs)
        results = self._gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=MIN_SCORE,
            text_threshold=MIN_SCORE,
            target_sizes=[rgb.shape[:2]],
        )[0]
        return (
            results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            results["labels"],
        )

    def _run_sam2(self, rgb, boxes):
        self._sam2.set_image(rgb)
        masks, _, _ = self._sam2.predict(box=boxes, multimask_output=False)
        # SAM 2 returns [N, 1, H, W] for boxed prompts.
        if masks.ndim == 4:
            masks = masks[:, 0]
        return masks


# ─── OWLv2 + SAM 2 backend ───────────────────────────────────────────────────
# OWLv2 is an open-vocabulary detector that conditions on a LIST of short
# text queries (not GroundingDINO's period-separated soup, and not
# LocateAnything's natural-language sentences). It returns boxes + a label
# index into the query list in a SINGLE forward pass — no autoregressive
# generation — so it is far cheaper in both VRAM (~0.6–1.5 GB vs
# LocateAnything's ~7.8 GB) and latency than the grounding-VLM path.
#
# We pass peg-queries and hole-queries in one combined list and remember
# which canonical class each query came from, so a single OWLv2 call labels
# both classes at once (the image is encoded once regardless of query count).
OWLV2_PEG_QUERIES = [
    "a metal peg", "a cylindrical peg", "an upright dowel pin",
    "a small vertical cylinder", "a wooden peg",
]
OWLV2_HOLE_QUERIES = [
    "a hole", "a socket", "a round hole", "a circular opening",
    "an insertion hole",
]

# OWLv2's open-vocabulary scores live on a lower scale than GroundingDINO's.
# Used both as the post-processing threshold and as the floor handed to
# filter_detection. Calibrated on a real workbench frame (debug_rgb.png):
# every box >=0.30 was a clean, tight peg/hole, while everything below was a
# large-area table/robot false positive that slips past the area filter.
# Lower it if recall drops on a different camera / lighting; raise it if junk
# reappears.
OWLV2_SCORE_THRESHOLD = 0.30

# Fraction of image HEIGHT to crop off the TOP before detection. The workbench
# camera frames the robot base / mount in the top ~15-20% of the image, and
# OWLv2 fires spurious "hole" boxes on its dark recesses (they look hole-like).
# Cropping to the table-only region removes those false positives at the
# source; detection coordinates are remapped back to the full frame afterwards,
# so the depth back-projection (which uses the full-resolution depth + K) is
# unaffected. Set to 0.0 to disable.
OWLV2_CROP_TOP_FRAC = 0.18


class OWLv2SAM2Detector:
    """Open-vocabulary boxes via OWLv2, masks via SAM 2.

    Drop-in alternative to GroundedSAM2Detector / LocateAnythingSAM2Detector
    for the cluttered tabletop scene. Chosen when GroundingDINO's lexical
    prior misses small / atypical objects but the heavier LocateAnything-3B
    won't co-reside with Isaac Sim + the matcher VLM on a 16 GB card.

    Single OWLv2 forward pass over a combined peg+hole query list, then one
    SAM 2 pass over all returned boxes (``set_image`` is the expensive step,
    so we batch peg + hole boxes together). Lazily instantiates the models on
    first call so importing this module stays cheap.

    Deps (only when used):
        pip install transformers torch   # for OWLv2
        pip install sam2                 # SAM 2 (small checkpoint by default)
    """

    def __init__(self, device: str = "cuda",
                 owlv2_model: str = "google/owlv2-base-patch16-ensemble",
                 sam2_checkpoint: str = "facebook/sam2-hiera-small",
                 score_threshold: float = OWLV2_SCORE_THRESHOLD,
                 crop_top_frac: float = OWLV2_CROP_TOP_FRAC,
                 dtype=None):
        self.device = device
        self.owlv2_model_id = owlv2_model
        self.sam2_checkpoint = sam2_checkpoint
        self.score_threshold = score_threshold
        self.crop_top_frac = float(crop_top_frac)
        self._dtype_override = dtype
        self._owlv2_processor = None
        self._owlv2 = None
        self._sam2 = None

        # Combined query list + a parallel canonical-label lookup, built once.
        self._queries = list(OWLV2_PEG_QUERIES) + list(OWLV2_HOLE_QUERIES)
        self._query_canonical = (
            ["peg"] * len(OWLV2_PEG_QUERIES)
            + ["hole"] * len(OWLV2_HOLE_QUERIES)
        )

    def _ensure_loaded(self) -> None:
        if self._owlv2 is not None:
            return
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        dtype = (self._dtype_override if self._dtype_override is not None
                 else torch.float32)
        self._dtype = dtype
        self._owlv2_processor = Owlv2Processor.from_pretrained(self.owlv2_model_id)
        self._owlv2 = Owlv2ForObjectDetection.from_pretrained(
            self.owlv2_model_id, torch_dtype=dtype,
        ).to(self.device).eval()
        self._sam2 = SAM2ImagePredictor.from_pretrained(self.sam2_checkpoint)

    def __call__(self, rgb: np.ndarray) -> list[RawDetection]:
        self._ensure_loaded()
        H, W = rgb.shape[:2]
        image_area = H * W

        # Crop the top band (robot base / mount) out of the detector's view,
        # then run detection on the table-only ROI. y_off is the row in the
        # FULL image where the ROI starts; everything below is remapped back.
        y_off = int(round(self.crop_top_frac * H))
        y_off = max(0, min(y_off, H - 1))
        rgb_roi = rgb[y_off:, :, :] if y_off > 0 else rgb

        boxes, scores, canon_labels = self._run_owlv2(rgb_roi)
        if len(boxes) == 0:
            return []

        masks = self._run_sam2(rgb_roi, boxes)

        detections: list[RawDetection] = []
        for box, score, canon, mask in zip(boxes, scores, canon_labels, masks):
            x0, y0, x1, y1 = (int(v) for v in box)
            if x1 <= x0 or y1 <= y0:
                continue
            # Remap ROI-local coordinates back to the full frame so the box
            # and mask align with the full-resolution depth + intrinsics that
            # pose_estimation.back_project uses.
            y0 += y_off
            y1 += y_off
            mask = mask.astype(bool)
            if y_off > 0:
                full_mask = np.zeros((H, W), dtype=bool)
                full_mask[y_off:, :] = mask
                mask = full_mask
            det = RawDetection(
                bbox_xyxy=(x0, y0, x1, y1),
                mask=mask,
                score=float(score),
                raw_label=f"owlv2:{canon}",
                canonical_label=canon,
            )
            if filter_detection(det, image_area, min_score=self.score_threshold):
                detections.append(det)

        return detections

    def _run_owlv2(self, rgb):
        import torch
        from PIL import Image

        pil = Image.fromarray(rgb if rgb.dtype == np.uint8
                              else rgb.astype(np.uint8))
        # OWLv2 wants text as a list (per image) of query lists.
        inputs = self._owlv2_processor(
            text=[self._queries], images=pil, return_tensors="pt",
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        with torch.no_grad():
            outputs = self._owlv2(**inputs)
        # post_process expects target_sizes as (height, width). We use the
        # "grounded" post-processor (the plain post_process_object_detection
        # is deprecated for OWLv2 since transformers 4.52); it returns the
        # same scores/labels/boxes keys, with "labels" still the query index.
        target_sizes = torch.tensor([(rgb.shape[0], rgb.shape[1])],
                                    device=self.device)
        results = self._owlv2_processor.post_process_grounded_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=target_sizes,
        )[0]
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        label_idx = results["labels"].detach().cpu().numpy()
        canon_labels = [self._query_canonical[int(i)] for i in label_idx]
        return boxes, scores, canon_labels

    def _run_sam2(self, rgb, boxes_xyxy):
        """Run SAM 2 with box prompts to get per-detection masks.

        Identical contract to the other detector backends' SAM 2 call. Kept
        inline rather than extracted to a shared helper so the detector
        classes remain self-contained.
        """
        self._sam2.set_image(rgb)
        masks, _, _ = self._sam2.predict(
            box=np.asarray(boxes_xyxy, dtype=np.float32), multimask_output=False,
        )
        # SAM 2 returns [N, 1, H, W] for boxed prompts.
        if masks.ndim == 4:
            masks = masks[:, 0]
        return masks


# ─── LocateAnything-3B + SAM 2 backend ───────────────────────────────────────
# Prompts for LocateAnything are natural-language descriptions, not the
# period-separated keyword soup GroundingDINO needs. LocateAnything was
# trained on dense / cluttered scenes and accepts color and spatial cues
# in the description, so we describe the objects as they actually appear
# in the rendered scene rather than relying on the lexical prior for
# generic "peg" and "hole" terms.
#
# Tune these on your own SOM frames; the model is sensitive to phrasing,
# and the "right" description depends on peg/hole materials, lighting,
# and the angle the perception camera is mounted at. Start descriptive,
# then trim once you see what's firing.
LA_PEG_PROMPT = "small colorful cylindrical peg standing upright on the table"
LA_HOLE_PROMPT = "small dark square hole on the top face of a purple block"

# LocateAnything emits boxes but no per-detection confidence. We assign
# a constant pseudo-score so the downstream filter still has a value to
# threshold on; the real per-box quality signal lives in the
# generation_mode (hybrid/slow) and in whether SAM 2 returns a sensible
# mask. Don't read too much into this number — it's not a real score.
LA_PSEUDO_SCORE = 0.80


class LocateAnythingSAM2Detector:
    """NVIDIA LocateAnything-3B for boxes, SAM 2 for masks.

    Alternative to GroundedSAM2Detector for cluttered tabletop scenes
    where GroundingDINO's lexical priors don't fire on small or
    visually-atypical objects (short stub pegs from a 35°-tilt
    workbench camera, square holes that look like dark dots at native
    resolution, etc).

    LocateAnything is a 3B Qwen2.5-VL-based grounding VLM. It returns
    bounding boxes as quantized text tokens (`<box><x1><y1><x2><y2></box>`,
    coords in [0, 1000]) — no masks, no per-box confidence scores. We
    keep SAM 2 in the pipeline to recover masks (driven by the boxes),
    which keeps the downstream pose-fitting interface identical to the
    GroundingDINO path.

    License note: LocateAnything-3B is under NVIDIA's research-only
    license. Fine for the thesis and benchmark figures; can't be ported
    into industry deployment without a re-license. The
    GroundedSAM2Detector remains available as the permissively-licensed
    baseline.

    Deps (only when used):
        pip install transformers>=4.57 opencv-python-headless decord lmdb peft
        pip install sam2
    """

    # Coordinate quantization grid LocateAnything emits its boxes on. Per
    # the model card, all `<box>` tokens are integers in [0, 1000] on the
    # original input image. We divide by this when scaling to pixels.
    COORD_GRID = 1000

    def __init__(self, device: str = "cuda",
                 la_model: str = "nvidia/LocateAnything-3B",
                 sam2_checkpoint: str = "facebook/sam2-hiera-large",
                 generation_mode: str = "hybrid",
                 max_new_tokens: int = 2048,
                 dtype=None):
        # Default dtype is bf16, matching the model card's inference recipe.
        # Resolved at load time to avoid importing torch at module import.
        self.device = device
        self.la_model_id = la_model
        self.sam2_checkpoint = sam2_checkpoint
        self.generation_mode = generation_mode
        self.max_new_tokens = max_new_tokens
        self._dtype_override = dtype

        self._la_tokenizer = None
        self._la_processor = None
        self._la_model = None
        self._sam2 = None

        # Stash the raw model output strings from the last call so the
        # runner / debug-perception layer can print them for prompt
        # iteration. Keyed by canonical label ("peg" / "hole").
        self.last_raw_responses: dict[str, str] = {}

    def _ensure_loaded(self) -> None:
        if self._la_model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        dtype = self._dtype_override if self._dtype_override is not None else torch.bfloat16
        self._dtype = dtype

        # trust_remote_code=True is mandatory — LocateAnything ships its
        # own modeling code (Parallel Box Decoding head, custom generate).
        self._la_tokenizer = AutoTokenizer.from_pretrained(
            self.la_model_id, trust_remote_code=True,
        )
        self._la_processor = AutoProcessor.from_pretrained(
            self.la_model_id, trust_remote_code=True,
        )
        self._la_model = AutoModel.from_pretrained(
            self.la_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        self._sam2 = SAM2ImagePredictor.from_pretrained(self.sam2_checkpoint)

    def __call__(self, rgb: np.ndarray) -> list[RawDetection]:
        self._ensure_loaded()
        H, W = rgb.shape[:2]
        image_area = H * W

        # Two grounding calls — one per canonical class — mirroring the
        # GroundingDINO two-prompt approach. Single combined prompts work
        # but the per-class signal is easier to debug and the cost is one
        # extra VLM forward pass.
        prompts: dict[str, str] = {"peg": LA_PEG_PROMPT, "hole": LA_HOLE_PROMPT}
        all_boxes_xyxy: list[tuple[int, int, int, int]] = []
        all_labels: list[str] = []
        self.last_raw_responses = {}

        for canonical_label, phrase in prompts.items():
            answer = self._run_locate_anything(rgb, phrase)
            self.last_raw_responses[canonical_label] = answer
            boxes_px = self._parse_boxes(answer, W, H)
            if not boxes_px:
                print(f"[locate_anything] no boxes for '{canonical_label}' "
                      f"prompt: {phrase!r}")
                continue
            all_boxes_xyxy.extend(boxes_px)
            all_labels.extend([canonical_label] * len(boxes_px))

        if not all_boxes_xyxy:
            return []

        # One SAM 2 pass over all boxes (peg + hole together) — cheaper
        # than two passes because set_image is the expensive step.
        masks = self._run_sam2(rgb, np.asarray(all_boxes_xyxy, dtype=np.float32))

        detections: list[RawDetection] = []
        for (x0, y0, x1, y1), canonical_label, mask in zip(
                all_boxes_xyxy, all_labels, masks):
            det = RawDetection(
                bbox_xyxy=(int(x0), int(y0), int(x1), int(y1)),
                mask=mask.astype(bool),
                score=LA_PSEUDO_SCORE,
                raw_label=f"locate_anything:{canonical_label}",
                canonical_label=canonical_label,
            )
            if filter_detection(det, image_area):
                detections.append(det)

        return detections

    def _run_locate_anything(self, rgb: np.ndarray, phrase: str) -> str:
        """One LocateAnything forward pass for a single description.

        Uses the model-card-recommended `ground_multi` prompt template
        ("Locate all the instances that match the following description: ...")
        which is intended for multi-instance phrase grounding — the right
        fit for "find all the pegs / find all the holes".
        """
        import torch
        from PIL import Image

        prompt = (
            f"Locate all the instances that match the following description: "
            f"{phrase}."
        )
        # The processor wants a PIL image; we have a numpy uint8 array.
        pil = Image.fromarray(rgb if rgb.dtype == np.uint8
                              else rgb.astype(np.uint8))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }]
        # NOTE: the HF model card uses processor.py_apply_chat_template;
        # this is LocateAnything's custom processor method and is correct
        # for this model (do NOT substitute apply_chat_template).
        text = self._la_processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        images, videos = self._la_processor.process_vision_info(messages)
        inputs = self._la_processor(
            text=[text], images=images, videos=videos, return_tensors="pt",
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self._dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        with torch.no_grad():
            response = self._la_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self._la_tokenizer,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                generation_mode=self.generation_mode,
                # Deterministic-ish decoding for detector use. Sampling
                # (do_sample=True) is for chat; for grounding we want
                # repeatable boxes across identical frames.
                do_sample=False,
                verbose=False,
            )
        # The custom generate returns either a string or a (str, history,
        # stats) tuple depending on the mode — handle both.
        if isinstance(response, tuple):
            return response[0]
        if isinstance(response, list):
            return response[0] if response else ""
        return response if isinstance(response, str) else str(response)

    def _parse_boxes(
        self, answer: str, W: int, H: int,
    ) -> list[tuple[int, int, int, int]]:
        """Parse `<box><x1><y1><x2><y2></box>` tokens into pixel xyxy."""
        import re
        boxes: list[tuple[int, int, int, int]] = []
        # The "4-token" pattern is the bbox format. There is also a
        # "2-token" point format used for `point` mode; we ignore it
        # since we asked for boxes.
        pattern = r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
        for m in re.finditer(pattern, answer):
            x1q, y1q, x2q, y2q = (int(g) for g in m.groups())
            x0 = int(x1q / self.COORD_GRID * W)
            y0 = int(y1q / self.COORD_GRID * H)
            x1 = int(x2q / self.COORD_GRID * W)
            y1 = int(y2q / self.COORD_GRID * H)
            # Guard against degenerate / inverted boxes the model
            # occasionally emits at the edges of the coord grid.
            if x1 <= x0 or y1 <= y0:
                continue
            x0 = max(0, min(W - 1, x0))
            y0 = max(0, min(H - 1, y0))
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            boxes.append((x0, y0, x1, y1))
        return boxes

    def _run_sam2(self, rgb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Run SAM 2 with box prompts to get per-detection masks.

        Identical to GroundedSAM2Detector's SAM 2 call. Kept inline
        rather than extracted to a shared helper so the two detector
        classes remain self-contained.
        """
        self._sam2.set_image(rgb)
        masks, _, _ = self._sam2.predict(box=boxes_xyxy, multimask_output=False)
        # SAM 2 returns [N, 1, H, W] for boxed prompts.
        if masks.ndim == 4:
            masks = masks[:, 0]
        return masks


# ─── A stub detector for use without model weights ───────────────────────────
class GroundTruthDetector:
    """Bypass detection by reading boxes directly from the simulator.

    Used during DMP-side development so the pipeline runs end-to-end
    without GroundingDINO weights. The runner only flips to
    GroundedSAM2Detector for the VLM evaluation runs.

    The "detections" are constructed from oracle ground-truth poses
    projected back into the image, then a perfectly tight bbox + a
    rasterized disk mask. This still exercises the full downstream
    pipeline including the VLM call, which is what we want during
    integration testing.
    """

    def __init__(self, oracle_objects, intrinsics, T_world_cam):
        self.oracle_objects = oracle_objects  # list of (label, world_pos, diameter, height)
        self.intrinsics = intrinsics
        self.T_world_cam = T_world_cam        # 4x4

    def __call__(self, rgb: np.ndarray) -> list[RawDetection]:
        H, W = rgb.shape[:2]
        detections: list[RawDetection] = []
        T_cam_world = np.linalg.inv(self.T_world_cam)
        K_fxfy = (self.intrinsics.fx, self.intrinsics.fy,
                  self.intrinsics.cx, self.intrinsics.cy)

        for i, (label, pos_w, diameter, height) in enumerate(self.oracle_objects):
            pos_c = (T_cam_world @ np.append(pos_w, 1.0))[:3]
            if pos_c[2] <= 0:
                continue
            u = K_fxfy[0] * pos_c[0] / pos_c[2] + K_fxfy[2]
            v = K_fxfy[1] * pos_c[1] / pos_c[2] + K_fxfy[3]
            # Approximate projected radius. For a peg this is the visible
            # height; for a hole, the diameter. Both work for a stub.
            extent = (height if label == "peg" and height is not None
                      else diameter)
            r_px = K_fxfy[0] * extent * 0.5 / pos_c[2]
            x0, y0 = int(max(0, u - r_px)), int(max(0, v - r_px))
            x1, y1 = int(min(W - 1, u + r_px)), int(min(H - 1, v + r_px))
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((H, W), dtype=bool)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] = (
                (xx - u) ** 2 + (yy - v) ** 2 <= r_px ** 2
            )
            detections.append(RawDetection(
                bbox_xyxy=(x0, y0, x1, y1),
                mask=mask,
                score=0.99,
                raw_label=label,
                canonical_label=label,
            ))
        return detections
