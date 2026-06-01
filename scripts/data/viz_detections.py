"""Standalone detection visualizer — no Isaac Sim needed.

Runs a detector backend on a saved RGB frame and writes an annotated PNG
(bounding boxes + mask overlays + class/score labels) so you can eyeball
detection quality offline. This is the analysis counterpart to the
in-sim --debug-perception dump: it answers "what did the detector actually
fire on, and how big is each box/mask?" without booting the simulator.

Usage::

    python scripts/data/viz_detections.py --image debug_rgb.png \
        --detector owlv2 --out reports/cluttered/debug/owlv2_annotated.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Distinct colors per canonical class (RGB).
CLASS_COLOR = {"peg": (40, 200, 80), "hole": (230, 60, 60), "unknown": (200, 200, 40)}


def build_detector(name: str):
    if name == "owlv2":
        from compliant_insertion.perception.detection import OWLv2SAM2Detector
        return OWLv2SAM2Detector()
    if name == "grounding_dino":
        from compliant_insertion.perception.detection import GroundedSAM2Detector
        return GroundedSAM2Detector()
    if name == "locate_anything":
        from compliant_insertion.perception.detection import LocateAnythingSAM2Detector
        return LocateAnythingSAM2Detector()
    raise SystemExit(f"unknown detector backend: {name}")


def annotate(rgb: np.ndarray, detections, out_path: Path) -> None:
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, d in enumerate(sorted(detections, key=lambda r: -r.score)):
        color = CLASS_COLOR.get(d.canonical_label, (255, 255, 255))
        # Mask tint.
        if d.mask is not None and d.mask.any():
            tint = np.zeros((*d.mask.shape, 4), dtype=np.uint8)
            tint[d.mask] = (*color, 110)
            overlay.alpha_composite(Image.fromarray(tint, "RGBA"))
        # Box + label.
        x0, y0, x1, y1 = d.bbox_xyxy
        draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=2)
        area = int(d.mask.sum()) if d.mask is not None else 0
        label = f"#{i} {d.canonical_label} {d.score:.2f} ({area}px)"
        ty = max(0, y0 - 16)
        draw.rectangle([x0, ty, x0 + 8 * len(label), ty + 14], fill=(*color, 220))
        draw.text((x0 + 2, ty), label, fill=(0, 0, 0, 255), font=font)

    out = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"[viz] wrote {out_path}  ({len(detections)} detections)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--detector", default="owlv2",
                    choices=["owlv2", "grounding_dino", "locate_anything"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    det = build_detector(args.detector)
    detections = det(rgb)
    print(f"[viz] {args.detector} -> {len(detections)} detections on {args.image}")
    for i, d in enumerate(sorted(detections, key=lambda r: -r.score)):
        x0, y0, x1, y1 = d.bbox_xyxy
        print(f"  #{i} {d.canonical_label:<4s} score={d.score:.3f} "
              f"box=({x0},{y0},{x1},{y1}) {x1-x0}x{y1-y0}px "
              f"mask={int(d.mask.sum())}px")
    annotate(rgb, detections, args.out)


if __name__ == "__main__":
    main()
