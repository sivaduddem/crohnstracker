#!/usr/bin/env python3
"""Generate clearly labeled DEMO placeholder images and mock metadata.

These are NOT medical images and NOT training data. They exist so the app is
runnable before a real dataset/model is available. UI/API always mark outputs
as mock/demo.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "data" / "demo"
IMAGES = DEMO_DIR / "images"
OVERLAYS = DEMO_DIR / "overlays"

LABELS = ["normal", "erythema", "erosion", "ulcer"]
COLORS = {
    "normal": (46, 125, 110),
    "erythema": (176, 74, 74),
    "erosion": (196, 120, 60),
    "ulcer": (120, 52, 90),
}


def _font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_placeholder(path: Path, label: str, case_id: str, kind: str = "image") -> None:
    color = COLORS[label]
    img = Image.new("RGB", (320, 240), color)
    draw = ImageDraw.Draw(img)
    # Non-anatomic abstract pattern (stripes) — not a synthetic endoscopy photo
    for i in range(0, 320, 16):
        shade = tuple(min(255, c + (i % 32)) for c in color)
        draw.rectangle([i, 0, i + 8, 240], fill=shade)
    if kind == "overlay":
        # Fake heatmap blob
        for r in range(80, 10, -8):
            alpha = 40 + r
            blob = Image.new("RGBA", img.size, (0, 0, 0, 0))
            bd = ImageDraw.Draw(blob)
            bd.ellipse([120 - r, 90 - r, 200 + r // 2, 170 + r // 2], fill=(255, 80, 0, min(180, alpha)))
            img = Image.alpha_composite(img.convert("RGBA"), blob).convert("RGB")
            draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 319, 28], fill=(20, 20, 20))
        draw.text((8, 6), "MOCK Grad-CAM — NOT real explanation", fill=(255, 220, 120), font=_font(14))
    else:
        draw.rectangle([0, 0, 319, 52], fill=(15, 15, 15))
        draw.text((8, 6), "DEMO PLACEHOLDER — NOT a medical image", fill=(255, 220, 120), font=_font(14))
        draw.text((8, 28), f"{case_id} | {label}", fill=(240, 240, 240), font=_font(16))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    OVERLAYS.mkdir(parents=True, exist_ok=True)

    cases = []
    projection_umap = []
    projection_tsne = []
    projection_pca = []
    events_by_seq: dict[str, list[str]] = {}

    idx = 0
    for label_i, label in enumerate(LABELS):
        for j in range(3):
            case_id = f"demo_{label}_{j+1:02d}"
            video_id = f"demo_video_{label}"
            fname = f"{case_id}.png"
            rel = f"images/{fname}"
            make_placeholder(IMAGES / fname, label, case_id, kind="image")
            make_placeholder(OVERLAYS / f"{case_id}_gradcam.png", label, case_id, kind="overlay")

            # Introduce one intentional mismatch / novelty-ish case
            pred = label
            conf = 0.88 + 0.02 * j
            if case_id == "demo_ulcer_03":
                pred = "erosion"
                conf = 0.51

            severity = ["mild", "moderate", "severe"][j]
            case = {
                "id": case_id,
                "source": "demo_fixture",
                "video_id": video_id,
                "frame_idx": j * 10,
                "path": rel,
                "gt_label": label,
                "severity": severity,
                "split": "demo",
                "pred_label": pred,
                "confidence": round(conf, 3),
            }
            cases.append(case)
            events_by_seq.setdefault(video_id, []).append(case_id)

            angle = (idx / 12) * 2 * math.pi
            r = 1.2 + 0.3 * label_i
            projection_umap.append(
                {
                    "case_id": case_id,
                    "x": round(r * math.cos(angle) + label_i * 0.2, 4),
                    "y": round(r * math.sin(angle) - label_i * 0.1, 4),
                    "gt_label": label,
                    "pred_label": pred,
                    "confidence": round(conf, 3),
                    "severity": severity,
                    "thumbnail_url": f"/media/images/{fname}",
                }
            )
            projection_tsne.append({**projection_umap[-1], "x": round(label_i + j * 0.3, 4), "y": round(j - label_i * 0.2, 4)})
            projection_pca.append({**projection_umap[-1], "x": round(idx * 0.4, 4), "y": round((label_i - 1.5) * 0.8, 4)})
            idx += 1

    events = []
    for video_id, cids in events_by_seq.items():
        label = cids[0].split("_")[1]
        events.append(
            {
                "sequence_id": video_id,
                "start_frame": 0,
                "end_frame": 20,
                "label": label,
                "peak_confidence": 0.92,
                "case_ids": cids,
            }
        )

    metrics = {
        "classification": {
            "accuracy": 0.83,
            "macro_f1": 0.81,
            "per_class_f1": {"normal": 0.90, "erythema": 0.78, "erosion": 0.76, "ulcer": 0.80},
            "confusion_matrix": [
                [3, 0, 0, 0],
                [0, 2, 1, 0],
                [0, 1, 2, 0],
                [0, 0, 1, 2],
            ],
            "labels": LABELS,
        },
        "retrieval": {
            "recall_at_k": 0.75,
            "precision_at_k": 0.68,
            "map": 0.71,
            "ndcg": 0.77,
            "latency_ms": 4.2,
            "embedding_ms": 12.5,
            "k": 5,
        },
    }

    (DEMO_DIR / "cases.json").write_text(
        json.dumps({"embedding_dim": 32, "cases": cases, "note": "DEMO FIXTURES — mock only"}, indent=2),
        encoding="utf-8",
    )
    (DEMO_DIR / "projections.json").write_text(
        json.dumps({"umap": projection_umap, "tsne": projection_tsne, "pca": projection_pca}, indent=2),
        encoding="utf-8",
    )
    (DEMO_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (DEMO_DIR / "sequences.json").write_text(json.dumps({"events": events}, indent=2), encoding="utf-8")
    (DEMO_DIR / "README.md").write_text(
        "# Demo fixtures\n\n"
        "These placeholders are **not** medical images and **not** training data.\n"
        "API/UI must display Demo mode banners. Mock predictions are deterministic.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} demo cases to {DEMO_DIR}")


if __name__ == "__main__":
    main()
