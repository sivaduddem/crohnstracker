#!/usr/bin/env python3
"""Preprocess Kvasir-Capsule into a flat manifest with remapped labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

from ml.data.label_map import is_target, map_label

REPO = Path(__file__).resolve().parents[2]


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield p


def infer_video_id(path: Path) -> str:
    # Prefer parent folder name as video/sequence id to support video-level splits
    return path.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=REPO / "data" / "raw" / "kvasir-capsule")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--include-other", action="store_true")
    args = parser.parse_args()

    img_out = args.out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    rows = []

    if not args.raw.exists():
        print(f"Raw dataset missing at {args.raw}. Run download_kvasir_capsule.py first.")
        # Write empty manifest so pipeline remains importable
        (args.out / "manifest.json").write_text(json.dumps({"cases": []}, indent=2), encoding="utf-8")
        return

    for i, src in enumerate(sorted(iter_images(args.raw))):
        # Label from parent folder or filename prefix
        raw_label = src.parent.name
        label = map_label(raw_label)
        if not args.include_other and not is_target(label):
            continue
        case_id = f"kv_{i:06d}"
        dst = img_out / f"{case_id}.jpg"
        try:
            with Image.open(src) as im:
                im = im.convert("RGB").resize((args.size, args.size))
                im.save(dst, quality=90)
        except OSError:
            continue
        rows.append(
            {
                "id": case_id,
                "source": "kvasir-capsule",
                "video_id": infer_video_id(src),
                "frame_idx": i,
                "path": str(dst.relative_to(args.out)).replace("\\", "/"),
                "gt_label": label,
                "raw_label": raw_label,
                "severity": None,
            }
        )

    (args.out / "manifest.json").write_text(json.dumps({"cases": rows}, indent=2), encoding="utf-8")
    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {len(rows)} processed cases to {args.out}")


if __name__ == "__main__":
    main()
