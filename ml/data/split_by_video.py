#!/usr/bin/env python3
"""Create video-level train/val/test splits (no frame leakage across splits)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO / "data" / "processed" / "manifest.json")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "processed" / "splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Missing manifest: {args.manifest}")

    cases = json.loads(args.manifest.read_text(encoding="utf-8"))["cases"]
    by_video: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        by_video[c["video_id"]].append(c["id"])

    videos = sorted(by_video.keys())
    rng = random.Random(args.seed)
    rng.shuffle(videos)

    n = len(videos)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    train_v = set(videos[:n_train])
    val_v = set(videos[n_train : n_train + n_val])
    test_v = set(videos[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    case_split = {}
    for vid, ids in by_video.items():
        split = "train" if vid in train_v else "val" if vid in val_v else "test"
        splits[split].extend(ids)
        for cid in ids:
            case_split[cid] = split

    # Leakage guard: no video in multiple splits
    assert train_v.isdisjoint(val_v) and train_v.isdisjoint(test_v) and val_v.isdisjoint(test_v)

    payload = {
        "seed": args.seed,
        "video_counts": {k: len(v) for k, v in [("train", train_v), ("val", val_v), ("test", test_v)]},
        "case_counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
        "case_split": case_split,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote video-level splits to {args.out}: {payload['case_counts']}")


if __name__ == "__main__":
    main()
