#!/usr/bin/env python3
"""Evaluate classification metrics on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from crohnstracker_shared.constants import TARGET_LABELS
from ml.explain.gradcam import load_model
from ml.train.dataset import LesionDataset

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--splits", type=Path, default=REPO / "data" / "processed" / "splits.json")
    parser.add_argument("--ckpt", type=Path, default=REPO / "artifacts" / "model.pt")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "cls_metrics.json")
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    ds = LesionDataset(args.processed, splits["test"], train=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    ys, preds = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            ys.extend(y.tolist())

    labels = list(range(len(TARGET_LABELS)))
    metrics = {
        "accuracy": float(accuracy_score(ys, preds)) if ys else 0.0,
        "macro_f1": float(f1_score(ys, preds, average="macro", labels=labels, zero_division=0)) if ys else 0.0,
        "per_class_f1": {
            TARGET_LABELS[i]: float(f1_score(ys, preds, average=None, labels=labels, zero_division=0)[i])
            for i in labels
        }
        if ys
        else {k: 0.0 for k in TARGET_LABELS},
        "confusion_matrix": confusion_matrix(ys, preds, labels=labels).tolist() if ys else [[0] * 4 for _ in range(4)],
        "labels": list(TARGET_LABELS),
    }
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
