#!/usr/bin/env python3
"""Train EfficientNet-B0 lesion classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from crohnstracker_shared.constants import TARGET_LABELS
from ml.train.dataset import LesionDataset
from ml.train.model import build_model

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--splits", type=Path, default=REPO / "data" / "processed" / "splits.json")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="Tiny run for CI/dev")
    args = parser.parse_args()

    if not args.splits.exists():
        raise SystemExit("Missing splits.json — run split_by_video.py after preprocess.")

    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    train_ids = splits["train"][:32] if args.smoke else splits["train"]
    val_ids = splits["val"][:16] if args.smoke else splits["val"]
    if not train_ids:
        raise SystemExit("No training cases. Preprocess Kvasir-Capsule first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(TARGET_LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    train_ds = LesionDataset(args.processed, train_ids, train=True)
    val_ds = LesionDataset(args.processed, val_ids, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    args.out.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    history = []

    for epoch in range(args.epochs if not args.smoke else 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(y)
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += len(y)
        train_acc = correct / max(total, 1)

        model.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                vcorrect += int((pred == y).sum().item())
                vtotal += len(y)
        val_acc = vcorrect / max(vtotal, 1)
        history.append({"epoch": epoch + 1, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch {epoch+1}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "labels": list(TARGET_LABELS),
                    "arch": "efficientnet_b0",
                },
                args.out / "model.pt",
            )

    (args.out / "label_map.json").write_text(
        json.dumps({"labels": list(TARGET_LABELS)}, indent=2), encoding="utf-8"
    )
    (args.out / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {args.out / 'model.pt'} (best val_acc={best_acc:.3f})")


if __name__ == "__main__":
    main()
