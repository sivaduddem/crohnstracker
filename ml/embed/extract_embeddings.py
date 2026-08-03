#!/usr/bin/env python3
"""Extract penultimate-layer embeddings for all processed images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from ml.explain.gradcam import load_model

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--ckpt", type=Path, default=REPO / "artifacts" / "model.pt")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts")
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Missing checkpoint: {args.ckpt}")

    manifest = json.loads((args.processed / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ids = []
    vecs = []
    with torch.no_grad():
        for c in tqdm(cases, desc="embed"):
            path = args.processed / c["path"]
            if not path.exists():
                continue
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            feat = model.forward_features(x).squeeze(0).cpu().numpy().astype(np.float32)
            feat /= np.linalg.norm(feat) + 1e-8
            ids.append(c["id"])
            vecs.append(feat)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embeddings.npy", np.stack(vecs) if vecs else np.zeros((0, 1)))
    np.save(args.out / "ids.npy", np.array(ids, dtype=object))
    meta = [{k: c[k] for k in ("id", "gt_label", "video_id", "path")} for c in cases if c["id"] in ids]
    (args.out / "embed_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(ids)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
