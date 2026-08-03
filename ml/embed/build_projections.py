#!/usr/bin/env python3
"""Precompute UMAP / t-SNE / PCA coordinates for the embedding explorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=REPO / "artifacts")
    args = parser.parse_args()

    emb = np.load(args.artifacts / "embeddings.npy")
    ids = np.load(args.artifacts / "ids.npy", allow_pickle=True)
    meta = {m["id"]: m for m in json.loads((args.artifacts / "embed_meta.json").read_text(encoding="utf-8"))}

    if len(emb) < 3:
        raise SystemExit("Need at least 3 embeddings for projections")

    pca_xy = PCA(n_components=2, random_state=42).fit_transform(emb)
    tsne_xy = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb) - 1)).fit_transform(emb)

    try:
        import umap

        umap_xy = umap.UMAP(n_components=2, random_state=42).fit_transform(emb)
    except Exception:
        umap_xy = pca_xy

    def pack(xy):
        points = []
        for i, cid in enumerate(ids):
            m = meta.get(str(cid), {})
            points.append(
                {
                    "case_id": str(cid),
                    "x": float(xy[i, 0]),
                    "y": float(xy[i, 1]),
                    "gt_label": m.get("gt_label", "other_unmapped"),
                    "pred_label": m.get("gt_label", "other_unmapped"),
                    "confidence": 0.0,
                    "severity": None,
                    "thumbnail_url": f"/media/processed/{m.get('path', '')}",
                }
            )
        return points

    payload = {"umap": pack(umap_xy), "tsne": pack(tsne_xy), "pca": pack(pca_xy)}
    (args.artifacts / "projections.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote projections for {len(ids)} points")


if __name__ == "__main__":
    main()
