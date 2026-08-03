#!/usr/bin/env python3
"""Build FAISS IndexFlatIP over L2-normalized embeddings (cosine similarity)."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=REPO / "artifacts")
    args = parser.parse_args()

    emb_path = args.artifacts / "embeddings.npy"
    if not emb_path.exists():
        raise SystemExit("Missing embeddings.npy — run extract_embeddings.py first")

    xb = np.load(emb_path).astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.write_index(index, str(args.artifacts / "index.faiss"))
    print(f"Wrote FAISS index with {index.ntotal} vectors -> {args.artifacts / 'index.faiss'}")


if __name__ == "__main__":
    main()
