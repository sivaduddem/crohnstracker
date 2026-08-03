#!/usr/bin/env python3
"""Evaluate retrieval quality: Recall@K, Precision@K, mAP, NDCG + latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def dcg(rels: list[int]) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=REPO / "artifacts")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "retrieval_metrics.json")
    args = parser.parse_args()

    emb = np.load(args.artifacts / "embeddings.npy").astype(np.float32)
    ids = np.load(args.artifacts / "ids.npy", allow_pickle=True)
    meta = {m["id"]: m for m in json.loads((args.artifacts / "embed_meta.json").read_text(encoding="utf-8"))}
    index = faiss.read_index(str(args.artifacts / "index.faiss"))

    faiss.normalize_L2(emb)
    recalls, precisions, aps, ndcgs = [], [], [], []
    t0 = time.perf_counter()
    for i, cid in enumerate(ids):
        q = emb[i : i + 1]
        sims, idxs = index.search(q, args.k + 1)
        neigh = [int(j) for j in idxs[0] if int(j) != i][: args.k]
        gt = meta[str(cid)]["gt_label"]
        rels = [1 if meta[str(ids[j])]["gt_label"] == gt else 0 for j in neigh]
        precisions.append(sum(rels) / args.k)
        recalls.append(1.0 if sum(rels) > 0 else 0.0)
        # AP
        hits, ap = 0, 0.0
        for rank, r in enumerate(rels, start=1):
            if r:
                hits += 1
                ap += hits / rank
        aps.append(ap / max(hits, 1) if hits else 0.0)
        ideal = sorted(rels, reverse=True)
        ndcgs.append((dcg(rels) / dcg(ideal)) if sum(ideal) else 0.0)

    latency_ms = (time.perf_counter() - t0) * 1000 / max(len(ids), 1)
    metrics = {
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "map": float(np.mean(aps)) if aps else 0.0,
        "ndcg": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "latency_ms": float(latency_ms),
        "embedding_ms": None,
        "k": args.k,
    }
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
