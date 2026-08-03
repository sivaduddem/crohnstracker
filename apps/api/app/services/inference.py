from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from apps.api.app.config import get_settings
from crohnstracker_shared.constants import DISCLAIMER, TARGET_LABELS
from crohnstracker_shared.schemas import (
    AppMode,
    HybridPredictResponse,
    PredictionResult,
    SimilarCase,
    SimilarityAnalysis,
)


class TrainedInferenceService:
    """Real model + FAISS path. Loaded only when artifacts exist."""

    def __init__(self, artifacts: Path) -> None:
        self.artifacts = artifacts
        self.model = None
        self.index = None
        self.ids: list[str] = []
        self.meta: dict[str, dict] = {}
        self.device = None
        self.tf = None
        self._load()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.index is not None

    def _load(self) -> None:
        ckpt = self.artifacts / "model.pt"
        index_path = self.artifacts / "index.faiss"
        if not ckpt.exists() or not index_path.exists():
            return
        import faiss
        import torch
        from torchvision import transforms

        from ml.explain.gradcam import load_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(ckpt, self.device)
        self.index = faiss.read_index(str(index_path))
        self.ids = [str(x) for x in np.load(self.artifacts / "ids.npy", allow_pickle=True)]
        meta_list = json.loads((self.artifacts / "embed_meta.json").read_text(encoding="utf-8"))
        self.meta = {m["id"]: m for m in meta_list}
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict_image(
        self,
        image: Image.Image,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        case_id: str | None = None,
    ) -> HybridPredictResponse:
        import torch

        assert self.ready and self.model is not None and self.index is not None
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs_t = torch.softmax(logits, dim=1)[0].cpu().numpy()
            feat = self.model.forward_features(x).cpu().numpy().astype(np.float32)
        feat /= np.linalg.norm(feat) + 1e-8
        probs = {TARGET_LABELS[i]: float(probs_t[i]) for i in range(len(TARGET_LABELS))}
        label = max(probs, key=probs.get)  # type: ignore[arg-type]
        pred = PredictionResult(label=label, confidence=float(probs[label]), probs=probs)  # type: ignore[arg-type]

        sims, idxs = self.index.search(feat.reshape(1, -1), top_k + 1)
        similar: list[SimilarCase] = []
        dists = []
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            cid = self.ids[int(idx)]
            if case_id and cid == case_id:
                continue
            if float(sim) < similarity_threshold:
                continue
            m = self.meta.get(cid, {})
            similar.append(
                SimilarCase(
                    case_id=cid,
                    similarity=round(float(sim), 4),
                    pred_label=m.get("gt_label", "other_unmapped"),  # type: ignore[arg-type]
                    gt_label=m.get("gt_label", "other_unmapped"),  # type: ignore[arg-type]
                    confidence=float(probs[label]),
                    thumbnail_url=f"/media/processed/{m.get('path', '')}",
                    gradcam_url=None,
                )
            )
            dists.append(1.0 - float(sim))
            if len(similar) >= top_k:
                break

        settings = get_settings()
        nn = dists[0] if dists else 1.0
        avg = float(sum(dists) / len(dists)) if dists else 1.0
        analysis = SimilarityAnalysis(
            nn_distance=round(nn, 4),
            avg_topk_distance=round(avg, 4),
            cluster_id=TARGET_LABELS.index(label) if label in TARGET_LABELS else None,
            similarity_confidence=round(max(0.0, 1.0 - avg), 4),
            novel_or_low_similarity=nn > settings.novelty_distance_threshold,
        )
        return HybridPredictResponse(
            mode=AppMode.trained,
            disclaimer=DISCLAIMER,
            case_id=case_id,
            prediction=pred,
            similarity_analysis=analysis,
            similar_cases=similar,
            is_mock=False,
        )


@lru_cache
def get_trained_service() -> TrainedInferenceService:
    return TrainedInferenceService(get_settings().artifacts_dir)
