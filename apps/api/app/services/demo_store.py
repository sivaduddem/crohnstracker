from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from apps.api.app.config import get_settings
from crohnstracker_shared.constants import DEMO_BANNER, DISCLAIMER, TARGET_LABELS
from crohnstracker_shared.schemas import (
    AppMode,
    ClassificationMetricsSnapshot,
    EmbeddingPoint,
    EmbeddingProjectionResponse,
    HealthResponse,
    HybridPredictResponse,
    ImageCase,
    LesionEvent,
    MetricsSummary,
    PredictionResult,
    RetrievalMetricsSnapshot,
    SimilarCase,
    SimilarityAnalysis,
)


class DemoStore:
    """Deterministic demo fixtures. Never presented as real model inference."""

    def __init__(self, demo_dir: Path) -> None:
        self.demo_dir = demo_dir
        self.meta_path = demo_dir / "cases.json"
        self.proj_path = demo_dir / "projections.json"
        self.metrics_path = demo_dir / "metrics.json"
        self.sequences_path = demo_dir / "sequences.json"
        self._cases: list[dict] = []
        self._embeddings: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.meta_path.exists():
            self._cases = []
            return
        payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._cases = payload.get("cases", [])
        dim = int(payload.get("embedding_dim", 32))
        rng = np.random.default_rng(42)
        for case in self._cases:
            seed = abs(hash(case["id"])) % (2**32)
            local = np.random.default_rng(seed)
            vec = local.normal(size=dim).astype(np.float32)
            # Class-conditioned drift for plausible neighborhoods
            label_idx = TARGET_LABELS.index(case["gt_label"]) if case["gt_label"] in TARGET_LABELS else 0
            vec[label_idx] += 3.0
            vec /= np.linalg.norm(vec) + 1e-8
            self._embeddings[case["id"]] = vec
        _ = rng  # reserved for future fixture noise

    @property
    def ready(self) -> bool:
        return bool(self._cases)

    def list_cases(self) -> list[ImageCase]:
        return [ImageCase(**c) for c in self._cases]

    def get_case(self, case_id: str) -> ImageCase | None:
        for c in self._cases:
            if c["id"] == case_id:
                return ImageCase(**c)
        return None

    def default_case_id(self) -> str:
        if not self._cases:
            raise LookupError("Demo fixtures missing. Run: python ml/data/generate_demo_fixtures.py")
        return self._cases[0]["id"]

    def health(self) -> HealthResponse:
        settings = get_settings()
        mode = AppMode.demo if settings.mode == "demo" else AppMode.trained
        trained_model = (settings.artifacts_dir / "model.pt").exists()
        faiss_idx = (settings.artifacts_dir / "index.faiss").exists()
        if mode == AppMode.demo:
            ok = self.ready
            return HealthResponse(
                status="ok" if ok else "degraded",
                mode=mode,
                model_loaded=False,
                faiss_loaded=self.ready,
                demo_banner=DEMO_BANNER,
            )
        ok = trained_model and faiss_idx
        return HealthResponse(
            status="ok" if ok else "degraded",
            mode=mode,
            model_loaded=trained_model,
            faiss_loaded=faiss_idx,
            demo_banner=None,
        )

    def _mock_prediction(self, case: ImageCase) -> PredictionResult:
        # Deterministic soft probs peaked at pred_label (or gt if missing)
        label = case.pred_label or case.gt_label
        conf = float(case.confidence if case.confidence is not None else 0.9)
        probs = {lab: 0.02 for lab in TARGET_LABELS}
        remain = max(conf, 0.4)
        probs[label if label in TARGET_LABELS else "normal"] = remain
        # renormalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        top = max(probs, key=probs.get)  # type: ignore[arg-type]
        return PredictionResult(label=top, confidence=float(probs[top]), probs=probs)  # type: ignore[arg-type]

    def similar(
        self,
        case_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> tuple[list[SimilarCase], SimilarityAnalysis]:
        if case_id not in self._embeddings:
            raise KeyError(case_id)
        q = self._embeddings[case_id]
        scored: list[tuple[str, float]] = []
        for cid, vec in self._embeddings.items():
            if cid == case_id:
                continue
            sim = float(np.dot(q, vec))
            if sim >= similarity_threshold:
                scored.append((cid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        similar_cases: list[SimilarCase] = []
        for cid, sim in top:
            case = self.get_case(cid)
            assert case is not None
            similar_cases.append(
                SimilarCase(
                    case_id=cid,
                    similarity=round(sim, 4),
                    pred_label=case.pred_label or case.gt_label,  # type: ignore[arg-type]
                    gt_label=case.gt_label,
                    confidence=float(case.confidence or 0.85),
                    thumbnail_url=f"/media/images/{Path(case.path).name}",
                    gradcam_url=f"/media/overlays/{cid}_gradcam.png",
                    severity=case.severity,
                )
            )
        dists = [1.0 - s for _, s in top] or [1.0]
        nn = dists[0]
        avg = float(sum(dists) / len(dists))
        settings = get_settings()
        novel = nn > settings.novelty_distance_threshold
        analysis = SimilarityAnalysis(
            nn_distance=round(nn, 4),
            avg_topk_distance=round(avg, 4),
            cluster_id=TARGET_LABELS.index(self.get_case(case_id).gt_label)  # type: ignore[union-attr]
            if self.get_case(case_id) and self.get_case(case_id).gt_label in TARGET_LABELS  # type: ignore[union-attr]
            else None,
            similarity_confidence=round(max(0.0, 1.0 - avg), 4),
            novel_or_low_similarity=novel,
        )
        return similar_cases, analysis

    def predict(
        self,
        case_id: str | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> HybridPredictResponse:
        cid = case_id or self.default_case_id()
        case = self.get_case(cid)
        if case is None:
            raise KeyError(cid)
        pred = self._mock_prediction(case)
        similar_cases, analysis = self.similar(cid, top_k=top_k, similarity_threshold=similarity_threshold)
        return HybridPredictResponse(
            mode=AppMode.demo,
            disclaimer=DISCLAIMER,
            case_id=cid,
            prediction=pred,
            similarity_analysis=analysis,
            similar_cases=similar_cases,
            is_mock=True,
        )

    def projections(self, method: str = "umap") -> EmbeddingProjectionResponse:
        if self.proj_path.exists():
            raw = json.loads(self.proj_path.read_text(encoding="utf-8"))
            points = [EmbeddingPoint(**p) for p in raw.get(method, raw.get("umap", []))]
        else:
            points = []
            for i, c in enumerate(self._cases):
                points.append(
                    EmbeddingPoint(
                        case_id=c["id"],
                        x=float(i % 5) + 0.1 * i,
                        y=float(i // 5) + 0.07 * i,
                        gt_label=c["gt_label"],
                        pred_label=c.get("pred_label") or c["gt_label"],
                        confidence=float(c.get("confidence") or 0.9),
                        severity=c.get("severity"),
                        thumbnail_url=f"/media/images/{Path(c['path']).name}",
                    )
                )
        return EmbeddingProjectionResponse(
            mode=AppMode.demo,
            method=method if method in ("umap", "tsne", "pca") else "umap",  # type: ignore[arg-type]
            points=points,
            is_mock=True,
        )

    def metrics(self) -> MetricsSummary:
        if self.metrics_path.exists():
            raw = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            return MetricsSummary(
                mode=AppMode.demo,
                disclaimer=DISCLAIMER,
                classification=ClassificationMetricsSnapshot(**raw["classification"]),
                retrieval=RetrievalMetricsSnapshot(**raw["retrieval"]),
            )
        labels = list(TARGET_LABELS)
        return MetricsSummary(
            mode=AppMode.demo,
            classification=ClassificationMetricsSnapshot(
                accuracy=0.0,
                macro_f1=0.0,
                per_class_f1={k: 0.0 for k in labels},
                confusion_matrix=[[0] * 4 for _ in range(4)],
                labels=labels,
            ),
            retrieval=RetrievalMetricsSnapshot(
                recall_at_k=0.0,
                precision_at_k=0.0,
                map=0.0,
                ndcg=0.0,
                latency_ms=0.0,
                embedding_ms=0.0,
                k=5,
            ),
        )

    def sequences(self) -> list[LesionEvent]:
        if not self.sequences_path.exists():
            return []
        raw = json.loads(self.sequences_path.read_text(encoding="utf-8"))
        return [LesionEvent(**e) for e in raw.get("events", [])]

    def sequence(self, sequence_id: str) -> list[LesionEvent]:
        return [e for e in self.sequences() if e.sequence_id == sequence_id]


@lru_cache
def get_demo_store() -> DemoStore:
    return DemoStore(get_settings().demo_data_dir)
