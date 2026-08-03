from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from crohnstracker_shared.constants import DISCLAIMER, TARGET_LABELS


class AppMode(str, Enum):
    demo = "demo"
    trained = "trained"


LesionLabel = Literal["normal", "erythema", "erosion", "ulcer", "other_unmapped"]
ReviewFlagReason = Literal[
    "novel_or_low_similarity",
    "low_confidence",
    "disagreement",
]


class PredictionResult(BaseModel):
    label: LesionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probs: dict[str, float]


class SimilarCase(BaseModel):
    case_id: str
    similarity: float = Field(ge=-1.0, le=1.0)
    pred_label: LesionLabel
    gt_label: LesionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    thumbnail_url: str
    gradcam_url: str | None = None
    severity: str | None = None


class SimilarityAnalysis(BaseModel):
    nn_distance: float
    avg_topk_distance: float
    cluster_id: int | None = None
    similarity_confidence: float = Field(ge=0.0, le=1.0)
    novel_or_low_similarity: bool = False


class HybridPredictResponse(BaseModel):
    mode: AppMode
    disclaimer: str = DISCLAIMER
    case_id: str | None = None
    prediction: PredictionResult
    similarity_analysis: SimilarityAnalysis
    similar_cases: list[SimilarCase]
    is_mock: bool = Field(
        description="True when outputs are deterministic demo mocks, not model inference."
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    mode: AppMode
    model_loaded: bool
    faiss_loaded: bool
    demo_banner: str | None = None
    version: str = "0.1.0"


class ImageCase(BaseModel):
    id: str
    source: str
    video_id: str
    frame_idx: int
    path: str
    gt_label: LesionLabel
    severity: str | None = None
    split: Literal["train", "val", "test", "demo"] = "demo"
    pred_label: LesionLabel | None = None
    confidence: float | None = None


class LesionEvent(BaseModel):
    sequence_id: str
    start_frame: int
    end_frame: int
    label: LesionLabel
    peak_confidence: float
    case_ids: list[str] = Field(default_factory=list)


class ReviewFlag(BaseModel):
    case_id: str
    reason: ReviewFlagReason
    status: Literal["open", "resolved"] = "open"
    note: str | None = None


class RetrievalMetricsSnapshot(BaseModel):
    recall_at_k: float
    precision_at_k: float
    map: float
    ndcg: float
    latency_ms: float
    embedding_ms: float | None = None
    k: int = 5


class ClassificationMetricsSnapshot(BaseModel):
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    labels: list[str] = Field(default_factory=lambda: list(TARGET_LABELS))


class MetricsSummary(BaseModel):
    mode: AppMode
    disclaimer: str = DISCLAIMER
    classification: ClassificationMetricsSnapshot
    retrieval: RetrievalMetricsSnapshot


class EmbeddingPoint(BaseModel):
    case_id: str
    x: float
    y: float
    gt_label: LesionLabel
    pred_label: LesionLabel
    confidence: float
    severity: str | None = None
    thumbnail_url: str


class EmbeddingProjectionResponse(BaseModel):
    mode: AppMode
    method: Literal["umap", "tsne", "pca"]
    points: list[EmbeddingPoint]
    is_mock: bool = True


class PredictRequest(BaseModel):
    case_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metric: Literal["cosine"] = "cosine"


class SimilarQuery(BaseModel):
    case_id: str
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metric: Literal["cosine"] = "cosine"
