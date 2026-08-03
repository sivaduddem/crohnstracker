from fastapi import APIRouter, HTTPException

from apps.api.app.services.demo_store import get_demo_store
from crohnstracker_shared.schemas import EmbeddingProjectionResponse

router = APIRouter(tags=["embeddings"])


@router.get("/embeddings/projection", response_model=EmbeddingProjectionResponse)
def projection(method: str = "umap") -> EmbeddingProjectionResponse:
    if method not in {"umap", "tsne", "pca"}:
        raise HTTPException(status_code=400, detail="method must be umap|tsne|pca")
    return get_demo_store().projections(method=method)
