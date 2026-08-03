from fastapi import APIRouter, HTTPException

from apps.api.app.services.demo_store import get_demo_store
from crohnstracker_shared.schemas import ImageCase, SimilarCase, SimilarityAnalysis, SimilarQuery

router = APIRouter(tags=["cases"])


@router.get("/cases", response_model=list[ImageCase])
def list_cases() -> list[ImageCase]:
    return get_demo_store().list_cases()


@router.get("/cases/{case_id}", response_model=ImageCase)
def get_case(case_id: str) -> ImageCase:
    case = get_demo_store().get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.get("/similar")
def similar(
    case_id: str,
    top_k: int = 5,
    similarity_threshold: float = 0.0,
    metric: str = "cosine",
) -> dict:
    if metric != "cosine":
        raise HTTPException(status_code=400, detail="MVP supports metric=cosine only")
    store = get_demo_store()
    try:
        cases, analysis = store.similar(case_id, top_k=top_k, similarity_threshold=similarity_threshold)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown case_id: {exc}") from exc
    return {
        "mode": "demo",
        "is_mock": True,
        "query": SimilarQuery(
            case_id=case_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metric="cosine",
        ).model_dump(),
        "similarity_analysis": SimilarityAnalysis.model_validate(analysis).model_dump(),
        "similar_cases": [SimilarCase.model_validate(c).model_dump() for c in cases],
    }
