from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from apps.api.app.config import get_settings
from apps.api.app.services.demo_store import get_demo_store
from apps.api.app.services.inference import get_trained_service
from crohnstracker_shared.schemas import HybridPredictResponse, PredictRequest

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=HybridPredictResponse)
def predict(body: PredictRequest | None = None) -> HybridPredictResponse:
    """Classify a case and return Top-K similar cases (hybrid view).

    Demo mode uses deterministic mock fixtures (`is_mock=true`).
    Trained mode expects a prior indexed case_id or use `/predict/upload`.
    """
    settings = get_settings()
    req = body or PredictRequest()
    store = get_demo_store()

    if settings.mode == "demo":
        if not store.ready:
            raise HTTPException(
                status_code=503,
                detail="Demo fixtures missing. Run: python ml/data/generate_demo_fixtures.py",
            )
        try:
            return store.predict(
                case_id=req.case_id,
                top_k=req.top_k,
                similarity_threshold=req.similarity_threshold,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown case_id: {exc}") from exc

    service = get_trained_service()
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail="Trained mode selected but model/FAISS artifacts are missing. Use APP_MODE=demo or train + index.",
        )
    raise HTTPException(
        status_code=400,
        detail="In trained mode, POST an image to /predict/upload (multipart) or provide a processed case workflow.",
    )


@router.post("/predict/upload", response_model=HybridPredictResponse)
async def predict_upload(
    file: UploadFile = File(...),
    top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> HybridPredictResponse:
    settings = get_settings()
    if settings.mode == "demo":
        # Keep demo runnable if someone uploads; still return mocks and label as mock
        store = get_demo_store()
        return store.predict(top_k=top_k, similarity_threshold=similarity_threshold)

    service = get_trained_service()
    if not service.ready:
        raise HTTPException(status_code=503, detail="Model/FAISS artifacts missing")
    raw = await file.read()
    image = Image.open(BytesIO(raw)).convert("RGB")
    return service.predict_image(image, top_k=top_k, similarity_threshold=similarity_threshold)


@router.post("/predict/{case_id}", response_model=HybridPredictResponse)
def predict_case(case_id: str, top_k: int = 5, similarity_threshold: float = 0.0) -> HybridPredictResponse:
    store = get_demo_store()
    try:
        return store.predict(case_id=case_id, top_k=top_k, similarity_threshold=similarity_threshold)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown case_id: {exc}") from exc
