from fastapi import APIRouter

from apps.api.app.services.demo_store import get_demo_store
from crohnstracker_shared.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return get_demo_store().health()
