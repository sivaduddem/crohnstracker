from fastapi import APIRouter

from apps.api.app.services.demo_store import get_demo_store
from crohnstracker_shared.schemas import MetricsSummary

router = APIRouter(tags=["metrics"])


@router.get("/metrics/summary", response_model=MetricsSummary)
def metrics_summary() -> MetricsSummary:
    return get_demo_store().metrics()
