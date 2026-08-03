from fastapi import APIRouter, HTTPException

from apps.api.app.services.demo_store import get_demo_store
from crohnstracker_shared.schemas import LesionEvent

router = APIRouter(tags=["sequences"])


@router.get("/sequences", response_model=list[LesionEvent])
def list_sequences() -> list[LesionEvent]:
    return get_demo_store().sequences()


@router.get("/sequences/{sequence_id}", response_model=list[LesionEvent])
def get_sequence(sequence_id: str) -> list[LesionEvent]:
    events = get_demo_store().sequence(sequence_id)
    if not events:
        raise HTTPException(status_code=404, detail="Sequence not found")
    return events
