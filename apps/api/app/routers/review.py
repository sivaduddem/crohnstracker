from fastapi import APIRouter

from crohnstracker_shared.schemas import ReviewFlag

router = APIRouter(tags=["review"])

_FLAGS: list[ReviewFlag] = []


@router.get("/review/flags", response_model=list[ReviewFlag])
def list_flags() -> list[ReviewFlag]:
    return _FLAGS


@router.post("/review/flags", response_model=ReviewFlag)
def upsert_flag(flag: ReviewFlag) -> ReviewFlag:
    global _FLAGS
    _FLAGS = [f for f in _FLAGS if not (f.case_id == flag.case_id and f.reason == flag.reason)]
    _FLAGS.append(flag)
    return flag
