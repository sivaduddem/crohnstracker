from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from apps.api.app.config import get_settings
from apps.api.app.db import init_db
from apps.api.app.routers import cases, embeddings, explain, health, metrics, predict, review, sequences
from crohnstracker_shared.constants import DISCLAIMER

settings = get_settings()

app = FastAPI(
    title="CrohnTracker API",
    description=(
        "AI-assisted colonoscopy lesion analysis for research/education. "
        + DISCLAIMER
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(cases.router)
app.include_router(explain.router)
app.include_router(embeddings.router)
app.include_router(metrics.router)
app.include_router(sequences.router)
app.include_router(review.router)

demo_dir = Path(settings.demo_data_dir)
if demo_dir.exists():
    app.mount("/media", StaticFiles(directory=str(demo_dir)), name="media")


@app.on_event("startup")
def _startup() -> None:
    try:
        init_db()
    except Exception:
        # DB is optional for demo read path
        pass


@app.get("/")
def root() -> dict:
    return {
        "name": "CrohnTracker API",
        "mode": settings.mode,
        "docs": "/docs",
        "health": "/health",
        "disclaimer": DISCLAIMER,
    }
