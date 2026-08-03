from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from apps.api.app.config import get_settings
from apps.api.app.services.demo_store import get_demo_store

router = APIRouter(tags=["explain"])


@router.post("/gradcam")
def gradcam(case_id: str) -> dict:
    """Return Grad-CAM overlay URL for a case.

    Demo mode serves pre-generated mock heatmaps labeled as non-clinical overlays.
    Trained mode can generate overlays under artifacts/gradcam/ when a checkpoint exists.
    """
    settings = get_settings()
    store = get_demo_store()

    if settings.mode == "demo":
        case = store.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found")
        overlay = settings.demo_data_dir / "overlays" / f"{case_id}_gradcam.png"
        if not overlay.exists():
            raise HTTPException(status_code=404, detail="Grad-CAM overlay missing; regenerate demo fixtures")
        return {
            "mode": settings.mode,
            "is_mock": True,
            "case_id": case_id,
            "gradcam_url": f"/media/overlays/{case_id}_gradcam.png",
            "disclaimer": "Demo Grad-CAM-style overlay - not a real model explanation.",
        }

    ckpt = settings.artifacts_dir / "model.pt"
    if not ckpt.exists():
        raise HTTPException(status_code=503, detail="No trained checkpoint for Grad-CAM")
    # Overlay expected to be produced offline by ml/explain tooling or future on-demand path
    out = settings.artifacts_dir / "gradcam" / f"{case_id}_gradcam.png"
    if not out.exists():
        raise HTTPException(status_code=404, detail="Grad-CAM not generated for this case yet")
    return {
        "mode": settings.mode,
        "is_mock": False,
        "case_id": case_id,
        "gradcam_url": f"/artifacts/gradcam/{case_id}_gradcam.png",
        "disclaimer": "Research Grad-CAM overlay - explanatory only, not a diagnosis.",
    }


@router.get("/gradcam/{case_id}")
def gradcam_file(case_id: str) -> FileResponse:
    settings = get_settings()
    path = settings.demo_data_dir / "overlays" / f"{case_id}_gradcam.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Overlay not found")
    return FileResponse(path)
