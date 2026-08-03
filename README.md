# CrohnTracker

AI-assisted colonoscopy / capsule endoscopy lesion analysis for **research and education**.

Similarity retrieval complements classification. **Similarity does not imply diagnosis.**

Demo mode ships with clearly labeled placeholder images and **mock predictions**. Never treat demo outputs as clinical results.

## Stack

| Layer | Tech |
|-------|------|
| Web | React + TypeScript + Vite |
| API | FastAPI |
| Model | EfficientNet-B0 (timm) |
| Retrieval | FAISS (cosine via Inner Product) |
| Dataset | Kvasir-Capsule (download separately) |

## Quick start (demo mode)

```bash
# Python 3.11+
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e packages/shared/python
pip install -r apps/api/requirements.txt
python ml/data/generate_demo_fixtures.py

# Terminal 1 — API
set PYTHONPATH=.
set APP_MODE=demo
uvicorn apps.api.app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Web
cd apps/web
npm install
npm run dev
```

Open http://localhost:5173 — you should see a **Demo data — mock predictions** banner and a hybrid prediction view.

Health check: http://localhost:8000/health

### Docker

```bash
docker compose up --build
```

## Train on Kvasir-Capsule

```bash
python ml/data/download_kvasir_capsule.py   # follow DOWNLOAD.md / set KVASIR_URL
python ml/data/preprocess.py
python ml/data/split_by_video.py
pip install -r ml/requirements.txt
python ml/train/train_cls.py --smoke          # tiny smoke run
python ml/train/train_cls.py                  # full train
python ml/embed/extract_embeddings.py
python ml/embed/build_faiss.py
python ml/embed/build_projections.py
python ml/eval/eval_cls.py
python ml/eval/eval_retrieval.py
```

Then run the API with `APP_MODE=trained` after artifacts exist under `artifacts/`.

## Repository layout

```
apps/api          FastAPI
apps/web          React UI
packages/shared   Pydantic + Zod contracts
ml/               data, train, Grad-CAM, FAISS, eval
data/demo         demo fixtures (placeholders)
data/cards        dataset + model cards
infra/docker      Dockerfiles
```

## Tests & CI

```bash
pytest -q
cd apps/web && npm run typecheck
```

GitHub Actions runs API tests + web typecheck on pull requests.

## Disclaimer

Research and educational use only. Mock predictions are labeled in API (`is_mock: true`) and UI. This software is not a medical device and must not be used as the sole basis for clinical decisions.
