.PHONY: help install demo-data api web test lint typecheck up down health

help:
	@echo "CrohnTracker targets:"
	@echo "  make install     - Install API + web deps"
	@echo "  make demo-data   - Generate labeled demo placeholders"
	@echo "  make api         - Run FastAPI (demo mode)"
	@echo "  make web         - Run Vite frontend"
	@echo "  make test        - Run API + unit tests"
	@echo "  make lint        - Ruff + web lint"
	@echo "  make typecheck   - TypeScript check"
	@echo "  make up          - docker compose up --build"
	@echo "  make down        - docker compose down"
	@echo "  make health      - Curl /health"

install:
	python -m pip install -e "packages/shared/python[dev]"
	python -m pip install -r apps/api/requirements.txt
	python -m pip install -r ml/requirements.txt
	cd apps/web && npm install

demo-data:
	python ml/data/generate_demo_fixtures.py

api:
	set APP_MODE=demo&& uvicorn apps.api.app.main:app --reload --app-dir . --host 0.0.0.0 --port 8000

web:
	cd apps/web && npm run dev

test:
	pytest apps/api/tests tests -q

lint:
	ruff check apps/api ml packages/shared/python
	cd apps/web && npm run lint

typecheck:
	cd apps/web && npm run typecheck

up:
	docker compose up --build

down:
	docker compose down

health:
	curl -s http://localhost:8000/health
