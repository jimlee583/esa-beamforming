.PHONY: install install-backend install-frontend \
       backend frontend test lint format typecheck ci \
       pre-commit

# ── Install ─────────────────────────────────────────────────────────────────

install-backend:
	cd backend && uv sync && uv pip install ruff mypy pytest httpx

install-frontend:
	cd frontend && npm install

install: install-backend install-frontend

# ── Run ─────────────────────────────────────────────────────────────────────

backend:
	cd backend && uv run uvicorn app.main:app --reload --reload-dir . --reload-dir ../array_engine --port 8000

frontend:
	cd frontend && npm run dev

# ── Quality ─────────────────────────────────────────────────────────────────

test:
	cd backend && uv run pytest -v --tb=short

lint:
	cd backend && uv run ruff check ../array_engine ../tests app
	cd frontend && npm run lint

format:
	cd backend && uv run ruff format ../array_engine ../tests app
	cd backend && uv run ruff check --fix ../array_engine ../tests app

format-check:
	cd backend && uv run ruff format --check ../array_engine ../tests app

typecheck:
	cd backend && uv run mypy

# ── CI (run everything) ────────────────────────────────────────────────────

ci: lint format-check typecheck test
	cd frontend && npm run build

# ── Pre-commit ──────────────────────────────────────────────────────────────

pre-commit:
	pre-commit run --all-files
