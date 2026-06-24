# aiforge-editor — developer tasks.
.DEFAULT_GOAL := help
.PHONY: help install install-train test test-cov lint format typecheck security \
        migrate run train eval onnx frontend-install frontend-build frontend-test \
        docker-up docker-down clean

BACKEND := backend
FRONTEND := frontend

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install backend runtime + dev deps
	cd $(BACKEND) && pip install -r requirements.txt

install-train: ## Install ML training deps (torch etc.)
	cd $(BACKEND) && pip install -r requirements-train.txt

test: ## Run backend tests
	cd $(BACKEND) && pytest -q

test-cov: ## Run backend tests with coverage gate (>=80% core)
	cd $(BACKEND) && pytest --cov=aiforge --cov-report=term-missing --cov-fail-under=80

lint: ## Ruff lint
	cd $(BACKEND) && ruff check aiforge

format: ## Ruff format
	cd $(BACKEND) && ruff format aiforge tests

typecheck: ## Mypy typecheck
	cd $(BACKEND) && mypy aiforge

security: ## Bandit + pip-audit
	cd $(BACKEND) && bandit -q -r aiforge -c pyproject.toml && pip-audit -r requirements-min.txt || true

migrate: ## Apply DB migrations
	cd $(BACKEND) && alembic upgrade head

run: ## Run the backend (offline mock backend)
	cd $(BACKEND) && uvicorn aiforge.api.server:app --reload --port 8000

train: ## Train the from-scratch code model (proof run)
	cd $(BACKEND) && python -m aiforge.ml.train --out runs/proof --max-steps 2000

eval: ## Evaluate the trained model
	cd $(BACKEND) && python -m aiforge.ml.eval --run runs/proof

onnx: ## Export the model to ONNX (+ verify)
	cd $(BACKEND) && python -m aiforge.ml.export_onnx --run runs/proof --verify

frontend-install: ## Install frontend deps
	cd $(FRONTEND) && npm install

frontend-build: ## Typecheck + build the frontend
	cd $(FRONTEND) && npm run build

frontend-test: ## Run frontend unit tests
	cd $(FRONTEND) && npm run test

docker-up: ## Bring up the full stack (compose)
	docker compose up --build

docker-down: ## Tear down the stack
	docker compose down -v

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache $(BACKEND)/.mypy_cache \
	       $(BACKEND)/.coverage $(BACKEND)/htmlcov $(FRONTEND)/dist
