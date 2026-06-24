# Contributing

Thanks for your interest in aiforge-editor! This guide covers local setup and
the quality bar.

## Setup

```bash
# Backend (offline; no keys needed)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pre-commit && pre-commit install   # from the repo root

# Frontend
cd ../frontend
npm install
```

Or use the Makefile from the repo root: `make install`, `make frontend-install`.

## Running locally

```bash
# Backend (auto-reload, offline mock backend)
make run            # -> http://localhost:8000  (/health, /ready, /metrics, /docs)

# Frontend (proxies /api -> :8000)
make frontend-build # or: cd frontend && npm run dev  -> http://localhost:5173
```

Register any credentials on the login screen — everything runs offline with the
deterministic mock model.

## Quality bar (must pass before a PR)

Backend:

```bash
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy (0 errors)
make test-cov    # pytest, >=80% coverage on the core
make security    # bandit + pip-audit
make migrate     # alembic upgrade head (schema applies)
```

Frontend:

```bash
cd frontend
npm run typecheck   # tsc --noEmit (0 errors)
npm run test        # vitest
npm run build       # tsc + vite build
```

CI runs all of the above across Python 3.9 / 3.11 / 3.12, plus a Docker buildx
of both images.

## Conventions

- **Backend stays Python 3.9 compatible.** Use `from __future__ import
  annotations` and `typing.List`/`Optional` (not PEP 604 unions at runtime).
- **torch is import-guarded.** Never import `aiforge.ml.model` (or other heavy
  ML modules) at backend startup — go through `aiforge.ml.load_module(...)` or
  lazy imports, so the API and CI run without torch.
- **No secrets in code.** Env vars only; add placeholders to `.env.example`.
- **New AI features** must work offline via the mock backend and have tests.
- **The diff engine and workspace sandbox** are security-/correctness-critical;
  changes there require tests.

## Tests

- Backend: `pytest` under `backend/tests/`. ML tests skip cleanly when torch is
  absent; the local-model integration test skips when no checkpoint exists.
- Frontend: `vitest` under `src/__tests__/`; a guarded Playwright smoke test
  under `e2e/` (runs only when `E2E_BASE_URL` is set).

## Training the model

See [`backend/aiforge/ml/README.md`](backend/aiforge/ml/README.md):

```bash
pip install -r backend/requirements-train.txt
make train   # python -m aiforge.ml.train --out runs/proof
make eval
make onnx
```

## License

By contributing, you agree your contributions are licensed under the MIT
License (see [LICENSE](LICENSE)).
