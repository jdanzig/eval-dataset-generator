.PHONY: install test build run
install:
	uv venv --python 3.12
	uv pip install fastapi "uvicorn[standard]" httpx "pydantic>=2" numpy scikit-learn \
		sentence-transformers pytest
test:      ## offline unit tests (no API key, no model download)
	.venv/bin/python -m pytest -q
build:     ## mine traffic -> label -> score difficulty -> proof (writes eval_set.json, coverage.html); needs ANTHROPIC_API_KEY
	.venv/bin/python -m app.cli build
run:       ## review-queue + coverage API on :8000 (serves the prebuilt eval_set.json)
	.venv/bin/python -m uvicorn app.api:app --reload --port 8000
