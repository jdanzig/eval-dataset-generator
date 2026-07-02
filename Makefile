.PHONY: install test build run
install:
	uv venv --python 3.12
	uv pip install fastapi "uvicorn[standard]" httpx "pydantic>=2" numpy scikit-learn pytest
test:
	.venv/bin/python -m pytest -q
build:      ## grow 50 -> 2000+ eval set + coverage heatmap (writes eval_set.json, coverage.html)
	.venv/bin/python -m app.cli build
run:        ## review-queue + coverage API on :8000
	.venv/bin/python -m uvicorn app.api:app --reload --port 8000
