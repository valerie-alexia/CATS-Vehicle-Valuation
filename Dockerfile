# syntax=docker/dockerfile:1
FROM python:3.11.18-slim-bookworm AS base

# OpenMP runtime required by the Linux XGBoost/LightGBM wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./

# Enforce the existing lock; never resolve upgrades or download another Python.
# Dashboard dependencies remain in the project manifest but are omitted here.
RUN uv sync --locked --no-python-downloads --python /usr/local/bin/python \
    --no-default-groups --no-group dev --no-group scraping \
    --no-install-project --no-editable \
    --no-install-package streamlit --no-install-package altair \
    && uv pip check \
    && .venv/bin/python -c "import fastapi, uvicorn, numpy, pandas, sklearn, xgboost, lightgbm, joblib; import importlib.util; assert all(importlib.util.find_spec(m) is None for m in ('selenium', 'undetected_chromedriver', 'pytest', 'streamlit', 'altair'))"

FROM base AS runtime
RUN groupadd --system cats \
    && useradd --system --gid cats --home-dir /app --no-create-home cats \
    && mkdir -p /data/evidence

COPY --from=dependencies /app/.venv /app/.venv

# Explicit runtime source allowlist: no Dashboard, scraper, cleaner, or notebook.
COPY src/Retraining/api.py src/Retraining/catalog.py \
    src/Retraining/cats_input.py src/Retraining/ev_api.py \
    src/Retraining/evidence.py src/Retraining/explanations.py \
    src/Retraining/granular_lookup.py src/Retraining/granular_runtime.py \
    src/Retraining/inference.py src/Retraining/input_values.py \
    src/Retraining/price_range.py src/Retraining/service.py ./src/Retraining/
COPY src/Training/pipeline.py src/Training/experiment_granular.py \
    src/Training/experiment_export.py ./src/Training/
COPY Data/audit/v2/cats_demo_app.py Data/audit/v2/cats_market_api_experiment.py \
    Data/audit/v2/ev_category_review.py Data/audit/v2/ev_combination_review.py \
    Data/audit/v2/ev_inference_experiment.py Data/audit/v2/market_price_category.py \
    Data/audit/v2/market_price_statistics.py ./Data/audit/v2/
COPY Data/audit/v2/ev_specialist_experiment/ev_specialist_candidate.joblib \
    Data/audit/v2/ev_specialist_experiment/ev_experiment_report.json \
    Data/audit/v2/ev_specialist_experiment/ev_training_combinations.json \
    ./Data/audit/v2/ev_specialist_experiment/
COPY Models/Mobil123/active.json ./Models/Mobil123/active.json
COPY Models/Mobil123/versions/20261022T093559Z-5eaa62ee/bundle.joblib \
    ./Models/Mobil123/versions/20261022T093559Z-5eaa62ee/bundle.joblib
COPY Models/Momotor/active.json ./Models/Momotor/active.json
COPY Models/Momotor/versions/20261022T095302Z-b492bfaf/bundle.joblib \
    ./Models/Momotor/versions/20261022T095302Z-b492bfaf/bundle.joblib

ENV PATH="/app/.venv/bin:$PATH" \
    VEHICLE_EVIDENCE_DIR=/data/evidence \
    CATS_MOBIL_MODEL=active8

# Validate the packaged module closure without starting a server or reading CSVs.
RUN .venv/bin/python -c "import sys; sys.path.insert(0, 'Data/audit/v2'); import cats_demo_app; cats_demo_app.app.openapi()"

USER cats
EXPOSE 8000
# /health checks models only; an empty evidence directory does not make it fail.
# Restrict published API access at the deployment network/proxy boundary.
CMD ["python", "-m", "uvicorn", "cats_demo_app:app", "--app-dir", "Data/audit/v2", "--host", "0.0.0.0", "--port", "8000"]
