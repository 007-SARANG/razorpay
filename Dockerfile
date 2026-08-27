# Trikon -- reconciliation controller.
#
# The image contains no credentials. Provide an LLM key at runtime if you want
# adjudication (`-e TRIKON_LLM_API_KEY=...`); without one the container runs the full
# deterministic pipeline and produces every metric.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY trikon ./trikon
RUN pip install --no-cache-dir -e .

COPY api ./api
COPY web ./web
COPY tests ./tests

# Fail the build if the reconciler does not actually work.
RUN python -m trikon.cli run --orders 300 > /dev/null && echo "self-check passed"

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
