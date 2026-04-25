# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt \
    -r requirements-api.txt

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8000

# Startup trains on synthetic data (~30s). For production, mount a model checkpoint
# and update api/recommender.py to load it instead of training at startup.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
