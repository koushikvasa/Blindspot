# Blindspot demo server: story page, live join page, stage view, and the /audit product page.
# Only the demo pieces are copied; the research code and raw data stay out of the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" "numpy>=1.26" "pydantic>=2.7"

COPY blindspot/__init__.py blindspot/__init__.py
COPY blindspot/api blindspot/api
# the audit engine (numpy only) with its calibration table and the two sample CSVs
COPY blindspot/audit blindspot/audit
COPY web web
# results/ui holds the exported (id-free) demo data; if it is missing the app falls back to web/mock
COPY results/ui results/ui

EXPOSE 8000
CMD ["sh", "-c", "uvicorn blindspot.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
