FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir -e .

ENV ROUTER_USE_FAKE_ENCODER=true
ENV ROUTER_RATE_LIMIT_PER_MINUTE=60
ENV DISPATCH_PROFILE=demo

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
