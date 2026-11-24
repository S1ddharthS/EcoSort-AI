# EcoSort AI production image.
# Build after training the model: docker build -t ecosort-ai .
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=1 \
    PORT=8080 \
    ECOSORT_DB=/data/ecosort.db \
    ECOSORT_UPLOADS=/data/uploads

# OpenCV needs libGL/libglib; libgomp is used by TensorFlow.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

# The database and uploads live outside /app so a mounted volume can hold them.
RUN mkdir -p /data/uploads /app/logs \
    && groupadd --system ecosort \
    && useradd --system --gid ecosort --home-dir /app ecosort \
    && chown -R ecosort:ecosort /data /app

USER ecosort

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/health', timeout=4)"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
