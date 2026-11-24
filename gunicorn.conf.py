"""Gunicorn configuration for EcoSort AI.

One worker is deliberate: the TensorFlow model is large, and the in-memory
rate limiter is per-process. Threads let health checks stay responsive while
a prediction is running.
"""

from __future__ import annotations

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_class = "gthread"
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
# Use /dev/shm for the temporary heartbeat file where available.
worker_tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
