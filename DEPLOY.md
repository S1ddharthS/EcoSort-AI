# EcoSort AI — Deployment & Production Guide

This guide describes how to run and deploy **EcoSort AI** across local environments, Docker containers, and cloud production platforms.

---

## 1. Local Development Deployment

```bash
# 1. Activate your virtual environment
.venv\Scripts\Activate.ps1    # Windows PowerShell
source .venv/bin/activate       # Linux / macOS

# 2. Run the application
python app.py
```

The app will start on `http://127.0.0.1:5000` (or `PORT` environment variable).

---

## 2. Docker Container Deployment

The project includes an optimized, production-ready `Dockerfile` based on `python:3.13-slim`.

### Build Image
```bash
docker build -t ecosort-ai:latest .
```

### Run Container
```bash
docker run -d \
  --name ecosort \
  -p 8080:8080 \
  -e SECRET_KEY="your-production-secret-key" \
  -v ecosort_data:/data \
  --restart unless-stopped \
  ecosort-ai:latest
```

### Persistent Volumes
- `/data/ecosort.db`: SQLite database file storing classification history and seeded recommendations.
- `/data/uploads/`: Stored preview thumbnails for user history.
Mounting a Docker volume to `/data` ensures user data and uploads persist across container updates.

---

## 3. Production WSGI Configuration (Linux / Cloud VM)

On Linux servers, EcoSort AI uses **Gunicorn**:

```bash
gunicorn -c gunicorn.conf.py app:app
```

### Key `gunicorn.conf.py` Settings
- **Workers**: 1-2 sync workers recommended (each worker loads the TensorFlow model in memory; keeping worker count modest prevents RAM exhaustion on 1GB/2GB cloud instances).
- **Timeout**: 120 seconds (ensures first-request model warm-up and Grad-CAM calculations finish cleanly without dropping connections).
- **Bind**: `0.0.0.0:8080` (or configured via `$PORT`).

---

## 4. Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `5000` (local) / `8080` (Docker) | HTTP listening port |
| `SECRET_KEY` | `ecosort-dev-key-...` | Flask session secret key (set strong random secret in prod) |
| `ECOSORT_DB` | `database/ecosort.db` | Absolute path to SQLite database file |
| `ECOSORT_UPLOADS` | `static/uploads` | Directory for storing uploaded history preview JPEGs |
| `ECOSORT_MODEL` | `ml/model/ecosort_model.keras` | Path to trained Keras model file |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `TF_CPP_MIN_LOG_LEVEL` | `1` | TensorFlow C++ console logging verbosity |

---

## 5. Cloud Platforms (Render, Railway, Fly.io, Heroku)

### Render / Railway / Heroku
- The repository includes a `Procfile`:
  ```text
  web: gunicorn -c gunicorn.conf.py app:app
  ```
- Ensure a persistent disk/volume is attached at `/data` if using Docker mode, or use default ephemeral paths for test staging.
- Set health check path to `/health` (expected HTTP 200).

---

## 6. Nginx Reverse Proxy Configuration Example

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```
