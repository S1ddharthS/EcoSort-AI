# EcoSort AI – Intelligent Waste Classification & Recycling Recommendation System

> **College Project — Emerging Technologies: A Primer (22ETP58)**  
> An end-to-end computer-vision web application that identifies six common waste categories from photographs or live webcam captures, provides transparent Grad-CAM visual explanations, and offers actionable, localized recycling and disposal instructions.

---

## 🌟 Key Features

1. **Six-Class Waste Classification**:
   - Accurately classifies items into: **Plastic**, **Paper**, **Cardboard**, **Glass**, **Metal**, and **Organic waste**.
2. **Transfer Learning with MobileNetV2**:
   - Real measured **93.52% test accuracy** and **92.48% macro F1-score** across 509 held-out test images, vastly outperforming a custom baseline CNN (46.56% accuracy).
3. **Transparent AI & Grad-CAM Explainability**:
   - Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps overlaid on the uploaded photo to show exactly which image regions influenced the prediction.
4. **Honest "Not Sure" Fallback (Safe AI)**:
   - Built-in confidence thresholding (`0.60`). If the model is uncertain or the photo is ambiguous, it admits uncertainty rather than making a wild guess that could contaminate recycling bins.
5. **Live Camera & Drag-and-Drop Upload**:
   - Supports live capture directly in browser with mobile-friendly camera framing and graceful permission fallback, as well as file upload (JPEG, PNG, WEBP).
6. **Local SQLite History & Filtering**:
   - All classifications are stored locally in SQLite with thumbnail previews. Supports category filtering, individual item deletion, and full history clearing.
7. **Educational Recycling Guides & Search**:
   - Comprehensive guides for all six categories detailing typical bin colors, accepted items ("What belongs"), excluded items ("Keep it out"), prep steps, and environmental impact facts.
8. **Clean, Accessible UI with Dark Mode**:
   - Mobile-first responsive layout (tested at 375px, 768px, and 1440px), WCAG-friendly contrast, skip links, keyboard navigation, and theme persistence.

---

## 📊 Measured Model Performance

Evaluated on 509 unseen test images (70/15/15 stratified train/val/test split):

| Waste Category | Test Images | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Plastic** | 72 | 92.7% | 88.9% | **90.8%** |
| **Paper** | 90 | 91.4% | 94.4% | **92.9%** |
| **Cardboard** | 61 | 96.3% | 85.3% | **90.4%** |
| **Glass** | 76 | 87.2% | 89.5% | **88.3%** |
| **Metal** | 62 | 88.4% | 98.4% | **93.1%** |
| **Organic** | 148 | 100.0% | 98.7% | **99.3%** |
| **Overall (Macro Avg)** | **509** | **92.7%** | **92.5%** | **92.5%** |
| **Overall Accuracy** | — | — | — | **93.52%** |

*Comparison*: Custom 4-block CNN trained from scratch achieved **46.56%** accuracy and **42.59%** macro F1, proving the clear advantage of transfer learning with ImageNet pre-trained weights on compact waste datasets.

---

## 🛠️ Tech Stack

- **Deep Learning / Computer Vision**: TensorFlow 2.x / Keras (`MobileNetV2`), OpenCV (`opencv-python`), Pillow, NumPy, Scikit-learn.
- **Backend / Web Framework**: Python 3.11+, Flask 3.1, SQLite 3 (WAL mode).
- **Frontend**: Semantic HTML5, CSS3 (custom responsive design system, CSS variables), Vanilla JavaScript (no heavy frontend frameworks required).
- **Testing & Verification**: Pytest, Playwright (end-to-end browser tests across viewports).
- **Containerization & Deployment**: Docker (multi-stage Python 3.13-slim), Gunicorn.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11 to 3.13 installed.
- Git (optional).

### 2. Setup Virtual Environment
```bash
# Clone or navigate to the project directory
cd "EcoSort AI"

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### 4. Run the Application
```bash
python app.py
```
Open your browser at **`http://127.0.0.1:5000`** (or the port displayed in terminal).

---

## 🧪 Running Tests

The test suite contains **62 tests** covering database operations, API endpoints, upload validation, security headers, rate limiting, and full Playwright end-to-end browser flows:

```bash
# Run unit and integration tests (fast, ~4s)
python -m pytest tests/test_*.py -v

# Run browser E2E tests with Playwright (~35s)
python -m pytest tests/e2e -v

# Run the entire test suite
python -m pytest
```

---

## 🔄 Machine Learning Pipeline

### Data Preparation
The dataset was curated from TrashNet and organic waste image datasets, preprocessed into standardized 224x224 RGB images with a stratified **70% train / 15% val / 15% test** split:
- `train`: 2,376 images
- `val`: 507 images
- `test`: 509 images

To re-run data ingestion and validation:
```bash
python ml/prepare_data.py
```

### Model Training
Training runs two phases:
1. **Feature Extraction**: Trains dense classification head on frozen MobileNetV2 base with dropout and class balancing.
2. **Fine-Tuning**: Unfreezes top convolutional blocks with reduced learning rate (1e-4) and cosine decay.
3. **Baseline Comparison**: Trains a custom 4-layer convolutional network from scratch.

To re-train:
```bash
python ml/train.py
```

### Model Evaluation
Evaluates the saved model on the held-out test split, updates `ml/model/metrics.json`, and exports confusion matrix and loss/accuracy curves:
```bash
python ml/evaluate.py
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | Healthcheck reporting database status, model status, and error state. |
| `/api/predict` | `POST` | Upload file (`multipart/form-data`) or base64 frame (`application/json`). Returns category, confidence, top-3 predictions, recommendations, and Grad-CAM heatmap data URL. |
| `/api/history` | `GET` | Retrieve classification history. Optional query parameter `?category=<name>`. |
| `/api/history/<id>` | `DELETE` | Delete single prediction item and associated preview image. |
| `/api/history` | `DELETE` | Clear all predictions and delete stored preview images. |
| `/api/stats` | `GET` | Aggregated sorting statistics (total items, most frequent category). |

---

## 🐳 Docker Deployment

Build and run the production container:
```bash
# Build the Docker image
docker build -t ecosort-ai .

# Run container with persistent storage
docker run -d -p 8080:8080 -v ecosort_data:/data --name ecosort ecosort-ai
```
Visit `http://localhost:8080`.

---

## 📁 Project Directory Layout

```
EcoSort AI/
├── app.py                      # Flask application factory, routes & API endpoints
├── config.py                   # Centralized application configuration
├── Dockerfile                  # Production container configuration
├── gunicorn.conf.py            # Production WSGI server settings
├── Procfile                    # Cloud process definition (Heroku / Render)
├── requirements.txt            # Production runtime dependencies
├── requirements-dev.txt        # Development and testing dependencies
├── pytest.ini                  # Pytest configuration & markers
├── database/
│   ├── db.py                   # SQLite data access layer
│   ├── schema.sql              # Database schema & seeded recommendations
│   └── ecosort.db              # SQLite database file
├── ml/
│   ├── model/                  # Model weights (ecosort_model.keras, metrics.json)
│   ├── prepare_data.py         # Dataset preparation & split script
│   ├── preprocess.py           # Shared image preprocessing & validation logic
│   ├── train.py                # MobileNetV2 transfer learning & training loop
│   ├── evaluate.py             # Held-out evaluation, metrics, and curve generation
│   ├── predict.py              # Model inference & prediction service
│   └── gradcam.py              # Grad-CAM heatmap generation
├── static/
│   ├── css/styles.css          # Design system & responsive styles
│   ├── fonts/                  # Self-hosted WOFF2 web fonts (Sora & Inter)
│   ├── img/                    # Icons, logo, confusion matrix, curves
│   └── js/                     # Modular frontend logic (main, classify, history, learn)
├── templates/                  # Jinja2 HTML templates (home, classify, history, etc.)
├── tests/                      # Unit, API, database, and Playwright e2e tests
└── screenshots/                # Responsive screenshots (375px, 768px, 1440px)
```

---

## 📜 Academic Integrity & Project Credit
This project was developed for **Emerging Technologies: A Primer (22ETP58)**. All metrics and evaluation outputs presented in this repository are from actual test runs on held-out data.
