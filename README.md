# 🛡️ DeepDetect AI - Enterprise Deepfake Video Detection System

State-of-the-art Spatial-Temporal Deepfake Forensics Center built with a
tested **InsightFace SCRFD** face detector and selectable **ResNet-18 +
Bidirectional LSTM**, **EfficientNet-V2 + BiLSTM**, or **MobileNet-V3 +
XGBoost** video classifiers.

Featuring a **Cyber-Forensics Dark Command Center** web application with full artifact inspection, side-by-side comparative forensics, batch threat scanning, and cryptographic PDF audit reports.

---

## 🌟 Key Frontend Features & Capabilities

### 1. 🎯 Deep Forensic Scanner
- **One-Click Quick-Load Presets**: Instantly load and test synthetic deepfakes (`Celeb-synthesis`), authentic human subjects (`Celeb-real`), or streaming media (`YouTube-real`).
- **Video Inputs**: Upload `.mp4`, `.avi`, `.mov`, `.webm`, `.mpeg`, `.mpg`, or `.m4v` files (up to 500 MB), or import a public video page/direct video-file URL supported by yt-dlp.
- **Evidence Telemetry HUD**: Resolution, framerate, video duration, file size, SHA-256 and MD5 cryptographic hashes.
- **Qualified Assessment**: Screening signal and score with clear uncertainty language; no automatic authenticity proof.
- **Interactive Semi-Circular Gauge**: Real-time synthetic manipulation probability score with adjustable sensitivity threshold.
- **Temporal Anomaly Timeline**: Frame-by-frame anomaly area chart identifying the exact temporal moments of facial distortion.
- **Facial Sequence Reel**: Interactive grid of cropped facial keyframes with per-frame anomaly risk badges.

### 2. 🔬 Artifact Laboratory (FFT & ELA)
- **2D Fast Fourier Transform (FFT) Spectrum**: Reveals hidden periodic grid artifacts and high-frequency noise typical of GAN and diffusion upsampling convolution layers.
- **Error Level Analysis (ELA)**: Simulates JPEG compression discrepancy to detect localized manipulation and splicing seams across facial regions.
- **Laplacian Edge & Texture Gradient**: Analyzes facial contour variance to identify boundary feathering and unnatural facial softening.
- **Inter-Frame Temporal Jitter**: Quantifies frame-to-frame pixel mean squared error to detect temporal flicker.

### 3. ⚖️ Side-by-Side Dual Video Comparator
- Synchronized comparison of a suspect video alongside an authentic reference video.
- Delta separation metric ($\Delta$) measuring statistical disparity.
- Overlaid dual temporal probability curves for side-by-side inspection.

### 4. 📂 Batch Forensic Scanner
- Automated batch evaluation for video archives and threat feeds (customizable batch size).
- Summary KPIs: Total Scanned, Synthetic Detection Ratio, Authentic Passes, and Average Processing Speed.
- Interactive sortable data matrix with one-click **CSV Report Export**.

### 5. 📄 Audit Dossier & Cryptographic PDF Report
- Generates official, tamper-evident forensic certificates using **ReportLab**.
- Includes Case ID, Lead Examiner, Agency, Hashes, Quantitative Metrics, Extracted Keyframe Evidence, and Chain-of-Custody Signatures.

### 6. 🧠 Neural Architecture & Explainability
- Complete mathematical breakdown of spatial convolution ($224 \times 224 \to 512\text{-dim}$) and temporal recurrent modeling ($512\text{-dim} \to \text{Bi-LSTM}$).
- Test set confusion matrix, accuracy metrics, and loss formulation.

### 7. 📊 Reported Evaluation Results

The best recorded validation checkpoint was restored from epoch 7:

| Checkpoint metric | Result |
|---|---:|
| Best validation accuracy | **85.00%** |
| Validation loss | **0.4210** |
| Restored checkpoint | **Epoch 7** |

This validation result is separate from the test-set results below. It
describes performance on the validation split used during training and does
not guarantee the same accuracy on unseen videos.

The current evaluation confusion matrix contains 45 test videos:

| Metric | Result |
|---|---:|
| Accuracy | **66.67%** (30/45) |
| Balanced accuracy | **67.20%** |
| Real-video recall | **43.48%** (10/23) |
| Fake-video recall | **90.91%** (20/22) |

The detector identifies most fake videos in this evaluation, but it also
misclassifies authentic videos as fake (13 false positives). Treat predictions
as an automated screening signal rather than definitive forensic proof. A low
score does not verify that the depicted event is authentic.

---

## 🚀 Quick Setup & How to Run

### 0. Install Dependencies
```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 1. Launch the API
```bash
uvicorn src.api:app --reload
```

### 2. Launch the React application
In a second terminal:
```bash
cd frontend
npm run dev
```
- Open `http://localhost:5173` in your browser. The Vite development server proxies API requests to FastAPI on port 8000.
- Select a local video or paste a public video page/direct public video-file URL, then review the queued analysis and results. Public pages supported by yt-dlp may be resolved automatically; login-protected, private, DRM, and age-restricted pages are not bypassed.

To serve the built frontend from FastAPI instead, run `npm run build` in `frontend`, then start the API without `--reload`; the API serves `frontend/dist` when it exists.

### Deploy on Render

This repository includes [`render.yaml`](./render.yaml), which creates a
FastAPI API service and a separate Render Static Site for the React frontend.
In Render, choose **New Blueprint Instance** and select this repository.
The Blueprint sets the frontend service's `VITE_API_URL` to
`https://deepdetect-api.onrender.com`. If Render assigns a different public
API URL, update that environment variable on the frontend Static Site and
redeploy it.

For the API service, the settings are:

```text
Build command:
python -m pip install --upgrade pip && python -m pip install -r requirements.txt

Start command:
gunicorn src.api:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

Set the API health check path to `/api/health`. The trained
`artifacts_full/mobilenet_xgboost.joblib` file must be committed to the
repository or downloaded during the Render build; otherwise the API cannot
load the selected production artifact. Render's free filesystem is ephemeral,
so use persistent storage if uploaded videos and analysis history must survive
deploys or restarts.

For the frontend Static Site, use:

```text
Root directory: frontend
Build command: npm ci && npm run build
Publish directory: dist (relative to the `frontend` root directory)
Environment variable: VITE_API_URL=https://deepdetect-api.onrender.com
```

### Importing videos

In the **DeepDetect workspace**, choose either a local file or a direct link to a
public video file. Link import supports direct HTTP(S) media responses, with
public-address checks, redirect validation, and a 500 MB limit. A page URL
from a streaming or social platform is not necessarily a media URL; links that
require sign-in, DRM, or platform-specific extraction are not fetched. Upload
an authorised export or use an official integration when available.

The analysis is a screening result, not proof or a legal verdict. Sampled face
scores are review cues, not validated timestamps pinpointing manipulation.

---

## 📂 Project Structure
- `src/preprocessing.py`: Uniform temporal sampling and pluggable face extraction.
- `src/face_detector.py`: InsightFace SCRFD detector; Haar is an explicit development-only fallback.
- `src/model.py`: ResNet-18 + Bi-LSTM spatial-temporal neural detector architecture.
- `src/models_zoo.py`: Multi-model zoo featuring EfficientNet-V2-S + BiLSTM and MobileNet-V3 + XGBoost.
- `src/inference.py`: Model loader, Grad-CAM integration, and sequence probability inference engine.
- `src/gradcam.py`: Explainability layer computing gradient-weighted class activation maps on suspect keyframes.
- `src/forensics.py`: 2D FFT spectrum, Error Level Analysis (ELA), Laplacian sharpness, and video hashing.
- `src/auth.py`: JWT-based user authentication, password hashing, and user account management.
- `src/database.py`: Durable SQLite persistence layer for jobs, user accounts, and batch workflows.
- `src/ingestion.py`: High-concurrency video validation and batch ingest pipeline.
- `src/report.py`: ReportLab engine for cryptographic forensic audit PDF reports with Grad-CAM overlays.
- `src/train.py`: Training loop with class-imbalance weighting and BCE loss.
- `src/benchmark.py`: Reproducible training, held-out evaluation, artifact saving, and winner selection for all three model candidates.
- `frontend/`: Modern React 19 + TypeScript + Vite user interface with dark cybersecurity aesthetics.
- `src/api.py`: FastAPI endpoints for auth, batch uploads, single video analysis, Grad-CAM previews, and PDF export.
- `tests/`: Automated test suite covering training balance, forensics, auth, and database persistence.
- `deepfake_detection.ipynb`: Documented step-by-step training and evaluation notebook.
- `confusion_matrix.png`: Model validation confusion matrix plot.

## Upgrade and model selection

Install the production face detector dependencies from `requirements.txt`, then
run the benchmark from the notebook or terminal:

```bash
python -m src.benchmark --data-dir data --output-dir artifacts_full --limit-per-class 150 --epochs 1 --num-frames 8 --no-pretrained
```

The command uses the same stratified train/validation/test split for every
candidate and writes `artifacts_full/benchmark_results.json` plus the trained model
artifacts. The winner is selected by held-out balanced accuracy, with ROC-AUC
as the tie-breaker. Configure the API with the selected artifact:

```powershell
$env:DEEPDETECT_MODEL = "mobilenet_xgboost"
$env:DEEPDETECT_MODEL_PATH = "artifacts_full\mobilenet_xgboost.joblib"
uvicorn src.api:app --reload
```

The API automatically selects the full winner when `artifacts_full/` exists.
`/api/models` reports the active model and artifact slots. `/api/health`
reports the active SCRFD detector. Set `DEEPDETECT_ALLOW_HAAR=1` only for local development when
InsightFace model assets are unavailable; production comparisons should never
silently fall back to Haar.

### Local benchmark result

The completed full balanced CPU benchmark used 150 real and 150 fake videos,
8 SCRFD face frames per video, one training epoch per candidate, and an
identical held-out split. The selected winner is **MobileNet-V3 + XGBoost**:

| Model | Accuracy | Balanced accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| ResNet-18 + BiLSTM | 49.33% | 50.00% | 0.661 | 0.548 |
| EfficientNet-V2 + BiLSTM | 54.67% | 55.19% | 0.673 | 0.647 |
| MobileNet-V3 + XGBoost | **66.67%** | **66.71%** | **0.675** | **0.671** |

The result is stored in `artifacts_full/benchmark_results.json`, and the
backend automatically uses the selected artifact when that file exists. It is
still a project evaluation rather than a publication-quality claim; increase
the dataset and epochs for formal research reporting.

The production MobileNet artifact was subsequently refit with pretrained
ImageNet MobileNet-V3 features before deployment. This avoids using random
backbone features in the CPU-safe comparison run. Its independent held-out
check was 60.00% accuracy, 60.14% balanced accuracy, F1 0.634, and ROC-AUC
0.659. Predictions remain screening signals and can still be wrong on
individual videos.

## Application architecture

The main web application pairs a modern React 19 + TypeScript frontend with a high-performance FastAPI backend:
- **Authentication**: JWT access and refresh token flow with encrypted session cookies/headers.
- **Persistence**: Durable SQLite database (`deepdetect.db`) with Write-Ahead Logging (WAL) ensuring jobs survive server restarts.
- **Explainability**: Frame-level anomaly scanning and automated Grad-CAM saliency heatmaps highlighting manipulated facial regions.
- **Batch Processing**: Parallel multi-video processing with per-file status updates and aggregate analytics.
- **Reporting**: Automated tamper-evident PDF dossier generation for case management and chain of custody.

## Accuracy-focused training defaults

The training pipeline uses a 32-frame temporal sequence by default, with
configurable frame counts for experiments. The pretrained
ResNet-18 backbone is fine-tuned with a learning rate 10x lower than the
temporal and classification heads. Training-only augmentation includes
horizontal flips, small affine pose changes, lighting/color variation,
occasional blur, and grayscale conversion. Early stopping restores the
checkpoint with the best validation accuracy, using validation loss as a
tie-breaker, to limit overfitting.

The reported best checkpoint used 32 sampled face frames per video and
restored epoch 7 with **85.00% validation accuracy**. The API
application supports temporal sequence depths from 8 to 48 frames, including
32, 40, and 48-frame options. If the frame count is changed, retrain and
reevaluate the model before comparing accuracy results.

Run a full training pass with:

```bash
python -m src.train
```

Use a CUDA-enabled PyTorch installation for practical training time. The
API defaults to the same 32-frame sequence and also exposes
additional frame-depth choices for longer videos.
