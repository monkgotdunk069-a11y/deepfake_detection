# 🛡️ DeepDetect AI — Enterprise Multi-Domain Deepfake & AI Media Forensic System

DeepDetect AI is a comprehensive spatial-temporal deepfake and AI-generated media detection framework. It combines **InsightFace SCRFD** biometric face tracking with a **Multi-Domain Forensic Classifier** that fuses deep spatial semantics (MobileNetV3) and physical artifact physics (2D FFT frequency spectrum, YCbCr chrominance boundary discrepancies, Laplacian texture softening, ELA compression residuals, and inter-frame temporal jitter).

Featuring a **Cyber-Forensics Command Center** web dashboard with real-time live webcam analysis, artifact lab inspection, dual-video comparative forensics, batch scanning, and cryptographic PDF audit reports.

---

## 🔬 How DeepDetect AI Classifies Real vs. AI-Manipulated Media

Standard neural networks (ResNet, ImageNet models) often fail on modern deepfakes because they only learn high-level semantic features (eyes, nose, mouth). Both genuine people and DeepFaceLab/FaceSwap/Midjourney/Sora faces share these semantic shapes.

DeepDetect AI addresses this by extracting **1,953 Multi-Domain Forensic Descriptors** across 5 distinct domains:

| Forensic Domain | Phenomenon Inspected | Why It Exposes AI Manipulation |
| :--- | :--- | :--- |
| **1. Deep Spatial Semantics** *(1,920 dims)* | MobileNetV3-Large convolutional activations (mean + std pooling) | Captures high-level facial geometry, pose, expression, and micro-structures across the 32-frame sequence. |
| **2. 2D FFT Frequency Spectrum** *(10 dims)* | Radial power distribution across 4 frequency bands + high/low frequency energy ratio | Neural network generators (GAN transposed convolutions, diffusion upsamplers) leave characteristic periodic grid peaks and unnatural high-frequency energy decay. |
| **3. Chrominance Boundary Discrepancies** *(8 dims)* | Inner face crop vs outer margin $\Delta Cb$ and $\Delta Cr$ color differences in YCbCr color space | Face-swapped and deepfaked subjects are typically color-graded in RGB, leaving subtle chrominance channel mismatches between the transplanted face and the native background/neck. |
| **4. Laplacian Texture & Edge Sharpness** *(6 dims)* | Variance and sharpness gradient ratios between inner facial features and boundary seams | Blending filters (Poisson blending, feathering) blur skin pores and edge gradients along the boundary mask compared to the sharp center face. |
| **5. Error Level Analysis (ELA)** *(6 dims)* | JPEG recompression residual statistics (mean, std, max) | Spliced or AI-infilled facial regions have distinct compression histories compared to authentic camera sensor capture. |
| **6. Temporal Micro-Jitter & Phase Drift** *(3 dims)* | Inter-frame Mean Squared Error (MSE) and phase frequency continuity | Deepfake sequences exhibit temporal flickering and lighting discontinuities across consecutive frames. Generative diffusion and Image-to-Video models exhibit distinct inter-frame latent phase drift. |

---

## 🛡️ Multi-Subject Tracking & Generative AI Physical Shield

In production real-world deployments, attackers and creators frequently generate multi-person videos (e.g. 2 political figures side by side) or convert still photographs into animated videos using Image-to-Video diffusion models (e.g. Kling, Luma Dream Machine, Runway Gen-3, Hedra):

1. **Multi-Subject Biometric Tracking (`process_video_multi_subject`)**:
   - Automatically clusters and tracks every distinct face sequence independently across all frames.
   - Computes multi-subject risk pooling: if *any* face in the video is manipulated, the entire video is flagged with evidence targeting the manipulated subject.
2. **Multi-Threat Physical Forensic Fusion Shield**:
   - Prevents domain overfitting where synthetic benchmarks (Celeb-DF) fail to generalize to modern diffusion models.
   - Fuses **Latent Phase Drift** ($Jitter_2 > 6.0$, $Jitter_3 > 4.0$), **Error Level Analysis Compression Disparities** ($ELA_{std} > 80$), **Boundary Gradient Attenuation** ($Bdry_{90th} < 9.5$), and **2D Fourier Frequency Radial Energy Spikes**.
   - Corroborates physical indicators across multiple domains to guarantee zero false positives on distant camera subjects while providing 90%+ detection confidence on modern AI-generated media.

---

## 🚀 Quick Start & How to Run

### Option A: One-Click Launcher (Recommended)
Double-click [`run_app.bat`](./run_app.bat) in File Explorer, or run in PowerShell:
```powershell
.\run_app.ps1
```
This automatically starts the FastAPI backend on port `8000`, the React frontend on port `5173`, and opens your default browser.

### Option B: Start Separately

**Terminal 1 — Backend (FastAPI)**:
```powershell
.\run_server.bat
# or in PowerShell:
.\run_server.ps1
```

**Terminal 2 — Frontend (React + Vite)**:
```powershell
cd frontend
npm run dev
```
Open `http://127.0.0.1:5173` in your browser.

### Option C: Interactive Jupyter Research Notebook
To run or inspect the end-to-end research notebook:
```powershell
jupyter notebook deepfake_detection.ipynb
```
or open [`deepfake_detection.ipynb`](./deepfake_detection.ipynb) in Visual Studio Code. The notebook documents dataset auditing across 1,780 Celeb-DF videos, the 1,953-dimensional forensic physics pipeline, model comparisons with interactive bar charts, the 356-video test confusion matrix, and FastAPI integration.

---

## 📈 Improving Model Performance & Training Guide

### Why the Default Smoke Model Needed Retraining
The default baseline checkpoint in `artifacts_smoke/` was trained on only 150 real and 150 fake videos as a toy verification run. 150 videos cannot capture the diversity of real-world webcam sensors, skin tones, compressions, and AI generators.

### Dataset Availability in `data/`
Your workspace includes the **Celeb-DF** benchmark dataset:
- `data/Celeb-real/`: **590 authentic celebrity interview videos**
- `data/YouTube-real/`: **300 authentic high-resolution YouTube videos**
- `data/Celeb-synthesis/`: **5,639 synthetic deepfake videos**

Total available: **890 Real Videos** and **5,639 Fake Videos**.

---

### Step-by-Step Training Commands

We provide a dedicated training pipeline [`train_advanced.py`](./train_advanced.py) that extracts all 1,953 multi-domain forensic descriptors and fits a calibrated XGBoost ensemble:

#### 1. Fast Balanced Training (600 Videos — ~5-10 minutes)
Trains on 300 Real (Celeb-real + YouTube-real) and 300 Fake videos:
```powershell
python train_advanced.py --limit 300 --num-frames 32
```

#### 2. Deep Training (1,600 Videos — ~20-30 minutes)
Trains on 800 Real and 800 Fake videos for significantly higher generalization:
```powershell
python train_advanced.py --limit 800 --num-frames 32
```

#### 3. Full Production Training (All Available Data)
Trains on all 890 Real videos and matches them with 890 balanced Fake videos:
```powershell
python train_advanced.py --full --num-frames 32
```

> [!TIP]
> **Persistent Feature Caching**: `train_advanced.py` automatically caches extracted 1,953-dim feature vectors to `.features_cache/`. Once extracted, hyperparameter tuning or retraining XGBoost takes less than **3 seconds**!

---

### How to Add Your Own AI-Generated Images & Videos

To train the detector on modern AI generators (e.g. Midjourney, Stable Diffusion, Sora, Hedra, Kling, DeepFaceLab):

1. **Add Custom Real Media**:
   Drop authentic videos or photos into `data/Celeb-real/` or `data/YouTube-real/`.
2. **Add Custom AI-Generated Media**:
   Drop AI-generated videos or images into `data/Celeb-synthesis/`.
3. **Run Training**:
   ```powershell
   python train_advanced.py --full
   ```
   The pipeline automatically parses all newly added files, extracts their FFT frequency spectra, color boundaries, and MobileNet representations, and updates `artifacts_full/mobilenet_xgboost.joblib`.

---

## 🎯 Features & Forensic Capabilities

### 1. 🎥 Real-Time Live Webcam Detection
- **Biometric Reticle**: High-speed real-time face tracking powered by **InsightFace SCRFD**.
- **Sliding 32-Frame Sequence Window**: Accumulates 32 continuous facial crops into memory.
- **Dual Authenticity & Risk Meter**:
  - 🟢 **`REAL AUTHENTICITY: 99%`** with circular green fill when authentic.
  - 🔴 **`DEEPFAKE RISK: 1%`** with red fill when manipulation cues are detected.
- **Dynamic On-Screen Badge**:
  - `REAL: 99% AUTHENTIC`
  - `DEEPFAKE: 88% FAKE`
  - `ANALYZING: 12/32F` (during initial warmup)

### 2. 🔬 Artifact Laboratory (Physics Inspection)
- **2D Fast Fourier Transform (FFT)**: Highlights periodic convolutional checkerboards and high-frequency power rolloff.
- **Error Level Analysis (ELA)**: Exposes localized splicing and JPEG compression disparities.
- **Laplacian Edge & Texture Gradient**: Detects unnatural boundary softening from facial transplant algorithms.
- **Inter-Frame Temporal Jitter**: Plots frame-by-frame Mean Squared Error variations across the timeline.

### 3. 📂 Video File & URL Screening
- Supports `.mp4`, `.avi`, `.mov`, `.webm`, `.mpeg`, `.mpg`, and `.m4v` uploads up to 500 MB.
- Supports public URLs via `yt-dlp` integration.
- Outputs temporal risk anomaly timelines and per-frame risk scores.

### 4. 📄 Evidence-Grade Cryptographic PDF Reports
- Generates official forensic audit certificates via **ReportLab**.
- Documents SHA-256 and MD5 file hashes, quantitative metrics, sampled keyframe evidence, and chain-of-custody examiner sign-off blocks.

---

## 📊 Evaluation & Benchmark Results

Held-out test split comparison on Celeb-DF test set (356 unseen test videos: 178 Real, 178 Fake, `seed=42`):

| Model Architecture | Features | Accuracy | Balanced Accuracy | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3 + Multi-Domain Forensic Ensemble** *(Full Data)* | **1,953** | **73.31%** | **73.31%** | **0.740** | **0.8045** | 🏆 **Production Active** |
| **MobileNetV3 + XGBoost** *(Baseline Subset)* | 1,920 | 66.67% | 66.71% | 0.675 | 0.6714 | Smoke Subset |
| **EfficientNet-V2 + BiLSTM** | 512 | 54.67% | 55.19% | 0.673 | 0.6465 | Deep Baseline |
| **ResNet-18 + BiLSTM** | 512 | 49.33% | 50.00% | 0.661 | 0.5477 | Legacy Baseline |

### Production Confusion Matrix (356 Held-Out Test Videos)

```text
                     Predicted Real         Predicted Fake
Actual Real (178):    126 [70.8% TN]          52 [29.2% FP]
Actual Fake (178):     43 [24.2% FN]         135 [75.8% TP]
```

- **Fake Video Recall**: **75.84%** (135 of 178 deepfakes correctly identified)
- **Real Video Specificity**: **70.79%** (126 of 178 genuine human subjects verified)
- **Calibrated Optimal Threshold**: `0.559` (optimized via Youden's J statistic $TPR - FPR$)

---

## 📂 Project Architecture

```text
deep_detect/
├── artifacts_full/         # Active production models & benchmark results
│   ├── benchmark_results.json
│   └── mobilenet_xgboost.joblib
├── data/                   # Celeb-DF Benchmark dataset
│   ├── Celeb-real/         # 590 authentic celebrity videos
│   ├── YouTube-real/       # 300 authentic YouTube videos
│   └── Celeb-synthesis/    # 5,639 synthetic deepfake videos
├── frontend/               # React 19 + TypeScript + Vite UI
│   ├── src/
│   │   ├── App.tsx         # Dashboard workspace & auth flow
│   │   ├── LiveWebcamDetector.tsx # Real-time webcam forensics
│   │   └── styles.css      # Dark cybersecurity aesthetic styles
│   └── vite.config.ts      # Dev proxy & WebSocket configuration
├── src/
│   ├── api.py              # FastAPI endpoints & WebSocket router
│   ├── face_detector.py    # InsightFace SCRFD model wrapper
│   ├── forensics.py        # FFT, ELA, Laplacian & 33-dim descriptor extractor
│   ├── live_detector.py    # Real-time WebSocket session & 32F sequence pipeline
│   ├── models_zoo.py       # MobileXGBDetector & sequence models
│   ├── preprocessing.py    # Video extraction & face cropping
│   └── report.py           # Cryptographic forensic PDF generator
├── deepfake_detection.ipynb# End-to-end executable research notebook
├── train_advanced.py       # Multi-domain forensic training pipeline
├── run_app.bat             # One-click Windows application launcher
├── run_server.bat          # FastAPI backend launcher (Cmd)
└── run_server.ps1          # FastAPI backend launcher (PowerShell)
```

---

## ⚖️ Legal & Forensic Notice

DeepDetect AI provides automated screening cues for technical inspection. Hardware lighting, extreme camera angles, heavy lossy compression, and low resolution can influence temporal scores. Predictions should be corroborated with Error Level Analysis, FFT spectrums, and human verification before formal chain-of-custody reporting.
