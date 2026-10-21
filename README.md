# 🛡️ DeepDetect AI - Enterprise Deepfake Video Detection System

State-of-the-art Spatial-Temporal Deepfake Forensics Center built with a
**fine-tuned ResNet-18 CNN** spatial feature projection and **Bidirectional
LSTM** temporal sequence modeling.

Featuring a **Cyber-Forensics Dark Command Center** web application with full artifact inspection, side-by-side comparative forensics, batch threat scanning, and cryptographic PDF audit reports.

---

## 🌟 Key Frontend Features & Capabilities

### 1. 🎯 Deep Forensic Scanner
- **One-Click Quick-Load Presets**: Instantly load and test synthetic deepfakes (`Celeb-synthesis`), authentic human subjects (`Celeb-real`), or streaming media (`YouTube-real`).
- **Drag & Drop Custom Upload**: Supports `.mp4`, `.avi`, `.mov`, `.webm`, `.mpeg4`.
- **Evidence Telemetry HUD**: Resolution, framerate, video duration, file size, SHA-256 and MD5 cryptographic hashes.
- **Dynamic Threat Verdict**: Glowing visual verdict badge (`CRITICAL: DEEPFAKE DETECTED` vs `VERIFIED: AUTHENTIC HUMAN MEDIA`).
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

The current evaluation confusion matrix contains 45 test videos:

| Metric | Result |
|---|---:|
| Accuracy | **66.67%** (30/45) |
| Balanced accuracy | **67.20%** |
| Real-video recall | **43.48%** (10/23) |
| Fake-video recall | **90.91%** (20/22) |

The detector identifies most fake videos in this evaluation, but it also
misclassifies authentic videos as fake (13 false positives). Treat predictions
as an automated screening signal rather than definitive forensic proof.

---

## 🚀 Quick Setup & How to Run

### 0. Install Dependencies
```bash
pip install -r requirements.txt
```

### 1. Launch the Forensic Web Application
```bash
streamlit run app.py
```
- Open `http://localhost:8501` in your browser.
- Adjust the **Sensitivity Alert Threshold** and **Temporal Sequence Depth** in the sidebar.
- Click **⚡ EXECUTE FORENSIC SCAN** or use the quick preset buttons.

---

## 📂 Project Structure
- `src/preprocessing.py`: Uniform temporal sampling and Haar cascade face extraction.
- `src/model.py`: ResNet-18 + Bi-LSTM spatial-temporal neural detector architecture.
- `src/inference.py`: Model loader and sequence probability inference engine.
- `src/forensics.py`: 2D FFT spectrum, Error Level Analysis (ELA), Laplacian sharpness, and video hashing.
- `src/report.py`: ReportLab engine for cryptographic forensic audit PDF reports.
- `src/train.py`: Training loop with class-imbalance weighting and BCE loss.
- `app.py`: Modern Cyberpunk & Sentinel Dark Theme Streamlit frontend application.
- `deepfake_detection.ipynb`: Documented step-by-step training and evaluation notebook.
- `confusion_matrix.png`: Model validation confusion matrix plot.

## Accuracy-focused training defaults

The training pipeline uses a longer 32-frame temporal sequence, 25 maximum
epochs, and early backbone unfreezing after the first epoch. The pretrained
ResNet-18 backbone is fine-tuned with a learning rate 10x lower than the
temporal and classification heads. Training-only augmentation includes
horizontal flips, small affine pose changes, lighting/color variation,
occasional blur, and grayscale conversion. Early stopping (patience 6) still
restores the best validation-loss checkpoint to limit overfitting.

Run a full training pass with:

```bash
python -m src.train
```

Use a CUDA-enabled PyTorch installation for practical training time. The
Streamlit app defaults to the same 32-frame sequence and also exposes 40- and
48-frame choices for longer videos.
