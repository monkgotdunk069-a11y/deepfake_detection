# 🛡️ DeepDetect AI - Deepfake Video Detection System

End-to-end Deepfake Video Detection pipeline matching the architecture diagram:
- **ResNeXt / ResNet Feature Extraction (CNN)** + **Bidirectional LSTM Temporal Sequence Modeling**
- **Complete Step-by-Step Jupyter Notebook**: [`deepfake_detection.ipynb`](file:///c:/Users/Nishant/OneDrive/Documents/deep_detect/deepfake_detection.ipynb)
- **Interactive Streamlit Web Application**: [`app.py`](file:///c:/Users/Nishant/OneDrive/Documents/deep_detect/app.py)

---

## 🚀 Quick Setup & How to Run

### 0. Install Dependencies
```bash
pip install -r requirements.txt
```

### 1. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
- Upload custom `.mp4`, `.avi`, `.mov` video files, or select from preloaded test datasets (`Celeb-real`, `YouTube-real`, `Celeb-synthesis`).
- Inspect extracted face frames in real-time.
- View the Real/Fake verdict, confidence percentage, and temporal anomaly line graph.

### 2. Run the Jupyter Notebook
Open [`deepfake_detection.ipynb`](file:///c:/Users/Nishant/OneDrive/Documents/deep_detect/deepfake_detection.ipynb) in VS Code or JupyterLab:
- Explains every step: Video frame extraction, Haar cascade face detection, dataset batching, ResNeXt+LSTM model design, training loop with BCE loss, confusion matrix, ROC curves, and single-video inference.

---

## 📂 Project Structure
- `src/preprocessing.py`: Uniform frame extraction and Haar face cropping.
- `src/model.py`: ResNeXt + LSTM spatial-temporal deepfake model architecture.
- `src/train.py`: Training and evaluation pipeline.
- `src/inference.py`: End-to-end video inference and temporal scoring engine.
- `app.py`: Streamlit frontend application with glassmorphic dark UI.
- `deepfake_detection.ipynb`: Fully documented notebook.
- `confusion_matrix.png`: Evaluation confusion matrix plot.
