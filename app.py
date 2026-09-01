import os
import sys
import tempfile
import glob
import textwrap

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import load_detector_model, predict_video


st.set_page_config(
    page_title="DeepDetect AI | ResNet18 + LSTM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def html(content: str):
    collapsed = " ".join(line.strip() for line in content.strip().splitlines())
    st.markdown(collapsed, unsafe_allow_html=True)


html(
    """
    <style>

    .stApp {
        background:
        radial-gradient(
            circle at 10% 20%,
            rgba(25, 30, 45, 1) 0%,
            rgba(10, 12, 20, 1) 85%
        );
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 17px;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    .section-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .fake-card {
        background: rgba(127, 29, 29, 0.30);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
    }

    .real-card {
        background: rgba(6, 78, 59, 0.30);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
    }

    .fake-title {
        color: #EF4444;
        font-size: 30px;
        font-weight: 800;
    }

    .real-title {
        color: #10B981;
        font-size: 30px;
        font-weight: 800;
    }

    .small-text {
        color: #94A3B8;
        font-size: 13px;
    }

    </style>
    """
)


with st.sidebar:

    html(
        """
        <div style="
            font-size:55px;
            text-align:center;
            margin-bottom:5px;
        ">
            🛡️
        </div>
        """
    )

    st.markdown(
        "<h2 style='text-align:center;'>DeepDetect AI</h2>",
        unsafe_allow_html=True
    )

    st.caption(
        "AI-powered Deepfake Video Forensics"
    )

    st.divider()

    st.markdown("### ⚙️ Model Settings")

    st.caption("Feature Extractor")

    backbone_choice = st.selectbox(
        "Backbone",
        ["ResNet-18"],
        index=0,
        label_visibility="collapsed"
    )

    backbone_type = "resnet18"

    st.caption("Temporal Sequence Length")

    seq_frames = st.slider(
        "Frames",
        min_value=8,
        max_value=32,
        value=16,
        step=4,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 📁 Input Method")

    input_mode = st.radio(
        "Choose Video Source",
        [
            "Upload Custom Video",
            "Select Sample Dataset Video"
        ]
    )

    selected_sample = None

    if input_mode == "Select Sample Dataset Video":

        real_samples = (
            glob.glob(
                os.path.join(
                    PROJECT_ROOT,
                    "data",
                    "Celeb-real",
                    "*.mp4"
                )
            )
            +
            glob.glob(
                os.path.join(
                    PROJECT_ROOT,
                    "data",
                    "YouTube-real",
                    "*.mp4"
                )
            )
        )

        fake_samples = glob.glob(
            os.path.join(
                PROJECT_ROOT,
                "data",
                "Celeb-synthesis",
                "*.mp4"
            )
        )

        sample_category = st.radio(
            "Sample Type",
            [
                "Real Human",
                "Deepfake Synthetic"
            ]
        )

        if sample_category == "Real Human":
            options = real_samples[:20]
        else:
            options = fake_samples[:20]

        if options:

            selected_sample = st.selectbox(
                "Select Test File",
                options,
                format_func=lambda x: os.path.basename(x)
            )

        else:

            st.warning(
                "No sample videos found in the dataset folders."
            )

    st.divider()

    st.markdown("### 🧠 Model Pipeline")

    st.markdown(
        """
        **1. Video Frames**

        Uniform temporal sampling

        **2. Face Detection**

        Facial region extraction

        **3. ResNet-18 CNN**

        Spatial feature extraction

        **4. Bi-LSTM**

        Temporal sequence analysis

        **5. Binary Classifier**

        Real / Fake prediction
        """
    )

    st.divider()


try:

    @st.cache_resource
    def get_cached_model():

        return load_detector_model(
            backbone_type="resnet18"
        )

    model, device = get_cached_model()

    device_name = (
        "CUDA GPU"
        if getattr(device, "type", "") == "cuda"
        else "CPU"
    )

    st.sidebar.success(
        f"✓ Model Ready\n\n"
        f"ResNet-18 + Bi-LSTM\n"
        f"Running on {device_name}"
    )

except Exception as e:

    model = None
    device = None

    st.sidebar.error(
        "Model loading failed."
    )

    st.sidebar.exception(e)


html(
    f"""
    <div style="
        background:linear-gradient(
            135deg,
            rgba(30,41,59,0.75),
            rgba(15,23,42,0.85)
        );
        border:1px solid rgba(255,255,255,0.10);
        border-radius:18px;
        padding:28px;
        margin-bottom:25px;
    ">
        <div style="
            display:inline-block;
            background:rgba(99,102,241,0.20);
            color:#A5B4FC;
            border:1px solid rgba(99,102,241,0.35);
            padding:5px 12px;
            border-radius:20px;
            font-size:12px;
            font-weight:700;
            margin-bottom:10px;
        ">
            AI FORENSICS & COMPUTER VISION
        </div>

        <div style="
            font-size:38px;
            font-weight:800;
            color:#F8FAFC;
            margin-bottom:8px;
        ">
            Spatial-Temporal Deepfake Detection
        </div>

        <div style="
            color:#94A3B8;
            font-size:16px;
            line-height:1.6;
        ">
            Analyze facial manipulation and synthetic video artifacts
            using a ResNet-18 CNN combined with a Bidirectional LSTM.
        </div>
    </div>
    """
)


def get_video_metadata(video_path):

    try:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if not fps or fps <= 0:
            fps = 30.0

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration = (
            total_frames / fps
            if total_frames > 0
            else 0
        )

        keyframes = []

        if total_frames > 0:

            indices = [
                0,
                int(total_frames * 0.25),
                int(total_frames * 0.50),
                int(total_frames * 0.75),
                total_frames - 1
            ]

            indices = sorted(
                set(
                    max(
                        0,
                        min(
                            total_frames - 1,
                            i
                        )
                    )
                    for i in indices
                )
            )

            for index in indices:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    index
                )

                ret, frame = cap.read()

                if ret and frame is not None:

                    frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    keyframes.append(
                        (index, frame)
                    )

        cap.release()

        return {
            "frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "resolution": f"{width} × {height}",
            "duration": duration,
            "keyframes": keyframes
        }

    except Exception:

        return None


video_path = None
video_name = None
video_size_mb = 0.0


if input_mode == "Upload Custom Video":

    st.markdown("### 📤 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video for forensic analysis",
        type=[
            "mp4",
            "avi",
            "mov",
            "webm",
            "mpeg4"
        ]
    )

    if uploaded_file is not None:

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        if not suffix:
            suffix = ".mp4"

        temporary_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        file_data = uploaded_file.getvalue()

        temporary_file.write(
            file_data
        )

        temporary_file.close()

        video_path = temporary_file.name

        video_name = uploaded_file.name

        video_size_mb = (
            len(file_data)
            / (1024 * 1024)
        )

else:

    if selected_sample:

        video_path = selected_sample

        video_name = os.path.basename(
            selected_sample
        )

        try:

            video_size_mb = (
                os.path.getsize(
                    selected_sample
                )
                / (1024 * 1024)
            )

        except Exception:

            video_size_mb = 0.0


if video_path:

    metadata = get_video_metadata(
        video_path
    )

    st.divider()

    left, right = st.columns(
        [1.05, 0.95],
        gap="large"
    )

    with left:

        st.markdown(
            "### 🎥 Source Video"
        )

        st.markdown(
            f"**{video_name}**"
        )

        if metadata:

            m1, m2, m3, m4 = st.columns(4)

            with m1:

                st.metric(
                    "Size",
                    f"{video_size_mb:.2f} MB"
                )

            with m2:

                st.metric(
                    "Resolution",
                    metadata["resolution"]
                )

            with m3:

                st.metric(
                    "Duration",
                    f"{metadata['duration']:.1f}s"
                )

            with m4:

                st.metric(
                    "FPS",
                    f"{metadata['fps']:.1f}"
                )

            st.markdown("")

            if metadata["keyframes"]:

                middle = (
                    len(metadata["keyframes"]) // 2
                )

                _, preview = metadata[
                    "keyframes"
                ][middle]

                st.image(
                    preview,
                    caption="Video Preview",
                    use_container_width=True
                )

                st.markdown(
                    "**Timeline Preview**"
                )

                preview_cols = st.columns(
                    min(
                        4,
                        len(
                            metadata[
                                "keyframes"
                            ]
                        )
                    )
                )

                for i, (
                    frame_index,
                    frame_image
                ) in enumerate(
                    metadata["keyframes"][:4]
                ):

                    with preview_cols[i]:

                        timestamp = (
                            frame_index
                            / metadata["fps"]
                        )

                        st.image(
                            frame_image,
                            caption=f"{timestamp:.1f}s",
                            use_container_width=True
                        )

        else:

            st.warning(
                "Unable to read video metadata."
            )

    with right:

        st.markdown(
            "### 🔍 Facial Sequence"
        )

        st.info(
            f"""
            The model will analyze **{seq_frames} frames**
            from the video.

            **Backbone:** ResNet-18

            **Temporal Model:** Bidirectional LSTM

            **Device:** {device_name if model is not None else "Unavailable"}
            """
        )

        st.markdown("")

        analyze_btn = st.button(
            "🔍 Analyze Video",
            type="primary",
            use_container_width=True
        )

        if not analyze_btn:

            st.caption(
                "Click the button above to extract facial frames and run the deepfake detector."
            )


    if analyze_btn:

        if model is None:

            st.error(
                "The model could not be loaded."
            )

            st.stop()

        progress = st.progress(
            0,
            text="Preparing video..."
        )

        try:

            progress.progress(
                20,
                text="Extracting video frames..."
            )

            progress.progress(
                40,
                text="Detecting and cropping faces..."
            )

            result = predict_video(
                video_path=video_path,
                model=model,
                device=device,
                num_frames=seq_frames,
                backbone_type="resnet18"
            )

            progress.progress(
                100,
                text="Analysis complete."
            )

        except Exception as e:

            progress.empty()

            st.error(
                "Video analysis failed."
            )

            st.exception(e)

            st.stop()

        progress.empty()

        st.divider()

        st.markdown(
            "## 📊 Forensic Analysis Result"
        )

        is_fake = result["is_fake"]

        fake_probability = (
            result["fake_probability"]
        )

        real_probability = (
            result["real_probability"]
        )

        confidence = result[
            "confidence"
        ]

        verdict_col, probability_col, frames_col = st.columns(
            [1.6, 1, 1]
        )

        with verdict_col:

            if is_fake:

                html(
                    f"""
                    <div class="fake-card">
                        <div class="fake-title">
                            ⚠️ DEEPFAKE DETECTED
                        </div>

                        <div style="
                            color:#FCA5A5;
                            font-size:16px;
                            margin-top:8px;
                        ">
                            Synthetic manipulation detected
                        </div>

                        <div style="
                            color:#F8FAFC;
                            font-size:22px;
                            font-weight:700;
                            margin-top:12px;
                        ">
                            Confidence: {confidence:.2f}%
                        </div>
                    </div>
                    """
                )

            else:

                html(
                    f"""
                    <div class="real-card">
                        <div class="real-title">
                            ✅ REAL / AUTHENTIC
                        </div>

                        <div style="
                            color:#6EE7B7;
                            font-size:16px;
                            margin-top:8px;
                        ">
                            No strong synthetic manipulation detected
                        </div>

                        <div style="
                            color:#F8FAFC;
                            font-size:22px;
                            font-weight:700;
                            margin-top:12px;
                        ">
                            Confidence: {confidence:.2f}%
                        </div>
                    </div>
                    """
                )

        with probability_col:

            st.metric(
                "Deepfake Probability",
                f"{fake_probability * 100:.2f}%"
            )

        with frames_col:

            st.metric(
                "Frames Analyzed",
                str(
                    result["num_frames"]
                )
            )

        st.markdown("")

        result_left, result_right = st.columns(
            [1, 1.5],
            gap="large"
        )

        with result_left:

            st.markdown(
                "### Probability Distribution"
            )

            pie = go.Figure(
                data=[
                    go.Pie(
                        labels=[
                            "Real",
                            "Deepfake"
                        ],
                        values=[
                            real_probability * 100,
                            fake_probability * 100
                        ],
                        hole=0.58,
                        marker=dict(
                            colors=[
                                "#10B981",
                                "#EF4444"
                            ]
                        ),
                        textinfo="label+percent"
                    )
                ]
            )

            pie.update_layout(
                showlegend=False,
                margin=dict(
                    t=20,
                    b=20,
                    l=20,
                    r=20
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color="#E2E8F0"
                )
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )

        with result_right:

            st.markdown(
                "### Temporal Anomaly Analysis"
            )

            frame_scores = result[
                "frame_probabilities"
            ]

            frame_data = pd.DataFrame(
                {
                    "Frame": [
                        i + 1
                        for i in range(
                            len(frame_scores)
                        )
                    ],
                    "Deepfake Probability": [
                        score * 100
                        for score in frame_scores
                    ]
                }
            )

            chart = px.line(
                frame_data,
                x="Frame",
                y="Deepfake Probability",
                markers=True
            )

            chart.add_hline(
                y=50,
                line_dash="dash",
                annotation_text="50% Threshold"
            )

            chart.update_layout(
                yaxis=dict(
                    range=[0, 100],
                    title="Deepfake Probability (%)"
                ),
                xaxis=dict(
                    title="Temporal Frame"
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color="#E2E8F0"
                ),
                margin=dict(
                    t=20,
                    b=30,
                    l=30,
                    r=20
                )
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

        st.divider()

        st.markdown(
            "### 🧑‍💻 Extracted Face Frames"
        )

        face_frames = result[
            "face_frames"
        ]

        if face_frames:

            frame_columns = st.columns(4)

            for i, frame in enumerate(
                face_frames
            ):

                with frame_columns[
                    i % 4
                ]:

                    st.image(
                        frame,
                        caption=f"Frame {i + 1}",
                        use_container_width=True
                    )

        else:

            st.warning(
                "No face frames were returned."
            )

        st.divider()

        with st.expander(
            "📘 How the Detection System Works"
        ):

            architecture_tab, flow_tab, metrics_tab = st.tabs(
                [
                    "🏗️ Architecture",
                    "🔄 Processing Flow",
                    "📈 Metrics"
                ]
            )

            with architecture_tab:

                st.markdown(
                    """
                    ### ResNet-18 + Bidirectional LSTM

                    **ResNet-18 CNN**

                    Each extracted face frame is resized to
                    224 × 224 pixels and passed through ResNet-18
                    to obtain spatial visual features.

                    **Feature Projection**

                    The CNN feature vector is projected into
                    a 512-dimensional representation.

                    **Bidirectional LSTM**

                    The sequence of frame features is passed
                    through the Bi-LSTM to model temporal
                    inconsistencies.

                    **Classification**

                    The temporal representation is passed to
                    the binary classifier to estimate the
                    probability that the video is synthetic.
                    """
                )

            with flow_tab:

                st.markdown(
                    """
                    ### Video Processing Pipeline

                    **Step 1 — Video Input**

                    The uploaded video is loaded.

                    **Step 2 — Frame Sampling**

                    A fixed number of temporal frames are selected.

                    **Step 3 — Face Detection**

                    Facial regions are extracted from the frames.

                    **Step 4 — Image Normalization**

                    Frames are resized and normalized using
                    ImageNet normalization.

                    **Step 5 — Spatial Analysis**

                    ResNet-18 extracts spatial features.

                    **Step 6 — Temporal Analysis**

                    Bi-LSTM analyzes the sequence.

                    **Step 7 — Final Prediction**

                    The classifier produces the final
                    Real / Deepfake probability.
                    """
                )

            with metrics_tab:

                st.markdown(
                    """
                    ### Output Metrics

                    **Deepfake Probability**

                    Probability assigned to the synthetic/deepfake
                    class.

                    **Real Probability**

                    Calculated as:

                    `1 - Deepfake Probability`

                    **Confidence**

                    The probability of the predicted class.

                    **Frame-Level Probability**

                    Probability generated for each analyzed
                    temporal frame.
                    """
                )

else:

    st.markdown("### 🎬 Start a Forensic Analysis")

    empty_left, empty_middle, empty_right = st.columns(
        [1, 2, 1]
    )

    with empty_middle:

        st.info(
            """
            Upload a video from the sidebar area above to begin.

            Supported formats:

            **MP4 • AVI • MOV • WEBM • MPEG4**

            The system will extract facial frames and analyze
            their spatial and temporal characteristics.
            """
        )

st.markdown("---")

st.caption(
    "DeepDetect AI • ResNet-18 + Bidirectional LSTM • Deepfake Video Forensics"
)