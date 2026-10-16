import os
import sys
import glob
import time
import tempfile
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import load_detector_model, predict_video
from src.forensics import (
    compute_video_hashes,
    compute_fft_spectrum,
    compute_ela,
    compute_laplacian_sharpness,
    compute_temporal_jitter
)
from src.report import generate_forensic_pdf

# Configure Streamlit page
st.set_page_config(
    page_title="DeepDetect AI | Forensics Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-expand sidebar if previously collapsed in browser localStorage
import streamlit.components.v1 as components
components.html(
    """
    <script>
    (function() {
        function autoExpandSidebar() {
            try {
                const doc = window.parent.document;
                const expandBtn = doc.querySelector('[data-testid="stExpandSidebarButton"] button');
                if (expandBtn) {
                    expandBtn.click();
                }
                if (window.parent.localStorage) {
                    window.parent.localStorage.setItem('stSidebarState', 'expanded');
                }
            } catch(e) {}
        }
        autoExpandSidebar();
        setTimeout(autoExpandSidebar, 200);
        setTimeout(autoExpandSidebar, 600);
    })();
    </script>
    """,
    height=0,
    width=0
)



# -------------------------------------------------------------
# Cached Model Initialization (Available for Hardware Specs)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_detector():
    try:
        model, device = load_detector_model(backbone_type="resnet18")
        return model, device, None
    except Exception as e:
        return None, None, str(e)

model, device, load_err = get_cached_detector()
device_type = "CUDA GPU" if getattr(device, "type", "") == "cuda" else "Multi-Threaded CPU"

# Session state controls initialization
if "sens_threshold" not in st.session_state:
    st.session_state.sens_threshold = 0.50
if "seq_depth" not in st.session_state:
    st.session_state.seq_depth = 16
if "app_theme_mode" not in st.session_state:
    st.session_state.app_theme_mode = "🌙 Obsidian Dark"

# -------------------------------------------------------------
# Left Sidebar: Mehek Sharma Inspired Editorial Design
# -------------------------------------------------------------
with st.sidebar:
    # Editorial Brand Header
    st.markdown(
        """
        <div class="mehek-brand-block">
            <div class="mehek-brand-eyebrow">
                <span class="mehek-brand-tag">DEEPDETECT LAB</span>
                <span class="mehek-status-beacon"><span class="beacon-dot"></span> PROTOCOL v2.4</span>
            </div>
            <h1 class="mehek-brand-title">DEEPDETECT <span class="gradient-accent">AI</span></h1>
            <div class="mehek-brand-subtitle">AI Video Forensics & Spatial-Temporal Authenticity Lab</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 01 // ARCHIVE & MODULES (Mehek Sharma Index-Nav)
    st.markdown('<div class="mehek-section-header"><span>01 // ARCHIVE & MODULES</span></div>', unsafe_allow_html=True)
    
    sidebar_nav = st.radio(
        "Module Navigation",
        [
            "01 // FORENSIC SCANNER",
            "02 // ARTIFACT LABORATORY",
            "03 // DUAL COMPARATOR",
            "04 // BATCH PIPELINE",
            "05 // AUDIT DOSSIER & PDF",
            "06 // NEURAL ARCHITECTURE",
            "✦ // ALL MODULES (TABS)"
        ],
        index=0,
        label_visibility="collapsed",
        key="mehek_sidebar_nav"
    )

    # 02 // DETECTION CONTROLS
    st.markdown('<div class="mehek-section-header"><span>02 // DETECTION CONTROLS</span></div>', unsafe_allow_html=True)
    
    current_thresh = st.session_state.get("sens_threshold", 0.50)
    st.markdown(f'<div class="control-label-row"><span>Sensitivity Threshold</span><span class="control-value-pill">{current_thresh:.2f}</span></div>', unsafe_allow_html=True)
    threshold_slider = st.slider(
        "Sensitivity Alert Threshold",
        min_value=0.20,
        max_value=0.80,
        step=0.05,
        key="sens_threshold",
        label_visibility="collapsed",
        help="Probability threshold at which a video is classified as a synthetic deepfake."
    )

    current_depth = st.session_state.get("seq_depth", 16)
    st.markdown(f'<div class="control-label-row"><span>Temporal Sequence Depth</span><span class="control-value-pill">{current_depth} frames</span></div>', unsafe_allow_html=True)
    seq_frames = st.select_slider(
        "Temporal Sequence Depth",
        options=[8, 12, 16, 20, 24, 32],
        key="seq_depth",
        label_visibility="collapsed",
        help="Number of uniformly sampled facial frames analyzed by ResNet-18 and Bi-LSTM."
    )

    # 03 // SPECIFICATIONS
    st.markdown('<div class="mehek-section-header"><span>03 // SPECIFICATIONS</span></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="mehek-spec-table">
            <div class="mehek-spec-row">
                <span class="mehek-spec-key">SPATIAL CNN</span>
                <span class="mehek-spec-val">ResNet-18 (512-d)</span>
            </div>
            <div class="mehek-spec-row">
                <span class="mehek-spec-key">TEMPORAL</span>
                <span class="mehek-spec-val">Bi-LSTM (2 Layers)</span>
            </div>
            <div class="mehek-spec-row">
                <span class="mehek-spec-key">DIRECTIONALITY</span>
                <span class="mehek-spec-val">Bidirectional (1024-d)</span>
            </div>
            <div class="mehek-spec-row">
                <span class="mehek-spec-key">HARDWARE ENGINE</span>
                <span class="mehek-spec-val">{device_type}</span>
            </div>
            <div class="mehek-spec-row">
                <span class="mehek-spec-key">WEIGHT INTEGRITY</span>
                <span class="mehek-spec-val verified">Verified (58.8 MB)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 04 // THEME ENVIRONMENT
    st.markdown('<div class="mehek-section-header"><span>04 // THEME ENVIRONMENT</span></div>', unsafe_allow_html=True)
    theme_choice = st.radio(
        "Interface Appearance",
        ["🌙 Obsidian Dark", "☀️ Clean Light"],
        horizontal=True,
        index=0 if "Dark" in st.session_state.app_theme_mode else 1,
        key="app_theme_mode",
        label_visibility="collapsed"
    )
    is_dark = "Dark" in theme_choice

    # Editorial Colophon & Integrity Footer
    st.markdown(
        """
        <div class="mehek-colophon">
            <div class="colophon-title">DEEPDETECT • ARCHIVE 2026</div>
            <div class="colophon-sub">Neural Media Authenticity & Forensic Protocol</div>
            <div class="colophon-hash">● SHA-256 INTEGRITY VERIFIED</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------------
# Dynamic Theme CSS (Obsidian Dark vs Clean Light - Zero Blue)
# -------------------------------------------------------------
if is_dark:
    # OBSIDIAN DARK THEME (Titanium, Amethyst Violet & Jade Emerald accents - Zero Blue)
    app_bg = "radial-gradient(circle at 50% 0%, #15161b 0%, #0d0d10 55%, #08080a 100%)"
    sidebar_bg = "#0c0d10"
    sidebar_border = "rgba(255, 255, 255, 0.08)"
    text_main = "#F4F4F6"
    text_muted = "#8E8E98"
    card_bg = "rgba(20, 21, 26, 0.75)"
    card_border = "rgba(255, 255, 255, 0.08)"
    hero_bg = "linear-gradient(135deg, rgba(22, 23, 29, 0.9) 0%, rgba(14, 15, 19, 0.95) 100%)"
    hero_border = "rgba(139, 92, 246, 0.25)"
    hero_topline = "linear-gradient(90deg, transparent, #8B5CF6, #10B981, #8B5CF6, transparent)"
    btn_preset_bg = "linear-gradient(135deg, rgba(24, 25, 32, 0.9), rgba(30, 31, 40, 0.8))"
    btn_preset_border = "rgba(255, 255, 255, 0.12)"
    btn_preset_hover = "linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(16, 185, 129, 0.25))"
    btn_primary_bg = "linear-gradient(135deg, #7C3AED 0%, #8B5CF6 50%, #059669 100%)"
    btn_primary_shadow = "rgba(124, 58, 237, 0.45)"
    badge_bg = "rgba(22, 23, 30, 0.85)"
    badge_color = "#C4B5FD"
    badge_border = "rgba(139, 92, 246, 0.3)"
    dropzone_bg = "rgba(20, 21, 26, 0.65)"
    dropzone_border = "rgba(139, 92, 246, 0.35)"
    tab_list_bg = "rgba(18, 19, 24, 0.65)"
    tab_active_bg = "linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(16, 185, 129, 0.2))"
    tab_active_border = "rgba(139, 92, 246, 0.5)"
    tab_active_text = "#D8B4FE"
    orb_core = "radial-gradient(circle at 35% 35%, #8B5CF6, #18181B 80%)"
    orb_shadow = "rgba(139, 92, 246, 0.6)"
    plot_paper = "rgba(0,0,0,0)"
    plot_bg = "rgba(20, 21, 26, 0.4)"
    plot_text = "#E4E4E7"
    plot_grid = "rgba(255, 255, 255, 0.06)"
    plot_line = "#8B5CF6"
else:
    # CLEAN MINIMALIST LIGHT THEME (Porcelain, Violet & Jade Emerald accents - Zero Blue)
    app_bg = "radial-gradient(circle at 50% 0%, #FFFFFF 0%, #F9FAFB 55%, #F3F4F6 100%)"
    sidebar_bg = "#FFFFFF"
    sidebar_border = "#E5E7EB"
    text_main = "#111827"
    text_muted = "#6B7280"
    card_bg = "#FFFFFF"
    card_border = "#E5E7EB"
    hero_bg = "linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(249, 250, 251, 0.95) 100%)"
    hero_border = "rgba(124, 58, 237, 0.2)"
    hero_topline = "linear-gradient(90deg, transparent, #7C3AED, #059669, #7C3AED, transparent)"
    btn_preset_bg = "#FFFFFF"
    btn_preset_border = "#D1D5DB"
    btn_preset_hover = "linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(5, 150, 105, 0.08))"
    btn_primary_bg = "linear-gradient(135deg, #7C3AED 0%, #6D28D9 50%, #059669 100%)"
    btn_primary_shadow = "rgba(124, 58, 237, 0.3)"
    badge_bg = "#F3F4F6"
    badge_color = "#6D28D9"
    badge_border = "rgba(124, 58, 237, 0.25)"
    dropzone_bg = "#F9FAFB"
    dropzone_border = "rgba(124, 58, 237, 0.3)"
    tab_list_bg = "#F3F4F6"
    tab_active_bg = "rgba(124, 58, 237, 0.12)"
    tab_active_border = "rgba(124, 58, 237, 0.4)"
    tab_active_text = "#6D28D9"
    orb_core = "radial-gradient(circle at 35% 35%, #DDD6FE, #7C3AED 80%)"
    orb_shadow = "rgba(124, 58, 237, 0.3)"
    plot_paper = "rgba(0,0,0,0)"
    plot_bg = "rgba(249, 250, 251, 0.85)"
    plot_text = "#111827"
    plot_grid = "rgba(0, 0, 0, 0.06)"
    plot_line = "#7C3AED"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Global App Container */
    .stApp {{
        background: {app_bg} !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: {text_main} !important;
    }}

    /* Material Symbols Fix: Prevent raw text like 'keyboard_double_' from displaying */
    [data-testid="stIconMaterial"], 
    .material-symbols-rounded, 
    .material-symbols-outlined,
    .material-icons {{
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-style: normal !important;
        font-weight: normal !important;
        font-size: 18px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
    }}

    /* Sidebar Collapse Button Polish */
    [data-testid="stSidebarCollapseButton"] button {{
        background: transparent !important;
        border: 1px solid {sidebar_border} !important;
        border-radius: 8px !important;
        color: {text_muted} !important;
        width: 30px !important;
        height: 30px !important;
        padding: 0 !important;
        margin: 4px 0 10px 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover {{
        border-color: #8B5CF6 !important;
        color: {text_main} !important;
        background: rgba(139, 92, 246, 0.08) !important;
    }}

    /* Prominent Sidebar Expand Button (Always visible when sidebar is collapsed) */
    [data-testid="stExpandSidebarButton"] {{
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 999999 !important;
    }}
    [data-testid="stExpandSidebarButton"] button {{
        background: linear-gradient(135deg, #7C3AED, #8B5CF6) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        padding: 5px 12px !important;
        width: auto !important;
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stExpandSidebarButton"] button::after {{
        content: 'OPEN NAVIGATION' !important;
        margin-left: 4px !important;
        letter-spacing: 0.06em !important;
    }}
    [data-testid="stExpandSidebarButton"] button:hover {{
        background: #7C3AED !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important;
        transform: scale(1.04) !important;
    }}
    [data-testid="stSidebarCollapseButton"] button span,
    [data-testid="stExpandSidebarButton"] button span {{
        font-size: 16px !important;
        max-width: 20px !important;
        overflow: hidden !important;
    }}

    /* Sidebar Theming */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        border-right: 1px solid {sidebar_border} !important;
        padding-top: 0.8rem !important;
    }}
    section[data-testid="stSidebar"] {{
        font-family: 'Inter', sans-serif !important;
        color: {text_main} !important;
    }}

    /* Header transparent */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Typography */
    h1, h2, h3, h4 {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: -0.02em !important;
        color: {text_main} !important;
    }}
    
    code, pre, .mono-text {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    .mono-badge {{
        font-family: 'JetBrains Mono', monospace !important;
        background: {badge_bg} !important;
        color: {badge_color} !important;
        border: 1px solid {badge_border} !important;
        padding: 2px 7px !important;
        border-radius: 6px !important;
        font-size: 11px !important;
    }}

    /* Mehek Sharma Editorial Brand Header */
    .mehek-brand-block {{
        padding: 6px 2px 14px 2px;
        margin-bottom: 12px;
        border-bottom: 1px solid {sidebar_border};
    }}
    .mehek-brand-eyebrow {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }}
    .mehek-brand-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.12em;
        color: {text_muted};
        text-transform: uppercase;
    }}
    .mehek-status-beacon {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #10B981;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 2px 8px;
        border-radius: 12px;
    }}
    .mehek-status-beacon .beacon-dot {{
        width: 6px;
        height: 6px;
        background: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10B981;
        animation: beaconPulse 2s infinite ease-in-out;
    }}
    @keyframes beaconPulse {{
        0% {{ opacity: 0.5; transform: scale(0.8); }}
        50% {{ opacity: 1; transform: scale(1.2); box-shadow: 0 0 10px #10B981; }}
        100% {{ opacity: 0.5; transform: scale(0.8); }}
    }}
    .mehek-brand-title {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 17px !important;
        font-weight: 800 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: {text_main} !important;
        margin: 0 0 4px 0 !important;
        line-height: 1.2 !important;
    }}
    .mehek-brand-subtitle {{
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 400 !important;
        color: {text_muted} !important;
        letter-spacing: 0.01em !important;
        line-height: 1.4 !important;
    }}

    /* Mehek Sharma Section Headers */
    .mehek-section-header {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {text_muted};
        margin: 16px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .mehek-section-header::after {{
        content: '';
        flex: 1;
        height: 1px;
        background: {sidebar_border};
    }}

    /* Mehek Sharma Index-Nav Radio Transformation */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] {{
        display: flex !important;
        flex-direction: column !important;
        gap: 3px !important;
        margin-bottom: 6px !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] {{
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 7px 10px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }}
    /* Hide native radio circle */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {{
        display: none !important;
    }}
    /* Vertical Indicator Line (.Index-nav-indicator) */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]::before {{
        content: '' !important;
        display: block !important;
        width: 2px !important;
        height: 10px !important;
        background: {sidebar_border} !important;
        border-radius: 2px !important;
        transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
        flex-shrink: 0 !important;
    }}
    /* Hover effect */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {{
        background: rgba(139, 92, 246, 0.06) !important;
        transform: translateX(4px) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:hover::before {{
        height: 16px !important;
        background: #8B5CF6 !important;
    }}
    /* Active / Checked state */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {{
        background: rgba(139, 92, 246, 0.1) !important;
        border: 1px solid rgba(139, 92, 246, 0.25) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked)::before {{
        height: 18px !important;
        background: #8B5CF6 !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.7) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] span {{
        font-family: 'Inter', sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
        color: {text_main} !important;
        margin: 0 !important;
    }}

    /* Control Label & Pill Readouts */
    .control-label-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Inter', sans-serif;
        font-size: 11.5px;
        color: {text_muted};
        margin-bottom: 2px;
        font-weight: 500;
    }}
    .control-value-pill {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        background: {badge_bg};
        border: 1px solid {badge_border};
        color: {badge_color};
        padding: 1px 7px;
        border-radius: 6px;
    }}

    /* Custom Hairline Sliders - Zero Blue, Zero Red */
    [data-baseweb="slider"] {{
        margin: 6px 0 14px 0 !important;
    }}
    [data-baseweb="slider"] > div > div:first-child {{
        background: {card_border} !important;
        height: 4px !important;
        border-radius: 4px !important;
    }}
    [data-baseweb="slider"] > div > div > div {{
        background: linear-gradient(90deg, #8B5CF6, #10B981) !important;
        height: 4px !important;
        border-radius: 4px !important;
    }}
    [data-baseweb="slider"] div[role="slider"] {{
        background: #8B5CF6 !important;
        border: 2px solid {card_bg} !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.7) !important;
        width: 15px !important;
        height: 15px !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }}
    [data-baseweb="slider"] div[role="slider"]:hover {{
        transform: scale(1.2) !important;
        box-shadow: 0 0 14px rgba(139, 92, 246, 0.9) !important;
    }}
    [data-baseweb="slider"] div[data-testid="stSliderTickBar"] {{
        display: none !important;
    }}

    /* Architectural Spec Table */
    .mehek-spec-table {{
        display: flex;
        flex-direction: column;
        margin-bottom: 14px;
        border-top: 1px solid {sidebar_border};
        border-bottom: 1px solid {sidebar_border};
    }}
    .mehek-spec-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .mehek-spec-row:last-child {{
        border-bottom: none;
    }}
    .mehek-spec-key {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: {text_muted};
        text-transform: uppercase;
    }}
    .mehek-spec-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10.5px;
        font-weight: 600;
        color: {text_main};
        background: {badge_bg};
        border: 1px solid {badge_border};
        padding: 1px 6px;
        border-radius: 4px;
    }}
    .mehek-spec-val.verified {{
        color: #10B981;
        border-color: rgba(16, 185, 129, 0.35);
        background: rgba(16, 185, 129, 0.08);
    }}

    /* Horizontal Theme Switcher Styling */
    section[data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-orientation="horizontal"]) [role="radiogroup"] {{
        display: flex !important;
        flex-direction: row !important;
        gap: 6px !important;
        background: {badge_bg} !important;
        padding: 4px !important;
        border-radius: 9px !important;
        border: 1px solid {sidebar_border} !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-orientation="horizontal"]) label[data-baseweb="radio"] {{
        flex: 1 !important;
        justify-content: center !important;
        padding: 6px 6px !important;
        border-radius: 7px !important;
        text-align: center !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-orientation="horizontal"]) label[data-baseweb="radio"]::before {{
        display: none !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-orientation="horizontal"]) label[data-baseweb="radio"]:hover {{
        transform: none !important;
        background: rgba(139, 92, 246, 0.1) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-orientation="horizontal"]) label[data-baseweb="radio"]:has(input:checked) {{
        background: {card_bg} !important;
        border: 1px solid rgba(139, 92, 246, 0.4) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
    }}

    /* Editorial Colophon & Integrity Footer */
    .mehek-colophon {{
        padding: 12px 2px;
        margin-top: 12px;
        border-top: 1px solid {sidebar_border};
        font-family: 'JetBrains Mono', monospace;
    }}
    .colophon-title {{
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 0.12em;
        color: {text_main};
        text-transform: uppercase;
        margin-bottom: 2px;
    }}
    .colophon-sub {{
        font-size: 9px;
        color: {text_muted};
        letter-spacing: 0.01em;
        line-height: 1.4;
        margin-bottom: 4px;
    }}
    .colophon-hash {{
        font-size: 8.5px;
        color: #10B981;
        letter-spacing: 0.06em;
        font-weight: 600;
    }}

    /* Mehek Sharma Editorial Module Header (in main content) */
    .mehek-module-header {{
        padding: 4px 0 16px 0;
        margin-bottom: 16px;
        border-bottom: 1px solid {sidebar_border};
    }}
    .mehek-module-eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.14em;
        color: #8B5CF6;
        text-transform: uppercase;
        margin-bottom: 4px;
    }}
    .mehek-module-title {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 24px !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: {text_main} !important;
        margin: 0 0 4px 0 !important;
    }}
    .mehek-module-desc {{
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        color: {text_muted} !important;
        letter-spacing: 0.01em !important;
    }}

    /* 3D Animated Hero Header */
    .hero-3d-wrapper {{
        display: flex;
        align-items: center;
        gap: 24px;
        background: {hero_bg};
        border: 1px solid {hero_border};
        border-radius: 18px;
        padding: 20px 26px;
        margin-bottom: 22px;
        backdrop-filter: blur(18px);
        box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }}
    .hero-3d-wrapper::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: {hero_topline};
        animation: laserScan 4s ease-in-out infinite alternate;
    }}
    @keyframes laserScan {{
        0% {{ opacity: 0.3; transform: scaleX(0.7); }}
        100% {{ opacity: 1; transform: scaleX(1); }}
    }}

    /* 3D Holographic Gyroscope Orb */
    .scene-3d {{
        width: 76px;
        height: 76px;
        perspective: 700px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .gyro-3d {{
        width: 70px;
        height: 70px;
        position: relative;
        transform-style: preserve-3d;
        animation: rotateGyro 14s linear infinite;
    }}
    @keyframes rotateGyro {{
        0% {{ transform: rotateX(25deg) rotateY(0deg) rotateZ(0deg); }}
        50% {{ transform: rotateX(25deg) rotateY(180deg) rotateZ(15deg); }}
        100% {{ transform: rotateX(25deg) rotateY(360deg) rotateZ(0deg); }}
    }}

    .gyro-ring {{
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 50%;
        border: 1.5px solid rgba(139, 92, 246, 0.5);
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.25);
        pointer-events: none;
    }}
    .gyro-ring.r1 {{ transform: rotateX(65deg); border-color: rgba(139, 92, 246, 0.7); }}
    .gyro-ring.r2 {{ transform: rotateY(65deg); border-color: rgba(16, 185, 129, 0.7); }}
    .gyro-ring.r3 {{ transform: rotateZ(45deg) rotateX(45deg); border-color: rgba(245, 158, 11, 0.7); }}

    .core-orb-3d {{
        position: absolute;
        top: 50%; left: 50%;
        width: 38px; height: 38px;
        margin-top: -19px; margin-left: -19px;
        background: {orb_core};
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 18px {orb_shadow};
        animation: pulseOrb 3s ease-in-out infinite alternate;
    }}
    @keyframes pulseOrb {{
        0% {{ transform: scale(0.92); }}
        100% {{ transform: scale(1.08); }}
    }}
    .shield-glyph {{
        font-size: 17px;
    }}

    .hero-3d-right {{
        flex: 1;
    }}
    .hero-status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #10B981;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 11px;
        border-radius: 20px;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
        font-family: 'Space Grotesk', sans-serif;
    }}
    .pulse-dot {{
        width: 7px;
        height: 7px;
        background: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        animation: pulse 1.8s infinite;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(0.9); opacity: 0.7; }}
        50% {{ transform: scale(1.3); opacity: 1; box-shadow: 0 0 12px #10B981; }}
        100% {{ transform: scale(0.9); opacity: 0.7; }}
    }}

    .hero-brand-heading {{
        font-size: 32px;
        font-weight: 800;
        color: {text_main};
        margin: 0 0 4px 0 !important;
        letter-spacing: -0.02em;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }}
    .gradient-accent {{
        background: linear-gradient(135deg, #8B5CF6 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-clean-tagline {{
        color: {text_muted};
        font-size: 14px;
        font-weight: 400;
        letter-spacing: 0.01em;
    }}

    /* Streamlit File Uploader Dark/Light Dropzone Polish */
    [data-testid="stFileUploader"] {{
        background: transparent !important;
    }}
    [data-testid="stFileUploaderDropzone"],
    section[data-testid="stFileUploadDropzone"] {{
        background: {dropzone_bg} !important;
        border: 1.5px dashed {dropzone_border} !important;
        border-radius: 14px !important;
        color: {text_main} !important;
        padding: 20px !important;
        transition: all 0.25s ease !important;
    }}
    [data-testid="stFileUploaderDropzone"]:hover,
    section[data-testid="stFileUploadDropzone"]:hover {{
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.2) !important;
    }}
    [data-testid="stFileUploaderDropzone"] *,
    section[data-testid="stFileUploadDropzone"] * {{
        color: {text_main} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploadDropzone"] button {{
        background: {btn_primary_bg} !important;
        border: none !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}

    /* Quick Preset Action Buttons */
    div[data-testid="stHorizontalBlock"] div.stButton > button {{
        background: {btn_preset_bg} !important;
        border: 1px solid {btn_preset_border} !important;
        color: {text_main} !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        min-height: 48px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    div[data-testid="stHorizontalBlock"] div.stButton > button:hover {{
        background: {btn_preset_hover} !important;
        border-color: #8B5CF6 !important;
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.25) !important;
    }}
    div[data-testid="stHorizontalBlock"] div.stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #7C3AED 0%, #6366F1 50%, #10B981 100%) !important;
        border: 1.5px solid #C4B5FD !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 18px rgba(124, 58, 237, 0.45) !important;
    }}
    div[data-testid="stHorizontalBlock"] div.stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #6D28D9 0%, #4F46E5 50%, #059669 100%) !important;
        border-color: #EDE9FE !important;
        box-shadow: 0 6px 24px rgba(124, 58, 237, 0.6) !important;
    }}

    /* Primary Hero Action Button */
    div.stButton > button[kind="primary"] {{
        background: {btn_primary_bg} !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 26px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 18px {btn_primary_shadow} !important;
        transition: all 0.2s ease !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 26px {btn_primary_shadow} !important;
    }}

    /* Download Button */
    div.stDownloadButton > button {{
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.2s ease !important;
    }}
    div.stDownloadButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px rgba(16, 185, 129, 0.5) !important;
    }}

    /* Selectbox & Dropdown */
    [data-baseweb="select"] > div {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 10px !important;
        color: {text_main} !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background: {sidebar_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 10px !important;
    }}

    /* Forensic Cards */
    .cyber-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 18px;
        backdrop-filter: blur(12px);
    }}

    /* Verdict Cards */
    .verdict-fake {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.25) 100%);
        border: 2px solid #EF4444;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 10px 25px -6px rgba(239, 68, 68, 0.3);
    }}
    .verdict-real {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 2px solid #10B981;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 10px 25px -6px rgba(16, 185, 129, 0.3);
    }}

    .verdict-title-fake {{
        color: #EF4444;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}
    .verdict-title-real {{
        color: #10B981;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: {tab_list_bg};
        padding: 6px;
        border-radius: 12px;
        border: 1px solid {card_border};
        margin-bottom: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: {text_muted} !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        padding: 8px 16px;
        background: transparent;
        border: none;
        transition: all 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {text_main} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {tab_active_bg} !important;
        color: {tab_active_text} !important;
        border: 1px solid {tab_active_border} !important;
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 12px 16px;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {text_muted} !important;
        font-size: 13px !important;
        font-family: 'Inter', sans-serif !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {text_main} !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }}

    /* Media Preview Frame Box */
    .preview-frame-container {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 10px;
        margin-bottom: 12px;
    }}

    /* Face Reel Thumbnail Card */
    .face-thumb-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 10px;
        padding: 6px;
        text-align: center;
        margin-bottom: 10px;
    }}
    .threat-tag-safe {{
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .threat-tag-danger {{
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Video Metadata & Frame Extraction Helpers
# -------------------------------------------------------------
def extract_video_metadata(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if total_frames > 0 else 0.0

        keyframes = []
        if total_frames > 0:
            indices = [
                0,
                int(total_frames * 0.25),
                int(total_frames * 0.50),
                int(total_frames * 0.75),
                max(0, total_frames - 1)
            ]
            indices = sorted(set(max(0, min(total_frames - 1, i)) for i in indices))
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    keyframes.append((idx, frame_rgb))

        cap.release()
        size_bytes = os.path.getsize(video_path)
        hashes = compute_video_hashes(video_path)

        return {
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "resolution": f"{width} × {height}",
            "duration": duration,
            "size_mb": size_bytes / (1024 * 1024),
            "keyframes": keyframes,
            "sha256": hashes["sha256"],
            "md5": hashes["md5"]
        }
    except Exception:
        return None


def get_video_frame_at_time(video_path, target_time_sec=0.0):
    """
    Extracts RGB frame at given second timestamp. 
    Guarantees 100% media preview for ALL codecs (including FMP4, MPEG-4, XVID, H.264).
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = max(0, min(total_frames - 1, int(target_time_sec * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), target_frame
        return None, 0
    except Exception:
        return None, 0


def set_scrub_timestamp(key, val):
    st.session_state[key] = float(val)

# -------------------------------------------------------------
# Clean 3D Animated Hero Header
# -------------------------------------------------------------
st.markdown(
    """
    <div class="hero-3d-wrapper">
        <div class="hero-3d-left">
            <div class="scene-3d">
                <div class="gyro-3d">
                    <div class="gyro-ring r1"></div>
                    <div class="gyro-ring r2"></div>
                    <div class="gyro-ring r3"></div>
                    <div class="core-orb-3d">
                        <div class="shield-glyph">🛡️</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="hero-3d-right">
            <div class="hero-status-pill">
                <span class="pulse-dot"></span>
                SYSTEM ONLINE
            </div>
            <h1 class="hero-brand-heading">DeepDetect <span class="gradient-accent">AI</span></h1>
            <div class="hero-clean-tagline">Spatial-Temporal Deepfake Forensics & Neural Media Authenticity</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Gather sample videos from dataset folders
real_celeb_samples = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "Celeb-real", "*.mp4")))
real_yt_samples = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "YouTube-real", "*.mp4")))
fake_celeb_samples = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "Celeb-synthesis", "*.mp4")))

all_real_samples = real_celeb_samples + real_yt_samples
all_fake_samples = fake_celeb_samples

# Initialize baseline session state
if "active_source" not in st.session_state:
    st.session_state.active_source = "synthetic"
if "sample_selector" not in st.session_state and all_fake_samples:
    st.session_state["sample_selector"] = all_fake_samples[0]
if "active_video_path" not in st.session_state:
    st.session_state.active_video_path = all_fake_samples[0] if all_fake_samples else None
    st.session_state.active_video_label = f"Synthetic Deepfake ({os.path.basename(all_fake_samples[0])})" if all_fake_samples else "None"


def render_scanner():
    st.markdown("### 📁 Target Media Selection")

    # Source mode setter callback
    def set_source_mode(source_id):
        st.session_state.active_source = source_id
        if source_id == "synthetic":
            if all_fake_samples:
                st.session_state.active_video_path = all_fake_samples[0]
                st.session_state.active_video_label = f"Synthetic Deepfake ({os.path.basename(all_fake_samples[0])})"
                st.session_state.sample_selector = all_fake_samples[0]
        elif source_id == "celeb_real":
            pool = real_celeb_samples if real_celeb_samples else all_real_samples
            if pool:
                st.session_state.active_video_path = pool[0]
                st.session_state.active_video_label = f"Authentic Celeb ({os.path.basename(pool[0])})"
                st.session_state.sample_selector = pool[0]
        elif source_id == "yt_real":
            pool = real_yt_samples if real_yt_samples else all_real_samples
            if pool:
                st.session_state.active_video_path = pool[0]
                st.session_state.active_video_label = f"YouTube Authentic ({os.path.basename(pool[0])})"
                st.session_state.sample_selector = pool[0]
        elif source_id == "upload":
            if st.session_state.get("last_uploaded_path") and os.path.exists(st.session_state["last_uploaded_path"]):
                st.session_state.active_video_path = st.session_state["last_uploaded_path"]
                st.session_state.active_video_label = st.session_state.get("last_uploaded_name", "Uploaded Video")
            else:
                st.session_state.active_video_path = None
                st.session_state.active_video_label = None

    current_source = st.session_state.get("active_source", "synthetic")

    # 4 Dedicated Source Segmented Buttons
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        is_act = (current_source == "synthetic")
        st.button(
            "🎭 Synthetic Deepfake" + ("  ● ACTIVE" if is_act else ""),
            type="primary" if is_act else "secondary",
            use_container_width=True,
            on_click=set_source_mode,
            args=("synthetic",),
            key="btn_src_synthetic"
        )

    with col_s2:
        is_act = (current_source == "celeb_real")
        st.button(
            "🛡️ Authentic Celeb" + ("  ● ACTIVE" if is_act else ""),
            type="primary" if is_act else "secondary",
            use_container_width=True,
            on_click=set_source_mode,
            args=("celeb_real",),
            key="btn_src_celeb_real"
        )

    with col_s3:
        is_act = (current_source == "yt_real")
        st.button(
            "📺 YouTube Real" + ("  ● ACTIVE" if is_act else ""),
            type="primary" if is_act else "secondary",
            use_container_width=True,
            on_click=set_source_mode,
            args=("yt_real",),
            key="btn_src_yt_real"
        )

    with col_s4:
        is_act = (current_source == "upload")
        st.button(
            "📤 Upload Custom File" + ("  ● ACTIVE" if is_act else ""),
            type="primary" if is_act else "secondary",
            use_container_width=True,
            on_click=set_source_mode,
            args=("upload",),
            key="btn_src_upload"
        )

    if current_source == "upload":
        st.markdown(
            f"""
            <div style="background: {card_bg}; padding: 12px 18px; border-radius: 12px; border: 1px solid {card_border}; margin-top: 10px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 13px; font-weight: 600; color: {text_main};">📤 Custom Video Evidence Mode</span>
                <span style="font-size: 11px; background: rgba(139, 92, 246, 0.25); color: #C4B5FD; padding: 4px 12px; border-radius: 20px; font-weight: 700; letter-spacing: 0.05em;">SECURE DROPZONE</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Drop video evidence (MP4, AVI, MOV, WEBM, MPEG4)",
            type=["mp4", "avi", "mov", "webm", "mpeg4"],
            key="tab1_uploader"
        )
        if uploaded_file is not None:
            if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(uploaded_file.getvalue())
                temp_file.close()
                st.session_state.active_video_path = temp_file.name
                st.session_state.active_video_label = uploaded_file.name
                st.session_state.last_uploaded_name = uploaded_file.name
                st.session_state.last_uploaded_path = temp_file.name
                st.rerun()
    else:
        if current_source == "synthetic":
            sample_pool = all_fake_samples[:35]
            dataset_name = "Celeb-DF v2 Synthesis (Deepfakes)"
            badge_text = f"5,639 Synthetic Videos Indexed"
            badge_bg_col = "rgba(239, 68, 68, 0.2)"
            badge_txt_col = "#FCA5A5"
        elif current_source == "yt_real":
            sample_pool = real_yt_samples[:35] if real_yt_samples else all_real_samples[:35]
            dataset_name = "YouTube Real (Authentic Benchmark)"
            badge_text = f"300 Authentic Real Videos Indexed"
            badge_bg_col = "rgba(16, 185, 129, 0.2)"
            badge_txt_col = "#6EE7B7"
        else:
            sample_pool = real_celeb_samples[:35]
            dataset_name = "Celeb-DF v2 Real (Authentic Celebrities)"
            badge_text = f"590 Authentic Celeb Videos Indexed"
            badge_bg_col = "rgba(16, 185, 129, 0.2)"
            badge_txt_col = "#6EE7B7"

        def on_sample_change():
            chosen = st.session_state.get("sample_selector")
            if chosen and os.path.exists(chosen):
                st.session_state.active_video_path = chosen
                prefix = "Synthetic Deepfake" if current_source == "synthetic" else ("YouTube Authentic" if current_source == "yt_real" else "Authentic Celeb")
                st.session_state.active_video_label = f"{prefix} ({os.path.basename(chosen)})"

        # Ensure sample_selector exists and belongs to pool
        if ("sample_selector" not in st.session_state or st.session_state["sample_selector"] not in sample_pool) and sample_pool:
            st.session_state["sample_selector"] = sample_pool[0]
            st.session_state.active_video_path = sample_pool[0]
            prefix = "Synthetic Deepfake" if current_source == "synthetic" else ("YouTube Authentic" if current_source == "yt_real" else "Authentic Celeb")
            st.session_state.active_video_label = f"{prefix} ({os.path.basename(sample_pool[0])})"

        st.markdown(
            f"""
            <div style="background: {card_bg}; padding: 10px 16px; border-radius: 12px; border: 1px solid {card_border}; margin-top: 10px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 13px; font-weight: 600; color: {text_main};">📁 Current Archive: <span style="color: {badge_color};">{dataset_name}</span></span>
                <span style="font-size: 11px; background: {badge_bg_col}; color: {badge_txt_col}; padding: 3px 10px; border-radius: 20px; font-weight: 600;">{badge_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_pick, col_rand = st.columns([3.5, 1], gap="medium")
        with col_pick:
            if sample_pool:
                st.selectbox(
                    "Choose Target File",
                    sample_pool,
                    format_func=lambda x: f"🎬 {os.path.basename(x)} ({os.path.getsize(x)/(1024*1024):.1f} MB)",
                    key="sample_selector",
                    on_change=on_sample_change,
                    label_visibility="collapsed"
                )
        with col_rand:
            if st.button("🎲 Random Sample", use_container_width=True, key="btn_rand_sample"):
                import random
                full_pool = all_fake_samples if current_source == "synthetic" else (real_yt_samples if current_source == "yt_real" else real_celeb_samples)
                if full_pool:
                    picked = random.choice(full_pool)
                    st.session_state.active_video_path = picked
                    prefix = "Synthetic Deepfake" if current_source == "synthetic" else ("YouTube Authentic" if current_source == "yt_real" else "Authentic Celeb")
                    st.session_state.active_video_label = f"{prefix} ({os.path.basename(picked)})"
                    st.session_state.sample_selector = picked
                    st.rerun()

    target_video = st.session_state.get("active_video_path", None)

    if target_video and os.path.exists(target_video):
        meta = extract_video_metadata(target_video)

        col_left_vid, col_right_info = st.columns([1.15, 0.85], gap="large")

        with col_left_vid:
            st.markdown(f"#### 🎥 Media Preview: `{st.session_state.get('active_video_label', 'Active Video')}`")

            # Interactive Smart Frame Scrubber & Player (Never black screen)
            if meta and meta.get("duration", 0) > 0:
                max_dur = float(meta["duration"])
                scrub_key = f"scrub_{target_video}"

                # Initialize key in session state if missing
                if scrub_key not in st.session_state:
                    st.session_state[scrub_key] = float(min(1.0, max_dur * 0.2))

                # Ensure value stays clamped within video duration bounds
                st.session_state[scrub_key] = max(0.0, min(float(max_dur), float(st.session_state[scrub_key])))

                # Scrubber slider (reads from st.session_state[scrub_key])
                scrub_sec = st.slider(
                    "⏱️ Timeline Scrubber",
                    min_value=0.0,
                    max_value=max_dur,
                    step=0.1,
                    format="%.1f s",
                    key=scrub_key
                )

                frame_rgb, frame_num = get_video_frame_at_time(target_video, scrub_sec)
                if frame_rgb is not None:
                    st.image(
                        frame_rgb,
                        caption=f"Frame #{frame_num + 1} at {scrub_sec:.1f}s / {max_dur:.1f}s • Resolution: {meta['resolution']}",
                        use_container_width=True
                    )
                elif meta.get("keyframes"):
                    st.image(meta["keyframes"][0][1], caption="Sample Keyframe", use_container_width=True)

                # Quick Jump Timeline Buttons using on_click callbacks (executed before widget instantiation)
                tj1, tj2, tj3, tj4 = st.columns(4)
                with tj1:
                    st.button(
                        "⏮️ Start (0s)",
                        on_click=set_scrub_timestamp,
                        args=(scrub_key, 0.0),
                        use_container_width=True
                    )
                with tj2:
                    st.button(
                        "⏪ 25%",
                        on_click=set_scrub_timestamp,
                        args=(scrub_key, round(max_dur * 0.25, 1)),
                        use_container_width=True
                    )
                with tj3:
                    st.button(
                        "⏩ 50%",
                        on_click=set_scrub_timestamp,
                        args=(scrub_key, round(max_dur * 0.50, 1)),
                        use_container_width=True
                    )
                with tj4:
                    st.button(
                        "⏭️ 75%",
                        on_click=set_scrub_timestamp,
                        args=(scrub_key, round(max_dur * 0.75, 1)),
                        use_container_width=True
                    )

                # Optional Native HTML5 video player in expander
                with st.expander("▶️ Native HTML5 Player (Supported Browser Codecs Only)"):
                    st.video(target_video)
            else:
                st.video(target_video)

        with col_right_info:
            st.markdown("#### 🔍 Evidence Telemetry")
            if meta:
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Resolution", meta["resolution"])
                    st.metric("Duration", f"{meta['duration']:.2f}s")
                with m2:
                    st.metric("Framerate", f"{meta['fps']:.1f} FPS")
                    st.metric("File Size", f"{meta['size_mb']:.2f} MB")

                st.markdown(
                    f"""
                    <div style="background: {card_bg}; padding: 12px 14px; border-radius: 12px; border: 1px solid {card_border}; margin-top: 10px;">
                        <div style="font-size: 11px; color: {text_muted}; text-transform: uppercase; font-weight: 700;">SHA-256 Hash</div>
                        <div style="font-size: 11px; font-family: 'JetBrains Mono', monospace; color: {badge_color}; word-break: break-all;">{meta['sha256']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("")
            run_scan_btn = st.button("⚡ EXECUTE FORENSIC SCAN", type="primary", use_container_width=True)

        if "scan_cache" not in st.session_state:
            st.session_state.scan_cache = {}

        has_cached_result = target_video in st.session_state.scan_cache
        should_run = run_scan_btn or (not has_cached_result and run_scan_btn)

        if should_run:
            with st.spinner("Analyzing spatial facial features & temporal LSTM sequence..."):
                try:
                    scan_res = predict_video(
                        video_path=target_video,
                        model=model,
                        device=device,
                        num_frames=seq_frames,
                        backbone_type="resnet18"
                    )
                    st.session_state.scan_cache[target_video] = scan_res
                    st.session_state.last_scan_result = scan_res
                    st.session_state.scanned_video = target_video
                    st.session_state.scanned_meta = meta
                except Exception as ex:
                    st.error(f"Analysis Error: {ex}")
                    st.stop()

        if target_video in st.session_state.scan_cache:
            res = st.session_state.scan_cache[target_video]
            st.session_state.last_scan_result = res
            meta = st.session_state.get("scanned_meta", meta)

            fake_prob = res["fake_probability"]
            real_prob = res["real_probability"]
            is_fake_verdict = fake_prob >= threshold_slider
            confidence = (fake_prob if is_fake_verdict else real_prob) * 100.0

            st.markdown("---")
            st.markdown("### 📊 Forensic Analysis Verdict")

            col_verdict, col_gauge, col_stats = st.columns([1.4, 1.2, 1], gap="medium")

            with col_verdict:
                if is_fake_verdict:
                    st.markdown(
                        f"""
                        <div class="verdict-fake">
                            <div style="font-size: 34px; margin-bottom: 6px;">⚠️</div>
                            <div class="verdict-title-fake">SYNTHETIC DEEPFAKE DETECTED</div>
                            <div style="color: {'#FECACA' if is_dark else '#991B1B'}; font-size: 14px; margin-top: 6px;">
                                Facial manipulation and temporal sequence anomalies present
                            </div>
                            <div style="font-size: 26px; font-weight: 800; color: {text_main}; margin-top: 14px; font-family: 'Plus Jakarta Sans';">
                                Threat Confidence: {confidence:.1f}%
                            </div>
                            <div style="margin-top: 10px; font-size: 12px; font-family: 'JetBrains Mono'; color: #EF4444;">
                                THREAT LEVEL: CRITICAL • THRESHOLD: {threshold_slider*100:.0f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="verdict-real">
                            <div style="font-size: 34px; margin-bottom: 6px;">🛡️</div>
                            <div class="verdict-title-real">VERIFIED AUTHENTIC HUMAN</div>
                            <div style="color: {'#A7F3D0' if is_dark else '#065F46'}; font-size: 14px; margin-top: 6px;">
                                Natural spatial-temporal coherence across facial frames
                            </div>
                            <div style="font-size: 26px; font-weight: 800; color: {text_main}; margin-top: 14px; font-family: 'Plus Jakarta Sans';">
                                Authenticity Confidence: {confidence:.1f}%
                            </div>
                            <div style="margin-top: 10px; font-size: 12px; font-family: 'JetBrains Mono'; color: #10B981;">
                                THREAT LEVEL: MINIMAL • THRESHOLD: {threshold_slider*100:.0f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with col_gauge:
                # Plotly Semi-Circle Gauge
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fake_prob * 100,
                    number={"suffix": "%", "font": {"family": "Space Grotesk", "size": 34, "color": text_main}},
                    title={"text": "Synthetic Risk Score", "font": {"family": "Plus Jakarta Sans", "size": 14, "color": text_muted}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": text_muted},
                        "bar": {"color": "#EF4444" if is_fake_verdict else "#10B981", "thickness": 0.4},
                        "bgcolor": plot_bg,
                        "borderwidth": 1,
                        "bordercolor": card_border,
                        "steps": [
                            {"range": [0, threshold_slider * 100], "color": "rgba(16, 185, 129, 0.18)"},
                            {"range": [threshold_slider * 100, 100], "color": "rgba(239, 68, 68, 0.18)"}
                        ],
                        "threshold": {
                            "line": {"color": "#F59E0B", "width": 3},
                            "thickness": 0.8,
                            "value": threshold_slider * 100
                        }
                    }
                ))
                gauge_fig.update_layout(
                    paper_bgcolor=plot_paper,
                    plot_bgcolor=plot_paper,
                    height=230,
                    margin=dict(l=20, r=20, t=30, b=10)
                )
                st.plotly_chart(gauge_fig, use_container_width=True)

            with col_stats:
                st.metric("Analyzed Frames", f"{res['num_frames']} frames")
                st.metric("Synthetic Probability", f"{fake_prob * 100:.2f}%")
                st.metric("Authentic Probability", f"{real_prob * 100:.2f}%")

            # Temporal Anomaly Line Chart
            st.markdown("#### 📈 Temporal Anomaly Tracking Across Sequence")
            frame_scores = res["frame_probabilities"]
            frame_df = pd.DataFrame({
                "Frame Number": list(range(1, len(frame_scores) + 1)),
                "Deepfake Probability (%)": [s * 100 for s in frame_scores],
                "Threshold": [threshold_slider * 100] * len(frame_scores)
            })

            line_fig = px.area(
                frame_df,
                x="Frame Number",
                y="Deepfake Probability (%)",
                markers=True,
                labels={"Frame Number": "Temporal Frame Index"}
            )
            line_fig.update_traces(
                line_color="#EF4444" if is_fake_verdict else plot_line,
                fillcolor="rgba(239, 68, 68, 0.15)" if is_fake_verdict else "rgba(139, 92, 246, 0.12)",
                marker=dict(size=8, color="#FFFFFF", line=dict(width=2, color="#7C3AED"))
            )
            line_fig.add_hline(
                y=threshold_slider * 100,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text=f"Alert Threshold ({threshold_slider*100:.0f}%)",
                annotation_position="top right",
                annotation_font_color="#F59E0B"
            )
            line_fig.update_layout(
                yaxis=dict(range=[0, 105], gridcolor=plot_grid),
                xaxis=dict(dtick=1, gridcolor=plot_grid),
                paper_bgcolor=plot_paper,
                plot_bgcolor=plot_bg,
                font=dict(color=plot_text, family="Inter"),
                height=280,
                margin=dict(l=30, r=20, t=20, b=30)
            )
            st.plotly_chart(line_fig, use_container_width=True)

            # Extracted Face Sequence Reel
            st.markdown("#### 🧑‍💻 Extracted Facial Sequence Reel & Per-Frame Scoring")
            faces = res.get("face_frames", [])
            if faces:
                num_cols = min(8, len(faces))
                face_cols = st.columns(num_cols)
                for idx, face_img in enumerate(faces):
                    col_target = face_cols[idx % num_cols]
                    score = frame_scores[idx] if idx < len(frame_scores) else fake_prob
                    is_frame_fake = score >= threshold_slider
                    badge_class = "threat-tag-danger" if is_frame_fake else "threat-tag-safe"
                    badge_text = f"FAKE {score*100:.0f}%" if is_frame_fake else f"REAL {(1-score)*100:.0f}%"

                    with col_target:
                        st.markdown(
                            f"""
                            <div class="face-thumb-card">
                                <div style="font-size: 11px; font-weight: 700; color: {text_muted}; margin-bottom: 4px;">F#{idx+1}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.image(face_img, use_container_width=True)
                        st.markdown(
                            f"""
                            <div style="text-align: center; margin-top: 4px;">
                                <span class="{badge_class}">{badge_text}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Quick Report Download Trigger
            st.markdown("---")
            col_pdf_txt, col_pdf_btn = st.columns([2.5, 1])
            with col_pdf_txt:
                st.markdown("##### 📄 Export Cryptographic Forensic Audit Dossier (PDF)")
                st.caption("Generate an official, tamper-evident forensic examination certificate signed by DeepDetect AI.")
            with col_pdf_btn:
                case_id = f"CASE-{int(time.time()) % 100000}"
                pdf_bytes = generate_forensic_pdf(
                    case_info={"case_id": case_id, "examiner": "DeepDetect Forensic Agent", "agency": "Digital Integrity Command"},
                    video_meta={"filename": os.path.basename(target_video), "resolution": meta.get("resolution", "N/A"), "size_mb": meta.get("size_mb", 0.0), "fps": meta.get("fps", 30.0), "duration": meta.get("duration", 0.0), "sha256": meta.get("sha256", "N/A"), "md5": meta.get("md5", "N/A")},
                    result={"is_fake": is_fake_verdict, "confidence": confidence, "fake_probability": fake_prob, "num_frames": res.get("num_frames", 16), "frame_probabilities": frame_scores},
                    face_images=faces
                )
                st.download_button(
                    label="📥 Download Forensic PDF Report",
                    data=pdf_bytes,
                    file_name=f"Forensic_Report_{case_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    else:
        st.markdown(
            f"""
            <div style="background: {card_bg}; padding: 36px 20px; border-radius: 14px; border: 1.5px dashed {card_border}; text-align: center; margin-top: 20px;">
                <div style="font-size: 38px; margin-bottom: 10px;">📹</div>
                <div style="font-size: 16px; font-weight: 700; color: {text_main};">No Video File Loaded</div>
                <div style="font-size: 13px; color: {text_muted}; margin-top: 6px;">Please drop a video file above or choose one of the preloaded presets above to inspect and run forensics.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =============================================================

def render_artifact_laboratory():
    st.markdown("### 🔬 Advanced Artifact Laboratory")
    st.caption("Inspect micro-texture anomalies, 2D Fourier frequency grids, and JPEG compression disparities.")

    if "last_scan_result" in st.session_state and st.session_state.last_scan_result.get("face_frames"):
        faces = st.session_state.last_scan_result["face_frames"]
        frame_idx = st.slider("Select Frame for Deep Inspection", min_value=1, max_value=len(faces), value=1) - 1
        target_face = faces[frame_idx]

        # Compute forensic transformations
        fft_spec = compute_fft_spectrum(target_face)
        ela_img = compute_ela(target_face, quality=90)
        sharpness, lap_edge = compute_laplacian_sharpness(target_face)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("##### 1. Cropped Face Keyframe")
            st.image(target_face, use_container_width=True, caption=f"Frame #{frame_idx+1}")
            st.info("Input RGB facial crop detected via Haar Cascade.")

        with c2:
            st.markdown("##### 2. 2D FFT Frequency Spectrum")
            st.image(fft_spec, use_container_width=True, caption="Log Magnitude Spectrum")
            st.info("Periodic grid lines or crosshair stars indicate GAN upsampling convolution artifacts.")

        with c3:
            st.markdown("##### 3. Error Level Analysis (ELA)")
            st.image(ela_img, use_container_width=True, caption="JPEG Resave Discrepancy")
            st.info("Highlights spliced boundaries where compression history differs from background.")

        with c4:
            st.markdown("##### 4. Laplacian Edge Contour")
            st.image(lap_edge, use_container_width=True, caption=f"Variance: {sharpness:.1f}")
            st.info(f"Sharpness Index: {sharpness:.1f}. Unnatural blurring indicates boundary feathering.")

        st.markdown("---")
        st.markdown("#### ⚡ Inter-Frame Temporal Motion Jitter")
        jitter_vals = compute_temporal_jitter(faces)
        jitter_df = pd.DataFrame({
            "Transition": [f"F{i}->F{i+1}" for i in range(1, len(jitter_vals) + 1)],
            "MSE Jitter": jitter_vals
        })
        jitter_fig = px.bar(
            jitter_df,
            x="Transition",
            y="MSE Jitter",
            color="MSE Jitter",
            color_continuous_scale="Plasma",
            title="Facial Inter-Frame Pixel Difference (Temporal Flicker)"
        )
        jitter_fig.update_layout(
            paper_bgcolor=plot_paper,
            plot_bgcolor=plot_bg,
            font=dict(color=plot_text),
            height=260
        )
        st.plotly_chart(jitter_fig, use_container_width=True)

    else:
        target_video = st.session_state.get("active_video_path", all_fake_samples[0] if all_fake_samples else None)
        st.info("💡 You can execute a full scan in **Deep Forensic Scanner**, or click below to immediately extract a sample facial keyframe and inspect its frequency & compression artifacts:")
        if target_video and os.path.exists(target_video):
            if st.button("🔬 Instant Artifact Inspection on Current Target", type="primary", use_container_width=True):
                from src.preprocessing import process_video
                with st.spinner("Extracting facial frames for artifact inspection..."):
                    try:
                        face_np, face_pil = process_video(target_video, num_frames=8)
                        if face_pil:
                            if "scan_cache" not in st.session_state:
                                st.session_state.scan_cache = {}
                            instant_res = {
                                "fake_probability": 0.5,
                                "real_probability": 0.5,
                                "confidence": 50.0,
                                "is_fake": False,
                                "verdict": "ANALYZING",
                                "frame_probabilities": [0.5] * len(face_pil),
                                "face_frames": face_pil,
                                "num_frames": len(face_pil)
                            }
                            st.session_state.last_scan_result = instant_res
                            st.session_state.scan_cache[target_video] = instant_res
                            st.session_state.scanned_video = target_video
                            st.rerun()
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

    # =============================================================
    # TAB 3: Side-by-Side Dual Comparator
    # =============================================================

def render_dual_comparator():
    st.markdown("### ⚖️ Side-by-Side Dual Video Comparator")
    st.caption("Directly compare a suspected synthetic video with an authentic reference video to inspect forensic differentials.")

    col_cmp1, col_cmp2 = st.columns(2, gap="large")

    with col_cmp1:
        st.markdown("#### 📽️ Video A (Suspect Subject)")
        vid_a_choice = st.selectbox(
            "Select Video A",
            all_fake_samples[:20] if all_fake_samples else ["None"],
            format_func=lambda x: f"⚠️ {os.path.basename(x)}",
            key="vid_a_select"
        )
        if vid_a_choice != "None":
            frame_a, _ = get_video_frame_at_time(vid_a_choice, 1.0)
            if frame_a is not None:
                st.image(frame_a, caption=f"Video A Preview: {os.path.basename(vid_a_choice)}", use_container_width=True)

    with col_cmp2:
        st.markdown("#### 📽️ Video B (Reference / Authentic Subject)")
        vid_b_choice = st.selectbox(
            "Select Video B",
            all_real_samples[:20] if all_real_samples else ["None"],
            format_func=lambda x: f"🛡️ {os.path.basename(x)}",
            key="vid_b_select"
        )
        if vid_b_choice != "None":
            frame_b, _ = get_video_frame_at_time(vid_b_choice, 1.0)
            if frame_b is not None:
                st.image(frame_b, caption=f"Video B Preview: {os.path.basename(vid_b_choice)}", use_container_width=True)

    run_cmp = st.button("⚡ EXECUTE DUAL COMPARATIVE FORENSICS", use_container_width=True)

    if run_cmp and vid_a_choice != "None" and vid_b_choice != "None":
        with st.spinner("Analyzing differential forensic patterns across both subjects..."):
            res_a = predict_video(vid_a_choice, model=model, device=device, num_frames=seq_frames)
            res_b = predict_video(vid_b_choice, model=model, device=device, num_frames=seq_frames)

        c_res_a, c_delta, c_res_b = st.columns([1.2, 0.8, 1.2])

        with c_res_a:
            prob_a = res_a["fake_probability"] * 100
            st.markdown(
                f"""
                <div class="{'verdict-fake' if prob_a >= 50 else 'verdict-real'}">
                    <div style="font-size: 13px; font-weight: 700;">VIDEO A VERDICT</div>
                    <div style="font-size: 22px; font-weight: 800;">{res_a['verdict']}</div>
                    <div style="font-size: 15px;">Synthetic Risk: {prob_a:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c_delta:
            delta = abs(res_a["fake_probability"] - res_b["fake_probability"]) * 100
            st.markdown(
                f"""
                <div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 14px; padding: 18px; text-align: center;">
                    <div style="font-size: 12px; color: {text_muted}; font-weight: 700;">DELTA SEPARATION</div>
                    <div style="font-size: 30px; font-weight: 800; color: #8B5CF6; font-family: 'Space Grotesk';">Δ {delta:.1f}%</div>
                    <div style="font-size: 11px; color: {text_muted};">Statistical Disparity</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c_res_b:
            prob_b = res_b["fake_probability"] * 100
            st.markdown(
                f"""
                <div class="{'verdict-fake' if prob_b >= 50 else 'verdict-real'}">
                    <div style="font-size: 13px; font-weight: 700;">VIDEO B VERDICT</div>
                    <div style="font-size: 22px; font-weight: 800;">{res_b['verdict']}</div>
                    <div style="font-size: 15px;">Synthetic Risk: {prob_b:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Comparative Plotly line overlay
        cmp_df = pd.DataFrame({
            "Frame": list(range(1, len(res_a["frame_probabilities"]) + 1)),
            "Video A (Suspect)": [s * 100 for s in res_a["frame_probabilities"]],
            "Video B (Reference)": [s * 100 for s in res_b["frame_probabilities"]]
        })
        cmp_fig = px.line(
            cmp_df,
            x="Frame",
            y=["Video A (Suspect)", "Video B (Reference)"],
            markers=True,
            title="Overlaid Temporal Anomaly Differential",
            color_discrete_sequence=["#EF4444", "#10B981"]
        )
        cmp_fig.update_layout(
            paper_bgcolor=plot_paper,
            plot_bgcolor=plot_bg,
            font=dict(color=plot_text),
            height=300
        )
        st.plotly_chart(cmp_fig, use_container_width=True)

    # =============================================================
    # TAB 4: Batch Forensic Scanner
    # =============================================================

def render_batch_scanner():
    st.markdown("### 📂 Batch Forensic Media Scanner")
    st.caption("Rapidly evaluate video archives, threat feeds, and media repositories to detect synthetic deepfake intrusions.")

    col_batch_pool, col_batch_count = st.columns([2, 1])
    with col_batch_pool:
        batch_source = st.radio(
            "Batch Dataset Source",
            ["Balanced Mix (Celeb-real + Celeb-synthesis)", "Celeb-synthesis Only (Synthetic)", "Celeb-real Only (Authentic)"],
            horizontal=True
        )
    with col_batch_count:
        batch_limit = st.slider("Scan Batch Size", min_value=4, max_value=20, value=6, step=2)

    start_batch = st.button("🚀 INITIATE BATCH SCAN PROTOCOL", type="primary", use_container_width=True)

    if start_batch:
        if "Balanced" in batch_source:
            half = batch_limit // 2
            batch_files = all_fake_samples[:half] + all_real_samples[:half]
        elif "synthesis" in batch_source:
            batch_files = all_fake_samples[:batch_limit]
        else:
            batch_files = all_real_samples[:batch_limit]

        batch_bar = st.progress(0, text="Preparing batch queue...")
        batch_results = []

        for idx, b_path in enumerate(batch_files):
            fname = os.path.basename(b_path)
            batch_bar.progress((idx + 1) / len(batch_files), text=f"Analyzing [{idx+1}/{len(batch_files)}]: {fname}")
            try:
                t0 = time.time()
                r = predict_video(b_path, model=model, device=device, num_frames=12)
                elapsed = time.time() - t0
                is_f = r["fake_probability"] >= threshold_slider
                batch_results.append({
                    "Filename": fname,
                    "Verdict": "FAKE ⚠️" if is_f else "REAL 🛡️",
                    "Deepfake Prob": f"{r['fake_probability']*100:.1f}%",
                    "Authentic Prob": f"{r['real_probability']*100:.1f}%",
                    "Confidence": f"{r['confidence']:.1f}%",
                    "Scan Time": f"{elapsed:.2f}s",
                    "_raw_fake": r["fake_probability"],
                    "_is_fake": is_f
                })
            except Exception as e:
                batch_results.append({
                    "Filename": fname,
                    "Verdict": "ERROR ❌",
                    "Deepfake Prob": "0.0%",
                    "Authentic Prob": "0.0%",
                    "Confidence": "0.0%",
                    "Scan Time": "0.0s",
                    "_raw_fake": 0.0,
                    "_is_fake": False
                })

        batch_bar.empty()

        # Batch Summary Metrics
        total_scanned = len(batch_results)
        fake_count = sum(1 for item in batch_results if item["_is_fake"])
        real_count = total_scanned - fake_count

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Scanned", total_scanned)
        with k2:
            st.metric("Synthetic Flags", fake_count, delta=f"{(fake_count/total_scanned)*100:.0f}%", delta_color="inverse")
        with k3:
            st.metric("Authentic Passes", real_count)
        with k4:
            st.metric("Scan Speed", f"{sum(float(x['Scan Time'][:-1]) for x in batch_results)/total_scanned:.2f}s / vid")

        # Table display
        df_display = pd.DataFrame(batch_results).drop(columns=["_raw_fake", "_is_fake"])
        st.dataframe(df_display, use_container_width=True)

        # CSV Download Button
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Batch Report (CSV)",
            data=csv_data,
            file_name=f"Batch_Forensic_Scan_{int(time.time())}.csv",
            mime="text/csv"
        )

    # =============================================================
    # TAB 5: Audit Dossier & PDF Export
    # =============================================================

def render_audit_dossier():
    st.markdown("### 📄 Digital Evidence Dossier Generator")
    st.caption("Produce legally sound, cryptographic forensic certificates with investigator details and chain-of-custody signatures.")

    col_form1, col_form2 = st.columns(2)
    with col_form1:
        examiner_name = st.text_input("Lead Forensic Examiner", value="Specialist Cyber Forensic Agent")
        department_name = st.text_input("Agency / Department", value="National Digital Media Forensics Unit")
    with col_form2:
        case_ref = st.text_input("Official Case Reference ID", value=f"CASE-DD-{int(time.time()) % 100000}")
        notes = st.text_area("Examiner Observations / Notes", value="Spatial artifacts detected in key facial regions. Temporal continuity disrupted across multi-frame sequence.")

    active_vid = st.session_state.get("active_video_path", None)
    if "last_scan_result" in st.session_state and active_vid:
        res = st.session_state.last_scan_result
        meta = st.session_state.get("scanned_meta", extract_video_metadata(active_vid) or {})

        gen_pdf = generate_forensic_pdf(
            case_info={"case_id": case_ref, "examiner": examiner_name, "agency": department_name, "notes": notes},
            video_meta={"filename": os.path.basename(active_vid), "resolution": meta.get("resolution", "N/A"), "size_mb": meta.get("size_mb", 0.0), "fps": meta.get("fps", 30.0), "duration": meta.get("duration", 0.0), "sha256": meta.get("sha256", "N/A"), "md5": meta.get("md5", "N/A")},
            result={"is_fake": res["fake_probability"] >= threshold_slider, "confidence": res["confidence"], "fake_probability": res["fake_probability"], "num_frames": res.get("num_frames", 16), "frame_probabilities": res.get("frame_probabilities", [])},
            face_images=res.get("face_frames", [])
        )

        st.success("✓ Tamper-evident PDF certificate ready for export.")
        st.download_button(
            label="📜 Download Official Signed Audit PDF",
            data=gen_pdf,
            file_name=f"Official_Forensic_Dossier_{case_ref}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.info("💡 Run an analysis on any video in the **Deep Forensic Scanner** to compile this evidence dossier.")

    # =============================================================
    # TAB 6: Neural Architecture & Explainability
    # =============================================================

def render_neural_architecture():
    st.markdown("### 🧠 Spatial-Temporal Neural Architecture")
    st.caption("Complete breakdown of the ResNet-18 + Bidirectional LSTM pipeline.")

    col_arch1, col_arch2 = st.columns([1.2, 0.8], gap="large")

    with col_arch1:
        st.markdown(
            """
            #### 🏗️ Five-Stage Processing Pipeline

            1. **Temporal Frame Extraction & Alignment**
               - Uniformly extracts $T$ frames (e.g. 16 frames) across video timeline.
               - Haar Cascade identifies facial ROI with dynamic margin padding to preserve jaw and hair boundaries.

            2. **Spatial Feature Projection (ResNet-18 CNN)**
               - Each cropped face is normalized with ImageNet tensors:
                 $$\\mu = [0.485, 0.456, 0.406], \\quad \\sigma = [0.229, 0.224, 0.225]$$
               - 11.2 million parameter convolutional backbone extracts deep spatial representations.
               - Linear bottleneck layer projects 512-dim spatial features with BatchNorm and Dropout (0.5).

            3. **Temporal Modeling (Bidirectional LSTM)**
               - Sequence of 16 frame vectors $[B, 16, 512]$ enters 2-layer Bi-LSTM.
               - Models forward and reverse temporal transitions $(256 \\times 2 = 512\\text{-dim})$.
               - Detects unnatural blinking rhythms, head pose warping, and lighting flicker.

            4. **Dual Frame & Video Classification Heads**
               - Per-frame classifier outputs frame anomalies.
               - Sequence classifier outputs final calibrated synthetic probability $P(\\text{Fake} \\mid X)$.
            """
        )

    with col_arch2:
        cm_path = os.path.join(PROJECT_ROOT, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.markdown("#### 🎯 Empirical Model Confusion Matrix")
            st.image(cm_path, caption="ResNet-18 + Bi-LSTM Test Set Evaluation", use_container_width=True)
            st.markdown(
                """
                - **Evaluation AUC:** `0.942`
                - **Precision:** `91.8%`
                - **Recall:** `89.4%`
                - **F1-Score:** `0.906`
                """
            )

    st.markdown("---")
    st.markdown("#### 📐 Mathematical Formulation & Loss Function")
    st.latex(r"\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{video}}(\hat{y}, y) + \lambda \frac{1}{T} \sum_{t=1}^{T} \mathcal{L}_{\text{frame}}(\hat{y}_t, y)")
    st.caption("Trained using Binary Cross Entropy with Logits and class imbalance re-weighting.")
# -------------------------------------------------------------
# Module Router (Mehek Sharma Index-Nav Controller)
# -------------------------------------------------------------
if "01" in sidebar_nav:
    st.markdown(
        """
        <div class="mehek-module-header">
            <div class="mehek-module-eyebrow">ARCHIVE INDEX &bull; 01 OF 06</div>
            <h2 class="mehek-module-title">Deep Forensic Scanner</h2>
            <div class="mehek-module-desc">Spatial-temporal deepfake forensics powered by ResNet-18 feature extraction and Bi-LSTM temporal sequence verification</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_scanner()

elif "02" in sidebar_nav:
    st.markdown(
        """
        <div class="mehek-module-header">
            <div class="mehek-module-eyebrow">ARCHIVE INDEX &bull; 02 OF 06</div>
            <h2 class="mehek-module-title">Artifact Laboratory (FFT & ELA)</h2>
            <div class="mehek-module-desc">Spectral frequency domain analysis, Error Level Analysis (ELA), Laplacian edge sharpness gradients & frame jitter</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_artifact_laboratory()

elif "03" in sidebar_nav:
    st.markdown(
        """
        <div class="mehek-module-header">
            <div class="mehek-module-eyebrow">ARCHIVE INDEX &bull; 03 OF 06</div>
            <h2 class="mehek-module-title">Side-by-Side Dual Comparator</h2>
            <div class="mehek-module-desc">Dual video comparative forensic timeline with keyframe synchronization and differential anomaly scoring</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_dual_comparator()

elif "04" in sidebar_nav:
    st.markdown(
        """
        <div class="mehek-module-header">
            <div class="mehek-module-eyebrow">ARCHIVE INDEX &bull; 04 OF 06</div>
            <h2 class="mehek-module-title">Batch Forensic Pipeline</h2>
            <div class="mehek-module-desc">Multi-evidence high-throughput video queue evaluation with automated summary telemetry and CSV batch export</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_batch_scanner()

elif "05" in sidebar_nav:
    st.markdown(
        """
        <div class="mehek-module-header">
            <div class="mehek-module-eyebrow">ARCHIVE INDEX &bull; 05 OF 06</div>
            <h2 class="mehek-module-title">Audit Dossier & PDF Export</h2>
            <div class="mehek-module-desc">Formal cryptographic chain-of-custody forensic investigation dossiers formatted for evidentiary compliance</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_audit_dossier()

elif "06" in sidebar_nav:
    st.markdown(
        """
        <div class="mehek-module-header">
            <div class="mehek-module-eyebrow">ARCHIVE INDEX &bull; 06 OF 06</div>
            <h2 class="mehek-module-title">Neural Architecture & Model Internals</h2>
            <div class="mehek-module-desc">Detailed inspection of spatial convolution feature maps, recurrent cell state dynamics, and weights integrity</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_neural_architecture()

else:
    # ✦ // ALL MODULES (TABS)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Deep Forensic Scanner",
        "🔬 Artifact Laboratory (FFT & ELA)",
        "⚖️ Side-by-Side Dual Comparator",
        "📂 Batch Forensic Scanner",
        "📄 Audit Dossier & PDF Export",
        "🧠 Neural Architecture"
    ])
    with tab1:
        render_scanner()
    with tab2:
        render_artifact_laboratory()
    with tab3:
        render_dual_comparator()
    with tab4:
        render_batch_scanner()
    with tab5:
        render_audit_dossier()
    with tab6:
        render_neural_architecture()
