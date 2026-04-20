"""
AI Digital Twin - Bearing Health Monitor
Multi-model Streamlit dashboard with SHAP explainability (DE_12k only).
"""

import os
import io
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy.io import loadmat

from utils.feature_extraction import (
    preprocess_signal, segment_signal, extract_features, compute_fft,
    FEATURE_COLUMNS, NORMAL_RANGES, WINDOW_SIZE,
)
from utils.model_loader import (
    get_config_name, load_bundle, list_available_models, predict, get_scaled_features,
)
from utils.shap_utils import get_shap_explanation, top_reasons

# ─────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Digital Twin - Bearing Health Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569; border-radius: 12px;
    padding: 16px 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
div[data-testid="stMetric"] label {
    color: #94a3b8 !important; font-size: 0.85rem !important;
    text-transform: uppercase; letter-spacing: 0.05em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important; font-weight: 700 !important;
}
.alert-critical {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border-left: 5px solid #ef4444; border-radius: 8px;
    padding: 16px 20px; color: #fecaca; margin-bottom: 16px;
}
.alert-warning {
    background: linear-gradient(135deg, #78350f, #92400e);
    border-left: 5px solid #f59e0b; border-radius: 8px;
    padding: 16px 20px; color: #fef3c7; margin-bottom: 16px;
}
.alert-healthy {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border-left: 5px solid #10b981; border-radius: 8px;
    padding: 16px 20px; color: #d1fae5; margin-bottom: 16px;
}
.badge-critical { background:#ef4444; color:white; padding:4px 14px; border-radius:20px; font-weight:700; }
.badge-warning  { background:#f59e0b; color:white; padding:4px 14px; border-radius:20px; font-weight:700; }
.badge-healthy  { background:#10b981; color:white; padding:4px 14px; border-radius:20px; font-weight:700; }
.badge-standby  { background:#6b7280; color:white; padding:4px 14px; border-radius:20px; font-weight:700; }
.info-box {
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    padding: 16px; margin-bottom: 16px; color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = "plotly_dark"

# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────

def _parse_label(label):
    if label == "Normal":
        return "Normal", 0.0
    mapping = {"IR": "Inner Race", "OR": "Outer Race", "Ball": "Ball"}
    parts = label.split("_")
    fault_type = mapping.get(parts[0], parts[0])
    fault_size = int(parts[1]) / 1000.0 if len(parts) > 1 else 0.0
    return fault_type, fault_size


def _severity(confidence, label):
    if label == "Normal":
        return "Healthy", "healthy", "No action needed. System operating normally."
    if confidence >= 0.90:
        return "Critical", "critical", "Immediate inspection recommended. Schedule bearing replacement."
    if confidence >= 0.70:
        return "Warning", "warning", "Monitor closely. Schedule maintenance within next service window."
    return "Low Risk", "healthy", "Continue monitoring. Confidence below threshold."


def _load_uploaded_signal(uploaded, sensor_key):
    """Parse .mat or .csv file. sensor_key is 'DE_time' or 'FE_time'."""
    name = uploaded.name.lower()
    if name.endswith(".mat"):
        data = loadmat(io.BytesIO(uploaded.read()))
        key = None
        for k in data.keys():
            if sensor_key in k:
                key = k
                break
        if key is None:
            # Try any *_time key as fallback
            for k in data.keys():
                if "_time" in k and not k.startswith("__"):
                    key = k
                    break
        if key is None:
            return None, f"No '{sensor_key}' signal found in this .mat file. This file may not contain data for the selected sensor type."
        signal = np.asarray(data[key]).squeeze().astype(np.float64)
        return signal, None

    elif name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(uploaded.read()))
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                return df[col].dropna().values.astype(np.float64), None
        return None, "No numeric column found in CSV."

    return None, "Unsupported file format. Upload .mat or .csv."


# ─────────────────────────────────────────────────
# PLOT BUILDERS
# ─────────────────────────────────────────────────

def _signal_fig(signal, fs, title):
    t = np.arange(len(signal)) / fs
    fig = go.Figure(go.Scatter(x=t, y=signal, mode="lines",
                               line=dict(color="#3b82f6", width=1)))
    fig.update_layout(template=PLOTLY_DARK, title=title,
                      xaxis_title="Time (s)", yaxis_title="Amplitude",
                      height=320, margin=dict(l=50, r=20, t=40, b=40),
                      hovermode="x unified")
    return fig


def _fft_fig(freqs, mags, peak_freq):
    fig = go.Figure(go.Scatter(x=freqs, y=mags, mode="lines",
                               line=dict(color="#8b5cf6", width=1)))
    fig.add_vline(x=peak_freq, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"Peak: {peak_freq:.0f} Hz",
                  annotation_font_color="#f59e0b")
    fig.update_layout(template=PLOTLY_DARK, title="Frequency Spectrum (FFT)",
                      xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
                      height=320, margin=dict(l=50, r=20, t=40, b=40))
    return fig


def _proba_fig(proba, class_names, pred_label):
    colors = ["#ef4444" if c == pred_label else "#475569" for c in class_names]
    fig = go.Figure(go.Bar(x=proba, y=class_names, orientation="h",
                           marker_color=colors,
                           text=[f"{p:.1%}" for p in proba], textposition="auto"))
    fig.update_layout(template=PLOTLY_DARK, title="Class Probability Distribution",
                      xaxis_title="Probability", height=400,
                      margin=dict(l=100, r=20, t=40, b=40), xaxis=dict(range=[0, 1]))
    return fig


def _shap_bar_fig(shap_vals, feature_names):
    order = np.argsort(np.abs(shap_vals))
    names = [feature_names[i] for i in order]
    vals = shap_vals[order]
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:+.3f}" for v in vals], textposition="auto"))
    fig.update_layout(template=PLOTLY_DARK,
                      title="SHAP Feature Contributions (this prediction)",
                      xaxis_title="SHAP Value", height=500,
                      margin=dict(l=120, r=20, t=40, b=40))
    return fig


def _shap_waterfall_fig(shap_vals, base_value, feature_names, feature_values):
    order = np.argsort(np.abs(shap_vals))[::-1]
    labels = [f"{feature_names[i]} = {feature_values[i]:.3f}" for i in order] + ["f(x)"]
    vals = [shap_vals[i] for i in order]
    measures = ["relative"] * len(vals) + ["total"]
    y_vals = vals + [0]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=labels, y=y_vals, base=base_value,
        connector=dict(line=dict(color="#64748b", width=1)),
        increasing=dict(marker=dict(color="#ef4444")),
        decreasing=dict(marker=dict(color="#3b82f6")),
        totals=dict(marker=dict(color="#8b5cf6")),
        text=[f"{v:+.3f}" for v in vals] + [f"{base_value + sum(vals):.3f}"],
        textposition="outside",
    ))
    fig.update_layout(template=PLOTLY_DARK,
                      title=f"SHAP Waterfall (base = {base_value:.3f})",
                      yaxis_title="Model Output", height=450,
                      margin=dict(l=50, r=20, t=60, b=120), xaxis_tickangle=-45)
    return fig


# ─────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []


def _add_history(label, confidence, filename, config):
    st.session_state.history.append({
        "file": filename, "prediction": label,
        "confidence": f"{confidence:.1%}", "model": config,
        "time": time.strftime("%H:%M:%S"),
    })
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]


# ─────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Control Panel")

    uploaded_file = st.file_uploader("Upload vibration signal", type=["mat", "csv"])

    sensor_type = st.selectbox("Sensor Location", ["Drive End (DE)", "Fan End (FE)"])

    # Fixed sampling rate — only 12k models are supported in this system
    sampling_rate = "12k"
    st.caption("Sampling Rate: 12 kHz (only 12k data supported)")

    config_name = get_config_name(sensor_type, sampling_rate)
    fs = 12000.0
    sensor_key = "DE_time" if "DE" in config_name else "FE_time"
    is_primary = (config_name == "DE_12k")

    st.markdown("---")

    # Model info
    bundle = load_bundle(config_name)
    if bundle is not None:
        st.markdown(f"**Active Model:** `{config_name}`")
        st.markdown(f"**Classes:** {len(bundle['class_names'])}")
        st.markdown(f"**Accuracy:** {bundle['accuracy']:.1%}")
        st.markdown(f"**Sampling Rate:** {bundle['fs']:.0f} Hz")
        if is_primary:
            st.success("SHAP: Full (Primary Model)")
        else:
            st.info("SHAP: Available (live computation)")
    else:
        available = list_available_models()
        st.error(f"Model `{config_name}` not found.")
        if available:
            st.info(f"Available: {', '.join(available)}")

    st.markdown("---")

    # Limitations panel
    with st.expander("Model Limitations", expanded=False):
        st.markdown("""
**This system is trained only on the CWRU dataset.**

Supported configurations:
- **12k Drive End (DE_12k)** - 12 classes, best accuracy (primary model)
- **12k Fan End (FE_12k)** - 10 classes (Fan End has no 0.028 fault size)

Both models use 12 kHz sampling rate.

SHAP explainability is **available for both models**.
Pre-generated report plots are only stored for DE_12k.
        """)

    with st.expander("How to Use", expanded=False):
        st.markdown("""
1. **Upload** a `.mat` file from the CWRU dataset or a single-column `.csv`
2. **Select** the correct sensor type (Drive End or Fan End)
3. View the diagnosis result, signal plots, SHAP explanation, and probability

**For best results:** Use files from `12k_Drive_End_Bearing_Fault_Data/` (DE_12k has 12 classes).
        """)


# ─────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────

col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown("# AI Digital Twin - Bearing Health Monitor")
    st.caption("Multi-model vibration analysis with XAI-powered fault diagnosis")
with col_status:
    if uploaded_file is None:
        st.markdown('<span class="badge-standby">STANDBY</span>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────
# NO FILE STATE
# ─────────────────────────────────────────────────

if uploaded_file is None:
    st.info("Upload a vibration signal file (.mat or .csv) from the sidebar to begin diagnosis.")

    if st.session_state.history:
        st.markdown("### Recent Predictions")
        st.dataframe(pd.DataFrame(st.session_state.history[::-1]), width="stretch")
    st.stop()

# ─────────────────────────────────────────────────
# MODEL CHECK
# ─────────────────────────────────────────────────

if bundle is None:
    st.error(f"Model `model_{config_name}.pkl` not found in `models/`. Run `python pipeline.py` first.")
    st.stop()

# ─────────────────────────────────────────────────
# LOAD SIGNAL
# ─────────────────────────────────────────────────

signal_raw, err = _load_uploaded_signal(uploaded_file, sensor_key)
if err:
    st.error(f"**Input Error:** {err}")
    st.markdown("""
    <div class="info-box">
    <strong>Troubleshooting:</strong><br>
    - Check that the file matches your selected sensor type<br>
    - Drive End files contain <code>DE_time</code> keys<br>
    - Fan End files contain <code>FE_time</code> keys<br>
    - Not all .mat files have both sensor channels
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if len(signal_raw) < WINDOW_SIZE:
    st.error(f"Signal too short ({len(signal_raw)} samples). Need at least {WINDOW_SIZE}.")
    st.stop()

if not np.isfinite(signal_raw).any():
    st.error("Signal contains no valid (finite) values.")
    st.stop()

# ─────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────

progress = st.progress(0, text="Preprocessing signal...")
filtered = preprocess_signal(signal_raw, fs=fs)
progress.progress(20, text="Segmenting windows...")

windows = segment_signal(filtered)
if len(windows) == 0:
    st.error("No valid windows after segmentation.")
    st.stop()

progress.progress(40, text="Extracting features...")
mid = len(windows) // 2
features = extract_features(windows[mid], fs=fs)
if features is None:
    st.error("Feature extraction failed (NaN/Inf in signal).")
    st.stop()

progress.progress(60, text="Running XGBoost model...")
try:
    pred_label, proba, class_names = predict(features, config_name)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

confidence = float(np.max(proba))
fault_type, fault_size = _parse_label(pred_label)
severity, sev_color, action = _severity(confidence, pred_label)

# SHAP — computed for ALL models
shap_vals = None
base_val = None
feat_names = None
progress.progress(80, text="Computing SHAP explanation...")
try:
    scaled = get_scaled_features(features, config_name)
    shap_vals, base_val, feat_names, _ = get_shap_explanation(scaled, config_name)
except Exception:
    shap_vals = None

progress.progress(100, text="Done!")
time.sleep(0.3)
progress.empty()

_add_history(pred_label, confidence, uploaded_file.name, config_name)

# Update header badge
with col_status:
    st.markdown(f'<span class="badge-{sev_color}">{severity.upper()}</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# DIAGNOSIS RESULT
# ─────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Fault Type", fault_type)
with c2:
    st.metric("Fault Class", pred_label)
with c3:
    st.metric("Confidence", f"{confidence:.1%}")
with c4:
    st.metric("Fault Size", f"{fault_size:.3f} in" if fault_size > 0 else "N/A")
with c5:
    st.metric("Model", config_name)

# ─────────────────────────────────────────────────
# ALERT
# ─────────────────────────────────────────────────

if pred_label != "Normal":
    st.markdown(f"""
    <div class="alert-{sev_color}">
        <strong>{"!!!" if sev_color == "critical" else "!"} {fault_type} Fault Detected &mdash; {pred_label}</strong><br>
        <strong>Severity:</strong> {severity} ({confidence:.1%} confidence)<br>
        <strong>Recommended Action:</strong> {action}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="alert-healthy">
        <strong>System Healthy</strong><br>
        Bearing operating within normal parameters. Confidence: {confidence:.1%}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────

tab_names = ["Signal View", "FFT Spectrum", "Prediction", "SHAP Explanation", "Feature Insights", "History"]
tabs = st.tabs(tab_names)
tab_idx = 0

# ── Signal View ──
with tabs[tab_idx]:
    n_show = min(20000, len(signal_raw))
    st.plotly_chart(_signal_fig(signal_raw[:n_show], fs, "Raw Vibration Signal"), width="stretch")
    st.plotly_chart(_signal_fig(filtered[:n_show], fs, f"Filtered Signal (Bandpass 20-5000 Hz)"), width="stretch")
tab_idx += 1

# ── FFT ──
with tabs[tab_idx]:
    freqs, mags = compute_fft(windows[mid], fs=fs)
    peak_f = float(freqs[np.argmax(mags)])
    st.plotly_chart(_fft_fig(freqs, mags, peak_f), width="stretch")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.metric("Peak Frequency", f"{peak_f:.0f} Hz")
    with fc2:
        st.metric("Freq Centroid", f"{features['FreqCentroid']:.0f} Hz")
    with fc3:
        st.metric("Spectral Entropy", f"{features['SpectralEntropy']:.2f}")
tab_idx += 1

# ── Probability ──
with tabs[tab_idx]:
    st.plotly_chart(_proba_fig(proba, class_names, pred_label), width="stretch")
    st.markdown("**Top 3 Predicted Classes:**")
    top3 = np.argsort(proba)[::-1][:3]
    pc1, pc2, pc3 = st.columns(3)
    for col, idx in zip([pc1, pc2, pc3], top3):
        with col:
            st.metric(class_names[idx], f"{proba[idx]:.1%}")
tab_idx += 1

# ── SHAP (all models) ──
with tabs[tab_idx]:
    if shap_vals is not None:
        if not is_primary:
            st.info(f"SHAP computed live for **{config_name}** model. Pre-generated report plots are only available for DE_12k.")

        reasons = top_reasons(shap_vals, feat_names, n=3)
        st.markdown("**Top 3 Reasons for This Prediction:**")
        for i, r in enumerate(reasons, 1):
            st.markdown(f"{i}. **{r['feature']}** (SHAP = `{r['shap_value']:+.3f}`) -- {r['text']}")
        st.markdown("---")

        col_bar, col_wf = st.columns(2)
        with col_bar:
            st.plotly_chart(_shap_bar_fig(shap_vals, feat_names), width="stretch")
        with col_wf:
            feat_vals = np.array([features[f] for f in feat_names])
            st.plotly_chart(_shap_waterfall_fig(shap_vals, base_val, feat_names, feat_vals),
                            width="stretch")
    else:
        st.warning("SHAP computation failed for this sample.")
tab_idx += 1

# ── Feature Insights ──
with tabs[tab_idx]:
    feat_df = pd.DataFrame([{
        "Feature": f, "Value": features[f],
        "Normal Low": NORMAL_RANGES.get(f, (None, None))[0],
        "Normal High": NORMAL_RANGES.get(f, (None, None))[1],
    } for f in FEATURE_COLUMNS])

    def _flag(row):
        lo, hi = row["Normal Low"], row["Normal High"]
        if lo is None:
            return ""
        return "ABNORMAL" if (row["Value"] < lo or row["Value"] > hi) else "OK"

    feat_df["Status"] = feat_df.apply(_flag, axis=1)

    def _highlight(row):
        if row["Status"] == "ABNORMAL":
            return ["background-color: rgba(239,68,68,0.2)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        feat_df.style.apply(_highlight, axis=1).format({"Value": "{:.4f}"}),
        width="stretch", height=620,
    )

tab_idx += 1

# ── History ──
with tabs[tab_idx]:
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(hist_df, width="stretch")
        if len(st.session_state.history) > 1:
            confs = [float(h["confidence"].strip("%")) / 100 for h in st.session_state.history]
            fig_trend = go.Figure(go.Scatter(y=confs, mode="lines+markers",
                                             line=dict(color="#3b82f6", width=2),
                                             marker=dict(size=8)))
            fig_trend.update_layout(template=PLOTLY_DARK, title="Confidence Trend",
                                    yaxis_title="Confidence", xaxis_title="Prediction #",
                                    height=300)
            st.plotly_chart(fig_trend, width="stretch")
    else:
        st.info("No predictions yet.")
