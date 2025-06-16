import streamlit as st
import tempfile
from pydub import AudioSegment, effects
import numpy as np
import librosa
import shap
import joblib
import os
from tensorflow.keras.models import load_model
import pandas as pd

X_train_cnn = joblib.load(r"C:\Users\ashritha reddy\Downloads\X_train_cnn.pkl")
X_train_log = joblib.load(r"C:\Users\ashritha reddy\Downloads\X_train_log.pkl")
model = load_model(r"C:\Users\ashritha reddy\Documents\DeepShield\Models\cnn_audio_model.keras")
log_model = joblib.load(r"C:\Users\ashritha reddy\Documents\DeepShield\Models\log_audio_model.pkl")
log_scaler = joblib.load(r"C:\Users\ashritha reddy\Documents\DeepShield\Models\log_audio_scaler.pkl")
cnn_scaler = joblib.load(r"C:\Users\ashritha reddy\Documents\DeepShield\Models\cnn_audio_scaler.pkl")


explainer_lr = shap.LinearExplainer(log_model, X_train_log)
explainer_cnn = shap.DeepExplainer(model, X_train_cnn)
def process_audio_inplace(path, target_sample_rate=16000, fixed_duration_ms=10000):
    ext = path.lower().split('.')[-1]
    audio = AudioSegment.from_file(path, format=ext)
    trimmed = audio.strip_silence(silence_len=300, silence_thresh=-40)
    normalized = effects.normalize(trimmed)
    if normalized.frame_rate != target_sample_rate:
        normalized = normalized.set_frame_rate(target_sample_rate)
    if len(normalized) < fixed_duration_ms:
        silence = AudioSegment.silent(duration=fixed_duration_ms - len(normalized))
        final_audio = normalized + silence
    else:
        final_audio = normalized[:fixed_duration_ms]
    final_audio.export(path, format=ext)

def log_predict(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    X = np.mean(mfcc.T, axis=0)
    X_scaled = log_scaler.transform(X.reshape(1, -1))
    
    pred = log_model.predict(X_scaled)[0]
    shap_values = explainer_lr.shap_values(X_scaled)
    
    return {
        "prediction": int(pred),
        "shap_values": shap_values,
        "feature_vector": X_scaled
    }

def cnn_predict(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)

    # Stack and reshape (same as training)
    X = np.vstack([mfcc, chroma, contrast, tonnetz, zcr, rmse])
    X = X.reshape(-1, 67)  # Shape: (time_steps, 67)
    X = cnn_scaler.transform(X)  # Scale
    X_input = X.reshape((-1, 67, 313, 1))  # Final shape: (1, 67, 313, 1)

    # Predict
    pred_prob = model.predict(X_input)[0][0]
    pred_class = int(pred_prob > 0.5)

    # Generate SHAP values
    shap_values_cnn = explainer_cnn.shap_values(X_input)

    return {
        "prediction": pred_class,
        "shap_values": shap_values_cnn.reshape(67,313).sum(axis=1).reshape(1,67),
        "feature_vector": X_input.reshape(67,313).sum(axis=1).reshape(1,67)
    }
def split_audio(path, chunk_ms=10000):
    ext = path.lower().split('.')[-1]
    audio = AudioSegment.from_file(path, format=ext)
    chunks = [audio[i:i + chunk_ms] for i in range(0, len(audio), chunk_ms)]
    return chunks, ext


def combined_predict(file_path, threshold_ms=10000):
    chunks, ext = split_audio(file_path, chunk_ms=threshold_ms)
    for chunk in chunks:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            chunk.export(tmp.name, format=ext)
            process_audio_inplace(tmp.name, fixed_duration_ms=threshold_ms)
            log_dict = log_predict(tmp.name)
            log_pred = log_dict["prediction"]
            cnn_dict = cnn_predict(tmp.name)
            cnn_pred = cnn_dict["prediction"]
            final_pred = int((log_pred + cnn_pred) >= 1)
            if final_pred == 1:
                if log_pred==1:
                    return 1,log_dict
                else:
                    return 1,cnn_dict
    return 0

# Dictionary mapping MFCC index to meaning
mfcc_meanings = {
    # MFCC_0 to MFCC_39 (you already defined this)
    'MFCC_0': "Overall audio energy (loudness)",
    'MFCC_1': "Low-frequency content (pitch contour)",
    'MFCC_2': "Formant structure (vocal tract resonance)",
    'MFCC_3': "Spectral tilt / brightness",
    'MFCC_4': "Harmonic structure (natural vs synthetic tone)",
    'MFCC_5': "Subtle timbral differences",
    'MFCC_6': "Higher spectral shape information",
    'MFCC_7': "Fine-grained timbral texture",
    'MFCC_8': "More detailed spectral variation",
    'MFCC_9': "High-frequency modulation pattern",
    'MFCC_10': "Local spectral envelope fluctuation",
    'MFCC_11': "Short-term spectral irregularities",
    'MFCC_12': "Finer articulation of timbral transitions",
    'MFCC_13': "Very subtle tonal variations",
    'MFCC_14': "Micro-variations in spectral distribution",
    'MFCC_15': "Fine spectral ripple patterns",
    'MFCC_16': "Higher-order harmonic deviation",
    'MFCC_17': "Rapid spectral slope fluctuations",
    'MFCC_18': "Temporal fine structure encoding",
    'MFCC_19': "Very high-frequency spectral modulation",
    'MFCC_20': "Minute amplitude modulation patterns",
    'MFCC_21': "Ultra-fine spectral texture components",
    'MFCC_22': "Subtle transient spectral deviations",
    'MFCC_23': "Higher resolution spectral envelope shape",
    'MFCC_24': "Very high-frequency ripple modulation",
    'MFCC_25': "Micro-level harmonic distortion cues",
    'MFCC_26': "Extremely fine spectral envelope dynamics",
    'MFCC_27': "High-resolution spectral peak alignment",
    'MFCC_28': "Subtle temporal-spectral interactions",
    'MFCC_29': "Very high-frequency formant perturbations",
    'MFCC_30': "Microscopic spectral envelope instability",
    'MFCC_31': "Ultra-fine grain timbral fluctuations",
    'MFCC_32': "Minute spectral edge behavior",
    'MFCC_33': "Extremely subtle spectral asymmetry",
    'MFCC_34': "Highly localized harmonic distortion",
    'MFCC_35': "Narrowband spectral interference patterns",
    'MFCC_36': "Subtle inter-harmonic spacing variation",
    'MFCC_37': "Fine spectral envelope instability",
    'MFCC_38': "High-frequency spectral envelope jitter",
    'MFCC_39': "Most detailed spectral modulation pattern"
}

# Add Chroma STFT
for i in range(12):
    mfcc_meanings[f'Chroma_{i}'] = f"Pitch class {i} energy – relates to vocal prosody and tone quality"

# Add Spectral Contrast
contrast_desc = [
    "Low-frequency spectral contrast",
    "Mid-low frequency band contrast",
    "Mid-frequency contrast",
    "Mid-high frequency contrast",
    "High-frequency contrast",
    "Very high-frequency contrast",
    "Overall spectral imbalance indicator"
]
for i, desc in enumerate(contrast_desc):
    mfcc_meanings[f'Contrast_{i}'] = desc

# Add Tonnetz
tonnetz_desc = [
    "Tonal centroid: First axis (C-A#)",
    "Tonal centroid: Second axis (D-B)",
    "Tonal centroid: Third axis (D#-C)",
    "Tonal tension: Harmonic instability",
    "Tonal flux: Change over time",
    "Key clarity: Degree of tonal focus"
]
for i, desc in enumerate(tonnetz_desc):
    mfcc_meanings[f'Tonnetz_{i}'] = desc

# Add ZCR and RMSE
mfcc_meanings['ZCR'] = "Zero-crossing rate – indicates unvoiced or noisy segments"
mfcc_meanings['RMSE'] = "Root-Mean-Square Energy – reflects short-term loudness patterns"

# Helper to render SHAP force plot in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# Feature names for Logistic Regression model
feature_names = [
    # MFCCs (0–39)
    'MFCC_0', 'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4',
    'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9',
    'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13', 'MFCC_14',
    'MFCC_15', 'MFCC_16', 'MFCC_17', 'MFCC_18', 'MFCC_19',
    'MFCC_20', 'MFCC_21', 'MFCC_22', 'MFCC_23', 'MFCC_24',
    'MFCC_25', 'MFCC_26', 'MFCC_27', 'MFCC_28', 'MFCC_29',
    'MFCC_30', 'MFCC_31', 'MFCC_32', 'MFCC_33', 'MFCC_34',
    'MFCC_35', 'MFCC_36', 'MFCC_37', 'MFCC_38', 'MFCC_39',

    # Chroma STFT (40–51)
    'Chroma_0', 'Chroma_1', 'Chroma_2', 'Chroma_3', 'Chroma_4',
    'Chroma_5', 'Chroma_6', 'Chroma_7', 'Chroma_8', 'Chroma_9',
    'Chroma_10', 'Chroma_11',

    # Spectral Contrast (52–58)
    'Contrast_0', 'Contrast_1', 'Contrast_2', 'Contrast_3',
    'Contrast_4', 'Contrast_5', 'Contrast_6',

    # Tonnetz (59–64)
    'Tonnetz_0', 'Tonnetz_1', 'Tonnetz_2', 'Tonnetz_3',
    'Tonnetz_4', 'Tonnetz_5',

    # ZCR and RMSE (65–66)
    'ZCR',
    'RMSE'
]


# Page config
st.set_page_config(page_title="DeepFake Audio Detector", page_icon="🎤", layout="wide")

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #1f2937, #111827);
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 3rem 1rem;
        text-align: center;
    }
    .title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #60a5fa;
        text-shadow: 0 0 8px #3b82f6;
    }
    .upload-label {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        color: #9ca3af;
    }
    .stFileUploader > div {
        border: 2px dashed #2563eb;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #1e40af20;
    }
    .result {
        margin-top: 2rem;
        font-size: 2rem;
        font-weight: 600;
    }
    .fake {
        color: #ef4444;
        text-shadow: 0 0 10px #ef4444aa;
    }
    .real {
        color: #22c55e;
        text-shadow: 0 0 10px #22c55eaa;
    }
    </style>
    <div class="main">
        <h1 class="title">🎤 DeepFake Audio Detection</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("", type=["wav", "mp3"])

if uploaded_file is not None:
    import os
    import tempfile

    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Analyzing audio..."):
        pred,result = combined_predict(temp_path)

    # Show Prediction Result
    if pred == 1:
        st.markdown('<p class="result fake">🔒 The audio is <strong>FAKE</strong></p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result real">🚨 The audio is <strong>REAL</strong></p>', unsafe_allow_html=True)

    # Show SHAP Explanation Section
    st.markdown("<h3 style='color:#60a5fa;'>🔍 Why Was This Audio Classified That Way?</h3>", unsafe_allow_html=True)

    # 1. Force Plot - Interactive SHAP visualization
    st.markdown("### 🧠 Interactive Explanation")
    try:
        shap.initjs()
        force_plot = shap.force_plot(
            explainer_lr.expected_value,
            result['shap_values'],
            result['feature_vector'],
            feature_names=feature_names[:result['shap_values'].shape[1]],
            matplotlib=False,
            out_names=["Real", "Fake"],
            plot_cmap=['#22c55e', '#ef4444']
        )

        st_shap(force_plot, height=300)
    except Exception as e:
        st.warning("Could not load interactive explanation.")

    # 2. Top Features Bar Chart
    st.markdown("### 📊 Top Influential MFCCs")
    try:
        top_n = 5
        top_indices = np.argsort(np.abs(result['shap_values'][0]))[::-1][:top_n]
        top_feats = [feature_names[i] for i in top_indices]
        top_vals = result['shap_values'][0][top_indices]

        df = pd.DataFrame({'Feature': top_feats, 'Impact': top_vals})
        st.bar_chart(df.set_index('Feature'))
    except Exception as e:
        st.warning("Could not generate bar chart.")

    # 3. Plain Text Summary
    st.markdown("### 📝 Plain English Summary")
    try:
        # Generate explanation
        explanation = ""
        for feat, val in zip(top_feats, top_vals):
            meaning = mfcc_meanings.get(feat, "General audio characteristic")
            
            if val > 0:
                explanation += f"- `{feat}`: {meaning} → High value suggests synthetic voice pattern\n"
            else:
                explanation += f"- `{feat}`: {meaning} → Low value suggests natural human speech\n"
        st.markdown(f"""
        The model believes this audio is {'fake' if pred == 1 else 'real'} because:
        {explanation}
        """)
    except Exception as e:
        st.warning("Could not generate text explanation.")
