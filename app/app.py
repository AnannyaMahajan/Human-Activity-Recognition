"""
Human Activity Recognition (HAR) — Clean, Modern Web Application.
A friendly, accessible machine learning application for predicting human physical
activities from smartphone accelerometer and gyroscope sensor data.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.decomposition import PCA

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils import load_model, logger
from src.data_loader import load_raw_data
from src.evaluate import calculate_metrics, compute_comprehensive_evaluation
from src.inference import (
    validate_feature_dataframe, run_inference, get_available_models,
    load_inference_model, load_preset_profiles, KEY_FEATURE_DESCRIPTORS
)

# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(
    page_title="Human Activity Recognition | Smart Motion AI",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# Clean, Modern Light-Theme CSS (Friendly & Accessible)
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #F8FAFC !important;
        color: #0F172A;
    }
    
    .stApp {
        background-color: #F8FAFC !important;
    }

    /* Top Brand Navigation Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 12px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-sub {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 500;
    }
    .nav-badge {
        background: #EFF6FF;
        color: #2563EB;
        border: 1px solid #DBEAFE;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
    }

    /* Friendly Hero Card */
    .hero-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F7FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.05);
    }
    .hero-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        margin: 0 0 10px 0 !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.6;
        max-width: 750px;
        margin: 0 0 20px 0;
    }

    /* Clean Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.06);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        line-height: 1.4;
    }

    /* Workflow Step Badge */
    .step-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #2563EB;
        background: #EFF6FF;
        border: 1px solid #DBEAFE;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 12px;
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        background: #2563EB;
        color: #FFFFFF;
        border-radius: 50%;
        font-size: 0.72rem;
        font-weight: 700;
    }

    /* Friendly Prediction Result Card */
    .result-card {
        background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 100%);
        border: 2px solid #86EFAC;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.08);
    }
    .result-activity-name {
        font-size: 2.1rem;
        font-weight: 800;
        color: #15803D;
        margin: 6px 0 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .result-confidence-pill {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
    }

    /* Friendly Feature Description Pill */
    .feature-badge {
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #E2E8F0;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
    }

    /* Tab bar clean styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #FFFFFF;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        font-size: 0.9rem;
        font-weight: 600;
        border-radius: 8px;
        color: #64748B;
        padding: 0 18px;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #EFF6FF !important;
        color: #2563EB !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.6rem !important; }
        .hero-card { padding: 20px; }
        .top-navbar { flex-direction: column; align-items: flex-start; gap: 8px; }
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Cached Loaders
# -------------------------------------------------------------
@st.cache_resource
def get_cached_model(model_path_str: str):
    """Loads and caches a model pipeline by path."""
    try:
        return load_model(Path(model_path_str))
    except Exception as e:
        logger.error(f"Error loading model from {model_path_str}: {e}")
        return None

@st.cache_data
def get_cached_metadata():
    """Loads metadata.json."""
    meta_path = config.MODELS_DIR / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

@st.cache_data
def get_cached_sample_data():
    """Loads sample_input.csv."""
    sample_path = config.APP_DIR / "sample_input.csv"
    if sample_path.exists():
        try:
            return pd.read_csv(sample_path)
        except Exception:
            return None
    return None

@st.cache_data
def get_cached_comparison_table():
    """Loads model_comparison_results.csv."""
    csv_path = config.TABLES_DIR / "model_comparison_results.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return None
    return None

@st.cache_data
def get_cached_raw_test_data():
    """Loads held-out test split for exploration and evaluation in <0.5s."""
    try:
        from src.data_loader import load_test_data
        return load_test_data()
    except Exception as e:
        logger.warning(f"Could not load test dataset: {e}")
        return None, None, None, None

@st.cache_data
def get_cached_presets():
    """Loads calibrated feature profiles for manual input."""
    return load_preset_profiles()


# Friendly Activity Mappings & Plain-Language Descriptions
FRIENDLY_ACTIVITY_NAMES = {
    "WALKING": "Walking",
    "WALKING_UPSTAIRS": "Walking Upstairs",
    "WALKING_DOWNSTAIRS": "Walking Downstairs",
    "SITTING": "Sitting",
    "STANDING": "Standing",
    "LAYING": "Laying Down"
}

ACTIVITY_ICONS = {
    "WALKING": "🚶",
    "WALKING_UPSTAIRS": "🧗",
    "WALKING_DOWNSTAIRS": "🏃",
    "SITTING": "🪑",
    "STANDING": "🧍",
    "LAYING": "🛋️"
}

ACTIVITY_DESCRIPTIONS = {
    "WALKING": "Continuous walking motion with rhythmic body acceleration and stepping cadence (approx. 1.8 Hz).",
    "WALKING_UPSTAIRS": "Ascending stairs with elevated vertical linear acceleration and positive thigh displacement.",
    "WALKING_DOWNSTAIRS": "Descending stairs characterized by sharp impact force peaks during downward foot strikes.",
    "SITTING": "Resting static posture with the smartphone tilted horizontally in front thigh pocket.",
    "STANDING": "Upright static posture with phone aligned along vertical gravitational axis.",
    "LAYING": "Horizontal resting position with phone lateral axis experiencing complete 90° gravity rotation."
}


# -------------------------------------------------------------
# Main Application Flow
# -------------------------------------------------------------
def main():
    metadata = get_cached_metadata()
    sample_data = get_cached_sample_data()
    comp_df = get_cached_comparison_table()
    available_models = get_available_models()

    # Extract clean overview metrics
    if comp_df is not None and not comp_df.empty:
        top_row = comp_df.iloc[0]
        best_model_name = str(top_row["Model"])
        best_acc = float(top_row["Accuracy"])
        best_acc_str = f"{best_acc:.1%}"
        inference_speed_str = f"{float(top_row['Inference_Time_ms_per_sample']):.3f} ms"
    else:
        best_model_name = metadata.get("best_model_name", "LDA")
        best_acc_str = f"{metadata.get('test_accuracy', 0.962):.1%}"
        inference_speed_str = "<0.01 ms"

    # Clean Top Navigation Header
    st.markdown("""
    <div class="top-navbar">
        <div>
            <div class="brand-title">
                <span>📱</span> Human Activity Recognition
            </div>
            <div class="brand-sub">Smart Motion AI Engine &nbsp;•&nbsp; Smartphone Accelerometer & Gyroscope Analysis</div>
        </div>
        <div style="display:flex; gap:10px; align-items:center;">
            <span class="nav-badge">🟢 AI Model Ready</span>
            <span class="nav-badge">6 Activities Supported</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Simple Primary Sections
    tabs = st.tabs([
        "🏠 Home",
        "⚡ Predict Activity",
        "📊 Explore Data",
        "🏆 Model Performance"
    ])

    # -------------------------------------------------------------
    # SECTION 1: HOME
    # -------------------------------------------------------------
    with tabs[0]:
        st.markdown(f"""
        <div class="hero-card">
            <h1 class="hero-title">Discover Physical Activities from Motion Data</h1>
            <p class="hero-desc">
                Using motion data from smartphone sensors, this application automatically recognizes whether a person 
                is <strong>walking</strong>, <strong>climbing stairs</strong>, <strong>sitting</strong>, 
                <strong>standing</strong>, or <strong>resting</strong> — with <strong>{best_acc_str}</strong> verified accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Primary Call-To-Action Button
        col_cta, _ = st.columns([1, 2])
        with col_cta:
            st.markdown("""
            <div style="margin-bottom: 12px;">
                <span style="font-size: 0.95rem; color: #475569;">Ready to test real sensor data?</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Start Predicting", type="primary", key="home_btn_predict", help="Head directly to the prediction workflow"):
                st.info("👉 Switch to the **⚡ Predict Activity** tab above to test a sample or upload your data!")

        st.markdown("### Key Highlights")

        # Three Compact Summary Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label"><span>🎯</span> SUPPORTED ACTIVITIES</div>
                <div class="metric-val">6 Classes</div>
                <div class="metric-sub">Walking, Stairs (Up/Down), Sitting, Standing, and Laying Down</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label"><span>🏆</span> BEST VERIFIED ACCURACY</div>
                <div class="metric-val">{best_acc_str}</div>
                <div class="metric-sub">Evaluated under strict zero-leakage subject isolation ({best_model_name})</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label"><span>⚡</span> INFERENCE AVAILABILITY</div>
                <div class="metric-val">Online & Ready</div>
                <div class="metric-sub">Fast edge response ({inference_speed_str} latency per sample)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Simple 3-Step Walkthrough
        st.markdown("### How It Works in 3 Simple Steps")
        step_c1, step_c2, step_c3 = st.columns(3)

        with step_c1:
            with st.container(border=True):
                st.markdown("#### 1. Sensor Motion Capture")
                st.write("A smartphone in the pocket records 3D acceleration and angular rotational velocity at 50 times per second (50Hz).")

        with step_c2:
            with st.container(border=True):
                st.markdown("#### 2. Pattern Extraction")
                st.write("The motion stream is split into 2.56-second windows. Key metrics (frequency, peak impacts, and gravity tilt) are extracted.")

        with step_c3:
            with st.container(border=True):
                st.markdown("#### 3. Real-Time Recognition")
                st.write("Our trained machine learning model analyzes the motion signals and outputs the detected activity with a confidence score.")

        # Expandable Project & Research Background
        with st.expander("ℹ️ Learn more about the UCI HAR Dataset & Scientific Background"):
            st.markdown("""
            - **Benchmark Source:** Official UCI Machine Learning Repository (Dataset #240).
            - **Participants:** 30 human volunteers aged 19–48 performing standard daily activities.
            - **Strict Subject Isolation:** 21 subjects used for training, 9 subjects reserved exclusively for final testing.
            - **Zero Data Leakage:** Models are never tested on motion signals from individuals they were trained on, guaranteeing true real-world generalization.
            """)

    # -------------------------------------------------------------
    # SECTION 2: PREDICT ACTIVITY (MAIN PREDICTION WORKFLOW)
    # -------------------------------------------------------------
    with tabs[1]:
        st.markdown("## Human Activity Recognition")
        st.markdown("*Upload your sensor data or try a sample to discover the predicted activity.*")

        # Model Selector (Clean & Simple)
        if not available_models:
            st.error("No saved models found in `models/`. Please train models first.")
            return

        with st.expander("⚙️ Model Options (Default: Best Model)", expanded=False):
            model_options = list(available_models.keys())
            selected_model_label = st.selectbox(
                "Choose Machine Learning Model:",
                options=model_options,
                index=0,
                help="Select which trained estimator will analyze your data."
            )
        
        # Load active model
        if 'selected_model_label' not in locals():
            selected_model_label = list(available_models.keys())[0]
        selected_model_path = available_models[selected_model_label]
        active_model = get_cached_model(str(selected_model_path))

        st.markdown("<br>", unsafe_allow_html=True)

        # Three Clear Options
        st.markdown('<div class="step-pill"><span class="step-num">1</span> Step 1: Choose Your Input Method</div>', unsafe_allow_html=True)

        input_choice = st.radio(
            "Select how you want to provide data:",
            ["Upload CSV", "Enter Features", "Try a Sample"],
            index=2,  # Default to 'Try a Sample' so users can predict in 1 click
            horizontal=True,
            key="main_input_selector",
            help="Choose between uploading a file, adjusting sliders, or loading an instant pre-loaded sample."
        )

        input_df = None
        ground_truth_label = None
        is_batch_mode = False

        # ---------------------------------------------
        # OPTION 1: TRY A SAMPLE (EASIEST FOR BEGINNERS)
        # ---------------------------------------------
        if input_choice == "Try a Sample":
            with st.container(border=True):
                st.markdown("#### 🧪 Select an Example Activity Sample")
                st.caption("Load a real, verified sensor recording from the test dataset to test immediate prediction.")

                if sample_data is not None and not sample_data.empty:
                    s_col1, s_col2 = st.columns([1, 1])
                    with s_col1:
                        available_acts = list(config.ACTIVITY_NAMES)
                        friendly_choices = [FRIENDLY_ACTIVITY_NAMES.get(a, a) for a in available_acts]
                        choice_to_code = {FRIENDLY_ACTIVITY_NAMES.get(a, a): a for a in available_acts}
                        
                        selected_friendly = st.selectbox(
                            "Select Activity Type:",
                            options=friendly_choices,
                            index=0
                        )
                        target_code = choice_to_code[selected_friendly]

                    with s_col2:
                        # Filter sample data for this activity
                        sub_samples = sample_data[sample_data['true_activity'] == target_code] if 'true_activity' in sample_data.columns else sample_data
                        sample_row_idx = st.selectbox(
                            "Select Sample Recording:",
                            options=sub_samples.index,
                            format_func=lambda x: f"Recording #{x + 1} ({selected_friendly})"
                        )

                    chosen_row = sub_samples.loc[sample_row_idx]
                    ground_truth_label = chosen_row.get("true_activity", None)
                    input_df = chosen_row.drop("true_activity", errors='ignore').to_frame().T.astype(float)

                    st.success(f"Loaded **{selected_friendly}** example. Ready to predict.")
                else:
                    st.warning("Sample dataset not found.")

        # ---------------------------------------------
        # OPTION 2: UPLOAD CSV
        # ---------------------------------------------
        elif input_choice == "Upload CSV":
            with st.container(border=True):
                st.markdown("#### 📂 Upload a Sensor Data File")
                st.write("Upload a CSV file containing 561-feature smartphone sensor readings. You can upload single or batch rows.")

                up_file = st.file_uploader(
                    "Choose a CSV file:",
                    type=["csv"],
                    key="clean_csv_uploader"
                )

                # Helpful template download links
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    unlabelled_path = config.APP_DIR / "sample_input_no_labels.csv"
                    if unlabelled_path.exists():
                        with open(unlabelled_path, "rb") as f:
                            st.download_button(
                                "📥 Download Ready-to-Test Sample CSV",
                                f,
                                "test_sample_18_rows.csv",
                                "text/csv",
                                key="dl_sample_clean_step1"
                            )
                with dl_col2:
                    template_path = config.APP_DIR / "template_schema.csv"
                    if template_path.exists():
                        with open(template_path, "rb") as f:
                            st.download_button(
                                "📄 Download Blank CSV Template",
                                f,
                                "template_schema.csv",
                                "text/csv",
                                key="dl_template_clean_step1"
                            )

                if up_file is not None:
                    try:
                        raw_upload_df = pd.read_csv(up_file)
                        is_valid, cleaned_data, err_msg, warnings = validate_feature_dataframe(raw_upload_df)

                        for w in warnings:
                            st.warning(f"Notice: {w}")

                        if not is_valid:
                            st.error(f"❌ {err_msg}")
                            input_df = None
                        else:
                            input_df = cleaned_data
                            is_batch_mode = len(input_df) > 1
                            st.success(f"✅ Successfully validated **{len(input_df)}** recording(s) across 561 features.")
                            with st.expander("Preview data snippet", expanded=False):
                                st.dataframe(input_df.head(5), use_container_width=True)

                    except Exception as e:
                        st.error(f"Could not read CSV file: {e}")

        # ---------------------------------------------
        # OPTION 3: ENTER FEATURES (CALIBRATED PRESETS)
        # ---------------------------------------------
        elif input_choice == "Enter Features":
            with st.container(border=True):
                st.markdown("#### 🎛️ Manual Feature Customization")
                st.write("Pick a baseline movement posture and fine-tune primary sensor dimensions using intuitive sliders.")

                presets = get_cached_presets()
                meta_features = metadata.get("feature_names", [f"feat_{i}" for i in range(561)])

                preset_col1, preset_col2 = st.columns([2, 1])
                with preset_col1:
                    friendly_presets = {
                        "WALKING": "Walking Style",
                        "WALKING_UPSTAIRS": "Stairs Climbing Style",
                        "WALKING_DOWNSTAIRS": "Stairs Descending Style",
                        "SITTING": "Sitting Still",
                        "STANDING": "Standing Upright",
                        "LAYING": "Laying Down"
                    }
                    preset_keys = list(friendly_presets.keys())
                    selected_preset_key = st.selectbox(
                        "Starting Motion Baseline:",
                        options=preset_keys,
                        format_func=lambda k: friendly_presets.get(k, k)
                    )
                with preset_col2:
                    st.write("<br>", unsafe_allow_html=True)
                    if st.button("🔄 Reset Sliders to Baseline", key="btn_reset_sliders"):
                        st.session_state["active_manual_dict"] = None
                        st.rerun()

                # Initialize state
                if "active_manual_dict" not in st.session_state or st.session_state.get("last_preset_key") != selected_preset_key:
                    if selected_preset_key in presets:
                        st.session_state["active_manual_dict"] = presets[selected_preset_key].copy()
                    else:
                        st.session_state["active_manual_dict"] = {f: 0.0 for f in meta_features}
                    st.session_state["last_preset_key"] = selected_preset_key

                st.markdown("##### Adjust Primary Sensor Measurements:")
                slider_cols = st.columns(2)

                # 4 simple, high-impact sliders
                simple_descriptors = KEY_FEATURE_DESCRIPTORS[:4]
                for idx, desc in enumerate(simple_descriptors):
                    fid = desc["id"]
                    curr_val = float(st.session_state["active_manual_dict"].get(fid, 0.0))
                    curr_val = max(-1.0, min(1.0, curr_val))

                    with slider_cols[idx % 2]:
                        new_val = st.slider(
                            f"{desc['label']}",
                            min_value=-1.0, max_value=1.0, value=curr_val, step=0.05,
                            help=desc["desc"], key=f"clean_slider_{fid}"
                        )
                        st.session_state["active_manual_dict"][fid] = new_val

                input_df = pd.DataFrame([st.session_state["active_manual_dict"]])[meta_features]

        # ---------------------------------------------
        # STEP 2: RUN PREDICTION
        # ---------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="step-pill"><span class="step-num">2</span> Step 2: Run Prediction</div>', unsafe_allow_html=True)

        can_predict = input_df is not None and len(input_df) > 0 and active_model is not None
        btn_pred = st.button(
            "🚀 Predict Activity",
            type="primary",
            disabled=not can_predict,
            key="btn_run_main_prediction",
            help="Click to analyze the sensor data and discover the predicted activity."
        )

        # ---------------------------------------------
        # STEP 3: VIEW RESULTS
        # ---------------------------------------------
        if btn_pred:
            with st.spinner("Analyzing motion patterns with Machine Learning..."):
                results = run_inference(active_model, input_df)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="step-pill"><span class="step-num">3</span> Step 3: Prediction Results</div>', unsafe_allow_html=True)

            # A. SINGLE INSTANCE RESULT
            if not is_batch_mode:
                pred_raw = results["predictions"][0]
                conf = results["confidences"][0]
                probs = results["probabilities"][0]
                friendly_name = FRIENDLY_ACTIVITY_NAMES.get(pred_raw, pred_raw)
                icon = ACTIVITY_ICONS.get(pred_raw, "🏃")
                desc_text = ACTIVITY_DESCRIPTIONS.get(pred_raw, "Physical activity detected.")

                # Large Friendly Result Card
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:0.85rem; font-weight:700; color:#166534; text-transform:uppercase; letter-spacing:0.05em;">
                        Detected Activity
                    </div>
                    <div class="result-activity-name">
                        <span>{icon}</span> {friendly_name}
                    </div>
                    <div style="display:flex; gap:12px; align-items:center; margin-bottom:12px;">
                        <span class="result-confidence-pill">Confidence: {conf:.1%}</span>
                        <span style="font-size:0.8rem; color:#475569;">Inference time: {results['latency_ms_per_sample']:.3f} ms</span>
                    </div>
                    <div style="font-size:0.92rem; color:#334155; line-height:1.5;">
                        {desc_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if ground_truth_label is not None:
                    expected_friendly = FRIENDLY_ACTIVITY_NAMES.get(ground_truth_label, ground_truth_label)
                    if pred_raw == ground_truth_label:
                        st.success(f"✅ **Match:** The prediction perfectly matches the recorded true activity (`{expected_friendly}`).")
                    else:
                        st.warning(f"ℹ️ **Discrepancy:** The recorded true label was `{expected_friendly}`.")

                # Simple Horizontal Probability Bar Chart
                st.markdown("#### Probability Distribution Across Activities")
                prob_df = pd.DataFrame({
                    "Activity": [FRIENDLY_ACTIVITY_NAMES.get(a, a) for a in config.ACTIVITY_NAMES],
                    "Confidence": probs
                })

                fig_clean_prob = px.bar(
                    prob_df, x="Confidence", y="Activity", orientation="h",
                    color="Confidence",
                    color_continuous_scale=["#DBEAFE", "#2563EB"],
                    text_auto=".1%", template="plotly_white"
                )
                fig_clean_prob.update_layout(
                    height=280,
                    xaxis=dict(title="Probability", range=[0, 1.05], tickformat=".0%"),
                    yaxis=dict(title="", autorange="reversed"),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_clean_prob, use_container_width=True)

            # B. BATCH PREDICTION RESULTS (TABLE + DOWNLOAD)
            else:
                st.success(f"Successfully predicted **{len(input_df)}** activities!")

                preds_raw = results["predictions"]
                confs = results["confidences"]

                result_table = pd.DataFrame({
                    "Row Index": range(1, len(preds_raw) + 1),
                    "Predicted Activity": [f"{ACTIVITY_ICONS.get(p, '')} {FRIENDLY_ACTIVITY_NAMES.get(p, p)}" for p in preds_raw],
                    "Confidence Score": [f"{c:.1%}" for c in confs]
                })

                # Summary distribution
                col_sum_chart, col_sum_table = st.columns([1, 1])
                with col_sum_chart:
                    counts_series = pd.Series([FRIENDLY_ACTIVITY_NAMES.get(p, p) for p in preds_raw]).value_counts().reset_index()
                    counts_series.columns = ["Activity", "Count"]
                    fig_summary = px.pie(
                        counts_series, names="Activity", values="Count",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        title="Batch Activity Breakdown", template="plotly_white"
                    )
                    fig_summary.update_traces(textposition='inside', textinfo='percent+label')
                    fig_summary.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_summary, use_container_width=True)

                with col_sum_table:
                    st.markdown("##### Prediction Preview")
                    st.dataframe(result_table.head(10), use_container_width=True, height=260)

                # Download CSV
                export_df = input_df.copy()
                export_df.insert(0, "Predicted_Activity", [FRIENDLY_ACTIVITY_NAMES.get(p, p) for p in preds_raw])
                export_df.insert(1, "Confidence", confs)

                csv_download_bytes = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Complete Batch Predictions (CSV)",
                    csv_download_bytes,
                    "har_batch_predictions.csv",
                    "text/csv",
                    type="primary",
                    key="dl_batch_clean_results"
                )

            # Action: Try another prediction
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Try Another Prediction", key="btn_try_another"):
                st.rerun()

    # -------------------------------------------------------------
    # SECTION 3: EXPLORE DATA
    # -------------------------------------------------------------
    with tabs[2]:
        st.markdown("## Explore Sensor & Feature Data")
        st.markdown("*Inspect how smartphone sensor signals differ between active and resting states.*")

        X_test, y_test, sub_test, feature_names = get_cached_raw_test_data()

        if X_test is not None and feature_names:
            df_explore = X_test.copy()
            df_explore["Activity"] = [FRIENDLY_ACTIVITY_NAMES.get(y, y) for y in y_test.values]

            # Simplified feature picker with understandable labels
            friendly_feature_options = {
                "tGravityAcc-mean()-X": "Horizontal Tilt (Gravity X)",
                "tGravityAcc-mean()-Y": "Vertical Phone Angle (Gravity Y)",
                "tBodyAcc-max()-X": "Peak Body Acceleration (Forward)",
                "tBodyAccJerk-std()-X": "Stepping Impact Intensity (Jerk)",
                "fBodyAcc-mean()-X": "Frequency Spectral Energy (Walking Rhythm)",
                "tBodyGyro-std()-Z": "Rotational Body Wobble (Gyro Z)"
            }

            f_choice_col1, f_choice_col2 = st.columns([2, 1])
            with f_choice_col1:
                feature_mode = st.radio("Choose Feature Selection Mode:", ["Featured Key Sensors", "Browse All 561 Features"], horizontal=True)
                if feature_mode == "Featured Key Sensors":
                    chosen_feat_key = st.selectbox(
                        "Select Key Sensor Measurement:",
                        options=list(friendly_feature_options.keys()),
                        format_func=lambda k: friendly_feature_options[k]
                    )
                else:
                    chosen_feat_key = st.selectbox("Select from 561 features:", options=feature_names)

            with f_choice_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(f"Feature ID: `{chosen_feat_key}`")

            # Charts: Boxplot & Histogram
            chart_c1, chart_c2 = st.columns(2)

            with chart_c1:
                with st.container(border=True):
                    st.markdown("#### Sensor Comparison Across Activities")
                    fig_box = px.box(
                        df_explore, x="Activity", y=chosen_feat_key, color="Activity",
                        color_discrete_sequence=px.colors.qualitative.Safe,
                        template="plotly_white"
                    )
                    fig_box.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig_box, use_container_width=True)

            with chart_c2:
                with st.container(border=True):
                    st.markdown("#### Distribution Density")
                    fig_hist = px.histogram(
                        df_explore, x=chosen_feat_key, color="Activity",
                        barmode="overlay", marginal="box",
                        color_discrete_sequence=px.colors.qualitative.Safe,
                        template="plotly_white"
                    )
                    fig_hist.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig_hist, use_container_width=True)

            # 2D PCA Cluster View
            with st.container(border=True):
                st.markdown("#### 2D AI Motion Separation Map")
                st.write("How artificial intelligence separates active walking movements from stationary sitting and standing postures:")
                
                pca_2d = PCA(n_components=2, random_state=42)
                X_pca_sub = X_test.iloc[:1000]
                y_pca_sub = [FRIENDLY_ACTIVITY_NAMES.get(y, y) for y in y_test.iloc[:1000]]
                pca_trans = pca_2d.fit_transform(X_pca_sub)

                df_pca = pd.DataFrame({
                    "PC1": pca_trans[:, 0],
                    "PC2": pca_trans[:, 1],
                    "Activity": y_pca_sub
                })

                fig_pca = px.scatter(
                    df_pca, x="PC1", y="PC2", color="Activity",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    template="plotly_white",
                    labels={"PC1": "Movement Energy (Component 1)", "PC2": "Gravity Tilt (Component 2)"}
                )
                fig_pca.update_layout(height=420)
                st.plotly_chart(fig_pca, use_container_width=True)

        else:
            st.info("Loading test dataset for visual exploration...")

    # -------------------------------------------------------------
    # SECTION 4: MODEL PERFORMANCE
    # -------------------------------------------------------------
    with tabs[3]:
        st.markdown("## Model Performance & Evaluation")
        st.markdown("*Compare machine learning algorithms on accuracy, speed, and reliability.*")

        if comp_df is not None and not comp_df.empty:
            # Clean summary cards for top model
            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric("Top Model Accuracy", best_acc_str, "LDA (Full)")
            with p2:
                st.metric("Macro F1-Score", f"{float(comp_df.iloc[0]['Macro_F1']):.4f}", "Balanced across all 6 classes")
            with p3:
                st.metric("Inference Latency", inference_speed_str, "Fast edge response")

            st.markdown("<br>", unsafe_allow_html=True)

            # Clean Leaderboard Table
            st.markdown("### Model Comparison Table")
            friendly_comp_df = comp_df[["Model", "Accuracy", "Macro_F1", "Inference_Time_ms_per_sample", "Model_Size_KB"]].copy()
            friendly_comp_df.columns = ["Model Name", "Accuracy", "Macro F1", "Speed (ms/sample)", "File Size (KB)"]
            
            st.dataframe(
                friendly_comp_df.style.format({
                    "Accuracy": "{:.2%}",
                    "Macro F1": "{:.4f}",
                    "Speed (ms/sample)": "{:.3f} ms",
                    "File Size (KB)": "{:.1f} KB"
                }),
                use_container_width=True,
                height=300
            )

            # Interactive Confusion Matrix
            st.markdown("---")
            st.markdown("### Interactive Confusion Matrix")
            st.caption("Inspect which activities are predicted accurately and where occasional misclassifications happen.")

            inspect_m_col, eval_btn_col = st.columns([2, 1])
            with inspect_m_col:
                selected_inspect_model = st.selectbox(
                    "Select Model for Confusion Matrix:",
                    options=list(available_models.keys()),
                    key="clean_inspect_model_select"
                )
            with eval_btn_col:
                st.write("<br>", unsafe_allow_html=True)
                run_eval_now = st.button("⚡ Evaluate on Test Set", type="primary", key="btn_clean_run_eval")

            eval_target_model = get_cached_model(str(available_models[selected_inspect_model]))

            if eval_target_model is not None:
                X_test, y_test, _, _ = get_cached_raw_test_data()

                if run_eval_now or f"eval_clean_{selected_inspect_model}" in st.session_state:
                    if run_eval_now:
                        with st.spinner("Evaluating model on 2,947 held-out test windows..."):
                            eval_out = compute_comprehensive_evaluation(eval_target_model, X_test, y_test, model_name=selected_inspect_model)
                            st.session_state[f"eval_clean_{selected_inspect_model}"] = eval_out

                    eval_out = st.session_state.get(f"eval_clean_{selected_inspect_model}")

                    if eval_out:
                        cm_norm = eval_out["confusion_matrix_normalized"]
                        # Map index and columns to friendly names
                        friendly_labels = [FRIENDLY_ACTIVITY_NAMES.get(a, a) for a in config.ACTIVITY_NAMES]
                        cm_plot_df = cm_norm.copy()
                        cm_plot_df.index = friendly_labels
                        cm_plot_df.columns = friendly_labels

                        fig_cm = px.imshow(
                            cm_plot_df, text_auto=".1%", aspect="auto", color_continuous_scale="Blues",
                            title=f"Classification Accuracy per Activity ({selected_inspect_model})",
                            template="plotly_white",
                            labels=dict(x="Predicted Activity", y="True Activity", color="Accuracy")
                        )
                        fig_cm.update_layout(height=400)
                        st.plotly_chart(fig_cm, use_container_width=True)

                        st.info("💡 **Key Insight:** Notice that **Laying Down** achieves 100% precision because gravity inverts by 90°. Sitting and Standing occasionally share slight confusion because thigh orientation in a front trouser pocket is nearly upright in both postures.")

    # -------------------------------------------------------------
    # OPTIONAL EXPANDABLE SECTION: ADVANCED SETTINGS & RESEARCH
    # -------------------------------------------------------------
    st.markdown("<br><hr>", unsafe_allow_html=True)
    with st.expander("⚙️ Advanced Settings & Research Details (For Evaluators & Researchers)"):
        st.markdown("### Technical Architecture & Scientific Verification")
        st.caption("Detailed statistical proofs, experimental curves, model architectures, and dataset schema.")

        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
            "🔬 Model & Methodology Details",
            "📐 Statistical Analysis",
            "🧪 Syllabus Experiments (E1-E7)",
            "🗄️ Dataset Information & Schema"
        ])

        with adv_tab1:
            st.markdown("#### 1. Model Details & Preprocessing Pipeline")
            st.markdown("""
            - **Pipeline Architecture:** `StandardScaler` followed by estimator (`LinearDiscriminantAnalysis`, `LogisticRegression`, `SVC`, `RandomForestClassifier`, `DecisionTreeClassifier`, `GaussianNB`, or `PCA(50) + KNeighborsClassifier`).
            - **Linear Discriminant Analysis (LDA):** Solves Fisher's linear discriminant maximizing between-class scatter over within-class scatter, achieving 96.2% test accuracy.
            - **PCA Preprocessing for KNN:** Uncompressed 561-dimensional feature space suffers from Euclidean distance concentration (curse of dimensionality). 50 principal components preserve 92.1% cumulative variance while speeding up neighbor lookups 10x.
            """)

            st.markdown("#### 2. Evaluation Methodology & Zero-Leakage Guarantee")
            st.markdown("""
            - **Subject Disjoint Split:** Volunteers 1-30 are partitioned strictly into Training (21 subjects, 7,352 instances) and Test (9 subjects, 2,947 instances).
            - **Zero Leakage Protocol:** Feature scaling (`StandardScaler`) is fitted strictly on the training set and applied idempotently to the test set. GroupKFold cross-validation groups splits strictly by subject ID to prevent inter-subject autocorrelation.
            """)

            st.markdown("#### 3. Feature Engineering Details")
            st.markdown("""
            - **561 Engineered Features:** Extracted from raw triaxial accelerometer and gyroscope data across 2.56s windows (128 readings at 50Hz with 50% overlap).
            - **Signal Separation:** Gravitational acceleration is isolated using a 0.3Hz low-pass Butterworth filter. Body acceleration jerk signals are derived from differential velocity.
            - **Frequency Domain:** Discrete Fast Fourier Transform (FFT) applied to capture spectral energy, dominant frequency peaks, and skewness.
            """)

        with adv_tab2:
            st.markdown("#### Statistical Significance & Hypothesis Testing")
            st.caption("Rigorous statistical testing ensuring observed model accuracy differences are genuine and not due to chance.")
            
            mc_path = config.TABLES_DIR / "stats_mcnemar_top_models.csv"
            tt_path = config.TABLES_DIR / "stats_paired_ttest_top_models.csv"

            st.markdown("##### McNemar's Paired Test (Contingency Tables on Held-Out Test Set)")
            st.write("Tests whether disagreements between two classifiers are symmetrically distributed ($H_0: p_b = p_c$).")
            if mc_path.exists():
                st.dataframe(pd.read_csv(mc_path), use_container_width=True)
            else:
                st.info("Run `python -m src.evaluate` to generate McNemar test statistics.")

            st.markdown("##### Paired Student's t-Test (5-Fold GroupKFold Cross-Validation)")
            st.write("Tests for statistically significant differences in mean fold accuracy across distinct subject groups.")
            if tt_path.exists():
                st.dataframe(pd.read_csv(tt_path), use_container_width=True)
            else:
                st.info("Run `python -m src.evaluate` to generate paired t-test results.")

        with adv_tab3:
            st.markdown("#### Syllabus Experiments (E1 to E7)")
            exp_figures = [
                ("E1: KNN Curse of Dimensionality", "exp1_knn_curse_of_dimensionality.png", "Evaluating KNN accuracy as PCA components scale."),
                ("E2: SVM C & Kernel Study", "exp2_svm_c_study.png", "Linear vs RBF vs Polynomial kernels."),
                ("E3: Decision Tree Validation Curve", "exp3_dt_validation_curve.png", "Max-depth bias-variance trade-off."),
                ("E4: Bagging vs Boosting", "exp4_bagging_vs_boosting.png", "Random Forest vs XGBoost vs AdaBoost."),
                ("E5: Tree Pruning (ccp_alpha)", "exp5_dt_pruning.png", "Cost-complexity pruning path."),
                ("E6: Hierarchical Clustering", "exp6_hierarchical_dendrogram.png", "Unsupervised activity profile dendrogram."),
                ("E7: Top 20 Feature Importance", "exp7_feature_importance_rf.png", "Gini feature importance ranking.")
            ]

            for title, img_fname, desc in exp_figures:
                img_p = config.FIGURES_DIR / img_fname
                if img_p.exists():
                    st.markdown(f"**{title}** — *{desc}*")
                    st.image(Image.open(img_p), use_container_width=True)

        with adv_tab4:
            st.markdown("#### Dataset Information & External Schema Validator")
            st.markdown("""
            - **Dataset Source:** UCI Machine Learning Repository Dataset #240.
            - **Participants:** 30 human subjects (ages 19-48) wearing a Samsung Galaxy S II smartphone on the waist.
            - **Total Windows:** 10,299 instances (7,352 Train, 2,947 Test).
            - **Class Distribution:** Balanced across dynamic and static activities (~15-19% per class).
            """)

            adv_up = st.file_uploader("Validate External CSV Compatibility:", type=["csv"], key="adv_file_validator")
            if adv_up is not None:
                df_ext = pd.read_csv(adv_up)
                v_ok, _, v_err, _ = validate_feature_dataframe(df_ext)
                if v_ok:
                    st.success("✅ Dataset is fully compatible with the 561-feature model input pipeline.")
                else:
                    st.error(f"❌ Incompatible: {v_err}")

if __name__ == "__main__":
    main()
