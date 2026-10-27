"""
Inference and Input Validation Module for UCI HAR Telemetry Engine.
Handles robust CSV validation, manual feature customization via calibrated presets,
model registry management, batch prediction execution, and latency benchmarking.
"""

from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

import config
from src.utils import logger, load_model

# Key top-ranking features across feature importance studies (E7) with intuitive physical descriptors
KEY_FEATURE_DESCRIPTORS = [
    {
        "id": "tGravityAcc-mean()-X",
        "label": "Gravity Accel X (Horizontal Tilt)",
        "desc": "Mean gravity component on horizontal X-axis. Distinguishes lying/lateral postures.",
        "min": -1.0, "max": 1.0, "step": 0.02
    },
    {
        "id": "tGravityAcc-mean()-Y",
        "label": "Gravity Accel Y (Vertical Plane)",
        "desc": "Mean gravity component along phone vertical length. Inverts when standing vs upside-down.",
        "min": -1.0, "max": 1.0, "step": 0.02
    },
    {
        "id": "tGravityAcc-mean()-Z",
        "label": "Gravity Accel Z (Orthogonal Depth)",
        "desc": "Perpendicular gravity component relative to screen face.",
        "min": -1.0, "max": 1.0, "step": 0.02
    },
    {
        "id": "tBodyAcc-max()-X",
        "label": "Peak Body Accel X (Forward Burst)",
        "desc": "Maximum linear acceleration during movement cycle. High in downstairs & running.",
        "min": -1.0, "max": 1.0, "step": 0.02
    },
    {
        "id": "tBodyAccJerk-std()-X",
        "label": "Body Jerk StdDev X (Impact Turbulence)",
        "desc": "Rate of change of body acceleration. High in dynamic stepping, zero when static.",
        "min": -1.0, "max": 1.0, "step": 0.02
    },
    {
        "id": "fBodyAcc-mean()-X",
        "label": "Frequency Body Accel X (Spectral Energy)",
        "desc": "FFT magnitude mean across body acceleration spectrum.",
        "min": -1.0, "max": 1.0, "step": 0.02
    },
    {
        "id": "fBodyAccJerk-bandsEnergy()-1,8",
        "label": "Cadence Band Energy (1-8 Hz)",
        "desc": "Energy in typical human stepping cadence bandwidth (1 to 2.5 Hz harmonic).",
        "min": -1.0, "max": 1.0, "step": 0.02
    },
    {
        "id": "tBodyGyro-std()-Z",
        "label": "Gyro Yaw StdDev (Rotational Wobble)",
        "desc": "Rotational velocity variance around vertical axis while walking.",
        "min": -1.0, "max": 1.0, "step": 0.02
    },
    {
        "id": "angle(Y,gravityMean)",
        "label": "Angle Y to Gravity Vector",
        "desc": "Geometric angle between device Y-axis and mean gravity vector. Critical for Sitting vs Standing.",
        "min": -1.0, "max": 1.0, "step": 0.02
    },
    {
        "id": "tBodyAcc-energy()-X",
        "label": "Kinetic Energy X",
        "desc": "Normalized energy of forward bodily oscillations.",
        "min": -1.0, "max": 1.0, "step": 0.02
    }
]


def get_available_models(models_dir: Path = config.MODELS_DIR) -> Dict[str, Path]:
    """
    Discovers all available trained model pipelines in the models directory.
    Returns dictionary mapping display name to Path.
    """
    model_dict = {}

    # Check metadata.json first
    meta_path = models_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if "available_model_files" in meta:
                for display_name, fname in meta["available_model_files"].items():
                    m_path = models_dir / fname
                    if m_path.exists():
                        model_dict[display_name] = m_path
        except Exception as e:
            logger.warning(f"Could not read metadata.json available models: {e}")

    # Fallback / augment by globbing models directory
    for joblib_file in models_dir.glob("*.joblib"):
        stem = joblib_file.stem
        if stem == "best_model":
            if "Best Model (Saved SOTA)" not in model_dict:
                model_dict["Best Model (Saved SOTA)"] = joblib_file
            continue
        
        display = stem.replace("_", " ").title()
        if display not in model_dict:
            model_dict[display] = joblib_file

    if not model_dict:
        # If nothing found, check best_model.joblib
        best_p = models_dir / "best_model.joblib"
        if best_p.exists():
            model_dict["Best Model (Default)"] = best_p

    return model_dict


def load_inference_model(model_path: Path):
    """Safely loads and caches a model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return load_model(model_path)


def validate_feature_dataframe(
    df: pd.DataFrame, 
    expected_feature_count: int = 561
) -> Tuple[bool, Optional[pd.DataFrame], Optional[str], List[str]]:
    """
    Thorough validation of user-supplied sensor feature DataFrame.
    
    Checks:
    1. Non-empty structure.
    2. Strips metadata columns ('true_activity', 'activity', 'subject', 'label').
    3. Handles non-numeric values.
    4. Detects NaN, nulls, and infs (imputing or flagging).
    5. Validates 561-feature dimensionality.
    
    Returns:
        (is_valid: bool, cleaned_df: Optional[pd.DataFrame], error_msg: Optional[str], warnings: List[str])
    """
    warnings = []

    if df is None or df.empty:
        return False, None, "Uploaded data is empty. Please supply a CSV with valid numerical data.", warnings

    # Identify and separate known label/subject columns
    drop_cols = [c for c in df.columns if str(c).strip().lower() in [
        'true_activity', 'activity', 'subject', 'label', 'target', 'y', 'subject_id'
    ]]
    
    if drop_cols:
        warnings.append(f"Detected and ignored non-feature metadata columns: {drop_cols}")
        cleaned = df.drop(columns=drop_cols)
    else:
        cleaned = df.copy()

    # Verify column count
    num_cols = cleaned.shape[1]
    if num_cols != expected_feature_count:
        return (
            False, None, 
            f"Dimension mismatch: Expected exactly {expected_feature_count} sensor features, but received {num_cols} columns.",
            warnings
        )

    # Attempt numeric conversion
    non_numeric_cols = []
    for col in cleaned.columns:
        if not pd.api.types.is_numeric_dtype(cleaned[col]):
            try:
                cleaned[col] = pd.to_numeric(cleaned[col])
            except (ValueError, TypeError):
                non_numeric_cols.append(col)

    if non_numeric_cols:
        return (
            False, None,
            f"Non-numeric values detected in {len(non_numeric_cols)} column(s) (e.g., {non_numeric_cols[:3]}). All 561 features must be numerical.",
            warnings
        )

    # Check for NaN / Infs
    nan_count = cleaned.isnull().sum().sum()
    inf_count = np.isinf(cleaned.values).sum()

    if nan_count > 0 or inf_count > 0:
        warnings.append(f"Found {nan_count} missing (NaN) values and {inf_count} infinite values. Automatically imputed using median column values.")
        # Impute safely with column medians or 0.0
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.fillna(cleaned.median().fillna(0.0))

    # Verify standard range warning for UCI HAR features (standard features are normalized in [-1, 1])
    out_of_bounds = ((cleaned < -10.0) | (cleaned > 10.0)).sum().sum()
    if out_of_bounds > 0:
        warnings.append(f"Notice: {out_of_bounds} feature values lie outside the typical [-1.0, 1.0] normalized UCI HAR range. Preprocessing scaler will adjust them, but verify your sensor unit scaling.")

    return True, cleaned, None, warnings


def run_inference(model: Any, X: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes real inference through the model pipeline and measures execution latency.
    
    Returns structured results:
        predictions: List[str]
        probabilities: np.ndarray (N x 6)
        confidences: List[float]
        latency_ms_per_sample: float
        class_names: List[str]
    """
    t0 = time.perf_counter()
    raw_preds = model.predict(X)
    raw_probs = model.predict_proba(X)
    elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
    latency_per_sample = elapsed_total_ms / max(1, len(X))

    # Normalize predictions to strings
    predictions = [
        config.ACTIVITY_NAMES[p] if isinstance(p, (int, np.integer)) else str(p)
        for p in raw_preds
    ]

    # Calculate confidence per sample
    confidences = []
    for i, pred_label in enumerate(predictions):
        if pred_label in config.ACTIVITY_NAMES:
            idx = config.ACTIVITY_NAMES.index(pred_label)
            conf = float(raw_probs[i, idx])
        else:
            conf = float(np.max(raw_probs[i]))
        confidences.append(conf)

    return {
        "predictions": predictions,
        "probabilities": raw_probs,
        "confidences": confidences,
        "latency_ms_per_sample": latency_per_sample,
        "total_latency_ms": elapsed_total_ms,
        "class_names": config.ACTIVITY_NAMES
    }


def load_preset_profiles() -> Dict[str, Dict[str, float]]:
    """Loads pre-calculated class feature centroids from preset_profiles.json."""
    preset_file = config.APP_DIR / "preset_profiles.json"
    if preset_file.exists():
        try:
            with open(preset_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read preset_profiles.json: {e}")

    # Fallback to zeros if missing
    feature_names = []
    meta_p = config.MODELS_DIR / "metadata.json"
    if meta_p.exists():
        with open(meta_p, 'r') as f:
            feature_names = json.load(f).get("feature_names", [])

    return {
        act: {f: 0.0 for f in feature_names}
        for act in config.ACTIVITY_NAMES
    }
