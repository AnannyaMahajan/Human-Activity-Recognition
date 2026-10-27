"""
Comprehensive Test Suite for UCI HAR Telemetry Engine.
Covers:
 1. Successful inference on a valid sample.
 2. Correct prediction output schema.
 3. Valid and invalid CSV uploads.
 4. Missing features and unexpected feature dimensions.
 5. Batch prediction.
 6. Model loading failures.
 7. Subject-disjoint train/test splits (Zero Leakage).
 8. Preprocessing without leakage.
 9. Metric calculation and confusion matrix generation.
10. Downloadable prediction results serialization.
11. Preset feature profiles and validation error handling.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
import numpy as np
import pandas as pd

import config
from src.data_loader import load_raw_data
from src.preprocessing import build_preprocessing_pipeline, ActivityClassifierPipeline
from src.utils import load_model
from src.evaluate import calculate_metrics, compute_comprehensive_evaluation
from src.inference import (
    validate_feature_dataframe, run_inference, get_available_models,
    load_preset_profiles, KEY_FEATURE_DESCRIPTORS
)


# -------------------------------------------------------------
# 1. Successful inference on a valid sample
# -------------------------------------------------------------
def test_successful_inference_on_valid_sample():
    """Verify that a saved model runs inference on a valid 561-feature row and yields a valid activity."""
    model_path = config.MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("best_model.joblib not present.")
    
    model = load_model(model_path)
    sample_path = config.APP_DIR / "sample_input.csv"
    assert sample_path.exists(), "sample_input.csv missing from app directory"

    df_sample = pd.read_csv(sample_path)
    X_single = df_sample.drop(columns=['true_activity'], errors='ignore').iloc[[0]]
    
    pred = model.predict(X_single)[0]
    pred_str = config.ACTIVITY_NAMES[pred] if isinstance(pred, (int, np.integer)) else str(pred)
    assert pred_str in config.ACTIVITY_NAMES, f"Invalid predicted activity: {pred_str}"


# -------------------------------------------------------------
# 2. Correct prediction output schema
# -------------------------------------------------------------
def test_prediction_output_schema():
    """Verify output schema of run_inference."""
    model_path = config.MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("best_model.joblib not present.")
    
    model = load_model(model_path)
    sample_path = config.APP_DIR / "sample_input.csv"
    df_sample = pd.read_csv(sample_path)
    X = df_sample.drop(columns=['true_activity'], errors='ignore').iloc[[0]]

    result = run_inference(model, X)
    assert "predictions" in result
    assert "probabilities" in result
    assert "confidences" in result
    assert "latency_ms_per_sample" in result
    assert "class_names" in result

    assert len(result["predictions"]) == 1
    assert result["predictions"][0] in config.ACTIVITY_NAMES
    assert result["probabilities"].shape == (1, 6)
    assert 0.0 <= result["confidences"][0] <= 1.0
    assert result["latency_ms_per_sample"] >= 0.0


# -------------------------------------------------------------
# 3. Valid and invalid CSV uploads
# -------------------------------------------------------------
def test_valid_and_invalid_csv_uploads():
    """Verify CSV validation handles empty, non-numeric, and valid data gracefully."""
    # Empty DataFrame
    is_valid, _, err, _ = validate_feature_dataframe(pd.DataFrame())
    assert not is_valid
    assert "empty" in err.lower()

    # None DataFrame
    is_valid, _, err, _ = validate_feature_dataframe(None)
    assert not is_valid

    # Non-numeric columns
    bad_df = pd.DataFrame({"col_" + str(i): ["string_val"] * 5 for i in range(561)})
    is_valid, _, err, _ = validate_feature_dataframe(bad_df)
    assert not is_valid
    assert "non-numeric" in err.lower()

    # Valid DataFrame
    valid_df = pd.DataFrame(np.random.uniform(-1, 1, size=(5, 561)))
    is_valid, cleaned, err, _ = validate_feature_dataframe(valid_df)
    assert is_valid
    assert err is None
    assert cleaned.shape == (5, 561)


# -------------------------------------------------------------
# 4. Missing features and unexpected feature dimensions
# -------------------------------------------------------------
def test_missing_features_and_unexpected_dimensions():
    """Verify validation flags dimension mismatches (too few or too many features)."""
    # 50 features instead of 561
    small_df = pd.DataFrame(np.random.randn(5, 50))
    is_valid, _, err, _ = validate_feature_dataframe(small_df)
    assert not is_valid
    assert "dimension mismatch" in err.lower()
    assert "50 columns" in err

    # 600 features
    large_df = pd.DataFrame(np.random.randn(5, 600))
    is_valid, _, err, _ = validate_feature_dataframe(large_df)
    assert not is_valid
    assert "dimension mismatch" in err.lower()


# -------------------------------------------------------------
# 5. Batch prediction
# -------------------------------------------------------------
def test_batch_prediction():
    """Verify batch prediction on multiple instances."""
    model_path = config.MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("best_model.joblib not present.")
    
    model = load_model(model_path)
    sample_path = config.APP_DIR / "sample_input.csv"
    df_sample = pd.read_csv(sample_path)
    X_batch = df_sample.drop(columns=['true_activity'], errors='ignore')

    result = run_inference(model, X_batch)
    n_samples = len(X_batch)
    assert len(result["predictions"]) == n_samples
    assert result["probabilities"].shape == (n_samples, 6)
    assert len(result["confidences"]) == n_samples

    # Probabilities should sum to approximately 1.0 per sample
    prob_sums = result["probabilities"].sum(axis=1)
    np.testing.assert_allclose(prob_sums, np.ones(n_samples), rtol=1e-2, atol=1e-2)


# -------------------------------------------------------------
# 6. Model loading failures
# -------------------------------------------------------------
def test_model_loading_failures():
    """Verify loading non-existent model files raises FileNotFoundError."""
    missing_path = config.MODELS_DIR / "non_existent_fake_model.joblib"
    with pytest.raises(FileNotFoundError):
        load_model(missing_path)


# -------------------------------------------------------------
# 7. Subject-disjoint train/test splits (Zero Leakage)
# -------------------------------------------------------------
def test_subject_disjoint_splits():
    """Verify train and test subjects are strictly disjoint and sum to 30."""
    X_train, y_train, sub_train, X_test, y_test, sub_test, _ = load_raw_data()
    train_subs = set(sub_train.unique())
    test_subs = set(sub_test.unique())

    assert train_subs.isdisjoint(test_subs), f"Leakage detected! Overlap: {train_subs & test_subs}"
    assert len(train_subs) == 21, f"Expected 21 train subjects, got {len(train_subs)}"
    assert len(test_subs) == 9, f"Expected 9 test subjects, got {len(test_subs)}"
    assert len(train_subs) + len(test_subs) == 30, f"Expected 30 total subjects."


# -------------------------------------------------------------
# 8. Preprocessing without leakage
# -------------------------------------------------------------
def test_preprocessing_without_leakage():
    """Verify transformers fit strictly on training data."""
    X_train, _, _, X_test, _, _, _ = load_raw_data()
    pipeline = build_preprocessing_pipeline(use_scaler=True, use_pca=False)
    
    # Not fitted before fit
    assert not hasattr(pipeline.named_steps['scaler'], 'mean_')
    
    # Fit only on train
    pipeline.fit(X_train)
    scaler_mean = pipeline.named_steps['scaler'].mean_
    
    # Train means match scaler mean
    np.testing.assert_allclose(scaler_mean, X_train.mean().values, rtol=1e-3, atol=1e-3)
    
    # Test means should NOT match
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(scaler_mean, X_test.mean().values, rtol=1e-2, atol=1e-2)


# -------------------------------------------------------------
# 9. Metric calculation and confusion matrix generation
# -------------------------------------------------------------
def test_metric_calculation_and_confusion_matrix():
    """Verify calculate_metrics and compute_comprehensive_evaluation."""
    y_true = pd.Series(["WALKING", "SITTING", "LAYING", "WALKING", "STANDING", "WALKING_UPSTAIRS"])
    y_pred = pd.Series(["WALKING", "SITTING", "LAYING", "WALKING_UPSTAIRS", "STANDING", "WALKING_UPSTAIRS"])

    metrics = calculate_metrics(y_true, y_pred, model_name="UnitTestModel")
    assert 0.0 <= metrics["Accuracy"] <= 1.0
    assert 0.0 <= metrics["Macro_F1"] <= 1.0
    assert metrics["Accuracy"] == 5 / 6

    # Test comprehensive evaluation function with a real model on sample
    model_path = config.MODELS_DIR / "best_model.joblib"
    if model_path.exists():
        model = load_model(model_path)
        sample_df = pd.read_csv(config.APP_DIR / "sample_input.csv")
        X_test = sample_df.drop(columns=['true_activity'], errors='ignore')
        y_test = sample_df['true_activity']

        comp_eval = compute_comprehensive_evaluation(model, X_test, y_test, model_name="BestModel")
        assert "summary" in comp_eval
        assert "confusion_matrix_counts" in comp_eval
        assert comp_eval["confusion_matrix_counts"].shape == (6, 6)
        assert "per_class_report" in comp_eval
        assert len(comp_eval["per_class_report"]) == 6


# -------------------------------------------------------------
# 10. Downloadable prediction results serialization
# -------------------------------------------------------------
def test_downloadable_prediction_results():
    """Verify conversion of prediction outputs to downloadable CSV format."""
    df_upload = pd.DataFrame(np.random.uniform(-1, 1, size=(4, 561)))
    predictions = ["WALKING", "SITTING", "LAYING", "STANDING"]
    confidences = [0.985, 0.942, 0.999, 0.891]

    result_df = df_upload.copy()
    result_df.insert(0, "Predicted_Activity", predictions)
    result_df.insert(1, "Confidence", confidences)

    csv_bytes = result_df.to_csv(index=False).encode('utf-8')
    assert len(csv_bytes) > 0

    # Read back and verify columns
    from io import BytesIO
    df_readback = pd.read_csv(BytesIO(csv_bytes))
    assert "Predicted_Activity" in df_readback.columns
    assert "Confidence" in df_readback.columns
    assert list(df_readback["Predicted_Activity"]) == predictions


# -------------------------------------------------------------
# 11. Preset feature profiles and validation error handling
# -------------------------------------------------------------
def test_preset_profiles_and_descriptors():
    """Verify preset profiles and key descriptors are available and valid."""
    presets = load_preset_profiles()
    assert len(presets) >= 6, "Expected at least 6 activity profiles in presets"
    for act in config.ACTIVITY_NAMES:
        assert act in presets, f"Activity {act} missing from presets"
        assert len(presets[act]) == 561, f"Expected 561 features in preset for {act}"

    assert len(KEY_FEATURE_DESCRIPTORS) >= 10
    for desc in KEY_FEATURE_DESCRIPTORS:
        assert "id" in desc and "label" in desc and "desc" in desc
