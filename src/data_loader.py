"""
Data loader module for UCI HAR Dataset.
Handles downloading, extracting, parsing, deduplicating feature names,
and verifying no subject leakage between train and test sets.
"""

import os
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

import config
from src.utils import logger

def download_and_extract_data(data_dir: Path = config.DATA_DIR) -> Path:
    """
    Downloads UCI HAR Dataset zip archive from official UCI URL if not present.
    Extracts to data_dir / 'UCI HAR Dataset'.
    """
    target_dir = config.RAW_DATA_DIR
    if (target_dir / "train" / "X_train.txt").exists():
        logger.info(f"Dataset already exists at {target_dir}")
        return target_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / config.ZIP_FILENAME

    # Step 1: Download zip if missing
    if not zip_path.exists():
        logger.info(f"Downloading dataset from {config.DATASET_URL}...")
        try:
            # Custom User-Agent header in case archive.ics.uci.edu blocks python default requests
            req = urllib.request.Request(
                config.DATASET_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            print("\n=======================================================")
            print("MANUAL DOWNLOAD INSTRUCTIONS:")
            print("1. Download UCI HAR Dataset manually from:")
            print("   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones")
            print(f"2. Place the extracted 'UCI HAR Dataset' folder into: {config.DATA_DIR}")
            print("=======================================================\n")
            raise e

    # Step 2: Extract primary zip
    logger.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Check if inner zip exists (UCI archive sometimes nests UCI HAR Dataset.zip inside outer zip)
    inner_zip = data_dir / config.INNER_ZIP_FILENAME
    if inner_zip.exists():
        logger.info(f"Extracting inner zip {inner_zip}...")
        with zipfile.ZipFile(inner_zip, 'r') as inner_ref:
            inner_ref.extractall(data_dir)

    if not (target_dir / "train" / "X_train.txt").exists():
        raise FileNotFoundError(f"Extraction failed: Expected files in {target_dir} not found.")

    logger.info(f"Data successfully prepared at {target_dir}")
    return target_dir


def load_raw_data(raw_data_dir: Path = config.RAW_DATA_DIR):
    """
    Loads raw UCI HAR Dataset text files into pandas DataFrames and numpy arrays.
    Handles duplicate feature names in features.txt by making them unique.
    
    Returns:
        X_train (pd.DataFrame), y_train (pd.Series), subject_train (pd.Series),
        X_test (pd.DataFrame), y_test (pd.Series), subject_test (pd.Series),
        feature_names (list)
    """
    if not (raw_data_dir / "train" / "X_train.txt").exists():
        download_and_extract_data(raw_data_dir.parent)

    # Load feature names
    features_path = raw_data_dir / "features.txt"
    raw_features = []
    with open(features_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                raw_features.append(parts[1])
            elif len(parts) == 1:
                raw_features.append(parts[0])

    # Handle duplicate feature names (e.g., tBodyAcc-bandsEnergy()-1,8 occurs multiple times)
    feature_names = []
    counts = {}
    for feat in raw_features:
        if feat in counts:
            counts[feat] += 1
            feature_names.append(f"{feat}_{counts[feat]}")
        else:
            counts[feat] = 0
            feature_names.append(feat)

    num_duplicates = sum(v > 0 for v in counts.values())
    if num_duplicates > 0:
        logger.info(f"Handled {num_duplicates} duplicate feature names by appending unique indices.")

    # Function to read space-delimited text matrices efficiently
    def read_matrix(filepath):
        return pd.read_csv(filepath, sep=r'\s+', header=None, names=feature_names)

    def read_vector(filepath):
        return pd.read_csv(filepath, sep=r'\s+', header=None).values.ravel()

    logger.info("Loading training set...")
    X_train = read_matrix(raw_data_dir / "train" / "X_train.txt")
    y_train_raw = read_vector(raw_data_dir / "train" / "y_train.txt")
    subject_train = pd.Series(read_vector(raw_data_dir / "train" / "subject_train.txt"), name="subject")

    logger.info("Loading test set...")
    X_test = read_matrix(raw_data_dir / "test" / "X_test.txt")
    y_test_raw = read_vector(raw_data_dir / "test" / "y_test.txt")
    subject_test = pd.Series(read_vector(raw_data_dir / "test" / "subject_test.txt"), name="subject")

    # Map target integer labels to string activity names
    y_train = pd.Series(y_train_raw, name="activity").map(config.ACTIVITY_MAP)
    y_test = pd.Series(y_test_raw, name="activity").map(config.ACTIVITY_MAP)

    # NO LEAKAGE PROOF: Subject-wise split check
    train_subjects = set(subject_train.unique())
    test_subjects = set(subject_test.unique())
    assert train_subjects.isdisjoint(test_subjects), (
        f"DATA LEAKAGE ERROR: Train and Test subjects overlap! Overlap: {train_subjects & test_subjects}"
    )
    print("=====================================================================")
    print(f"PROOF OF NO SUBJECT LEAKAGE:")
    print(f"  Training Subjects ({len(train_subjects)}): {sorted(list(train_subjects))}")
    print(f"  Testing Subjects  ({len(test_subjects)}):  {sorted(list(test_subjects))}")
    print(f"  Disjoint Check:   SUCCESS (Intersection = {train_subjects & test_subjects})")
    print("=====================================================================")

    return X_train, y_train, subject_train, X_test, y_test, subject_test, feature_names


def load_test_data(raw_data_dir: Path = config.RAW_DATA_DIR):
    """
    Fast loader for test split only (skips reading 66MB training matrix).
    Loads in <0.5s for snappy UI and responsive exploration.
    """
    if not (raw_data_dir / "test" / "X_test.txt").exists():
        download_and_extract_data(raw_data_dir.parent)

    features_path = raw_data_dir / "features.txt"
    raw_features = []
    with open(features_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                raw_features.append(parts[1])
            elif len(parts) == 1:
                raw_features.append(parts[0])

    feature_names = []
    counts = {}
    for feat in raw_features:
        if feat in counts:
            counts[feat] += 1
            feature_names.append(f"{feat}_{counts[feat]}")
        else:
            counts[feat] = 0
            feature_names.append(feat)

    X_test = pd.read_csv(raw_data_dir / "test" / "X_test.txt", sep=r'\s+', header=None, names=feature_names)
    y_test_raw = pd.read_csv(raw_data_dir / "test" / "y_test.txt", sep=r'\s+', header=None).values.ravel()
    subject_test = pd.Series(pd.read_csv(raw_data_dir / "test" / "subject_test.txt", sep=r'\s+', header=None).values.ravel(), name="subject")
    y_test = pd.Series(y_test_raw, name="activity").map(config.ACTIVITY_MAP)

    return X_test, y_test, subject_test, feature_names


def perform_data_quality_checks(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Performs comprehensive data quality checks and returns summary dict.
    """
    checks = {}
    checks["missing_train"] = X_train.isnull().sum().sum()
    checks["missing_test"] = X_test.isnull().sum().sum()
    checks["duplicates_train"] = X_train.duplicated().sum()
    checks["duplicates_test"] = X_test.duplicated().sum()
    checks["n_features"] = X_train.shape[1]
    checks["n_train_samples"] = X_train.shape[0]
    checks["n_test_samples"] = X_test.shape[0]

    # Feature range check
    checks["feature_min"] = float(X_train.min().min())
    checks["feature_max"] = float(X_train.max().max())

    # Variance check
    variances = X_train.var()
    near_zero_var = (variances < 1e-4).sum()
    checks["near_zero_var_count"] = int(near_zero_var)

    # Correlation check (|r| > 0.95)
    # Compute on a subset if speed is needed, but 561 features correlation matrix is 561x561 (fast enough)
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = (upper_tri > 0.95).sum().sum()
    checks["high_corr_pairs_gt_0.95"] = int(high_corr_pairs)

    logger.info("Data Quality Checks:")
    for k, v in checks.items():
        logger.info(f"  {k}: {v}")

    return checks

if __name__ == "__main__":
    X_train, y_train, sub_train, X_test, y_test, sub_test, feats = load_raw_data()
    checks = perform_data_quality_checks(X_train, y_train, X_test, y_test)
