"""
Utility functions for logging, reproducibility, model saving/loading, and table export.
"""

import logging
import random
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import config

def get_logger(name: str = "UCI_HAR") -> logging.Logger:
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = get_logger()

def set_seed(seed: int = config.RANDOM_SEED):
    """Sets random seed across numpy, python random, and environment."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")

def save_model(model, filepath: Path, compress: int = 3):
    """Saves a fitted model or pipeline using joblib compression."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath, compress=compress)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"Saved model to {filepath} ({size_mb:.2f} MB, compress={compress})")

def load_model(filepath: Path):
    """Loads a joblib saved model."""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at {filepath}")
    return joblib.load(filepath)

def save_table(df: pd.DataFrame, filename_stem: str, tables_dir: Path = config.TABLES_DIR):
    """Saves a DataFrame as both CSV and Markdown table."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{filename_stem}.csv"
    md_path = tables_dir / f"{filename_stem}.md"
    
    df.to_csv(csv_path, index=True)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(df.to_markdown())
    logger.info(f"Saved table '{filename_stem}' to CSV and Markdown in {tables_dir}")
