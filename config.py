"""
Configuration module for UCI HAR project.
Contains paths, constants, seed, activity label mapping, and style settings.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "UCI HAR Dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
APP_DIR = BASE_DIR / "app"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
TESTS_DIR = BASE_DIR / "tests"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR, APP_DIR, NOTEBOOKS_DIR, TESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_URL = "https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip"
DATASET_BACKUP_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ZIP_FILENAME = "UCI_HAR_Dataset.zip"
INNER_ZIP_FILENAME = "UCI HAR Dataset.zip"

# Global settings
RANDOM_SEED = 42

# Activity Mapping
ACTIVITY_MAP = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

ACTIVITY_IDS = list(ACTIVITY_MAP.keys())
ACTIVITY_NAMES = list(ACTIVITY_MAP.values())

DYNAMIC_ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
STATIC_ACTIVITIES = ["SITTING", "STANDING", "LAYING"]

# Color palette for plots (harmonious, modern palette)
ACTIVITY_COLORS = {
    "WALKING": "#1f77b4",          # Blue
    "WALKING_UPSTAIRS": "#ff7f0e", # Orange
    "WALKING_DOWNSTAIRS": "#2ca02c",# Green
    "SITTING": "#d62728",           # Red
    "STANDING": "#9467bd",          # Purple
    "LAYING": "#8c564b"             # Brown
}

# Project Metadata for Report / App
PROJECT_TITLE = "Human Activity Recognition (HAR) using Smartphone Sensor Data"
PROJECT_SUBTITLE = "A Comparative Study of Machine Learning Models with a Streamlit Web App"
TEAM_MEMBERS = "Machine Learning Project Team"
GUIDE_NAME = "Faculty Supervisor"
ACADEMIC_YEAR = "2024-2025"
