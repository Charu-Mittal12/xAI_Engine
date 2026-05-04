import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "log_price"

# ─── Logging ──────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# ─── I/O helpers ──────────────────────────────────────────────────────────────
def save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def save_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

# ─── Data loader ──────────────────────────────────────────────────────────────
def load_raw_data(filename: str = "Airbnb_Data.csv") -> pd.DataFrame:
    """Load raw Airbnb listings CSV, stripping $ and commas from price."""
    logger = get_logger("utils")
    path = DATA_DIR / filename
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols from {path}")
    if "host_response_rate" in df.columns:
        df["host_response_rate"] = (
            df["host_response_rate"]
            .str.replace("%", "", regex=False)
            .astype(float)
        )
    # Then add "host_response_rate" to NUMERICAL_FEATURES
    # Drop rows with missing/zero target
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] > 0)].reset_index(drop=True)
    logger.info(f"After price cleaning: {df.shape[0]} rows")
    return df

# ─── Feature catalog ──────────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "accommodates", "bathrooms", "bedrooms", "beds",
    "number_of_reviews", "review_scores_rating",
    "latitude", "longitude",
]

CATEGORICAL_FEATURES = [
    "neighbourhood", "room_type", "property_type",
    "city", "bed_type", "cancellation_policy",
    "instant_bookable", "host_identity_verified",
    "host_has_profile_pic",
]

def get_feature_lists(df: pd.DataFrame):
    """Return only the feature lists that actually exist in df."""
    num_feats = [c for c in NUMERICAL_FEATURES if c in df.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num_feats, cat_feats