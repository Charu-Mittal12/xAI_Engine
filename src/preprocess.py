import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    TARGET_COL, RANDOM_STATE, TEST_SIZE,
    get_feature_lists, get_logger, load_raw_data,
    save_pickle, OUTPUT_DIR,
)

logger = get_logger("preprocess")

# ─── Outlier treatment ────────────────────────────────────────────────────────
def remove_price_outliers(df, z_thresh=3.5):
    from scipy.stats import zscore
    mask = np.abs(zscore(df[TARGET_COL])) < z_thresh
    before = len(df)
    df = df[mask].reset_index(drop=True)
    logger.info(f"Outlier removal: {before} → {len(df)} rows")
    return df

# ─── Preprocessing pipeline builder ──────────────────────────────────────────
def build_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ],
        remainder="drop",
    )

def get_feature_names_out(preprocessor: ColumnTransformer, num_features: list, cat_features: list) -> list:
    """Retrieve full feature names post-transform."""
    num_names = num_features
    cat_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(cat_features)
        .tolist()
    )
    return num_names + cat_names

# ─── End-to-end data preparation ─────────────────────────────────────────────
def prepare_data(filename: str = "Airbnb_Data.csv"):
    """
    Returns:
        X_train_t, X_val_t, X_test_t  — transformed numpy arrays
        y_train, y_val, y_test         — log1p-transformed targets (Series)
        feature_names                  — list of final feature names
        preprocessor                   — fitted ColumnTransformer
        X_test_raw                     — raw test DataFrame (for LIME context)
    """
    df = load_raw_data(filename)
    df = remove_price_outliers(df)

    num_feats, cat_feats = get_feature_lists(df)
    logger.info(f"Numerical: {num_feats}")
    logger.info(f"Categorical: {cat_feats}")

    X = df[num_feats + cat_feats].copy()
    y = df[TARGET_COL].copy()  # log-transform for regression stability

    # Train / val / test split  (60 / 20 / 20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE * 2, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(num_feats, cat_feats)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t   = preprocessor.transform(X_val)
    X_test_t  = preprocessor.transform(X_test)

    feature_names = get_feature_names_out(preprocessor, num_feats, cat_feats)
    logger.info(f"Feature matrix shape: train={X_train_t.shape}, val={X_val_t.shape}, test={X_test_t.shape}")

    # Persist preprocessor for inference
    save_pickle(preprocessor, OUTPUT_DIR / "preprocessor.pkl")
    save_pickle(feature_names, OUTPUT_DIR / "feature_names.pkl")

    return X_train_t, X_val_t, X_test_t, y_train, y_val, y_test, feature_names, preprocessor, X_test