import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

from src.utils import RANDOM_STATE, get_logger, save_pickle, OUTPUT_DIR

logger = get_logger("experiments")

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ─── Parameter sweep definitions ─────────────────────────────────────────────
SWEEP_CONFIG = {
    "LinearRegression": {
        "param_name": "alpha",
        "param_values": [0.01, 0.1, 1.0, 10.0, 100.0],
        "build_fn": lambda v: Ridge(alpha=v),
    },
    "RandomForest": {
        "param_name": "n_estimators",
        "param_values": [50, 100, 150, 200, 300],
        "build_fn": lambda v: RandomForestRegressor(
            n_estimators=v, max_depth=15, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE
        ),
    },
    "XGBoost": {
        "param_name": "max_depth",
        "param_values": [3, 4, 5, 6, 8, 10],
        "build_fn": lambda v: XGBRegressor(
            n_estimators=200, max_depth=v, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0
        ),
    },
    "CatBoost": {
        "param_name": "iterations",
        "param_values": [50, 100, 150, 200, 300],
        "build_fn": lambda v: CatBoostRegressor(
            iterations=v, depth=6, learning_rate=0.05,
            random_seed=RANDOM_STATE, verbose=0
        ),
    },
}

# ─── Run sweeps ───────────────────────────────────────────────────────────────
def run_parameter_sweeps(X_train, y_train, X_val, y_val) -> dict:
    """
    Returns:
        history = {
            model_name: {
                param_value: {"train_loss": float, "val_loss": float}
            }
        }
    """
    history = {}
    for model_name, cfg in SWEEP_CONFIG.items():
        logger.info(f"Sweeping {model_name} over {cfg['param_name']} ...")
        model_history = {}
        for val in cfg["param_values"]:
            model = cfg["build_fn"](val)
            model.fit(X_train, y_train)
            train_loss = _rmse(y_train, model.predict(X_train))
            val_loss   = _rmse(y_val,   model.predict(X_val))
            model_history[val] = {"train_loss": train_loss, "val_loss": val_loss}
            logger.info(f"  {cfg['param_name']}={val} | train={train_loss:.4f} val={val_loss:.4f}")
        history[model_name] = model_history

    save_pickle(history, OUTPUT_DIR / "learning_curves.pkl")
    logger.info("Experiment history saved.")
    return history