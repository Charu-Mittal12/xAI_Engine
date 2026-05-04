import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import get_logger, save_json, OUTPUT_DIR

logger = get_logger("evaluate")

def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

def evaluate_all_models(trained_models: dict, X_test, y_test) -> dict:
    """Evaluate all models on test set. Returns dict and saves JSON."""
    results = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, name)
        results[name] = metrics
        logger.info(f"{name}: RMSE={metrics['RMSE']:.4f} MAE={metrics['MAE']:.4f} R²={metrics['R2']:.4f}")

    save_json(results, OUTPUT_DIR / "metrics.json")
    return results

def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    rows = list(metrics.values())
    df = pd.DataFrame(rows).set_index("model")
    return df.round(4)