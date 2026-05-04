import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.utils import RANDOM_STATE, get_logger, save_pickle, OUTPUT_DIR

logger   = get_logger("train")
PLOT_DIR = OUTPUT_DIR / "eda"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "LinearRegression": "#90CAF9",
    "RandomForest":     "#A5D6A7",
    "XGBoost":          "#FFCC80",
    "CatBoost":         "#CE93D8",
}


# ══════════════════════════════════════════════════════════════════════════════
# Model configs
# ══════════════════════════════════════════════════════════════════════════════

def get_model_configs() -> dict:
    return {

        "LinearRegression": [
            ("C1: alpha=0.01",  {"alpha": 0.01},  Ridge(alpha=0.01)),
            ("C2: alpha=0.1",   {"alpha": 0.1},   Ridge(alpha=0.1)),
            ("C3: alpha=1.0",   {"alpha": 1.0},   Ridge(alpha=1.0)),
            ("C4: alpha=10.0",  {"alpha": 10.0},  Ridge(alpha=10.0)),
            ("C5: alpha=50.0",  {"alpha": 50.0},  Ridge(alpha=50.0)),
            ("C6: alpha=100.0", {"alpha": 100.0}, Ridge(alpha=100.0)),
        ],

        "RandomForest": [
            ("C1: n=100,d=10,leaf=5,feat=sqrt",
             {"n_estimators": 100, "max_depth": 10,  "min_samples_leaf": 5,  "max_features": "sqrt"},
             RandomForestRegressor(n_estimators=100, max_depth=10,  min_samples_leaf=5,
                                   max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
            ("C2: n=200,d=15,leaf=5,feat=sqrt",
             {"n_estimators": 200, "max_depth": 15,  "min_samples_leaf": 5,  "max_features": "sqrt"},
             RandomForestRegressor(n_estimators=200, max_depth=15,  min_samples_leaf=5,
                                   max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
            ("C3: n=200,d=20,leaf=2,feat=sqrt",
             {"n_estimators": 200, "max_depth": 20,  "min_samples_leaf": 2,  "max_features": "sqrt"},
             RandomForestRegressor(n_estimators=200, max_depth=20,  min_samples_leaf=2,
                                   max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
            ("C4: n=200,d=None,leaf=2,feat=sqrt",
             {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2,  "max_features": "sqrt"},
             RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=2,
                                   max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
            ("C5: n=300,d=None,leaf=1,feat=sqrt",
             {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,  "max_features": "sqrt"},
             RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=1,
                                   max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
            ("C6: n=300,d=None,leaf=1,feat=0.5",
             {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,  "max_features": 0.5},
             RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=1,
                                   max_features=0.5,  n_jobs=-1, random_state=RANDOM_STATE)),
        ],

        "XGBoost": [
            ("C1: n=100,d=4,lr=0.10,sub=0.8",
             {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10,
              "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
             XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.10,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
                          eval_metric="rmse", random_state=RANDOM_STATE, verbosity=0)),
            ("C2: n=200,d=6,lr=0.05,sub=0.8",
             {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
             XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
                          eval_metric="rmse", random_state=RANDOM_STATE, verbosity=0)),
            ("C3: n=300,d=6,lr=0.05,sub=0.9",
             {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
              "subsample": 0.9, "colsample_bytree": 0.8, "reg_alpha": 0.0, "reg_lambda": 1.0},
             XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
                          eval_metric="rmse", random_state=RANDOM_STATE, verbosity=0)),
            ("C4: n=300,d=8,lr=0.03,sub=0.9",
             {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03,
              "subsample": 0.9, "colsample_bytree": 0.7, "reg_alpha": 0.0, "reg_lambda": 1.0},
             XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.03,
                          subsample=0.9, colsample_bytree=0.7, reg_alpha=0.0, reg_lambda=1.0,
                          eval_metric="rmse", random_state=RANDOM_STATE, verbosity=0)),
            ("C5: n=300,d=8,lr=0.03,L1=0.1",
             {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03,
              "subsample": 0.9, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0},
             XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.03,
                          subsample=0.9, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                          eval_metric="rmse", random_state=RANDOM_STATE, verbosity=0)),
            ("C6: n=500,d=6,lr=0.02,L1=0.1,L2=2",
             {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.02,
              "subsample": 0.9, "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 2.0},
             XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.02,
                          subsample=0.9, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=2.0,
                          eval_metric="rmse", random_state=RANDOM_STATE, verbosity=0)),
        ],

        "CatBoost": [
            ("C1: iter=200,d=6,lr=0.10,l2=3",
             {"iterations": 200, "depth": 6, "learning_rate": 0.10,
              "l2_leaf_reg": 3, "bagging_temperature": 1.0, "border_count": 128},
             CatBoostRegressor(iterations=200, depth=6, learning_rate=0.10,
                               l2_leaf_reg=3, bagging_temperature=1.0, border_count=128,
                               loss_function="RMSE", random_seed=RANDOM_STATE, verbose=0)),
            ("C2: iter=300,d=6,lr=0.05,l2=3",
             {"iterations": 300, "depth": 6, "learning_rate": 0.05,
              "l2_leaf_reg": 3, "bagging_temperature": 1.0, "border_count": 128},
             CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                               l2_leaf_reg=3, bagging_temperature=1.0, border_count=128,
                               loss_function="RMSE", random_seed=RANDOM_STATE, verbose=0)),
            ("C3: iter=500,d=8,lr=0.03,l2=5",
             {"iterations": 500, "depth": 8, "learning_rate": 0.03,
              "l2_leaf_reg": 5, "bagging_temperature": 1.0, "border_count": 128},
             CatBoostRegressor(iterations=500, depth=8, learning_rate=0.03,
                               l2_leaf_reg=5, bagging_temperature=1.0, border_count=128,
                               loss_function="RMSE", random_seed=RANDOM_STATE, verbose=0)),
            ("C4: iter=500,d=8,lr=0.03,l2=10",
             {"iterations": 500, "depth": 8, "learning_rate": 0.03,
              "l2_leaf_reg": 10, "bagging_temperature": 1.0, "border_count": 128},
             CatBoostRegressor(iterations=500, depth=8, learning_rate=0.03,
                               l2_leaf_reg=10, bagging_temperature=1.0, border_count=128,
                               loss_function="RMSE", random_seed=RANDOM_STATE, verbose=0)),
            ("C5: iter=500,d=8,lr=0.03,l2=5,bt=0.5",
             {"iterations": 500, "depth": 8, "learning_rate": 0.03,
              "l2_leaf_reg": 5, "bagging_temperature": 0.5, "border_count": 64},
             CatBoostRegressor(iterations=500, depth=8, learning_rate=0.03,
                               l2_leaf_reg=5, bagging_temperature=0.5, border_count=64,
                               loss_function="RMSE", random_seed=RANDOM_STATE, verbose=0)),
            ("C6: iter=700,d=6,lr=0.02,l2=5,bt=0.5",
             {"iterations": 700, "depth": 6, "learning_rate": 0.02,
              "l2_leaf_reg": 5, "bagging_temperature": 0.5, "border_count": 64},
             CatBoostRegressor(iterations=700, depth=6, learning_rate=0.02,
                               l2_leaf_reg=5, bagging_temperature=0.5, border_count=64,
                               loss_function="RMSE", random_seed=RANDOM_STATE, verbose=0)),
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Metrics helper
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 6),
        "R2":   round(float(r2_score(y_true, y_pred)), 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Save sweep results to JSON
# ══════════════════════════════════════════════════════════════════════════════

def _save_sweep_json(sweep_record: dict):
    """
    Saves all config results to outputs/model_sweep_results.json

    Structure:
    {
        "LinearRegression": {
            "configs": [
                {
                    "config_label": "C1: alpha=0.01",
                    "hyperparameters": {"alpha": 0.01},
                    "metrics": {"RMSE": 0.412, "MAE": 0.298, "R2": 0.631},
                    "is_best": false
                },
                ...
            ],
            "best_config": "C3: alpha=1.0",
            "best_metrics": {"RMSE": 0.410, "MAE": 0.295, "R2": 0.635}
        },
        ...
    }
    """
    json_out = {}

    for model_name, records in sweep_record.items():
        best_idx  = int(np.argmin([r["metrics"]["RMSE"] for r in records]))
        best_rec  = records[best_idx]

        configs_list = []
        for i, rec in enumerate(records):
            configs_list.append({
                "config_label":   rec["config_label"],
                "hyperparameters": rec["hyperparameters"],
                "metrics":         rec["metrics"],
                "is_best":         (i == best_idx),
            })

        json_out[model_name] = {
            "configs":      configs_list,
            "best_config":  best_rec["config_label"],
            "best_metrics": best_rec["metrics"],
        }

    json_path = OUTPUT_DIR / "model_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=4)
    logger.info(f"All config results saved → {json_path}")
    return json_out


# ══════════════════════════════════════════════════════════════════════════════
# Bar chart
# ══════════════════════════════════════════════════════════════════════════════

def _plot_comparison_bar(json_out: dict):
    models  = list(json_out.keys())
    metrics = ["RMSE", "MAE", "R2"]
    colors  = [MODEL_COLORS[m] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "Model Comparison — Best Configuration Metrics (Actual Values)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, metric in zip(axes, metrics):
        vals = [json_out[m]["best_metrics"][metric] for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="white",
                      width=0.5, linewidth=1.2)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.012,
                f"{val:.4f}",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#1A237E",
            )

        best_idx = int(np.argmin(vals)) if metric in ["RMSE", "MAE"] else int(np.argmax(vals))
        bars[best_idx].set_edgecolor("#E53935")
        bars[best_idx].set_linewidth(3.0)

        best_val = min(vals) if metric in ["RMSE", "MAE"] else max(vals)
        ax.axhline(best_val, color="#E53935", linestyle="--",
                   linewidth=1.5, alpha=0.6, label=f"Best: {best_val:.4f}")

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric, fontsize=10)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.legend(fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = PLOT_DIR / "model_comparison_bar.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter tables
# ══════════════════════════════════════════════════════════════════════════════

def _plot_hyperparameter_tables(json_out: dict):
    fig = plt.figure(figsize=(26, 22))
    fig.suptitle(
        "All Hyperparameter Configurations & Actual Metric Scores",
        fontsize=15, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.55, wspace=0.22,
        top=0.93, bottom=0.03,
        left=0.02, right=0.98,
    )

    header_color = "#1565C0"
    row_colors   = ["#EEF2FF", "#FFFFFF"]

    for idx, (model_name, data) in enumerate(json_out.items()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.axis("off")

        # Build flat DataFrame from all configs
        rows = []
        for cfg in data["configs"]:
            row = {"Config": cfg["config_label"]}
            row.update(cfg["hyperparameters"])
            row.update(cfg["metrics"])
            rows.append(row)
        df = pd.DataFrame(rows)

        # Round metric cols
        for col in ["RMSE", "MAE", "R2"]:
            if col in df.columns:
                df[col] = df[col].round(4)

        best_row_idx = int(df["RMSE"].idxmin())
        col_labels   = list(df.columns)
        cell_vals    = df.values.tolist()

        # Cell colors
        cell_colors = []
        for ri in range(len(df)):
            if ri == best_row_idx:
                cell_colors.append(["#FFF9C4"] * len(col_labels))
            else:
                cell_colors.append([row_colors[ri % 2]] * len(col_labels))

        tbl = ax.table(
            cellText=cell_vals,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            cellColours=cell_colors,
        )

        # Header styling
        for j in range(len(col_labels)):
            cell = tbl[(0, j)]
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)
            cell.set_height(0.13)

        # Row styling
        for ri in range(len(df)):
            for j in range(len(col_labels)):
                cell = tbl[(ri + 1, j)]
                cell.set_height(0.11)
                if ri == best_row_idx:
                    cell.set_edgecolor("#E53935")
                    cell.set_linewidth(1.8)
                    cell.set_text_props(fontweight="bold", fontsize=8.5)
                else:
                    cell.set_text_props(fontsize=8.0)

        tbl.auto_set_font_size(False)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))
        tbl.scale(1.1, 1.9)

        bm = data["best_metrics"]
        ax.set_title(
            f"{model_name}\n"
            f"★ Best: {data['best_config']}  —  "
            f"RMSE={bm['RMSE']:.4f}  |  "
            f"MAE={bm['MAE']:.4f}  |  "
            f"R²={bm['R2']:.4f}",
            fontsize=9.5, fontweight="bold",
            color=header_color, pad=14, loc="left",
        )

        ax.add_patch(plt.Rectangle(
            (-0.012, 0.0), 0.012, 1.0,
            transform=ax.transAxes,
            color=MODEL_COLORS[model_name],
            clip_on=False,
        ))

        ax.annotate(
            "★ Best config — yellow fill, red border",
            xy=(0.0, -0.04), xycoords="axes fraction",
            fontsize=7.5, color="#B71C1C", style="italic",
        )

    path = PLOT_DIR / "model_hyperparameter_tables.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(X_train, y_train, X_test, y_test) -> dict:
    """
    Train all configs for all models.
    Saves:
      - outputs/trained_models.pkl
      - outputs/model_sweep_results.json   ← ALL config results
      - outputs/eda/model_comparison_bar.png
      - outputs/eda/model_hyperparameter_tables.png

    Returns:
        {model_name: best_fitted_model}
    """
    configs      = get_model_configs()
    best_models  = {}
    sweep_record = {}

    for model_name, config_list in configs.items():
        logger.info(f"\n{'='*55}")
        logger.info(f"Training {model_name} — {len(config_list)} configs")
        sweep_record[model_name] = []
        best_rmse  = float("inf")
        best_model = None

        for cfg_label, params, model in config_list:
            logger.info(f"  Fitting {cfg_label} ...")
            model.fit(X_train, y_train)
            preds   = model.predict(X_test)
            metrics = _compute_metrics(y_test, preds)

            # Store full record
            sweep_record[model_name].append({
                "config_label":    cfg_label,
                "hyperparameters": {
                    k: (None if v is None else v)   # JSON-safe None
                    for k, v in params.items()
                },
                "metrics": metrics,
            })

            logger.info(
                f"    RMSE={metrics['RMSE']:.4f} | "
                f"MAE={metrics['MAE']:.4f} | "
                f"R²={metrics['R2']:.4f}"
            )

            if metrics["RMSE"] < best_rmse:
                best_rmse  = metrics["RMSE"]
                best_model = model

        best_models[model_name] = best_model
        logger.info(f"  ✓ Best RMSE for {model_name}: {best_rmse:.4f}")

    # ── Save all outputs ───────────────────────────────────────────────────────
    save_pickle(best_models, OUTPUT_DIR / "trained_models.pkl")

    json_out = _save_sweep_json(sweep_record)   # → model_sweep_results.json
    _plot_comparison_bar(json_out)              # → model_comparison_bar.png
    _plot_hyperparameter_tables(json_out)       # → model_hyperparameter_tables.png

    return best_models