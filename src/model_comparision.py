import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from src.utils import get_logger, OUTPUT_DIR

logger = get_logger("model_comparison")
PLOT_DIR = OUTPUT_DIR / "eda"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETER CONFIGS + METRICS
# Replace metric values with your actual outputs from metrics.json
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DATA = {

    "Linear Regression": {
        "hyperparams": pd.DataFrame([
            {"fit_intercept": True,  "normalize": False, "RMSE": 0.412, "MAE": 0.298, "R²": 0.631},
            {"fit_intercept": True,  "normalize": True,  "RMSE": 0.412, "MAE": 0.298, "R²": 0.631},
            {"fit_intercept": False, "normalize": False, "RMSE": 0.445, "MAE": 0.321, "R²": 0.598},
        ]),
        "best": {"RMSE": 0.412, "MAE": 0.298, "R²": 0.631},
        "color": "#90CAF9",
    },

    "Random Forest": {
        "hyperparams": pd.DataFrame([
            {"n_estimators": 50,  "max_depth": 10,  "min_samples_split": 5, "RMSE": 0.368, "MAE": 0.263, "R²": 0.701},
            {"n_estimators": 100, "max_depth": 15,  "min_samples_split": 2, "RMSE": 0.341, "MAE": 0.244, "R²": 0.742},
            {"n_estimators": 200, "max_depth": 20,  "min_samples_split": 2, "RMSE": 0.335, "MAE": 0.239, "R²": 0.751},
            {"n_estimators": 200, "max_depth": None,"min_samples_split": 2, "RMSE": 0.330, "MAE": 0.236, "R²": 0.759},
        ]),
        "best": {"RMSE": 0.330, "MAE": 0.236, "R²": 0.759},
        "color": "#A5D6A7",
    },

    "XGBoost": {
        "hyperparams": pd.DataFrame([
            {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.8, "RMSE": 0.321, "MAE": 0.229, "R²": 0.771},
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "RMSE": 0.308, "MAE": 0.219, "R²": 0.789},
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "RMSE": 0.301, "MAE": 0.214, "R²": 0.798},
            {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03, "subsample": 0.9, "RMSE": 0.297, "MAE": 0.211, "R²": 0.803},
        ]),
        "best": {"RMSE": 0.297, "MAE": 0.211, "R²": 0.803},
        "color": "#FFCC80",
    },

    "CatBoost": {
        "hyperparams": pd.DataFrame([
            {"iterations": 200, "depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 3,  "RMSE": 0.312, "MAE": 0.222, "R²": 0.781},
            {"iterations": 300, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3,  "RMSE": 0.299, "MAE": 0.213, "R²": 0.797},
            {"iterations": 500, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 5,  "RMSE": 0.291, "MAE": 0.207, "R²": 0.809},
            {"iterations": 500, "depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 10, "RMSE": 0.288, "MAE": 0.205, "R²": 0.813},
        ]),
        "best": {"RMSE": 0.288, "MAE": 0.205, "R²": 0.813},
        "color": "#CE93D8",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. BAR CHART — All Models on Best Config Metrics
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison_bar(model_data: dict, path=None):
    models  = list(model_data.keys())
    metrics = ["RMSE", "MAE", "R²"]
    colors  = [model_data[m]["color"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle(
        "Model Comparison — Best Configuration Metrics",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, metric in zip(axes, metrics):
        vals = [model_data[m]["best"][metric] for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="white",
                      width=0.5, linewidth=1.2)

        # Annotate values on top of bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#1A237E",
            )

        # Highlight best bar with red border
        best_idx = int(np.argmin(vals)) if metric in ["RMSE", "MAE"] else int(np.argmax(vals))
        bars[best_idx].set_edgecolor("#E53935")
        bars[best_idx].set_linewidth(3.0)

        # Best reference dashed line
        best_val = min(vals) if metric in ["RMSE", "MAE"] else max(vals)
        ax.axhline(best_val, color="#E53935", linestyle="--",
                   linewidth=1.5, alpha=0.6, label=f"Best: {best_val:.3f}")

        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylabel(metric, fontsize=10)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.set_ylim(0, max(vals) * 1.20)
        ax.legend(fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 2. HYPERPARAMETER + METRICS TABLES — One per Model (4 total)
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_hyperparameter_tables(model_data: dict, path=None):
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        "Model Hyperparameter Configurations & Corresponding Metrics",
        fontsize=15, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.6, wspace=0.25,
        top=0.93, bottom=0.03,
        left=0.03, right=0.97,
    )

    header_color = "#1565C0"
    row_colors   = ["#EEF2FF", "#FFFFFF"]

    for idx, (model_name, data) in enumerate(model_data.items()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.axis("off")

        df           = data["hyperparams"].copy()
        model_color  = data["color"]
        best_row_idx = int(df["RMSE"].idxmin())

        # Round metric columns
        df["RMSE"] = df["RMSE"].round(4)
        df["MAE"]  = df["MAE"].round(4)
        df["R²"]   = df["R²"].round(4)

        # Add Config label column
        df.insert(0, "Config", [f"C{i+1}" for i in range(len(df))])

        col_labels = list(df.columns)
        cell_vals  = df.values.tolist()

        # Build cell colors row by row
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

        # Header row styling
        for j in range(len(col_labels)):
            cell = tbl[(0, j)]
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
            cell.set_height(0.14)

        # Data row styling + best row red border
        for ri in range(len(df)):
            for j in range(len(col_labels)):
                cell = tbl[(ri + 1, j)]
                cell.set_height(0.12)
                if ri == best_row_idx:
                    cell.set_edgecolor("#E53935")
                    cell.set_linewidth(1.8)
                    cell.set_text_props(fontweight="bold", fontsize=9)
                else:
                    cell.set_text_props(fontsize=8.5)

        tbl.auto_set_font_size(False)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))
        tbl.scale(1.2, 2.0)

        # Title with best metrics summary
        ax.set_title(
            f"{model_name}\n"
            f"Best Config (★ C{best_row_idx+1}):  "
            f"RMSE = {data['best']['RMSE']:.4f}  |  "
            f"MAE = {data['best']['MAE']:.4f}  |  "
            f"R² = {data['best']['R²']:.4f}",
            fontsize=10, fontweight="bold",
            color=header_color, pad=16,
            loc="left",
        )

        # Color accent strip on left edge
        ax.add_patch(plt.Rectangle(
            (-0.015, 0.0), 0.015, 1.0,
            transform=ax.transAxes,
            color=model_color,
            clip_on=False,
        ))

        # Best row legend annotation
        ax.annotate(
            "★ Best Config (highlighted in yellow, red border)",
            xy=(0.0, -0.04),
            xycoords="axes fraction",
            fontsize=8, color="#B71C1C", style="italic",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_model_comparison(metrics_dict: dict = None):
    """
    Optionally pass metrics_dict loaded from metrics.json to override
    best metric values automatically.

    Expected format:
        {
            "Linear Regression": {"RMSE": x, "MAE": y, "R2": z},
            "Random Forest":     {"RMSE": x, "MAE": y, "R2": z},
            ...
        }
    """
    if metrics_dict:
        for model_name, vals in metrics_dict.items():
            if model_name in MODEL_DATA:
                MODEL_DATA[model_name]["best"]["RMSE"] = vals.get("RMSE", MODEL_DATA[model_name]["best"]["RMSE"])
                MODEL_DATA[model_name]["best"]["MAE"]  = vals.get("MAE",  MODEL_DATA[model_name]["best"]["MAE"])
                MODEL_DATA[model_name]["best"]["R²"]   = vals.get("R2",   MODEL_DATA[model_name]["best"]["R²"])
                logger.info(f"Loaded actual metrics for {model_name}: {vals}")

    plot_model_comparison_bar(
        MODEL_DATA,
        path=PLOT_DIR / "model_comparison_bar.png",
    )

    plot_model_hyperparameter_tables(
        MODEL_DATA,
        path=PLOT_DIR / "model_hyperparameter_tables.png",
    )

    logger.info("Model comparison plots complete.")


if __name__ == "__main__":
    run_model_comparison()