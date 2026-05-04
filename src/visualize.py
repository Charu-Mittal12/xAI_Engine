import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import get_logger, TARGET_COL

logger = get_logger("visualize")

# ─── Style ────────────────────────────────────────────────────────────────────
PALETTE = "Blues_r"
FIG_DPI = 120
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

def _savefig(fig, path=None):
    if path:
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    return fig

# ─── EDA plots ────────────────────────────────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[TARGET_COL], bins=80, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_title("Price Distribution")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Count")

    log_price = np.log1p(df[TARGET_COL])
    axes[1].hist(log_price, bins=80, color="#FF5722", edgecolor="white", alpha=0.85)
    axes[1].set_title("Log(1+Price) Distribution")
    axes[1].set_xlabel("log1p(Price)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Univariate Price Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _savefig(fig, path)

def plot_price_vs_categorical(df: pd.DataFrame, cat_col: str, path=None):
    order = df.groupby(cat_col)[TARGET_COL].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(data=df, x=cat_col, y=TARGET_COL, order=order, ax=ax,
                   palette="muted", inner="quartile")
    ax.set_title(f"Price by {cat_col}")
    ax.set_xlabel(cat_col)
    ax.set_ylabel("Price ($)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    return _savefig(fig, path)

def plot_price_vs_numerical(df: pd.DataFrame, num_col: str, path=None):
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample[num_col], sample[TARGET_COL], alpha=0.3, s=10, color="#1565C0")
    m, b = np.polyfit(sample[num_col].fillna(0), sample[TARGET_COL], 1)
    ax.plot(sorted(sample[num_col].fillna(0)), [m * x + b for x in sorted(sample[num_col].fillna(0))],
            "r--", linewidth=1.5, label="Trend")
    ax.set_title(f"Price vs {num_col}")
    ax.set_xlabel(num_col)
    ax.set_ylabel("Price ($)")
    ax.legend()
    plt.tight_layout()
    return _savefig(fig, path)

def plot_correlation_heatmap(df: pd.DataFrame, num_features: list, path=None):
    corr = df[num_features + [TARGET_COL]].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return _savefig(fig, path)

def plot_missing_values(df: pd.DataFrame, path=None):
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("No missing values found.")
        return None
    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.35)))
    missing.plot(kind="barh", ax=ax, color="#EF5350")
    ax.set_xlabel("Missing Fraction")
    ax.set_title("Missing Value Analysis", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    return _savefig(fig, path)

def plot_outlier_boxplots(df: pd.DataFrame, num_features: list, path=None):
    n_cols = 4
    n_rows = (len(num_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()
    for i, feat in enumerate(num_features):
        sns.boxplot(y=df[feat], ax=axes[i], color="#42A5F5", fliersize=2)
        axes[i].set_title(feat, fontsize=9)
        axes[i].set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Outlier Boxplots — Numerical Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _savefig(fig, path)

# ─── Experiment / training curves ─────────────────────────────────────────────
def plot_learning_curves(history: dict, model_name: str, path=None):
    cfg = history[model_name]
    param_vals = list(cfg.keys())
    train_losses = [cfg[v]["train_loss"] for v in param_vals]
    val_losses   = [cfg[v]["val_loss"]   for v in param_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_vals, train_losses, "o-", label="Train RMSE", color="#1976D2")
    ax.plot(param_vals, val_losses,   "s-", label="Val RMSE",   color="#E53935")
    ax.set_title(f"{model_name} — Training vs Validation Loss", fontsize=12)
    ax.set_xlabel("Parameter Value")
    ax.set_ylabel("RMSE (log-price space)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return _savefig(fig, path)

# ─── Evaluation ───────────────────────────────────────────────────────────────
def plot_model_performance_table(metrics_df, path=None):
    fig, ax = plt.subplots(figsize=(9, len(metrics_df) * 0.8 + 1.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=metrics_df.round(4).values,
        colLabels=metrics_df.columns,
        rowLabels=metrics_df.index,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.6)
    ax.set_title("Model Evaluation — Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return _savefig(fig, path)

# ─── LIME plots ───────────────────────────────────────────────────────────────
def plot_lime_explanation(lime_contrib: dict, model_name: str, top_k: int = 12, path=None):
    sorted_items = sorted(lime_contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    features, values = zip(*sorted_items) if sorted_items else ([], [])
    colors = ["#1565C0" if v > 0 else "#C62828" for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(features))
    ax.barh(y_pos, values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"LIME Explanation — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Contribution")
    ax.invert_yaxis()
    plt.tight_layout()
    return _savefig(fig, path)

# ─── SHAP plots ───────────────────────────────────────────────────────────────
def plot_shap_waterfall(shap_contrib: dict, model_name: str, top_k: int = 12, path=None):
    sorted_items = sorted(shap_contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    features, values = zip(*sorted_items) if sorted_items else ([], [])
    colors = ["#1B5E20" if v > 0 else "#B71C1C" for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(features))
    ax.barh(y_pos, values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"SHAP Waterfall — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("SHAP Value (impact on log-price)")
    ax.invert_yaxis()
    plt.tight_layout()
    return _savefig(fig, path)

# ─── Comparison ───────────────────────────────────────────────────────────────
def plot_lime_shap_comparison(lime_contrib: dict, shap_contrib: dict, model_name: str, top_k: int = 10, path=None):
    lime_top = dict(sorted(lime_contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k])
    shap_top = dict(sorted(shap_contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k])
    all_feats = list(dict.fromkeys(list(lime_top) + list(shap_top)))

    lime_vals = [lime_top.get(f, 0) for f in all_feats]
    shap_vals = [shap_top.get(f, 0) for f in all_feats]

    x = np.arange(len(all_feats))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, lime_vals, width, label="LIME", color="#1976D2", alpha=0.85)
    ax.bar(x + width / 2, shap_vals, width, label="SHAP", color="#388E3C", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(all_feats, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"LIME vs SHAP Feature Contributions — {model_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Contribution")
    ax.legend()
    plt.tight_layout()
    return _savefig(fig, path)