"""
EDA module — Airbnb Price Prediction
Includes: standard EDA, description word cloud, name sentiment analysis,
          and price/log-price distribution plots (merged from plot_price_distributions.py)
Changes:
  - Price bucket: bar chart only (no pie)
  - Geospatial heatmap: visually enhanced with colorbar, city annotations, grid
  - Discarded features plot: REMOVED
  - Violin plots → Box plots everywhere
  - Top-5 keyword price buckets: mean, median, mode annotations added
  - THEME: All plots use white background, blue/purple palette,
           red/green/blue highlights for important annotations
  - plot_price_distributions merged directly into run_standard_eda
"""
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy.stats import gaussian_kde

from src.utils import (
    load_raw_data, get_feature_lists,
    TARGET_COL, OUTPUT_DIR, get_logger,
)
from src import visualize as viz

warnings.filterwarnings("ignore")
logger = get_logger("eda")
EDA_DIR = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# ── Global light theme ────────────────────────────────────────────────────────
BG_COLOR    = "#FFFFFF"
PANEL_COLOR = "#F5F5FB"
GRID_COLOR  = "#DCDCEE"
TEXT_COLOR  = "#1A1A2E"
SPINE_COLOR = "#AAAACC"

PRIMARY_PURPLE  = "#6A3DB8"
PRIMARY_BLUE    = "#2B7FD4"
MID_VIOLET      = "#8A56C8"
LIGHT_BLUE      = "#5BA4E0"
DEEP_INDIGO     = "#3A2A7A"

HL_RED   = "#C0392B"
HL_GREEN = "#1E7B4B"
HL_BLUE  = "#1A5FA8"


def _bar_palette(n: int):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "bp", [DEEP_INDIGO, PRIMARY_PURPLE, PRIMARY_BLUE, LIGHT_BLUE], N=256
    )
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _apply_light_theme():
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    PANEL_COLOR,
        "axes.edgecolor":    SPINE_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "axes.grid":         True,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.6,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "legend.facecolor":  "#EEEEF8",
        "legend.edgecolor":  SPINE_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "savefig.facecolor": BG_COLOR,
        "savefig.edgecolor": BG_COLOR,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
    })


_apply_light_theme()

sns.set_theme(style="whitegrid", font_scale=1.05, rc={
    "axes.facecolor":   PANEL_COLOR,
    "figure.facecolor": BG_COLOR,
    "grid.color":       GRID_COLOR,
    "axes.edgecolor":   SPINE_COLOR,
    "text.color":       TEXT_COLOR,
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
})

STOP_WORDS = {
    "the","a","an","and","or","in","of","to","is","with","for","at",
    "on","by","this","that","are","be","it","as","from","you","your",
    "our","we","has","have","will","can","was","but","not","all",
    "more","very","also","s","room","apartment","airbnb","home","house",
}


# ══════════════════════════════════════════════════════════════════════════════
# MERGED: Price & Log-Price Distribution (from plot_price_distributions.py)
# ══════════════════════════════════════════════════════════════════════════════

def _kde_overlay(ax, data, color, alpha=0.18):
    """Draw a KDE curve and fill below it on a twin axis."""
    kde = gaussian_kde(data, bw_method="scott")
    xs  = np.linspace(data.min(), data.max(), 500)
    ys  = kde(xs)
    ax2 = ax.twinx()
    ax2.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9, zorder=5)
    ax2.fill_between(xs, ys, alpha=alpha, color=color, zorder=4)
    ax2.set_ylabel("Density", color=SPINE_COLOR, fontsize=9)
    ax2.tick_params(axis="y", colors=SPINE_COLOR, labelsize=8)
    ax2.spines[:].set_edgecolor(SPINE_COLOR)
    return ax2


def _stat_lines(ax, data, y_max, unit=""):
    """Draw mean, median, ±1σ lines with annotations."""
    from scipy.stats import skew, kurtosis
    mean   = data.mean()
    median = data.median()
    std    = data.std()

    ax.axvline(mean,        color=HL_RED,    linestyle="--", linewidth=2.0,
               label=f"Mean: {mean:.2f}{unit}", zorder=6)
    ax.axvline(median,      color=HL_GREEN,  linestyle="-.", linewidth=2.0,
               label=f"Median: {median:.2f}{unit}", zorder=6)
    ax.axvline(mean - std,  color=LIGHT_BLUE, linestyle=":",  linewidth=1.4,
               alpha=0.8, label=f"−1σ: {mean-std:.2f}{unit}", zorder=5)
    ax.axvline(mean + std,  color=LIGHT_BLUE, linestyle=":",  linewidth=1.4,
               alpha=0.8, label=f"+1σ: {mean+std:.2f}{unit}", zorder=5)

    sk  = skew(data)
    ku  = kurtosis(data)
    txt = f"Skew: {sk:+.3f}\nKurt: {ku:+.3f}\nn: {len(data):,}"
    ax.text(0.97, 0.97, txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", fc="#EEF0FF",
                      ec=HL_BLUE, alpha=0.9))


def plot_price_and_logprice_distributions(df: pd.DataFrame, path=None):
    """
    Two-panel plot:
      Left  — log_price distribution (raw TARGET_COL)
      Right — actual price distribution (exp(log_price))
    Merged from the standalone plot_price_distributions.py script.
    """
    log_price  = df[TARGET_COL].dropna().astype(float)
    real_price = np.exp(log_price)

    fig, (ax_log, ax_real) = plt.subplots(
        1, 2, figsize=(16, 6),
        facecolor=BG_COLOR,
        gridspec_kw={"wspace": 0.38},
    )

    # ── Left: log_price ──────────────────────────────────────────────────────
    ax_log.set_facecolor(PANEL_COLOR)
    ax_log.spines[:].set_edgecolor(SPINE_COLOR)

    counts_log, _, _ = ax_log.hist(
        log_price, bins=60,
        color=PRIMARY_PURPLE, edgecolor="white",
        linewidth=0.4, alpha=0.85,
        label="log_price count",
    )
    _kde_overlay(ax_log, log_price, color=HL_BLUE)
    _stat_lines(ax_log, log_price, y_max=counts_log.max())

    p5, p95 = np.percentile(log_price, [5, 95])
    ax_log.axvspan(p5, p95, alpha=0.07, color=PRIMARY_BLUE, label="5th–95th pct")

    ax_log.set_title("Log-Price Distribution  (log_price)",
                     fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax_log.set_xlabel("log_price  (natural log of $)", fontsize=11, color=TEXT_COLOR)
    ax_log.set_ylabel("Number of Listings",            fontsize=11, color=TEXT_COLOR)
    ax_log.tick_params(colors=TEXT_COLOR)
    ax_log.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_log.legend(fontsize=9, loc="upper left")

    # ── Right: actual price ──────────────────────────────────────────────────
    ax_real.set_facecolor(PANEL_COLOR)
    ax_real.spines[:].set_edgecolor(SPINE_COLOR)

    clip_p99      = real_price.quantile(0.99)
    price_clipped = real_price.clip(upper=clip_p99)

    counts_r, _, _ = ax_real.hist(
        price_clipped, bins=60,
        color=PRIMARY_BLUE, edgecolor="white",
        linewidth=0.4, alpha=0.85,
        label="Actual price count",
    )
    _kde_overlay(ax_real, price_clipped, color=HL_RED)
    _stat_lines(ax_real, price_clipped, y_max=counts_r.max(), unit="$")

    true_max = real_price.max()
    ax_real.text(
        0.97, 0.60,
        f"True max: ${true_max:,.0f}\n(clipped at 99th pct\nfor display)",
        transform=ax_real.transAxes, ha="right", va="top",
        fontsize=8.5, color=HL_RED,
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF0F0", ec=HL_RED, alpha=0.9),
    )
    ax_real.set_title("Actual Price Distribution  (exp(log_price) = $)",
                      fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax_real.set_xlabel("Price ($)",           fontsize=11, color=TEXT_COLOR)
    ax_real.set_ylabel("Number of Listings",  fontsize=11, color=TEXT_COLOR)
    ax_real.tick_params(colors=TEXT_COLOR)
    ax_real.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_real.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_real.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Airbnb Price Analysis — Log-Scale vs Actual Dollar Distribution",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    fig.text(
        0.5, -0.03,
        "Left: log_price stored in dataset (approx. normal).   "
        "Right: exp(log_price) = actual nightly price $ (right-skewed, long tail).",
        ha="center", va="top", fontsize=9, color=SPINE_COLOR, style="italic",
    )

    out_path = path or (EDA_DIR / "price_and_logprice_distributions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Standard EDA
# ══════════════════════════════════════════════════════════════════════════════

def run_standard_eda(df: pd.DataFrame, num_feats: list, cat_feats: list):
    # ── Merged price distribution plot ───────────────────────────────────────
    plot_price_and_logprice_distributions(
        df, path=EDA_DIR / "price_and_logprice_distributions.png"
    )

    viz.plot_price_distribution(df, path=EDA_DIR / "price_distribution.png")
    viz.plot_correlation_heatmap(df, num_feats, path=EDA_DIR / "correlation_heatmap.png")
    viz.plot_missing_values(df, path=EDA_DIR / "missing_values.png")
    viz.plot_outlier_boxplots(df, num_feats, path=EDA_DIR / "outlier_boxplots.png")

    for cat in cat_feats[:4]:
        safe = cat.replace("/", "_")
        _plot_price_vs_cat_boxplot(df, cat, path=EDA_DIR / f"price_vs_{safe}.png")

    for num in ["accommodates", "bedrooms", "review_scores_rating"]:
        if num in num_feats:
            viz.plot_price_vs_numerical(df, num, path=EDA_DIR / f"price_vs_{num}.png")


def _plot_price_vs_cat_boxplot(df: pd.DataFrame, cat_col: str, path=None):
    order = (
        df.groupby(cat_col)[TARGET_COL]
        .median()
        .sort_values(ascending=False)
        .index
    )
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    palette = {cat: c for cat, c in zip(order, _bar_palette(len(order)))}
    sns.boxplot(
        data=df, x=cat_col, y=TARGET_COL,
        order=order, ax=ax,
        palette=palette,
        flierprops=dict(marker="o", markersize=2, alpha=0.4,
                        markerfacecolor=PRIMARY_PURPLE, markeredgecolor="none"),
        boxprops=dict(edgecolor=PRIMARY_BLUE),
        whiskerprops=dict(color=PRIMARY_BLUE),
        capprops=dict(color=LIGHT_BLUE),
        medianprops=dict(color=HL_GREEN, linewidth=2),
        width=0.5,
    )
    for i, cat_val in enumerate(order):
        med = df[df[cat_col] == cat_val][TARGET_COL].median()
        ax.text(i, med + 0.02, f"{med:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=HL_BLUE, fontweight="bold")

    ax.set_title(f"Price Distribution by {cat_col}",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel(cat_col, color=TEXT_COLOR)
    ax.set_ylabel("log_price", color=TEXT_COLOR)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", color=TEXT_COLOR)
    ax.spines[:].set_edgecolor(SPINE_COLOR)
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Advanced EDA (7 Points)
# ══════════════════════════════════════════════════════════════════════════════

def run_advanced_eda(df: pd.DataFrame, num_feats: list, cat_feats: list):
    logger.info("Running Advanced Future Work EDA — 7 points ...")
    _plot_price_buckets(df)
    if "latitude" in df.columns and "longitude" in df.columns:
        _plot_geo_price(df)
    if "host_since" in df.columns:
        _plot_host_experience_vs_price(df)
    if "amenities" in df.columns:
        _plot_amenity_count_vs_price(df)
    if "cancellation_policy" in df.columns:
        _plot_cancellation_vs_price(df)
    if "availability_365" in df.columns:
        _plot_availability_distribution(df)
    if "room_type" in df.columns and "city" in df.columns:
        _plot_room_city_interaction(df)
    logger.info("Advanced EDA (7 points) complete.")


def _plot_price_buckets(df: pd.DataFrame):
    price  = np.exp(df[TARGET_COL]) if df[TARGET_COL].max() < 20 else df[TARGET_COL]
    bins   = [0, 50, 100, 150, 200, 300, 500, 1000, float("inf")]
    labels = ["<$50","$50-100","$100-150","$150-200",
              "$200-300","$300-500","$500-1k",">$1k"]

    df2 = df.copy()
    df2["price_bucket"] = pd.cut(price, bins=bins, labels=labels)
    counts = df2["price_bucket"].value_counts().reindex(labels)
    pct    = counts / counts.sum() * 100

    palette = _bar_palette(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    bars = ax.bar(labels, counts.values, color=palette, edgecolor=SPINE_COLOR, linewidth=0.8)

    for bar, cnt, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.012,
                f"{cnt:,}\n({p:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=HL_BLUE)

    mean_price   = price.mean()
    median_price = price.median()
    ax.axvline(x=_price_to_bucket_x(mean_price,   bins, labels),
               color=HL_RED,   linestyle="--", linewidth=2.2,
               label=f"Mean: ${mean_price:.0f}")
    ax.axvline(x=_price_to_bucket_x(median_price, bins, labels),
               color=HL_GREEN, linestyle="-.", linewidth=2.2,
               label=f"Median: ${median_price:.0f}")

    ax.set_title("① Price Bucket Distribution — Listing Count",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Price Range ($)", fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel("Number of Listings", fontsize=11, color=TEXT_COLOR)
    ax.tick_params(axis="x", rotation=20, colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[:].set_edgecolor(SPINE_COLOR)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "adv_01_price_buckets.png", dpi=120,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info("Saved: adv_01_price_buckets.png")


def _price_to_bucket_x(price_val, bins, labels):
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if lo <= price_val < hi:
            return i
    return len(labels) - 1


def _plot_geo_price(df: pd.DataFrame):
    sample = df[["latitude", "longitude", TARGET_COL]].dropna().sample(
        min(8000, len(df)), random_state=42
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG_COLOR)
    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)

    sc = axes[0].scatter(
        sample["longitude"], sample["latitude"],
        c=sample[TARGET_COL], cmap="RdPu",
        s=6, alpha=0.75, linewidths=0,
        vmin=sample[TARGET_COL].quantile(0.05),
        vmax=sample[TARGET_COL].quantile(0.95),
    )
    cb = fig.colorbar(sc, ax=axes[0], pad=0.02, shrink=0.85)
    cb.set_label("log_price", color=TEXT_COLOR, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    axes[0].set_title("Listing Price by Location", color=TEXT_COLOR,
                       fontsize=12, fontweight="bold", pad=10)
    axes[0].set_xlabel("Longitude", color=TEXT_COLOR, fontsize=10)
    axes[0].set_ylabel("Latitude",  color=TEXT_COLOR, fontsize=10)
    axes[0].tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in axes[0].spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    if "city" in df.columns:
        city_centers = (
            df.dropna(subset=["latitude","longitude","city"])
            .groupby("city")[["latitude","longitude"]]
            .mean()
        )
        for city, row in city_centers.iterrows():
            axes[0].annotate(
                city,
                xy=(row["longitude"], row["latitude"]),
                fontsize=8, color=HL_BLUE, fontweight="bold",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#EEF0FF", ec=PRIMARY_BLUE, alpha=0.85),
            )

    bp_cmap = mcolors.LinearSegmentedColormap.from_list(
        "bp_hex", [PANEL_COLOR, MID_VIOLET, PRIMARY_BLUE, LIGHT_BLUE], N=256
    )
    hb = axes[1].hexbin(
        sample["longitude"], sample["latitude"],
        C=sample[TARGET_COL], gridsize=40, cmap=bp_cmap,
        reduce_C_function=np.mean, linewidths=0.05,
    )
    cb2 = fig.colorbar(hb, ax=axes[1], pad=0.02, shrink=0.85)
    cb2.set_label("Mean log_price", color=TEXT_COLOR, fontsize=10)
    cb2.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    axes[1].set_title("Hexbin Mean Price Density", color=TEXT_COLOR,
                       fontsize=12, fontweight="bold", pad=10)
    axes[1].set_xlabel("Longitude", color=TEXT_COLOR, fontsize=10)
    axes[1].set_ylabel("Latitude",  color=TEXT_COLOR, fontsize=10)
    axes[1].tick_params(colors=TEXT_COLOR, labelsize=8)
    for spine in axes[1].spines.values():
        spine.set_edgecolor(SPINE_COLOR)

    fig.suptitle("② Geospatial Price Heatmap", color=TEXT_COLOR,
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "adv_02_geo_price.png", dpi=150,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info("Saved: adv_02_geo_price.png")


def _plot_host_experience_vs_price(df: pd.DataFrame):
    df2 = df[["host_since", TARGET_COL]].dropna().copy()
    df2["host_since"] = pd.to_datetime(df2["host_since"], errors="coerce")
    df2["years_active"] = (pd.Timestamp("2024-01-01") - df2["host_since"]).dt.days / 365
    df2 = df2[df2["years_active"].between(0, 20)]
    df2["experience_bin"] = pd.cut(
        df2["years_active"],
        bins=[0,1,2,3,5,8,12,20],
        labels=["<1yr","1-2yr","2-3yr","3-5yr","5-8yr","8-12yr","12+yr"]
    )
    agg = df2.groupby("experience_bin", observed=True)[TARGET_COL].mean()
    n = len(agg)
    palette = _bar_palette(n)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    agg.plot(kind="bar", ax=ax, color=palette, edgecolor=SPINE_COLOR)

    max_idx = agg.values.argmax()
    ax.patches[max_idx].set_facecolor(HL_RED)
    ax.patches[max_idx].set_edgecolor(HL_RED)
    ax.text(max_idx, agg.values[max_idx] + 0.002,
            f"Peak\n{agg.values[max_idx]:.2f}",
            ha="center", va="bottom", color=HL_RED, fontsize=8, fontweight="bold")

    ax.set_title("③ Host Experience vs Average log_price",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Years Active as Host", color=TEXT_COLOR)
    ax.set_ylabel("Mean log_price", color=TEXT_COLOR)
    ax.tick_params(axis="x", rotation=30, colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)
    ax.spines[:].set_edgecolor(SPINE_COLOR)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "adv_03_host_experience.png", dpi=120,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def _plot_amenity_count_vs_price(df: pd.DataFrame):
    df2 = df[["amenities", TARGET_COL]].dropna().copy()
    df2["amenity_count"] = df2["amenities"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) else 0
    )
    df2 = df2[df2["amenity_count"] > 0]
    df2["amenity_bin"] = pd.cut(
        df2["amenity_count"],
        bins=[0,5,10,15,20,30,50,100],
        labels=["1-5","6-10","11-15","16-20","21-30","31-50","50+"]
    )
    agg = df2.groupby("amenity_bin", observed=True)[TARGET_COL].mean()
    palette = _bar_palette(len(agg))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    agg.plot(kind="bar", ax=ax, color=palette, edgecolor=SPINE_COLOR)

    max_idx = agg.values.argmax()
    ax.patches[max_idx].set_facecolor(HL_GREEN)
    ax.patches[max_idx].set_edgecolor(HL_GREEN)
    ax.text(max_idx, agg.values[max_idx] + 0.002,
            f"Max\n{agg.values[max_idx]:.2f}",
            ha="center", va="bottom", color=HL_GREEN, fontsize=8, fontweight="bold")

    ax.set_title("④ Amenity Count vs Average log_price",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Number of Amenities", color=TEXT_COLOR)
    ax.set_ylabel("Mean log_price", color=TEXT_COLOR)
    ax.tick_params(axis="x", rotation=30, colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)
    ax.spines[:].set_edgecolor(SPINE_COLOR)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "adv_04_amenity_count.png", dpi=120,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def _plot_cancellation_vs_price(df: pd.DataFrame):
    order = (
        df.groupby("cancellation_policy")[TARGET_COL]
        .median()
        .sort_values(ascending=False)
        .index
    )
    palette = {cat: c for cat, c in zip(order, _bar_palette(len(order)))}

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    sns.boxplot(
        data=df, x="cancellation_policy", y=TARGET_COL,
        order=order, ax=ax, palette=palette,
        flierprops=dict(marker="o", markersize=2, alpha=0.35,
                        markerfacecolor=PRIMARY_PURPLE, markeredgecolor="none"),
        boxprops=dict(edgecolor=PRIMARY_BLUE),
        whiskerprops=dict(color=PRIMARY_BLUE),
        capprops=dict(color=LIGHT_BLUE),
        medianprops=dict(color=HL_GREEN, linewidth=2.2),
        width=0.5,
    )
    for i, policy in enumerate(order):
        med = df[df["cancellation_policy"] == policy][TARGET_COL].median()
        ax.text(i, med + 0.01, f"{med:.2f}", ha="center", va="bottom",
                fontsize=8, color=HL_BLUE, fontweight="bold")

    ax.set_title("⑤ Cancellation Policy vs Price Distribution",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Cancellation Policy", color=TEXT_COLOR)
    ax.set_ylabel("log_price", color=TEXT_COLOR)
    ax.tick_params(axis="x", rotation=20, colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)
    ax.spines[:].set_edgecolor(SPINE_COLOR)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "adv_05_cancellation_price.png", dpi=120,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def _plot_availability_distribution(df: pd.DataFrame):
    avail = df["availability_365"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_COLOR)
    for ax in axes:
        ax.set_facecolor(PANEL_COLOR)
        ax.spines[:].set_edgecolor(SPINE_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

    axes[0].hist(avail, bins=50, color=PRIMARY_PURPLE, edgecolor=SPINE_COLOR, alpha=0.85)
    axes[0].set_title("Availability (days/year)", color=TEXT_COLOR, fontweight="bold")
    axes[0].set_xlabel("Days Available", color=TEXT_COLOR)
    axes[0].set_ylabel("Count", color=TEXT_COLOR)
    mean_avail = avail.mean()
    axes[0].axvline(mean_avail, color=HL_RED, linestyle="--", linewidth=2,
                    label=f"Mean: {mean_avail:.0f}d")
    axes[0].legend(fontsize=9)

    corr_df = df[["availability_365", TARGET_COL]].dropna().sample(
        min(3000, len(df)), random_state=42
    )
    axes[1].scatter(corr_df["availability_365"], corr_df[TARGET_COL],
                    alpha=0.25, s=8, color=PRIMARY_BLUE)
    axes[1].set_title("⑥ Availability vs log_price", color=TEXT_COLOR, fontweight="bold")
    axes[1].set_xlabel("Availability (days)", color=TEXT_COLOR)
    axes[1].set_ylabel("log_price", color=TEXT_COLOR)

    fig.suptitle("⑥ Booking Availability Analysis",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "adv_06_availability.png", dpi=120,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def _plot_room_city_interaction(df: pd.DataFrame):
    pivot = df.groupby(["city", "room_type"])[TARGET_COL].mean().unstack()
    n_series = len(pivot.columns)
    series_colors = _bar_palette(n_series)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    pivot.plot(kind="bar", ax=ax, edgecolor=SPINE_COLOR, color=series_colors)

    all_vals = pivot.values.flatten()
    global_max = np.nanmax(all_vals)
    for patch in ax.patches:
        if abs(patch.get_height() - global_max) < 1e-6:
            patch.set_facecolor(HL_RED)
            patch.set_edgecolor(HL_RED)
            ax.text(patch.get_x() + patch.get_width() / 2, global_max + 0.01,
                    f"Max\n{global_max:.2f}", ha="center", va="bottom",
                    color=HL_RED, fontsize=7.5, fontweight="bold")

    ax.set_title("⑦ Room Type × City — Mean log_price",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("City", color=TEXT_COLOR)
    ax.set_ylabel("Mean log_price", color=TEXT_COLOR)
    ax.tick_params(axis="x", rotation=35, colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)
    ax.legend(title="Room Type", bbox_to_anchor=(1.01, 1), loc="upper left",
              title_fontsize=9, labelcolor=TEXT_COLOR,
              facecolor="#EEEEF8", edgecolor=SPINE_COLOR)
    ax.spines[:].set_edgecolor(SPINE_COLOR)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "adv_07_room_city_interaction.png", dpi=120,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Description: Word Cloud + Top-5 Words + Price Buckets
# ══════════════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> list:
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    return [w for w in text.split() if w not in STOP_WORDS and len(w) > 3]


def run_description_analysis(df: pd.DataFrame):
    if "description" not in df.columns:
        logger.warning("No 'description' column — skipping.")
        return

    logger.info("Running description word cloud + top-5 price analysis ...")
    df2 = df[["description", TARGET_COL]].dropna().copy()
    df2["tokens"] = df2["description"].apply(_clean_text)

    all_words = [w for tokens in df2["tokens"] for w in tokens]
    word_freq = Counter(all_words)
    top_words = [w for w, _ in word_freq.most_common(5)]

    try:
        from wordcloud import WordCloud

        def _bp_color_func(word, font_size, position, orientation,
                           random_state=None, **kwargs):
            options = [PRIMARY_PURPLE, MID_VIOLET, PRIMARY_BLUE, LIGHT_BLUE, DEEP_INDIGO]
            import random
            return random.choice(options)

        wc = WordCloud(
            width=1200, height=600,
            background_color=BG_COLOR,
            color_func=_bp_color_func,
            max_words=150,
            stopwords=STOP_WORDS,
            collocations=False,
        ).generate(" ".join(all_words))

        fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Description Word Cloud — Most Frequent Words",
                     fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=15)
        fig.savefig(EDA_DIR / "desc_wordcloud.png", dpi=150,
                    bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        logger.info("Saved: desc_wordcloud.png")
    except ImportError:
        logger.warning("Run: pip install wordcloud")

    top20 = word_freq.most_common(20)
    words20, freqs20 = zip(*top20)
    bar_colors = [HL_RED if w in top_words else PRIMARY_BLUE for w in words20]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)
    ax.bar(words20, freqs20, color=bar_colors, edgecolor=SPINE_COLOR)
    ax.set_title("Top-20 Words in Listing Descriptions  (Red = Top-5)",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylabel("Frequency", color=TEXT_COLOR)
    ax.tick_params(axis="x", rotation=40, colors=TEXT_COLOR)
    ax.tick_params(axis="y", colors=TEXT_COLOR)
    ax.spines[:].set_edgecolor(SPINE_COLOR)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "desc_top_words.png", dpi=120,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

    price_vals = np.exp(df2[TARGET_COL]) if df2[TARGET_COL].max() < 20 else df2[TARGET_COL]
    bins         = [0, 50, 100, 150, 200, 300, 500, float("inf")]
    bucket_labels = ["<$50","$50-100","$100-150","$150-200","$200-300","$300-500",">$500"]
    df2["price_bucket"] = pd.cut(price_vals, bins=bins, labels=bucket_labels)

    n = len(top_words)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 6), sharey=False, facecolor=BG_COLOR)
    if n == 1:
        axes = [axes]

    bucket_palette = _bar_palette(len(bucket_labels))

    for ax, word in zip(axes, top_words):
        ax.set_facecolor(PANEL_COLOR)
        ax.spines[:].set_edgecolor(SPINE_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

        mask   = df2["tokens"].apply(lambda t: word in t)
        subset = df2[mask].copy()
        counts = subset["price_bucket"].value_counts().reindex(bucket_labels).fillna(0)

        bars = ax.bar(bucket_labels, counts.values,
                      color=bucket_palette, edgecolor=SPINE_COLOR)

        word_prices = price_vals[mask]
        mean_p   = word_prices.mean()
        median_p = word_prices.median()
        mode_bucket = subset["price_bucket"].mode()
        mode_label  = mode_bucket.iloc[0] if not mode_bucket.empty else "N/A"

        mean_x   = _price_to_bucket_x(mean_p,   bins, bucket_labels)
        median_x = _price_to_bucket_x(median_p, bins, bucket_labels)

        ax.axvline(mean_x,   color=HL_RED,   linestyle="--", linewidth=2,
                   label=f"Mean ${mean_p:.0f}")
        ax.axvline(median_x, color=HL_GREEN, linestyle="-.",  linewidth=2,
                   label=f"Median ${median_p:.0f}")
        ax.text(0.97, 0.97, f"Mode bucket:\n{mode_label}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=HL_BLUE,
                bbox=dict(boxstyle="round,pad=0.3", fc="#EEF0FF", ec=HL_BLUE, alpha=0.9))

        for bar, cnt in zip(bars, counts.values):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + counts.max() * 0.02,
                        f"{int(cnt):,}", ha="center", fontsize=7.5, color=PRIMARY_BLUE)

        ax.set_title(f'"{word}"', fontweight="bold", fontsize=11, color=TEXT_COLOR)
        ax.set_xlabel("Price Bucket ($)", fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel("Count", fontsize=9, color=TEXT_COLOR)
        ax.tick_params(axis="x", rotation=40, colors=TEXT_COLOR)
        ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(
        "Price Bucket Distribution for Top-5 Description Keywords\n"
        "(with Mean [RED], Median [GREEN], and Modal Bucket [BLUE])",
        fontsize=13, fontweight="bold", color=TEXT_COLOR,
    )
    fig.patch.set_facecolor(BG_COLOR)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "desc_top5_price_buckets.png", dpi=130,
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info("Saved: desc_top5_price_buckets.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Name sentiment analysis (VADER + TF-IDF + Logistic Regression)
# ══════════════════════════════════════════════════════════════════════════════

def run_name_sentiment_analysis(df: pd.DataFrame):
    if "name" not in df.columns:
        logger.warning("No 'name' column — skipping.")
        return

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        logger.warning("Run: pip install vaderSentiment")
        return

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline

    logger.info("Running VADER-based name sentiment analysis ...")

    df2 = df[["name", TARGET_COL]].dropna().copy()
    df2["name_clean"] = df2["name"].astype(str).str.lower().str.strip()

    analyzer = SentimentIntensityAnalyzer()

    def get_vader_label(text):
        score = analyzer.polarity_scores(str(text))["compound"]
        if score >= 0.05:   return "Positive"
        elif score <= -0.05: return "Negative"
        else:                return "Neutral"

    def get_vader_score(text):
        return analyzer.polarity_scores(str(text))["compound"]

    df2["vader_label"] = df2["name_clean"].apply(get_vader_label)
    df2["vader_score"] = df2["name_clean"].apply(get_vader_score)
    logger.info(f"VADER label distribution:\n{df2['vader_label'].value_counts().to_string()}")

    price_vals = np.exp(df2[TARGET_COL]) if df2[TARGET_COL].max() < 20 else df2[TARGET_COL]
    df2["price"] = price_vals

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = {"Positive": "#1B5E20", "Neutral": "#1565C0", "Negative": "#B71C1C"}

    for ax, label in zip(axes[:3], ["Positive", "Neutral", "Negative"]):
        group   = df2[df2["vader_label"] == label]["price"]
        clipped = group.clip(upper=group.quantile(0.99))
        ax.hist(clipped, bins=60, color=colors[label], edgecolor="white", alpha=0.85)
        ax.axvline(group.mean(),   color="white",  linestyle="--", linewidth=2,
                   label=f"Mean: ${group.mean():.0f}")
        ax.axvline(group.median(), color="yellow", linestyle="-.", linewidth=2,
                   label=f"Median: ${group.median():.0f}")
        ax.set_title(f"{label} Names\n(n={len(group):,})", fontweight="bold", color=colors[label])
        ax.set_xlabel("Price ($)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[3]
    for label in ["Positive", "Neutral", "Negative"]:
        group   = df2[df2["vader_label"] == label]["price"]
        clipped = group.clip(upper=group.quantile(0.99))
        if len(clipped) > 10:
            kde = gaussian_kde(clipped)
            xs  = np.linspace(clipped.min(), clipped.max(), 400)
            ax.plot(xs, kde(xs), color=colors[label], linewidth=2.5, label=label)
            ax.fill_between(xs, kde(xs), alpha=0.12, color=colors[label])
    ax.set_title("KDE Overlay\nAll Sentiment Groups", fontweight="bold")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Price Distribution by Emotional Sentiment of Listing Name (VADER)\n"
        "Sentiment assigned first from language — price trend observed after",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(EDA_DIR / "name_sentiment_price_dist.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: name_sentiment_price_dist.png")

    sample = df2.sample(min(5000, len(df2)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        sample["vader_score"],
        sample["price"].clip(upper=sample["price"].quantile(0.99)),
        c=sample["vader_score"], cmap="RdYlGn", alpha=0.35, s=8, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="VADER Compound Score")
    z  = np.polyfit(sample["vader_score"],
                    sample["price"].clip(upper=sample["price"].quantile(0.99)), 1)
    xs = np.linspace(sample["vader_score"].min(), sample["vader_score"].max(), 200)
    ax.plot(xs, np.poly1d(z)(xs), "k--", linewidth=2, label=f"Trend (slope={z[0]:.1f})")
    ax.axvline( 0.05, color="#1B5E20", linestyle=":", linewidth=1.5, alpha=0.7, label="Positive threshold")
    ax.axvline(-0.05, color="#B71C1C", linestyle=":", linewidth=1.5, alpha=0.7, label="Negative threshold")
    ax.set_title("VADER Sentiment Score vs Listing Price", fontsize=12, fontweight="bold")
    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "name_vader_score_vs_price.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: name_vader_score_vs_price.png")

    df_binary = df2[df2["vader_label"].isin(["Positive", "Negative"])].copy()
    df_binary["label_enc"] = (df_binary["vader_label"] == "Positive").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df_binary["name_clean"], df_binary["label_enc"],
        test_size=0.2, random_state=42, stratify=df_binary["label_enc"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                   max_features=10000, sublinear_tf=True, min_df=2)),
        ("clf",   LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                      random_state=42, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)
    logger.info(
        f"TF-IDF + LR report:\n"
        f"{classification_report(y_te, model.predict(X_te), target_names=['Negative','Positive'])}"
    )

    feat_names = np.array(model.named_steps["tfidf"].get_feature_names_out())
    coefs      = model.named_steps["clf"].coef_[0]
    top_pos = feat_names[np.argsort(coefs)[-15:][::-1]]
    top_neg = feat_names[np.argsort(coefs)[:15]]
    pos_coefs = coefs[np.argsort(coefs)[-15:][::-1]]
    neg_coefs = coefs[np.argsort(coefs)[:15]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(range(15), pos_coefs[::-1], color="#1B5E20", edgecolor="white")
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels(top_pos[::-1], fontsize=9)
    axes[0].set_title("Words Most Associated with\nEmotionally Positive Names",
                      fontweight="bold", color="#1B5E20")
    axes[0].set_xlabel("LR Coefficient")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].barh(range(15), np.abs(neg_coefs[::-1]), color="#B71C1C", edgecolor="white")
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels(top_neg[::-1], fontsize=9)
    axes[1].set_title("Words Most Associated with\nEmotionally Negative Names",
                      fontweight="bold", color="#B71C1C")
    axes[1].set_xlabel("|LR Coefficient|")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle(
        "Linguistic Patterns in Listing Names — TF-IDF + Logistic Regression\n"
        "(Sentiment labels from VADER, coefficients show word-level price influence)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(EDA_DIR / "name_sentiment_words.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: name_sentiment_words.png")

    summary = df2.groupby("vader_label")["price"].agg(["mean", "median", "count"])
    logger.info(f"\nPrice by VADER sentiment:\n{summary.to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(filename: str = "Airbnb_Data.csv") -> pd.DataFrame:
    df = load_raw_data(filename)
    num_feats, cat_feats = get_feature_lists(df)

    logger.info("=== Summary Statistics ===")
    logger.info(f"\n{df[num_feats + [TARGET_COL]].describe().to_string()}")

    logger.info("── Standard EDA ──")
    run_standard_eda(df, num_feats, cat_feats)

    logger.info("── Advanced Future Work (7 Points) ──")
    run_advanced_eda(df, num_feats, cat_feats)

    logger.info("── Description Word Cloud + Top-5 Price Buckets ──")
    run_description_analysis(df)

    logger.info("── Name Sentiment → Price Distribution ──")
    run_name_sentiment_analysis(df)

    logger.info(f"✓ All EDA outputs saved to: {EDA_DIR}")
    return df