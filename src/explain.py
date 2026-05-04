"""
Standalone Explainability Runner.
Loads pre-trained models and data splits from outputs/
Runs: LIME (4 plots) + SHAP (4 plots) + Overlap score per model.
Run independently: python -m src.explain
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path

from src.utils import get_logger, load_pickle, OUTPUT_DIR

logger    = get_logger("explain")
LIME_DIR  = OUTPUT_DIR / "lime"
SHAP_DIR  = OUTPUT_DIR / "shap"
LIME_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

OBS_IDX          = 0
TOP_N            = 5
N_VALUES         = [100, 500, 1000]
KERNEL_WIDTHS    = [0.25, 0.75, 2.0]
SAMPLING_METHODS = ["gaussian", "lhs"]
DEFAULT_N        = 1000
DEFAULT_KERNEL   = 0.75
DEFAULT_SAMPLE   = "gaussian"

MODEL_COLORS = {
    "LinearRegression": "#90CAF9",
    "RandomForest":     "#A5D6A7",
    "XGBoost":          "#FFCC80",
    "CatBoost":         "#CE93D8",
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD PRE-TRAINED MODELS + DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_artifacts():
    logger.info("Loading trained_models.pkl ...")
    trained_models = load_pickle(OUTPUT_DIR / "trained_models.pkl")

    logger.info("Loading data_splits.pkl ...")
    splits = load_pickle(OUTPUT_DIR / "data_splits.pkl")

    X_train      = splits["X_train"]
    X_test       = splits["X_test"]
    y_test       = splits["y_test"]
    feature_names = splits["feature_names"]
    X_test_raw   = splits.get("X_test_raw", None)

    # Convert to numpy if DataFrame
    X_train_arr = X_train.values if hasattr(X_train, "values") else X_train
    X_test_arr  = X_test.values  if hasattr(X_test,  "values") else X_test
    y_test_arr  = np.array(y_test)

    logger.info(f"Loaded {len(trained_models)} models | "
                f"X_train: {X_train_arr.shape} | X_test: {X_test_arr.shape}")

    return trained_models, X_train_arr, X_test_arr, y_test_arr, feature_names, X_test_raw


# ══════════════════════════════════════════════════════════════════════════════
# ── LIME SECTION ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _lime_explainer(X_train, feature_names, kernel_width):
    """sampling_method is NOT a constructor arg in older LIME — pass to explain_instance instead."""
    return LimeTabularExplainer(
        training_data        = X_train,
        feature_names        = feature_names,
        mode                 = "regression",
        discretize_continuous= True,
        kernel_width         = kernel_width,
        random_state         = 42,
    )



def _lime_top_features(model, X_train, X_test, feature_names,
                       n_samples, kernel_width, sampling_method, obs_idx):
    explainer = _lime_explainer(X_train, feature_names, kernel_width)
    try:
        # newer LIME versions support sampling_method in explain_instance
        exp = explainer.explain_instance(
            data_row        = X_test[obs_idx],
            predict_fn      = model.predict,
            num_features    = len(feature_names),
            num_samples     = n_samples,
            sampling_method = sampling_method,   # ← here, not in constructor
        )
    except TypeError:
        # older LIME versions — sampling_method not supported at all, fallback
        logger.warning(f"sampling_method='{sampling_method}' not supported — using default")
        exp = explainer.explain_instance(
            data_row   = X_test[obs_idx],
            predict_fn = model.predict,
            num_features = len(feature_names),
            num_samples  = n_samples,
        )
    raw = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)[:TOP_N]
    return {f: w for f, w in raw}


# ── LIME Plot 1: Feature Weight Bar Chart ─────────────────────────────────────

def _lime_plot_weights(model_name, top_feats, pred, actual, color, path):
    feats   = list(top_feats.keys())
    weights = list(top_feats.values())
    colors  = ["#2E7D32" if w > 0 else "#C62828" for w in weights]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(feats[::-1], weights[::-1], color=colors[::-1],
            edgecolor="white", height=0.6)
    for feat, w in zip(feats[::-1], weights[::-1]):
        ax.text(w + (0.002 if w >= 0 else -0.002),
                feats[::-1].index(feat),
                f"{w:+.4f}", va="center",
                ha="left" if w >= 0 else "right",
                fontsize=9, fontweight="bold", color="#1A237E")

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title(
        f"{model_name} — LIME Top-{TOP_N} Feature Weights\n"
        f"N={DEFAULT_N} | Kernel={DEFAULT_KERNEL} | Sampling={DEFAULT_SAMPLE}\n"
        f"Predicted log_price: {pred:.4f}  |  Actual log_price: {actual:.4f}",
        fontsize=11, fontweight="bold")
    ax.set_xlabel("LIME Weight  (+) pushes price UP  |  (−) pushes price DOWN",
                  fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2E7D32", label="Increases prediction"),
        Patch(facecolor="#C62828", label="Decreases prediction"),
    ], loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── LIME Plot 2: N Sensitivity ────────────────────────────────────────────────

def _lime_plot_n_sensitivity(model_name, n_results, path):
    all_feats = list(dict.fromkeys(
        f for res in n_results.values() for f in res))[:TOP_N]
    n_vals = sorted(n_results.keys())
    cmap   = plt.cm.get_cmap("tab10", len(all_feats))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, feat in enumerate(all_feats):
        weights = [n_results[n].get(feat, 0) for n in n_vals]
        ax.plot([str(n) for n in n_vals], weights,
                marker="o", linewidth=2.2, markersize=8,
                color=cmap(i), label=feat[:40])
        for n, w in zip(n_vals, weights):
            ax.annotate(f"{w:+.3f}", (str(n), w),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7.5, color=cmap(i))
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_title(
        f"{model_name} — LIME N Sensitivity\n"
        f"Kernel={DEFAULT_KERNEL} | Sampling={DEFAULT_SAMPLE} | Obs={OBS_IDX}",
        fontsize=11, fontweight="bold")
    ax.set_xlabel("N (Perturbed Samples)", fontsize=10)
    ax.set_ylabel("LIME Feature Weight", fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── LIME Plot 3: Kernel Sensitivity ──────────────────────────────────────────

def _lime_plot_kernel_sensitivity(model_name, kernel_results, path):
    all_feats = list(dict.fromkeys(
        f for res in kernel_results.values() for f in res))[:TOP_N]
    k_vals = sorted(kernel_results.keys())
    cmap   = plt.cm.get_cmap("tab10", len(all_feats))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, feat in enumerate(all_feats):
        weights = [kernel_results[k].get(feat, 0) for k in k_vals]
        ax.plot([str(k) for k in k_vals], weights,
                marker="s", linewidth=2.2, markersize=8,
                color=cmap(i), label=feat[:40])
        for k, w in zip(k_vals, weights):
            ax.annotate(f"{w:+.3f}", (str(k), w),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7.5, color=cmap(i))
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_title(
        f"{model_name} — LIME Kernel Width Sensitivity\n"
        f"N={DEFAULT_N} | Sampling={DEFAULT_SAMPLE} | Obs={OBS_IDX}",
        fontsize=11, fontweight="bold")
    ax.set_xlabel("Kernel Width  (tight ← 0.25 → wide 2.0)", fontsize=10)
    ax.set_ylabel("LIME Feature Weight", fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── LIME Plot 4: Sampling Method Comparison ───────────────────────────────────

def _lime_plot_sampling(model_name, sampling_results, path):
    all_feats = list(dict.fromkeys(
        f for res in sampling_results.values() for f in res))[:TOP_N]
    methods = list(sampling_results.keys())
    x = np.arange(len(all_feats))
    width = 0.35
    method_colors = {"gaussian": "#1565C0", "lhs": "#E65100"}

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, method in enumerate(methods):
        weights = [sampling_results[method].get(f, 0) for f in all_feats]
        offset  = (i - 0.5) * width
        bars = ax.bar(x + offset, weights, width,
                      label=method, color=method_colors[method],
                      alpha=0.82, edgecolor="white")
        for bar, w in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.001 if w >= 0 else -0.003),
                    f"{w:+.3f}", ha="center",
                    va="bottom" if w >= 0 else "top",
                    fontsize=8, color="#1A237E")
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f[:30] for f in all_feats], rotation=15, ha="right", fontsize=8.5)
    ax.set_title(
        f"{model_name} — Sampling Method Comparison (Gaussian vs LHS)\n"
        f"N={DEFAULT_N} | Kernel={DEFAULT_KERNEL} | Obs={OBS_IDX}",
        fontsize=11, fontweight="bold")
    ax.set_ylabel("LIME Feature Weight", fontsize=10)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── LIME Master ───────────────────────────────────────────────────────────────

def run_lime(trained_models, X_train, X_test, y_test, feature_names):
    results = {}
    for model_name, model in trained_models.items():
        logger.info(f"\n{'='*50}\nLIME — {model_name}")
        model_dir = LIME_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        pred   = float(model.predict(X_test[[OBS_IDX]])[0])
        actual = float(y_test[OBS_IDX])

        # Graph 1 — default weights
        default = _lime_top_features(model, X_train, X_test, feature_names,
                                     DEFAULT_N, DEFAULT_KERNEL, DEFAULT_SAMPLE, OBS_IDX)
        _lime_plot_weights(model_name, default, pred, actual,
                           MODEL_COLORS[model_name],
                           model_dir / f"{model_name}_1_feature_weights.png")

        # Graph 2 — N sensitivity
        n_results = {n: _lime_top_features(model, X_train, X_test, feature_names,
                                           n, DEFAULT_KERNEL, DEFAULT_SAMPLE, OBS_IDX)
                     for n in N_VALUES}
        _lime_plot_n_sensitivity(model_name, n_results,
                                 model_dir / f"{model_name}_2_n_sensitivity.png")

        # Graph 3 — kernel sensitivity
        kernel_results = {k: _lime_top_features(model, X_train, X_test, feature_names,
                                                 DEFAULT_N, k, DEFAULT_SAMPLE, OBS_IDX)
                          for k in KERNEL_WIDTHS}
        _lime_plot_kernel_sensitivity(model_name, kernel_results,
                                      model_dir / f"{model_name}_3_kernel_sensitivity.png")

        # Graph 4 — sampling comparison
        sampling_results = {sm: _lime_top_features(model, X_train, X_test, feature_names,
                                                    DEFAULT_N, DEFAULT_KERNEL, sm, OBS_IDX)
                             for sm in SAMPLING_METHODS}
        _lime_plot_sampling(model_name, sampling_results,
                            model_dir / f"{model_name}_4_sampling_comparison.png")

        results[model_name] = {
            "top_features":       default,
            "n_sensitivity":      n_results,
            "kernel_sensitivity": kernel_results,
            "sampling_results":   sampling_results,
        }
        logger.info(f"✓ LIME complete — {model_name}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ── SHAP SECTION ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _get_shap_explainer(model_name, model, X_train):
    if model_name in ["XGBoost", "RandomForest", "CatBoost"]:
        return shap.TreeExplainer(model)
    else:
        return shap.LinearExplainer(model, X_train)


def run_shap(trained_models, X_train, X_test, y_test, feature_names):
    results = {}
    for model_name, model in trained_models.items():
        logger.info(f"\n{'='*50}\nSHAP — {model_name}")
        model_dir = SHAP_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        color = MODEL_COLORS[model_name]

        explainer   = _get_shap_explainer(model_name, model, X_train)
        shap_values = explainer.shap_values(X_test)

        # ── Plot 1: SHAP Waterfall for obs_idx ────────────────────────────
        shap_single = shap_values[OBS_IDX]
        top_idx     = np.argsort(np.abs(shap_single))[-TOP_N:][::-1]
        top_feats   = [feature_names[i] for i in top_idx]
        top_vals    = [shap_single[i]   for i in top_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        bar_colors = ["#2E7D32" if v > 0 else "#C62828" for v in top_vals]
        ax.barh(top_feats[::-1], top_vals[::-1],
                color=bar_colors[::-1], edgecolor="white", height=0.6)
        for feat, v in zip(top_feats[::-1], top_vals[::-1]):
            ax.text(v + (0.002 if v >= 0 else -0.002),
                    top_feats[::-1].index(feat),
                    f"{v:+.4f}", va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=9, fontweight="bold", color="#1A237E")
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_title(
            f"{model_name} — SHAP Top-{TOP_N} Feature Contributions\n"
            f"Obs={OBS_IDX} | Predicted: {float(model.predict(X_test[[OBS_IDX]])[0]):.4f} "
            f"| Actual: {float(y_test[OBS_IDX]):.4f}",
            fontsize=11, fontweight="bold")
        ax.set_xlabel("SHAP Value  (+) increases prediction  |  (−) decreases prediction",
                      fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(model_dir / f"{model_name}_1_shap_local.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {model_name}_1_shap_local.png")

        # ── Plot 2: Global Feature Importance (mean |SHAP|) ───────────────
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_global_idx  = np.argsort(mean_abs)[-TOP_N:][::-1]
        top_global_feat = [feature_names[i] for i in top_global_idx]
        top_global_vals = [mean_abs[i]       for i in top_global_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top_global_feat[::-1], top_global_vals[::-1],
                color=color, edgecolor="white", height=0.6)
        for feat, v in zip(top_global_feat[::-1], top_global_vals[::-1]):
            ax.text(v + 0.001, top_global_feat[::-1].index(feat),
                    f"{v:.4f}", va="center", ha="left",
                    fontsize=9, fontweight="bold", color="#1A237E")
        ax.set_title(
            f"{model_name} — SHAP Global Feature Importance\n"
            f"Mean |SHAP value| across all {X_test.shape[0]} test samples",
            fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(model_dir / f"{model_name}_2_shap_global.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {model_name}_2_shap_global.png")

        # ── Plot 3: SHAP Beeswarm (summary plot) ──────────────────────────
        top_feat_idx = np.argsort(mean_abs)[-TOP_N:][::-1]
        shap_top     = shap_values[:, top_feat_idx]
        X_test_top   = X_test[:, top_feat_idx]
        feat_top     = [feature_names[i] for i in top_feat_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (feat, sidx) in enumerate(zip(feat_top, top_feat_idx)):
            sv    = shap_values[:, sidx]
            fv    = X_test[:, sidx]
            norm  = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
            ax.scatter(sv, [i] * len(sv), c=norm, cmap="coolwarm",
                       alpha=0.3, s=8)
        ax.set_yticks(range(len(feat_top)))
        ax.set_yticklabels(feat_top, fontsize=9)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_title(
            f"{model_name} — SHAP Beeswarm (Top-{TOP_N} Features)\n"
            "Color = feature value (blue=low, red=high)",
            fontsize=11, fontweight="bold")
        ax.set_xlabel("SHAP Value", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(model_dir / f"{model_name}_3_shap_beeswarm.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {model_name}_3_shap_beeswarm.png")

        # ── Plot 4: SHAP Dependence (top feature) ─────────────────────────
        top1_idx  = top_global_idx[0]
        top1_feat = feature_names[top1_idx]
        sv_top1   = shap_values[:, top1_idx]
        fv_top1   = X_test[:, top1_idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        sc = ax.scatter(fv_top1, sv_top1, c=sv_top1, cmap="coolwarm",
                        alpha=0.4, s=10, linewidths=0)
        plt.colorbar(sc, ax=ax, label="SHAP Value")
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_title(
            f"{model_name} — SHAP Dependence Plot\n"
            f"Top feature: {top1_feat}",
            fontsize=11, fontweight="bold")
        ax.set_xlabel(top1_feat, fontsize=10)
        ax.set_ylabel("SHAP Value", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(model_dir / f"{model_name}_4_shap_dependence.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {model_name}_4_shap_dependence.png")

        results[model_name] = {feat_top[i]: float(top_global_vals[i])
                               for i in range(len(feat_top))}
        logger.info(f"✓ SHAP complete — {model_name}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAP SCORE
# ══════════════════════════════════════════════════════════════════════════════

def compute_overlap(lime_results, shap_results):
    overlaps = {}
    for model_name in lime_results:
        lime_feats = set(list(lime_results[model_name]["top_features"].keys())[:TOP_N])
        shap_feats = set(list(shap_results[model_name].keys())[:TOP_N])
        score = len(lime_feats & shap_feats) / len(lime_feats | shap_feats) \
                if (lime_feats | shap_feats) else 0.0
        overlaps[model_name] = round(score, 4)
        logger.info(f"LIME-SHAP Overlap ({model_name}): {score:.2%}")
    return overlaps


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    trained_models, X_train, X_test, y_test, feature_names, X_test_raw = \
        load_artifacts()

    logger.info("══════════ LIME ══════════")
    lime_results = run_lime(trained_models, X_train, X_test,
                            y_test, feature_names)

    # logger.info("══════════ SHAP ══════════")                         requires stronger CPU/RAM, so run separately if needed.
    # shap_results = run_shap(trained_models, X_train, X_test,
    #                         y_test, feature_names)

    logger.info("══════════ OVERLAP SCORES ══════════")
    overlaps = compute_overlap(lime_results, shap_results)

    from src.utils import save_pickle
    save_pickle({
        "lime":     lime_results,
        "shap":     shap_results,
        "overlaps": overlaps,
        "obs_idx":  OBS_IDX,
    }, OUTPUT_DIR / "explanations.pkl")

    logger.info("\nAll explanation outputs saved.")
    logger.info(f"LIME → {LIME_DIR}")
    logger.info(f"SHAP → {SHAP_DIR}")