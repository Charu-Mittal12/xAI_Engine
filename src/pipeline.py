"""
pipeline.py — Full end-to-end runner for the Airbnb Price Prediction project.

Run with:
    python pipeline.py
    python pipeline.py --file MyData.csv --obs 5

Output folder layout after a successful run
───────────────────────────────────────────
outputs/
├── eda/
│   ├── price_and_logprice_distributions.png   ← merged from plot_price_distributions.py
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── missing_values.png
│   ├── outlier_boxplots.png
│   ├── price_vs_<cat>.png                     (up to 4 categoricals)
│   ├── price_vs_<num>.png                     (accommodates / bedrooms / review_scores_rating)
│   ├── adv_01_price_buckets.png
│   ├── adv_02_geo_price.png
│   ├── adv_03_host_experience.png
│   ├── adv_04_amenity_count.png
│   ├── adv_05_cancellation_price.png
│   ├── adv_06_availability.png
│   ├── adv_07_room_city_interaction.png
│   ├── desc_wordcloud.png
│   ├── desc_top_words.png
│   ├── desc_top5_price_buckets.png
│   ├── name_sentiment_price_dist.png
│   ├── name_vader_score_vs_price.png
│   ├── name_sentiment_words.png
│   ├── model_comparison_bar.png               ← produced by train.py
│   └── model_hyperparameter_tables.png        ← produced by train.py
├── lime/
│   └── <ModelName>/
│       ├── <ModelName>_1_feature_weights.png
│       ├── <ModelName>_2_n_sensitivity.png
│       ├── <ModelName>_3_kernel_sensitivity.png
│       └── <ModelName>_4_sampling_comparison.png
# ├── shap/                                          [DISABLED — hardware constraints]
# │   └── <ModelName>/
# │       ├── <ModelName>_1_shap_local.png
# │       ├── <ModelName>_2_shap_global.png
# │       ├── <ModelName>_3_shap_beeswarm.png
# │       └── <ModelName>_4_shap_dependence.png
├── preprocessor.pkl
├── feature_names.pkl
├── trained_models.pkl
├── model_sweep_results.json
├── metrics.json
├── learning_curves.pkl
├── explanations.pkl
└── data_splits.pkl
"""

import argparse
from src.preprocess import prepare_data
from src.train import train_all_models
from src.experiments import run_parameter_sweeps
from src.evaluate import evaluate_all_models, metrics_to_dataframe
from src.explain import run_lime  # , run_shap, compute_overlap  # DISABLED — hardware constraints
from src.eda import run_eda
from src.utils import get_logger, save_pickle, OUTPUT_DIR
from src.visualize import plot_learning_curves, plot_model_performance_table

logger = get_logger("pipeline")

# Output sub-directories (created on demand)
EDA_DIR  = OUTPUT_DIR / "eda"
LIME_DIR = OUTPUT_DIR / "lime"
# SHAP_DIR = OUTPUT_DIR / "shap"   # DISABLED — hardware constraints
for _d in [EDA_DIR, LIME_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def run_pipeline(filename: str = "Airbnb_Data.csv", obs_idx: int = 0):
    # ══════════════════════════════════════════════════════════════════════
    # 1. EDA  — all plots saved to outputs/eda/
    #           includes the merged price_and_logprice_distributions.png
    # ══════════════════════════════════════════════════════════════════════
    logger.info("══════════ EDA ══════════")
    run_eda(filename)

    # ══════════════════════════════════════════════════════════════════════
    # 2. PREPROCESSING
    # ══════════════════════════════════════════════════════════════════════
    logger.info("══════════ PREPROCESSING ══════════")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names, preprocessor, X_test_raw) = prepare_data(filename)

    # Persist splits for the standalone explain runner
    save_pickle({
        "X_train":       X_train,
        "X_val":         X_val,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_val":         y_val,
        "y_test":        y_test,
        "feature_names": feature_names,
        "X_test_raw":    X_test_raw,
    }, OUTPUT_DIR / "data_splits.pkl")

    # ══════════════════════════════════════════════════════════════════════
    # 3. TRAINING
    #    Produces:
    #      outputs/trained_models.pkl
    #      outputs/model_sweep_results.json
    #      outputs/eda/model_comparison_bar.png
    #      outputs/eda/model_hyperparameter_tables.png
    # ══════════════════════════════════════════════════════════════════════
    logger.info("══════════ TRAINING ══════════")
    trained_models = train_all_models(X_train, y_train, X_test, y_test)

    # ══════════════════════════════════════════════════════════════════════
    # 4. PARAMETER SWEEP EXPERIMENTS
    #    Produces:
    #      outputs/learning_curves.pkl
    #      outputs/eda/learning_curve_<ModelName>.png  (one per model)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("══════════ EXPERIMENTS ══════════")
    history = run_parameter_sweeps(X_train, y_train, X_val, y_val)

    for model_name in history:
        plot_learning_curves(
            history, model_name,
            path=EDA_DIR / f"learning_curve_{model_name}.png",
        )
        logger.info(f"Saved learning curve: {model_name}")

    # ══════════════════════════════════════════════════════════════════════
    # 5. EVALUATION
    #    Produces:
    #      outputs/metrics.json
    #      outputs/eda/model_performance_table.png
    # ══════════════════════════════════════════════════════════════════════
    logger.info("══════════ EVALUATION ══════════")
    metrics    = evaluate_all_models(trained_models, X_test, y_test)
    metrics_df = metrics_to_dataframe(metrics)
    logger.info(f"\n{metrics_df.to_string()}")

    plot_model_performance_table(
        metrics_df,
        path=EDA_DIR / "model_performance_table.png",
    )
    logger.info("Saved: model_performance_table.png")

    # ══════════════════════════════════════════════════════════════════════
    # 6. LIME  — 4 plots per model saved to outputs/lime/<ModelName>/
    # ══════════════════════════════════════════════════════════════════════
    logger.info("══════════ LIME ══════════")
    import numpy as np
    y_test_arr  = np.array(y_test)
    X_train_arr = X_train if not hasattr(X_train, "values") else X_train.values
    X_test_arr  = X_test  if not hasattr(X_test,  "values") else X_test.values

    lime_results = run_lime(
        trained_models, X_train_arr, X_test_arr, y_test_arr, feature_names
    )

    # ══════════════════════════════════════════════════════════════════════
    # 7. SHAP  — DISABLED due to hardware constraints
    #    To re-enable: uncomment this block and the run_shap / compute_overlap
    #    imports at the top of the file.
    # ══════════════════════════════════════════════════════════════════════
    # logger.info("══════════ SHAP ══════════")
    # shap_results = run_shap(
    #     trained_models, X_train_arr, X_test_arr, y_test_arr, feature_names
    # )

    # ══════════════════════════════════════════════════════════════════════
    # 8. OVERLAP SCORES  — DISABLED (requires SHAP results)
    # ══════════════════════════════════════════════════════════════════════
    # logger.info("══════════ OVERLAP SCORES ══════════")
    # overlaps = compute_overlap(lime_results, shap_results)

    # ══════════════════════════════════════════════════════════════════════
    # 9. PERSIST EVERYTHING
    # ══════════════════════════════════════════════════════════════════════
    save_pickle({
        "lime":    lime_results,
        # "shap":    shap_results,   # DISABLED — hardware constraints
        # "overlaps": overlaps,      # DISABLED — hardware constraints
        "obs_idx": obs_idx,
    }, OUTPUT_DIR / "explanations.pkl")

    logger.info("══════════════════════════════════════════════")
    logger.info("Pipeline complete.  All outputs in /outputs/")
    logger.info(f"  EDA plots   → {EDA_DIR}")
    logger.info(f"  LIME plots  → {LIME_DIR}")
    # logger.info(f"  SHAP plots  → {SHAP_DIR}")   # DISABLED — hardware constraints
    logger.info("══════════════════════════════════════════════")

    return trained_models, metrics, history, lime_results  # shap_results, overlaps removed


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Airbnb price-prediction pipeline."
    )
    parser.add_argument(
        "--file", default="Airbnb_Data.csv",
        help="CSV filename inside the data/ directory (default: Airbnb_Data.csv)",
    )
    parser.add_argument(
        "--obs", type=int, default=0,
        help="Test-set observation index used for LIME/SHAP local explanations (default: 0)",
    )
    args = parser.parse_args()
    run_pipeline(filename=args.file, obs_idx=args.obs)