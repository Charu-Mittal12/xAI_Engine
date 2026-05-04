#  Airbnb Price Prediction Pipeline

An end-to-end ML pipeline to predict Airbnb listing nightly prices using log-transformed regression models with full explainability.

---

##  Problem Statement

Given a listing's attributes (location, room type, amenities, host experience, reviews, availability), **predict the nightly price** of an Airbnb listing. Price is log-transformed to handle skewness. Post-hoc explainability is provided via LIME.

---

##  Project Structure

```
.
├── data/AirbnbData.csv         # Raw dataset
├── src/
│   ├── eda.py                  # 14+ exploratory plots
│   ├── preprocess.py           # Feature engineering & splits
│   ├── train.py                # Multi-model training + hyperparameter sweep
│   ├── experiments.py          # Learning curve tracking
│   ├── evaluate.py             # RMSE / MAE / R² metrics
│   ├── explain.py              # LIME + SHAP explainability
│   ├── model_comparision.py    # Comparison visualizations
│   ├── visualize.py            # Shared plotting utilities
│   └── utils.py                # Logging, I/O helpers
├── pipeline.py                 # Main entry point
└── outputs/                    # All artifacts saved here (auto-created)
```

---

##  Models

Each model is trained across 6 hyperparameter configs; best by RMSE is selected.

| Model | Type |
|---|---|
| Ridge Regression | Linear baseline |
| Random Forest | Ensemble (bagging) |
| XGBoost | Gradient boosting |
| CatBoost | Gradient boosting |

---

##  Setup

```bash
# 1. Clone & activate environment
git clone <your-repo-url> && cd <repo-folder>
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install scikit-learn xgboost catboost shap lime pandas numpy matplotlib seaborn scipy wordcloud

# 3. Place your data
cp /path/to/your/data.csv data/AirbnbData.csv
```

---

## Run the Pipeline

```bash
# Default run
python pipeline.py

# Custom file and observation index for LIME/SHAP
python pipeline.py --file MyData.csv --obs 5
```

| Argument | Default | Description |
|---|---|---|
| `--file` | `AirbnbData.csv` | CSV filename inside `data/` |
| `--obs` | `0` | Test-set row index for local explanation |

---

##  Pipeline Stages

1. **EDA** — Distributions, geo heatmaps, host experience, amenity analysis, word clouds
2. **Preprocessing** — Encoding, imputation, scaling, train/val/test split
3. **Training** — 4 models × 6 configs, best saved to `trained_models.pkl`
4. **Experiments** — Parameter sweeps + learning curves
5. **Evaluation** — RMSE, MAE, R² on test set → `metrics.json`
6. **Explainability** — LIME (4 plots/model) 

---

## Key Outputs

```
outputs/
├── eda/                        # All EDA + training plots
├── lime/<ModelName>/           # 4 LIME plots per model
├── trained_models.pkl
├── metrics.json
└── model_sweep_results.json
```

---

## Run Explainability Standalone

```bash
# Requires trained_models.pkl and data_splits.pkl to exist
python -m src.explain
```
