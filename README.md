# Customer Churn Prediction

End-to-end machine learning project that predicts customer churn using the Telco Customer Churn dataset (7,043 customers, 21 features). The project covers exploratory data analysis, feature engineering, training of three classification models, threshold optimization, and hyperparameter tuning — all organized as a modular, production-style Python pipeline.

---

## Results

| Model               | Accuracy | Precision | Recall | F1     | AUC-ROC |
|---------------------|----------|-----------|--------|--------|---------|
| Logistic Regression | 0.7589   | 0.5298    | 0.8075 | 0.6398 | 0.8470  |
| Random Forest       | 0.7617   | 0.5373    | 0.7326 | 0.6199 | 0.8125  |
| **XGBoost (tuned)** | **0.7872** | **0.5830** | **0.6952** | **0.6341** | **0.8467** |

> Recall is prioritized over precision: in a churn context, missing a churner (false negative) is more costly than a false alarm (false positive). All models use class imbalance handling (`class_weight="balanced"` / `scale_pos_weight`). Random Forest and XGBoost use optimized decision thresholds found via F1 maximization on the cross-validation set (thresholds constrained to 0.3–0.7 to avoid degenerate values). AUC-ROC is reported as a threshold-independent measure of model quality.

---

## Key Findings from EDA

- **Contract type** is the strongest predictor. Month-to-month customers churn at **42.7%** vs **2.6%** for two-year contracts.
- **Tenure** is the strongest numeric predictor (correlation: **-0.35**). Churned customers have a median tenure of 10 months vs 38 months for retained customers — the first year is the highest-risk window.
- **Internet service**: Fiber optic customers churn at **41.9%**, more than double DSL users (19%).
- **Add-on services** (OnlineSecurity, TechSupport) cut churn roughly in half (~41% without vs ~15% with).
- **Payment method**: Electronic check users churn at **45.3%** vs ~16% for automatic payment methods.
- **Gender** shows virtually no difference (26.9% vs 26.2%) — not a useful predictor.
- Class distribution: **73.5% retained / 26.5% churned** — imbalanced but workable.

---

## Project Structure

```
churn-prediction/
│
├── data/
│   └── raw/
│       └── Telco_Customer_Churn.csv
│
├── models/                        # Serialized model files (.pkl)
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Full exploratory data analysis
│   └── 02_model_debug.ipynb       # Model debugging and inspection
│
├── src/
│   ├── data/
│   │   ├── ingestion.py           # CSV loading with existence and type checks
│   │   ├── transformation.py      # Cleaning, encoding, imputation, feature engineering
│   │   └── validation.py          # Schema checks before processing
│   │
│   ├── models/
│   │   ├── train.py               # Splits data, fits imputer on train only, trains LR/RF/XGBoost
│   │   ├── evaluate.py            # Metrics, AUC-ROC, threshold optimization, confusion matrices
│   │   ├── tune.py                # GridSearchCV hyperparameter tuning for XGBoost
│   │   └── predict.py             # Inference on new data
│   │
│   └── pipelines/
│       ├── training_pipeline.py   # Orchestrates full training flow
│       └── inference_pipeline.py  # Orchestrates prediction on new data
│
├── tests/
│   ├── test_validation.py         # Tests for validate() — empty df, missing columns
│   └── test_transformation.py     # Tests for fix_column_types, encode_binary, engineer_features
├── conftest.py                    # pytest configuration
├── requirements.txt               # Pinned dependencies
├── main.py                        # Entry point — runs training pipeline
└── README.md
```

---

## Pipeline Overview

```
Raw CSV → validate() → clean_and_encode() → train() → tune_xgboost() → evaluate()
```

1. **Ingestion** — loads CSV, checks file type and existence
2. **Validation** — verifies required columns before any processing
3. **Transformation** — normalizes column names, fixes `TotalCharges` dtype, binary + one-hot encoding, feature engineering. Does not impute — no statistical fitting happens here.
4. **Training** — 80/10/10 split first, then `SimpleImputer` fitted on `X_train` only and applied to all three subsets. Trains Logistic Regression, Random Forest, XGBoost with imbalance handling.
5. **Tuning** — GridSearchCV on XGBoost across learning rate, max depth, and n_estimators
6. **Evaluation** — metrics on CV and test sets, optimized thresholds (0.3–0.7 range), AUC-ROC, confusion matrices

---

## Feature Engineering

Three features created during transformation:

| Feature                 | Formula                                  | Rationale                                           |
|-------------------------|------------------------------------------|-----------------------------------------------------|
| `charges_per_tenure`    | `MonthlyCharges / (tenure + 1)`          | Cost efficiency per month of customer lifetime      |
| `charge_to_total_ratio` | `MonthlyCharges / (TotalCharges + 1)`    | Detects early-stage customers with low accumulation |
| `new_customer`          | `1 if tenure <= 12 else 0`               | Flags first-year customers (highest churn risk)     |

---

## Tech Stack

- **Language:** Python 3.13
- **Data:** pandas, numpy
- **Modeling:** scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn
- **Utilities:** joblib, pathlib, pytest
- **Environment:** Jupyter Notebooks, VS Code

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/SantiLopBla/churn-prediction.git
cd churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add dataset
# Place Telco_Customer_Churn.csv in data/raw/

# 4. Run the training pipeline
python main.py
```

---

## Testing

The project includes 5 automated tests covering the data validation and transformation layers.

```bash
# Run all tests
pytest -v

# Run a specific file
pytest tests/test_validation.py -v
pytest tests/test_transformation.py -v
```

---

## Dataset

**Telco Customer Churn** — IBM sample dataset available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
7,043 customers · 21 features · Binary target: `Churn` (Yes/No)

---

## Author

**Santiago López Blanco**  
Data Science Engineering student — Universidad Fidélitas, Costa Rica  
[GitHub](https://github.com/SantiLopBla) · [LinkedIn](https://www.linkedin.com/in/santiago-l%C3%B3pez-blanco-420886342/)
