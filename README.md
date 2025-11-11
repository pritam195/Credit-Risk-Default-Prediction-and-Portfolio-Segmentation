# 💳 Credit Risk Default Prediction and Portfolio Segmentation

An end-to-end machine learning pipeline that simulates borrower credit profiles, analyzes key risk drivers, trains Probability-of-Default (PD) models, and segments a portfolio into actionable risk bands.  

It automatically produces visual EDA dashboards, model explainability insights, performance metrics, and reproducible artifacts under `results/`, with the selected model persisted under `models/`.

---

## 🚀 Key Features

✅ **Synthetic Portfolio Generator** — Creates realistic borrower profiles with financial and behavioral attributes.  
✅ **EDA Dashboards** — Explore distributions of credit score, DTI, utilization, delinquencies, and income.  
✅ **Model Training** — Trains Logistic Regression and Random Forest models, selects the best using AUC.  
✅ **Feature Importance** — Visualizes top risk drivers with ranked importance bars.  
✅ **Portfolio Segmentation** — Maps predicted PDs into Low, Moderate, High, and Very High Risk bands.  
✅ **Reproducible Outputs** — All plots, reports, and CSVs are saved automatically under `results/`.  
✅ **Model Persistence** — Best-performing model serialized and stored in `models/`.

---
## 🧩 Requirements

- Python 3.8+
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- joblib

---

## 🗂️ Project Structure

```text
├── main.py                 # Runs the complete pipeline end-to-end
├── requirements.txt        # Python dependencies
├── results/                # Generated plots, reports, and CSVs (auto-created)
├── models/                 # Serialized best model (auto-created)
└── .venv/                  # Local virtual environment (not committed)
```

---

## 📁 Where to Find Outputs
results/

File	Description
portfolio_analysis.png	Visual EDA dashboard (credit score, DTI, etc.)
feature_importance.png	Ranked bar chart of key risk drivers
portfolio_segmentation.csv	Borrower-level predicted PD and risk bands
model_performance.txt	Performance metrics and top drivers summary

models/

File	Description
best_credit_risk_model.pkl	Serialized best-performing model (via joblib)

---

## 🧠 How It Works (Pipeline Overview)

### Data Generation
Generates ~5,000 synthetic borrower profiles with realistic attributes — credit history, DTI, utilization, income, and delinquencies.
A binary PD label (default) is derived using domain-based rules plus random noise.

### EDA (Exploratory Data Analysis)
Visualizes key distributions and relationships — such as defaults by credit score, utilization, and income.

### Preprocessing
Label-encodes categorical variables and scales numeric features using StandardScaler.

### Model Training
- Fits Logistic Regression and Random Forest classifiers.
- Selects the best model by AUC score on a stratified test split.
- Saves model metrics and ROC curve plots.
- Feature Importance & Explainability
- Extracts feature importances or normalized coefficients.
- Produces a bar plot (feature_importance.png) with color-coded feature groups.

### Portfolio Segmentation
- Maps predicted PDs into four bands:
- Low Risk: PD < 0.02
- Moderate Risk: 0.02 ≤ PD < 0.08
- High Risk: 0.08 ≤ PD < 0.20
- Very High Risk: PD ≥ 0.20
- Each segment includes borrower counts, mean PD, and actionable insights.

---

## 🏁 Summary

This project demonstrates a complete and reproducible credit risk modeling workflow, covering:
- Synthetic data generation

- EDA and visualization

- Model development and evaluation

- Risk segmentation and reporting
It can easily be extended to real credit datasets or production deployment scenarios.
