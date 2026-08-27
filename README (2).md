# 💳 Credit Card Revenue Intelligence Engine

**An end-to-end analytics case study: how a bank with 50MM+ credit card customers can increase revenue and spend per card — using EDA, feature engineering, XGBoost + SHAP, and a live Streamlit decision tool.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 The problem

A major bank wants to grow **revenue per credit card customer**. Most public
tutorials on this dataset stop at *churn prediction*. This project reframes
it around the harder, more useful business question: **what actually drives
revenue, and which customers represent the biggest untapped opportunity?**

## 📊 Dataset

[Credit Card Customers (BankChurners)](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
— 10,127 customers, 21 features, Kaggle. Includes transaction volume,
utilization, credit limit, relationship depth, and card tier — everything
needed to model revenue instead of just churn.

## 🏗️ What this does

1. **Engineers a real revenue target** — `Estimated_Annual_Revenue` built
   from interchange fees + interest income + annual fees, decomposed by
   source so you can see exactly where the money comes from.
2. **Trains an XGBoost regressor** (R² = 0.98) to explain revenue from
   behavioral, demographic, and relationship features.
3. **Uses SHAP** to identify the true top revenue drivers — not assumptions,
   model-derived:
   1. Credit Utilization
   2. Credit Limit
   3. Transaction Frequency
   4. Card Tier
4. **Builds a "Revenue Opportunity Score"** — the project's key differentiator.
   Instead of chasing top spenders, it flags customers with unused credit
   capacity **and** strong engagement who are still under-monetized versus
   peers on the same card tier. On this dataset that segment (~2,100
   customers) represents an estimated **$451K in revenue uplift potential —
   14% of total portfolio revenue.**
5. **Ships a 5-tab Streamlit app**: Overview, Revenue Drivers (SHAP),
   Segmentation (K-Means), Hidden Revenue Finder, and a live What-If
   Simulator for testing how utilization/limit/frequency changes move
   predicted revenue.

## 🖥️ App preview

| Tab | What it shows |
|---|---|
| 📊 Overview | Portfolio KPIs, revenue Pareto curve, revenue mix by source |
| 🔍 Revenue Drivers | SHAP feature importance + the 4-factor breakdown |
| 🧩 Segmentation | K-Means customer segments by spend behavior |
| 🎯 Hidden Revenue Finder | The opportunity-scoring twist, with target account list |
| 🧮 What-If Simulator | Adjust a customer profile, see predicted revenue update live |

## ⚙️ Tech stack

`Python` · `pandas` / `numpy` · `scikit-learn` · `XGBoost` · `SHAP` ·
`Streamlit` · `Plotly`

## 🚀 Quickstart

```bash
git clone https://github.com/<your-username>/credit-card-revenue-engine.git
cd credit-card-revenue-engine
pip install -r requirements.txt

python 01_prepare_data.py   # builds the revenue target + opportunity score
python 02_train_model.py    # trains XGBoost, computes SHAP importances
streamlit run app.py        # launches the app at localhost:8501
```

## 📁 Project structure

```
├── app.py                   # Streamlit app (5 tabs)
├── 01_prepare_data.py       # data cleaning + revenue feature engineering
├── 02_train_model.py        # XGBoost training + SHAP explainability
├── BankChurners.csv         # raw Kaggle dataset
├── artifacts/                # generated: model, encoders, SHAP values, processed data
├── requirements.txt
└── README.md
```

## 📌 Key results

| Metric | Value |
|---|---|
| Model R² | 0.98 |
| Model MAE | ~$15 |
| Top revenue driver | Credit Utilization |
| Opportunity segment size | ~2,100 customers (top quartile) |
| Estimated uplift potential | $451K (≈14% of portfolio revenue) |

## ⚠️ Caveats

- Revenue figures (interchange rate, APR, fee schedule) are illustrative,
  industry-typical assumptions — not real bank pricing. Swap in actual P&L
  inputs for a production version.
- Dataset is ~10K US customers; directionally useful for a case study, not
  a substitute for a real 50MM-customer portfolio.

## 📄 License

MIT — see [LICENSE](LICENSE).

---
*Built as a portfolio project applying marketing analytics and ML to a bank revenue-optimization case study.*
