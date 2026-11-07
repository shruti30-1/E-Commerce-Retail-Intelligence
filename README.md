# 🛒 Retail Demand & Customer Intelligence Platform

> **End-to-end data science portfolio project** using the Olist Brazilian E-Commerce public dataset.
> Built to demonstrate SQL analytics, EDA, feature engineering, unsupervised ML, supervised ML, time series forecasting, and an interactive Streamlit dashboard — skills directly tested in data analyst and data scientist interviews.

---

## 📊 Project Overview

This project simulates a **real-world retail analytics platform** where I act as the Data Scientist responsible for understanding customer behavior, predicting demand, and surfacing actionable business insights for the Olist marketplace.

**Dataset:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 9 relational tables, ~100K orders, 2016–2018

**Database:** Supabase (hosted PostgreSQL) connected via SQLAlchemy

---

## 🏗️ Architecture

```
📁 E-Commerce-Retail-Intelligence/
├── 📁 data/                        # Raw CSVs (not tracked in git)
│   ├── olist_orders_dataset.csv
│   ├── olist_customers_dataset.csv
│   └── ... (9 total CSVs)
├── 📁 notebooks/
│   ├── 01_setup_connection.ipynb   ✅ DB setup & data ingestion
│   ├── 02_analytical_sql.ipynb     ✅ Phase 1: SQL Analytics
│   ├── 03_eda.ipynb                ✅ Phase 2: Exploratory Data Analysis
│   ├── 04_rfm_feature_engineering.ipynb  ✅ Phase 3: Feature Engineering + RFM
│   ├── 05_customer_segmentation.ipynb    ✅ Phase 4: K-Means Clustering
│   ├── 06_demand_forecasting.ipynb       ✅ Phase 5: Prophet Forecasting
│   └── 07_review_prediction.ipynb        ✅ Phase 6: Random Forest Classifier
├── 📁 src/
│   └── dashboard.py                ✅ Phase 7: Streamlit Dashboard
├── .env                            🔒 (not tracked — credentials)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📚 Phase Breakdown

| # | Phase | Notebook | Key Skills |
|---|-------|----------|-----------|
| 1 | **Analytical SQL** | `02_analytical_sql.ipynb` | 6-table JOINs, CTEs, Window Functions, Date Arithmetic |
| 2 | **EDA** | `03_eda.ipynb` | Data Quality, Distribution Analysis, Correlation Heatmaps |
| 3 | **Feature Engineering + RFM** | `04_rfm_feature_engineering.ipynb` | RFM Scoring, `pd.qcut`, Feature Derivation |
| 4 | **Customer Segmentation** | `05_customer_segmentation.ipynb` | K-Means, Elbow Method, Silhouette Score, PCA |
| 5 | **Demand Forecasting** | `06_demand_forecasting.ipynb` | Time Series, Prophet, MAE/RMSE/MAPE, Temporal Split |
| 6 | **Review Prediction** | `07_review_prediction.ipynb` | Random Forest, ROC-AUC, Class Imbalance, Cross-Validation |
| 7 | **Dashboard** | `src/dashboard.py` | Streamlit, Plotly, Multi-page App |

---

## 🔑 Key Insights

- 📦 **~97% of orders are delivered** — Olist has strong fulfillment
- 🌍 **São Paulo accounts for ~40% of orders** — heavy regional concentration
- ⭐ **Late delivery is the #1 predictor of 1-star reviews** — confirmed by feature importance in Phase 6
- 👥 **80%+ of customers made only 1 purchase** — low repeat rate characteristic of Olist's marketplace model
- 📈 **~300% order volume growth** from Q4 2016 to Q4 2017 — strong early-stage growth

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python) | Core language |
| ![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas) | Data manipulation |
| ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5-f7931e?logo=scikit-learn) | ML models |
| ![Prophet](https://img.shields.io/badge/Prophet-1.1-blue) | Time series forecasting |
| ![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit) | Dashboard |
| ![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?logo=plotly) | Interactive charts |
| ![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?logo=supabase) | Cloud database |
| ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red) | DB connection |
| ![Jupyter](https://img.shields.io/badge/JupyterLab-4.3-orange?logo=jupyter) | Notebooks |

---

## 🚀 Setup & Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/retail-intelligence-olist.git
cd retail-intelligence-olist
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```
SUPABASE_URL=postgresql+psycopg2://postgres.<project-ref>:<password>@<host>:5432/postgres
```

### 4. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all 9 CSV files in the `data/` folder.

### 5. Load data into Supabase
Run `notebooks/01_setup_connection.ipynb` — this loads all 9 CSVs into Supabase as PostgreSQL tables.

### 6. Run the notebooks in order
```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

### 7. Launch the Streamlit dashboard
```bash
streamlit run src/dashboard.py
```

---

## 📂 Data Sources

- **Dataset:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **License:** CC BY-NC-SA 4.0
- **Raw files:** Not tracked in this repository. Download from Kaggle link above.

---

## 👩‍💻 Author

Built as a learning-first data science portfolio project. Every notebook contains **interview-ready markdown explanations** of the *why* behind each analytical and ML decision.

---

*Last updated: November 2026*
