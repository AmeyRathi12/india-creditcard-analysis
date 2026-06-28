# 💳 Decoding Indian Spending: Credit Card Behaviour Analysis

> Surfacing 4 actionable business insights from 26,000+ transactions across 6 Indian cities using advanced SQL and Python.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/spaces/Amey12/india-creditcard-analysis)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![SQL](https://img.shields.io/badge/SQL-Advanced-orange?style=flat-square)]()
[![Pandas](https://img.shields.io/badge/Pandas-2.0-green?style=flat-square)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square)](https://huggingface.co/spaces/Amey12/india-creditcard-analysis)

---

## 🚀 Live Demo

**👉 [Try the interactive dashboard here](https://huggingface.co/spaces/Amey12/india-creditcard-analysis)**

Explore all 4 insights interactively — filter by city, card type, and year in real time.

---

## 📊 Key findings

| Finding | Detail |
|---|---|
| 📅 Festive spending spike | **40%+ increase** in Oct–Dec — missed entirely in raw data |
| 💳 Platinum vs Silver gap | **2.8x higher** spend per transaction for Platinum cardholders |
| 👥 Gender-category preference | Statistically significant differences by gender across spend categories |
| 🏙️ City-level patterns | Distinct spending profiles across Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Kolkata |

---

## 🛠 Tech stack

- **Language:** Python 3.10+
- **Data wrangling:** Pandas, NumPy
- **SQL:** Window functions, GROUP BY, subqueries, CTEs
- **Visualisation:** Matplotlib, Seaborn
- **App:** Streamlit (deployed on Hugging Face Spaces)

---

## 📁 Project structure

```
india-creditcard-analysis/
│
├── app.py                         # Streamlit dashboard (live on HF Spaces)
├── requirements.txt               # Dependencies
│
├── data/
│   └── credit_card_transactions.csv   # Raw dataset (26,000+ rows)
│
├── sql/
│   ├── spending_by_city.sql
│   ├── festive_season_analysis.sql
│   ├── card_tier_comparison.sql
│   └── gender_category_preferences.sql
│
├── notebooks/
│   └── analysis_walkthrough.ipynb     # End-to-end analysis with commentary
│
└── README.md
```

---

## 🔬 Methodology

- **Outlier removal:** IQR-based filtering to remove transaction anomalies
- **Feature engineering:** Datetime decomposition (month, quarter, day-of-week, festive flag)
- **SQL analysis:** Advanced aggregations with window functions for rank and percentile analysis
- **Statistical validation:** Group comparisons with significance checks

---

## ▶️ Run locally

```bash
git clone https://github.com/AmeyRathi12/india-creditcard-analysis
cd india-creditcard-analysis
pip install -r requirements.txt
streamlit run app.py
```

---

## 💡 Business applications

- **Card product targeting:** Platinum vs Silver spend gap informs premium card marketing
- **Festive campaigns:** 40%+ spend spike signals optimal timing for cashback/reward offers
- **Category promotions:** Gender-based category preferences enable personalised offer design

---

Built by [Amey Rathi](https://github.com/AmeyRathi12) · ML/AI Engineer · Data Analyst
