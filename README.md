# 📈 StockAlpha – Explainable Stock Signals (ML + LLM)

Generate interpretable stock movement signals using real financial data, machine learning models, and GenAI explanations — all visualized in an interactive Streamlit dashboard.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13%2B-blue?logo=python">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-Boosting-blueviolet">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit">
  <img alt="OpenAI" src="https://img.shields.io/badge/GenAI-LLM%20Explanations-8A2BE2">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> 🎓 Developed for a graduate-level **Finance Information Processing** course.  
> ⚠️ Educational use only – **not financial or trading advice**.

---

## 📚 Table of Contents

1. [Overview](#1-overview)  
2. [Problem Statement](#2-problem-statement)  
3. [Solution Architecture](#3-solution-architecture)  
4. [Tech Stack](#4-tech-stack)  
5. [Project Structure](#5-project-structure)  
6. [Features](#6-features)  
7. [Installation & Usage](#7-installation--usage)  
8. [Limitations](#8-limitations)  
9. [Future Work](#9-future-work)  

---

## 1. 📌 Overview

**StockAlpha** predicts whether a stock’s next-day closing price will be higher than today’s and explains the prediction in plain English using a Large Language Model (LLM).

The full pipeline includes:
- Live market data (via Alpha Vantage API)
- Feature engineering (technical indicators)
- ML models (Logistic Regression, Random Forest, XGBoost)
- LLM-based explanations
- Streamlit dashboard

---

## 2. ❓ Problem Statement

Predict whether a stock’s **next closing price** will be **up** or **not up**, and explain that prediction in human-readable language.

### Key Constraints
- Use real-time data from a **financial API**
- Output must be **interpretable** to non-technical users
- Provide a **complete working codebase and dashboard**

---

## 3. Solution Architecture

```text
                ┌────────────────────┐
                │  Alpha Vantage API │
                └────────┬───────────┘
                         ▼
                  fetch_data.py
                         ▼
              ┌────────────────────┐
              │ Feature Engineering│
              └────────┬───────────┘
                         ▼
                  features.py
                         ▼
              ┌────────────────────┐
              │  Model Training    │
              └────────┬───────────┘
                         ▼
                   models.py
                         ▼
              ┌────────────────────┐
              │   Prediction +     │
              │  LLM Explanation   │
              └────────┬───────────┘
                         ▼
         Streamlit UI ← dashboard.py → llm_explain.py
````

---

## 4. Tech Stack

### 🧪 Programming

* Python 3.13+
* Conda environment (`finance_stock_env`)

### 🔧 Libraries

* **Data**: `pandas`, `numpy`
* **Modeling**: `scikit-learn`, `xgboost`
* **API**: `requests`
* **App**: `streamlit`
* **Environment**: `python-dotenv`
* **LLM**: `openai` (or any compatible API client)

### 🌐 Services

* **Alpha Vantage API** – Stock data
* **OpenAI API** – LLM-based explanations

---

## 5. Project Structure

```
stockalpha-explainable-ml-llm/
│
├── app/
│   └── dashboard.py             # Streamlit dashboard interface
│
├── src/
│   ├── config.py                # Configurations and environment loading
│   ├── fetch_data.py            # Fetch stock data via API
│   ├── features.py              # Generate features (MAs, returns, vol)
│   ├── models.py                # Train and evaluate ML models
│   ├── llm_explain.py           # Generate LLM-based explanations
│   └── __init__.py
│
├── data/                        # Not version controlled
│   ├── raw/                     # Raw downloaded stock data
│   └── processed/               # Processed feature sets
│
├── demo_prediction.py           # Command-line demo for predictions
├── .env.example                 # Environment variable template
├── .gitignore
└── README.md
```

---

## 6. Features

### ✅ Stock Signal Modeling

* Stocks: `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`
* Target: `1` if next close > today’s, else `0`
* Features:

  * Moving averages (`short_ma`, `long_ma`)
  * Returns (`return`, `ret_lag1`, `ret_lag2`)
  * Volatility (`short_vol`, `long_vol`)
* Models:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Metric: **Directional accuracy** (53–55%)

### 💡 Explainability with LLM

* Inputs:

  * Symbol, date, prediction, feature values
* Outputs:

  * Natural-language explanations
  * Disclaimers: risk, weak signals, educational use
* Fallback:

  * Rule-based explanation if API fails

### 📊 Streamlit Dashboard

* Select stock and date
* See:

  * Prediction + probability
  * LLM explanation
  * Price chart (adjusted close)
  * Model vs actual (30-day bar chart)
  * Ranked signals for all stocks

---

## 7. Installation & Usage

### 1. Clone Repository

```bash
git clone https://github.com/ritwikm14/stockalpha-explainable-ml-llm.git
cd stockalpha-explainable-ml-llm
```

### 2. Set Up Environment

```bash
conda create -n finance_stock_env python=3.13 -y
conda activate finance_stock_env
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

> Or manually:

```bash
pip install pandas numpy scikit-learn xgboost streamlit python-dotenv requests openai
```

### 4. Configure API Keys

Create a `.env` file using the template:

```bash
cp .env.example .env
```

Add your credentials to `.env`:

```env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=optional_custom_url
```

### 5. Run the ML Pipeline

```bash
python -m src.fetch_data
python -m src.features
python -m src.models
```

### 6. Run Demo Script (Optional)

```bash
python demo_prediction.py
```

### 7. Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 8. Limitations

* Only 5 stocks are supported
* No fundamental or sentiment-based signals
* No portfolio-level backtesting or transaction cost modeling

---

## 9. Future Work

* Add news/sentiment using RAG or NLP
* Support additional stocks and timeframes
* Use advanced models (e.g., LSTM, Transformer)
* Add FastAPI backend and full-stack deployment

---

## 👤 Contributor – Vishak Shashikumar

- Worked on API-driven data ingestion using the Alpha Vantage stock market API.
- Assisted in feature engineering and preprocessing for stock trend prediction.
- Contributed to machine learning model evaluation and performance analysis.
- Supported explainable AI (XAI) components to improve transparency of model predictions.
- Improved project documentation and GitHub structure for portfolio readiness.
