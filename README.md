# StockAlpha – Explainable Stock Signals (ML + LLM)

<p align="center">
  API-driven stock movement signals using the Alpha Vantage stock market API, Machine Learning, and GenAI (LLM explanations), visualized in a Streamlit dashboard.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML%20models-orange?logo=scikitlearn&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-numerical%20computing-informational?logo=numpy&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-interactive%20dashboard-ff4b4b?logo=streamlit&logoColor=white">
  <img alt="GenAI LLM" src="https://img.shields.io/badge/GenAI-LLM%20explanations-8A2BE2">
  <img alt="Editor" src="https://img.shields.io/badge/Editor-VS%20Code-007ACC?logo=visualstudiocode&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> Educational prototype for a graduate **Finance Information Processing** course.  
> Focus: stock direction prediction + explainability (not trading advice).

---

## 1. Problem Statement

Given recent daily OHLCV data for large-cap U.S. stocks, predict whether **tomorrow’s closing price will be higher than today’s** (UP vs NOT UP), and generate a **human-readable explanation** of the model’s output.

Constraints:

- Data must come from a **real financial API** (no Kaggle, no Wikipedia).
- Project must provide **working code, a demo, and documentation**.
- Predictions must be **interpretable** to non-technical users.

---

## 2. High-Level Solution

1. **Data Ingestion (API)**  
   - Pull daily adjusted OHLCV time series for symbols (e.g. `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`) from the **Alpha Vantage Time Series Daily Adjusted** endpoint.

2. **Feature Engineering**  
   - Build a **panel dataset** with:
     - Moving averages (short / long)  
     - Daily and lagged returns  
     - Rolling volatility measures  

3. **Machine Learning Models**

   - Train and evaluate:
     - **Logistic Regression**  
     - **Random Forest Classifier**  
     - **XGBoost Classifier**  
   - Use test-set **directional accuracy** to select the deployed model.

4. **GenAI Explanation Layer (LLM)**  
   - Use a **Large Language Model** to turn the numeric prediction + features into a **plain-English explanation** with risk disclaimers and guardrails.

5. **Interactive Dashboard (Streamlit)**  
   - Provide a UI to:
     - Select symbol + date  
     - Run prediction and explanation  
     - Visualize historical price and recent directional accuracy  
     - Rank all stocks by probability of UP (cross-sectional signal)

---

## 3. Tech Stack

**Languages & Runtime**

- Python 3.13  
- Conda environment (`finance_stock_env`)

**Core Libraries**

- Data: `pandas`, `numpy`  
- ML: `scikit-learn`, `xgboost`  
- API: `requests`  
- App: `streamlit`  
- Config / Env: `python-dotenv`  
- GenAI / LLM: `openai` (or OpenAI-compatible client / local proxy)

**External Services**

- **Alpha Vantage** – stock market API (Time Series Daily Adjusted)  
- **OpenAI / compatible endpoint** – for LLM explanations (optional; project has a safe fallback explanation)

---

## 4. Project Structure

```text
.
├─ app/
│   └─ dashboard.py          # Streamlit UI for predictions & visualizations
├─ src/
│   ├─ config.py             # Paths, symbols, environment variables
│   ├─ fetch_data.py         # Calls Alpha Vantage API, saves raw CSV files
│   ├─ features.py           # Feature engineering (MAs, returns, volatility)
│   ├─ models.py             # Train & evaluate Logistic / RF / XGBoost
│   ├─ llm_explain.py        # LLM-based explanation with guardrails
│   └─ __init__.py
├─ data/
│   ├─ raw/                  # Raw API data (gitignored)
│   └─ processed/            # Engineered panel features (gitignored)
├─ demo_prediction.py        # CLI demo: one prediction + explanation
├─ .env.example              # Template for API keys (no secrets)
├─ .gitignore
└─ README.md

5. Features
5.1 Stock Signal Modeling

Universe: large-cap U.S. equities (AAPL, MSFT, AMZN, GOOGL, META).

Features per (symbol, date):

short_ma, long_ma – moving averages of adjusted close

return, ret_lag1, ret_lag2 – daily & lagged returns

short_vol, long_vol – rolling volatility

Target:

up_next_day = 1 if next close > current close, else 0.

Models:

Logistic Regression

Random Forest

XGBoost

Metric:

Directional accuracy on a held-out test set

Typical values around 53–55%, realistic for noisy financial series.

5.2 Explainable AI (XAI) with LLM

Inputs to LLM:

Symbol, date

Predicted probability of UP

Binary label (UP / NOT UP)

Key feature values (MAs, returns, volatility)

LLM prompt focuses on:

Short vs long MA (trend)

Recent returns (momentum / mean reversion)

Short vs long volatility (stability vs choppiness)

Explicit disclaimer: weak statistical signal, not investment advice.

Fallback:

If LLM call fails, a rule-based explanation string is generated.

5.3 Streamlit Dashboard

Inputs

Stock symbol selector

Date selector (latest at bottom)

Outputs

Probability of UP and predicted label

LLM-generated explanation

Feature snapshot (JSON view)

Historical price chart (Adjusted Close)

Recent Direction: Model vs Actual bar chart (last 30 trading days)

Cross-Sectional Signal Table:

All symbols ranked by probability of UP

Latest adjusted close for each symbol

6. Setup & Usage
6.1 Clone the repository
git clone https://github.com/ritwikm14/stockalpha-explainable-ml-llm.git
cd stockalpha-explainable-ml-llm

6.2 Create and activate environment
conda create -n finance_stock_env python=3.13 -y
conda activate finance_stock_env

Install dependencies (adapt once you export requirements.txt):
pip install pandas numpy scikit-learn xgboost streamlit python-dotenv requests openai

6.3 Configure environment variables

Create a .env file in the project root (copy from .env.example) and fill in:
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
OPENAI_API_KEY=your_openai_or_proxy_key_here
OPENAI_BASE_URL=optional_custom_base_url_if_using_proxy

6.4 Fetch data, build features, train models
python -m src.fetch_data
python -m src.features
python -m src.models

6.5 Run CLI demo (optional)
python demo_prediction.py

6.6 Launch Streamlit dashboard
streamlit run app/dashboard.py

Then open http://localhost:8501 in your browser.

7. Limitations & Extensions

Current limitations

Small symbol universe and short history window.

Only OHLCV features; no fundamentals, macro data, or news/sentiment.

No transaction cost modeling or portfolio optimization.

Potential extensions

Expand universe and history; test cross-validation by regime.

Add news / sentiment signals using NLP and RAG.

Replace the baseline with more advanced time-series models.

Wrap the core pipeline in FastAPI and serve via a production frontend.

8. Status

✅ End-to-end pipeline working (API → features → models → dashboard).

✅ GenAI explanation layer integrated with safe fallbacks.

✅ Ready as a portfolio project for ML / GenAI / Quant / FinTech roles.