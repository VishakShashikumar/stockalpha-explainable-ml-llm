import os
import sys

# Ensure the project root (parent of app/) is on the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR
from src.models import load_model_and_features
from src.llm_explain import make_prediction, explain_with_llm

# ---------- Load data and model ----------
PANEL_PATH = PROCESSED_DIR / "panel_features.csv"
df_panel = pd.read_csv(PANEL_PATH, parse_dates=["date"])

model, feature_cols = load_model_and_features()

# ---------- Streamlit page config ----------
st.set_page_config(
    page_title="StockInsight AI Dashboard",
    layout="centered",
)

st.title("📈 StockInsight AI")
st.write(
    "API-driven stock movement prediction using Machine Learning (ML) "
    "and Large Language Model (LLM) explanations.\n\n"
    "_This is an educational demo of model outputs, not investment advice._"
)

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")

symbols = sorted(df_panel["symbol"].unique().tolist())
if not symbols:
    st.error("No symbols found in the feature panel. Did you run src.features and src.models?")
    st.stop()

symbol = st.sidebar.selectbox("Choose stock symbol", symbols, index=0)

dates_for_symbol = (
    df_panel[df_panel["symbol"] == symbol]["date"]
    .sort_values()
    .dt.date
    .unique()
)
latest_date = dates_for_symbol[-1]

date_choice = st.sidebar.selectbox(
    "Choose date (latest at bottom)",
    dates_for_symbol,
    index=len(dates_for_symbol) - 1,
)

st.write(f"### Selected: `{symbol}` on `{date_choice}`")

# ---------- Select rows for that symbol ----------
df_sym = df_panel[df_panel["symbol"] == symbol].copy()
df_sym = df_sym[df_sym["date"].dt.date <= date_choice]

if df_sym.empty:
    st.error("No data for this symbol/date combination.")
    st.stop()

# Sort by date for time-series plots
df_sym = df_sym.sort_values("date")

# For the model, drop rows where rolling features are still NaN
df_sym_model = df_sym.dropna(subset=feature_cols).copy()
if df_sym_model.empty:
    st.error("Not enough non-missing feature rows for this symbol/date selection.")
    st.stop()

# Last valid row is the one we use for the current-day prediction
row = df_sym_model.iloc[-1]
feature_row = row[feature_cols]

# ---------- Pre-compute model probabilities over history (clean subset) ----------
hist_X = df_sym_model[feature_cols]
df_sym_model["prob_up"] = model.predict_proba(hist_X)[:, 1]  # probability of class 1 (UP)

# Build columns for evaluation: next-day move and predicted direction
df_sym_model["next_return"] = df_sym_model["return"].shift(-1)
df_eval = df_sym_model.dropna(subset=["next_return"]).copy()
df_eval["actual_up"] = (df_eval["next_return"] > 0).astype(int)
df_eval["pred_up"] = (df_eval["prob_up"] >= 0.5).astype(int)

# ---------- Prediction button ----------
if st.button("Run prediction"):
    with st.spinner("Running model and generating explanation..."):
        # --- Single-symbol prediction & explanation ---
        pred = make_prediction(
            model=model,
            feature_row=feature_row,
            symbol=symbol,
            date_str=row["date"].strftime("%Y-%m-%d"),
        )

        st.subheader("Model Output")
        st.metric(
            label="Probability of UP (next day)",
            value=f"{pred.prob_up:.2%}",
        )
        st.write(f"Predicted label: **{'UP' if pred.label_up == 1 else 'NOT UP'}**")

        st.subheader("LLM Explanation")
        explanation = explain_with_llm(pred)
        st.write(explanation)

        st.subheader("Feature Snapshot (key values used by model)")
        st.json(pred.features_summary)

        # ---------- Visualization 1: Historical price ----------
        st.subheader("Historical Price (Adjusted Close)")
        price_df = df_sym.set_index("date")[["adjusted_close"]]
        price_df.columns = ["Adjusted Close"]
        st.line_chart(price_df)

        # ---------- Visualization 2: Recent prediction vs actual ----------
        st.subheader("Recent Direction: Model vs Actual (Last 30 trading days)")

        if df_eval.empty:
            st.info("Not enough clean rows to evaluate predictions yet.")
        else:
            # Compute accuracy over this evaluation window
            accuracy = (df_eval["actual_up"] == df_eval["pred_up"]).mean()
            st.write(f"Directional accuracy over this window: **{accuracy:.2%}**")

            # Use last 30 days for a readable chart
            df_plot = df_eval.set_index("date")[["actual_up", "pred_up"]].tail(30)
            df_plot.columns = ["Actual UP (1=yes)", "Predicted UP (1=yes)"]

            st.bar_chart(df_plot)

        # ---------- NEW: Cross-sectional ranking (all symbols on this date) ----------
        st.subheader("Cross-Sectional Signal: Top Stocks by Probability of UP")

        snapshot_rows = []
        for sym in symbols:
            df_s = df_panel[df_panel["symbol"] == sym].copy()
            df_s = df_s[df_s["date"].dt.date <= date_choice]
            df_s = df_s.sort_values("date")
            df_s = df_s.dropna(subset=feature_cols)

            if df_s.empty:
                continue

            last_row = df_s.iloc[-1]
            X_sym = last_row[feature_cols].to_frame().T
            prob_sym = model.predict_proba(X_sym)[0][1]

            snapshot_rows.append(
                {
                    "Symbol": sym,
                    "Prob. of UP (%)": round(prob_sym * 100.0, 2),
                    "Latest Adjusted Close": round(last_row["adjusted_close"], 2),
                }
            )

        if not snapshot_rows:
            st.info("Not enough clean data across symbols for this date.")
        else:
            snap_df = pd.DataFrame(snapshot_rows)
            snap_df = snap_df.sort_values("Prob. of UP (%)", ascending=False)

            st.caption(
                "Ranking of all symbols based on the model's probability that "
                "the next day's close will be higher than today's close."
            )
            st.dataframe(snap_df.head(5), use_container_width=True)

else:
    st.info("Click **Run prediction** to see the model output, explanation, and charts.")
