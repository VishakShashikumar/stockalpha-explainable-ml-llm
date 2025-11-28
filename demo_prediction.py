import pandas as pd

from src.config import PROCESSED_DIR
from src.models import load_model_and_features
from src.llm_explain import make_prediction, explain_with_llm


def main(symbol: str = "AAPL"):
    # Load the full feature panel
    panel_path = PROCESSED_DIR / "panel_features.csv"
    df = pd.read_csv(panel_path, parse_dates=["date"])

    # Filter for the chosen symbol and sort by date
    df_sym = df[df["symbol"] == symbol].dropna().sort_values("date")
    if df_sym.empty:
        raise ValueError(f"No data for symbol {symbol}")

    # Use the most recent row as "today"
    last_row = df_sym.iloc[-1]

    # Load best model + feature columns
    model, feature_cols = load_model_and_features()
    feature_row = last_row[feature_cols]

    # Run prediction
    pred = make_prediction(
        model=model,
        feature_row=feature_row,
        symbol=symbol,
        date_str=last_row["date"].strftime("%Y-%m-%d"),
    )

    print(f"Symbol: {pred.symbol}")
    print(f"Date: {pred.date}")
    print(f"Predicted probability of UP: {pred.prob_up:.2%}")
    print(f"Predicted label: {'UP' if pred.label_up == 1 else 'NOT UP'}")

    # Get LLM explanation
    explanation = explain_with_llm(pred)
    print("\nLLM Explanation:")
    print(explanation)


if __name__ == "__main__":
    main()
