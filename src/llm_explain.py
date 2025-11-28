from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from openai import OpenAI, APIConnectionError

from .config import OPENAI_API_KEY


@dataclass
class PredictionResult:
    """
    Container for model prediction + key features,
    so we can pass it cleanly to the LLM or fallback generator.
    """
    symbol: str
    date: str
    prob_up: float
    label_up: int
    features_summary: Dict[str, Any]


def make_prediction(model, feature_row, symbol: str, date_str: str) -> PredictionResult:
    """
    Run the trained model on a single feature row and return
    a structured PredictionResult object.
    """
    # Ensure 2D shape for scikit-learn / XGBoost style models
    if getattr(feature_row, "ndim", 1) == 1:
        X = feature_row.to_frame().T
    else:
        X = feature_row

    # Probability that next day is UP (class 1)
    proba = model.predict_proba(X)[0][1]
    prob_up = float(np.clip(proba, 0.0, 1.0))
    label_up = int(prob_up >= 0.5)

    feat = feature_row.to_dict()
    summary = {
        "latest_close": feat.get("adjusted_close"),
        "short_ma": feat.get("ma_5"),
        "long_ma": feat.get("ma_10"),
        "short_vol": feat.get("vol_5"),
        "long_vol": feat.get("vol_10"),
        "last_return": feat.get("return"),
        "ret_lag1": feat.get("ret_lag1"),
        "ret_lag2": feat.get("ret_lag2"),
    }

    return PredictionResult(
        symbol=symbol,
        date=date_str,
        prob_up=prob_up,
        label_up=label_up,
        features_summary=summary,
    )


def build_prompt(pred: PredictionResult) -> str:
    """
    Build a careful, guardrailed prompt for the LLM.
    """
    direction = "up" if pred.label_up == 1 else "not up (flat or down)"

    fs = pred.features_summary
    text_summary = (
        f"Latest adjusted close: {fs.get('latest_close')}\n"
        f"5-day moving average: {fs.get('short_ma')}\n"
        f"10-day moving average: {fs.get('long_ma')}\n"
        f"5-day volatility (approx): {fs.get('short_vol')}\n"
        f"10-day volatility (approx): {fs.get('long_vol')}\n"
        f"Today return: {fs.get('last_return')}\n"
        f"Return lag1: {fs.get('ret_lag1')}\n"
        f"Return lag2: {fs.get('ret_lag2')}\n"
    )

    prompt = f"""
You are a cautious financial education assistant.

You are given the output of a statistical model that looks only at past prices and simple technical indicators
for a stock. The model is NOT perfect and this information is for educational illustration only.

STRICT RULES (GUARDRAILS):
- DO NOT give investment advice.
- DO NOT say 'buy', 'sell', or 'hold', or recommend trading any stock.
- DO NOT claim certainty or guarantee future moves.
- Speak in neutral, educational language.
- Always include a short disclaimer at the end:
  "This is an educational explanation of model outputs, not investment advice."

Here is the context:

- Stock symbol: {pred.symbol}
- Date of prediction: {pred.date}
- Model prediction: price is more likely to move {direction} on the next trading day.
- Model probability that price moves up: {pred.prob_up:.2%}

Technical summary (from time-series features):
{text_summary}

TASK:
In 3–5 sentences, explain in simple language why the model might think the stock is more likely to move {direction}.
Mention short-term trend (moving averages), recent returns, and volatility if they are relevant.
Be honest that the signal can be noisy, and end with the required disclaimer.
"""
    return prompt.strip()


def _fallback_explanation(pred: PredictionResult) -> str:
    """
    If the LLM is unavailable (no server / bad connection), return
    a safe explanation that actually uses the features in a simple,
    rule-based way.
    """
    fs = pred.features_summary
    prob_pct = f"{pred.prob_up:.2%}"

    # 1) Direction / confidence from probability
    if pred.prob_up > 0.55:
        direction_text = "slightly more likely to move up than down"
        confidence_text = "a bit stronger than a pure 50-50 coin flip"
    elif pred.prob_up < 0.45:
        direction_text = "slightly more likely to move down or stay flat"
        confidence_text = "a bit stronger than a pure 50-50 coin flip"
    else:
        direction_text = "very close to a 50-50 outcome (up vs down/flat)"
        confidence_text = "very weak and close to random guessing"

    # 2) Trend based on moving averages
    short_ma = fs.get("short_ma")
    long_ma = fs.get("long_ma")
    if short_ma is not None and long_ma is not None:
        if short_ma > long_ma:
            trend_text = (
                "The short-term moving average is above the longer-term average, "
                "which usually indicates a mild upward short-term trend."
            )
        elif short_ma < long_ma:
            trend_text = (
                "The short-term moving average is below the longer-term average, "
                "which usually points to a mild downward or weakening trend."
            )
        else:
            trend_text = (
                "Short-term and longer-term moving averages are very similar, "
                "so there is no clear trend signal from them."
            )
    else:
        trend_text = (
            "Moving average information is limited here, so the trend signal is unclear."
        )

    # 3) Recent return
    last_ret = fs.get("last_return")
    if last_ret is not None:
        if last_ret > 0:
            ret_text = (
                "The most recent daily return was positive, so the stock has just moved up."
            )
        elif last_ret < 0:
            ret_text = (
                "The most recent daily return was negative, so the stock has just moved down."
            )
        else:
            ret_text = (
                "The most recent daily return was close to zero, so price was fairly unchanged."
            )
    else:
        ret_text = "Recent return information is limited."

    # 4) Volatility comparison
    short_vol = fs.get("short_vol")
    long_vol = fs.get("long_vol")
    if short_vol is not None and long_vol is not None:
        if short_vol > long_vol:
            vol_text = (
                "Short-term volatility is higher than longer-term volatility, "
                "which means recent price movements have been relatively choppy."
            )
        elif short_vol < long_vol:
            vol_text = (
                "Short-term volatility is lower than longer-term volatility, "
                "so recent price movements have been somewhat calmer than usual."
            )
        else:
            vol_text = (
                "Short-term and longer-term volatility are similar, "
                "so risk levels have been relatively stable."
            )
    else:
        vol_text = "Volatility information is limited."

    direction_sentence = (
        f"For stock {pred.symbol} on {pred.date}, the time-series model estimates a {prob_pct} "
        f"chance for the next day's closing price, which makes the outcome {direction_text}. "
        f"This signal is {confidence_text}."
    )

    explanation = (
        direction_sentence + " " +
        trend_text + " " +
        ret_text + " " +
        vol_text + " " +
        "Because the dataset is small and markets are noisy, these signals should be treated as rough statistics "
        "rather than reliable forecasts. This is an educational explanation of model outputs, not investment advice."
    )

    return explanation


def explain_with_llm(pred: PredictionResult, model_name: str = "gpt-4o-mini") -> str:
    """
    Try to call a local OpenAI-compatible server running on http://localhost:4000.

    - OPENAI_API_KEY can be your local proxy key (e.g., sk-local-...).
    - model_name must match a model that your local server exposes.
    - If the connection fails, we return a safe fallback explanation instead of crashing.
    """
    if not OPENAI_API_KEY:
        # No key at all – just return fallback
        return _fallback_explanation(pred)

    # Route to your local proxy instead of api.openai.com
    client = OpenAI(
        base_url="http://localhost:4000/v1",
        api_key=OPENAI_API_KEY,
    )

    prompt = build_prompt(pred)

    try:
        response = client.chat.completions.create(
            model=model_name,  # e.g. "gpt-4o-mini" or whatever your proxy serves
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful, neutral financial education assistant. "
                        "You MUST follow all safety rules given in the user message."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=250,
        )

        return response.choices[0].message.content.strip()

    except APIConnectionError as e:
        # Local server not running or unreachable
        print("LLM connection error, using fallback explanation instead:", e)
        return _fallback_explanation(pred)
    except Exception as e:
        # Any other LLM-related error – we also fall back safely
        print("LLM error, using fallback explanation instead:", e)
        return _fallback_explanation(pred)
