import time
import requests
import pandas as pd

from .config import ALPHA_VANTAGE_API_KEY, RAW_DIR, STOCK_SYMBOLS

BASE_URL = "https://www.alphavantage.co/query"


def fetch_daily(symbol: str) -> pd.DataFrame:
    """
    Fetch daily time series for a symbol from Alpha Vantage using the FREE endpoint.
    We use 'outputsize=compact' (last ~100 days), which is allowed on the free tier.
    """
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY is not set. Put it in your .env file.")

    params = {
        "function": "TIME_SERIES_DAILY",   # FREE endpoint
        "symbol": symbol,
        "outputsize": "compact",           # FREE: last ~100 days (no premium 'full')
        "datatype": "json",
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Free TIME_SERIES_DAILY still returns "Time Series (Daily)"
    if "Time Series (Daily)" not in data:
        raise RuntimeError(f"Unexpected response for {symbol}: {data}")

    ts = data["Time Series (Daily)"]

    # Convert dict to DataFrame
    df = (
        pd.DataFrame.from_dict(ts, orient="index")
        .rename_axis("date")
        .reset_index()
    )

    # For TIME_SERIES_DAILY the fields are:
    # 1. open, 2. high, 3. low, 4. close, 5. volume
    df = df.rename(
        columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    )

    df["date"] = pd.to_datetime(df["date"])

    float_cols = ["open", "high", "low", "close"]
    for c in float_cols:
        df[c] = df[c].astype(float)

    df["volume"] = df["volume"].astype(int)

    # We don't have adjusted close in free endpoint, so use close as proxy
    df["adjusted_close"] = df["close"]

    # Placeholder columns for compatibility with rest of the pipeline
    df["dividend_amount"] = 0.0
    df["split_coefficient"] = 1.0

    df = df.sort_values("date").reset_index(drop=True)
    df["symbol"] = symbol

    return df


def download_all_symbols(symbols=None, sleep_seconds: int = 15):
    """
    Download daily data for all symbols and save to data/raw/{symbol}_daily.csv.
    Alpha Vantage free tier allows 5 calls/minute, so we pause between calls.
    """
    if symbols is None:
        symbols = STOCK_SYMBOLS

    for i, sym in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] Fetching {sym}...")
        df = fetch_daily(sym)
        out_path = RAW_DIR / f"{sym}_daily.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")

        # Sleep between symbols to respect rate limits
        if i < len(symbols):
            print(f"Sleeping {sleep_seconds} seconds to respect rate limits...")
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    download_all_symbols()
