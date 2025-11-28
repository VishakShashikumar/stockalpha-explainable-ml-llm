import pandas as pd

from .config import RAW_DIR, PROCESSED_DIR, STOCK_SYMBOLS


def load_symbol_raw(symbol: str) -> pd.DataFrame:
    """
    Load raw daily CSV for one symbol from data/raw.
    """
    path = RAW_DIR / f"{symbol}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data for {symbol} not found at {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def add_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple technical features and a next-day up/down target.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Daily return
    df["return"] = df["adjusted_close"].pct_change()

    # Moving averages
    df["ma_5"] = df["adjusted_close"].rolling(window=5).mean()
    df["ma_10"] = df["adjusted_close"].rolling(window=10).mean()

    # Volatility (rolling std of returns)
    df["vol_5"] = df["return"].rolling(window=5).std()
    df["vol_10"] = df["return"].rolling(window=10).std()

    # Lagged returns
    df["ret_lag1"] = df["return"].shift(1)
    df["ret_lag2"] = df["return"].shift(2)
    df["ret_lag3"] = df["return"].shift(3)

    # Next-day return and binary target
    df["return_next"] = df["adjusted_close"].shift(-1) / df["adjusted_close"] - 1
    df["y_up"] = (df["return_next"] > 0).astype(int)

    # Remove last row (no next-day data)
    df = df.iloc[:-1, :]

    return df


def build_panel(symbols=None) -> pd.DataFrame:
    """
    Build a combined panel of all symbols with features + target.
    No hard date filters, works even with ~100 days of data per symbol.
    """
    if symbols is None:
        symbols = STOCK_SYMBOLS

    all_dfs = []
    for sym in symbols:
        df_raw = load_symbol_raw(sym)
        df_feat = add_features_and_target(df_raw)
        all_dfs.append(df_feat)

    panel = pd.concat(all_dfs, ignore_index=True)

    out_path = PROCESSED_DIR / "panel_features.csv"
    panel.to_csv(out_path, index=False)
    print(f"Saved feature panel with {len(panel)} rows to {out_path}")
    return panel


if __name__ == "__main__":
    build_panel()
