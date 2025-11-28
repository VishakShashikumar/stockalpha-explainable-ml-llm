import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier  # New: XGBoost

from .config import PROCESSED_DIR


def load_panel() -> pd.DataFrame:
    """
    Load the feature panel from data/processed.
    """
    path = PROCESSED_DIR / "panel_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Feature panel not found at {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def get_feature_and_target(df: pd.DataFrame):
    """
    Select feature columns and target.
    """
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "return",
        "ma_5",
        "ma_10",
        "vol_5",
        "vol_10",
        "ret_lag1",
        "ret_lag2",
        "ret_lag3",
    ]
    X = df[feature_cols].copy()
    y = df["y_up"].copy()
    return X, y, feature_cols


def train_models():
    """
    Train three models on the panel:
    1) Logistic Regression (baseline)
    2) Random Forest (tree ensemble)
    3) XGBoost (gradient boosting)

    Use a 70% / 30% chronological split for train / test.
    Save whichever model gets the highest test accuracy as the "production" model.
    """
    df = load_panel()
    # Drop rows with NaNs from rolling features
    df = df.dropna().reset_index(drop=True)

    # Sort by time and symbol to respect chronology
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    n = len(df)
    if n < 50:
        print(f"Warning: only {n} rows after cleaning; results may be unstable.")

    # Chronological split: first 70% = train, last 30% = test
    split_idx = int(n * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train, y_train, feature_cols = get_feature_and_target(train_df)
    X_test, y_test, _ = get_feature_and_target(test_df)

    results = {}

    # 1) Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print("=== Logistic Regression ===")
    print("Accuracy:", acc_lr)
    print(classification_report(y_test, y_pred_lr))
    results["log_reg"] = (acc_lr, log_reg)

    # 2) Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("=== Random Forest ===")
    print("Accuracy:", acc_rf)
    print(classification_report(y_test, y_pred_rf))
    results["random_forest"] = (acc_rf, rf)

    # 3) XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",     # avoids label encoder warnings
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print("=== XGBoost ===")
    print("Accuracy:", acc_xgb)
    print(classification_report(y_test, y_pred_xgb))
    results["xgboost"] = (acc_xgb, xgb)

    # Pick the best model by accuracy
    best_name, (best_acc, best_model) = max(results.items(), key=lambda kv: kv[1][0])

    print("=== Model Comparison ===")
    for name, (acc, _) in results.items():
        print(f"{name:15s} -> accuracy: {acc:.4f}")
    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")

    # Save the best model + feature list
    model_path = PROCESSED_DIR / "model_best.pkl"
    feat_path = PROCESSED_DIR / "feature_cols.pkl"
    meta_path = PROCESSED_DIR / "model_meta.txt"

    joblib.dump(best_model, model_path)
    joblib.dump(feature_cols, feat_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"best_model={best_name}\n")
        f.write(f"test_accuracy={best_acc:.4f}\n")

    print(f"Saved best model ({best_name}) to {model_path}")
    print(f"Saved feature columns to {feat_path}")
    print(f"Saved model metadata to {meta_path}")


def load_model_and_features():
    """
    Load the best model (whatever won during training) and the feature list.
    """
    model_path = PROCESSED_DIR / "model_best.pkl"
    feat_path = PROCESSED_DIR / "feature_cols.pkl"
    model = joblib.load(model_path)
    feature_cols = joblib.load(feat_path)
    return model, feature_cols


if __name__ == "__main__":
    train_models()
