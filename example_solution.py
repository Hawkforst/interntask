"""
Example solution for the intern take-home task.

This is the kind of answer we would be happy to see from a strong candidate.
It is deliberately not the absolute best possible model — it is a clean,
well-reasoned baseline-plus-one-iteration solution with a short narrative.

Pipeline:
    1. Load and peek at the data.
    2. Quick EDA (prints only — a candidate would add plots).
    3. Baseline: linear regression on raw features.
    4. Improved model: dow dummies, log-price, promo x weekend interaction,
       Fourier features for yearly seasonality, linear trend.
    5. Cross-validate both on a time-based split.
    6. Refit the improved model on all train data, predict test.
    7. (Interviewer only) score against the hidden truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    return train, test


# ---------------------------------------------------------------------------
# 2. Quick EDA summary (a real submission would include plots)
# ---------------------------------------------------------------------------

def eda(train: pd.DataFrame) -> None:
    print("=== Shape ===")
    print(train.shape)
    print("\n=== Describe ===")
    print(train.describe())
    print("\n=== Mean sales by day of week ===")
    print(train.groupby("dow")["sales"].mean().round(1))
    print("\n=== Mean sales by promo ===")
    print(train.groupby("promo")["sales"].mean().round(1))
    print("\n=== Correlation of raw numeric features with sales ===")
    print(train[["t", "dow", "price", "promo", "sales"]].corr()["sales"].round(3))
    # Observations a candidate should write up:
    # - Clear weekly pattern: Sat/Sun much higher than mid-week.
    # - Promo lifts sales on average; effect appears larger on weekends.
    # - `price` correlates negatively with sales, but a scatter plot shows
    #   the relationship is curved (log-shaped), not linear.
    # - Slow upward drift over two years suggests a mild trend.
    # - A residual plot after removing weekly + trend shows a yearly wave.


# ---------------------------------------------------------------------------
# 3 & 4. Feature engineering
# ---------------------------------------------------------------------------

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    # Linear trend.
    X["t"] = df["t"]

    # Day-of-week dummies (drop Monday as reference).
    for d in range(1, 7):
        X[f"dow_{d}"] = (df["dow"] == d).astype(int)

    # Nonlinear price effect.
    X["log_price"] = np.log(df["price"])

    # Promo main effect and weekend interaction.
    is_weekend = df["dow"].isin([5, 6]).astype(int)
    X["promo"] = df["promo"]
    X["promo_x_weekend"] = df["promo"] * is_weekend

    # Yearly seasonality via one pair of Fourier terms.
    angle = 2 * np.pi * df["t"] / 365.25
    X["yr_sin"] = np.sin(angle)
    X["yr_cos"] = np.cos(angle)

    return X


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["t", "dow", "price", "promo"]].copy()


# ---------------------------------------------------------------------------
# 5. Time-based validation
# ---------------------------------------------------------------------------

def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def time_split_eval(train: pd.DataFrame, feature_fn, label: str) -> dict[str, float]:
    # Use the last 90 days of train as a validation window — same horizon as test.
    cutoff = len(train) - 90
    tr, va = train.iloc[:cutoff], train.iloc[cutoff:]

    X_tr, X_va = feature_fn(tr), feature_fn(va)
    model = Ridge(alpha=1.0).fit(X_tr, tr["sales"])
    pred = model.predict(X_va)

    s = score(va["sales"].to_numpy(), pred)
    print(f"{label:30s}  rmse={s['rmse']:6.2f}  mape={s['mape']:.3f}")
    return s


# ---------------------------------------------------------------------------
# 6. Fit on full train, predict test
# ---------------------------------------------------------------------------

def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    X_tr = make_features(train)
    X_te = make_features(test)
    model = Ridge(alpha=1.0).fit(X_tr, train["sales"])
    pred = model.predict(X_te)

    out = pd.DataFrame({"date": test["date"], "sales_pred": pred.round(2)})
    out.to_csv(Path(__file__).parent / "predictions.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 7. Interviewer-only: score against hidden truth if available.
# ---------------------------------------------------------------------------

def grade_against_truth(predictions: pd.DataFrame) -> None:
    truth_path = DATA_DIR / "test_with_truth.csv"
    if not truth_path.exists():
        return
    truth = pd.read_csv(truth_path, parse_dates=["date"])
    merged = predictions.merge(truth[["date", "sales"]], on="date")
    s = score(merged["sales"].to_numpy(), merged["sales_pred"].to_numpy())
    print(f"\n[interviewer] test  rmse={s['rmse']:6.2f}  mape={s['mape']:.3f}")


def main() -> None:
    train, test = load()

    eda(train)

    print("\n=== Time-based validation (last 90 days of train) ===")
    time_split_eval(train, raw_features, "baseline (raw features)")
    time_split_eval(train, make_features, "engineered features")

    preds = fit_and_predict(train, test)
    print(f"\nWrote {len(preds)} predictions to predictions.csv")

    grade_against_truth(preds)


if __name__ == "__main__":
    main()
