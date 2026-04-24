"""
Generator for the intern take-home task.

Produces three files in data/:
    train.csv            - features + sales (given to candidate)
    test.csv             - features only   (given to candidate)
    test_with_truth.csv  - features + true sales (interviewer only)

The underlying data-generating process is intentionally layered so that a
naive linear regression on raw features will get a mediocre score, while a
candidate who does EDA and engineers sensible features (day-of-week effect,
log-price, promo x weekend interaction, yearly seasonality) will do
meaningfully better. None of the structure is exotic, but it is not all
obvious from looking at one scatter plot.

Tune the constants below if you want to make the task easier or harder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_TRAIN = 730   # 2 years of daily data
N_TEST = 90     # 3 months forecast horizon
START_DATE = "2024-01-01"

OUT_DIR = Path(__file__).parent / "data"


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    n = N_TRAIN + N_TEST

    t = np.arange(n)
    dow = t % 7
    is_weekend = ((dow == 5) | (dow == 6)).astype(float)

    # Features the candidate sees.
    price = rng.uniform(5.0, 20.0, size=n)
    promo = rng.binomial(1, 0.15, size=n).astype(float)

    # ---- Hidden structure ----

    # Slow linear trend.
    trend = 0.05 * t

    # Weekly effect: weekends meaningfully higher, Tue/Wed slightly lower.
    # Captured cleanly by dow dummies; a raw linear model on `dow` as an int
    # will fit this badly.
    weekly_by_dow = np.array([0.0, -5.0, -10.0, 0.0, 15.0, 35.0, 30.0])
    weekly = weekly_by_dow[dow]

    # Yearly seasonality (one full cycle per 365.25 days).
    yearly = 40.0 * np.sin(2 * np.pi * t / 365.25)

    # Nonlinear price effect: log elasticity around a reference price of 10.
    # A linear model on raw `price` will underfit at the tails.
    price_effect = -50.0 * np.log(price / 10.0)

    # Promo uplift, boosted on weekends (interaction).
    promo_effect = 40.0 * promo * (1.0 + 0.8 * is_weekend)

    # Gaussian noise.
    noise = rng.normal(0.0, 15.0, size=n)

    base = 200.0
    sales = base + trend + weekly + yearly + price_effect + promo_effect + noise
    sales = np.clip(sales, 0.0, None)

    dates = pd.date_range(START_DATE, periods=n, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "t": t,
            "dow": dow,
            "price": price.round(2),
            "promo": promo.astype(int),
            "sales": sales.round(2),
        }
    )

    train = df.iloc[:N_TRAIN].reset_index(drop=True)
    test_full = df.iloc[N_TRAIN:].reset_index(drop=True)
    return train, test_full


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test_full = generate()
    test = test_full.drop(columns=["sales"])

    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    test_full.to_csv(OUT_DIR / "test_with_truth.csv", index=False)

    print(f"Wrote {len(train)} train rows and {len(test)} test rows to {OUT_DIR}/")


if __name__ == "__main__":
    main()
