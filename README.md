# Data Scientist Intern — Take-Home Task

## Scenario

You are given 2 years of daily sales data for a single product. Sales depend
on the day, the price we charged, whether a promotion was running, and the
day of the week. Your job is to understand the structure of the data, build
a model that predicts sales from the available features, and forecast the
next 90 days.

This task is designed to take roughly **one focused working day**. Don't
over-engineer. We care about how you think, not about squeezing the last
0.1% of accuracy out of a model.

## Data

Two CSV files in `data/`:

- `train.csv` — 730 rows, columns: `date`, `t`, `dow`, `price`, `promo`, `sales`
- `test.csv`  — 90 rows, same columns **except `sales` is withheld**

Column meanings:

| column  | description                                              |
|---------|----------------------------------------------------------|
| `date`  | calendar date                                            |
| `t`     | day index (0, 1, 2, …), useful as a continuous time axis |
| `dow`   | day of week, 0 = Monday … 6 = Sunday                     |
| `price` | price we charged that day (continuous, in currency units)|
| `promo` | 1 if a promotion was running that day, else 0            |
| `sales` | units sold that day (target)                             |

## What we want from you

1. **Explore the data.** Describe what structure you find. Plots are welcome.
2. **Build a model** that predicts `sales` from the features.
3. **Submit predictions** for every row in `test.csv` as `predictions.csv`
   with columns `date,sales_pred`.
4. **Write a short report** (1–2 pages, markdown or PDF):
   - What structure did you find in the data?
   - What models did you try? What worked, what didn't, and why?
   - How did you validate your choices?
   - What would you do with more time?

## How we evaluate

- **Understanding of the data** — the write-up matters at least as much as the number.
- **Predictive performance** on the held-out test set (we'll use RMSE and MAPE).
- **Code quality** — readable, reproducible, reasonable structure.
- **Honest reasoning** — we'd rather see a simple model you understand than a
  complex one you don't.

## Stack

Python. Use any libraries you like (pandas, scikit-learn, statsmodels,
Prophet, PyTorch — your call). Include a `requirements.txt` or a note of
what you used so we can reproduce.

## Deliverables

A zip / repo containing:

- your code
- `predictions.csv`
- the write-up
- a note on how to run it

Good luck, and have fun with it.
