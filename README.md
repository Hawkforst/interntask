# Stáž úkol 

## Scénář:

Dostal/a jste 2 roky denních dat o prodejích jednoho produktu. Prodeje závisí na dni, ceně, kterou jsme účtovali, na tom, zda běžela promo akce, a na dni v týdnu. Vaším úkolem je porozumět struktuře dat, postavit model, který predikuje prodeje z dostupných featur, a vytvořit předpověď na dalších 90 dní - je na vás, zda k úkolu přistoupíte spíše koncepčně nebo prakticky a programátorsky. 

Chceme ale v každém případě vidět alespoň část vámi psaného kódu, ideálně vč. enkapsulace do class(es) a využití nějaké machine learning knihovny.

Pokuste se vzít zadaná data a zamyslet se nad úkolem. Cílem není mít perfektní kód, ale je to způsob, jakým my můžeme zhodnotit Váš způsob přemýšlení, přístup k zadaným úkolům, schopnost pracovat (částečně) samostně, apod. 

Přesnou interpretaci úkolu necháme na vás, může to být např., ale ne nutně:
 - EDA
 - feature engineering
 - vizualizace dat
 - tvorba modelu
 - porovnávání více modelů
 - teoretický návrh metodiky
 - praktická implementace
 - apod.


## Toolchain
Programujte v Pythonu, můžete volně používat zdroje na internetu. Částečně smíte používat LLMka (a je to do rozumné míry i žádoucí, abyste se zbytečně neznevýhodnili), nicméně byste měli všemu, co nám submitnete, detailně rozumět, včetně jakéhokoliv AI generated kódu. Ideálně AI kód i označte, například tagem <LLM> /code </LLM> 


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
