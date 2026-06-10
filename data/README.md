# XAUUSD Data — how to add more history

The repo uses CSV price data as its database. To add more/better data:

## 1. Export from MetaTrader 5
- **Tools → Options → Charts → "Max bars in chart" → Unlimited**
- Open a XAUUSD chart, press **Home** repeatedly to pull deep history from the broker
- Export the **finest timeframe you can get** (M1 or M5 preferred; M15 works)
  - Either: **View → Symbols → XAUUSD → Bars → Export** (GUI)
  - Or: the MT5 Python API (`mt5_data_fetcher.py`) with `copy_rates_range`

## 2. Drop the file(s) here
Put raw exports in **`data/raw/`**. The filename **must contain the timeframe**
(e.g. `XAUUSD_M5_2019_2026.csv`, `XAUUSD_M15_...csv`).

## 3. Run the ingest
```
python3 ingest_data.py
```
It auto-detects the format, writes the canonical `data/XAUUSD_<TF>.csv`, and
resamples a clean `XAUUSD_H1.csv` from the finest intraday file.

## Canonical schema (what the backtester loads)
```
datetime,open,high,low,close,tick_vol,spread
2025-01-07 12:40:00,2644.48,2644.75,2642.77,2643.46,2253,23.0
```

## What helps most
- **Finer timeframe** (M5 > M15) — the strategy is entry-timing sensitive
- **More years** — especially more bear/calm regimes for robustness
- **Same broker as your challenge account** — so spread/commission match
