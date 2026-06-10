"""
Cross-Validation by Market Structure — XAUUSD M5

Splits the full dataset into distinct market regimes and runs the
combined breakout strategy fresh ($10k balance) on each slice.

Goal: confirm the strategy can pass The5ers challenge (+8%, <10% DD,
<5% daily DD) across ALL known market structures, not just the backtest
average.

Periods are defined by XAUUSD price structure Jan 2025 – Jun 2026:
  1. Steady Uptrend   Jan 07 – Mar 31 2025  (gradual rally to 2900)
  2. Rocket Bull      Apr 01 – May 31 2025  (ATH break, 2900→3500)
  3. Post-ATH Correct Jun 01 – Aug 31 2025  (range/pullback after peak)
  4. Range Recovery   Sep 01 – Dec 31 2025  (choppy consolidation)
  5. High-Vol Chop    Jan 01 – Mar 31 2026  (volatile, directionless)
  6. Current          Apr 01 – Jun 09 2026  (most recent)
"""

import numpy as np
import pandas as pd
from backtester import Backtester, load
from strategy_combined import (
    CombinedBreakout,
    compute_ranges,
    compute_h1_trend,
    compute_regime,
)

BALANCE  = 10_000.0
LOTS     = 0.1
SL_PTS   = 1200
TP_MULT  = 2.5

PERIODS = [
    ("Steady Uptrend",    "2025-01-07", "2025-03-31"),
    ("Rocket Bull",       "2025-04-01", "2025-05-31"),
    ("Post-ATH Correct",  "2025-06-01", "2025-08-31"),
    ("Range Recovery",    "2025-09-01", "2025-12-31"),
    ("High-Vol Chop",     "2026-01-01", "2026-03-31"),
    ("Current",           "2026-04-01", "2026-06-09"),
]


def run_period(df_full, h1_trend_full, regime_full,
               start: str, end: str,
               lots=LOTS, balance=BALANCE,
               sl_points=SL_PTS, tp_mult=TP_MULT):
    """
    Slice the pre-computed data to [start, end] and run the strategy fresh.
    Returns (report, log, summary_dict).
    """
    df = df_full.loc[start:end].copy()
    if len(df) < 10:
        return None, None, None

    # Slice the pre-computed series to this period
    h1_trend = h1_trend_full.reindex(df.index, method="ffill")
    regime   = regime_full.reindex(df.index, method="ffill")

    arb_ranges, nyo_ranges = compute_ranges(df)

    strat = CombinedBreakout(balance, sl_points, tp_mult)
    strat.arb_ranges = arb_ranges
    strat.nyo_ranges = nyo_ranges
    strat.h1_trend   = h1_trend
    strat.regime     = regime

    bt     = Backtester(df, strat, lots=lots, initial_balance=balance)
    report = bt.run()
    log    = report.trade_log()

    if log.empty:
        return report, log, None

    # Build summary
    s = report.summary()
    log["date"]  = log["exit_time"].dt.date
    log["month"] = log["exit_time"].dt.to_period("M")
    daily   = log.groupby("date")["pnl"].sum()
    monthly = log.groupby("month")["pnl"].sum()
    worst_day_pct = daily.min() / balance * 100

    # Challenge simulation — does it hit +8% before 10% DD?
    bal  = balance
    peak = balance
    challenge_result = "⏳ Not reached"
    challenge_date   = None
    for _, t in log.iterrows():
        bal  += t["pnl"]
        peak  = max(peak, bal)
        if bal >= balance * 1.08:
            challenge_result = "✅ PASS"
            challenge_date   = t["exit_time"].date()
            break
        if peak - bal >= balance * 0.10:
            challenge_result = "❌ FAIL (DD)"
            challenge_date   = t["exit_time"].date()
            break
    # Daily DD check
    if worst_day_pct <= -5.0:
        challenge_result = "❌ FAIL (daily DD)"

    return report, log, {
        "trades":         len(log),
        "win_rate":       f"{(log['pnl']>0).mean()*100:.0f}%",
        "monthly_avg":    monthly.mean(),
        "profit_factor":  s["profit_factor"],
        "max_dd":         s["max_drawdown"],
        "worst_day":      f"{worst_day_pct:.2f}%",
        "net_pnl":        s["net_pnl"],
        "challenge":      challenge_result,
        "ch_date":        challenge_date,
    }


def main(lots=LOTS, sl_points=SL_PTS, tp_mult=TP_MULT):
    print("Loading data …")
    df_full = load("M5")

    print("Computing trend + regime (H1) …")
    h1_trend_full = compute_h1_trend(df_full)
    regime_full   = compute_regime(df_full)

    print(f"\n{'='*70}")
    print(f"  Cross-Validation by Market Structure  —  {lots} lot | SL={sl_points} | TP×{tp_mult}")
    print(f"{'='*70}")
    print(f"  Starting balance: ${BALANCE:,.0f} | Target: +8% = ${BALANCE*1.08:,.0f} | Max DD: 10%")
    print(f"{'='*70}\n")

    results = []

    for label, start, end in PERIODS:
        r, log, info = run_period(
            df_full, h1_trend_full, regime_full,
            start, end, lots, BALANCE, sl_points, tp_mult
        )
        if info is None:
            print(f"  [{label:22s}]  {start} → {end}  — no trades / no data\n")
            results.append((label, start, end, None))
            continue

        ch = info["challenge"]
        cd = f"  ({info['ch_date']})" if info["ch_date"] else ""

        print(f"  [{label:22s}]  {start} → {end}")
        print(f"    trades={info['trades']:>3}  win={info['win_rate']:>4}  "
              f"monthly_avg=${info['monthly_avg']:>7,.0f}  "
              f"PF={info['profit_factor']:>5}  "
              f"max_dd={info['max_dd']:>10}  "
              f"worst_day={info['worst_day']:>7}")
        print(f"    net_pnl={info['net_pnl']:>10}   challenge → {ch}{cd}\n")

        results.append((label, start, end, info))

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Period':<22}  {'Trades':>6}  {'Monthly':>9}  {'Net PnL':>10}  {'Challenge'}")
    print(f"  {'-'*62}")
    passes = fails = skips = 0
    for label, start, end, info in results:
        if info is None:
            print(f"  {label:<22}  {'—':>6}  {'—':>9}  {'—':>10}  (no data)")
            skips += 1
        else:
            ch = info["challenge"]
            if "PASS" in ch: passes += 1
            elif "FAIL" in ch: fails += 1
            else: skips += 1
            print(f"  {label:<22}  {info['trades']:>6}  "
                  f"${info['monthly_avg']:>8,.0f}  "
                  f"{info['net_pnl']:>10}  {ch}")

    print(f"  {'-'*62}")
    total = passes + fails + skips
    print(f"  Passed: {passes}/{total-skips} regimes  |  Failed: {fails}/{total-skips}  |  "
          f"Insufficient data: {skips}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
