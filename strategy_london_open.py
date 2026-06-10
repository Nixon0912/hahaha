"""
London Pre-Open Breakout — XAUUSD M5 Scalper

Edge:
    The 30 minutes before London open (07:30–07:55) is the tightest
    consolidation of the day. London institutions break it at 08:00
    with conviction. Catching this break on M5 gives:
      - Clear structural entry (the pre-open high/low)
      - Natural SL placement (other side of the range)
      - Explosive first-hour momentum to ride

Why this is better than EMA cross:
    - Trades off a REAL level with institutional logic behind it
    - Tight pre-open range = tight SL = larger lot size within rules
    - Only 1 trade per day = clean, no overtrading

Parameters:
    Pre-open window : 07:30–07:55 (6 M5 bars)
    Entry window    : 08:00–09:00 (first London hour only)
    SL              : range_size × SL_MULT (other side of range)
    TP              : range_size × TP_MULT
    Min range pts   : minimum range to trade (filter dead mornings)
    Max range pts   : maximum range (filter news-gap mornings)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backtester import Backtester, Strategy, Report, load

DATA_DIR = Path(__file__).parent / "data"

# ── Parameters ────────────────────────────────────────────────────────────────

PRE_OPEN_START  = (7, 30)   # server time tuple (hour, minute)
PRE_OPEN_END    = (8,  0)   # London open
ENTRY_END       = (9,  0)   # stop hunting after first hour
FORCE_CLOSE_H   = 21

MIN_RANGE_PTS   = 50    # pre-open range must be at least $0.50 — filter dead sessions
MAX_RANGE_PTS   = 600   # pre-open range max $6 — filter gapped-up mornings

SL_MULT         = 1.0   # SL = range_size (other side of breakout)
TP_MULT         = 3.0   # TP = 3 × range (aggressive target, first-hour momentum)

MAX_DD_GUARD    = 0.08
DAILY_DD_GUARD  = 0.04


# ── Pre-compute pre-open ranges ───────────────────────────────────────────────

def compute_preopen_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading date, compute the High/Low of 07:30–07:55.
    """
    mask  = ((df.index.hour == 7) & (df.index.minute >= 30))
    pre   = df[mask].copy()
    pre["date"] = pre.index.date

    ranges = pre.groupby("date").agg(
        pre_high=("high", "max"),
        pre_low=("low",  "min"),
    )
    ranges["range_pts"] = ((ranges["pre_high"] - ranges["pre_low"]) / 0.01).round(0)
    return ranges


# ── Strategy ──────────────────────────────────────────────────────────────────

class LondonOpenBreakout(Strategy):

    def __init__(
        self,
        initial_balance : float = 10_000.0,
        sl_mult         : float = SL_MULT,
        tp_mult         : float = TP_MULT,
        min_range_pts   : int   = MIN_RANGE_PTS,
        max_range_pts   : int   = MAX_RANGE_PTS,
        daily_dd_guard  : float = DAILY_DD_GUARD,
        max_dd_guard    : float = MAX_DD_GUARD,
    ):
        self.sl_mult       = sl_mult
        self.tp_mult       = tp_mult
        self.min_range_pts = min_range_pts
        self.max_range_pts = max_range_pts
        self.daily_guard   = daily_dd_guard
        self.max_dd_guard  = max_dd_guard

        self.ranges        = None   # injected

        self.initial_bal   = initial_balance
        self.balance       = initial_balance
        self.peak_bal      = initial_balance
        self._in_trade     = False
        self._dir          = None
        self._sl           = None
        self._tp           = None
        self._day          = None
        self._day_traded   = False
        self._day_start    = initial_balance

    def _new_day(self, today):
        self._day        = today
        self._day_traded = False
        self._day_start  = self.balance

    def next(self, i: int, df: pd.DataFrame) -> str | None:
        bar  = df.iloc[i]
        t    = df.index[i]
        today = t.date()
        pt   = 0.01

        if today != self._day:
            self._new_day(today)

        # Force-close
        if self._in_trade and t.hour >= FORCE_CLOSE_H:
            return self._close()

        # Manage trade
        if self._in_trade:
            if self._dir == "buy":
                if bar["low"]  <= self._sl: return self._close()
                if bar["high"] >= self._tp: return self._close()
            else:
                if bar["high"] >= self._sl: return self._close()
                if bar["low"]  <= self._tp: return self._close()
            return None

        # Entry window: 08:00–08:55
        if not (t.hour == 8 and t.minute < 60) or (t.hour >= ENTRY_END[0] and t.minute >= ENTRY_END[1]):
            return None
        if t.hour < PRE_OPEN_END[0]:
            return None
        if self._day_traded:
            return None

        # Risk guards
        self.peak_bal = max(self.peak_bal, self.balance)
        if (self._day_start - self.balance) / self.initial_bal >= self.daily_guard:
            return None
        if (self.peak_bal - self.balance) / self.initial_bal >= self.max_dd_guard:
            return None

        # Pre-open range for today
        if today not in self.ranges.index:
            return None
        r       = self.ranges.loc[today]
        rng_pts = int(r["range_pts"])
        if rng_pts < self.min_range_pts or rng_pts > self.max_range_pts:
            return None

        pre_high  = r["pre_high"]
        pre_low   = r["pre_low"]
        rng_size  = pre_high - pre_low
        close     = bar["close"]
        spread    = bar["spread"]

        # Long breakout
        if close > pre_high:
            entry      = close + (spread / 2) * pt
            self._sl   = entry - rng_size * self.sl_mult
            self._tp   = entry + rng_size * self.tp_mult
            self._in_trade  = True
            self._dir       = "buy"
            self._day_traded = True
            return "buy"

        # Short breakout
        if close < pre_low:
            entry      = close - (spread / 2) * pt
            self._sl   = entry + rng_size * self.sl_mult
            self._tp   = entry - rng_size * self.tp_mult
            self._in_trade  = True
            self._dir       = "sell"
            self._day_traded = True
            return "sell"

        return None

    def _close(self) -> str:
        self._in_trade = False
        self._dir = self._sl = self._tp = None
        return "close"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_london(
    lots         : float = 0.1,
    balance      : float = 10_000.0,
    sl_mult      : float = SL_MULT,
    tp_mult      : float = TP_MULT,
    min_range    : int   = MIN_RANGE_PTS,
    max_range    : int   = MAX_RANGE_PTS,
    verbose      : bool  = True,
) -> tuple[Report, pd.DataFrame]:
    df     = load("M5")
    ranges = compute_preopen_ranges(df)

    strat          = LondonOpenBreakout(balance, sl_mult, tp_mult, min_range, max_range)
    strat.ranges   = ranges
    bt             = Backtester(df, strat, lots=lots, initial_balance=balance)
    report         = bt.run()
    log            = report.trade_log()

    if verbose and not log.empty:
        report.print()
        overnight = (log["entry_time"].dt.date != log["exit_time"].dt.date).sum()
        log["date"]  = log["exit_time"].dt.date
        log["month"] = log["exit_time"].dt.to_period("M")
        daily   = log.groupby("date")["pnl"].sum()
        monthly = log.groupby("month")["pnl"].sum()
        worst_day_pct = daily.min() / balance * 100

        print(f"\n  Overnight holds  : {overnight} {'✅' if overnight == 0 else '⚠️'}")
        print(f"  Trades/month avg : {len(log)/len(monthly):.1f}")
        print(f"  Avg hold         : {log['bars_held'].mean()*5:.0f} min")
        print(f"  Best trade       : ${log['pnl'].max():.2f}")
        print(f"  Worst trade      : ${log['pnl'].min():.2f}")
        print(f"\n  Daily DD  worst  : ${daily.min():.2f} ({worst_day_pct:.2f}%) "
              f"{'✅' if worst_day_pct > -5 else '❌'}")
        print(f"\n  Monthly PnL:")
        for m, v in monthly.items():
            bar_char = "▓" * int(abs(v) / 50)
            sign     = "+" if v > 0 else ""
            print(f"    {m}  {sign}${v:>8,.2f}  {bar_char}")
        print(f"\n  Monthly avg    : ${monthly.mean():.2f}")
        print(f"  Monthly median : ${monthly.median():.2f}")
        print(f"  Best month     : ${monthly.max():.2f}")
        print(f"  Worst month    : ${monthly.min():.2f}")
        print(f"  Profitable months: {(monthly > 0).sum()} / {len(monthly)}")

    return report, log


# ── Parameter scan ────────────────────────────────────────────────────────────

def scan(lots: float = 0.1, balance: float = 10_000.0):
    print(f"\n{'sl_x':>5} {'tp_x':>5} {'min_r':>6} {'max_r':>6} | "
          f"{'trades':>7} {'win%':>6} {'monthly_avg':>12} {'pf':>5} "
          f"{'max_dd':>9} {'worst_day%':>11}")
    print("-" * 85)

    best = {}
    best_score = -999999

    for sl_m in [0.5, 1.0, 1.5]:
        for tp_m in [2.0, 3.0, 4.0]:
            for min_r in [50, 100]:
                for max_r in [400, 600]:
                    r, log = run_london(lots, balance, sl_mult=sl_m, tp_mult=tp_m,
                                        min_range=min_r, max_range=max_r, verbose=False)
                    if not r.trades or log.empty:
                        continue
                    pnls  = [t.pnl for t in r.trades]
                    wins  = [p for p in pnls if p > 0]
                    s     = r.summary()
                    log["date"]  = log["exit_time"].dt.date
                    log["month"] = log["exit_time"].dt.to_period("M")
                    daily   = log.groupby("date")["pnl"].sum()
                    monthly = log.groupby("month")["pnl"].sum()
                    worst_day_pct = daily.min() / balance * 100

                    score = monthly.mean() if worst_day_pct > -5 else -9999
                    print(f"{sl_m:>5} {tp_m:>5} {min_r:>6} {max_r:>6} | "
                          f"{len(pnls):>7} {len(wins)/len(pnls)*100:>5.1f}% "
                          f"${monthly.mean():>11,.2f} {s['profit_factor']:>5} "
                          f"{s['max_drawdown']:>9} {worst_day_pct:>10.2f}%")

                    if score > best_score:
                        best_score = score
                        best = dict(sl_m=sl_m, tp_m=tp_m, min_r=min_r, max_r=max_r,
                                    monthly_avg=monthly.mean(),
                                    worst_day_pct=worst_day_pct)

    print("-" * 85)
    if best:
        print(f"\n✅ Best: SL×{best['sl_m']} | TP×{best['tp_m']} | "
              f"Range {best['min_r']}–{best['max_r']} pts")
        print(f"   Monthly avg: ${best['monthly_avg']:.2f} | "
              f"Worst day: {best['worst_day_pct']:.2f}%")
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  London Pre-Open Breakout — XAUUSD M5")
    print("=" * 55)
    run_london(lots=0.1, balance=10_000.0)

    print("\n\n=== PARAMETER SCAN ===")
    best = scan(lots=0.1, balance=10_000.0)

    if best:
        # Show at the maximum safe lot size
        # SL = rng_size * sl_mult — median pre-open range
        df     = load("M5")
        ranges = compute_preopen_ranges(df)
        valid  = ranges[(ranges["range_pts"] >= best["min_r"]) &
                        (ranges["range_pts"] <= best["max_r"])]
        median_rng_price = float(valid["range_pts"].median()) * 0.01
        sl_price = median_rng_price * best["sl_m"]
        sl_pts_median = sl_price / 0.01

        # Max lot: 3 losses/day < $400
        max_lot = round((400 / 3) / (sl_pts_median * 0.01 * 100) / 0.01) * 0.01
        max_lot = min(max_lot, 0.5)

        print(f"\n\n=== BEST CONFIG AT {max_lot} LOT ===")
        print(f"(Median SL = {sl_pts_median:.0f} pts, max 3 losses/day < $400)")
        run_london(lots=max_lot, balance=10_000.0,
                   sl_mult=best["sl_m"], tp_mult=best["tp_m"],
                   min_range=best["min_r"], max_range=best["max_r"])
