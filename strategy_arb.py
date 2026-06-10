"""
Asian Range Breakout (ARB) v2 — XAUUSD Intraday

Improvements over v1:
    1. H1 trend filter  — only long if H1 EMA50 bullish, only short if bearish
    2. Breakout buffer  — require close X pts BEYOND Asian level (kills false breaks)
    3. Tight entry window — 08:00–10:30 London momentum only
    4. Range quality filter — medium-width ranges break cleanest (1000–8000 pts)

The5ers rules (10k account):
    Profit target : +8%  = +$800
    Max drawdown  : 10%  = $1,000 (guard at 8% = $800)
    Daily DD      : 5%   = $500   (guard at 4% = $400)

Scale:
    0.1 lot, 1 pt = $0.10 P&L
    SL 1000 pts = $100 = 1% of $10k
    TP 2000 pts = $200 = 2% of $10k
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backtester import Backtester, Strategy, Report, load

DATA_DIR = Path(__file__).parent / "data"

# ── Parameters ────────────────────────────────────────────────────────────────

ASIAN_END_H     = 8      # range locks at 07:55
ENTRY_START_H   = 8      # London open
ENTRY_END_H     = 11     # cut entries at 11:00 (past London momentum)
FORCE_CLOSE_H   = 21     # hard close before rollover

MIN_RANGE_PTS   = 1000   # ignore quiet sessions below $10 range
MAX_RANGE_PTS   = 8000   # ignore chaotic news sessions above $80 range
BREAKOUT_BUFFER = 50     # extra pts close must exceed Asian level (anti-false-break)

SL_POINTS       = 1000   # $10 price move → $100 at 0.1 lot = 1% of $10k
TP_MULT         = 2.0    # 2:1 RR

MAX_TRADES_DAY  = 1      # one trade per day (clean, avoids overtrading)
DAILY_DD_GUARD  = 0.04   # stop if daily loss ≥ 4% (buffer before 5% limit)
MAX_DD_GUARD    = 0.08   # stop if total DD  ≥ 8% (buffer before 10% limit)

H1_EMA_PERIOD   = 50


# ── Pre-computation ───────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_asian_ranges(df: pd.DataFrame) -> pd.DataFrame:
    asian = df[df.index.hour < ASIAN_END_H].copy()
    asian["date"] = asian.index.date
    ranges = asian.groupby("date").agg(
        asian_high=("high", "max"),
        asian_low=("low",  "min"),
    )
    ranges["range_pts"] = ((ranges["asian_high"] - ranges["asian_low"]) / 0.01).round(0)
    return ranges


def compute_h1_trend(df_m5: pd.DataFrame) -> pd.Series:
    """H1 EMA50 direction mapped onto M5 index (forward-filled, no look-ahead)."""
    h1 = load("H1")
    h1["ema50"]  = ema(h1["close"], H1_EMA_PERIOD)
    h1["trend"]  = np.where(h1["close"] > h1["ema50"], 1, -1)
    # Shift 1 bar to avoid look-ahead, then align to M5
    trend_m5 = h1["trend"].shift(1).reindex(df_m5.index, method="ffill").fillna(0)
    return trend_m5


# ── Strategy ──────────────────────────────────────────────────────────────────

class AsianRangeBreakout(Strategy):

    def __init__(
        self,
        initial_balance : float = 10_000.0,
        sl_points       : int   = SL_POINTS,
        tp_mult         : float = TP_MULT,
        breakout_buffer : int   = BREAKOUT_BUFFER,
        min_range_pts   : int   = MIN_RANGE_PTS,
        max_range_pts   : int   = MAX_RANGE_PTS,
        entry_end_h     : int   = ENTRY_END_H,
        max_trades_day  : int   = MAX_TRADES_DAY,
        daily_dd_guard  : float = DAILY_DD_GUARD,
        max_dd_guard    : float = MAX_DD_GUARD,
    ):
        self.sl_points      = sl_points
        self.tp_points      = int(sl_points * tp_mult)
        self.buffer         = breakout_buffer
        self.min_range_pts  = min_range_pts
        self.max_range_pts  = max_range_pts
        self.entry_end_h    = entry_end_h
        self.max_trades_day = max_trades_day
        self.daily_dd_guard = daily_dd_guard
        self.max_dd_guard   = max_dd_guard

        # Injected before run
        self.ranges   : pd.DataFrame = None
        self.h1_trend : pd.Series    = None

        # State
        self.initial_bal    = initial_balance
        self.balance        = initial_balance
        self.peak_balance   = initial_balance
        self._in_trade      = False
        self._direction     = None
        self._sl            = None
        self._tp            = None
        self._day_date      = None
        self._day_trades    = 0
        self._day_start_bal = initial_balance

    def _reset_day(self, today):
        self._day_date      = today
        self._day_trades    = 0
        self._day_start_bal = self.balance

    def next(self, i: int, df: pd.DataFrame) -> str | None:
        bar      = df.iloc[i]
        bar_time = df.index[i]
        hour     = bar_time.hour
        today    = bar_time.date()
        pt       = 0.01

        if today != self._day_date:
            self._reset_day(today)

        # Force-close
        if self._in_trade and hour >= FORCE_CLOSE_H:
            return self._close()

        # Manage open trade
        if self._in_trade:
            if self._direction == "buy":
                if bar["low"]  <= self._sl: return self._close()
                if bar["high"] >= self._tp: return self._close()
            else:
                if bar["high"] >= self._sl: return self._close()
                if bar["low"]  <= self._tp: return self._close()
            return None

        # Entry window
        if hour < ENTRY_START_H or hour >= self.entry_end_h:
            return None

        # Risk guards
        if self._day_trades >= self.max_trades_day:
            return None
        self.peak_balance = max(self.peak_balance, self.balance)
        daily_loss  = (self._day_start_bal - self.balance) / self.initial_bal
        total_dd    = (self.peak_balance - self.balance) / self.initial_bal
        if daily_loss >= self.daily_dd_guard or total_dd >= self.max_dd_guard:
            return None

        # Asian range for today
        if today not in self.ranges.index:
            return None
        r       = self.ranges.loc[today]
        rng_pts = int(r["range_pts"])
        if rng_pts < self.min_range_pts or rng_pts > self.max_range_pts:
            return None

        asian_high = r["asian_high"]
        asian_low  = r["asian_low"]
        close      = bar["close"]
        spread     = bar["spread"]
        buf_price  = self.buffer * pt

        # H1 trend
        trend = int(self.h1_trend.iloc[i]) if self.h1_trend is not None else 0

        # Long: bullish trend + close breaks well above Asian high
        if trend >= 0 and close > asian_high + buf_price:
            entry = close + (spread / 2) * pt
            self._sl = entry - self.sl_points * pt
            self._tp = entry + self.tp_points * pt
            self._in_trade = True
            self._direction = "buy"
            self._day_trades += 1
            return "buy"

        # Short: bearish trend + close breaks well below Asian low
        if trend <= 0 and close < asian_low - buf_price:
            entry = close - (spread / 2) * pt
            self._sl = entry + self.sl_points * pt
            self._tp = entry - self.tp_points * pt
            self._in_trade = True
            self._direction = "sell"
            self._day_trades += 1
            return "sell"

        return None

    def _close(self) -> str:
        self._in_trade  = False
        self._direction = None
        self._sl = self._tp = None
        return "close"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_arb(
    lots            : float = 0.1,
    balance         : float = 10_000.0,
    sl_points       : int   = SL_POINTS,
    tp_mult         : float = TP_MULT,
    breakout_buffer : int   = BREAKOUT_BUFFER,
    entry_end_h     : int   = ENTRY_END_H,
    verbose         : bool  = True,
) -> tuple[Report, pd.DataFrame]:
    df     = load("M5")
    ranges = compute_asian_ranges(df)
    trend  = compute_h1_trend(df)

    strategy                = AsianRangeBreakout(
        initial_balance  = balance,
        sl_points        = sl_points,
        tp_mult          = tp_mult,
        breakout_buffer  = breakout_buffer,
        entry_end_h      = entry_end_h,
    )
    strategy.ranges   = ranges
    strategy.h1_trend = trend

    bt     = Backtester(df, strategy, lots=lots, initial_balance=balance)
    report = bt.run()
    log    = report.trade_log()

    if verbose and not log.empty:
        report.print()
        overnight = (log["entry_time"].dt.date != log["exit_time"].dt.date).sum()
        log["date"] = log["exit_time"].dt.date
        daily = log.groupby("date")["pnl"].sum()
        worst_pct = daily.min() / balance * 100
        print(f"\n  Overnight holds : {overnight} {'✅' if overnight == 0 else '⚠️'}")
        print(f"  Avg hold        : {log['bars_held'].mean()*5:.0f} min")
        print(f"  Best trade      : ${log['pnl'].max():.2f}")
        print(f"  Worst trade     : ${log['pnl'].min():.2f}")
        print(f"\n  Daily PnL:")
        print(f"    Worst day     : ${daily.min():.2f}  ({worst_pct:.2f}%) "
              f"{'✅' if worst_pct > -5 else '❌ breaches 5% rule'}")
        print(f"    Best day      : ${daily.max():.2f}")
        print(f"    Avg day       : ${daily.mean():.2f}")
        print(f"    Profitable days: {(daily > 0).sum()} / {len(daily)} "
              f"({(daily > 0).mean()*100:.0f}%)")
    elif verbose:
        print("  No trades taken.")

    return report, log


# ── Parameter scan ────────────────────────────────────────────────────────────

def scan(lots: float = 0.1, balance: float = 10_000.0):
    print("\n=== PARAMETER SCAN — Asian Range Breakout v2 ===")
    print(f"{'sl_pts':>7} {'tp_x':>5} {'buf':>5} {'end_h':>6} | "
          f"{'trades':>7} {'win%':>6} {'net_pnl':>9} {'pf':>5} "
          f"{'max_dd':>9} {'worst_day%':>11}")
    print("-" * 80)

    best_pnl = -999999
    best_cfg = {}

    for sl in [800, 1000, 1200]:
        for tp_m in [1.5, 2.0, 2.5]:
            for buf in [30, 80, 150]:
                for end_h in [10, 11, 13]:
                    r, log = run_arb(lots, balance, sl_points=sl, tp_mult=tp_m,
                                     breakout_buffer=buf, entry_end_h=end_h,
                                     verbose=False)
                    if not r.trades or log.empty:
                        continue
                    pnls  = [t.pnl for t in r.trades]
                    wins  = [p for p in pnls if p > 0]
                    s     = r.summary()
                    log["date"] = log["exit_time"].dt.date
                    daily = log.groupby("date")["pnl"].sum()
                    worst_day_pct = daily.min() / balance * 100

                    net = sum(pnls)
                    print(f"{sl:>7} {tp_m:>5} {buf:>5} {end_h:>6} | "
                          f"{len(pnls):>7} {len(wins)/len(pnls)*100:>5.1f}% "
                          f"${net:>8,.0f} {s['profit_factor']:>5} "
                          f"{s['max_drawdown']:>9} {worst_day_pct:>10.2f}%")

                    if net > best_pnl:
                        best_pnl = net
                        best_cfg = dict(sl=sl, tp_m=tp_m, buf=buf, end_h=end_h,
                                        net=net, wr=len(wins)/len(pnls)*100,
                                        worst_day_pct=worst_day_pct)

    print("-" * 80)
    print(f"\n✅ Best config: SL={best_cfg['sl']} | TP×{best_cfg['tp_m']} | "
          f"Buffer={best_cfg['buf']} | Entry_end={best_cfg['end_h']}:00")
    print(f"   Net PnL: ${best_cfg['net']:,.2f} | "
          f"Win rate: {best_cfg['wr']:.1f}% | "
          f"Worst day: {best_cfg['worst_day_pct']:.2f}%")
    return best_cfg


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  Asian Range Breakout v2 — default params")
    print("=" * 52)
    run_arb(lots=0.1, balance=10_000.0)

    print()
    best = scan(lots=0.1, balance=10_000.0)

    # Re-run best config with full reporting
    print("\n\n" + "=" * 52)
    print("  Best config — full report")
    print("=" * 52)
    run_arb(lots=0.1, balance=10_000.0,
            sl_points=best["sl"], tp_mult=best["tp_m"],
            breakout_buffer=best["buf"], entry_end_h=best["end_h"])
