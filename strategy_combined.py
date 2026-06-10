"""
Combined Structural Breakout Strategy — XAUUSD M5

Two sessions, same structural logic:

  Session 1 — Asian Range Breakout (ARB)
      Range   : 00:00–07:55 (Asian accumulation)
      Entry   : 08:00–10:00 London open break
      Edge    : 8 hrs of institutional accumulation → explosive London move

  Session 2 — NY Open Breakout (NYO)
      Range   : 08:00–12:55 (London morning range)
      Entry   : 13:00–15:00 NY open break
      Edge    : London sets the day's range → NY institutions break it

Both use the same structural logic:
    - Define a consolidation range
    - Wait for institutional session to break it with conviction
    - Breakout buffer to filter fakeouts
    - H1 EMA(50) trend filter — only trade in dominant direction
    - Hard close at 21:00, zero overnight holds

Max 1 trade per session = max 2 trades per day.
Daily DD guard at 4%, Max DD guard at 8%.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backtester import Backtester, Strategy, Report, load

DATA_DIR = Path(__file__).parent / "data"


# ── Helpers ───────────────────────────────────────────────────────────────────

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def compute_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ARB range : 00:00–07:55 per date
    NYO range : 08:00–12:55 per date
    """
    df["date"] = df.index.date

    asian = df[df.index.hour < 8]
    arb   = asian.groupby("date").agg(
        high=("high", "max"), low=("low", "min")
    )
    arb["range_pts"] = ((arb["high"] - arb["low"]) / 0.01).round(0)

    london = df[(df.index.hour >= 8) & (df.index.hour < 13)]
    nyo    = london.groupby("date").agg(
        high=("high", "max"), low=("low", "min")
    )
    nyo["range_pts"] = ((nyo["high"] - nyo["low"]) / 0.01).round(0)

    df.drop(columns=["date"], inplace=True)
    return arb, nyo


def compute_h1_trend(df_m5: pd.DataFrame) -> pd.Series:
    h1 = load("H1")
    h1["ema50"] = ema(h1["close"], 50)
    h1["trend"] = np.where(h1["close"] > h1["ema50"], 1, -1)
    return h1["trend"].shift(1).reindex(df_m5.index, method="ffill").fillna(0)


def compute_regime(
    df_m5       : pd.DataFrame,
    adx_period  : int   = 14,
    atr_ma_per  : int   = 50,
    min_adx     : float = 22.0,
    min_atr_r   : float = 0.75,   # must be ≥ 75 % of its 50-bar avg (not too quiet)
    max_atr_r   : float = 3.5,    # must be ≤ 3.5× its 50-bar avg (not chaotic)
) -> pd.Series:
    """
    H1 market-regime filter.  Returns 1 (trade allowed) or 0 (sit out).

    Condition for trading:
      ADX(14)  ≥ min_adx          — some directional momentum
      ATR(14)  ≥ min_atr_r × MA   — not a dead/compressed session (Aug-25 style)
      ATR(14)  ≤ max_atr_r × MA   — not chaotically wide (Feb-26 style)

    All signals are shifted 1 bar (no look-ahead) then forward-filled
    onto the M5 index.
    """
    h1 = load("H1")

    # ── ATR(14) ──────────────────────────────────────────────────────────────
    prev_c = h1["close"].shift(1)
    tr = pd.concat([
        h1["high"] - h1["low"],
        (h1["high"] - prev_c).abs(),
        (h1["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr    = tr.ewm(span=adx_period, adjust=False).mean()
    atr_ma = atr.rolling(atr_ma_per, min_periods=atr_ma_per // 2).mean()
    atr_ratio = atr / atr_ma.replace(0, np.nan)

    # ── ADX(14) via Wilder smoothing ─────────────────────────────────────────
    up   = (h1["high"] - h1["high"].shift(1)).clip(lower=0)
    dn   = (h1["low"].shift(1) - h1["low"]).clip(lower=0)
    dm_p = up.where(up >= dn, 0.0)
    dm_m = dn.where(dn >  up, 0.0)

    atr_safe  = atr.replace(0, np.nan)
    di_p = 100 * dm_p.ewm(span=adx_period, adjust=False).mean() / atr_safe
    di_m = 100 * dm_m.ewm(span=adx_period, adjust=False).mean() / atr_safe
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx  = dx.ewm(span=adx_period, adjust=False).mean()

    tradeable = (
        (adx       >= min_adx)  &
        (atr_ratio >= min_atr_r) &
        (atr_ratio <= max_atr_r)
    ).astype(int)

    return tradeable.shift(1).reindex(df_m5.index, method="ffill").fillna(0).astype(int)


# ── Strategy ──────────────────────────────────────────────────────────────────

class CombinedBreakout(Strategy):

    # ARB session params
    ARB_ENTRY_START = 8
    ARB_ENTRY_END   = 10
    ARB_MIN_RANGE   = 1000
    ARB_MAX_RANGE   = 8000

    # NYO session params
    NYO_ENTRY_START = 13
    NYO_ENTRY_END   = 15
    NYO_MIN_RANGE   = 500
    NYO_MAX_RANGE   = 6000

    FORCE_CLOSE_H   = 21
    BREAKOUT_BUFFER = 30   # pts close must exceed the level

    def __init__(
        self,
        initial_balance : float = 10_000.0,
        sl_points       : int   = 1200,
        tp_mult         : float = 2.5,
        daily_dd_guard  : float = 0.04,
        max_dd_guard    : float = 0.085,  # triggers at 8.5% → actual max DD stays ≤ 9%
    ):
        self.sl_points    = sl_points
        self.tp_points    = int(sl_points * tp_mult)
        self.buf          = self.BREAKOUT_BUFFER * 0.01
        self.daily_guard  = daily_dd_guard
        self.max_dd_guard = max_dd_guard

        # Injected
        self.arb_ranges  = None
        self.nyo_ranges  = None
        self.h1_trend    = None
        self.regime      = None   # 1 = trending/trade, 0 = choppy/sit-out

        # Account state
        self.initial_bal = initial_balance
        self.balance     = initial_balance
        self.peak_bal    = initial_balance

        # Day state
        self._day        = None
        self._day_start  = initial_balance
        self._arb_done   = False   # one ARB trade per day
        self._nyo_done   = False   # one NYO trade per day

        # Trade state
        self._in_trade   = False
        self._dir        = None
        self._sl         = None
        self._tp         = None

    def _new_day(self, today):
        self._day      = today
        self._day_start = self.balance
        self._arb_done  = False
        self._nyo_done  = False

    def _guards_ok(self) -> bool:
        self.peak_bal = max(self.peak_bal, self.balance)
        daily_loss = (self._day_start - self.balance) / self.initial_bal
        total_dd   = (self.peak_bal  - self.balance) / self.initial_bal
        return daily_loss < self.daily_guard and total_dd < self.max_dd_guard

    def next(self, i: int, df: pd.DataFrame) -> str | None:
        bar   = df.iloc[i]
        t     = df.index[i]
        today = t.date()
        hour  = t.hour
        pt    = 0.01

        if today != self._day:
            self._new_day(today)

        # Force-close: rollover OR max DD breached
        if self._in_trade:
            self.peak_bal = max(self.peak_bal, self.balance)
            total_dd = (self.peak_bal - self.balance) / self.initial_bal
            if hour >= self.FORCE_CLOSE_H or total_dd >= self.max_dd_guard:
                return self._close()

        # Manage open trade
        if self._in_trade:
            if self._dir == "buy":
                if bar["low"]  <= self._sl: return self._close()
                if bar["high"] >= self._tp: return self._close()
            else:
                if bar["high"] >= self._sl: return self._close()
                if bar["low"]  <= self._tp: return self._close()
            return None

        if not self._guards_ok():
            return None

        # Regime filter: skip choppy / dead / chaotic sessions
        if self.regime is not None and int(self.regime.iloc[i]) == 0:
            return None

        close  = bar["close"]
        spread = bar["spread"]
        trend  = int(self.h1_trend.iloc[i])

        # ── Session 1: ARB (London open) ──────────────────────────────────────
        if (self.ARB_ENTRY_START <= hour < self.ARB_ENTRY_END
                and not self._arb_done
                and today in self.arb_ranges.index):

            r = self.arb_ranges.loc[today]
            if self.ARB_MIN_RANGE <= r["range_pts"] <= self.ARB_MAX_RANGE:
                if trend >= 0 and close > r["high"] + self.buf:
                    self._enter("buy", close, spread, pt)
                    self._arb_done = True
                    return "buy"
                if trend <= 0 and close < r["low"] - self.buf:
                    self._enter("sell", close, spread, pt)
                    self._arb_done = True
                    return "sell"

        # ── Session 2: NYO (NY open) ──────────────────────────────────────────
        if (self.NYO_ENTRY_START <= hour < self.NYO_ENTRY_END
                and not self._nyo_done
                and today in self.nyo_ranges.index):

            r = self.nyo_ranges.loc[today]
            if self.NYO_MIN_RANGE <= r["range_pts"] <= self.NYO_MAX_RANGE:
                if trend >= 0 and close > r["high"] + self.buf:
                    self._enter("buy", close, spread, pt)
                    self._nyo_done = True
                    return "buy"
                if trend <= 0 and close < r["low"] - self.buf:
                    self._enter("sell", close, spread, pt)
                    self._nyo_done = True
                    return "sell"

        return None

    def _enter(self, direction, close, spread, pt):
        if direction == "buy":
            entry    = close + (spread / 2) * pt
            self._sl = entry - self.sl_points * pt
            self._tp = entry + self.tp_points * pt
        else:
            entry    = close - (spread / 2) * pt
            self._sl = entry + self.sl_points * pt
            self._tp = entry - self.tp_points * pt
        self._in_trade = True
        self._dir      = direction

    def _close(self) -> str:
        self._in_trade = False
        self._dir = self._sl = self._tp = None
        return "close"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_combined(
    lots      : float = 0.1,
    balance   : float = 10_000.0,
    sl_points : int   = 1200,
    tp_mult   : float = 2.5,
    verbose   : bool  = True,
) -> tuple[Report, pd.DataFrame]:

    df  = load("M5")
    arb_ranges, nyo_ranges = compute_ranges(df)
    h1_trend = compute_h1_trend(df)
    regime   = compute_regime(df)

    strat              = CombinedBreakout(balance, sl_points, tp_mult)
    strat.arb_ranges   = arb_ranges
    strat.nyo_ranges   = nyo_ranges
    strat.h1_trend     = h1_trend
    strat.regime       = regime

    bt     = Backtester(df, strat, lots=lots, initial_balance=balance)
    report = bt.run()
    log    = report.trade_log()

    if verbose and not log.empty:
        report.print()
        overnight = (log["entry_time"].dt.date != log["exit_time"].dt.date).sum()
        log["date"]  = log["exit_time"].dt.date
        log["month"] = log["exit_time"].dt.to_period("M")
        daily   = log.groupby("date")["pnl"].sum()
        monthly = log.groupby("month")["pnl"].sum()
        worst_pct = daily.min() / balance * 100

        print(f"\n  Overnight holds  : {overnight} {'✅' if overnight == 0 else '⚠️'}")
        print(f"  Trades/month avg : {len(log)/len(monthly):.1f}")
        print(f"  Avg hold         : {log['bars_held'].mean()*5:.0f} min")
        print(f"  Best trade       : ${log['pnl'].max():.2f}")
        print(f"  Worst trade      : ${log['pnl'].min():.2f}")
        print(f"\n  Daily DD worst   : ${daily.min():.2f} ({worst_pct:.2f}%) "
              f"{'✅' if worst_pct > -5 else '❌'}")

        print(f"\n  Monthly PnL:")
        for m, v in monthly.items():
            bar_c = "▓" * int(abs(v) / 30)
            sign  = "+" if v > 0 else ""
            print(f"    {m}  {sign}${v:>8,.2f}  {bar_c}")

        print(f"\n  Monthly avg      : ${monthly.mean():.2f}")
        print(f"  Monthly median   : ${monthly.median():.2f}")
        print(f"  Best month       : ${monthly.max():.2f}")
        print(f"  Worst month      : ${monthly.min():.2f}")
        print(f"  Profitable months: {(monthly > 0).sum()} / {len(monthly)}")

        # Challenge simulation
        print(f"\n  === THE5ERS CHALLENGE SIMULATION ===")
        bal   = balance
        peak  = balance
        for _, t in log.iterrows():
            bal  += t["pnl"]
            peak  = max(peak, bal)
            if bal >= balance * 1.08:
                print(f"  ✅ Target hit on {t['exit_time'].date()} — ${bal:.2f}")
                break
            if peak - bal >= balance * 0.10:
                print(f"  ❌ Max DD breached on {t['exit_time'].date()} — ${bal:.2f}")
                break
        else:
            print(f"  Final balance: ${bal:.2f}")

    return report, log


# ── Parameter scan ────────────────────────────────────────────────────────────

def scan(lots: float = 0.1, balance: float = 10_000.0):
    print(f"\n{'sl_pts':>7} {'tp_x':>5} | "
          f"{'trades':>7} {'win%':>6} {'monthly':>9} {'pf':>5} "
          f"{'max_dd':>9} {'worst_day%':>11}")
    print("-" * 70)

    best = {}
    best_score = -999999

    for sl in [800, 1000, 1200, 1500]:
        for tp_m in [2.0, 2.5, 3.0]:
            r, log = run_combined(lots, balance, sl_points=sl,
                                  tp_mult=tp_m, verbose=False)
            if not r.trades or log.empty:
                continue
            pnls  = [t.pnl for t in r.trades]
            wins  = [p for p in pnls if p > 0]
            s     = r.summary()
            log["date"]  = log["exit_time"].dt.date
            log["month"] = log["exit_time"].dt.to_period("M")
            daily   = log.groupby("date")["pnl"].sum()
            monthly = log.groupby("month")["pnl"].sum()
            worst_pct = daily.min() / balance * 100

            score = monthly.mean() if worst_pct > -5 else -9999
            print(f"{sl:>7} {tp_m:>5} | "
                  f"{len(pnls):>7} {len(wins)/len(pnls)*100:>5.1f}% "
                  f"${monthly.mean():>8,.0f} {s['profit_factor']:>5} "
                  f"{s['max_drawdown']:>9} {worst_pct:>10.2f}%")

            if score > best_score:
                best_score = score
                best = dict(sl=sl, tp_m=tp_m, monthly_avg=monthly.mean(),
                            worst_pct=worst_pct)

    print("-" * 70)
    if best:
        print(f"\n✅ Best: SL={best['sl']} | TP×{best['tp_m']}")
        print(f"   Monthly avg: ${best['monthly_avg']:.2f} | "
              f"Worst day: {best['worst_pct']:.2f}%")
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Combined ARB + NY Open — default params (0.1 lot)")
    print("=" * 55)
    run_combined(lots=0.1, balance=10_000.0)

    print("\n\n=== PARAMETER SCAN ===")
    best = scan(lots=0.1, balance=10_000.0)

    if best:
        print(f"\n\n=== BEST CONFIG — full report ===")
        run_combined(lots=0.1, balance=10_000.0,
                     sl_points=best["sl"], tp_mult=best["tp_m"])
