"""
Adaptive Strategy — XAUUSD M5

Routes between three sub-strategies based on H1 market regime:

  REGIME_TREND  ADX≥25, ATR 0.75–3.0× MA  →  CombinedBreakout (wide SL/TP)
                  Edge: institutional session breakouts with full momentum
  REGIME_SLOW   ADX 18–25, ATR < 1.1× MA  →  EMA Pullback
                  Edge: slow trends; enter on M5 pullback to H1 EMA
  REGIME_RANGE  ADX < 20                   →  Tight Structural Breakout
                  Same Asian/NY range logic as TREND but with tighter SL
                  and TP (moves complete quicker in a range); more entries
                  by also checking M15 and H1 session ranges
  REGIME_CHOP   ATR > 3× MA               →  Sit out
                  No edge when volatility is chaotic; protect capital

Regime is determined once per H1 bar (no look-ahead) and forwarded
to the M5 loop via a pre-computed Series.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backtester import Backtester, Strategy, Report, load
from strategy_combined import compute_ranges, ema

DATA_DIR = Path(__file__).parent / "data"

# ── Regime labels ─────────────────────────────────────────────────────────────

REGIME_TREND = 0   # breakout
REGIME_SLOW  = 1   # EMA pullback
REGIME_RANGE = 2   # BB mean reversion
REGIME_CHOP  = 3   # sit out
REGIME_BEAR  = 4   # bearish trending — short-only breakouts, wide TP

REGIME_NAMES = {
    REGIME_TREND: "TREND",
    REGIME_SLOW:  "SLOW ",
    REGIME_RANGE: "RANGE",
    REGIME_CHOP:  "CHOP ",
    REGIME_BEAR:  "BEAR ",
}

# ── Params ────────────────────────────────────────────────────────────────────

# Breakout (CombinedBreakout reused internally)
ARB_ENTRY_START = 8;  ARB_ENTRY_END = 10
NYO_ENTRY_START = 13; NYO_ENTRY_END = 15
ARB_MIN_RANGE = 1000; ARB_MAX_RANGE = 8000
NYO_MIN_RANGE = 500;  NYO_MAX_RANGE = 6000
BREAKOUT_BUFFER = 30   # pts
BO_SL_PTS  = 1200
BO_TP_MULT = 2.5

# EMA Pullback
PB_ENTRY_START = 8;   PB_ENTRY_END = 16
PB_SL_PTS  = 120      # pts
PB_TP_MULT = 2.0      # TP = 240 pts
PB_M5_EMA  = 21       # pullback EMA on M5

# EMA Pullback
PB_MAX_PER_DAY = 2    # allow 2 pullback trades per day
PB_MIN_EXT_PTS = 30   # price must have been at least 30 pts beyond EMA before crossing back

# Tight Structural Breakout (RANGE regime) — same Asian/NY logic, smaller moves
# In a ranging market the Asian range and London range still break intraday,
# they just don't run as far. Use tighter SL + smaller TP to match.
RNG_SL_PTS        = 500  # tighter SL (vs 1200 in TREND)
RNG_TP_MULT       = 1.5  # TP = 750 pts (vs 2.5× in TREND) — range moves finish sooner
RNG_BUFFER        = 15   # smaller breakout buffer in range (pts)
RNG_ARB_MIN       = 500  # allow tighter Asian ranges
RNG_ARB_MAX       = 5000
RNG_NYO_MIN       = 300
RNG_NYO_MAX       = 4000
RNG_MAX_ARB       = 2    # up to 2 ARB attempts per day in range
RNG_MAX_NYO       = 2    # up to 2 NYO attempts per day in range

# Bearish Trend Breakout — short-only, wide TP to ride the down-move
BEAR_SL_PTS      = 1000   # give it room, trading with trend
BEAR_TP_MULT     = 4.0    # 4000 pts TP — let the trend run
BEAR_BUFFER      = 20     # pts beyond range low to confirm breakdown
BEAR_EMA_SLOPE_N = 10     # H1 bars to measure EMA50 slope
BEAR_SLOPE_MIN   = -0.1   # EMA50 must be falling ≥ 0.1 pts/H1 bar (~$2.4/day)

FORCE_CLOSE_H  = 21
DAILY_DD_GUARD = 0.04
MAX_DD_GUARD   = 0.085


# ── Regime computation ────────────────────────────────────────────────────────

def compute_all_h1(df_m5: pd.DataFrame,
                   adx_period    : int   = 14,
                   atr_ma_per    : int   = 50,
                   min_adx_trend : float = 25.0,
                   min_adx_slow  : float = 18.0,
                   min_atr_r     : float = 0.75,
                   max_atr_r     : float = 3.0,
                   slow_atr_cap  : float = 1.1,
                   ) -> pd.DataFrame:
    """
    Returns a DataFrame indexed to df_m5 with columns:
        regime      int    REGIME_* constant (H1-derived, no look-ahead)
        h1_trend    int    +1 / -1 (H1 EMA50)
        h1_ema50    float
        m5_ema21    float  (M5 EMA21, for pullback entries)
        m5_bb_upper float  (M5 BB upper, for range mean reversion)
        m5_bb_lower float
        m5_bb_mid   float
        m5_rsi      float  (M5 RSI-9, for range entry confirmation)
    """
    h1 = load("H1")

    # ── H1 EMA50 trend ───────────────────────────────────────────────────────
    h1["ema50"] = ema(h1["close"], 50)
    h1["trend"] = np.where(h1["close"] > h1["ema50"], 1, -1)

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

    # ── ADX(14) ──────────────────────────────────────────────────────────────
    up   = (h1["high"] - h1["high"].shift(1)).clip(lower=0)
    dn   = (h1["low"].shift(1) - h1["low"]).clip(lower=0)
    dm_p = up.where(up >= dn, 0.0)
    dm_m = dn.where(dn >  up, 0.0)
    atr_s  = atr.replace(0, np.nan)
    di_p   = 100 * dm_p.ewm(span=adx_period, adjust=False).mean() / atr_s
    di_m   = 100 * dm_m.ewm(span=adx_period, adjust=False).mean() / atr_s
    dx     = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx    = dx.ewm(span=adx_period, adjust=False).mean()

    # (H1 BB removed — range sub-strategy uses structural breakout levels)

    # ── H1 EMA50 slope (bearish trend detection) ─────────────────────────────
    # slope = change in EMA50 per H1 bar over last BEAR_EMA_SLOPE_N bars
    ema50_slope = h1["ema50"].diff(BEAR_EMA_SLOPE_N) / BEAR_EMA_SLOPE_N

    # ── Regime classification ─────────────────────────────────────────────────
    chop  = atr_ratio > max_atr_r
    # Bearish trend: price below EMA50, EMA50 sloping down, ADX trending
    bear  = (~chop) & (h1["trend"] == -1) & (ema50_slope <= BEAR_SLOPE_MIN) & (adx >= min_adx_slow) & (atr_ratio >= min_atr_r)
    trend = (~chop) & (~bear) & (adx >= min_adx_trend) & (atr_ratio >= min_atr_r)
    slow  = (~chop) & (~bear) & (~trend) & (adx >= min_adx_slow) & (atr_ratio <= slow_atr_cap)
    regime_h1 = pd.Series(REGIME_RANGE, index=h1.index)
    regime_h1[trend] = REGIME_TREND
    regime_h1[slow]  = REGIME_SLOW
    regime_h1[bear]  = REGIME_BEAR
    regime_h1[chop]  = REGIME_CHOP

    # ── Shift 1 H1 bar (no look-ahead), reindex to M5 ───────────────────────
    def _ff(s): return s.shift(1).reindex(df_m5.index, method="ffill")

    result = pd.DataFrame(index=df_m5.index)
    result["regime"]   = _ff(regime_h1).fillna(REGIME_RANGE).astype(int)
    result["h1_trend"] = _ff(h1["trend"]).fillna(0).astype(int)
    result["h1_ema50"] = _ff(h1["ema50"])

    # M5 EMA21 for pullback entries
    result["m5_ema21"] = ema(df_m5["close"], PB_M5_EMA)

    result["h1_ema_slope"] = _ff(ema50_slope).fillna(0)

    return result


# ── Adaptive strategy ─────────────────────────────────────────────────────────

class AdaptiveStrategy(Strategy):

    FORCE_CLOSE_H = FORCE_CLOSE_H

    def __init__(
        self,
        initial_balance : float = 10_000.0,
        daily_dd_guard  : float = DAILY_DD_GUARD,
        max_dd_guard    : float = MAX_DD_GUARD,
    ):
        self.daily_guard  = daily_dd_guard
        self.max_dd_guard = max_dd_guard

        # Injected pre-computed data
        self.h1_data    = None   # DataFrame from compute_all_h1
        self.arb_ranges = None
        self.nyo_ranges = None

        # Account
        self.initial_bal = initial_balance
        self.balance     = initial_balance
        self.peak_bal    = initial_balance

        # Day state
        self._day        = None
        self._day_start  = initial_balance
        # TREND regime counters (1 trade per session)
        self._arb_done   = False
        self._nyo_done   = False
        # SLOW regime counter
        self._pb_today   = 0
        # RANGE regime counters (allow multiple per session)
        self._rng_arb    = 0   # ARB attempts used today in range mode
        self._rng_nyo    = 0   # NYO attempts used today in range mode
        self._bear_arb   = 0   # short ARB trades today in bear mode
        self._bear_nyo   = 0   # short NYO trades today in bear mode

        # Trade state
        self._in_trade   = False
        self._dir        = None
        self._sl         = None
        self._tp         = None
        self._is_bb      = False  # unused now but kept for _close() compat

    def _new_day(self, today):
        self._day      = today
        self._day_start = self.balance
        self._arb_done  = False
        self._nyo_done  = False
        self._pb_today  = 0
        self._rng_arb   = 0
        self._rng_nyo   = 0
        self._bear_arb   = 0
        self._bear_nyo   = 0

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

        h = self.h1_data.iloc[i]
        regime = int(h["regime"])

        # ── Force-close: rollover or max DD ──────────────────────────────────
        if self._in_trade:
            self.peak_bal = max(self.peak_bal, self.balance)
            total_dd = (self.peak_bal - self.balance) / self.initial_bal
            if hour >= self.FORCE_CLOSE_H or total_dd >= self.max_dd_guard:
                return self._close(i, self._is_bb)

        # ── Manage open trade (TP/SL) ─────────────────────────────────────────
        if self._in_trade:
            if self._dir == "buy":
                if bar["low"]  <= self._sl: return self._close(i, self._is_bb)
                if bar["high"] >= self._tp: return self._close(i, self._is_bb)
            else:
                if bar["high"] >= self._sl: return self._close(i, self._is_bb)
                if bar["low"]  <= self._tp: return self._close(i, self._is_bb)
            return None

        if not self._guards_ok():
            return None

        close  = bar["close"]
        spread = bar["spread"]
        trend  = int(h["h1_trend"])

        # ── CHOP: sit out ─────────────────────────────────────────────────────
        if regime == REGIME_CHOP:
            return None

        # ── TREND: breakout (ARB + NYO) ───────────────────────────────────────
        if regime == REGIME_TREND:
            buf = BREAKOUT_BUFFER * pt

            if (ARB_ENTRY_START <= hour < ARB_ENTRY_END
                    and not self._arb_done
                    and today in self.arb_ranges.index):
                r = self.arb_ranges.loc[today]
                if ARB_MIN_RANGE <= r["range_pts"] <= ARB_MAX_RANGE:
                    if trend >= 0 and close > r["high"] + buf:
                        self._enter_fixed("buy",  close, spread, pt, BO_SL_PTS, BO_TP_MULT)
                        self._arb_done = True
                        return "buy"
                    if trend <= 0 and close < r["low"] - buf:
                        self._enter_fixed("sell", close, spread, pt, BO_SL_PTS, BO_TP_MULT)
                        self._arb_done = True
                        return "sell"

            if (NYO_ENTRY_START <= hour < NYO_ENTRY_END
                    and not self._nyo_done
                    and today in self.nyo_ranges.index):
                r = self.nyo_ranges.loc[today]
                if NYO_MIN_RANGE <= r["range_pts"] <= NYO_MAX_RANGE:
                    if trend >= 0 and close > r["high"] + buf:
                        self._enter_fixed("buy",  close, spread, pt, BO_SL_PTS, BO_TP_MULT)
                        self._nyo_done = True
                        return "buy"
                    if trend <= 0 and close < r["low"] - buf:
                        self._enter_fixed("sell", close, spread, pt, BO_SL_PTS, BO_TP_MULT)
                        self._nyo_done = True
                        return "sell"

        # ── SLOW: EMA pullback (up to PB_MAX_PER_DAY per day) ────────────────
        elif regime == REGIME_SLOW:
            if not (PB_ENTRY_START <= hour < PB_ENTRY_END):
                return None
            if self._pb_today >= PB_MAX_PER_DAY:
                return None
            m5_ema = h["m5_ema21"]
            if pd.isna(m5_ema):
                return None
            prev_close = df["close"].iloc[i - 1]
            # Require meaningful extension before pullback cross (not just noise)
            extended_long  = prev_close < m5_ema - PB_MIN_EXT_PTS * pt
            extended_short = prev_close > m5_ema + PB_MIN_EXT_PTS * pt
            if trend > 0 and extended_long and close >= m5_ema:
                self._enter_fixed("buy",  close, spread, pt, PB_SL_PTS, PB_TP_MULT)
                self._pb_today += 1
                return "buy"
            if trend < 0 and extended_short and close <= m5_ema:
                self._enter_fixed("sell", close, spread, pt, PB_SL_PTS, PB_TP_MULT)
                self._pb_today += 1
                return "sell"

        # ── RANGE: tight structural breakout (same Asian/NY levels, tighter SL) ─
        # In a ranging market the intraday structure still produces breakouts —
        # they just complete quicker. Use the same range levels but with a
        # tighter SL and smaller TP, and allow re-entry after each close.
        elif regime == REGIME_RANGE:
            buf = RNG_BUFFER * pt

            if (ARB_ENTRY_START <= hour < ARB_ENTRY_END
                    and self._rng_arb < RNG_MAX_ARB
                    and today in self.arb_ranges.index):
                r = self.arb_ranges.loc[today]
                if RNG_ARB_MIN <= r["range_pts"] <= RNG_ARB_MAX:
                    if close > r["high"] + buf:
                        self._enter_fixed("buy",  close, spread, pt, RNG_SL_PTS, RNG_TP_MULT)
                        self._rng_arb += 1
                        return "buy"
                    if close < r["low"] - buf:
                        self._enter_fixed("sell", close, spread, pt, RNG_SL_PTS, RNG_TP_MULT)
                        self._rng_arb += 1
                        return "sell"

            if (NYO_ENTRY_START <= hour < NYO_ENTRY_END
                    and self._rng_nyo < RNG_MAX_NYO
                    and today in self.nyo_ranges.index):
                r = self.nyo_ranges.loc[today]
                if RNG_NYO_MIN <= r["range_pts"] <= RNG_NYO_MAX:
                    if close > r["high"] + buf:
                        self._enter_fixed("buy",  close, spread, pt, RNG_SL_PTS, RNG_TP_MULT)
                        self._rng_nyo += 1
                        return "buy"
                    if close < r["low"] - buf:
                        self._enter_fixed("sell", close, spread, pt, RNG_SL_PTS, RNG_TP_MULT)
                        self._rng_nyo += 1
                        return "sell"

        # ── BEAR: short-only structural breakout, wide TP ─────────────────────
        # In a confirmed bearish trend (EMA50 sloping down, price below EMA50),
        # only take breakdown shorts. Skip longs — they fade quickly.
        elif regime == REGIME_BEAR:
            buf = BEAR_BUFFER * pt

            if (ARB_ENTRY_START <= hour < ARB_ENTRY_END
                    and self._bear_arb < 1
                    and today in self.arb_ranges.index):
                r = self.arb_ranges.loc[today]
                if ARB_MIN_RANGE <= r["range_pts"] <= ARB_MAX_RANGE:
                    if close < r["low"] - buf:
                        self._enter_fixed("sell", close, spread, pt, BEAR_SL_PTS, BEAR_TP_MULT)
                        self._bear_arb += 1
                        return "sell"

            if (NYO_ENTRY_START <= hour < NYO_ENTRY_END
                    and self._bear_nyo < 1
                    and today in self.nyo_ranges.index):
                r = self.nyo_ranges.loc[today]
                if NYO_MIN_RANGE <= r["range_pts"] <= NYO_MAX_RANGE:
                    if close < r["low"] - buf:
                        self._enter_fixed("sell", close, spread, pt, BEAR_SL_PTS, BEAR_TP_MULT)
                        self._bear_nyo += 1
                        return "sell"

        return None

    def _enter_fixed(self, direction, close, spread, pt, sl_pts, tp_mult):
        tp_pts = int(sl_pts * tp_mult)
        if direction == "buy":
            entry    = close + (spread / 2) * pt
            self._sl = entry - sl_pts * pt
            self._tp = entry + tp_pts * pt
        else:
            entry    = close - (spread / 2) * pt
            self._sl = entry + sl_pts * pt
            self._tp = entry - tp_pts * pt
        self._in_trade = True
        self._dir      = direction
        self._is_bb    = False


    def _close(self, bar_i: int = -1, was_bb: bool = False) -> str:
        self._in_trade = False
        self._dir = self._sl = self._tp = None
        return "close"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_adaptive(
    lots    : float = 0.1,
    balance : float = 10_000.0,
    verbose : bool  = True,
    df_full : pd.DataFrame = None,
    h1_data : pd.DataFrame = None,
) -> tuple[Report, pd.DataFrame]:

    if df_full is None:
        df_full = load("M5")
    if h1_data is None:
        h1_data = compute_all_h1(df_full)

    arb_ranges, nyo_ranges = compute_ranges(df_full)

    strat = AdaptiveStrategy(balance)
    strat.h1_data    = h1_data
    strat.arb_ranges = arb_ranges
    strat.nyo_ranges = nyo_ranges

    bt     = Backtester(df_full, strat, lots=lots, initial_balance=balance)
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
        print(f"  Avg hold (bars)  : {log['bars_held'].mean()*5:.0f} min")
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

        # Regime breakdown
        if "regime" in h1_data.columns:
            log2 = log.copy()
            log2["regime"] = log2["entry_time"].map(
                lambda ts: REGIME_NAMES.get(int(h1_data.loc[h1_data.index.asof(ts), "regime"])
                                            if ts in h1_data.index or True else REGIME_RANGE, "?")
                if hasattr(h1_data.index, "asof") else "?"
            )
            try:
                log2["regime_at_entry"] = [
                    REGIME_NAMES.get(int(h1_data["regime"].asof(ts)), "?")
                    for ts in log2["entry_time"]
                ]
                rg = log2.groupby("regime_at_entry")["pnl"].agg(["count", "sum", "mean"])
                print(f"\n  By regime:")
                for rname, row in rg.iterrows():
                    print(f"    {rname}: {int(row['count']):>3} trades  "
                          f"net=${row['sum']:>8,.2f}  avg=${row['mean']:>7,.2f}")
            except Exception:
                pass

        # Challenge simulation
        print(f"\n  === THE5ERS CHALLENGE SIMULATION ===")
        bal, peak = balance, balance
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


# ── Cross-validation by market structure ──────────────────────────────────────

PERIODS = [
    ("Steady Uptrend",    "2025-01-07", "2025-03-31"),
    ("Rocket Bull",       "2025-04-01", "2025-05-31"),
    ("Post-ATH Correct",  "2025-06-01", "2025-08-31"),
    ("Range Recovery",    "2025-09-01", "2025-12-31"),
    ("High-Vol Chop",     "2026-01-01", "2026-03-31"),
    ("Current",           "2026-04-01", "2026-06-09"),
]


def find_bear_periods(df_full: pd.DataFrame, h1_data: pd.DataFrame,
                      min_days: int = 5) -> list[tuple[str, str]]:
    """
    Find contiguous date ranges in the historical data where REGIME_BEAR
    was active for at least min_days consecutive trading days.
    Returns list of (start_date, end_date) strings.
    """
    bear_mask = h1_data["regime"] == REGIME_BEAR
    # Map to daily: day is "bear" if majority of its H1 bars are REGIME_BEAR
    bear_mask.index = pd.to_datetime(bear_mask.index)
    daily_bear = bear_mask.resample("D").mean()
    daily_bear = (daily_bear >= 0.3).reindex(
        pd.date_range(daily_bear.index[0], daily_bear.index[-1], freq="D")
    ).fillna(False)

    periods = []
    in_bear = False
    start = None
    streak = 0

    for date, is_bear in daily_bear.items():
        if is_bear:
            if not in_bear:
                start = date
                streak = 0
                in_bear = True
            streak += 1
        else:
            if in_bear and streak >= min_days:
                periods.append((start.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")))
            in_bear = False
            streak = 0

    if in_bear and streak >= min_days:
        periods.append((start.strftime("%Y-%m-%d"), daily_bear.index[-1].strftime("%Y-%m-%d")))

    return periods


def validate_bear(lots=0.1, balance=10_000.0, df_full=None, h1_data=None):
    """Run the adaptive strategy on all identified BEAR periods."""
    if df_full is None:
        df_full = load("M5")
    if h1_data is None:
        h1_data = compute_all_h1(df_full)

    periods = find_bear_periods(df_full, h1_data)
    if not periods:
        print("  No bearish trending periods found in data.")
        return

    print(f"\n{'='*70}")
    print(f"  Bearish Trend Periods — validation ({lots} lot)")
    print(f"{'='*70}")
    print(f"  Detected {len(periods)} BEAR period(s) ≥ 10 trading days\n")

    for start, end in periods:
        df   = df_full.loc[start:end].copy()
        h1_s = h1_data.loc[start:end].copy()
        if len(df) < 20:
            continue

        arb_ranges, nyo_ranges = compute_ranges(df)
        strat = AdaptiveStrategy(balance)
        strat.h1_data    = h1_s
        strat.arb_ranges = arb_ranges
        strat.nyo_ranges = nyo_ranges

        bt     = Backtester(df, strat, lots=lots, initial_balance=balance)
        report = bt.run()
        log    = report.trade_log()

        if log.empty:
            print(f"  [{start} → {end}]  0 trades\n")
            continue

        s = report.summary()
        log["date"]  = log["exit_time"].dt.date
        log["month"] = log["exit_time"].dt.to_period("M")
        daily   = log.groupby("date")["pnl"].sum()
        monthly = log.groupby("month")["pnl"].sum()
        worst_pct = daily.min() / balance * 100

        # Count regime breakdown
        try:
            reg_counts = {}
            for ts in log["entry_time"]:
                rn = REGIME_NAMES.get(int(h1_s["regime"].asof(ts)), "?")
                reg_counts[rn] = reg_counts.get(rn, 0) + 1
            reg_str = " ".join(f"{k}:{v}" for k, v in sorted(reg_counts.items()))
        except Exception:
            reg_str = ""

        # Challenge sim
        bal, peak = balance, balance
        ch = "⏳ Not reached"
        ch_date = None
        for _, t in log.iterrows():
            bal  += t["pnl"]
            peak  = max(peak, bal)
            if bal >= balance * 1.08:
                ch = "✅ PASS"; ch_date = t["exit_time"].date(); break
            if peak - bal >= balance * 0.10:
                ch = "❌ FAIL(DD)"; ch_date = t["exit_time"].date(); break
        if worst_pct <= -5.0:
            ch = "❌ FAIL(daily)"

        days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        print(f"  [{start} → {end}]  ({days}d)")
        print(f"    trades={len(log):>3}  win={(log['pnl']>0).mean()*100:.0f}%  "
              f"monthly=${monthly.mean():>7,.0f}  PF={s['profit_factor']:>5}  "
              f"max_dd={s['max_drawdown']:>10}  worst_day={worst_pct:.2f}%")
        print(f"    regimes: {reg_str}")
        print(f"    {ch}{f'  ({ch_date})' if ch_date else ''}")

        print(f"    Monthly PnL:")
        for m, v in monthly.items():
            bar_c = "▓" * max(0, int(abs(v) / 30))
            sign  = "+" if v > 0 else ""
            print(f"      {m}  {sign}${v:>8,.2f}  {bar_c}")
        print()


def cross_validate(lots=0.1, balance=10_000.0, df_full=None, h1_data=None):
    if df_full is None:
        df_full = load("M5")
    if h1_data is None:
        print("Computing H1 indicators …")
        h1_data = compute_all_h1(df_full)

    print(f"\n{'='*72}")
    print(f"  Adaptive Strategy — Cross-Validation by Market Structure")
    print(f"{'='*72}")
    print(f"  Balance ${balance:,.0f} | Target +8% | Max DD 10% | {lots} lot")
    print(f"{'='*72}\n")

    rows = []

    for label, start, end in PERIODS:
        df   = df_full.loc[start:end].copy()
        h1_s = h1_data.loc[start:end].copy()
        if len(df) < 20:
            print(f"  [{label:<22}]  skipped (no data)\n")
            continue

        arb_ranges, nyo_ranges = compute_ranges(df)
        strat = AdaptiveStrategy(balance)
        strat.h1_data    = h1_s
        strat.arb_ranges = arb_ranges
        strat.nyo_ranges = nyo_ranges

        bt     = Backtester(df, strat, lots=lots, initial_balance=balance)
        report = bt.run()
        log    = report.trade_log()

        if log.empty:
            print(f"  [{label:<22}]  {start} → {end}  — 0 trades\n")
            rows.append((label, 0, 0, "$0", "⏳ No trades"))
            continue

        s = report.summary()
        log["date"]  = log["exit_time"].dt.date
        log["month"] = log["exit_time"].dt.to_period("M")
        daily   = log.groupby("date")["pnl"].sum()
        monthly = log.groupby("month")["pnl"].sum()
        worst_pct = daily.min() / balance * 100

        # Regime mix
        try:
            reg_counts = {}
            for ts in log["entry_time"]:
                rn = REGIME_NAMES.get(int(h1_s["regime"].asof(ts)), "?")
                reg_counts[rn] = reg_counts.get(rn, 0) + 1
            reg_str = " ".join(f"{k}:{v}" for k, v in sorted(reg_counts.items()))
        except Exception:
            reg_str = ""

        # Challenge sim
        bal, peak = balance, balance
        ch = "⏳ Not reached"
        ch_date = None
        for _, t in log.iterrows():
            bal  += t["pnl"]
            peak  = max(peak, bal)
            if bal >= balance * 1.08:
                ch = "✅ PASS"; ch_date = t["exit_time"].date(); break
            if peak - bal >= balance * 0.10:
                ch = "❌ FAIL(DD)"; ch_date = t["exit_time"].date(); break
        if worst_pct <= -5.0:
            ch = "❌ FAIL(daily)"

        print(f"  [{label:<22}]  {start} → {end}")
        print(f"    trades={len(log):>3}  win={(log['pnl']>0).mean()*100:.0f}%  "
              f"monthly_avg=${monthly.mean():>7,.0f}  PF={s['profit_factor']:>5}  "
              f"max_dd={s['max_drawdown']:>10}  worst_day={worst_pct:.2f}%")
        print(f"    net={s['net_pnl']:>10}   regimes: {reg_str}")
        print(f"    challenge → {ch}{f'  ({ch_date})' if ch_date else ''}\n")

        rows.append((label, len(log), monthly.mean(), s["net_pnl"], ch))

    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Period':<22}  {'Trades':>6}  {'Monthly':>9}  {'Net PnL':>10}  {'Challenge'}")
    print(f"  {'-'*65}")
    passes = fails = 0
    for label, trades, monthly_avg, net_pnl, ch in rows:
        if isinstance(monthly_avg, (int, float)):
            print(f"  {label:<22}  {trades:>6}  ${monthly_avg:>8,.0f}  {net_pnl:>10}  {ch}")
        else:
            print(f"  {label:<22}  {trades:>6}  {'—':>9}  {net_pnl:>10}  {ch}")
        if "PASS" in ch: passes += 1
        elif "FAIL" in ch: fails += 1
    total = passes + fails
    print(f"  {'-'*65}")
    print(f"  Passed: {passes}/{total}  |  Failed: {fails}/{total}")
    print(f"{'='*72}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading M5 data …")
    df_full = load("M5")
    print("Computing H1 regime + indicators …")
    h1_data = compute_all_h1(df_full)

    print("\n" + "=" * 60)
    print("  Adaptive Strategy — full period (0.1 lot)")
    print("=" * 60)
    run_adaptive(lots=0.1, balance=10_000.0, df_full=df_full, h1_data=h1_data)

    cross_validate(lots=0.1, balance=10_000.0, df_full=df_full, h1_data=h1_data)

    print("\n\n")
    validate_bear(lots=0.1, balance=10_000.0, df_full=df_full, h1_data=h1_data)
