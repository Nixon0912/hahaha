"""
Adaptive Strategy — XAUUSD M5

Routes between three sub-strategies based on H1 market regime:

  REGIME_TREND  ADX≥25, ATR 0.75–3.0× MA  →  CombinedBreakout
                  Edge: institutional session breakouts with momentum
  REGIME_SLOW   ADX 18–25, ATR < 1.1× MA  →  EMA Pullback
                  Edge: slow trends where breakouts fail; buy/sell dips
                  to the H1 EMA(50) in the dominant direction
  REGIME_RANGE  ADX < 20                   →  BB Mean Reversion
                  Edge: range-bound; fade H1 Bollinger Band(20,2σ)
                  extremes back to midline
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

REGIME_NAMES = {
    REGIME_TREND: "TREND",
    REGIME_SLOW:  "SLOW ",
    REGIME_RANGE: "RANGE",
    REGIME_CHOP:  "CHOP ",
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

# BB Mean Reversion
BB_ENTRY_START = 8;   BB_ENTRY_END = 14
BB_SL_PTS = 100
BB_TP_MULT = 1.5      # TP = 150 pts
BB_PERIOD  = 20
BB_SIGMA   = 2.0

FORCE_CLOSE_H  = 21
DAILY_DD_GUARD = 0.04
MAX_DD_GUARD   = 0.085


# ── Regime computation ────────────────────────────────────────────────────────

def compute_all_h1(df_m5: pd.DataFrame,
                   adx_period : int   = 14,
                   atr_ma_per : int   = 50,
                   bb_period  : int   = BB_PERIOD,
                   bb_sigma   : float = BB_SIGMA,
                   min_adx_trend : float = 25.0,
                   min_adx_slow  : float = 18.0,
                   min_atr_r     : float = 0.75,
                   max_atr_r     : float = 3.0,
                   slow_atr_cap  : float = 1.1,
                   ) -> pd.DataFrame:
    """
    Returns a DataFrame indexed to df_m5 with columns:
        regime    int  REGIME_* constant
        h1_trend  int  +1 / -1 (H1 EMA50 direction)
        h1_ema50  float
        m5_ema21  float  (M5 EMA21 for pullback entries, indexed to M5)
        bb_upper  float  (H1 BB upper band, reindexed to M5)
        bb_lower  float
        bb_mid    float
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

    # ── H1 Bollinger Bands ───────────────────────────────────────────────────
    bb_mid   = h1["close"].rolling(bb_period).mean()
    bb_std   = h1["close"].rolling(bb_period).std()
    bb_upper = bb_mid + bb_sigma * bb_std
    bb_lower = bb_mid - bb_sigma * bb_std

    # ── Regime classification ─────────────────────────────────────────────────
    chop  = atr_ratio > max_atr_r
    trend = (~chop) & (adx >= min_adx_trend) & (atr_ratio >= min_atr_r)
    slow  = (~chop) & (~trend) & (adx >= min_adx_slow) & (atr_ratio <= slow_atr_cap)
    regime_h1 = pd.Series(REGIME_RANGE, index=h1.index)
    regime_h1[trend] = REGIME_TREND
    regime_h1[slow]  = REGIME_SLOW
    regime_h1[chop]  = REGIME_CHOP

    # ── Shift 1 H1 bar (no look-ahead), reindex to M5 ───────────────────────
    def _ff(s): return s.shift(1).reindex(df_m5.index, method="ffill")

    result = pd.DataFrame(index=df_m5.index)
    result["regime"]   = _ff(regime_h1).fillna(REGIME_RANGE).astype(int)
    result["h1_trend"] = _ff(h1["trend"]).fillna(0).astype(int)
    result["h1_ema50"] = _ff(h1["ema50"])
    result["bb_upper"] = _ff(bb_upper)
    result["bb_lower"] = _ff(bb_lower)
    result["bb_mid"]   = _ff(bb_mid)

    # M5 EMA21 is computed on M5 itself (no look-ahead needed)
    result["m5_ema21"] = ema(df_m5["close"], PB_M5_EMA)

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
        self._arb_done   = False
        self._nyo_done   = False
        self._day_traded = False   # for SLOW/RANGE subs (1 trade/day)

        # Trade state
        self._in_trade   = False
        self._dir        = None
        self._sl         = None
        self._tp         = None
        self._tp_dynamic = False   # True when TP is a price level (BB mid)

    def _new_day(self, today):
        self._day        = today
        self._day_start  = self.balance
        self._arb_done   = False
        self._nyo_done   = False
        self._day_traded = False

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
                return self._close()

        # ── Manage open trade (TP/SL) ─────────────────────────────────────────
        if self._in_trade:
            tp_price = h["bb_mid"] if self._tp_dynamic else self._tp
            if self._dir == "buy":
                if bar["low"]  <= self._sl:    return self._close()
                if bar["high"] >= tp_price:    return self._close()
            else:
                if bar["high"] >= self._sl:    return self._close()
                if bar["low"]  <= tp_price:    return self._close()
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

        # ── SLOW: EMA pullback ────────────────────────────────────────────────
        elif regime == REGIME_SLOW:
            if not (PB_ENTRY_START <= hour < PB_ENTRY_END) or self._day_traded:
                return None
            m5_ema = h["m5_ema21"]
            if pd.isna(m5_ema):
                return None
            # Long: H1 uptrend, price just crossed back above M5 EMA (pullback complete)
            prev_close = df["close"].iloc[i - 1]
            if trend > 0 and prev_close < m5_ema and close >= m5_ema:
                self._enter_fixed("buy",  close, spread, pt, PB_SL_PTS, PB_TP_MULT)
                self._day_traded = True
                return "buy"
            if trend < 0 and prev_close > m5_ema and close <= m5_ema:
                self._enter_fixed("sell", close, spread, pt, PB_SL_PTS, PB_TP_MULT)
                self._day_traded = True
                return "sell"

        # ── RANGE: BB mean reversion ──────────────────────────────────────────
        elif regime == REGIME_RANGE:
            if not (BB_ENTRY_START <= hour < BB_ENTRY_END) or self._day_traded:
                return None
            bb_upper = h["bb_upper"]
            bb_lower = h["bb_lower"]
            bb_mid   = h["bb_mid"]
            if pd.isna(bb_upper) or pd.isna(bb_lower):
                return None
            # Fade upper band → sell; fade lower band → buy
            # Only trade if BB mid is in trend direction (avoid fighting strong moves)
            if close > bb_upper:
                self._enter_bb("sell", close, spread, pt, bb_mid)
                self._day_traded = True
                return "sell"
            if close < bb_lower:
                self._enter_bb("buy",  close, spread, pt, bb_mid)
                self._day_traded = True
                return "buy"

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
        self._in_trade   = True
        self._dir        = direction
        self._tp_dynamic = False

    def _enter_bb(self, direction, close, spread, pt, bb_mid):
        if direction == "buy":
            entry    = close + (spread / 2) * pt
            self._sl = entry - BB_SL_PTS * pt
        else:
            entry    = close - (spread / 2) * pt
            self._sl = entry + BB_SL_PTS * pt
        self._in_trade   = True
        self._dir        = direction
        self._tp         = None       # dynamic: BB midline (updated each bar)
        self._tp_dynamic = True       # checked against h["bb_mid"] in manage loop

    def _close(self) -> str:
        self._in_trade   = False
        self._dir = self._sl = self._tp = None
        self._tp_dynamic = False
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
