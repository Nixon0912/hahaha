"""
Alpha Hunt — systematic search for edge across 5 realistic market periods.
Rocket Bull (Apr–May 2025) excluded throughout.

Tested dimensions:
  1. SL/TP sizing    fixed pts  vs  ATR-adaptive  vs  range-adaptive
  2. TP multiplier   1.5 / 2.0 / 2.5 / 3.0
  3. Sessions        ARB-only / NYO-only / both
  4. ATR gate        minimum H1 ATR(14) to trade (skip dead days)
  5. Range/ATR ratio Asian/London range must be ≥ X% of H1 ATR
  6. Trend filter    H1 EMA50 required / optional
  7. Day-of-week     all / Mon–Thu only / exclude Friday

Scoring: a config is "alpha" if:
  - PF > 1.10 in ALL 5 test periods
  - Monthly avg > $0 in ALL 5 test periods
  - Max DD never breaches $1,500 in any period
  - Zero failed challenges (no period hits 10% DD)
"""

import numpy as np
import pandas as pd
from itertools import product
from backtester import Backtester, Strategy, Report, load
from strategy_combined import compute_ranges, ema

# ── Test periods (no Rocket Bull) ─────────────────────────────────────────────

PERIODS = [
    ("Steady Uptrend",   "2025-01-07", "2025-03-31"),
    ("Post-ATH Correct", "2025-06-01", "2025-08-31"),
    ("Range Recovery",   "2025-09-01", "2025-12-31"),
    ("High-Vol Chop",    "2026-01-01", "2026-03-31"),
    ("Current",          "2026-04-01", "2026-06-09"),
]

BALANCE = 10_000.0
LOTS    = 0.1

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def compute_h1_atr(df_m5, period=14, ma_period=50):
    """Return H1 ATR(14) and its 50-bar MA, reindexed to M5."""
    h1 = load("H1")
    prev_c = h1["close"].shift(1)
    tr = pd.concat([
        h1["high"] - h1["low"],
        (h1["high"] - prev_c).abs(),
        (h1["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr    = tr.ewm(span=period, adjust=False).mean()
    atr_ma = atr.rolling(ma_period, min_periods=ma_period // 2).mean()
    h1["ema50"] = _ema(h1["close"], 50)
    trend = pd.Series(np.where(h1["close"] > h1["ema50"], 1, -1), index=h1.index)

    def _ff(s): return s.shift(1).reindex(df_m5.index, method="ffill")
    out = pd.DataFrame(index=df_m5.index)
    out["h1_atr"]    = _ff(atr)
    out["h1_atr_ma"] = _ff(atr_ma)
    out["h1_trend"]  = _ff(trend).fillna(0).astype(int)
    return out


# ── Configurable breakout strategy ────────────────────────────────────────────

class HuntStrategy(Strategy):
    """
    Structural breakout strategy with every dimension configurable.
    Used by the scanner to test thousands of parameter combinations.
    """

    def __init__(
        self,
        # SL/TP sizing mode
        sl_mode        : str   = "fixed",  # "fixed" | "atr" | "range"
        sl_fixed_pts   : int   = 1200,
        tp_mult        : float = 2.5,
        sl_atr_mult    : float = 0.6,   # SL = H1_ATR * sl_atr_mult (in price → pts)
        sl_min_pts     : int   = 400,
        sl_max_pts     : int   = 2000,
        # Session
        use_arb        : bool  = True,
        use_nyo        : bool  = True,
        # Filters
        min_h1_atr     : float = 0.0,    # skip bars where H1 ATR < this ($)
        min_range_atr_r: float = 0.0,    # range must be ≥ X × H1_ATR
        require_trend  : bool  = True,   # H1 EMA50 direction filter
        dow_filter     : str   = "all",  # "all" | "mon-thu" | "no-fri"
        breakout_buf   : int   = 30,
        # Risk
        initial_balance: float = BALANCE,
        daily_dd_guard : float = 0.04,
        max_dd_guard   : float = 0.085,
    ):
        self.sl_mode         = sl_mode
        self.sl_fixed_pts    = sl_fixed_pts
        self.tp_mult         = tp_mult
        self.sl_atr_mult     = sl_atr_mult
        self.sl_min_pts      = sl_min_pts
        self.sl_max_pts      = sl_max_pts
        self.use_arb         = use_arb
        self.use_nyo         = use_nyo
        self.min_h1_atr      = min_h1_atr
        self.min_range_atr_r = min_range_atr_r
        self.require_trend   = require_trend
        self.dow_filter      = dow_filter
        self.buf             = breakout_buf * 0.01
        self.daily_guard     = daily_dd_guard
        self.max_dd_guard    = max_dd_guard

        # Injected
        self.arb_ranges = None
        self.nyo_ranges = None
        self.h1_info    = None   # DataFrame with h1_atr, h1_trend

        # Account
        self.initial_bal = initial_balance
        self.balance     = initial_balance
        self.peak_bal    = initial_balance

        # Day
        self._day       = None
        self._day_start = initial_balance
        self._arb_done  = False
        self._nyo_done  = False

        # Trade
        self._in_trade = False
        self._dir      = None
        self._sl       = None
        self._tp       = None

    def _new_day(self, today):
        self._day       = today
        self._day_start = self.balance
        self._arb_done  = False
        self._nyo_done  = False

    def _guards_ok(self):
        self.peak_bal = max(self.peak_bal, self.balance)
        return (
            (self._day_start - self.balance) / self.initial_bal < self.daily_guard
            and (self.peak_bal - self.balance) / self.initial_bal < self.max_dd_guard
        )

    def _sl_pts(self, h1_atr, range_pts):
        if self.sl_mode == "atr":
            pts = int(h1_atr * self.sl_atr_mult / 0.01)
        elif self.sl_mode == "range":
            pts = int(range_pts * 0.5)   # half the session range
        else:
            pts = self.sl_fixed_pts
        return max(self.sl_min_pts, min(self.sl_max_pts, pts))

    def next(self, i, df):
        bar   = df.iloc[i]
        t     = df.index[i]
        today = t.date()
        hour  = t.hour
        pt    = 0.01

        if today != self._day:
            self._new_day(today)

        # Day-of-week filter
        dow = t.dayofweek  # 0=Mon … 4=Fri
        if self.dow_filter == "mon-thu" and dow >= 4:
            return None
        if self.dow_filter == "no-fri" and dow == 4:
            return None

        h = self.h1_info.iloc[i]

        # Force-close
        if self._in_trade:
            self.peak_bal = max(self.peak_bal, self.balance)
            if t.hour >= 21 or (self.peak_bal - self.balance) / self.initial_bal >= self.max_dd_guard:
                return self._close()

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

        # ATR gate
        h1_atr = float(h["h1_atr"]) if not pd.isna(h["h1_atr"]) else 0
        if h1_atr < self.min_h1_atr:
            return None

        close  = bar["close"]
        spread = bar["spread"]
        trend  = int(h["h1_trend"])

        # ── ARB ───────────────────────────────────────────────────────────────
        if self.use_arb and 8 <= hour < 10 and not self._arb_done and today in self.arb_ranges.index:
            r = self.arb_ranges.loc[today]
            rng = float(r["range_pts"])
            if 500 <= rng <= 9000:
                if self.min_range_atr_r > 0 and h1_atr > 0:
                    if (rng * 0.01) < self.min_range_atr_r * h1_atr:
                        pass  # range too tight relative to ATR
                    else:
                        sl_pts = self._sl_pts(h1_atr, rng)
                        tp_pts = int(sl_pts * self.tp_mult)
                        if (not self.require_trend or trend >= 0) and close > r["high"] + self.buf:
                            e = close + (spread/2)*pt
                            self._sl = e - sl_pts*pt; self._tp = e + tp_pts*pt
                            self._in_trade = True; self._dir = "buy"
                            self._arb_done = True; return "buy"
                        if (not self.require_trend or trend <= 0) and close < r["low"] - self.buf:
                            e = close - (spread/2)*pt
                            self._sl = e + sl_pts*pt; self._tp = e - tp_pts*pt
                            self._in_trade = True; self._dir = "sell"
                            self._arb_done = True; return "sell"
                else:
                    sl_pts = self._sl_pts(h1_atr, rng)
                    tp_pts = int(sl_pts * self.tp_mult)
                    if (not self.require_trend or trend >= 0) and close > r["high"] + self.buf:
                        e = close + (spread/2)*pt
                        self._sl = e - sl_pts*pt; self._tp = e + tp_pts*pt
                        self._in_trade = True; self._dir = "buy"
                        self._arb_done = True; return "buy"
                    if (not self.require_trend or trend <= 0) and close < r["low"] - self.buf:
                        e = close - (spread/2)*pt
                        self._sl = e + sl_pts*pt; self._tp = e - tp_pts*pt
                        self._in_trade = True; self._dir = "sell"
                        self._arb_done = True; return "sell"

        # ── NYO ───────────────────────────────────────────────────────────────
        if self.use_nyo and 13 <= hour < 15 and not self._nyo_done and today in self.nyo_ranges.index:
            r = self.nyo_ranges.loc[today]
            rng = float(r["range_pts"])
            if 300 <= rng <= 7000:
                sl_pts = self._sl_pts(h1_atr, rng)
                tp_pts = int(sl_pts * self.tp_mult)
                if (not self.require_trend or trend >= 0) and close > r["high"] + self.buf:
                    e = close + (spread/2)*pt
                    self._sl = e - sl_pts*pt; self._tp = e + tp_pts*pt
                    self._in_trade = True; self._dir = "buy"
                    self._nyo_done = True; return "buy"
                if (not self.require_trend or trend <= 0) and close < r["low"] - self.buf:
                    e = close - (spread/2)*pt
                    self._sl = e + sl_pts*pt; self._tp = e - tp_pts*pt
                    self._in_trade = True; self._dir = "sell"
                    self._nyo_done = True; return "sell"

        return None

    def _close(self):
        self._in_trade = False
        self._dir = self._sl = self._tp = None
        return "close"


# ── Single-period backtest ────────────────────────────────────────────────────

def run_period(cfg: dict, df_slice, h1_slice, arb_ranges, nyo_ranges,
               balance=BALANCE, lots=LOTS):
    strat = HuntStrategy(**cfg, initial_balance=balance)
    strat.arb_ranges = arb_ranges
    strat.nyo_ranges = nyo_ranges
    strat.h1_info    = h1_slice

    bt     = Backtester(df_slice, strat, lots=lots, initial_balance=balance)
    report = bt.run()
    log    = report.trade_log()

    if log.empty:
        return None

    s = report.summary()
    log["date"]  = log["exit_time"].dt.date
    log["month"] = log["exit_time"].dt.to_period("M")
    monthly = log.groupby("month")["pnl"].sum()
    daily   = log.groupby("date")["pnl"].sum()
    pnls    = [t.pnl for t in report.trades]
    wins    = [p for p in pnls if p > 0]

    # Parse max_dd numeric
    max_dd_str = s["max_drawdown"]
    try:
        max_dd = float(max_dd_str.replace("$","").replace(",",""))
    except Exception:
        max_dd = 0

    return {
        "trades":      len(pnls),
        "win_rate":    len(wins)/len(pnls) if pnls else 0,
        "monthly_avg": monthly.mean(),
        "pf":          float(s["profit_factor"].replace("∞","99")) if isinstance(s["profit_factor"], str) else s["profit_factor"],
        "max_dd":      max_dd,
        "worst_day":   daily.min() / balance * 100,
        "net_pnl":     sum(pnls),
    }


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan(lots=LOTS, balance=BALANCE, top_n=20):
    print("Loading data …")
    df_full  = load("M5")
    print("Computing H1 ATR + trend …")
    h1_info  = compute_h1_atr(df_full)

    # Pre-slice data for each period
    slices = []
    for label, start, end in PERIODS:
        df_s  = df_full.loc[start:end].copy()
        h1_s  = h1_info.loc[start:end].copy()
        arb_r, nyo_r = compute_ranges(df_s)
        slices.append((label, df_s, h1_s, arb_r, nyo_r))

    # Parameter grid
    grid = list(product(
        ["fixed", "atr", "range"],     # sl_mode
        [800, 1200, 1500],             # sl_fixed_pts
        [1.5, 2.0, 2.5, 3.0],         # tp_mult
        [0.4, 0.6, 0.8],              # sl_atr_mult
        [True, False],                 # use_arb
        [True, False],                 # use_nyo
        [0.0, 5.0, 8.0],              # min_h1_atr
        [0.0, 0.3, 0.5],              # min_range_atr_r
        [True, False],                 # require_trend
        ["all", "mon-thu", "no-fri"], # dow_filter
    ))

    # Filter: must use at least one session
    grid = [g for g in grid if g[4] or g[5]]

    print(f"Testing {len(grid):,} configurations across {len(PERIODS)} periods …\n")

    results = []
    for (sl_mode, sl_fixed, tp_mult, sl_atr_m, use_arb, use_nyo,
         min_atr, min_rng_r, req_trend, dow) in grid:

        cfg = dict(
            sl_mode=sl_mode, sl_fixed_pts=sl_fixed, tp_mult=tp_mult,
            sl_atr_mult=sl_atr_m, use_arb=use_arb, use_nyo=use_nyo,
            min_h1_atr=min_atr, min_range_atr_r=min_rng_r,
            require_trend=req_trend, dow_filter=dow,
        )

        period_results = []
        ok = True
        for label, df_s, h1_s, arb_r, nyo_r in slices:
            pr = run_period(cfg, df_s, h1_s, arb_r, nyo_r, balance, lots)
            if pr is None:
                ok = False; break
            period_results.append((label, pr))

        if not ok:
            continue

        # Alpha criteria
        all_pf_ok      = all(r["pf"] >= 1.10          for _, r in period_results)
        all_monthly_ok = all(r["monthly_avg"] >= 0     for _, r in period_results)
        no_dd_fail     = all(r["max_dd"] >= -1500      for _, r in period_results)
        no_worst_day   = all(r["worst_day"] >= -5.0    for _, r in period_results)
        min_trades     = all(r["trades"] >= 5          for _, r in period_results)

        avg_monthly = np.mean([r["monthly_avg"] for _, r in period_results])
        avg_pf      = np.mean([r["pf"]          for _, r in period_results])
        min_pf      = min(r["pf"]               for _, r in period_results)
        worst_dd    = min(r["max_dd"]           for _, r in period_results)

        is_alpha = all_pf_ok and all_monthly_ok and no_dd_fail and no_worst_day and min_trades

        results.append({
            "cfg":         cfg,
            "is_alpha":    is_alpha,
            "avg_monthly": avg_monthly,
            "avg_pf":      avg_pf,
            "min_pf":      min_pf,
            "worst_dd":    worst_dd,
            "details":     period_results,
        })

    alphas = [r for r in results if r["is_alpha"]]
    alphas.sort(key=lambda r: r["avg_monthly"], reverse=True)

    print(f"\n{'='*75}")
    print(f"  ALPHA HUNT RESULTS")
    print(f"  {len(alphas)} alpha configs found out of {len(results):,} tested")
    print(f"{'='*75}\n")

    if not alphas:
        # Show best non-alpha configs anyway
        results.sort(key=lambda r: r["avg_monthly"], reverse=True)
        print("  No full alpha found. Top 10 closest:\n")
        for rank, r in enumerate(results[:10], 1):
            cfg = r["cfg"]
            issues = []
            for label, pr in r["details"]:
                if pr["pf"] < 1.10:      issues.append(f"{label[:10]} PF={pr['pf']:.2f}")
                if pr["monthly_avg"] < 0: issues.append(f"{label[:10]} mon=${pr['monthly_avg']:.0f}")
            print(f"  #{rank:2d}  sl={cfg['sl_mode']:<5} tp×{cfg['tp_mult']}  "
                  f"atr_m={cfg['sl_atr_mult']}  arb={cfg['use_arb']} nyo={cfg['use_nyo']}  "
                  f"min_atr={cfg['min_h1_atr']}  rng_r={cfg['min_range_atr_r']}  "
                  f"trend={cfg['require_trend']}  dow={cfg['dow_filter']}")
            print(f"       avg_monthly=${r['avg_monthly']:>7,.0f}  avg_pf={r['avg_pf']:.2f}  "
                  f"min_pf={r['min_pf']:.2f}  worst_dd=${r['worst_dd']:,.0f}")
            print(f"       issues: {'; '.join(issues[:3]) if issues else 'none'}\n")
        return results

    print(f"  {'Rank':<4} {'sl_mode':<6} {'tp×':<5} {'atr_m':<6} {'arb':<4} {'nyo':<4} "
          f"{'min_atr':<8} {'rng_r':<6} {'trend':<6} {'dow':<8} "
          f"{'avg_mo':>8} {'avg_pf':>7} {'min_pf':>7} {'worst_dd':>10}")
    print(f"  {'-'*105}")

    for rank, r in enumerate(alphas[:top_n], 1):
        cfg = r["cfg"]
        print(f"  {rank:<4} {cfg['sl_mode']:<6} {cfg['tp_mult']:<5} "
              f"{cfg['sl_atr_mult']:<6} {str(cfg['use_arb']):<4} {str(cfg['use_nyo']):<4} "
              f"{cfg['min_h1_atr']:<8} {cfg['min_range_atr_r']:<6} "
              f"{str(cfg['require_trend']):<6} {cfg['dow_filter']:<8} "
              f"${r['avg_monthly']:>7,.0f} {r['avg_pf']:>7.2f} "
              f"{r['min_pf']:>7.2f} ${r['worst_dd']:>9,.0f}")

    # Full detail on the best alpha
    best = alphas[0]
    print(f"\n{'='*75}")
    print(f"  BEST ALPHA — detailed breakdown")
    print(f"{'='*75}")
    cfg = best["cfg"]
    for k, v in cfg.items():
        print(f"    {k:<20} {v}")
    print()
    print(f"  {'Period':<22} {'Trades':>6} {'Win%':>6} {'Monthly':>9} "
          f"{'PF':>6} {'MaxDD':>10} {'WrstDay':>9}")
    print(f"  {'-'*75}")
    for label, pr in best["details"]:
        print(f"  {label:<22} {pr['trades']:>6} {pr['win_rate']*100:>5.0f}% "
              f"${pr['monthly_avg']:>8,.0f} {pr['pf']:>6.2f} "
              f"${pr['max_dd']:>9,.0f} {pr['worst_day']:>8.2f}%")
    print(f"  {'-'*75}")
    avgs = best["details"]
    print(f"  {'AVERAGE':<22} "
          f"{'':>6} {'':>6} "
          f"${best['avg_monthly']:>8,.0f} {best['avg_pf']:>6.2f} "
          f"${best['worst_dd']:>9,.0f}")

    return alphas


if __name__ == "__main__":
    alphas = scan(lots=LOTS, balance=BALANCE)
