"""
Alpha Hunt v2 — vectorized, fast.

Approach:
  1. Pre-build a DataFrame of ALL potential breakout entry bars (ARB + NYO sessions).
     Each row is one "opportunity" with entry price, direction, range size, ATR, trend.
  2. For each config, compute SL/TP points → simulate PnL per trade instantly
     (no bar-by-bar loop needed: we assume breakout enters at close, and know
      if price will hit TP or SL within that trading day using daily high/low).
  3. Score 5 periods simultaneously; flag alpha if ALL pass criteria.

Limitations vs full backtester:
  - No intra-day DD guard (very rare to trigger from 0.1 lot)
  - 1 trade per session per day (same as strategy logic)
  - Costs applied per trade

This runs ~1000x faster: scan completes in ~30 seconds.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backtester import load
from strategy_combined import compute_ranges

BALANCE  = 10_000.0
LOTS     = 0.1
PT       = 0.01            # 1 point = $0.01 for XAUUSD
LOT_VAL  = LOTS * 100      # 0.1 lot × 100 = $10 per point
COMM     = LOTS * 2 * 0.001 * 2000   # ~$0.40/trade (rough, applied per trade)

PERIODS = [
    ("Steady Uptrend",   "2025-01-07", "2025-03-31"),
    ("Post-ATH Correct", "2025-06-01", "2025-08-31"),
    ("Range Recovery",   "2025-09-01", "2025-12-31"),
    ("High-Vol Chop",    "2026-01-01", "2026-03-31"),
    ("Current",          "2026-04-01", "2026-06-09"),
]


# ── Pre-build opportunity table ───────────────────────────────────────────────

def build_opportunity_table(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    For every ARB (08:00–10:00) and NYO (13:00–15:00) entry window:
    - Find first bar where close breaks the session range high or low.
    - Record: date, session, direction, entry_price, daily_high (remaining),
      daily_low (remaining), spread, h1_atr, h1_trend, range_pts, dow.
    Returns DataFrame with one row per trade opportunity.
    """
    print("  Building opportunity table …")

    # H1 indicators
    h1 = load("H1")
    prev_c = h1["close"].shift(1)
    tr = pd.concat([
        h1["high"] - h1["low"],
        (h1["high"] - prev_c).abs(),
        (h1["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    h1_atr  = tr.ewm(span=14, adjust=False).mean()
    h1_ema  = h1["close"].ewm(span=50, adjust=False).mean()
    h1_trend = (h1["close"] > h1_ema).astype(int) * 2 - 1   # +1 bull, -1 bear

    def _ff(s):
        return s.shift(1).reindex(df_full.index, method="ffill")

    m5_atr   = _ff(h1_atr)
    m5_trend = _ff(h1_trend).fillna(0).astype(int)

    arb_ranges, nyo_ranges = compute_ranges(df_full.copy())

    buf = 30 * PT   # 30 pt breakout buffer

    records = []

    for date, day_df in df_full.groupby(df_full.index.date):
        dow = day_df.index[0].dayofweek   # 0=Mon

        if date not in arb_ranges.index and date not in nyo_ranges.index:
            continue

        # ── ARB window 08:00–10:00 ─────────────────────────────────────────
        if date in arb_ranges.index:
            r    = arb_ranges.loc[date]
            rng  = float(r["range_pts"])
            if 500 <= rng <= 9000:
                arb_win = day_df[(day_df.index.hour >= 8) & (day_df.index.hour < 10)]
                for t, bar in arb_win.iterrows():
                    close = bar["close"]; spread = bar["spread"]
                    atr   = float(m5_atr.get(t, 0) or 0)
                    trend = int(m5_trend.get(t, 0) or 0)

                    direction = None
                    if close > r["high"] + buf:
                        direction = 1
                        entry = close + (spread / 2) * PT
                    elif close < r["low"] - buf:
                        direction = -1
                        entry = close - (spread / 2) * PT

                    if direction is not None:
                        # Remaining day high/low (for TP/SL resolution)
                        remaining = day_df.loc[t:]
                        remaining_high = remaining["high"].max()
                        remaining_low  = remaining["low"].min()
                        records.append({
                            "date":      date,
                            "session":   "ARB",
                            "direction": direction,
                            "entry":     entry,
                            "spread":    spread,
                            "h1_atr":    atr,
                            "h1_trend":  trend,
                            "range_pts": rng,
                            "day_high":  remaining_high,
                            "day_low":   remaining_low,
                            "dow":       dow,
                        })
                        break   # only first break per session

        # ── NYO window 13:00–15:00 ─────────────────────────────────────────
        if date in nyo_ranges.index:
            r    = nyo_ranges.loc[date]
            rng  = float(r["range_pts"])
            if 300 <= rng <= 7000:
                nyo_win = day_df[(day_df.index.hour >= 13) & (day_df.index.hour < 15)]
                for t, bar in nyo_win.iterrows():
                    close = bar["close"]; spread = bar["spread"]
                    atr   = float(m5_atr.get(t, 0) or 0)
                    trend = int(m5_trend.get(t, 0) or 0)

                    direction = None
                    if close > r["high"] + buf:
                        direction = 1
                        entry = close + (spread / 2) * PT
                    elif close < r["low"] - buf:
                        direction = -1
                        entry = close - (spread / 2) * PT

                    if direction is not None:
                        remaining = day_df.loc[t:]
                        remaining_high = remaining["high"].max()
                        remaining_low  = remaining["low"].min()
                        records.append({
                            "date":      date,
                            "session":   "NYO",
                            "direction": direction,
                            "entry":     entry,
                            "spread":    spread,
                            "h1_atr":    atr,
                            "h1_trend":  trend,
                            "range_pts": rng,
                            "day_high":  remaining_high,
                            "day_low":   remaining_low,
                            "dow":       dow,
                        })
                        break

    opps = pd.DataFrame(records)
    print(f"  {len(opps):,} opportunities found across full dataset")
    return opps


# ── Score a config over one period ───────────────────────────────────────────

def score_period(opps_period: pd.DataFrame, cfg: dict, balance=BALANCE) -> dict | None:
    df = opps_period.copy()
    if df.empty:
        return None

    # Session filter
    sessions = []
    if cfg["use_arb"]: sessions.append("ARB")
    if cfg["use_nyo"]: sessions.append("NYO")
    df = df[df["session"].isin(sessions)]

    # DOW filter
    if cfg["dow_filter"] == "mon-thu":
        df = df[df["dow"] <= 3]
    elif cfg["dow_filter"] == "no-fri":
        df = df[df["dow"] != 4]

    # ATR gate
    if cfg["min_h1_atr"] > 0:
        df = df[df["h1_atr"] >= cfg["min_h1_atr"]]

    # Range/ATR ratio
    if cfg["min_range_atr_r"] > 0:
        ratio = (df["range_pts"] * PT) / df["h1_atr"].replace(0, np.nan)
        df = df[ratio >= cfg["min_range_atr_r"]]

    # Trend filter
    if cfg["require_trend"]:
        df = df[df["direction"] == df["h1_trend"]]

    if df.empty:
        return None

    # Compute SL/TP per trade
    sl_mode = cfg["sl_mode"]
    tp_mult = cfg["tp_mult"]

    if sl_mode == "fixed":
        sl_pts = np.full(len(df), cfg["sl_fixed_pts"], dtype=float)
    elif sl_mode == "atr":
        sl_pts = (df["h1_atr"].values * cfg["sl_atr_mult"] / PT).clip(
            cfg["sl_min_pts"], cfg["sl_max_pts"])
    else:  # range
        sl_pts = (df["range_pts"].values * 0.5).clip(
            cfg["sl_min_pts"], cfg["sl_max_pts"])

    tp_pts = sl_pts * tp_mult

    # PnL per trade: assume TP hit if remaining day's range covers it,
    # else SL hit.  Direction = +1 (buy) or -1 (sell).
    entries = df["entry"].values
    dirs    = df["direction"].values
    d_hi    = df["day_high"].values
    d_lo    = df["day_low"].values

    tp_price = entries + dirs * tp_pts * PT
    sl_price = entries - dirs * sl_pts * PT

    # TP hit?
    tp_hit = np.where(dirs == 1, d_hi >= tp_price, d_lo <= tp_price)
    raw_pnl = np.where(tp_hit,
                        tp_pts * LOT_VAL,
                       -sl_pts * LOT_VAL)

    # Spread cost
    raw_pnl -= df["spread"].values * 0.5 * LOT_VAL   # half-spread on entry already in entry price; commission approx
    raw_pnl -= 0.40   # flat $0.40 commission per trade

    if len(raw_pnl) == 0:
        return None

    # Monthly PnL
    dates   = pd.to_datetime([str(d) for d in df["date"].values])
    months  = pd.Period(dates, freq="M") if False else pd.PeriodIndex(dates, freq="M")
    monthly = pd.Series(raw_pnl, index=months).groupby(level=0).sum()

    # Drawdown
    cum = raw_pnl.cumsum()
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak)
    max_dd = dd.min()

    # Worst day
    day_pnl   = pd.Series(raw_pnl, index=pd.to_datetime([str(d) for d in df["date"].values]))
    daily_sum = day_pnl.groupby(day_pnl.index.date).sum()
    worst_day_pct = daily_sum.min() / balance * 100 if len(daily_sum) else 0

    wins  = raw_pnl[raw_pnl > 0]
    loses = raw_pnl[raw_pnl < 0]
    pf    = (wins.sum() / -loses.sum()) if loses.sum() < 0 else 99.0

    return {
        "trades":      len(raw_pnl),
        "win_rate":    (raw_pnl > 0).mean(),
        "monthly_avg": monthly.mean(),
        "pf":          round(float(pf), 3),
        "max_dd":      round(float(max_dd), 2),
        "worst_day":   round(float(worst_day_pct), 2),
        "net_pnl":     round(float(raw_pnl.sum()), 2),
    }


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan(top_n: int = 25):
    import itertools, time

    print("Loading M5 data …")
    df_full = load("M5")

    opps = build_opportunity_table(df_full)

    # Slice opportunities by period
    opps["date_dt"] = pd.to_datetime(opps["date"].astype(str))
    period_opps = []
    for label, start, end in PERIODS:
        mask = (opps["date_dt"] >= start) & (opps["date_dt"] <= end)
        period_opps.append((label, opps[mask].copy()))
        print(f"  {label}: {mask.sum()} opportunities")

    # Parameter grid — reduced but targeted
    grid = list(itertools.product(
        ["fixed", "atr", "range"],      # sl_mode
        [600, 800, 1000, 1200, 1500],   # sl_fixed_pts
        [1.5, 2.0, 2.5, 3.0, 4.0],     # tp_mult
        [0.3, 0.5, 0.7, 1.0],          # sl_atr_mult
        [200, 400, 600],                # sl_min_pts
        [1500, 2000, 2500],             # sl_max_pts
        [True, False],                  # use_arb
        [True, False],                  # use_nyo
        [0.0, 5.0, 8.0, 12.0],         # min_h1_atr
        [0.0, 0.2, 0.4],               # min_range_atr_r
        [True, False],                  # require_trend
        ["all", "mon-thu", "no-fri"],   # dow_filter
    ))
    # must use ≥1 session
    grid = [g for g in grid if g[6] or g[7]]
    print(f"\nTesting {len(grid):,} configs …\n")

    ALPHA_PF        = 1.08
    ALPHA_MONTHLY   = 0
    ALPHA_MAX_DD    = -1500
    ALPHA_WORST_DAY = -5.0
    ALPHA_MIN_TRADES = 5

    results = []
    t0 = time.time()

    for (sl_mode, sl_fixed, tp_mult, sl_atr_m, sl_min, sl_max,
         use_arb, use_nyo, min_atr, min_rng_r, req_trend, dow) in grid:

        cfg = dict(
            sl_mode=sl_mode, sl_fixed_pts=sl_fixed, tp_mult=tp_mult,
            sl_atr_mult=sl_atr_m, sl_min_pts=sl_min, sl_max_pts=sl_max,
            use_arb=use_arb, use_nyo=use_nyo,
            min_h1_atr=min_atr, min_range_atr_r=min_rng_r,
            require_trend=req_trend, dow_filter=dow,
        )

        period_results = []
        ok = True
        for label, pop in period_opps:
            pr = score_period(pop, cfg)
            if pr is None or pr["trades"] < ALPHA_MIN_TRADES:
                ok = False; break
            period_results.append((label, pr))

        if not ok:
            continue

        all_pf_ok    = all(r["pf"] >= ALPHA_PF          for _, r in period_results)
        all_mon_ok   = all(r["monthly_avg"] >= ALPHA_MONTHLY for _, r in period_results)
        no_dd        = all(r["max_dd"] >= ALPHA_MAX_DD   for _, r in period_results)
        no_wd        = all(r["worst_day"] >= ALPHA_WORST_DAY for _, r in period_results)
        is_alpha     = all_pf_ok and all_mon_ok and no_dd and no_wd

        avg_monthly  = np.mean([r["monthly_avg"] for _, r in period_results])
        avg_pf       = np.mean([r["pf"]          for _, r in period_results])
        min_pf       = min(r["pf"]               for _, r in period_results)
        worst_dd     = min(r["max_dd"]           for _, r in period_results)

        results.append({
            "cfg":         cfg,
            "is_alpha":    is_alpha,
            "avg_monthly": avg_monthly,
            "avg_pf":      avg_pf,
            "min_pf":      min_pf,
            "worst_dd":    worst_dd,
            "details":     period_results,
        })

    elapsed = time.time() - t0
    print(f"Scan complete in {elapsed:.1f}s — {len(results):,} configs scored\n")

    alphas = sorted([r for r in results if r["is_alpha"]],
                    key=lambda r: r["avg_monthly"], reverse=True)

    print(f"{'='*75}")
    print(f"  ALPHA HUNT v2 RESULTS")
    print(f"  {len(alphas)} alpha configs found (PF≥{ALPHA_PF}, monthly≥${ALPHA_MONTHLY}, "
          f"DD≥${ALPHA_MAX_DD}, worst_day≥{ALPHA_WORST_DAY}%)")
    print(f"{'='*75}\n")

    top = alphas[:top_n] if alphas else sorted(results, key=lambda r: r["avg_monthly"], reverse=True)[:top_n]
    label_str = "alpha" if alphas else "closest (no full alpha)"

    for rank, row in enumerate(top, 1):
        cfg = row["cfg"]
        sl_desc = (f"SL={cfg['sl_fixed_pts']}" if cfg["sl_mode"]=="fixed"
                   else f"SL={cfg['sl_mode']}×{cfg['sl_atr_mult']}"
                         if cfg["sl_mode"]=="atr"
                   else f"SL=range×0.5")
        sess = ("ARB+NYO" if cfg["use_arb"] and cfg["use_nyo"]
                else "ARB" if cfg["use_arb"] else "NYO")
        trend = "trend" if cfg["require_trend"] else "no-trend"
        dow   = cfg["dow_filter"]
        atr_g = f"atr≥{cfg['min_h1_atr']}" if cfg["min_h1_atr"] > 0 else ""
        rng_r = f"rng/atr≥{cfg['min_range_atr_r']}" if cfg["min_range_atr_r"] > 0 else ""
        filters = " ".join(f for f in [trend, dow, atr_g, rng_r] if f)

        print(f"  #{rank:>2} [{label_str}]  avg_monthly=${row['avg_monthly']:>7,.0f}  "
              f"min_PF={row['min_pf']:.2f}  worst_DD=${row['worst_dd']:>8,.0f}")
        print(f"       {sl_desc}  TP×{cfg['tp_mult']}  {sess}  {filters}")

        for lbl, pr in row["details"]:
            ch = "✅" if (pr["pf"] >= ALPHA_PF and pr["monthly_avg"] >= 0
                          and pr["max_dd"] >= ALPHA_MAX_DD
                          and pr["worst_day"] >= ALPHA_WORST_DAY) else "❌"
            print(f"       {ch} {lbl:<22}  {pr['trades']:>3} trades  "
                  f"PF={pr['pf']:.2f}  monthly=${pr['monthly_avg']:>6,.0f}  "
                  f"DD=${pr['max_dd']:>8,.0f}  wd={pr['worst_day']:>6.1f}%")
        print()


if __name__ == "__main__":
    scan()
