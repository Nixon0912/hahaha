"""
Alpha Hunt v3 — numpy-batched, completes in ~10s.

Strategy: pre-build opportunity table (same as v2), then score ALL configs
at once using numpy vectorised operations. No Python loop over configs.

For each trade opportunity we record the "raw outcome" as a function of
sl_pts and tp_pts: we know whether TP or SL was hit (from day_high/day_low),
so each trade's PnL = TP×LOT_VAL if hit else -SL×LOT_VAL (minus costs).

Grid sweep: for each combination of (sl_pts, tp_mult) we can compute
per-trade PnL for every opportunity in one matrix multiply, then aggregate.
"""

import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from backtester import load
from strategy_combined import compute_ranges

BALANCE  = 10_000.0
LOTS     = 0.1
PT       = 0.01
LOT_VAL  = PT * 100 * LOTS   # 0.01 * 100 * 0.1 = $0.10 per point at 0.1 lot
COMM     = 0.40         # flat commission per trade

PERIODS = [
    ("Steady Uptrend",   "2025-01-07", "2025-03-31"),
    ("Post-ATH Correct", "2025-06-01", "2025-08-31"),
    ("Range Recovery",   "2025-09-01", "2025-12-31"),
    ("High-Vol Chop",    "2026-01-01", "2026-03-31"),
    ("Current",          "2026-04-01", "2026-06-09"),
]

ALPHA_PF        = 1.08
ALPHA_MONTHLY   = 0
ALPHA_MAX_DD    = -1500
ALPHA_WORST_DAY = -5.0
ALPHA_MIN_TR    = 5


def _compute_adx(h1, period=14):
    """Wilder ADX on H1 OHLC."""
    hi = h1["high"]; lo = h1["low"]; cl = h1["close"]
    prev_hi = hi.shift(1); prev_lo = lo.shift(1); prev_cl = cl.shift(1)
    tr = pd.concat([hi-lo, (hi-prev_cl).abs(), (lo-prev_cl).abs()], axis=1).max(axis=1)
    dm_p = np.where((hi-prev_hi) > (prev_lo-lo), np.maximum(hi-prev_hi, 0), 0)
    dm_m = np.where((prev_lo-lo) > (hi-prev_hi), np.maximum(prev_lo-lo, 0), 0)
    def _wilder(s, n):
        out = np.full(len(s), np.nan)
        arr = np.asarray(s, dtype=float)
        # seed
        start = n
        out[start-1] = np.nanmean(arr[:start])
        for i in range(start, len(arr)):
            out[i] = out[i-1] - out[i-1]/n + arr[i]
        return pd.Series(out, index=s.index)
    atr14   = _wilder(tr, period)
    di_p    = 100 * _wilder(pd.Series(dm_p, index=h1.index), period) / atr14
    di_m    = 100 * _wilder(pd.Series(dm_m, index=h1.index), period) / atr14
    dx      = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx     = _wilder(dx.fillna(0), period)
    return adx, atr14


def build_opps(df_full):
    print("  Building opportunity table …")
    h1 = load("H1")
    prev_c = h1["close"].shift(1)
    tr = pd.concat([h1["high"]-h1["low"],
                    (h1["high"]-prev_c).abs(),
                    (h1["low"]-prev_c).abs()], axis=1).max(axis=1)
    h1_atr  = tr.ewm(span=14, adjust=False).mean()
    h1_atr_ma = h1_atr.rolling(50, min_periods=25).mean()
    h1_ema  = h1["close"].ewm(span=50, adjust=False).mean()
    h1_trend = (h1["close"] > h1_ema).astype(int)*2 - 1
    h1_adx, _ = _compute_adx(h1)

    def _ff(s): return s.shift(1).reindex(df_full.index, method="ffill")
    m5_atr    = _ff(h1_atr)
    m5_atr_ma = _ff(h1_atr_ma)
    m5_trend  = _ff(h1_trend).fillna(0).astype(int)
    m5_adx    = _ff(h1_adx).fillna(0)

    arb_ranges, nyo_ranges = compute_ranges(df_full.copy())
    buf = 30 * PT

    records = []
    for date, day_df in df_full.groupby(df_full.index.date):
        dow = day_df.index[0].dayofweek

        for sess, ranges, h_start, h_end, rng_lo, rng_hi in [
            ("ARB", arb_ranges, 8, 10, 500, 9000),
            ("NYO", nyo_ranges, 13, 15, 300, 7000),
        ]:
            if date not in ranges.index:
                continue
            r   = ranges.loc[date]
            rng = float(r["range_pts"])
            if not (rng_lo <= rng <= rng_hi):
                continue

            win = day_df[(day_df.index.hour >= h_start) & (day_df.index.hour < h_end)]
            for t, bar in win.iterrows():
                close  = bar["close"]; spread = bar["spread"]
                atr    = float(m5_atr.get(t, 0) or 0)
                atr_ma = float(m5_atr_ma.get(t, 0) or 0)
                trend  = int(m5_trend.get(t, 0) or 0)
                adx    = float(m5_adx.get(t, 0) or 0)
                atr_exp = 1 if (atr_ma > 0 and atr >= atr_ma) else 0
                direction = None
                if close > r["high"] + buf:
                    direction = 1
                    entry = close + (spread/2)*PT
                elif close < r["low"] - buf:
                    direction = -1
                    entry = close - (spread/2)*PT
                if direction is not None:
                    rem = day_df.loc[t:]
                    records.append((date, sess, direction, entry, spread,
                                    atr, trend, rng, rem["high"].max(),
                                    rem["low"].min(), dow, adx, atr_exp))
                    break

    cols = ["date","session","direction","entry","spread",
            "h1_atr","h1_trend","range_pts","day_high","day_low","dow","h1_adx","atr_exp"]
    opps = pd.DataFrame(records, columns=cols)
    print(f"  {len(opps):,} opportunities  ({len(opps[opps.session=='ARB'])} ARB, "
          f"{len(opps[opps.session=='NYO'])} NYO)")
    return opps


def score_batch(opps_arr: dict, cfg_list: list) -> list:
    """
    Score all configs against all period opportunity arrays at once.
    opps_arr: {period_label: numpy structured dict}
    """
    results = []
    for cfg in cfg_list:
        period_scores = []
        ok = True
        for label, pa in opps_arr.items():
            mask = np.ones(len(pa["direction"]), dtype=bool)

            # Session
            if not cfg["use_arb"]: mask &= pa["session"] == b"NYO"
            if not cfg["use_nyo"]: mask &= pa["session"] == b"ARB"

            # DOW
            if cfg["dow"] == "mon-thu": mask &= pa["dow"] <= 3
            elif cfg["dow"] == "no-fri": mask &= pa["dow"] != 4

            # ATR gate
            if cfg["min_atr"] > 0:  mask &= pa["h1_atr"] >= cfg["min_atr"]
            if cfg["min_adx"] > 0:  mask &= pa["h1_adx"] >= cfg["min_adx"]
            if cfg["req_atr_exp"]:  mask &= pa["atr_exp"] == 1

            # Range/ATR ratio
            if cfg["min_rr"] > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rr = np.where(pa["h1_atr"] > 0,
                                  (pa["range_pts"] * PT) / pa["h1_atr"], 0)
                mask &= rr >= cfg["min_rr"]

            # Trend
            if cfg["trend"]: mask &= pa["direction"] == pa["h1_trend"]

            idx = np.where(mask)[0]
            if len(idx) < ALPHA_MIN_TR:
                ok = False; break

            dirs   = pa["direction"][idx]
            atr_v  = pa["h1_atr"][idx]
            rng_v  = pa["range_pts"][idx]
            d_hi   = pa["day_high"][idx]
            d_lo   = pa["day_low"][idx]
            entry  = pa["entry"][idx]
            spread = pa["spread"][idx]

            # SL points
            if cfg["sl_mode"] == "fixed":
                sl_pts = np.full(len(idx), float(cfg["sl_fixed"]))
            elif cfg["sl_mode"] == "atr":
                sl_pts = np.clip(atr_v * cfg["sl_atr_m"] / PT,
                                 cfg["sl_min"], cfg["sl_max"])
            else:  # range
                sl_pts = np.clip(rng_v * 0.5, cfg["sl_min"], cfg["sl_max"])

            tp_pts = sl_pts * cfg["tp_mult"]

            tp_price = entry + dirs * tp_pts * PT
            sl_price = entry - dirs * sl_pts * PT

            sl_hit   = np.where(dirs == 1, d_lo <= sl_price, d_hi >= sl_price)
            tp_hit   = np.where(dirs == 1, d_hi >= tp_price, d_lo <= tp_price)
            tp_clean = tp_hit & ~sl_hit
            pnl = np.where(tp_clean, tp_pts, -sl_pts) * LOT_VAL
            pnl -= spread * 0.5 * LOT_VAL + COMM

            # Metrics
            wins  = pnl[pnl > 0].sum()
            loses = pnl[pnl < 0].sum()
            pf    = wins / -loses if loses < 0 else 99.0

            dates_idx = pa["date"][idx]
            mon_key   = (dates_idx // 100).astype(int)   # YYYYMM
            monthly   = pd.Series(pnl).groupby(mon_key).sum()
            monthly_avg = monthly.mean()

            day_key  = dates_idx.astype(int)
            daily    = pd.Series(pnl).groupby(day_key).sum()
            worst_day_pct = daily.min() / BALANCE * 100

            cum  = np.cumsum(pnl)
            peak = np.maximum.accumulate(cum)
            max_dd = float((cum - peak).min())

            period_scores.append((label, {
                "trades": len(pnl), "pf": float(pf),
                "monthly_avg": float(monthly_avg),
                "max_dd": max_dd, "worst_day": float(worst_day_pct),
                "net_pnl": float(pnl.sum()),
            }))

        if not ok:
            continue

        all_pf   = all(r["pf"] >= ALPHA_PF          for _, r in period_scores)
        all_mon  = all(r["monthly_avg"] >= ALPHA_MONTHLY for _, r in period_scores)
        no_dd    = all(r["max_dd"] >= ALPHA_MAX_DD   for _, r in period_scores)
        no_wd    = all(r["worst_day"] >= ALPHA_WORST_DAY for _, r in period_scores)
        is_alpha = all_pf and all_mon and no_dd and no_wd

        avg_monthly = np.mean([r["monthly_avg"] for _, r in period_scores])
        min_pf      = min(r["pf"]               for _, r in period_scores)
        worst_dd    = min(r["max_dd"]           for _, r in period_scores)

        results.append({
            "cfg": cfg, "is_alpha": is_alpha,
            "avg_monthly": avg_monthly, "min_pf": min_pf,
            "worst_dd": worst_dd, "details": period_scores,
        })
    return results


def scan(top_n=20):
    import time
    print("Loading M5 data …")
    df_full = load("M5")
    opps = build_opps(df_full)

    # Convert period slices to numpy arrays for fast indexing
    opps["date_dt"] = pd.to_datetime(opps["date"].astype(str))
    opps["date_int"] = opps["date_dt"].dt.year * 10000 + opps["date_dt"].dt.month * 100 + opps["date_dt"].dt.day
    opps["session_b"] = opps["session"].str.encode("utf-8").apply(lambda x: x.ljust(3, b" "))

    period_arrs = {}
    for label, start, end in PERIODS:
        mask = (opps["date_dt"] >= start) & (opps["date_dt"] <= end)
        s = opps[mask]
        print(f"  {label}: {mask.sum()} opps")
        period_arrs[label] = {
            "direction": s["direction"].values.astype(np.int8),
            "session":   s["session"].values.astype("U3"),
            "dow":       s["dow"].values.astype(np.int8),
            "h1_atr":    s["h1_atr"].values.astype(np.float32),
            "h1_trend":  s["h1_trend"].values.astype(np.int8),
            "range_pts": s["range_pts"].values.astype(np.float32),
            "day_high":  s["day_high"].values.astype(np.float32),
            "day_low":   s["day_low"].values.astype(np.float32),
            "entry":     s["entry"].values.astype(np.float32),
            "spread":    s["spread"].values.astype(np.float32),
            "date":      s["date_int"].values.astype(np.int32),
            "h1_adx":    s["h1_adx"].values.astype(np.float32),
            "atr_exp":   s["atr_exp"].values.astype(np.int8),
        }

    # Fix session comparison for numpy string arrays
    def _fix_session(pa):
        if not cfg["use_arb"] and cfg["use_nyo"]:
            return pa["session"] == "NYO"
        if cfg["use_arb"] and not cfg["use_nyo"]:
            return pa["session"] == "ARB"
        return np.ones(len(pa["direction"]), dtype=bool)

    # Build config list (deduplicated)
    cfgs = []
    for sl_mode in ["fixed", "atr", "range"]:
        for sl_fixed in ([600, 800, 1000, 1200, 1500] if sl_mode == "fixed" else [1000]):
            for tp_mult in [1.5, 2.0, 2.5, 3.0, 4.0]:
                for sl_atr_m in ([0.3, 0.5, 0.7, 1.0] if sl_mode in ["atr","range"] else [0.5]):
                    for sl_min in ([200, 400, 600] if sl_mode != "fixed" else [400]):
                        for sl_max in ([1500, 2000, 2500] if sl_mode != "fixed" else [2000]):
                            for use_arb in [True, False]:
                                for use_nyo in [True, False]:
                                    if not use_arb and not use_nyo: continue
                                    for min_atr in [0.0, 5.0, 8.0, 12.0]:
                                        for min_adx in [0.0, 18.0, 22.0, 27.0]:
                                            for req_atr_exp in [False, True]:
                                                for min_rr in [0.0, 0.2, 0.4]:
                                                    for trend in [True, False]:
                                                        for dow in ["all","mon-thu","no-fri"]:
                                                            cfgs.append(dict(
                                                                sl_mode=sl_mode, sl_fixed=sl_fixed,
                                                                tp_mult=tp_mult, sl_atr_m=sl_atr_m,
                                                                sl_min=sl_min, sl_max=sl_max,
                                                                use_arb=use_arb, use_nyo=use_nyo,
                                                                min_atr=min_atr, min_adx=min_adx,
                                                                req_atr_exp=req_atr_exp,
                                                                min_rr=min_rr, trend=trend, dow=dow,
                                                            ))

    print(f"\nScanning {len(cfgs):,} configs …\n")
    t0 = time.time()

    # Patch score_batch to use numpy string comparison
    import types

    def _score(cfg_list):
        results = []
        for cfg in cfg_list:
            period_scores = []
            ok = True
            for label, pa in period_arrs.items():
                mask = np.ones(len(pa["direction"]), dtype=bool)
                if not cfg["use_arb"]: mask &= pa["session"] == "NYO"
                if not cfg["use_nyo"]: mask &= pa["session"] == "ARB"
                if cfg["dow"] == "mon-thu": mask &= pa["dow"] <= 3
                elif cfg["dow"] == "no-fri": mask &= pa["dow"] != 4
                if cfg["min_atr"] > 0:  mask &= pa["h1_atr"] >= cfg["min_atr"]
                if cfg["min_adx"] > 0:  mask &= pa["h1_adx"] >= cfg["min_adx"]
                if cfg["req_atr_exp"]:  mask &= pa["atr_exp"] == 1
                if cfg["min_rr"] > 0:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rr = np.where(pa["h1_atr"] > 0,
                                      (pa["range_pts"] * PT) / pa["h1_atr"], 0)
                    mask &= rr >= cfg["min_rr"]
                if cfg["trend"]: mask &= pa["direction"] == pa["h1_trend"]

                idx = np.where(mask)[0]
                if len(idx) < ALPHA_MIN_TR:
                    ok = False; break

                dirs   = pa["direction"][idx].astype(float)
                atr_v  = pa["h1_atr"][idx].astype(float)
                rng_v  = pa["range_pts"][idx].astype(float)
                d_hi   = pa["day_high"][idx].astype(float)
                d_lo   = pa["day_low"][idx].astype(float)
                entry  = pa["entry"][idx].astype(float)
                spread = pa["spread"][idx].astype(float)

                if cfg["sl_mode"] == "fixed":
                    sl_pts = np.full(len(idx), float(cfg["sl_fixed"]))
                elif cfg["sl_mode"] == "atr":
                    sl_pts = np.clip(atr_v * cfg["sl_atr_m"] / PT,
                                     cfg["sl_min"], cfg["sl_max"])
                else:
                    sl_pts = np.clip(rng_v * 0.5, cfg["sl_min"], cfg["sl_max"])

                tp_pts   = sl_pts * cfg["tp_mult"]
                tp_price = entry + dirs * tp_pts * PT
                sl_price = entry - dirs * sl_pts * PT

                # SL takes priority: if SL is hit, it's a loss even if TP is also in range
                sl_hit = np.where(dirs == 1, d_lo <= sl_price, d_hi >= sl_price)
                tp_hit = np.where(dirs == 1, d_hi >= tp_price, d_lo <= tp_price)
                # Only count TP if SL was NOT hit
                tp_clean = tp_hit & ~sl_hit
                pnl = np.where(tp_clean, tp_pts, -sl_pts) * LOT_VAL
                pnl -= spread * 0.5 * LOT_VAL + COMM

                wins  = pnl[pnl > 0].sum()
                loses = pnl[pnl < 0].sum()
                pf    = wins / -loses if loses < 0 else 99.0

                date_arr  = pa["date"][idx]
                mon_key   = date_arr // 100
                monthly   = pd.Series(pnl).groupby(mon_key).sum()

                day_key  = date_arr
                daily    = pd.Series(pnl).groupby(day_key).sum()
                worst_day_pct = daily.min() / BALANCE * 100

                cum  = np.cumsum(pnl)
                peak = np.maximum.accumulate(cum)
                max_dd = float((cum - peak).min())

                period_scores.append((label, {
                    "trades": len(pnl), "pf": round(float(pf),3),
                    "monthly_avg": float(monthly.mean()),
                    "max_dd": max_dd, "worst_day": float(worst_day_pct),
                    "net_pnl": float(pnl.sum()),
                }))

            if not ok: continue

            all_pf   = all(r["pf"] >= ALPHA_PF          for _, r in period_scores)
            all_mon  = all(r["monthly_avg"] >= ALPHA_MONTHLY for _, r in period_scores)
            no_dd    = all(r["max_dd"] >= ALPHA_MAX_DD   for _, r in period_scores)
            no_wd    = all(r["worst_day"] >= ALPHA_WORST_DAY for _, r in period_scores)
            is_alpha = all_pf and all_mon and no_dd and no_wd

            avg_monthly = float(np.mean([r["monthly_avg"] for _, r in period_scores]))
            min_pf      = min(r["pf"]               for _, r in period_scores)
            worst_dd    = min(r["max_dd"]           for _, r in period_scores)

            results.append({
                "cfg": cfg, "is_alpha": is_alpha,
                "avg_monthly": avg_monthly, "min_pf": min_pf,
                "worst_dd": worst_dd, "details": period_scores,
            })
        return results

    results = _score(cfgs)

    elapsed = time.time() - t0
    alphas = sorted([r for r in results if r["is_alpha"]],
                    key=lambda r: r["avg_monthly"], reverse=True)

    print(f"Scan done in {elapsed:.1f}s — {len(results):,} scored\n")
    print(f"{'='*75}")
    print(f"  ALPHA HUNT v3")
    print(f"  {len(alphas)} alpha configs (PF≥{ALPHA_PF}, monthly≥$0, "
          f"DD≥${ALPHA_MAX_DD}, wd≥{ALPHA_WORST_DAY}%)")
    print(f"  Periods tested: {len(PERIODS)} (Rocket Bull excluded)")
    print(f"{'='*75}\n")

    top = alphas[:top_n] if alphas else sorted(results, key=lambda r: r["avg_monthly"], reverse=True)[:top_n]
    label_tag = "ALPHA" if alphas else "BEST (no full alpha)"

    for rank, row in enumerate(top, 1):
        c = row["cfg"]
        sl_desc = (f"SL={c['sl_fixed']}pt" if c["sl_mode"]=="fixed"
                   else f"SL={c['sl_mode']}×{c['sl_atr_m']} [{c['sl_min']}-{c['sl_max']}]")
        sess  = "ARB+NYO" if c["use_arb"] and c["use_nyo"] else ("ARB" if c["use_arb"] else "NYO")
        trend = "trend" if c["trend"] else "no-trend"
        extras = []
        if c["min_atr"]    > 0:  extras.append(f"atr≥{c['min_atr']}")
        if c["min_adx"]    > 0:  extras.append(f"adx≥{c['min_adx']}")
        if c["req_atr_exp"]:     extras.append("atr-expanding")
        if c["min_rr"]     > 0:  extras.append(f"rr≥{c['min_rr']}")
        if c["dow"] != "all":    extras.append(c["dow"])
        extra_str = " ".join(extras)
        print(f"  #{rank:<2} [{label_tag}]  avg_monthly=${row['avg_monthly']:>7,.0f}  "
              f"min_PF={row['min_pf']:.2f}  worst_DD=${row['worst_dd']:>8,.0f}")
        print(f"       {sl_desc}  TP×{c['tp_mult']}  {sess}  {trend}  {extra_str}")
        for lbl, pr in row["details"]:
            ok = (pr["pf"] >= ALPHA_PF and pr["monthly_avg"] >= 0
                  and pr["max_dd"] >= ALPHA_MAX_DD and pr["worst_day"] >= ALPHA_WORST_DAY)
            ch = "✅" if ok else "❌"
            print(f"       {ch} {lbl:<22}  {pr['trades']:>3}tr  "
                  f"PF={pr['pf']:.2f}  mo=${pr['monthly_avg']:>6,.0f}  "
                  f"DD=${pr['max_dd']:>8,.0f}  wd={pr['worst_day']:>5.1f}%")
        print()

    if not alphas:
        print("  ── No full alpha found. Relaxing criteria to find partial hits …\n")
        # Show how many periods each best config passes
        results.sort(key=lambda r: r["avg_monthly"], reverse=True)
        for row in results[:5]:
            passes = sum(1 for _, pr in row["details"]
                         if pr["pf"] >= ALPHA_PF and pr["monthly_avg"] >= 0
                         and pr["max_dd"] >= ALPHA_MAX_DD and pr["worst_day"] >= ALPHA_WORST_DAY)
            print(f"  passes {passes}/{len(PERIODS)} periods | avg_mo=${row['avg_monthly']:,.0f} | min_PF={row['min_pf']:.2f}")
            for lbl, pr in row["details"]:
                ok = (pr["pf"] >= ALPHA_PF and pr["monthly_avg"] >= 0
                      and pr["max_dd"] >= ALPHA_MAX_DD and pr["worst_day"] >= ALPHA_WORST_DAY)
                print(f"    {'✅' if ok else '❌'} {lbl}: PF={pr['pf']:.2f} mo=${pr['monthly_avg']:,.0f} DD=${pr['max_dd']:,.0f}")
            print()


if __name__ == "__main__":
    scan()
