"""
Portfolio refinement — top-expR streams only + per-stream Kelly weighting.

From the full scan, 9 OOS survivors. Two weakest streams:
  DAX40-ARB  expR +0.055  drag down average
  UK100-ARB  expR +0.068  drag down average

This script re-runs Monte Carlo on:
  A) Top-7 only (drop DAX40 + UK100)
  B) Top-5 only (drop also ESXEUR-NYO + USDJPY-ARB, keep highest-expR)
  C) Top-7 with per-stream Kelly risk scaling (weight by expR)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings, glob, re
warnings.filterwarnings("ignore")

# Reuse extractors from multi_asset_scan
from multi_asset_scan import (
    load_raw, build_mtf,
    extract_arb, extract_nyo, extract_mom,
    oos_split, edge_ok, monte_carlo,
    sym_from_path, EXCLUDE
)

RAW_DIR  = Path(__file__).parent
INIT_BAL = 10_000.0


def load_stream(sym: str, arch: str) -> list:
    files = sorted(glob.glob(str(RAW_DIR / f"{sym}_M15_*.csv")))
    if not files:
        raise FileNotFoundError(sym)
    fpath = max(files, key=lambda f: Path(f).stat().st_size)
    m15 = load_raw(fpath)
    mtf = build_mtf(m15)
    fn  = {"ARB": extract_arb, "NYO": extract_nyo, "MOM": extract_mom}[arch]
    return fn(m15, mtf)


def monte_carlo_kelly(streams: list, base_risk: float, n_sim=5000) -> dict:
    """
    Per-stream Kelly-weighted sizing: risk_i = base_risk * (expR_i / max_expR).
    Each trade carries its stream's own risk fraction.
    """
    if not streams:
        return {}

    # Compute per-stream expR weight
    expRs   = np.array([np.mean([t["R"] for t in s["trades"]]) for s in streams])
    max_expR = expRs.max()
    weights  = expRs / max_expR   # normalise so best stream gets base_risk

    # Flatten trades with per-trade risk fraction
    tagged = []
    for s, w in zip(streams, weights):
        for t in s["trades"]:
            tagged.append({"R": t["R"], "risk": base_risk * w,
                           "date": t["date"], "entry_t": t["entry_t"]})
    tagged.sort(key=lambda x: x["entry_t"])

    n   = len(tagged)
    Rs  = np.array([t["R"]    for t in tagged])
    Rk  = np.array([t["risk"] for t in tagged])
    d0  = pd.Timestamp(tagged[0]["date"])
    d1  = pd.Timestamp(tagged[-1]["date"])
    tpm = n / max((d1 - d0).days / 30.44, 0.1)

    rng  = np.random.default_rng(42)
    draw = max(n * 4, 300)
    pc = bc = 0; t2p = []

    for _ in range(n_sim):
        idx = rng.choice(len(tagged), size=draw, replace=True)
        bal = peak = INIT_BAL; done = False
        for k, ii in enumerate(idx):
            bal  += bal * Rk[ii] * Rs[ii]
            peak  = max(peak, bal)
            if bal >= INIT_BAL * 1.08:
                pc += 1; t2p.append(k + 1); done = True; break
            if bal <= INIT_BAL * 0.90 or peak - bal >= INIT_BAL * 0.10:
                bc += 1; done = True; break

    med = float(np.median(t2p)) if t2p else np.nan
    return {
        "pass_pct": pc / n_sim * 100, "bust_pct": bc / n_sim * 100,
        "med_mo":   med / tpm if not np.isnan(med) else np.nan,
        "tpm": tpm, "expR": float(Rs.mean()), "n": n,
        "weights": {s["key"]: round(w, 3) for s, w in zip(streams, weights)},
    }


def run():
    # ── Stream definitions ────────────────────────────────────────────────
    STREAM_DEFS = [
        ("ASXAUD", "NYO", +0.145),
        ("DAX40",  "ARB", +0.055),
        ("ESXEUR", "NYO", +0.102),
        ("SP500",  "MOM", +0.134),
        ("UK100",  "ARB", +0.068),
        ("USDCAD", "MOM", +0.194),
        ("USDJPY", "ARB", +0.107),
        ("USDJPY", "NYO", +0.206),
        ("XAGUSD", "ARB", +0.140),
    ]

    print("Loading all 9 survivor streams …")
    streams = []
    for sym, arch, exp_r in STREAM_DEFS:
        try:
            trades = load_stream(sym, arch)
            streams.append({"sym": sym, "arch": arch, "expR": exp_r,
                            "trades": trades, "key": f"{sym}-{arch}"})
            print(f"  {sym:<8} {arch}  {len(trades):>3} trades  expR={exp_r:+.3f}")
        except Exception as e:
            print(f"  {sym} {arch}: ERROR — {e}")

    # ── Portfolio subsets ──────────────────────────────────────────────────
    top7 = [s for s in streams if s["expR"] >= 0.10]
    top5 = [s for s in streams if s["expR"] >= 0.13]

    portfolios = [
        ("All-9  (full)",  streams),
        ("Top-7  (≥0.10)", top7),
        ("Top-5  (≥0.13)", top5),
    ]

    print(f"\n{'='*80}")
    print(f"  Portfolio Comparison — Flat Risk Monte Carlo")
    print(f"{'='*80}")
    print(f"  {'Portfolio':<22}  {'Trades':>6}  {'expR':>6}  "
          f"{'t/mo':>5}  {'Risk%':>6}  {'Pass%':>6}  {'Bust%':>6}  {'Med.Mo':>7}")
    print(f"  {'-'*78}")

    for label, pool in portfolios:
        all_t = sorted([t for s in pool for t in s["trades"]],
                       key=lambda x: x["entry_t"])
        if not all_t: continue
        Rs = [t["R"] for t in all_t]
        d0 = pd.Timestamp(all_t[0]["date"])
        d1 = pd.Timestamp(all_t[-1]["date"])
        tpm = len(all_t) / max((d1 - d0).days / 30.44, 0.1)
        expR = float(np.mean(Rs))

        for risk in [0.0015, 0.0020, 0.0025]:
            mc = monte_carlo(all_t, risk)
            flag = ""
            if mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 3.0:
                flag = "  *** TARGET ***"
            elif mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 4.5:
                flag = "  ** close **"
            print(f"  {label:<22}  {len(all_t):>6}  {expR:>+6.3f}  "
                  f"{tpm:>4.1f}  {risk*100:>5.2f}%  "
                  f"{mc['pass_pct']:>5.1f}%  {mc['bust_pct']:>5.1f}%  "
                  f"{mc['med_mo']:>6.1f}{flag}")

    # ── Kelly-weighted scan ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  Kelly-Weighted Risk (per-stream, proportional to expR)")
    print(f"{'='*80}")
    print(f"  {'Portfolio':<22}  {'Base risk':>9}  {'Pass%':>6}  {'Bust%':>6}  {'Med.Mo':>7}")
    print(f"  {'-'*60}")

    for label, pool in portfolios:
        if not pool: continue
        for base_risk in [0.0015, 0.0020, 0.0025, 0.0030]:
            mc = monte_carlo_kelly(pool, base_risk)
            if not mc: continue
            flag = ""
            if mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 3.0:
                flag = "  *** TARGET ***"
            elif mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 4.0:
                flag = "  ** close **"
            print(f"  {label:<22}  {base_risk*100:>8.2f}%  "
                  f"{mc['pass_pct']:>5.1f}%  {mc['bust_pct']:>5.1f}%  "
                  f"{mc['med_mo']:>6.1f}{flag}")
        print()

    # ── Best portfolio full risk scan ──────────────────────────────────────
    best_pool = top5 if len(top5) >= 4 else top7
    best_label = "Top-5" if len(top5) >= 4 else "Top-7"
    all_t = sorted([t for s in best_pool for t in s["trades"]],
                   key=lambda x: x["entry_t"])
    Rs = [t["R"] for t in all_t]
    d0 = pd.Timestamp(all_t[0]["date"])
    d1 = pd.Timestamp(all_t[-1]["date"])
    tpm = len(all_t) / max((d1 - d0).days / 30.44, 0.1)

    print(f"\n{'='*80}")
    print(f"  Full Risk Scan — {best_label} Portfolio")
    print(f"  {len(all_t)} trades  expR={np.mean(Rs):+.3f}  WR={sum(r>0 for r in Rs)/len(Rs)*100:.0f}%  {tpm:.1f} t/mo")
    print(f"{'='*80}")
    print(f"  {'Risk%':>6}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}")
    sweet = None
    for risk in [0.0010, 0.0015, 0.0018, 0.0020, 0.0022, 0.0025,
                 0.0028, 0.0030, 0.0035, 0.0040, 0.0050]:
        mc = monte_carlo(all_t, risk)
        flag = ""
        if mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 3.0:
            flag = "  *** SWEET SPOT ***"
            sweet = (risk, mc)
        elif mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 4.0:
            flag = "  ** close **"
        print(f"  {risk*100:>5.2f}%  {mc['pass_pct']:>6.1f}%  {mc['bust_pct']:>5.1f}%  "
              f"{mc['med_mo']:>6.1f}{flag}")

    if sweet:
        risk, mc = sweet
        print(f"\n  ✅ TARGET HIT: {risk*100:.2f}% risk/trade")
        print(f"     Pass {mc['pass_pct']:.1f}%  Bust {mc['bust_pct']:.1f}%  "
              f"Median {mc['med_mo']:.1f} months")
        print(f"     Streams: " + ", ".join(f"{s['sym']}-{s['arch']}" for s in best_pool))
    else:
        print(f"\n  ⚠️  Gap remains. Closest to target:")
        print(f"     0.20%: ~5-6% bust, ~8-9 months")
        print(f"     Consider per-stream Kelly weighting (see table above).")


if __name__ == "__main__":
    run()
