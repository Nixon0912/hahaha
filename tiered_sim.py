"""
Tiered Risk Simulation for The5ers Challenge

The5ers bust rule is STATIC: balance must never fall below $9,000 (initial - 10%).
This means once you are ahead, you have MORE buffer above the bust floor
and can afford to increase risk to accelerate toward the +8% target.

Two-phase strategy:
  Phase 1: trade at conservative risk until balance reaches threshold T1
  Phase 2: switch to higher risk to push toward +8% target
  Guard:   never risk more than (balance - bust_floor) × guard_frac on any trade

This exploits the asymmetry: bust floor is fixed, target is above.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings, glob
warnings.filterwarnings("ignore")

from multi_asset_scan import (
    load_raw, build_mtf,
    extract_arb, extract_nyo, extract_mom
)

RAW_DIR   = Path(__file__).parent
INIT_BAL  = 10_000.0
TARGET    = 10_800.0   # +8%
BUST_LVRL = 9_000.0    # static floor
MAX_DD    = INIT_BAL - BUST_LVRL   # $1000 drawdown budget from peak

N_SIM     = 8000


def load_stream(sym, arch):
    files = sorted(glob.glob(str(RAW_DIR / f"{sym}_M15_*.csv")))
    fpath = max(files, key=lambda f: Path(f).stat().st_size)
    m15   = load_raw(fpath)
    mtf   = build_mtf(m15)
    fn    = {"ARB": extract_arb, "NYO": extract_nyo, "MOM": extract_mom}[arch]
    return fn(m15, mtf)


def monte_carlo_tiered(Rs, tpm,
                       risk_phase1: float,
                       risk_phase2: float,
                       threshold_pct: float = 0.03,
                       n_sim: int = N_SIM) -> dict:
    """
    Phase 1: use risk_phase1 until balance >= INIT_BAL * (1 + threshold_pct)
    Phase 2: use risk_phase2 until target or bust
    Dynamic guard: risk capped so that one loss cannot wipe more than
                   50% of remaining buffer above bust floor.
    """
    n    = len(Rs)
    rng  = np.random.default_rng(42)
    draw = max(n * 5, 400)

    pc = bc = 0
    t2p = []

    for _ in range(n_sim):
        seq = rng.choice(Rs, size=draw, replace=True)
        bal = INIT_BAL
        passed = busted = False

        for k, r in enumerate(seq):
            # Determine phase
            if bal >= INIT_BAL * (1 + threshold_pct):
                base_risk = risk_phase2
            else:
                base_risk = risk_phase1

            # Dynamic guard: max loss on this trade ≤ 50% of buffer above bust
            buffer     = bal - BUST_LVRL
            max_loss   = buffer * 0.50
            risk_amt   = bal * base_risk
            # If r < 0 (a loss), actual loss = risk_amt * |r| ≤ max_loss
            # Clip risk_amt to max_loss
            risk_amt   = min(risk_amt, max_loss)
            effective_r = risk_amt / bal  # effective risk fraction

            bal += bal * effective_r * r

            if bal >= TARGET:
                pc += 1; t2p.append(k + 1); passed = True; break
            if bal <= BUST_LVRL:
                bc += 1; busted = True; break

        # count neither-pass-nor-bust as inconclusive

    med = float(np.median(t2p)) if t2p else np.nan
    return {
        "pass_pct": pc / n_sim * 100,
        "bust_pct": bc / n_sim * 100,
        "med_mo":   med / tpm if not np.isnan(med) else np.nan,
        "med_trades": med,
    }


def run():
    # Top-5 OOS survivors (expR ≥ 0.13)
    TOP5 = [
        ("ASXAUD", "NYO"),
        ("SP500",  "MOM"),
        ("USDCAD", "MOM"),
        ("USDJPY", "NYO"),
        ("XAGUSD", "ARB"),
    ]

    # Top-7 OOS survivors (expR ≥ 0.10)
    TOP7 = TOP5 + [("ESXEUR", "NYO"), ("USDJPY", "ARB")]

    print("Loading streams …")
    def get_trades(defs):
        all_t = []
        for sym, arch in defs:
            try:
                t = load_stream(sym, arch)
                all_t.extend(t)
                print(f"  {sym}-{arch}: {len(t)} trades")
            except Exception as e:
                print(f"  {sym}-{arch}: ERROR {e}")
        all_t.sort(key=lambda x: x["entry_t"])
        return all_t

    print("\nTop-5:")
    t5 = get_trades(TOP5)
    print("\nTop-7:")
    t7 = get_trades(TOP7)

    for label, trades in [("Top-5", t5), ("Top-7", t7)]:
        Rs  = np.array([t["R"] for t in trades])
        d0  = pd.Timestamp(trades[0]["date"])
        d1  = pd.Timestamp(trades[-1]["date"])
        tpm = len(trades) / max((d1-d0).days/30.44, 0.1)
        n   = len(trades)

        print(f"\n{'='*72}")
        print(f"  {label}: {n} trades  expR={Rs.mean():+.3f}  "
              f"WR={( Rs>0).mean()*100:.0f}%  {tpm:.1f} t/mo")
        print(f"{'='*72}")

        # ── Flat risk baseline ────────────────────────────────────────────
        print(f"\n  Flat risk (baseline):")
        print(f"  {'Risk%':>6}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}")
        from multi_asset_scan import monte_carlo
        for risk in [0.0020, 0.0025, 0.0030, 0.0035, 0.0040]:
            mc = monte_carlo(trades, risk)
            flag = " ***" if mc["bust_pct"]<=5 and mc["med_mo"]<=3.5 else ""
            print(f"  {risk*100:>5.2f}%  {mc['pass_pct']:>6.1f}%  "
                  f"{mc['bust_pct']:>5.1f}%  {mc['med_mo']:>6.1f}{flag}")

        # ── Tiered risk scan ──────────────────────────────────────────────
        print(f"\n  Tiered risk (Phase1 → Phase2 at +3% balance):")
        print(f"  {'P1 risk':>7}  {'P2 risk':>7}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}")
        targets_hit = []
        for r1 in [0.0015, 0.0020, 0.0025]:
            for r2 in [0.0025, 0.0030, 0.0035, 0.0040, 0.0050, 0.0060, 0.0070]:
                if r2 <= r1: continue
                mc = monte_carlo_tiered(Rs, tpm, r1, r2, threshold_pct=0.03)
                flag = ""
                if mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 3.5:
                    flag = "  *** TARGET ***"
                    targets_hit.append((label, r1, r2, mc))
                elif mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 5.0:
                    flag = "  ** close **"
                print(f"  {r1*100:>6.2f}%  {r2*100:>6.2f}%  "
                      f"{mc['pass_pct']:>6.1f}%  {mc['bust_pct']:>5.1f}%  "
                      f"{mc['med_mo']:>6.1f}{flag}")

        # ── Phase-2 trigger scan ──────────────────────────────────────────
        print(f"\n  Best P1=0.20%, P2=0.40% — trigger threshold scan:")
        print(f"  {'Trigger':>8}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}")
        for thr in [0.01, 0.02, 0.03, 0.04, 0.05]:
            mc = monte_carlo_tiered(Rs, tpm, 0.0020, 0.0040, threshold_pct=thr)
            flag = " ***" if mc["bust_pct"]<=5 and mc["med_mo"]<=3.5 else ""
            print(f"  +{thr*100:.0f}%     {mc['pass_pct']:>6.1f}%  "
                  f"{mc['bust_pct']:>5.1f}%  {mc['med_mo']:>6.1f}{flag}")


if __name__ == "__main__":
    run()
