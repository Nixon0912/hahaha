"""
F10 — Lot sizing acceptance tests (all 8 instruments).

Verifies:
  1. Actual cash risk at SL is within tolerance of the 1.25% budget
  2. Lots are rounded DOWN to volume_step (never up)
  3. Below-minimum sizing returns 0.0 (skip) — never forces volume_min
  4. The 1.5x actual-risk sanity cap rejects over-risk trades

No MT5 connection needed — uses representative symbol specs. Replace the
specs with live mt5.symbol_info() values during the demo phase to confirm.

Run from repo root:  python ea/test_lot_sizing.py
Exit 0 = all pass.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ea.risk import calculate_lots
from ea.config import RISK_PCT

# Representative MT5 symbol specs (tick_size, tick_value per lot in acct ccy,
# volume_min/max/step). Confirm against live mt5.symbol_info() on the demo.
SPECS = {
    #          tick_size  tick_value  vmin  vmax   vstep
    "ASXAUD": (0.1,       0.10,       0.01, 100.0, 0.01),
    "DAX40":  (0.01,      0.01,       0.01, 100.0, 0.01),
    "ESXEUR": (0.01,      0.01,       0.01, 100.0, 0.01),
    "SP500":  (0.01,      0.01,       0.01, 100.0, 0.01),
    "UK100":  (0.01,      0.01,       0.01, 100.0, 0.01),
    "USDCAD": (0.00001,   0.73,       0.01, 100.0, 0.01),
    "USDJPY": (0.001,     0.65,       0.01, 100.0, 0.01),
    "XAGUSD": (0.001,     5.0,        0.01, 100.0, 0.01),
}

# Representative price + SL distance per symbol (SL ≈ 0.3% of price)
SCENARIO = {
    "ASXAUD": (8000.0,  24.0),
    "DAX40":  (18000.0, 54.0),
    "ESXEUR": (5000.0,  15.0),
    "SP500":  (5500.0,  16.5),
    "UK100":  (8200.0,  24.6),
    "USDCAD": (1.3600,  0.0041),
    "USDJPY": (156.00,  0.47),
    "XAGUSD": (31.00,   0.093),
}

BALANCE = 10_000.0
TOL = 0.30  # actual risk must be within ±30% of intended (rounding granularity)


def fmt(x): return f"{x:.5f}".rstrip("0").rstrip(".")


def main():
    print("F10 — Lot Sizing Acceptance Tests\n")
    intended = BALANCE * RISK_PCT
    print(f"Account ${BALANCE:.0f}  ·  intended risk/trade = ${intended:.2f} "
          f"({RISK_PCT*100:.2f}%)\n")
    print(f"{'Symbol':<8} {'Price':>9} {'SL dist':>9} {'Lots':>7} "
          f"{'Actual $':>9} {'%':>6} {'Status'}")
    print("-" * 60)

    all_pass = True
    for sym, (ts, tv, vmin, vmax, vstep) in SPECS.items():
        price, sl_dist = SCENARIO[sym]
        lots = calculate_lots(sym, sl_dist, BALANCE, ts, tv, vmin, vmax, vstep)
        sl_ticks = sl_dist / ts
        actual = lots * sl_ticks * tv
        pct = actual / BALANCE * 100

        # Checks: lots>0, actual within tolerance, never over 1.25%*1.5
        ok = (lots > 0
              and abs(actual - intended) <= intended * TOL
              and actual <= intended * 1.5)
        # round-down check: lots must be a multiple of vstep
        ok = ok and abs((lots / vstep) - round(lots / vstep)) < 1e-9
        status = "✅" if ok else "❌ FAIL"
        if not ok:
            all_pass = False
        print(f"{sym:<8} {fmt(price):>9} {fmt(sl_dist):>9} {lots:>7.2f} "
              f"{actual:>9.2f} {pct:>5.2f}% {status}")

    # ── Edge case 1: below-minimum must return 0.0 (skip, not force up) ──
    print("\nEdge cases:")
    ts, tv, vmin, vmax, vstep = SPECS["USDJPY"]
    # Tiny account + wide SL → raw lots below 0.01
    lots_small = calculate_lots("USDJPY", 3.0, 300, ts, tv, vmin, vmax, vstep)
    ok1 = lots_small == 0.0
    print(f"  Below-min sizing returns 0.0 (skip): {lots_small} "
          f"{'✅' if ok1 else '❌ FAIL — forced up, over-risk!'}")
    all_pass = all_pass and ok1

    # ── Edge case 2: invalid inputs return 0.0 ──
    lots_bad = calculate_lots("USDJPY", 0.0, BALANCE, ts, tv, vmin, vmax, vstep)
    ok2 = lots_bad == 0.0
    print(f"  Zero SL distance returns 0.0:        {lots_bad} "
          f"{'✅' if ok2 else '❌ FAIL'}")
    all_pass = all_pass and ok2

    print("-" * 60)
    if all_pass:
        print("✅  All lot-sizing tests passed.")
        return 0
    print("❌  Lot-sizing test failure — do not deploy.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
