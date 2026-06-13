"""
F13 — Signal reconciliation: live path vs backtest path.

The backtest extracts trades with multi_asset_scan.extract_arb/nyo/mom.
The live EA detects trades with ea/signals.py check_arb/check_nyo/check_mom.
If these two ever diverge, the live system trades differently from what was
validated. This script proves they produce the SAME (entry_t, direction)
set, stream by stream.

Method: for each stream, replay the live check_* functions over the exact
same candidate entry bars the extractor scans (one trade per day, first
qualifying bar), then diff the resulting trade sets against the extractor's.

Run from repo root:  python ea/reconcile.py
Exit 0 = perfect match on all streams. Exit 1 = divergence found.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import multi_asset_scan as mas
from ml_filter import load_m15_mtf
from multi_asset_scan import extract_arb, extract_nyo, extract_mom
import ea.signals as live
from ea.config import STREAMS

# ── Memoize build_mtf so the live check_* calls don't rebuild it per bar ──
# build_mtf is deterministic given m15; the live functions call it internally.
_mtf_cache = {}
_orig_build_mtf = mas.build_mtf
def _cached_build_mtf(m15):
    key = id(m15)
    if key not in _mtf_cache:
        _mtf_cache[key] = _orig_build_mtf(m15)
    return _mtf_cache[key]
# Patch both the backtest module and the live module's reference
mas.build_mtf = _cached_build_mtf
live.build_mtf = _cached_build_mtf


def replay_live(m15, mtf, sym, arch):
    """Replay live check_* over candidate bars, one trade/day, first qualifying."""
    fired = []
    if arch == "ARB":
        windows = (8, 10)
    elif arch == "NYO":
        windows = (13, 15)
    else:
        windows = (8, 20)

    for d in sorted(set(m15.index.date)):
        if pd.Timestamp(d).dayofweek >= 4:
            continue
        bars = m15[(m15.index.date == d)
                   & (m15.index.hour >= windows[0])
                   & (m15.index.hour < windows[1])]
        prev_above = None
        traded = False
        for t, _ in bars.iterrows():
            if traded:
                break
            if arch == "ARB":
                sig = live.check_arb(m15, t)
            elif arch == "NYO":
                sig = live.check_nyo(m15, t)
            else:
                sig, prev_above = live.check_mom(m15, t, prev_above)
            if sig is not None:
                fired.append((t, sig["direction"]))
                traded = True
    return set(fired)


def main():
    streams = sorted(set((s[0], s[1]) for s in STREAMS))
    all_ok = True
    print("F13 Signal Reconciliation — live check_* vs backtest extract_*\n")
    print(f"{'Stream':<14} {'Backtest':>9} {'Live':>6} {'Match':>6} {'Status'}")
    print("-" * 50)

    for sym, arch in streams:
        m15, mtf = load_m15_mtf(sym)
        _mtf_cache.clear()
        _mtf_cache[id(m15)] = mtf  # seed with the canonical mtf

        fn = {"ARB": extract_arb, "NYO": extract_nyo, "MOM": extract_mom}[arch]
        bt_trades = fn(m15, mtf)
        bt_set = set((t["entry_t"], t["d"]) for t in bt_trades)

        live_set = replay_live(m15, mtf, sym, arch)

        match = bt_set == live_set
        n_match = len(bt_set & live_set)
        only_bt = bt_set - live_set
        only_live = live_set - bt_set

        status = "✅ OK" if match else "❌ DIVERGE"
        print(f"{sym}-{arch:<9} {len(bt_set):>9} {len(live_set):>6} "
              f"{n_match:>6} {status}")
        if not match:
            all_ok = False
            if only_bt:
                print(f"    only in backtest ({len(only_bt)}): "
                      f"{sorted(only_bt)[:3]}")
            if only_live:
                print(f"    only in live ({len(only_live)}): "
                      f"{sorted(only_live)[:3]}")

    print("-" * 50)
    if all_ok:
        print("✅  PERFECT MATCH — live signal path == backtest signal path.")
        print("    The EA will trade exactly the setups that were validated.")
        return 0
    else:
        print("❌  DIVERGENCE FOUND — live and backtest disagree. Do not deploy.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
