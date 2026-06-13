"""
F11 — Signal handshake test (execution leg).

Verifies the Python → signal-file → EA → broker path end to end by injecting
ONE tiny test order on the demo, confirming the EA fills it, then flattening it.

This exercises the OPPOSITE direction from the bar export: it proves the EA
actually reads apex9_signals.json and places/closes real (demo) orders.

USAGE — stop the live runner first so it doesn't race on the signal file:
    1. Ctrl-C the running `python ea/runner.py`
    2. python ea/test_handshake.py            # injects a 0.01-lot test BUY
       → watch MT5 Trade tab: a position appears within ~10s
    3. python ea/test_handshake.py --close    # flattens the test position
       → position disappears within ~10s
    4. Restart the live runner: caffeinate -i python ea/runner.py

Default test symbol is XAUUSD (liquid, on your chart). Override with --symbol.
Nothing here touches the validated strategy — it's a plumbing test only.
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from ea.config import SIGNAL_FILE


def _load():
    if SIGNAL_FILE.exists():
        try:
            return json.load(open(SIGNAL_FILE))
        except Exception:
            return []
    return []


def inject(symbol: str, direction: int, lots: float):
    # Wide SL/TP so the test position does NOT auto-close before we flatten it.
    # Direction 1=buy. SL/TP are placeholders; the EA normalizes to digits.
    sigs = _load()
    sigs.append({
        "status":    "pending",
        "symbol":    symbol,
        "direction": direction,
        "lots":      lots,
        "sl":        0.0,      # 0 = no SL (test only — EA sends as-is)
        "tp":        0.0,      # 0 = no TP
        "comment":   "APEX9-HANDSHAKE-TEST",
        "timestamp": str(datetime.now()),
    })
    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sigs, open(SIGNAL_FILE, "w"), indent=2)
    print(f"✅ Injected test {'BUY' if direction==1 else 'SELL'} {lots} {symbol}")
    print(f"   → {SIGNAL_FILE}")
    print("   Watch the MT5 Trade tab — a position should appear within ~10s.")
    print("   Poll the file for the EA's status write-back …")
    for _ in range(12):
        time.sleep(5)
        for s in _load():
            if s.get("comment") == "APEX9-HANDSHAKE-TEST":
                st = s.get("status")
                if st != "pending":
                    print(f"   EA wrote back status = '{st}'  "
                          f"(executed_at={s.get('executed_at','?')})")
                    return
        print("   … still pending, waiting for EA poll")
    print("   ⚠️  No status change after 60s — is Algo Trading enabled on the chart?")


def close(symbol: str):
    sigs = _load()
    sigs.append({
        "status":    "close_all",
        "symbol":    symbol,
        "comment":   "APEX9-HANDSHAKE-TEST-CLOSE",
        "timestamp": str(datetime.now()),
    })
    json.dump(sigs, open(SIGNAL_FILE, "w"), indent=2)
    print(f"✅ Wrote close_all for {symbol} — position should vanish within ~10s.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--lots", type=float, default=0.01)
    ap.add_argument("--sell", action="store_true", help="test a SELL instead of BUY")
    ap.add_argument("--close", action="store_true", help="flatten the test position")
    a = ap.parse_args()
    if a.close:
        close(a.symbol)
    else:
        inject(a.symbol, -1 if a.sell else 1, a.lots)
