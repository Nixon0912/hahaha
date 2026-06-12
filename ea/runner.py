"""
APEX-9 Main Loop — runs every minute, checks all streams, manages guards.

Usage (Windows with MT5 installed):
    python -m ea.runner

Usage (Mac/Linux — signal-file mode, pair with MQL5 EA):
    python -m ea.runner --signal-file-mode

The loop:
  1. Connect to MT5 (or signal-file mode)
  2. Every new M15 bar: evaluate each stream's entry conditions
  3. Score with ML model — take only P(win) > 35%
  4. Pre-trade: check all guards (floor, daily stop, spread, CB)
  5. Place order with correct lot size
  6. Force-close all positions at 21:00 server time
  7. Record outcomes, update state, evaluate circuit breaker
"""
import sys
import time
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path

# Add parent dir to path so we can import from hahaha root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ea.config import (
    MODEL_PATH, SIGNAL_FILE, LOG_FILE, STATE_FILE,
    STREAMS, ML_THRESHOLD, RISK_PCT, FORCE_CLOSE_H,
    INIT_BALANCE, TARGET_BALANCE, BUST_FLOOR
)
from ea.features import compute_features_live, FEAT_COLS_EXT
from ea.signals import check_arb, check_nyo, check_mom
from ea.risk import (
    load_state, save_state, calculate_lots, reset_daily,
    check_floor_guard, check_daily_stop, check_spread,
    check_circuit_breaker, evaluate_circuit_breaker,
    check_inactivity, record_trade
)
from ea.executor import (
    connect_mt5, get_account_info, get_server_time, get_symbol_info,
    get_open_positions, place_order, force_close_all, write_close_signal
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("apex9.runner")


# ── Globals ────────────────────────────────────────────────────────────────
_model        = None  # loaded once
_model_feats  = None
_traded_today = set()  # (symbol, arch) pairs already traded today
_mom_state    = {}     # (symbol, arch) → prev_above_ema
_last_bar_dt  = {}     # symbol → last processed M15 bar timestamp


def load_model():
    global _model, _model_feats
    payload   = joblib.load(MODEL_PATH)
    _model    = payload["model"]
    _model_feats = payload["feat_cols"]
    log.info(f"Model loaded: trained to {payload['train_end']}  "
             f"n_train={payload['n_train']}  threshold={payload['threshold']:.0%}")


def score_signal(feats: dict) -> float:
    """Return P(win) for a candidate trade."""
    X = np.array([[feats.get(c, 0.0) for c in _model_feats]])
    return float(_model.predict_proba(X)[0, 1])


def get_m15_data(symbol: str, n_bars: int = 2000) -> pd.DataFrame | None:
    """
    Fetch M15 OHLCV from MT5 (live) or from the most recent CSV (fallback).
    Returns dataframe indexed by server-time datetime.
    """
    try:
        import MetaTrader5 as mt5
        from multi_asset_scan import load_raw
        bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, n_bars)
        if bars is None or len(bars) == 0:
            return None
        df = pd.DataFrame(bars)
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("datetime")[["open", "high", "low", "close",
                                       "tick_volume", "spread"]].copy()
        df.rename(columns={"tick_volume": "tick_vol"}, inplace=True)
        df = df[(df["high"] > df["low"]) & (df["close"] > 0)]
        return df
    except ImportError:
        # Fallback: read from local CSV (for testing / signal-file mode)
        import glob
        from multi_asset_scan import load_raw
        files = sorted(glob.glob(str(Path(__file__).parent.parent / f"{symbol}_M15_*.csv")))
        if not files:
            return None
        fpath = max(files, key=lambda f: Path(f).stat().st_size)
        return load_raw(fpath)


def compute_rolling_wr(state: dict) -> float:
    history = state.get("trade_history", [])
    if len(history) < 3:
        return 0.5
    last10 = [t["R"] for t in history[-10:]]
    return float(np.mean([r > 0 for r in last10]))


def reset_daily_traded(server_time: datetime):
    """Reset per-stream trading flag at start of server day."""
    global _traded_today
    today_str = str(server_time.date())
    if not hasattr(reset_daily_traded, "_last_date") or \
       reset_daily_traded._last_date != today_str:
        _traded_today = set()
        _mom_state.clear()
        reset_daily_traded._last_date = today_str
        log.info(f"Daily reset: {today_str}")


def is_new_bar(symbol: str, m15: pd.DataFrame) -> bool:
    latest = m15.index[-1]
    if _last_bar_dt.get(symbol) == latest:
        return False
    _last_bar_dt[symbol] = latest
    return True


def handle_force_close(server_time: datetime, state: dict):
    """Force-close at 20:55, 20:57, 20:59, 21:00 server time."""
    h, m = server_time.hour, server_time.minute
    close_windows = [(FORCE_CLOSE_H - 1, 55), (FORCE_CLOSE_H - 1, 57),
                     (FORCE_CLOSE_H - 1, 59), (FORCE_CLOSE_H, 0)]
    if (h, m) in close_windows:
        positions = get_open_positions()
        if positions:
            n = force_close_all("FORCE_21H")
            log.info(f"Force-close window {h}:{m:02d}: closed {n} positions")


def check_challenge_status(balance: float) -> str | None:
    if balance >= TARGET_BALANCE:
        return "PASSED"
    if balance <= BUST_FLOOR:
        return "BUSTED"
    return None


def process_stream(sym: str, arch: str,
                   m15: pd.DataFrame, state: dict,
                   account: dict, server_time: datetime,
                   signal_file_mode: bool):
    """Evaluate one stream for a trading signal on the current bar."""
    key = (sym, arch)
    if key in _traded_today:
        return

    entry_t = m15.index[-1]
    h = entry_t.hour

    # ── Check entry window ─────────────────────────────────────────────────
    stream_cfg = next((s for s in STREAMS if s[0]==sym and s[1]==arch), None)
    if stream_cfg is None:
        return
    _, _, entry_start, entry_end = stream_cfg
    if not (entry_start <= h < entry_end):
        return

    # ── Skip weekends ──────────────────────────────────────────────────────
    if entry_t.dayofweek >= 5:
        return

    # ── Generate raw signal ────────────────────────────────────────────────
    signal = None
    if arch == "ARB":
        signal = check_arb(m15, entry_t)
    elif arch == "NYO":
        signal = check_nyo(m15, entry_t)
    elif arch == "MOM":
        prev_above = _mom_state.get(key)
        signal, new_prev = check_mom(m15, entry_t, prev_above)
        _mom_state[key] = new_prev

    if signal is None:
        return

    # ── ML filter ─────────────────────────────────────────────────────────
    rolling_wr = compute_rolling_wr(state)
    feats = compute_features_live(m15, sym, arch, entry_t,
                                  signal["direction"], rolling_wr)
    if feats is None:
        return

    prob = score_signal(feats)
    log.info(f"Signal: {sym}-{arch}  dir={'LONG' if signal['direction']==1 else 'SHORT'}  "
             f"P(win)={prob:.3f}")

    if prob < ML_THRESHOLD:
        log.info(f"  → Rejected: P(win)={prob:.3f} < {ML_THRESHOLD:.0%}")
        return

    log.info(f"  → ACCEPTED: P(win)={prob:.3f}")

    # ── Pre-trade guards ───────────────────────────────────────────────────
    balance = account["balance"]
    equity  = account["equity"]

    ok, msg = check_circuit_breaker(state)
    if not ok: log.warning(f"  → BLOCKED: {msg}"); return

    ok, msg = check_floor_guard(balance)
    if not ok: log.warning(f"  → BLOCKED: {msg}"); return

    ok, msg = check_daily_stop(equity, state["daily_start_balance"])
    if not ok: log.warning(f"  → BLOCKED: {msg}"); return

    sym_info = get_symbol_info(sym)
    if sym_info:
        ok, msg = check_spread(sym, sym_info["spread"])
        if not ok: log.warning(f"  → BLOCKED: {msg}"); return
    else:
        sym_info = {"tick_size": 1e-5, "tick_value": 1.0,
                    "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01}

    # ── Lot sizing ─────────────────────────────────────────────────────────
    lots = calculate_lots(
        sym, signal["sl_dist"], balance,
        sym_info["tick_size"], sym_info["tick_value"],
        sym_info["volume_min"], sym_info["volume_max"], sym_info["volume_step"]
    )
    if lots <= 0:
        log.warning(f"  → BLOCKED: lot size = 0")
        return

    # ── Place order ────────────────────────────────────────────────────────
    result = place_order(
        sym, signal["direction"], lots,
        signal["sl"], signal["tp"],
        comment=f"APEX9_{arch}_p{prob:.2f}"
    )

    if result.get("success"):
        _traded_today.add(key)
        log.info(f"  ✅ ORDER SENT: {sym}-{arch}  lots={lots}  "
                 f"SL={signal['sl']:.5f}  TP={signal['tp']:.5f}")
    else:
        log.error(f"  ❌ ORDER FAILED: {result}")


def run(signal_file_mode: bool = False):
    log.info("=" * 60)
    log.info("  APEX-9 EA starting …")
    log.info(f"  Threshold: {ML_THRESHOLD:.0%}  Risk: {RISK_PCT*100:.2f}%")
    log.info("=" * 60)

    load_model()

    # On Mac the MetaTrader5 Python package is unavailable — always signal-file mode
    import platform
    if platform.system() == "Darwin":
        signal_file_mode = True
        log.info("Mac detected — using signal-file mode (apex9_signals.json)")
    elif not signal_file_mode:
        if not connect_mt5():
            log.warning("MT5 not available — switching to signal-file mode")
            signal_file_mode = True

    state = load_state()

    while True:
        try:
            server_time = get_server_time() or datetime.utcnow()
            account     = get_account_info() or {
                "balance": INIT_BALANCE, "equity": INIT_BALANCE
            }

            # ── Challenge status ─────────────────────────────────────────
            status = check_challenge_status(account["balance"])
            if status == "PASSED":
                log.info("🎉 CHALLENGE PASSED — stopping EA")
                break
            if status == "BUSTED":
                log.critical("💀 ACCOUNT BUSTED — stopping EA")
                break

            # ── Daily reset ──────────────────────────────────────────────
            reset_daily_traded(server_time)
            reset_daily(state, account["balance"])

            # ── Force-close windows ──────────────────────────────────────
            handle_force_close(server_time, state)

            # ── Circuit breaker evaluation ───────────────────────────────
            if not state.get("circuit_breaker_active"):
                evaluate_circuit_breaker(state, INIT_BALANCE)

            # ── Inactivity alerts ────────────────────────────────────────
            for alert in check_inactivity(state):
                log.warning(alert)

            # ── Evaluate each stream ─────────────────────────────────────
            h = server_time.hour
            if 7 <= h < FORCE_CLOSE_H:  # only evaluate during market hours
                for sym, arch, entry_start, entry_end in STREAMS:
                    if not (entry_start <= h < entry_end):
                        continue
                    m15 = get_m15_data(sym)
                    if m15 is None or len(m15) < 200:
                        continue
                    if not is_new_bar(sym, m15):
                        continue
                    process_stream(sym, arch, m15, state, account,
                                   server_time, signal_file_mode)

            save_state(state)

        except KeyboardInterrupt:
            log.info("EA stopped by user")
            break
        except Exception as e:
            log.error(f"Loop error: {e}", exc_info=True)

        time.sleep(30)  # poll every 30 seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX-9 EA Runner")
    parser.add_argument("--signal-file-mode", action="store_true",
                        help="Write signals to file instead of placing MT5 orders directly")
    args = parser.parse_args()
    run(signal_file_mode=args.signal_file_mode)
