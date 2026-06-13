"""
Risk management: lot sizing, guards, circuit breaker, spread check.
All guards are evaluated before any order is sent.
"""
import math
import json
import logging
from datetime import datetime, date
from pathlib import Path
import numpy as np

from ea.config import (
    RISK_PCT, FLOOR_GUARD_BAL, DAILY_STOP_PCT, SPREAD_MAX_MULT,
    INIT_BALANCE, BUST_FLOOR, TARGET_BALANCE,
    CB_T1_TRADES, CB_T1_MIN_R, CB_T2_DRAWDOWN,
    CB_T3_TRADES, CB_T3_MIN_EXPR, CB_T3_MIN_DAYS,
    INACTIVITY_ALERT_DAYS, STATE_FILE
)

log = logging.getLogger("apex9.risk")

# Historical median spreads per symbol (points) — computed from backtest data
# Tuned per symbol/session; used for live spread guard
HIST_SPREAD_MEDIAN = {
    "ASXAUD": 5.0, "DAX40": 8.0, "ESXEUR": 5.0, "SP500": 5.0,
    "UK100": 8.0, "USDCAD": 2.0, "USDJPY": 2.0, "XAGUSD": 30.0,
}


# ── State persistence ─────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "daily_start_balance": INIT_BALANCE,
        "daily_start_date": str(date.today()),
        "trade_history": [],          # list of {"date", "R", "sym", "arch"}
        "last_trade_date": None,
        "circuit_breaker_active": False,
        "circuit_breaker_reason": None,
        "circuit_breaker_since": None,
    }

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Lot sizing ────────────────────────────────────────────────────────────

def calculate_lots(symbol: str, sl_dist: float, balance: float,
                   tick_size: float, tick_value: float,
                   volume_min: float, volume_max: float,
                   volume_step: float) -> float:
    """
    Universal lot sizing — works for FX, indices, metals.
    Uses MT5's own tick_value (in account currency per lot) to avoid
    manual currency conversion.

    risk_amount = balance * RISK_PCT
    sl_ticks    = sl_dist / tick_size
    lots        = risk_amount / (sl_ticks * tick_value)
    Always round DOWN to volume_step.

    CRITICAL: if raw_lots < volume_min, return 0.0 and skip the trade.
    We never force-round up to volume_min — that would make actual risk
    exceed the intended risk budget, violating the 1.25% cap.
    """
    if sl_dist <= 0 or tick_size <= 0 or tick_value <= 0:
        log.warning(f"{symbol}: invalid sizing inputs — skip")
        return 0.0

    risk_amount = balance * RISK_PCT
    sl_ticks    = sl_dist / tick_size
    raw_lots    = risk_amount / (sl_ticks * tick_value)

    # Round DOWN to volume_step — never up
    lots = math.floor(raw_lots / volume_step) * volume_step

    # Skip if below broker minimum — do NOT force up to volume_min.
    # Forcing up would make actual risk = volume_min * sl_ticks * tick_value,
    # which can meaningfully exceed the 1.25% budget on small accounts or
    # wide SLs.
    if lots < volume_min:
        actual_if_forced = volume_min * sl_ticks * tick_value
        log.warning(f"{symbol}: sized lots={lots:.4f} < volume_min={volume_min} — "
                    f"trade skipped (forcing min would risk "
                    f"${actual_if_forced:.2f} = {actual_if_forced/balance*100:.2f}%)")
        return 0.0

    lots = min(lots, volume_max)

    # Verify: actual cash risk must not exceed 1.5× intended (sanity check)
    actual_risk = lots * sl_ticks * tick_value
    if actual_risk > risk_amount * 1.5:
        log.warning(f"{symbol}: actual_risk=${actual_risk:.2f} > 1.5× intended "
                    f"${risk_amount:.2f} — trade skipped")
        return 0.0

    log.info(f"{symbol}: risk=${risk_amount:.2f}  SL={sl_dist:.5f}  "
             f"lots={lots:.2f}  actual_risk=${actual_risk:.2f}  "
             f"({actual_risk/balance*100:.2f}%)")
    return lots


# ── Pre-trade guards ──────────────────────────────────────────────────────

def check_floor_guard(balance: float) -> tuple[bool, str]:
    if balance <= FLOOR_GUARD_BAL:
        return False, f"Floor guard: balance ${balance:.2f} ≤ ${FLOOR_GUARD_BAL}"
    return True, ""

def check_daily_stop(equity: float, daily_start_balance: float) -> tuple[bool, str]:
    dd = (equity - daily_start_balance) / daily_start_balance
    if dd <= -DAILY_STOP_PCT:
        return False, f"Daily stop: equity {dd*100:.2f}% below daily start"
    return True, ""

def check_spread(symbol: str, live_spread_pts: float) -> tuple[bool, str]:
    median = HIST_SPREAD_MEDIAN.get(symbol, 10.0)
    if live_spread_pts > median * SPREAD_MAX_MULT:
        return False, (f"Spread guard: {symbol} live={live_spread_pts:.1f}pt "
                       f"> {median * SPREAD_MAX_MULT:.1f}pt limit")
    return True, ""

def check_circuit_breaker(state: dict) -> tuple[bool, str]:
    if state.get("circuit_breaker_active"):
        reason = state.get("circuit_breaker_reason", "unknown")
        return False, f"Circuit breaker active: {reason}"
    return True, ""

def evaluate_circuit_breaker(state: dict, starting_balance: float,
                             live_balance: float | None = None,
                             live_equity: float | None = None) -> bool:
    """
    Evaluate three independent CB triggers. If any fires, activate and return True.

    Args:
        state:           persistent state dict
        starting_balance: challenge-start balance (used as T2 reference)
        live_balance:    actual MT5 account balance (preferred for T2)
        live_equity:     actual MT5 equity including open positions (preferred for T2)

    T2 uses live account equity/balance when available (fixes auditor finding:
    simulated reconstruction misses slippage, partial fills, real float P/L).
    Falls back to trade-history simulation only if MT5 data unavailable.
    """
    history = state.get("trade_history", [])
    if not history:
        return False

    Rs = [t["R"] for t in history]

    # T1: last 5 trades total R ≤ -3.0
    if len(Rs) >= CB_T1_TRADES:
        last5_total = sum(Rs[-CB_T1_TRADES:])
        if last5_total <= CB_T1_MIN_R:
            reason = f"T1: last {CB_T1_TRADES} trades total R={last5_total:.2f} ≤ {CB_T1_MIN_R}"
            _activate_cb(state, reason)
            return True

    # T2: drawdown ≥ -4% from starting balance.
    # Use real MT5 equity (includes open position float) when available.
    # Fall back to simulated reconstruction only if MT5 data is absent.
    if live_equity is not None:
        ref = live_equity
        source = "live_equity"
    elif live_balance is not None:
        ref = live_balance
        source = "live_balance"
    else:
        # Fallback: simulate from trade history (less accurate — misses
        # slippage, real lot sizes, partial fills, commission mismatch)
        ref = starting_balance
        for t in history:
            ref += ref * RISK_PCT * t["R"]
        source = "simulated"

    dd = (ref - starting_balance) / starting_balance
    if dd <= CB_T2_DRAWDOWN:
        reason = (f"T2: drawdown {dd*100:.2f}% ≤ {CB_T2_DRAWDOWN*100:.0f}% "
                  f"(source={source}  ref=${ref:.2f}  start=${starting_balance:.2f})")
        _activate_cb(state, reason)
        return True

    # T3: last 10 trades expR < 0 AND ≥10 calendar days elapsed
    if len(Rs) >= CB_T3_TRADES:
        last10_expr = np.mean(Rs[-CB_T3_TRADES:])
        if last10_expr < CB_T3_MIN_EXPR:
            first_date = history[-CB_T3_TRADES]["date"]
            last_date  = history[-1]["date"]
            days_elapsed = (datetime.fromisoformat(last_date) -
                            datetime.fromisoformat(first_date)).days
            if days_elapsed >= CB_T3_MIN_DAYS:
                reason = (f"T3: last {CB_T3_TRADES} trades expR={last10_expr:.3f} < 0 "
                          f"over {days_elapsed} days")
                _activate_cb(state, reason)
                return True

    return False

def _activate_cb(state: dict, reason: str):
    state["circuit_breaker_active"] = True
    state["circuit_breaker_reason"] = reason
    state["circuit_breaker_since"]  = str(datetime.now())
    log.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}")


# ── Inactivity monitor ────────────────────────────────────────────────────

def check_inactivity(state: dict,
                     server_date: date | None = None) -> list[str]:
    """
    Return alert messages and apply deterministic contingency actions.

    At day 28 the contingency fires automatically: new trades are blocked
    (by activating the circuit breaker) until manual review clears it.
    This satisfies The5ers compliant-contingency requirement and is the
    only inactivity action that does not require human presence to enforce.

    Uses server_date when supplied (MT5 server time); falls back to local
    date only if unavailable.
    """
    last = state.get("last_trade_date")
    if last is None:
        return []
    ref_date = server_date or date.today()
    days = (ref_date - date.fromisoformat(last)).days
    alerts = []
    for threshold in INACTIVITY_ALERT_DAYS:
        if days >= threshold:
            alerts.append(f"INACTIVITY ALERT: {days} days since last trade "
                          f"(threshold: {threshold}d)")

    # Day-28 contingency: block new trades until manual review.
    # The compliant action at this point is to stop trading and contact
    # The5ers to confirm the account is still in good standing, then
    # manually reset this flag once confirmed.
    if days >= 28 and not state.get("circuit_breaker_active"):
        reason = (f"INACTIVITY CONTINGENCY: {days} days without a trade — "
                  f"auto-blocking new orders. Manual reset required after review.")
        state["circuit_breaker_active"] = True
        state["circuit_breaker_reason"] = reason
        state["circuit_breaker_since"]  = str(datetime.now())
        log.critical(reason)
        alerts.append(reason)

    return alerts


# ── Post-trade state update ───────────────────────────────────────────────

def record_trade(state: dict, sym: str, arch: str, R: float, trade_date: str):
    state["trade_history"].append({
        "date": trade_date, "sym": sym, "arch": arch, "R": round(R, 4)
    })
    state["last_trade_date"] = trade_date
    log.info(f"Trade recorded: {sym}-{arch}  R={R:+.3f}  ({trade_date})")

def reset_daily(state: dict, balance: float,
                server_date: date | None = None):
    """
    Reset daily tracking baseline.
    Uses MT5 server date when supplied — critical for prop challenges where
    the server calendar (UTC+3) differs from local Mac/VPS date.
    Falls back to local date only if server time is unavailable.
    """
    ref_date = server_date or date.today()
    today_str = str(ref_date)
    if state.get("daily_start_date") != today_str:
        state["daily_start_date"]    = today_str
        state["daily_start_balance"] = balance
        source = "server" if server_date else "local"
        log.info(f"Daily reset ({source} date {today_str}): "
                 f"start_balance=${balance:.2f}")
