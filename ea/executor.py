"""
MT5 execution bridge — Mac signal-file mode.

On Mac, the MetaTrader5 Python package is not available.
Python writes orders to apex9_signals.json in the MT5 common files directory:
  ~/Library/Application Support/MetaTrader 5/MQL5/Files/apex9_signals.json

The MQL5 EA (APEX9_EA.mq5) polls this file every 10 seconds and executes orders.
Python reads back the updated file to confirm execution status.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from ea.config import FORCE_CLOSE_H, SIGNAL_FILE, LOG_FILE

log = logging.getLogger("apex9.executor")

# ── MT5 connection ─────────────────────────────────────────────────────────

def connect_mt5() -> bool:
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            log.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False
        info = mt5.account_info()
        log.info(f"MT5 connected: account={info.login}  balance=${info.balance:.2f}  "
                 f"server={info.server}")
        return True
    except ImportError:
        log.warning("MetaTrader5 package not available — running in signal-file mode")
        return False

def get_account_info() -> dict | None:
    try:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info is None:
            return None
        return {"balance": info.balance, "equity": info.equity,
                "margin": info.margin, "free_margin": info.margin_free}
    except ImportError:
        return None

def get_server_time() -> datetime | None:
    try:
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick("EURUSD")
        if tick:
            return datetime.fromtimestamp(tick.time)
        return None
    except ImportError:
        return datetime.utcnow()

def get_symbol_info(symbol: str) -> dict | None:
    try:
        import MetaTrader5 as mt5
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "tick_size":   info.trade_tick_size,
            "tick_value":  info.trade_tick_value,  # in account currency per lot
            "volume_min":  info.volume_min,
            "volume_max":  info.volume_max,
            "volume_step": info.volume_step,
            "spread":      info.spread,             # in points
            "digits":      info.digits,
            "point":       info.point,
        }
    except ImportError:
        return None

def get_open_positions(symbol: str = None) -> list:
    try:
        import MetaTrader5 as mt5
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        if positions is None:
            return []
        return list(positions)
    except ImportError:
        return []


# ── Order placement ────────────────────────────────────────────────────────

def place_order(symbol: str, direction: int, lots: float,
                sl: float, tp: float, comment: str = "APEX9") -> dict:
    """
    Place a market order.
    direction: 1=buy, -1=sell
    Returns result dict.
    """
    try:
        import MetaTrader5 as mt5
        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "error": "no tick"}

        price = tick.ask if direction == 1 else tick.bid
        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   symbol,
            "volume":   lots,
            "type":     order_type,
            "price":    price,
            "sl":       round(sl, mt5.symbol_info(symbol).digits),
            "tp":       round(tp, mt5.symbol_info(symbol).digits),
            "deviation": 20,
            "magic":    20260101,
            "comment":  comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"ORDER PLACED: {symbol} {'BUY' if direction==1 else 'SELL'} "
                     f"{lots} lots  SL={sl:.5f}  TP={tp:.5f}  ticket={result.order}")
            return {"success": True, "ticket": result.order, "price": result.price}
        else:
            log.error(f"ORDER FAILED: {symbol}  retcode={result.retcode}  "
                      f"comment={result.comment}")
            return {"success": False, "retcode": result.retcode,
                    "error": result.comment}
    except ImportError:
        # Signal-file mode
        _write_signal(symbol, direction, lots, sl, tp, comment)
        return {"success": True, "mode": "signal_file"}


def close_position(ticket: int, symbol: str, lots: float,
                   direction: int, reason: str = "") -> bool:
    """Close a specific position by ticket."""
    try:
        import MetaTrader5 as mt5
        close_type = mt5.ORDER_TYPE_SELL if direction == 1 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        price = tick.bid if direction == 1 else tick.ask
        request = {
            "action":   mt5.TRADE_ACTION_DEAL,
            "symbol":   symbol,
            "volume":   lots,
            "type":     close_type,
            "position": ticket,
            "price":    price,
            "deviation": 30,
            "magic":    20260101,
            "comment":  f"APEX9_CLOSE_{reason}",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            log.info(f"CLOSED: ticket={ticket} {symbol}  reason={reason}")
        else:
            log.error(f"CLOSE FAILED: ticket={ticket}  retcode={result.retcode}")
        return ok
    except ImportError:
        log.info(f"[signal-file mode] close signal written for {symbol}")
        return True


def force_close_all(reason: str = "FORCE_CLOSE_21H") -> int:
    """
    Redundant force-close: tries up to 4 times at 20:55/20:57/20:59/21:00.
    Returns number of positions closed.
    """
    closed = 0
    positions = get_open_positions()
    for pos in positions:
        for attempt in range(3):
            ok = close_position(pos.ticket, pos.symbol, pos.volume,
                                1 if pos.type == 0 else -1, reason)
            if ok:
                closed += 1
                break
            time.sleep(2)
        else:
            log.critical(f"FAILED to close {pos.symbol} ticket={pos.ticket} "
                         f"after 3 attempts — MANUAL CLOSE REQUIRED")
    return closed


# ── Signal file mode (Mac/Linux fallback) ─────────────────────────────────

def _write_signal(symbol: str, direction: int, lots: float,
                  sl: float, tp: float, comment: str):
    """
    Write a pending signal to signals.json for the MQL5 EA to pick up.
    The MQL5 EA polls this file every tick and executes pending signals.
    """
    try:
        signals = []
        if SIGNAL_FILE.exists():
            with open(SIGNAL_FILE) as f:
                signals = json.load(f)
    except Exception:
        signals = []

    signals.append({
        "status":    "pending",
        "symbol":    symbol,
        "direction": direction,   # 1=buy, -1=sell
        "lots":      lots,
        "sl":        sl,
        "tp":        tp,
        "comment":   comment,
        "timestamp": str(datetime.now()),
    })
    with open(SIGNAL_FILE, "w") as f:
        json.dump(signals, f, indent=2)
    log.info(f"Signal written to file: {symbol} {'BUY' if direction==1 else 'SELL'} {lots}")


def write_close_signal(symbol: str):
    """Write a close-all signal for a symbol to the signal file."""
    try:
        signals = []
        if SIGNAL_FILE.exists():
            with open(SIGNAL_FILE) as f:
                signals = json.load(f)
    except Exception:
        signals = []

    signals.append({
        "status":    "close_all",
        "symbol":    symbol,
        "timestamp": str(datetime.now()),
    })
    with open(SIGNAL_FILE, "w") as f:
        json.dump(signals, f, indent=2)
