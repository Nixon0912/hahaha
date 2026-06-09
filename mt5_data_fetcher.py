"""
MT5 Data Fetcher
Fetches broker-specific trading data from MetaTrader 5:
- Symbol info (spreads, swaps, commissions, contract size)
- OHLCV historical price data
- Tick data (real-time bid/ask)
- Account info

Requirements:
    pip install MetaTrader5 pandas
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime


# ── Connection ────────────────────────────────────────────────────────────────

def connect(login: int = None, password: str = None, server: str = None) -> bool:
    """Initialize and connect to the MT5 terminal."""
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialize failed: {mt5.last_error()}")
        return False

    if login and password and server:
        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            print(f"[ERROR] Login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

    info = mt5.terminal_info()
    print(f"[OK] Connected to: {info.name} | Build: {info.build}")
    return True


def disconnect():
    mt5.shutdown()
    print("[OK] Disconnected from MT5.")


# ── Broker-Specific Symbol Info ───────────────────────────────────────────────

def get_symbol_info(symbol: str) -> dict:
    """
    Returns broker-specific details for a symbol:
    spread, swap long/short, commission, contract size, margin, etc.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"[ERROR] Symbol '{symbol}' not found: {mt5.last_error()}")
        return {}

    return {
        "symbol":           info.name,
        "description":      info.description,
        "currency_base":    info.currency_base,
        "currency_profit":  info.currency_profit,
        "currency_margin":  info.currency_margin,
        "digits":           info.digits,
        "point":            info.point,
        # Broker costs
        "spread":           info.spread,             # in points
        "spread_float":     info.spread_float,       # True = variable spread
        "swap_long":        info.swap_long,          # per lot per night (long)
        "swap_short":       info.swap_short,         # per lot per night (short)
        "swap_mode":        info.swap_mode,          # how swap is calculated
        "swap_rollover3days": info.swap_rollover3days,  # day of triple swap
        "commission":       getattr(info, "commission", None),  # not always exposed
        # Contract details
        "contract_size":    info.trade_contract_size,
        "volume_min":       info.volume_min,
        "volume_max":       info.volume_max,
        "volume_step":      info.volume_step,
        # Margin
        "margin_initial":   info.margin_initial,
        "margin_maintenance": info.margin_maintenance,
    }


def get_multiple_symbols_info(symbols: list) -> pd.DataFrame:
    """Returns a DataFrame of broker-specific info for a list of symbols."""
    rows = [get_symbol_info(s) for s in symbols]
    rows = [r for r in rows if r]  # drop failed lookups
    return pd.DataFrame(rows)


# ── Historical OHLCV Data ─────────────────────────────────────────────────────

TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


def get_ohlcv(
    symbol: str,
    timeframe: str = "H1",
    date_from: datetime = None,
    date_to: datetime = None,
    count: int = 1000,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars for a symbol.

    Pass date_from + date_to for a date range, or just count for the latest N bars.
    """
    tf = TIMEFRAMES.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Options: {list(TIMEFRAMES)}")

    if date_from and date_to:
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
    else:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

    if rates is None or len(rates) == 0:
        print(f"[WARN] No OHLCV data for {symbol} {timeframe}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "tick_vol", "real_volume": "real_vol"}, inplace=True)
    return df[["open", "high", "low", "close", "tick_vol", "real_vol", "spread"]]


# ── Tick Data ─────────────────────────────────────────────────────────────────

def get_ticks(
    symbol: str,
    date_from: datetime = None,
    date_to: datetime = None,
    count: int = 1000,
) -> pd.DataFrame:
    """
    Fetch raw tick data (bid, ask, last, volume) for a symbol.
    Useful for analysing actual spread behaviour over time.
    """
    if date_from and date_to:
        ticks = mt5.copy_ticks_range(symbol, date_from, date_to, mt5.COPY_TICKS_ALL)
    else:
        ticks = mt5.copy_ticks_from(symbol, datetime.now(), count, mt5.COPY_TICKS_ALL)

    if ticks is None or len(ticks) == 0:
        print(f"[WARN] No tick data for {symbol}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df["spread_points"] = ((df["ask"] - df["bid"]) * 10 ** mt5.symbol_info(symbol).digits).round(1)
    return df


# ── Account Info ──────────────────────────────────────────────────────────────

def get_account_info() -> dict:
    """Returns current account balance, equity, margin, leverage, etc."""
    acc = mt5.account_info()
    if acc is None:
        print(f"[ERROR] Could not get account info: {mt5.last_error()}")
        return {}
    return {
        "login":        acc.login,
        "server":       acc.server,
        "currency":     acc.currency,
        "leverage":     acc.leverage,
        "balance":      acc.balance,
        "equity":       acc.equity,
        "margin":       acc.margin,
        "margin_free":  acc.margin_free,
        "margin_level": acc.margin_level,
        "profit":       acc.profit,
    }


# ── Example Usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Connect — leave login/password/server as None to use the already-open terminal
    if not connect():
        exit(1)

    symbols = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "BTCUSD"]

    # 1. Broker-specific costs per symbol
    print("\n=== Broker Symbol Info ===")
    info_df = get_multiple_symbols_info(symbols)
    print(info_df[["symbol", "spread", "spread_float", "swap_long", "swap_short",
                    "contract_size", "volume_min"]].to_string(index=False))

    # 2. Historical OHLCV (last 500 H1 bars for EURUSD)
    print("\n=== EURUSD H1 OHLCV (last 10 rows) ===")
    ohlcv = get_ohlcv("EURUSD", timeframe="H1", count=500)
    print(ohlcv.tail(10))

    # 3. Recent ticks (last 200 ticks for EURUSD)
    print("\n=== EURUSD Recent Ticks (last 5 rows) ===")
    ticks = get_ticks("EURUSD", count=200)
    print(ticks.tail(5))

    # 4. Account info
    print("\n=== Account Info ===")
    acc = get_account_info()
    for k, v in acc.items():
        print(f"  {k}: {v}")

    disconnect()
