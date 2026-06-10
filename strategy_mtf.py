"""
Multi-Timeframe Intraday Gold Scalper — XAUUSD

Timeframe stack:
  H1  → trend filter  : trade only in the direction of H1 EMA(50)
  M15 → signal        : EMA(9) x EMA(21) cross + RSI(14) momentum
                        signal only valid for SIGNAL_TTL bars after the cross
  M5  → execution     : first M5 close that aligns with the signal

Rules:
  Entry  : M5 close in signal direction + H1 trend aligned + session active
  SL     : 30 points from entry
  TP     : 60 points from entry  (2:1 RR)
  Forced close : any open trade at/after 22:00 server time
  Session      : 03:00 – 22:00 server time only
  Cooldown     : no new entry for COOLDOWN bars after any exit
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backtester import Backtester, Strategy, Report, load
from broker_costs import SYMBOLS

DATA_DIR = Path(__file__).parent / "data"
SYMBOL   = "XAUUSD"
SPEC     = SYMBOLS[SYMBOL]
POINT    = SPEC["point"]


# ── Indicators ────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_indicators(m5: pd.DataFrame, signal_ttl: int = 6) -> pd.DataFrame:
    """
    Align H1 and M15 indicators onto M5 bars via forward-fill.
    M15 cross signal expires after signal_ttl M5 bars (default = 30 min).
    No look-ahead: HTF indicators are shifted by 1 bar before aligning.
    """
    # ── H1 trend ─────────────────────────────────────────────────────────────
    h1 = load("H1")
    h1["ema50"]      = ema(h1["close"], 50)
    h1["h1_bullish"] = (h1["close"] > h1["ema50"]).astype(int)

    # ── M15 signal ────────────────────────────────────────────────────────────
    m15 = load("M15")
    m15["ema9"]  = ema(m15["close"], 9)
    m15["ema21"] = ema(m15["close"], 21)
    m15["rsi14"] = rsi(m15["close"], 14)

    above_now  = (m15["ema9"] > m15["ema21"]).astype(bool)
    above_prev = above_now.shift(1).fillna(False).astype(bool)

    # Raw cross: +1 long cross, -1 short cross, 0 nothing
    m15["cross"] = np.where(
        (~above_prev) & above_now,    1,
        np.where(above_prev & (~above_now), -1, 0)
    )
    m15["m15_rsi"] = m15["rsi14"]

    # ── Align to M5 (shift HTF by 1 to avoid look-ahead) ─────────────────────
    df = m5.copy()
    df["h1_bullish"] = h1["h1_bullish"].shift(1).reindex(df.index, method="ffill").fillna(0)
    df["m15_cross"]  = m15["cross"].shift(1).reindex(df.index, method="ffill").fillna(0)
    df["m15_rsi"]    = m15["m15_rsi"].shift(1).reindex(df.index, method="ffill").fillna(50)
    df["m5_ema20"]   = ema(df["close"], 20)

    # Signal TTL: mark M5 bars within signal_ttl bars of each cross
    # We give each cross a unique ID then count bars since the cross
    cross_times = df.index[df["m15_cross"] != 0]
    df["signal_dir"] = 0
    df["signal_age"]  = 999

    for ct in cross_times:
        direction = int(df.loc[ct, "m15_cross"])
        mask = (df.index >= ct) & (df.index <= df.index[df.index.get_loc(ct) + signal_ttl]
                                    if df.index.get_loc(ct) + signal_ttl < len(df)
                                    else df.index[-1])
        current_age = (df.index[mask] - ct).total_seconds() / 300  # in M5 bars
        # Only update where this cross is more recent than previous
        for idx in df.index[mask]:
            age = int((idx - ct).total_seconds() / 300)
            if age < df.loc[idx, "signal_age"]:
                df.loc[idx, "signal_age"] = age
                df.loc[idx, "signal_dir"] = direction

    return df


# ── Strategy ──────────────────────────────────────────────────────────────────

class MTFScalper(Strategy):
    SL_POINTS     = 30
    TP_POINTS     = 60
    SESSION_START = 3
    SESSION_END   = 22
    COOLDOWN_BARS = 3   # bars to wait after any exit before re-entering

    def __init__(self):
        self.df_ind     = None
        self._in_trade  = False
        self._direction = None
        self._sl        = None
        self._tp        = None
        self._cooldown  = 0

    def next(self, i: int, df: pd.DataFrame) -> str | None:
        bar      = df.iloc[i]
        bar_time = df.index[i]
        hour     = bar_time.hour
        ind      = self.df_ind.iloc[i]

        # ── Cooldown counter ──────────────────────────────────────────────────
        if self._cooldown > 0:
            self._cooldown -= 1

        # ── Force-close at session end ────────────────────────────────────────
        if self._in_trade and hour >= self.SESSION_END:
            return self._close("session_end")

        # ── Manage open trade: SL / TP ────────────────────────────────────────
        if self._in_trade:
            if self._direction == "buy":
                if bar["low"] <= self._sl or bar["high"] >= self._tp:
                    return self._close("sl_tp")
            else:
                if bar["high"] >= self._sl or bar["low"] <= self._tp:
                    return self._close("sl_tp")
            return None

        # ── No new entries outside session or during cooldown ─────────────────
        if hour < self.SESSION_START or hour >= self.SESSION_END:
            return None
        if self._cooldown > 0:
            return None

        # ── Entry conditions ──────────────────────────────────────────────────
        h1_bull   = ind["h1_bullish"] == 1
        sig_dir   = int(ind["signal_dir"])
        sig_age   = int(ind["signal_age"])
        m15_rsi   = ind["m15_rsi"]
        close     = bar["close"]
        m5_ema20  = ind["m5_ema20"]
        spread    = bar["spread"]

        signal_fresh = sig_age < 6  # only use cross within 6 M5 bars (~30 min)

        # Long entry
        if (h1_bull and sig_dir == 1 and signal_fresh
                and m15_rsi > 50 and close > m5_ema20):
            self._enter("buy", close, spread)
            return "buy"

        # Short entry
        if (not h1_bull and sig_dir == -1 and signal_fresh
                and m15_rsi < 50 and close < m5_ema20):
            self._enter("sell", close, spread)
            return "sell"

        return None

    def _enter(self, direction: str, price: float, spread_pts: float):
        if direction == "buy":
            entry      = price + (spread_pts / 2) * POINT
            self._sl   = entry - self.SL_POINTS * POINT
            self._tp   = entry + self.TP_POINTS * POINT
        else:
            entry      = price - (spread_pts / 2) * POINT
            self._sl   = entry + self.SL_POINTS * POINT
            self._tp   = entry - self.TP_POINTS * POINT
        self._in_trade  = True
        self._direction = direction

    def _close(self, reason: str = "") -> str:
        self._in_trade  = False
        self._direction = None
        self._sl = self._tp = None
        self._cooldown  = self.COOLDOWN_BARS
        return "close"  # close-only, no reverse position opened


# ── Runner ────────────────────────────────────────────────────────────────────

def run_mtf(
    lots: float = 0.1,
    balance: float = 10_000.0,
    signal_ttl: int = 6,
) -> Report:
    print("Loading and aligning indicators across H1 / M15 / M5...")
    m5  = load("M5")
    df  = build_indicators(m5, signal_ttl=signal_ttl)

    strategy         = MTFScalper()
    strategy.df_ind  = df

    bt = Backtester(df, strategy, lots=lots, initial_balance=balance)
    report = bt.run()
    return report


if __name__ == "__main__":
    report = run_mtf(lots=0.1, balance=10_000.0)
    report.print()

    log = report.trade_log()
    if not log.empty:
        print(f"\nTotal trades: {len(log)}")

        # Overnight check — must be zero
        log["held_overnight"] = (
            log["entry_time"].dt.date != log["exit_time"].dt.date
        )
        overnight = log["held_overnight"].sum()
        print(f"Overnight holds: {overnight} "
              f"({'✅ none' if overnight == 0 else '⚠️  check logic'})")

        print(f"\nPnL distribution:")
        print(f"  Best trade:  ${log['pnl'].max():.2f}")
        print(f"  Worst trade: ${log['pnl'].min():.2f}")
        print(f"  Median:      ${log['pnl'].median():.2f}")

        print(f"\nFirst 10 trades:")
        print(log.head(10).to_string(index=False))
