"""
Backtester for XAUUSD — plugs into broker_costs.py for real cost modelling.

Usage:
    1. Subclass Strategy and implement next(i, df) — return "buy", "sell", or None
    2. Call Backtester(df, strategy).run()
    3. Print or inspect the returned Report

Costs modelled:
    - Spread   : half spread added on entry, half on exit (market order simulation)
    - Commission: round-turn charged on entry
    - Swap     : nightly cost deducted for every bar that crosses midnight
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from broker_costs import SYMBOLS, swap_cost_per_night, commission_round_turn, point_value

DATA_DIR = Path(__file__).parent / "data"
SYMBOL = "XAUUSD"


# ── Data loader ───────────────────────────────────────────────────────────────

def load(timeframe: str = "M5") -> pd.DataFrame:
    path = DATA_DIR / f"{SYMBOL}_{timeframe}.csv"
    raw = pd.read_csv(path, sep=None, engine="python", nrows=0).columns.tolist()

    if "<DATE>" in raw:
        # Raw MT5 export format (tab-separated, angle-bracket headers)
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.strip("<>").lower() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
        df.set_index("datetime", inplace=True)
        df.drop(columns=["date", "time"], inplace=True)
        df.rename(columns={"tickvol": "tick_vol", "vol": "real_vol"}, inplace=True)
    else:
        # Resampled CSV format (datetime index already combined)
        df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")

    return df


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    direction:   str        # "buy" or "sell"
    entry_time:  pd.Timestamp
    entry_price: float
    lots:        float
    exit_time:   pd.Timestamp  = None
    exit_price:  float         = None
    pnl:         float         = None   # USD, after all costs
    bars_held:   int           = 0
    swap_paid:   float         = 0.0
    commission:  float         = 0.0
    spread_cost: float         = 0.0


# ── Strategy base class ───────────────────────────────────────────────────────

class Strategy:
    """
    Subclass this. Override next().

    next(i, df) is called on every bar.
    It receives the current bar index and the full DataFrame.
    Return:
        "buy"  — open a long (or close an existing short)
        "sell" — open a short (or close an existing long)
        None   — do nothing
    """

    def next(self, i: int, df: pd.DataFrame) -> Optional[str]:
        raise NotImplementedError


# ── Backtester ────────────────────────────────────────────────────────────────

class Backtester:
    def __init__(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        lots: float = 0.1,
        initial_balance: float = 10_000.0,
        symbol: str = SYMBOL,
    ):
        self.df = df.copy()
        self.strategy = strategy
        self.lots = lots
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.symbol = symbol
        self.spec = SYMBOLS[symbol]
        self.trades: list[Trade] = []
        self._open: Optional[Trade] = None

    # ── Cost helpers ──────────────────────────────────────────────────────────

    def _half_spread_cost(self, bar_spread_pts: float) -> float:
        """Half the bar's spread in USD (charged on entry AND exit)."""
        return (bar_spread_pts / 2) * self.spec["point"] * self.spec["contract_size"] * self.lots

    def _swap_for_bar(self, trade: Trade, bar_time: pd.Timestamp) -> float:
        """Return swap cost if this bar crosses midnight (daily rollover)."""
        if trade.entry_time.date() == bar_time.date():
            return 0.0
        day_name = bar_time.strftime("%A")
        direction = "long" if trade.direction == "buy" else "short"
        return swap_cost_per_night(self.symbol, direction, self.lots, day_name)

    # ── Open / close ──────────────────────────────────────────────────────────

    def _open_trade(self, direction: str, bar: pd.Series, bar_time: pd.Timestamp):
        spread_pts = bar["spread"]
        # Buyer pays ask (close + half spread); seller receives bid (close - half spread)
        if direction == "buy":
            entry_price = bar["close"] + (spread_pts / 2) * self.spec["point"]
        else:
            entry_price = bar["close"] - (spread_pts / 2) * self.spec["point"]

        commission = commission_round_turn(self.symbol, entry_price, self.lots)
        spread_cost = self._half_spread_cost(spread_pts)

        self._open = Trade(
            direction=direction,
            entry_time=bar_time,
            entry_price=entry_price,
            lots=self.lots,
            commission=commission,
            spread_cost=spread_cost,
        )

    def _close_trade(self, bar: pd.Series, bar_time: pd.Timestamp):
        t = self._open
        spread_pts = bar["spread"]
        t.exit_time = bar_time
        t.bars_held = len(self.df.loc[t.entry_time:bar_time]) - 1

        # Exit spread cost (other half)
        exit_spread = self._half_spread_cost(spread_pts)
        t.spread_cost += exit_spread

        if t.direction == "buy":
            exit_price = bar["close"] - (spread_pts / 2) * self.spec["point"]
            raw_pnl = (exit_price - t.entry_price) * self.spec["contract_size"] * self.lots
        else:
            exit_price = bar["close"] + (spread_pts / 2) * self.spec["point"]
            raw_pnl = (t.entry_price - exit_price) * self.spec["contract_size"] * self.lots

        t.exit_price = exit_price
        t.pnl = raw_pnl - t.commission - t.swap_paid
        self.balance += t.pnl
        self.trades.append(t)
        self._open = None

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> "Report":
        df = self.df
        equity_curve = []

        for i in range(1, len(df)):
            bar = df.iloc[i]
            bar_time = df.index[i]

            # Accrue swap on open trade
            if self._open:
                swap = self._swap_for_bar(self._open, bar_time)
                if swap != 0.0:
                    self._open.swap_paid += swap

            signal = self.strategy.next(i, df)

            if signal == "close":
                # Close-only: no new position opened
                if self._open:
                    self._close_trade(bar, bar_time)

            elif signal == "buy":
                if self._open and self._open.direction == "sell":
                    self._close_trade(bar, bar_time)
                if not self._open:
                    self._open_trade("buy", bar, bar_time)

            elif signal == "sell":
                if self._open and self._open.direction == "buy":
                    self._close_trade(bar, bar_time)
                if not self._open:
                    self._open_trade("sell", bar, bar_time)

            # Track equity (unrealised)
            if self._open:
                t = self._open
                mid = bar["close"]
                unreal = ((mid - t.entry_price) if t.direction == "buy"
                          else (t.entry_price - mid))
                unreal *= self.spec["contract_size"] * self.lots
                equity_curve.append(self.balance + unreal)
            else:
                equity_curve.append(self.balance)

        # Force-close any open trade at the last bar
        if self._open:
            self._close_trade(df.iloc[-1], df.index[-1])

        return Report(self.trades, equity_curve, self.initial_balance)


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class Report:
    trades: list
    equity_curve: list
    initial_balance: float

    def summary(self) -> dict:
        if not self.trades:
            return {"error": "No trades taken"}

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        equity = np.array(self.equity_curve)
        drawdowns = equity - np.maximum.accumulate(equity)

        return {
            "trades":           len(self.trades),
            "win_rate":         f"{len(wins)/len(pnls)*100:.1f}%",
            "net_pnl":          f"${sum(pnls):,.2f}",
            "avg_win":          f"${np.mean(wins):.2f}" if wins else "$0",
            "avg_loss":         f"${np.mean(losses):.2f}" if losses else "$0",
            "profit_factor":    f"{sum(wins)/abs(sum(losses)):.2f}" if losses else "∞",
            "max_drawdown":     f"${drawdowns.min():,.2f}",
            "final_balance":    f"${self.initial_balance + sum(pnls):,.2f}",
            "return":           f"{sum(pnls)/self.initial_balance*100:.2f}%",
            "total_commission": f"${sum(t.commission for t in self.trades):,.2f}",
            "total_swap":       f"${sum(t.swap_paid for t in self.trades):,.2f}",
            "total_spread":     f"${sum(t.spread_cost for t in self.trades):,.2f}",
        }

    def print(self):
        s = self.summary()
        print("\n" + "="*40)
        print("  BACKTEST REPORT")
        print("="*40)
        for k, v in s.items():
            print(f"  {k:<20} {v}")
        print("="*40)

    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "direction":    t.direction,
            "entry_time":   t.entry_time,
            "exit_time":    t.exit_time,
            "entry_price":  t.entry_price,
            "exit_price":   t.exit_price,
            "bars_held":    t.bars_held,
            "pnl":          round(t.pnl, 2),
            "commission":   round(t.commission, 2),
            "swap":         round(t.swap_paid, 2),
            "spread_cost":  round(t.spread_cost, 2),
        } for t in self.trades])


# ── Example strategy: simple MA crossover ─────────────────────────────────────

class MACross(Strategy):
    """
    Buy when fast MA crosses above slow MA.
    Sell when fast MA crosses below slow MA.
    Baseline strategy — not a recommendation, just to verify the engine works.
    """
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow
        self._fast_ma = None
        self._slow_ma = None

    def next(self, i: int, df: pd.DataFrame) -> Optional[str]:
        if i < self.slow:
            return None
        close = df["close"]
        fast_now  = close.iloc[i - self.fast + 1 : i + 1].mean()
        fast_prev = close.iloc[i - self.fast     : i    ].mean()
        slow_now  = close.iloc[i - self.slow + 1 : i + 1].mean()
        slow_prev = close.iloc[i - self.slow     : i    ].mean()

        if fast_prev <= slow_prev and fast_now > slow_now:
            return "buy"
        if fast_prev >= slow_prev and fast_now < slow_now:
            return "sell"
        return None


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for tf in ["M5", "M15", "M30", "H1"]:
        print(f"\nRunning MACross(20,50) on XAUUSD {tf}...")
        df = load(tf)
        bt = Backtester(df, MACross(20, 50), lots=0.1, initial_balance=10_000)
        report = bt.run()
        report.print()
        log = report.trade_log()
        if not log.empty:
            print(f"\n  First 3 trades:\n{log.head(3).to_string(index=False)}")
