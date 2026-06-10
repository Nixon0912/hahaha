"""
M5 EMA Momentum Scalper — XAUUSD

Logic:
    Trend  : H1 EMA(50) — only trade in the dominant direction
    Signal : M5 EMA(8) crosses EMA(21) in trend direction
    Filter : RSI(14) on M5 must confirm momentum (>50 long, <50 short)
    Entry  : Next bar after cross confirmation
    SL/TP  : Fixed points, tight SL enables larger lot size within rules
    Session: 08:00–21:00 server time (London + NY only, skip thin Asian)
    Max 4 trades/day, hard daily DD stop at 4% of initial balance

Key insight on lot sizing vs SL:
    SL 150 pts × 0.01 × 100 × lots = loss per trade
    3 losses/day < $400 daily guard → lots < 0.89
    → Can use 0.5–0.8 lot with tight SL vs only 0.09 lot on the ARB strategy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backtester import Backtester, Strategy, Report, load

DATA_DIR = Path(__file__).parent / "data"

# ── Indicators ────────────────────────────────────────────────────────────────

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    h1 = load("H1")
    h1["ema50"]   = ema(h1["close"], 50)
    h1["h1_trend"] = np.where(h1["close"] > h1["ema50"], 1, -1)

    out = df.copy()
    out["h1_trend"] = h1["h1_trend"].shift(1).reindex(out.index, method="ffill").fillna(0)
    out["ema8"]     = ema(out["close"], 8)
    out["ema21"]    = ema(out["close"], 21)
    out["rsi14"]    = rsi(out["close"], 14)
    return out


# ── Strategy ──────────────────────────────────────────────────────────────────

class EMAScalper(Strategy):

    SESSION_START = 8
    SESSION_END   = 21

    def __init__(
        self,
        initial_balance : float = 10_000.0,
        sl_points       : int   = 150,
        tp_mult         : float = 2.0,
        max_trades_day  : int   = 4,
        daily_dd_guard  : float = 0.04,
        max_dd_guard    : float = 0.08,
    ):
        self.sl_points     = sl_points
        self.tp_points     = int(sl_points * tp_mult)
        self.max_trades    = max_trades_day
        self.daily_guard   = daily_dd_guard
        self.max_dd_guard  = max_dd_guard

        self.df_ind        = None   # injected

        self.initial_bal   = initial_balance
        self.balance       = initial_balance
        self.peak_bal      = initial_balance
        self._in_trade     = False
        self._dir          = None
        self._sl           = None
        self._tp           = None
        self._day          = None
        self._day_trades   = 0
        self._day_start    = initial_balance
        self._cooldown     = 0

    def _new_day(self, today):
        self._day        = today
        self._day_trades = 0
        self._day_start  = self.balance

    def next(self, i: int, df: pd.DataFrame) -> str | None:
        bar      = df.iloc[i]
        t        = df.index[i]
        hour     = t.hour
        today    = t.date()
        pt       = 0.01

        if today != self._day:
            self._new_day(today)

        if self._cooldown > 0:
            self._cooldown -= 1

        # Force-close
        if self._in_trade and hour >= self.SESSION_END:
            return self._close()

        # SL / TP
        if self._in_trade:
            if self._dir == "buy":
                if bar["low"]  <= self._sl: return self._close()
                if bar["high"] >= self._tp: return self._close()
            else:
                if bar["high"] >= self._sl: return self._close()
                if bar["low"]  <= self._tp: return self._close()
            return None

        # Session / guards
        if hour < self.SESSION_START or hour >= self.SESSION_END:
            return None
        if self._day_trades >= self.max_trades or self._cooldown > 0:
            return None

        self.peak_bal = max(self.peak_bal, self.balance)
        if (self._day_start - self.balance) / self.initial_bal >= self.daily_guard:
            return None
        if (self.peak_bal - self.balance) / self.initial_bal >= self.max_dd_guard:
            return None

        ind  = self.df_ind.iloc[i]
        prev = self.df_ind.iloc[i - 1]
        trend = int(ind["h1_trend"])
        close = bar["close"]
        spread = bar["spread"]

        # Cross detection
        cross_up   = prev["ema8"] <= prev["ema21"] and ind["ema8"] > ind["ema21"]
        cross_down = prev["ema8"] >= prev["ema21"] and ind["ema8"] < ind["ema21"]

        # Long
        if trend >= 0 and cross_up and ind["rsi14"] > 50:
            entry = close + (spread / 2) * pt
            self._sl = entry - self.sl_points * pt
            self._tp = entry + self.tp_points * pt
            self._in_trade = True
            self._dir = "buy"
            self._day_trades += 1
            return "buy"

        # Short
        if trend <= 0 and cross_down and ind["rsi14"] < 50:
            entry = close - (spread / 2) * pt
            self._sl = entry + self.sl_points * pt
            self._tp = entry - self.tp_points * pt
            self._in_trade = True
            self._dir = "sell"
            self._day_trades += 1
            return "sell"

        return None

    def _close(self) -> str:
        self._in_trade = False
        self._dir = None
        self._sl = self._tp = None
        self._cooldown = 2
        return "close"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_scalp(
    lots        : float = 0.1,
    balance     : float = 10_000.0,
    sl_points   : int   = 150,
    tp_mult     : float = 2.0,
    max_trades  : int   = 4,
    verbose     : bool  = True,
) -> tuple[Report, pd.DataFrame]:
    df  = load("M5")
    ind = build_indicators(df)

    strat           = EMAScalper(balance, sl_points, tp_mult, max_trades)
    strat.df_ind    = ind

    bt     = Backtester(df, strat, lots=lots, initial_balance=balance)
    report = bt.run()
    log    = report.trade_log()

    if verbose and not log.empty:
        report.print()
        overnight = (log["entry_time"].dt.date != log["exit_time"].dt.date).sum()
        log["date"]  = log["exit_time"].dt.date
        log["month"] = log["exit_time"].dt.to_period("M")
        daily   = log.groupby("date")["pnl"].sum()
        monthly = log.groupby("month")["pnl"].sum()
        worst_day_pct = daily.min() / balance * 100

        print(f"\n  Overnight holds  : {overnight} {'✅' if overnight == 0 else '⚠️'}")
        print(f"  Avg hold         : {log['bars_held'].mean()*5:.0f} min")
        print(f"  Trades/month avg : {len(log)/len(monthly):.1f}")
        print(f"\n  Daily:")
        print(f"    Worst day      : ${daily.min():.2f} ({worst_day_pct:.2f}%) "
              f"{'✅' if worst_day_pct > -5 else '❌'}")
        print(f"    Best day       : ${daily.max():.2f}")
        print(f"\n  Monthly PnL:")
        for m, v in monthly.items():
            print(f"    {m}  ${v:>8,.2f}")
        print(f"\n  Monthly avg      : ${monthly.mean():.2f}")
        print(f"  Monthly median   : ${monthly.median():.2f}")
        print(f"  Best month       : ${monthly.max():.2f}")
        print(f"  Worst month      : ${monthly.min():.2f}")

    return report, log


# ── Parameter scan ────────────────────────────────────────────────────────────

def scan(lots: float = 0.1, balance: float = 10_000.0):
    print(f"\n{'sl':>5} {'tp_x':>5} {'max_t':>6} | "
          f"{'trades':>7} {'win%':>6} {'monthly':>9} {'pf':>5} "
          f"{'max_dd':>9} {'worst_day%':>11} {'overnight':>10}")
    print("-" * 85)

    best = {}
    best_score = -999999

    for sl in [100, 150, 200, 300]:
        for tp_m in [1.5, 2.0, 3.0]:
            for mt in [3, 4, 6]:
                r, log = run_scalp(lots, balance, sl_points=sl,
                                   tp_mult=tp_m, max_trades=mt, verbose=False)
                if not r.trades or log.empty:
                    continue
                pnls = [t.pnl for t in r.trades]
                wins = [p for p in pnls if p > 0]
                s    = r.summary()
                log["date"]  = log["exit_time"].dt.date
                log["month"] = log["exit_time"].dt.to_period("M")
                daily   = log.groupby("date")["pnl"].sum()
                monthly = log.groupby("month")["pnl"].sum()
                worst_day_pct = daily.min() / balance * 100
                overnight = (log["entry_time"].dt.date != log["exit_time"].dt.date).sum()

                score = monthly.mean() if worst_day_pct > -5 else -9999

                print(f"{sl:>5} {tp_m:>5} {mt:>6} | "
                      f"{len(pnls):>7} {len(wins)/len(pnls)*100:>5.1f}% "
                      f"${monthly.mean():>8,.0f} {s['profit_factor']:>5} "
                      f"{s['max_drawdown']:>9} {worst_day_pct:>10.2f}% "
                      f"{overnight:>10}")

                if score > best_score:
                    best_score = score
                    best = dict(sl=sl, tp_m=tp_m, mt=mt,
                                monthly_avg=monthly.mean(), worst_day_pct=worst_day_pct)

    print("-" * 85)
    if best:
        print(f"\n✅ Best: SL={best['sl']} | TP×{best['tp_m']} | MaxTrades={best['mt']}")
        print(f"   Monthly avg: ${best['monthly_avg']:.2f} | Worst day: {best['worst_day_pct']:.2f}%")
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  EMA M5 Scalper — default params (0.1 lot)")
    print("=" * 55)
    run_scalp(lots=0.1, balance=10_000.0)

    print("\n\n=== PARAMETER SCAN ===")
    best = scan(lots=0.1, balance=10_000.0)

    # Re-run best at larger lot size to show 1-month potential
    if best:
        safe_lot = round(min(
            0.1 * (400 / (best["sl"] * 0.01 * 100 * 0.1 / 0.1 * best["mt"])),
            0.1 * (800 / abs(best.get("max_dd_raw", 879)))
        ) / 0.01) * 0.01
        safe_lot = max(0.1, min(safe_lot, 0.5))

        print(f"\n\n=== BEST CONFIG AT {safe_lot} LOT ===")
        run_scalp(lots=safe_lot, balance=10_000.0,
                  sl_points=best["sl"], tp_mult=best["tp_m"],
                  max_trades=best["mt"])
