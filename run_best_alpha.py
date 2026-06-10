"""
Full backtest of best alpha config:
  SL = H1_ATR × 0.3 (clipped 400–1500 pts)
  TP = SL × 4.0
  Sessions: ARB + NYO
  Trend filter: OFF
  DOW: all days
"""

import numpy as np
import pandas as pd
from backtester import Backtester, Strategy, load
from strategy_combined import compute_ranges

PT      = 0.01
LOTS    = 0.1
BALANCE = 10_000.0

SL_ATR_MULT = 0.3
TP_MULT     = 4.0
SL_MIN_PTS  = 400
SL_MAX_PTS  = 1500
BUF         = 30 * PT

PERIODS = [
    ("Steady Uptrend",   "2025-01-07", "2025-03-31"),
    ("Post-ATH Correct", "2025-06-01", "2025-08-31"),
    ("Range Recovery",   "2025-09-01", "2025-12-31"),
    ("High-Vol Chop",    "2026-01-01", "2026-03-31"),
    ("Current",          "2026-04-01", "2026-06-09"),
]

class ATRBreakout(Strategy):
    def __init__(self, balance):
        self.initial_bal = balance
        self.balance     = balance
        self.peak_bal    = balance
        self._day        = None
        self._day_start  = balance
        self._arb_done   = False
        self._nyo_done   = False
        self._in_trade   = False
        self._dir = self._sl = self._tp = None
        self.arb_ranges = self.nyo_ranges = self.h1_atr = None

    def _new_day(self, today):
        self._day       = today
        self._day_start = self.balance
        self._arb_done  = self._nyo_done = False

    def next(self, i, df):
        bar   = df.iloc[i]
        t     = df.index[i]
        today = t.date()
        hour  = t.hour

        if today != self._day:
            self._new_day(today)

        h1_atr = float(self.h1_atr.iloc[i]) if not pd.isna(self.h1_atr.iloc[i]) else 0

        # Force-close overnight
        if self._in_trade:
            self.peak_bal = max(self.peak_bal, self.balance)
            if hour >= 21 or (self.peak_bal - self.balance) / self.initial_bal >= 0.085:
                return self._close()
            if self._dir == "buy":
                if bar["low"]  <= self._sl: return self._close()
                if bar["high"] >= self._tp: return self._close()
            else:
                if bar["high"] >= self._sl: return self._close()
                if bar["low"]  <= self._tp: return self._close()
            return None

        # Guards
        self.peak_bal = max(self.peak_bal, self.balance)
        daily_dd  = (self._day_start - self.balance) / self.initial_bal
        max_dd    = (self.peak_bal  - self.balance) / self.initial_bal
        if daily_dd >= 0.04 or max_dd >= 0.085:
            return None

        close  = bar["close"]
        spread = bar["spread"]

        def _sl_pts():
            if h1_atr <= 0: return SL_MIN_PTS
            return int(np.clip(h1_atr * SL_ATR_MULT / PT, SL_MIN_PTS, SL_MAX_PTS))

        # ARB 08:00–10:00
        if 8 <= hour < 10 and not self._arb_done and today in self.arb_ranges.index:
            r = self.arb_ranges.loc[today]
            rng = float(r["range_pts"])
            if 500 <= rng <= 9000:
                sl = _sl_pts(); tp = int(sl * TP_MULT)
                if close > r["high"] + BUF:
                    e = close + (spread/2)*PT
                    self._sl=e-sl*PT; self._tp=e+tp*PT
                    self._in_trade=True; self._dir="buy"; self._arb_done=True; return "buy"
                if close < r["low"] - BUF:
                    e = close - (spread/2)*PT
                    self._sl=e+sl*PT; self._tp=e-tp*PT
                    self._in_trade=True; self._dir="sell"; self._arb_done=True; return "sell"

        # NYO 13:00–15:00
        if 13 <= hour < 15 and not self._nyo_done and today in self.nyo_ranges.index:
            r = self.nyo_ranges.loc[today]
            rng = float(r["range_pts"])
            if 300 <= rng <= 7000:
                sl = _sl_pts(); tp = int(sl * TP_MULT)
                if close > r["high"] + BUF:
                    e = close + (spread/2)*PT
                    self._sl=e-sl*PT; self._tp=e+tp*PT
                    self._in_trade=True; self._dir="buy"; self._nyo_done=True; return "buy"
                if close < r["low"] - BUF:
                    e = close - (spread/2)*PT
                    self._sl=e+sl*PT; self._tp=e-tp*PT
                    self._in_trade=True; self._dir="sell"; self._nyo_done=True; return "sell"
        return None

    def _close(self):
        self._in_trade=False; self._dir=self._sl=self._tp=None; return "close"


def run_period(df_full, h1_atr_full, arb_r, nyo_r, start, end, label):
    df = df_full.loc[start:end].copy()
    h1 = h1_atr_full.reindex(df.index, method="ffill")

    strat = ATRBreakout(BALANCE)
    strat.arb_ranges = arb_r
    strat.nyo_ranges = nyo_r
    strat.h1_atr     = h1

    bt     = Backtester(df, strat, lots=LOTS, initial_balance=BALANCE)
    report = bt.run()
    log    = report.trade_log()
    if log.empty:
        return

    s = report.summary()
    log["date"]  = log["exit_time"].dt.date
    log["month"] = log["exit_time"].dt.to_period("M")
    daily   = log.groupby("date")["pnl"].sum()
    monthly = log.groupby("month")["pnl"].sum()

    # Hold time
    log["hold_min"] = (log["exit_time"] - log["entry_time"]).dt.total_seconds() / 60
    avg_hold = log["hold_min"].mean()

    # Sharpe (annualised, daily returns)
    daily_ret = daily / BALANCE
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    try:
        net = float(str(s["net_pnl"]).replace("$","").replace(",",""))
        mdd = float(str(s["max_drawdown"]).replace("$","").replace(",",""))
    except:
        net = mdd = 0

    n_years2 = len(daily) / 252
    ann_ret2 = (net / BALANCE) / n_years2
    calmar   = ann_ret2 / abs(mdd / BALANCE) if mdd != 0 else 0

    # Fee breakdown
    wins  = log[log["pnl"] > 0]
    loses = log[log["pnl"] < 0]
    pf = wins["pnl"].sum() / -loses["pnl"].sum() if len(loses) > 0 else 99

    print(f"\n{'='*62}")
    print(f"  {label}  ({start} → {end})")
    print(f"{'='*62}")
    print(f"  Trades        : {len(log):>6}  ({len(wins)} wins / {len(loses)} losses)")
    print(f"  Win rate      : {len(wins)/len(log)*100:>5.1f}%")
    print(f"  Profit factor : {pf:>6.2f}")
    print(f"  Net PnL       : ${net:>8,.2f}  ({net/BALANCE*100:.1f}%)")
    print(f"  Max drawdown  : ${mdd:>8,.2f}  ({mdd/BALANCE*100:.2f}%)")
    print(f"  Monthly avg   : ${monthly.mean():>8,.2f}")
    print(f"  Monthly best  : ${monthly.max():>8,.2f}")
    print(f"  Monthly worst : ${monthly.min():>8,.2f}")
    print(f"  Sharpe ratio  : {sharpe:>6.2f}  (annualised, daily)")
    print(f"  Calmar ratio  : {calmar:>6.2f}  (ann.ret / max DD)")
    print(f"  Avg hold time : {avg_hold:>5.0f} min  ({avg_hold/60:.1f} hrs)")
    print(f"  Worst day     : ${daily.min():>8,.2f}  ({daily.min()/BALANCE*100:.2f}%)")
    print(f"  Avg win       : ${wins['pnl'].mean():>7,.2f}")
    print(f"  Avg loss      : ${loses['pnl'].mean():>7,.2f}")
    print(f"  Commissions   : ${log['commission'].sum():>7,.2f}")
    print(f"  Spread cost   : ${log['spread_cost'].sum():>7,.2f}")
    print(f"  Total costs   : ${log['commission'].sum()+log['spread_cost'].sum():>7,.2f}")
    gross = net + log['commission'].sum() + log['spread_cost'].sum()
    print(f"  Gross PnL     : ${gross:>8,.2f}  (before costs)")
    print(f"  Cost drag     : {(log['commission'].sum()+log['spread_cost'].sum())/gross*100:.1f}% of gross")

    # Challenge sim
    bal = BALANCE; peak = BALANCE; result = "⏳ Not reached"; ch_date = None
    for _, t in log.iterrows():
        bal += t["pnl"]; peak = max(peak, bal)
        if bal >= BALANCE * 1.08:
            result = "✅ PASS"; ch_date = t["exit_time"].date(); break
        if peak - bal >= BALANCE * 0.10:
            result = "❌ FAIL (DD)"; ch_date = t["exit_time"].date(); break
    print(f"\n  The5ers challenge: {result}", f" ({ch_date})" if ch_date else "")

    print(f"\n  Monthly PnL:")
    for m, v in monthly.items():
        bar_len = int(abs(v) / 50)
        sign = "+" if v >= 0 else "-"
        print(f"    {m}  {sign}${abs(v):>7,.2f}  {'▓'*min(bar_len,40)}")

    return {"net": net, "mdd": mdd, "sharpe": sharpe, "calmar": calmar,
            "trades": len(log), "monthly_avg": monthly.mean(),
            "avg_hold": avg_hold, "pf": pf}


def main():
    print("Loading M5 data …")
    df_full = load("M5")

    print("Computing H1 ATR …")
    h1 = load("H1")
    prev_c = h1["close"].shift(1)
    tr = pd.concat([h1["high"]-h1["low"],
                    (h1["high"]-prev_c).abs(),
                    (h1["low"]-prev_c).abs()], axis=1).max(axis=1)
    h1_atr = tr.ewm(span=14, adjust=False).mean()
    h1_atr_m5 = h1_atr.shift(1).reindex(df_full.index, method="ffill")

    arb_r, nyo_r = compute_ranges(df_full.copy())

    print(f"\n{'='*62}")
    print(f"  BEST ALPHA — Full Analysis")
    print(f"  SL = H1_ATR × 0.3  (clip 400–1500 pts)")
    print(f"  TP = SL × 4.0  |  ARB + NYO  |  No trend filter")
    print(f"  Balance $10,000  |  0.1 lot")
    print(f"{'='*62}")

    all_results = []
    for label, start, end in PERIODS:
        r = run_period(df_full, h1_atr_m5, arb_r, nyo_r, start, end, label)
        if r: all_results.append((label, r))

    # Summary across all periods
    print(f"\n\n{'='*62}")
    print(f"  CROSS-PERIOD SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Period':<22}  {'Trades':>6}  {'PF':>5}  {'Monthly':>9}  {'Sharpe':>7}  {'Calmar':>7}")
    print(f"  {'-'*58}")
    for label, r in all_results:
        print(f"  {label:<22}  {r['trades']:>6}  {r['pf']:>5.2f}  "
              f"${r['monthly_avg']:>8,.0f}  {r['sharpe']:>7.2f}  {r['calmar']:>7.2f}")
    if all_results:
        avg_sharpe  = np.mean([r["sharpe"]      for _, r in all_results])
        avg_calmar  = np.mean([r["calmar"]       for _, r in all_results])
        avg_monthly = np.mean([r["monthly_avg"]  for _, r in all_results])
        avg_hold    = np.mean([r["avg_hold"]     for _, r in all_results])
        total_tr    = sum(r["trades"]            for _, r in all_results)
        print(f"  {'-'*58}")
        print(f"  {'AVERAGE':<22}  {total_tr//len(all_results):>6}  "
              f"{'—':>5}  ${avg_monthly:>8,.0f}  {avg_sharpe:>7.2f}  {avg_calmar:>7.2f}")
        print(f"\n  Total trades (5 periods): {total_tr}")
        print(f"  Avg hold time           : {avg_hold:.0f} min ({avg_hold/60:.1f} hrs)")

if __name__ == "__main__":
    main()
