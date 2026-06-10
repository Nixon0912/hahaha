"""Deep drill on Current period (Apr 1 - Jun 9 2026) with regime-switch ARB."""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from backtester import Backtester, load
from strategy_combined import compute_ranges
from test_regime_switch import build_h1_indicators, RegimeSwitchARB, BALANCE, LOTS, PT

pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)

df_full = load("M5")
ind = build_h1_indicators(df_full)
arb_r, _ = compute_ranges(df_full.copy())

START, END = "2026-04-01", "2026-06-09"
df = df_full.loc[START:END].copy()
iv = ind.reindex(df.index, method="ffill")

strat = RegimeSwitchARB(BALANCE, arb_r, iv, adx_thresh=25)
bt = Backtester(df, strat, lots=LOTS, initial_balance=BALANCE)
rep = bt.run()
log = rep.trade_log()

print("="*90)
print("  CURRENT PERIOD — full trade log (regime-switch ARB)")
print("="*90)
log["hold_h"] = (log["exit_time"]-log["entry_time"]).dt.total_seconds()/3600
log["cum"] = log["pnl"].cumsum()
log["bal"] = BALANCE + log["cum"]
show = log[["direction","entry_time","exit_time","entry_price","exit_price",
           "pnl","cum","bal","hold_h"]].copy()
show["entry_time"]=show["entry_time"].dt.strftime("%m-%d %H:%M")
show["exit_time"]=show["exit_time"].dt.strftime("%m-%d %H:%M")
for c in ["entry_price","exit_price"]: show[c]=show[c].round(2)
for c in ["pnl","cum","bal"]: show[c]=show[c].round(1)
show["hold_h"]=show["hold_h"].round(1)
print(show.to_string(index=False))

print("\n" + "="*90)
wins=log[log.pnl>0]; losses=log[log.pnl<0]
print(f"  Trades: {len(log)}  ({len(log[log.direction=='buy'])}B / {len(log[log.direction=='sell'])}S)")
print(f"  Wins: {len(wins)} (avg ${wins.pnl.mean():.1f})  Losses: {len(losses)} (avg ${losses.pnl.mean():.1f})")
print(f"  Net: ${log.pnl.sum():.1f} ({log.pnl.sum()/BALANCE*100:.2f}%)")
print(f"  Final balance: ${BALANCE+log.pnl.sum():.1f}")

# Gold price trajectory over the period
print(f"\n  Gold price: start ${df['close'].iloc[0]:.1f} → end ${df['close'].iloc[-1]:.1f} "
      f"({(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.1f}%)")
print(f"  H1 ATR avg: {iv['atr'].mean():.2f}  trend (% bars bull): {(iv['trend']>0).mean()*100:.0f}%")

# Daily breakdown
print("\n  Daily PnL:")
log["date"]=log["exit_time"].dt.date
daily = log.groupby("date").agg(n=("pnl","size"), pnl=("pnl","sum"))
print(daily.to_string())

# What happened to BUY trades vs SELL
print("\n  By direction:")
for d in ["buy","sell"]:
    sub=log[log.direction==d]
    if len(sub):
        w=len(sub[sub.pnl>0])
        print(f"    {d:>4}: {len(sub)} trades, {w} wins ({w/len(sub)*100:.0f}%), net ${sub.pnl.sum():.1f}")

# Missed opportunities: days with ARB range but no trade (DD guard or no breakout)
print("\n  Days in period with valid ARB range:")
traded_days = set(log["entry_time"].dt.date)
n_range_days=0; n_traded=0
for date in pd.date_range(START, END):
    d=date.date()
    if d in arb_r.index and d.weekday()<4:
        r=arb_r.loc[d]
        if 500<=float(r["range_pts"])<=9000:
            n_range_days+=1
            if d in traded_days: n_traded+=1
print(f"    {n_range_days} valid-range weekdays, {n_traded} produced a trade "
      f"({n_range_days-n_traded} no-breakout or guard-blocked)")
