"""Run the AdaptiveStrategy (regime router) on 4-year M15 data, all 9 periods."""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from backtester import Backtester, load
from strategy_combined import compute_ranges
from strategy_adaptive import AdaptiveStrategy, compute_all_h1

LOT=0.04; BALANCE=10_000.0
PERIODS=[("2022 Bear","2022-03-18","2022-10-31"),
         ("2022-23 Recov","2022-11-01","2023-04-30"),
         ("2023 Range","2023-05-01","2023-09-30"),
         ("2023-24 Brk","2023-10-01","2024-03-31"),
         ("2024 Bull","2024-04-01","2024-10-31"),
         ("2024 Corr","2024-11-01","2025-01-31"),
         ("2025 H1 Bull","2025-02-01","2025-05-31"),
         ("2025 H2 Bull","2025-06-01","2025-12-31"),
         ("2026 Para+Crash","2026-01-01","2026-06-10")]

print("Loading 4yr M15 + computing H1 regimes …")
m15=load("M15")
h1_data=compute_all_h1(m15)
arb_all,nyo_all=compute_ranges(m15.copy())

def run(st,en):
    df=m15.loc[st:en].copy()
    if len(df)<10:return None
    hd=h1_data.reindex(df.index,method="ffill")
    s=AdaptiveStrategy(initial_balance=BALANCE)
    s.h1_data=hd; s.arb_ranges=arb_all; s.nyo_ranges=nyo_all
    bt=Backtester(df,s,lots=LOT,initial_balance=BALANCE)
    log=bt.run().trade_log()
    if log.empty:return dict(n=0,ret=0,pf=0,mdd=0,wd=0,wr=0,hold=0,sharpe=0,ok=True,flat=True,ch="flat")
    w=log[log.pnl>0];l=log[log.pnl<0]
    pf=w.pnl.sum()/-l.pnl.sum() if len(l)>0 else 99
    log["hold_min"]=(log["exit_time"]-log["entry_time"]).dt.total_seconds()/60
    log["date"]=log["exit_time"].dt.date;daily=log.groupby("date")["pnl"].sum()
    cum=log["pnl"].cumsum();mdd=(cum-cum.cummax()).min();net=log.pnl.sum()
    dr=daily/BALANCE;sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    bal=BALANCE;pk=BALANCE;ch="slow"
    for _,t in log.iterrows():
        bal+=t.pnl;pk=max(pk,bal)
        if bal>=BALANCE*1.08:ch="PASS";break
        if pk-bal>=BALANCE*0.10:ch="FAIL";break
    ok=(net>=0 and mdd>=-1200*(LOT/0.10) and daily.min()/BALANCE*100>=-4)
    return dict(n=len(log),ret=net/BALANCE*100,pf=pf,mdd=mdd/BALANCE*100,
                wd=daily.min()/BALANCE*100,wr=len(w)/len(log)*100,
                hold=log.hold_min.mean(),sharpe=sharpe,ok=ok,flat=False,ch=ch)

print(f"\n{'='*92}")
print(f"  ADAPTIVE REGIME ROUTER — 4-year M15 @ {LOT} lot")
print(f"{'='*92}")
print(f"  {'Period':<17}{'N':>4}{'WR%':>6}{'PF':>6}{'Ret%':>7}{'MaxDD%':>8}{'WD%':>6}"
      f"{'Shrp':>6}{'Hold':>6}  {'Result'}")
ok=0;rets=[];tot=0
for lbl,st,en in PERIODS:
    r=run(st,en)
    if r is None:continue
    if r["ok"]:ok+=1
    rets.append(r["ret"]);tot+=r["ret"]
    mark="✓" if r["ok"] else "✗"
    if r["flat"]: mark="·"
    print(f"  {mark} {lbl:<15}{r['n']:>4}{r['wr']:>6.1f}{r['pf']:>6.2f}{r['ret']:>+6.1f}%"
          f"{r['mdd']:>7.2f}%{r['wd']:>5.1f}%{r['sharpe']:>6.2f}{r['hold']:>5.0f}m  {r['ch']}")
print(f"  {'-'*88}")
print(f"  {ok}/9 OK (positive or flat)   total {tot:+.1f}%   avg/period {np.mean(rets):+.1f}%")
