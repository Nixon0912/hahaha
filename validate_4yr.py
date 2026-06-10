"""
Validate the regime-switch ARB strategy on 4.2 years of M15 data (2022-2026).
Includes regimes never previously tested: 2022 bear, 2023 range, 2024 bull.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from backtester import Backtester, load
from strategy_combined import compute_ranges
from test_regime_switch import build_h1_indicators, RegimeSwitchARB, BALANCE, PT

LOT = 0.04

# Distinct market regimes across the 4-year span
PERIODS = [
    ("2022 Bear (hikes)",    "2022-03-18", "2022-10-31"),  # 1937->1633 downtrend
    ("2022-23 Recovery",     "2022-11-01", "2023-04-30"),  # 1633->1990 rally
    ("2023 Range/Chop",      "2023-05-01", "2023-09-30"),  # 1990->1849 sideways
    ("2023-24 Breakout",     "2023-10-01", "2024-03-31"),  # 1849->2234 to ATH
    ("2024 Steady Bull",     "2024-04-01", "2024-10-31"),  # 2234->2743 grind
    ("2024 Correction",      "2024-11-01", "2025-01-31"),  # pullback then bounce
    ("2025 H1 Bull",         "2025-02-01", "2025-05-31"),  # strong bull
    ("2025 H2 Bull",         "2025-06-01", "2025-12-31"),  # another leg
    ("2026 Parabola+Crash",  "2026-01-01", "2026-06-10"),  # +11,+11,-12.7 then bear
]

def run(label, start, end, m15, ind, arb_r):
    df = m15.loc[start:end].copy()
    iv = ind.reindex(df.index, method="ffill")
    s  = RegimeSwitchARB(BALANCE, arb_r, iv, adx_thresh=25)
    bt = Backtester(df, s, lots=LOT, initial_balance=BALANCE)
    log = bt.run().trade_log()
    if log.empty: return None
    log["hold_min"]=(log["exit_time"]-log["entry_time"]).dt.total_seconds()/60
    log["date"]=log["exit_time"].dt.date; log["month"]=log["exit_time"].dt.to_period("M")
    w=log[log.pnl>0]; l=log[log.pnl<0]
    daily=log.groupby("date")["pnl"].sum(); monthly=log.groupby("month")["pnl"].sum()
    net=log.pnl.sum(); pf=w.pnl.sum()/-l.pnl.sum() if len(l)>0 else 99
    cum=log["pnl"].cumsum(); mdd=(cum-cum.cummax()).min()
    n_days=len(daily); dr=daily/BALANCE
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    ann=(net/BALANCE)/(n_days/252) if n_days>0 else 0
    calmar=ann/(abs(mdd)/BALANCE) if mdd!=0 else 0
    bal=BALANCE;pk=BALANCE;ch="slow";chd=None
    for _,t in log.iterrows():
        bal+=t.pnl;pk=max(pk,bal)
        if bal>=BALANCE*1.08:ch="PASS";chd=t["exit_time"].date();break
        if pk-bal>=BALANCE*0.10:ch="FAIL";chd=t["exit_time"].date();break
    ok=(pf>=1.10 and monthly.mean()>=0 and mdd>=-1200*(LOT/0.10) and
        daily.min()/BALANCE*100>=-4 and len(log)>=8)
    return dict(n=len(log),nb=len(log[log.direction=="buy"]),ns=len(log[log.direction=="sell"]),
                wr=len(w)/len(log)*100,net=net,ret=net/BALANCE*100,pf=pf,
                mdd=mdd/BALANCE*100,wd=daily.min()/BALANCE*100,hold=log.hold_min.mean(),
                sharpe=sharpe,calmar=calmar,mo=monthly.mean(),ok=ok,ch=ch,chd=chd,
                bpnl=w.pnl.sum() if False else log[log.direction=="buy"].pnl.sum(),
                spnl=log[log.direction=="sell"].pnl.sum())

print("Loading 4-year M15 + H1 …")
m15=load("M15"); ind=build_h1_indicators(m15); arb_r,_=compute_ranges(m15.copy())

print(f"\n{'='*108}")
print(f"  4-YEAR VALIDATION — regime-switch ARB @ {LOT} lot  (SL=ATR×0.7[400-1600], TP×3.5, Mon-Thu)")
print(f"{'='*108}")
print(f"  {'Period':<22}{'N':>4}{'B/S':>7}{'WR%':>6}{'PF':>6}{'Ret%':>7}{'MaxDD%':>8}"
      f"{'WD%':>6}{'Shrp':>6}{'Clmr':>7}{'Hold':>6}  {'BUY$/SELL$':>14} {'Pass'}")
passes=0; rets=[]
for lbl,st,en in PERIODS:
    r=run(lbl,st,en,m15,ind,arb_r)
    if r is None:
        print(f"  {lbl:<22}  no trades"); continue
    if r["ok"]: passes+=1
    rets.append(r["ret"])
    fl="✅" if r["ok"] else "❌"
    bs=f"{r['nb']}/{r['ns']}"; bspnl=f"{r['bpnl']:+.0f}/{r['spnl']:+.0f}"
    print(f"  {fl}{lbl:<20}{r['n']:>4}{bs:>7}{r['wr']:>6.1f}{r['pf']:>6.2f}"
          f"{r['ret']:>+6.1f}%{r['mdd']:>7.2f}%{r['wd']:>5.1f}%{r['sharpe']:>6.2f}{r['calmar']:>7.1f}"
          f"{r['hold']:>5.0f}m  {bspnl:>14}  {r['ch']}")
print(f"  {'-'*104}")
print(f"  {passes}/{len(PERIODS)} periods pass     avg return/period {np.mean(rets):+.1f}%     "
      f"(at {LOT} lot; DD scales with lot)")
