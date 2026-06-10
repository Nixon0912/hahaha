"""Complete performance matrix for the validated ARB-trend strategy at 0.04 lot."""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from stress_test import df_full, arb_r, ind, ARB, PERIODS, BALANCE, PT
from backtester import Backtester

LOT = 0.04
SCALE = LOT / 0.10   # pnl in stress_test ARB is computed at 0.10 lot internally? No—
# backtester uses lots param. We pass lots=LOT so pnl already at 0.04 lot.

def run_full(start, end):
    df = df_full.loc[start:end].copy()
    iv = ind.reindex(df.index, method="ffill")
    s  = ARB(BALANCE, arb_r, iv)           # default params SL×0.7[400-1600] TP×3.5
    bt = Backtester(df, s, lots=LOT, initial_balance=BALANCE)
    rep = bt.run(); log = rep.trade_log()
    if log.empty: return None
    log["hold_min"] = (log["exit_time"]-log["entry_time"]).dt.total_seconds()/60
    log["date"]  = log["exit_time"].dt.date
    log["month"] = log["exit_time"].dt.to_period("M")
    w = log[log.pnl>0]; l = log[log.pnl<0]
    daily   = log.groupby("date")["pnl"].sum()
    monthly = log.groupby("month")["pnl"].sum()
    net = log.pnl.sum()
    pf  = w.pnl.sum()/-l.pnl.sum() if len(l)>0 else 99
    cum = log["pnl"].cumsum(); peak = cum.cummax(); mdd = (cum-peak).min()
    n_days = len(daily); n_yrs = n_days/252 if n_days else 0
    dr = daily/BALANCE
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    sortino_dn = dr[dr<0].std()
    sortino = dr.mean()/sortino_dn*np.sqrt(252) if sortino_dn>0 else 0
    ann = (net/BALANCE)/n_yrs if n_yrs>0 else 0
    calmar = ann/(abs(mdd)/BALANCE) if mdd!=0 else 0
    # expectancy & streaks
    wr = len(w)/len(log)
    expectancy = log.pnl.mean()
    # max consecutive losses
    streak=mx=0
    for x in log.pnl:
        if x<0: streak+=1; mx=max(mx,streak)
        else: streak=0
    # challenge
    bal=BALANCE;pk=BALANCE;ch="slow";chd=None
    for _,t in log.iterrows():
        bal+=t.pnl;pk=max(pk,bal)
        if bal>=BALANCE*1.08:ch="PASS";chd=t["exit_time"].date();break
        if pk-bal>=BALANCE*0.10:ch="FAIL";chd=t["exit_time"].date();break
    return dict(
        n=len(log), nb=len(log[log.direction=="buy"]), ns=len(log[log.direction=="sell"]),
        wr=wr*100, net=net, ret=net/BALANCE*100, pf=pf, mdd=mdd, mdd_pct=mdd/BALANCE*100,
        avg_win=w.pnl.mean() if len(w) else 0, avg_loss=l.pnl.mean() if len(l) else 0,
        best=log.pnl.max(), worst=log.pnl.min(),
        expectancy=expectancy, rr=(w.pnl.mean()/-l.pnl.mean()) if len(l) and len(w) else 0,
        hold=log.hold_min.mean(), hold_med=log.hold_min.median(),
        sharpe=sharpe, sortino=sortino, calmar=calmar,
        worst_day=daily.min()/BALANCE*100, best_day=daily.max()/BALANCE*100,
        n_days=n_days, n_months=len(monthly), mo_avg=monthly.mean(),
        max_streak=mx, ch=ch, chd=chd,
        costs=log["commission"].sum()+log.get("spread_cost",pd.Series([0])).sum(),
        log=log,
    )

results={}
for lbl,st,en in PERIODS:
    results[lbl]=run_full(st,en)

# ── Per-period matrix ─────────────────────────────────────────────────────────
print("="*120)
print(f"  FULL PERFORMANCE MATRIX  —  ARB-trend  @  {LOT} lot  (SL=ATR×0.7[400-1600], TP×3.5, Mon-Thu)")
print("="*120)
rows=[("Trades (B/S)", lambda r:f"{r['n']} ({r['nb']}/{r['ns']})"),
      ("Win rate %",   lambda r:f"{r['wr']:.1f}"),
      ("Net return %", lambda r:f"{r['ret']:+.2f}"),
      ("Profit factor",lambda r:f"{r['pf']:.2f}"),
      ("Expectancy $/tr",lambda r:f"{r['expectancy']:+.2f}"),
      ("Avg win $",    lambda r:f"{r['avg_win']:+.2f}"),
      ("Avg loss $",   lambda r:f"{r['avg_loss']:+.2f}"),
      ("R:R (win/loss)",lambda r:f"{r['rr']:.2f}"),
      ("Best / worst $",lambda r:f"{r['best']:+.0f}/{r['worst']:+.0f}"),
      ("Max DD %",     lambda r:f"{r['mdd_pct']:.2f}"),
      ("Worst day %",  lambda r:f"{r['worst_day']:.2f}"),
      ("Best day %",   lambda r:f"{r['best_day']:+.2f}"),
      ("Max loss streak",lambda r:f"{r['max_streak']}"),
      ("Avg hold (min)",lambda r:f"{r['hold']:.0f}"),
      ("Median hold(min)",lambda r:f"{r['hold_med']:.0f}"),
      ("Sharpe",       lambda r:f"{r['sharpe']:.2f}"),
      ("Sortino",      lambda r:f"{r['sortino']:.2f}"),
      ("Calmar",       lambda r:f"{r['calmar']:.2f}"),
      ("Trading days", lambda r:f"{r['n_days']}"),
      ("Avg $/month",  lambda r:f"{r['mo_avg']:+.0f}"),
      ("Challenge",    lambda r:f"{r['ch']}"+(f" {r['chd']}" if r['chd'] else "")),
      ]
hdr=f"  {'Metric':<18}"+"".join(f"{lbl.split()[0][:11]:>13}" for lbl,_,_ in PERIODS)
print(hdr); print("  "+"-"*(18+13*5))
for name,fn in rows:
    line=f"  {name:<18}"
    for lbl,_,_ in PERIODS:
        r=results[lbl]
        line+=f"{(fn(r) if r else '—'):>13}"
    print(line)

# ── Aggregate (pooled, all periods) ───────────────────────────────────────────
allpnl=np.concatenate([results[l]["log"]["pnl"].values for l,_,_ in PERIODS if results[l]])
allhold=np.concatenate([results[l]["log"]["hold_min"].values for l,_,_ in PERIODS if results[l]])
alldir=np.concatenate([(results[l]["log"]["direction"]=="buy").values for l,_,_ in PERIODS if results[l]])
w=allpnl[allpnl>0]; l_=allpnl[allpnl<0]
print("\n"+"="*120)
print("  AGGREGATE  (all 5 periods pooled — proxy for continuous trading)")
print("="*120)
print(f"  Total trades        : {len(allpnl)}  ({alldir.sum()} buy / {(~alldir).sum()} sell)")
print(f"  Total net           : ${allpnl.sum():+.0f}  ({allpnl.sum()/BALANCE*100:+.1f}% on $10k)")
print(f"  Win rate            : {(allpnl>0).mean()*100:.1f}%")
print(f"  Profit factor       : {w.sum()/-l_.sum():.2f}")
print(f"  Expectancy / trade  : ${allpnl.mean():+.2f}")
print(f"  Avg win / avg loss  : ${w.mean():+.2f} / ${l_.mean():+.2f}  (R:R {w.mean()/-l_.mean():.2f})")
print(f"  Avg hold time       : {allhold.mean():.0f} min  (median {np.median(allhold):.0f} min)")
print(f"  Largest win / loss  : ${allpnl.max():+.0f} / ${allpnl.min():+.0f}")
# pooled sharpe on per-trade basis (annualised by trades/yr ~ len/1.4yr)
yrs=1.42
tr_per_yr=len(allpnl)/yrs
sharpe_tr=allpnl.mean()/allpnl.std()*np.sqrt(tr_per_yr)
print(f"  Per-trade Sharpe(ann): {sharpe_tr:.2f}")
print(f"  Total commission+spread cost across all periods: "
      f"${sum(results[l]['costs'] for l,_,_ in PERIODS if results[l]):.0f}")
print(f"\n  Note: metrics at {LOT} lot. Sharpe/Calmar are scale-invariant (same at any lot).")
print(f"        Net $ and DD $ scale linearly with lot size.")
