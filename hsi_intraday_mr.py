"""
HSI intraday mean-reversion — fade short-term overshoots, flat by session close.

Directly tests the 'range-bound intraday' hypothesis WITHOUT overnight gap risk
(the thing that gave daily MR its 31% drawdown).

On M15 bars (resampled from M5), within each session:
  anchor = rolling W-bar mean ; dev = (close - anchor)/rolling W-bar std
  enter -sign(dev) when |dev| >= ENTRY ; exit when |dev| <= EXIT or session end
Cost = half-spread per turnover unit. No-lookahead (signal t -> return t+1).
IS(70%)/OOS(30%) split; best curve plotted.
"""
import sys, warnings, glob
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

m5 = load_raw(glob.glob("HSIHKD_M5_*.csv")[0]); m5 = m5[m5["close"]>0]
m15 = m5.resample("15min", label="left", closed="left").agg(
    {"open":"first","high":"max","low":"min","close":"last","spread":"mean"}).dropna()
m15["date"] = m15.index.date
m15["ret"] = m15["close"].pct_change()
HALF_SPREAD_PTS = 3.5

def intraday_signal(W, ENTRY, EXIT):
    # rolling stats computed WITHIN each session (no cross-day leakage)
    g = m15.groupby("date")["close"]
    ma = g.transform(lambda s: s.rolling(W, min_periods=W).mean())
    sd = g.transform(lambda s: s.rolling(W, min_periods=W).std())
    dev = (m15["close"]-ma)/sd
    # stateful position within day, flat at session end
    pos = np.zeros(len(m15)); cur=0.0; dates=m15["date"].values; dv=dev.values
    for i in range(len(m15)):
        if i>0 and dates[i]!=dates[i-1]: cur=0.0     # new session -> flat
        if np.isnan(dv[i]): pos[i]=cur; continue
        if cur==0:
            if dv[i]>=ENTRY: cur=-1
            elif dv[i]<=-ENTRY: cur=1
        else:
            if abs(dv[i])<=EXIT: cur=0
        pos[i]=cur
    return pd.Series(pos, index=m15.index)

def backtest(pos, name):
    # close intraday position at last bar of each session (no overnight)
    eod = m15["date"].values
    pos = pos.copy()
    pos_shift = pos.shift(1).fillna(0)               # no lookahead
    # zero out the return that would cross a day boundary
    cross = pd.Series(eod, index=m15.index) != pd.Series(eod, index=m15.index).shift(1)
    r = m15["ret"].copy(); r[cross]=0.0
    turn = pos_shift.diff().abs().fillna(pos_shift.abs())
    cost = turn*(HALF_SPREAD_PTS/m15["close"])
    net = pos_shift*r - cost
    eq = (1+net.fillna(0)).cumprod()
    nbars=len(net); yrs=nbars/ (252* (len(m15)/m15['date'].nunique()))
    sh = net.mean()/net.std()*np.sqrt(252*(len(m15)/m15['date'].nunique())) if net.std()>0 else 0
    split=int(len(net)*0.7); ism=np.arange(len(net))<split
    iss=net[ism]; oos=net[~ism]
    ppy=252*(len(m15)/m15['date'].nunique())
    sh_is=iss.mean()/iss.std()*np.sqrt(ppy) if iss.std()>0 else 0
    sh_oos=oos.mean()/oos.std()*np.sqrt(ppy) if oos.std()>0 else 0
    dd=(eq/eq.cummax()-1).min()
    cagr=eq.iloc[-1]**(1/max(yrs,0.1))-1 if eq.iloc[-1]>0 else -1
    ntr=int((turn>0).sum())
    lsh=(pos_shift>0).mean(); ssh=(pos_shift<0).mean()
    return dict(name=name,eq=eq,sh=sh,sh_is=sh_is,sh_oos=sh_oos,dd=dd,cagr=cagr,
                ntr=ntr,lsh=lsh,ssh=ssh,split=split,net=net)

results=[]
for W in [6,10,14]:
    for ENTRY in [1.5,2.0,2.5]:
        for EXIT in [0.0,0.5]:
            results.append(backtest(intraday_signal(W,ENTRY,EXIT),
                                    f"W{W} ent{ENTRY} exit{EXIT}"))

print("="*94)
print("  HSI INTRADAY mean-reversion (M15, flat overnight) — cost-adj, IS70/OOS30")
print(f"  {'config':<20}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'IS Sh':>7}{'OOS Sh':>8}{'#tr':>7}{'L/S%':>10}")
print("="*94)
for r in sorted(results,key=lambda x:-x["sh_oos"]):
    rob=" ✅" if (r["sh_is"]>0.5 and r["sh_oos"]>0.5) else ""
    print(f"  {r['name']:<20}{r['cagr']*100:>7.1f}%{r['sh']:>8.2f}{r['dd']*100:>7.1f}%"
          f"{r['sh_is']:>7.2f}{r['sh_oos']:>8.2f}{r['ntr']:>7d}{r['lsh']*100:>4.0f}/{r['ssh']*100:>3.0f}{rob}")

best=max(results,key=lambda x:min(x["sh_is"],x["sh_oos"]))
print(f"\n  Robust pick: {best['name']} | Sharpe {best['sh']:.2f} CAGR {best['cagr']*100:.1f}% "
      f"maxDD {best['dd']*100:.1f}%  ({best['ntr']} trades)")

fig,ax=plt.subplots(figsize=(12,6))
ax.plot(best["eq"].index,best["eq"].values,lw=1.2,color="#2ca02c",label=f"intraday MR {best['name']}")
ax.axvline(m15.index[best["split"]],color="red",ls="--",alpha=0.7,label="IS/OOS split")
ax.set_yscale("log")
ax.set_title(f"HSI intraday mean-reversion (log) — Sharpe {best['sh']:.2f}, "
             f"CAGR {best['cagr']*100:.0f}%, maxDD {best['dd']*100:.0f}%")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("hsi_intraday_equity.png",dpi=110)
print("  saved -> hsi_intraday_equity.png")
