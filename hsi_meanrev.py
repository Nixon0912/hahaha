"""
HSI mean-reversion battery — find a real, steady edge (challenge-agnostic).

Hypothesis (user): HSI is range-bound / mean-reverting -> fade extremes.
Daily lag-1 autocorr = -0.025 (negative => reversion) supports this.

Tests (daily bars engineered from 17yr M5), all cost-adjusted, IS/OOS split:
  A. Short-term reversal: fade the last k-day move, hold m days
  B. Z-score reversion: fade |close-SMA(n)|/std(n) > entry, exit at < exit
  C. RSI(2) Connors: long RSI2<lo, short RSI2>hi, exit on SMA cross
  D. Bollinger reversion

No-lookahead: signal at close(t) -> position for return(t+1). Cost = half-spread
(3.5pts) per unit turnover. Ranked by OOS Sharpe; best curve plotted.
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
d1 = m5.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
d1["ret"] = d1["close"].pct_change()
HALF_SPREAD = 3.5  # pts per turnover unit; flip(=2) -> 7pt round trip
d1["cost_unit"] = HALF_SPREAD / d1["close"]
SPLIT = int(len(d1)*0.70)
ISmask = np.arange(len(d1)) < SPLIT

def metrics(net):
    net = net.fillna(0)
    eq = (1+net).cumprod()
    n = len(net); yrs = n/252
    cagr = eq.iloc[-1]**(1/yrs)-1 if eq.iloc[-1]>0 else -1
    sh = net.mean()/net.std()*np.sqrt(252) if net.std()>0 else 0
    dd = (eq/eq.cummax()-1).min()
    return eq, cagr, sh, dd

def run(pos, name):
    pos = pos.shift(1).fillna(0)               # no lookahead
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos*d1["ret"] - turn*d1["cost_unit"]
    eq, cagr, sh, dd = metrics(net)
    iss = net[ISmask]; oos = net[~ISmask]
    sh_is = iss.mean()/iss.std()*np.sqrt(252) if iss.std()>0 else 0
    sh_oos = oos.mean()/oos.std()*np.sqrt(252) if oos.std()>0 else 0
    ntr = int(turn[turn>0].count())
    longshare = (pos>0).mean(); shortshare=(pos<0).mean()
    return dict(name=name, cagr=cagr, sh=sh, dd=dd, sh_is=sh_is, sh_oos=sh_oos,
                eq=eq, net=net, ntr=ntr, lsh=longshare, ssh=shortshare, pos=pos)

def rsi(series, n):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

results=[]

# A. short-term reversal
for k in [1,2,3,5]:
    for m in [1,2,3]:
        mom = d1["close"].pct_change(k)
        sig = -np.sign(mom)                    # fade the move
        if m>1: sig = sig.rolling(m).mean().apply(np.sign)  # smooth hold
        results.append(run(sig, f"A reversal k{k} hold{m}"))

# B. z-score reversion
for n in [10,20,30]:
    ma = d1["close"].rolling(n).mean(); sd = d1["close"].rolling(n).std()
    z = (d1["close"]-ma)/sd
    for entry in [1.0,1.5,2.0]:
        # hold a fade until z crosses back through 0
        pos = pd.Series(0.0, index=d1.index); cur=0.0
        zv=z.values; out=np.zeros(len(zv))
        for i in range(len(zv)):
            if np.isnan(zv[i]): out[i]=cur; continue
            if cur==0:
                if zv[i]>=entry: cur=-1
                elif zv[i]<=-entry: cur=1
            else:
                if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0

            out[i]=cur
        results.append(run(pd.Series(out,index=d1.index), f"B zscore n{n} e{entry}"))

# C. RSI(2) Connors
r2 = rsi(d1["close"],2)
sma200 = d1["close"].rolling(200).mean(); sma5=d1["close"].rolling(5).mean()
for lo,hi in [(10,90),(5,95),(15,85)]:
    pos=np.zeros(len(d1)); cur=0.0
    r2v=r2.values; c=d1["close"].values; s5=sma5.values
    for i in range(len(d1)):
        if np.isnan(r2v[i]): pos[i]=cur; continue
        if cur==0:
            if r2v[i]<lo: cur=1
            elif r2v[i]>hi: cur=-1
        else:
            if cur==1 and c[i]>s5[i]: cur=0
            elif cur==-1 and c[i]<s5[i]: cur=0
        pos[i]=cur
    results.append(run(pd.Series(pos,index=d1.index), f"C RSI2 {lo}/{hi}"))

# D. Bollinger reversion
for n in [20,30]:
    ma=d1["close"].rolling(n).mean(); sd=d1["close"].rolling(n).std()
    up=ma+2*sd; dn=ma-2*sd
    pos=np.zeros(len(d1)); cur=0.0
    c=d1["close"].values; uu=up.values; dd_=dn.values; mm=ma.values
    for i in range(len(d1)):
        if np.isnan(mm[i]): pos[i]=cur; continue
        if cur==0:
            if c[i]>uu[i]: cur=-1
            elif c[i]<dd_[i]: cur=1
        else:
            if (cur==-1 and c[i]<=mm[i]) or (cur==1 and c[i]>=mm[i]): cur=0
        pos[i]=cur
    results.append(run(pd.Series(pos,index=d1.index), f"D boll n{n}"))

print("="*92)
print(f"  HSI mean-reversion battery — daily bars, cost-adj, IS(70%)/OOS(30%)")
print(f"  {'strategy':<22}{'CAGR':>7}{'Sharpe':>7}{'maxDD':>7}{'IS Sh':>7}{'OOS Sh':>7}{'#tr':>6}{'L/S%':>10}")
print("="*92)
for r in sorted(results, key=lambda x:-x["sh_oos"]):
    robust = " ✅" if (r["sh_is"]>0.3 and r["sh_oos"]>0.3) else ""
    print(f"  {r['name']:<22}{r['cagr']*100:>6.1f}%{r['sh']:>7.2f}{r['dd']*100:>6.1f}%"
          f"{r['sh_is']:>7.2f}{r['sh_oos']:>7.2f}{r['ntr']:>6d}"
          f"{r['lsh']*100:>4.0f}/{r['ssh']*100:>3.0f}{robust}")

best = max(results, key=lambda x: min(x["sh_is"], x["sh_oos"]))  # robust pick
print("\n  Robust pick (max of min(IS,OOS) Sharpe):", best["name"],
      f"| Sharpe {best['sh']:.2f}  CAGR {best['cagr']*100:.1f}%  maxDD {best['dd']*100:.1f}%")

# plot equity curve
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(best["eq"].index, best["eq"].values, lw=1.3, color="#1f77b4", label=best["name"])
bh = (1+d1["ret"].fillna(0)).cumprod()
ax.plot(bh.index, bh.values, lw=1.0, color="gray", alpha=0.6, label="HSI buy & hold")
ax.axvline(d1.index[SPLIT], color="red", ls="--", alpha=0.7, label="IS/OOS split")
ax.set_yscale("log"); ax.set_title(f"HSI mean-reversion equity (log) — {best['name']}  "
    f"Sharpe {best['sh']:.2f}, CAGR {best['cagr']*100:.0f}%, maxDD {best['dd']*100:.0f}%")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("hsi_equity.png", dpi=110)
print("\n  saved equity curve -> hsi_equity.png")
