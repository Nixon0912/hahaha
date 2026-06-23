"""
LONG-HISTORY hunt — 17 years, the data that can actually validate an edge.

XAUUSD H1 (2009-2026, 100k bars) and HSIHKD M5 (2009-2026, 386k bars).

Discipline:
  - Walk-forward: expanding-window train, score the NEXT unseen year, every year.
  - The signal PARAM is re-chosen each year on past data only.
  - Combined out-of-sample equity = stitched next-year blocks (never seen at pick).
  - Test trend (TSMOM, breakout, EMA) and intraday mean-reversion.
  17 years of walk-forward OOS is a real test, not a backtest.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

ann_h1 = 24*252      # hourly bars/yr (approx, FX ~24h)
VT=0.15; CAP=3.0

FILES={"XAUUSD_H1":("XAUUSD_H1_200904240100_202606101800.csv",0.15,24*252),
       "HSIHKD_M5":("HSIHKD_M5_200902020300_202606102155.csv",3.5,0)}

def sharpe(x,per): return x.mean()/x.std()*np.sqrt(per) if x.std()>0 else 0

def load(fp):
    d=load_raw(fp); return d[d["close"]>0]

# ── signal families (bar-based, generic) ──────────────────────────────────────
def tsmom(p,L): return np.sign(p.pct_change(L)).fillna(0)
def ema_x(p,f,s): return np.sign(p.ewm(span=f).mean()-p.ewm(span=s).mean()).fillna(0)
def breakout(p,L):
    hi=p.rolling(L).max(); lo=p.rolling(L).min()
    sig=pd.Series(0.0,index=p.index)
    sig[p>=hi.shift(1)]=1; sig[p<=lo.shift(1)]=-1
    return sig.replace(0,np.nan).ffill().fillna(0)
def mr_z(p,n,e):
    z=(p-p.rolling(n).mean())/p.rolling(n).std()
    zv=z.values; out=np.zeros(len(zv)); cur=0.0
    for i in range(len(zv)):
        if np.isnan(zv[i]): out[i]=cur; continue
        if cur==0:
            if zv[i]>=e: cur=-1
            elif zv[i]<=-e: cur=1
        else:
            if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
        out[i]=cur
    return pd.Series(out,index=p.index)

def net_of(p,h,sig,vol_win=100):
    ret=p.pct_change().fillna(0)
    rv=ret.rolling(vol_win).std()
    pos=(sig*(rv.median()/rv.replace(0,np.nan)).clip(upper=CAP).fillna(0)).shift(1).fillna(0)
    trn=pos.diff().abs().fillna(pos.abs())
    return (pos*ret-trn*(h/p)).fillna(0)

# parameter grid of trend + MR signals (universal, generic)
def make_grid(p):
    g={}
    for L in [24,48,96,168,336,720]:        # hours: 1d,2d,4d,7d,14d,30d
        g[f"TSMOM{L}"]=tsmom(p,L)
        g[f"BRK{L}"]=breakout(p,L)
    for f,s in [(24,96),(48,200),(96,400)]:
        g[f"EMA{f}x{s}"]=ema_x(p,f,s)
    for n in [48,96,200]:
        for e in [2.0,2.5]:
            g[f"MR{n}e{e}"]=mr_z(p,n,e)
    return g

def walk_forward(name, fp, h, per):
    p=load(fp)["close"]
    grid=make_grid(p)
    nets={k:net_of(p,h,v) for k,v in grid.items()}
    idx=p.index
    years=sorted(set(idx.year))
    # need at least 3 yrs to start
    start_year=years[3]
    wf_blocks=[]; picks=[]
    for y in years:
        if y<start_year: continue
        train_mask=idx.year<y
        test_mask =idx.year==y
        if train_mask.sum()<1000 or test_mask.sum()<200: continue
        # pick best signal on training (past only)
        best=None
        for k,nt in nets.items():
            s=sharpe(nt[train_mask],per)
            if best is None or s>best[0]: best=(s,k)
        _,k=best
        test_ret=nets[k][test_mask]
        wf_blocks.append(test_ret); picks.append((y,k,sharpe(test_ret,per)))
    wf=pd.concat(wf_blocks)
    return wf, picks, nets, p, per

print("="*80)
print("  LONG-HISTORY WALK-FORWARD (17 yrs) — re-pick each year, score next year")
print("="*80)

results={}
for name,(fp,h,per) in FILES.items():
    if per==0:  # estimate bars/yr for M5
        ptmp=load(fp); per=int(len(ptmp)/((ptmp.index[-1]-ptmp.index[0]).days/365))
    wf,picks,nets,p,per=walk_forward(name,fp,h,per)
    eq=(1+wf).cumprod()
    dd=(eq/eq.cummax()-1).min()
    cagr=eq.iloc[-1]**(per/len(wf))-1 if eq.iloc[-1]>0 else -1
    sh=sharpe(wf,per)
    pos_years=sum(1 for _,_,s in picks if s>0)
    results[name]=dict(wf=wf,sh=sh,cagr=cagr,dd=dd,picks=picks,per=per)
    print(f"\n  ── {name}  ({per} bars/yr) ──")
    print(f"     Walk-forward OOS Sharpe {sh:.2f}   CAGR {cagr*100:.1f}%   maxDD {dd*100:.1f}%")
    print(f"     Positive test-years: {pos_years}/{len(picks)}")
    print(f"     Yearly picks & test Sharpe:")
    for y,k,s in picks:
        print(f"       {y}: {k:<12} -> {s:+.2f}")

# ── Combine the two markets (diversify) on overlapping years ──────────────────
print("\n"+"="*80)
print("  COMBINED 2-MARKET WALK-FORWARD PORTFOLIO")
print("="*80)
wfg=results["XAUUSD_H1"]["wf"]; wfh=results["HSIHKD_M5"]["wf"]
# resample both to daily PnL to combine cleanly
dg=(1+wfg).groupby(wfg.index.normalize()).prod()-1
dh=(1+wfh).groupby(wfh.index.normalize()).prod()-1
comb=pd.concat([dg.rename("XAU"),dh.rename("HSI")],axis=1).dropna()
port=comb.mean(axis=1)
sh=sharpe(port,252); eq=(1+port).cumprod(); dd=(eq/eq.cummax()-1).min()
cagr=eq.iloc[-1]**(252/len(port))-1
by=port.groupby(port.index.year).apply(lambda s:(1+s).prod()-1)
corr=comb.corr().iloc[0,1]
print(f"  Daily-combined walk-forward portfolio:")
print(f"     Sharpe {sh:.2f}   CAGR {cagr*100:.1f}%   maxDD {dd*100:.1f}%   corr(XAU,HSI) {corr:.2f}")
print(f"     Positive years: {(by>0).sum()}/{len(by)}")
print("     "+"  ".join(f"{y}:{v*100:+.0f}%" for y,v in by.items()))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(2,1,figsize=(15,9),gridspec_kw={"height_ratios":[3,1]})
for name in results:
    wf=results[name]["wf"]; e=(1+wf).cumprod()
    ax[0].plot(e.index,e.values,lw=1.3,alpha=0.7,
               label=f"{name} WF Sh {results[name]['sh']:.2f} CAGR {results[name]['cagr']*100:.0f}%")
ecomb=(1+port).cumprod()
ax[0].plot(ecomb.index,ecomb.values,lw=2.5,color="black",
           label=f"COMBINED WF Sh {sh:.2f} CAGR {cagr*100:.0f}% DD {dd*100:.0f}%")
ax[0].set_yscale("log"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
ax[0].set_title("17-Year Walk-Forward (re-pick each year, score next) — honest OOS")
uw=(ecomb/ecomb.cummax()-1)*100
ax[1].fill_between(ecomb.index,uw.values,0,color="black",alpha=0.3)
ax[1].set_title(f"Combined drawdown  maxDD {dd*100:.1f}%"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("longhist_hunt.png",dpi=130)
print("\n  Saved → longhist_hunt.png")
