"""
HSI final system — vol-targeted Mean-Reversion + Trend blend.
Polished deliverable: equity (log) + underwater drawdown + year-by-year table.
All systems vol-targeted to 10% annual for fair visual comparison.
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
d1["ret"]=d1["close"].pct_change().fillna(0)
HALF=3.5; cost_unit=HALF/d1["close"]; SPLIT=int(len(d1)*0.70); ann=252
realvol=d1["ret"].rolling(20).std()*np.sqrt(ann)

def vt(pos,target=0.10,cap=3.0):
    return pos*(target/realvol.replace(0,np.nan)).clip(upper=cap).fillna(0)
def net_of(pos):
    pos=pos.shift(1).fillna(0); turn=pos.diff().abs().fillna(pos.abs())
    return (pos*d1["ret"]-turn*cost_unit).fillna(0)
def stats(net):
    eq=(1+net).cumprod(); sh=net.mean()/net.std()*np.sqrt(ann)
    dd=(eq/eq.cummax()-1); cagr=eq.iloc[-1]**(ann/len(net))-1
    shi=net.iloc[:SPLIT].mean()/net.iloc[:SPLIT].std()*np.sqrt(ann)
    sho=net.iloc[SPLIT:].mean()/net.iloc[SPLIT:].std()*np.sqrt(ann)
    return eq,sh,shi,sho,dd,cagr

# components
nn=30; ma=d1["close"].rolling(nn).mean(); sd=d1["close"].rolling(nn).std(); z=(d1["close"]-ma)/sd
zv=z.values; mr=np.zeros(len(d1)); cur=0.0
for i in range(len(d1)):
    if np.isnan(zv[i]): mr[i]=cur; continue
    if cur==0: cur=-1 if zv[i]>=2 else (1 if zv[i]<=-2 else 0)
    elif (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
    mr[i]=cur
mr=pd.Series(mr,index=d1.index)
trend=np.sign(d1["close"].pct_change(100)).fillna(0)
blend=0.5*mr+0.5*trend

systems={"Mean-Reversion":vt(mr),"Trend(100d)":vt(trend),"Blend MR+Trend":vt(blend)}
nets={k:net_of(v) for k,v in systems.items()}
S={k:stats(v) for k,v in nets.items()}

print("="*78)
print(f"  {'system':<20}{'CAGR':>8}{'Sharpe':>8}{'IS Sh':>8}{'OOS Sh':>8}{'maxDD':>8}")
print("="*78)
for k in systems:
    eq,sh,shi,sho,dd,cagr=S[k]
    print(f"  {k:<20}{cagr*100:>7.1f}%{sh:>8.2f}{shi:>8.2f}{sho:>8.2f}{dd.min()*100:>7.1f}%")

# year-by-year for the blend
print("\n  Blend year-by-year return (vol-tgt 10%):")
by=nets["Blend MR+Trend"].groupby(d1.index.year).apply(lambda s:(1+s).prod()-1)
print("   "+"  ".join(f"{y}:{v*100:+.1f}%" for y,v in by.items()))
pos_yrs=(by>0).sum(); print(f"   positive years: {pos_yrs}/{len(by)}")

# plot: equity + underwater
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={"height_ratios":[3,1]})
colors={"Mean-Reversion":"#1f77b4","Trend(100d)":"#ff7f0e","Blend MR+Trend":"#2ca02c"}
for k in systems:
    eq,sh,shi,sho,dd,cagr=S[k]
    ax1.plot(eq.index,eq.values,lw=1.4 if "Blend" in k else 1.0,
             alpha=1.0 if "Blend" in k else 0.6,color=colors[k],
             label=f"{k}  Sh {sh:.2f} (OOS {sho:.2f}) DD {dd.min()*100:.0f}%")
ax1.axvline(d1.index[SPLIT],color="red",ls="--",alpha=0.6,label="IS/OOS split")
ax1.set_yscale("log"); ax1.legend(loc="upper left",fontsize=9); ax1.grid(alpha=0.3)
ax1.set_title("HSI vol-targeted (10%) systems — 2009-2026, costs included")
eqb,_,_,_,ddb,_=S["Blend MR+Trend"]
ax2.fill_between(ddb.index,ddb.values*100,0,color="#2ca02c",alpha=0.4)
ax2.set_ylabel("Blend DD %"); ax2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("hsi_final_equity.png",dpi=120)
print("\n  saved -> hsi_final_equity.png")
