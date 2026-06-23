"""
ROBUST strategy — built to survive out-of-sample, not to win the backtest.

Anti-overfit discipline:
  1. ONE signal family, ONE parameter set, applied IDENTICALLY to all instruments.
     (Degrees of freedom: ~5 global choices, not 750 per-instrument picks.)
  2. Equal weight. No fitted weights.
  3. The single global parameter is chosen on IS only.
  4. Honest measurement on the OOS holdout that was never touched.
  5. Walk-forward check: re-evaluate on rolling unseen windows.

Tested universal signals:
  - TSMOM(L)        : sign of L-day return, same L for all
  - MR(n,e)         : same z-score reversion params for all
  - TSMOM + MR blend: equal mix, universal params
"""
import sys, warnings, glob, re
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

ann=252; VT=0.15; CAP=3.0; SF=0.70

HALF={"AUDUSD":0.00008,"EURUSD":0.00007,"USDJPY":0.008,"USDCAD":0.00010,
      "USDCHF":0.00009,"AUDNZD":0.00012,"XAUUSD":0.15,"XAGUSD":0.02,
      "XPDUSD":3.0,"XPTUSD":0.8,"XTIUSD":0.04,"XBRUSD":0.04,"NGCUSD":0.005,
      "CUCUSD":0.0025,"NAS100":1.5,"SP500":0.5,"US30":3.0,"DAX40":2.0,
      "ESXEUR":1.0,"F40EUR":0.8,"IBXEUR":4.0,"UK100":1.5,"ASXAUD":2.0,
      "JPN225":8.0,"HSIHKD":3.5}
def hs(sym,p):
    for k,v in HALF.items():
        if sym.startswith(k[:4]): return v
    return p.mean()*0.0002
def load_d1(fp):
    try:
        df=load_raw(fp); df=df[df["close"]>0]
        d=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        return d[d["close"]>0]
    except: return None
def mr_sig(p,n,e):
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
def tsmom(p,L): return np.sign(p.pct_change(L)).fillna(0)
def make_net(p,h,sig):
    ret=p.pct_change().fillna(0); rv=ret.rolling(20).std()*np.sqrt(ann)
    pos=(sig*(VT/rv.replace(0,np.nan)).clip(upper=CAP).fillna(0)).shift(1).fillna(0)
    trn=pos.diff().abs().fillna(pos.abs())
    return (pos*ret-trn*(h/p)).fillna(0)
def sharpe(x): return x.mean()/x.std()*np.sqrt(ann) if x.std()>0 else 0

# ── Load every instrument ─────────────────────────────────────────────────────
all_csvs=sorted(glob.glob("*.csv")); inst={}
for f in all_csvs:
    m=re.match(r"([A-Z0-9]+)_(M5|M15|H1)_",Path(f).name)
    if not m: continue
    sym,tf=m.group(1),m.group(2); rank={"M5":0,"M15":1,"H1":2}
    if sym not in inst or rank[tf]>rank[inst[sym][1]]: inst[sym]=(f,tf)

prices={}
for sym,(fp,tf) in sorted(inst.items()):
    d1=load_d1(fp)
    if d1 is None or len(d1)<400: continue
    prices[sym]=d1["close"]
allsyms=list(prices.keys())
print(f"  Universe: {len(allsyms)} instruments")

# ── Build a universal-signal portfolio (same params for ALL) ──────────────────
def universal_port(kind, **kw):
    """Apply identical signal to every instrument, equal-weight the net streams."""
    nets={}
    for s in allsyms:
        p=prices[s]; h=hs(s,p)
        if kind=="TSMOM": sig=tsmom(p,kw["L"])
        elif kind=="MR":  sig=mr_sig(p,kw["n"],kw["e"])
        elif kind=="BLEND":
            sig=0.5*tsmom(p,kw["L"])+0.5*mr_sig(p,kw["n"],kw["e"])
        nets[s]=make_net(p,h,sig)
    # align
    common=None
    for s in allsyms:
        common=nets[s].index if common is None else common.intersection(nets[s].index)
    df=pd.DataFrame({s:nets[s].reindex(common).fillna(0) for s in allsyms})
    return df.mean(axis=1), common, df

# ── Choose global parameter on IS only, then report OOS ───────────────────────
def split_metrics(port, common):
    sp=int(len(port)*SF)
    return sharpe(port.iloc[:sp]), sharpe(port.iloc[sp:]), sp

print("\n"+"="*78)
print("  UNIVERSAL SIGNAL SCAN — same params for all instruments")
print("  (choose on IS, report OOS — honest)")
print("="*78)
print(f"  {'signal':<22}{'IS_Sh':>8}{'OOS_Sh':>8}{'full_Sh':>9}{'decay':>8}")

configs=[]
# TSMOM family
for L in [20,50,75,100,150,200]:
    port,common,df=universal_port("TSMOM",L=L)
    iss,oss,sp=split_metrics(port,common)
    configs.append((f"TSMOM L={L}",iss,oss,sharpe(port),port,common,df))
# MR family
for n in [10,20,30]:
    for e in [1.5,2.0,2.5]:
        port,common,df=universal_port("MR",n=n,e=e)
        iss,oss,sp=split_metrics(port,common)
        configs.append((f"MR n={n} e={e}",iss,oss,sharpe(port),port,common,df))
# Blend
for L in [50,100,200]:
    port,common,df=universal_port("BLEND",L=L,n=20,e=2.0)
    iss,oss,sp=split_metrics(port,common)
    configs.append((f"BLEND L={L}+MR20",iss,oss,sharpe(port),port,common,df))

for nm,iss,oss,fs,_,_,_ in configs:
    decay=oss-iss
    tag=" ✅" if oss>0.7 else (" ⚠" if oss<0.3 else "")
    print(f"  {nm:<22}{iss:>8.2f}{oss:>8.2f}{fs:>9.2f}{decay:>+8.2f}{tag}")

# ── Pick the config with best IS (honest: we choose on IS, not OOS) ───────────
chosen=max(configs,key=lambda c:c[1])   # best IS Sharpe
nm,iss,oss,fs,port,common,df=chosen
print(f"\n  Chosen on IS: [{nm}]  IS {iss:.2f} -> OOS {oss:.2f}")

# Also show: which config a naive over-fitter would pick (best OOS) for contrast
cheat=max(configs,key=lambda c:c[2])
print(f"  (For contrast, best-OOS config is [{cheat[0]}] OOS {cheat[2]:.2f} "
      f"-- but choosing on OOS is cheating)")

# ── Walk-forward: rolling re-selection, measure only on next unseen block ──────
print("\n"+"="*78)
print("  WALK-FORWARD (re-pick best universal param each block, score next block)")
print("="*78)
# build a param grid of universal configs once
grid=[("TSMOM",dict(L=L)) for L in [20,50,75,100,150,200]]+\
     [("MR",dict(n=n,e=e)) for n in [10,20,30] for e in [1.5,2.0,2.5]]
# precompute each config's net df
gnets={}
for kind,kw in grid:
    p_,c_,d_=universal_port(kind,**kw)
    gnets[(kind,tuple(sorted(kw.items())))]=(p_,c_)
# common index across all
base_common=None
for (p_,c_) in gnets.values():
    base_common=c_ if base_common is None else base_common.intersection(c_)
L0=len(base_common)
fold=L0//5
wf_returns=[]
for k in range(1,5):
    tr_end=fold*k
    te_end=fold*(k+1) if k<4 else L0
    tr_idx=base_common[:tr_end]; te_idx=base_common[tr_end:te_end]
    # pick best config on training portion
    best=None
    for key,(p_,c_) in gnets.items():
        ptr=p_.reindex(tr_idx).fillna(0)
        s=sharpe(ptr)
        if best is None or s>best[0]: best=(s,key,p_)
    _,key,p_=best
    pte=p_.reindex(te_idx).fillna(0)
    wf_returns.append(pte)
    print(f"  Fold {k}: train->{tr_idx[-1].date()}  test {te_idx[0].date()}..{te_idx[-1].date()}"
          f"  picked {key[0]}{dict(key[1])}  test Sharpe {sharpe(pte):.2f}")
wf=pd.concat(wf_returns)
print(f"\n  WALK-FORWARD combined OOS Sharpe: {sharpe(wf):.2f}  "
      f"CAGR {((1+wf).prod()**(ann/len(wf))-1)*100:.1f}%  "
      f"maxDD {((1+wf).cumprod()/(1+wf).cumprod().cummax()-1).min()*100:.1f}%")

# ── Final honest portfolio: the IS-chosen universal config, full metrics ──────
sp=int(len(port)*SF)
port_oos=port.iloc[sp:]
oos_idx=common[sp:]
eq_oos=(1+port_oos).cumprod()
print("\n"+"="*78)
print(f"  FINAL — universal [{nm}], equal-weight all {len(allsyms)} instruments")
print("="*78)
def show(name,seg,idx):
    eq=(1+seg).cumprod(); dd=(eq/eq.cummax()-1).min()
    cg=eq.iloc[-1]**(ann/len(seg))-1 if eq.iloc[-1]>0 else -1
    by=seg.groupby(idx.year).apply(lambda s:(1+s).prod()-1)
    print(f"  [{name}] Sharpe {sharpe(seg):.2f}  CAGR {cg*100:.1f}%  maxDD {dd*100:.1f}%")
    print("    "+"  ".join(f"{y}:{v*100:+.1f}%" for y,v in by.items()))
show("IN-SAMPLE",port.iloc[:sp],common[:sp])
show("OUT-OF-SAMPLE (honest)",port_oos,oos_idx)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(2,1,figsize=(15,9),gridspec_kw={"height_ratios":[3,1]})
eq_full=(1+port).cumprod()
ax[0].plot(common,eq_full.values,color="navy",lw=2,label=f"Universal {nm}")
ax[0].axvline(common[sp],color="red",ls="--",alpha=0.5,label="IS/OOS split")
ax[0].plot(wf.index,(1+wf).cumprod().values*eq_full.reindex(wf.index).iloc[0],
           color="orange",lw=1.5,alpha=0.7,label=f"Walk-forward OOS (Sh {sharpe(wf):.2f})")
ax[0].set_yscale("log"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
ax[0].set_title(f"Robust universal strategy [{nm}]  "
                f"IS Sh {iss:.2f} -> OOS Sh {oss:.2f}  WF Sh {sharpe(wf):.2f}")
uw=(eq_full/eq_full.cummax()-1)*100
ax[1].fill_between(common,uw.values,0,color="navy",alpha=0.4)
ax[1].axvline(common[sp],color="red",ls="--",alpha=0.5)
ax[1].set_title("Drawdown"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("robust_strategy.png",dpi=130)
print("\n  Saved → robust_strategy.png")
