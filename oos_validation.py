"""
TRUE out-of-sample validation — answers "is the fix just in-sample?"

Protocol (no peeking):
  1. SELECT signal per instrument using IS data ONLY (first 70%).
  2. SELECT basket + weights using IS data ONLY.
  3. CALIBRATE the hard daily-risk cap using IS data ONLY.
  4. Then FREEZE everything and MEASURE on the OOS holdout (last 30%) that was
     never touched during any decision.

This is the only number that means anything. We report:
  - IS metrics (what we fit — optimistic)
  - OOS metrics (the honest holdout)
  - Challenge MC on OOS-only daily returns
"""
import sys, warnings, glob, re
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
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
def ema(p,f,s): return np.sign(p.ewm(span=f).mean()-p.ewm(span=s).mean()).fillna(0)
def rev(p,L): return -np.sign(p.pct_change(L)).fillna(0)
def valsig(p,n=252):
    z=(p-p.rolling(n).mean())/p.rolling(n).std()
    return (-np.sign(z)).fillna(0)
def make_net(p,h,sig):
    ret=p.pct_change().fillna(0); rv=ret.rolling(20).std()*np.sqrt(ann)
    pos=(sig*(VT/rv.replace(0,np.nan)).clip(upper=CAP).fillna(0)).shift(1).fillna(0)
    trn=pos.diff().abs().fillna(pos.abs())
    return (pos*ret-trn*(h/p)).fillna(0)
def sharpe(x): return x.mean()/x.std()*np.sqrt(ann) if x.std()>0 else 0

all_csvs=sorted(glob.glob("*.csv")); inst={}
for f in all_csvs:
    m=re.match(r"([A-Z0-9]+)_(M5|M15|H1)_",Path(f).name)
    if not m: continue
    sym,tf=m.group(1),m.group(2); rank={"M5":0,"M15":1,"H1":2}
    if sym not in inst or rank[tf]>rank[inst[sym][1]]: inst[sym]=(f,tf)

# ── STEP 1: select signal per instrument on IS only ───────────────────────────
survivors={}
for sym,(fp,tf) in sorted(inst.items()):
    d1=load_d1(fp)
    if d1 is None or len(d1)<300: continue
    p=d1["close"]; h=hs(sym,p); maxL=min(200,int(len(p)*0.20)); cands=[]
    for L in [5,10,15,20,30,50,75,100,150,200]:
        if L<=maxL: cands.append((f"MOM{L}",tsmom(p,L)))
    for n in [10,15,20,30,40]:
        for e in [1.5,2.0,2.5]: cands.append((f"MR{n}e{e}",mr_sig(p,n,e)))
    for L in [1,2,3,5,7,10]: cands.append((f"REV{L}",rev(p,L)))
    for fa,sl in [(5,20),(10,50),(20,100),(5,50)]:
        if sl<=maxL: cands.append((f"EMA{fa}x{sl}",ema(p,fa,sl)))
    if len(p)>=252: cands.append(("VAL1y",valsig(p,252)))
    best=None
    for nm,sig in cands:
        net=make_net(p,h,sig); sp=int(len(net)*SF)
        shi=sharpe(net.iloc[:sp])            # IS-only selection metric
        if best is None or shi>best[2]: best=(nm,net,shi)
    nm,net,shi=best
    if shi>0.30:
        survivors[sym]=dict(net=net,shi=shi,sig=nm)

syms=list(survivors.keys())

# ── STEP 2+3: basket, weights, cap — all from IS only ─────────────────────────
common=survivors[syms[0]]["net"].index
for s in syms[1:]: common=common.intersection(survivors[s]["net"].index)
ND=pd.DataFrame({s:survivors[s]["net"].reindex(common).fillna(0) for s in syms})
SPLIT=int(len(common)*SF)
IS_idx=common[:SPLIT]; OOS_idx=common[SPLIT:]
ND_is=ND.iloc[:SPLIT]

corr_is=ND_is.corr()   # correlation measured on IS only
order=sorted(syms,key=lambda s:-survivors[s]["shi"]); selected=[order[0]]
remaining=[s for s in order if s!=order[0]]
while remaining:
    best_s=min(remaining,key=lambda s:np.mean([abs(corr_is.loc[s,x]) for x in selected]))
    selected.append(best_s); remaining.remove(best_s)

def challenge(r_arr, base_risk, max_risk, max_safe, target=0.08, max_loss=0.06,
              daily_lim=0.05, brake_at=0.04, max_horizon=None, iters=8000, seed=7):
    rng=np.random.default_rng(seed); floor=1.0-max_loss; N=len(r_arr)
    if max_horizon is None: max_horizon=min(400,N-1)
    base_risk=min(base_risk,max_safe); max_risk=min(max_risk,max_safe)
    P=B=NR=0; dtp=[]
    for _ in range(iters):
        s=rng.integers(0,max(1,N-max_horizon)); w=r_arr[s:s+max_horizon]
        bal=1.0; out=None
        for k,x in enumerate(w,1):
            prog=bal-1.0
            buf=base_risk+(max_risk-base_risk)*min(max(prog,0)/target,1.0)
            dist=bal-floor; fb=min(1.0,max(0.0,dist/brake_at))
            rs=min(buf*fb,max_safe); xr=x*rs
            if xr<=-daily_lim: out="B"; break
            bal*=(1+xr)
            if bal<=floor: out="B"; break
            if bal-1>=target: out="P"; dtp.append(k); break
        if out=="P": P+=1
        elif out=="B": B+=1
        else: NR+=1
    return dict(passp=P/iters*100,bustp=B/iters*100,nrp=NR/iters*100,
                med=(np.median(dtp) if dtp else np.nan))

print("="*82)
print("  TRUE OUT-OF-SAMPLE VALIDATION  (select on IS, measure on OOS holdout)")
print(f"  IS days: {SPLIT}   OOS days: {len(common)-SPLIT}   "
      f"OOS span: {OOS_idx[0].date()} -> {OOS_idx[-1].date()}")
print("="*82)
print(f"  {'N':>3}  {'IS_Sh':>7}{'OOS_Sh':>7}  {'IS_pass%':>9}{'OOS_pass%':>10}"
      f"{'OOS_bust%':>10}{'OOS_med':>8}")

for N in [5,7,10,15]:
    if N>len(selected): continue
    basket=selected[:N]
    # weights from IS sharpe (frozen)
    ws={s:max(survivors[s]["shi"],0.01) for s in basket}
    tot=sum(ws.values()); ws={s:v/tot for s,v in ws.items()}
    port=sum(ws[s]*ND[s] for s in basket)
    port_is=port.iloc[:SPLIT]; port_oos=port.iloc[SPLIT:]
    # hard cap calibrated on IS worst day (frozen), applied to OOS
    max_safe=(0.05/port_is.abs().max())*0.90
    is_sh=sharpe(port_is); oos_sh=sharpe(port_oos)
    mc_is =challenge(port_is.values, 2.0,4.0,max_safe)
    mc_oos=challenge(port_oos.values,2.0,4.0,max_safe,max_horizon=min(250,len(port_oos)-1))
    print(f"  {N:>3}  {is_sh:>7.2f}{oos_sh:>7.2f}  {mc_is['passp']:>8.1f}%"
          f"{mc_oos['passp']:>9.1f}%{mc_oos['bustp']:>9.1f}%{mc_oos['med']:>8.0f}")

# ── Detail on N=10 ────────────────────────────────────────────────────────────
N=10; basket=selected[:N]
ws={s:max(survivors[s]["shi"],0.01) for s in basket}
tot=sum(ws.values()); ws={s:v/tot for s,v in ws.items()}
port=sum(ws[s]*ND[s] for s in basket)
port_is=port.iloc[:SPLIT]; port_oos=port.iloc[SPLIT:]
max_safe=(0.05/port_is.abs().max())*0.90

def block(name,seg,idx):
    eq=(1+seg).cumprod(); dd=(eq/eq.cummax()-1).min()
    cagr=eq.iloc[-1]**(ann/len(seg))-1 if eq.iloc[-1]>0 else -1
    by=seg.groupby(idx.year).apply(lambda s:(1+s).prod()-1)
    print(f"\n  [{name}]  Sharpe {sharpe(seg):.2f}  CAGR {cagr*100:.1f}%  maxDD {dd*100:.1f}%")
    print("     "+"  ".join(f"{y}:{v*100:+.1f}%" for y,v in by.items()))

print("\n"+"="*82)
print(f"  DETAIL — N={N} basket: {basket}")
print("="*82)
block("IN-SAMPLE (fitted)",port_is,IS_idx)
block("OUT-OF-SAMPLE (holdout, honest)",port_oos,OOS_idx)
print(f"\n  Hard daily cap (calibrated on IS): {max_safe:.2f}x")
mc_oos=challenge(port_oos.values,2.0,4.0,max_safe,max_horizon=min(250,len(port_oos)-1))
print(f"  OOS challenge: pass {mc_oos['passp']:.1f}%  bust {mc_oos['bustp']:.1f}%  "
      f"none {mc_oos['nrp']:.1f}%  median {mc_oos['med']:.0f} days")
print(f"\n  Per-instrument IS->OOS Sharpe (the honest decay):")
for s in basket:
    iss=sharpe(ND[s].iloc[:SPLIT]); oss=sharpe(ND[s].iloc[SPLIT:])
    tag="⚠ negative OOS" if oss<0 else ""
    print(f"    {s:<8} {survivors[s]['sig']:<12} IS {iss:5.2f} -> OOS {oss:5.2f}  {tag}")
