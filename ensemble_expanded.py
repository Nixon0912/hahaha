"""
Expanded ensemble — increase VARIETY of streams to smooth daily returns and
lift the 5ers no-time-limit pass rate further.

Builds the best signal per instrument across all 25 instruments (4 premia),
keeps survivors, greedily selects a low-correlation basket of increasing size
(3 -> 5 -> 7 -> 10 -> all), and runs the no-time-limit challenge MC on each
to see how pass/bust improves with added variety.
"""
import sys, warnings, glob, re
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

ann = 252; VT = 0.15; CAP = 3.0; SF = 0.70

HALF = {
    "AUDUSD":0.00008,"EURUSD":0.00007,"USDJPY":0.008,"USDCAD":0.00010,
    "USDCHF":0.00009,"AUDNZD":0.00012,"XAUUSD":0.15,"XAGUSD":0.02,
    "XPDUSD":3.0,"XPTUSD":0.8,"XTIUSD":0.04,"XBRUSD":0.04,"NGCUSD":0.005,
    "CUCUSD":0.0025,"NAS100":1.5,"SP500":0.5,"US30":3.0,"DAX40":2.0,
    "ESXEUR":1.0,"F40EUR":0.8,"IBXEUR":4.0,"UK100":1.5,"ASXAUD":2.0,
    "JPN225":8.0,"HSIHKD":3.5,
}
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

def run(p,h,sig,name):
    ret=p.pct_change().fillna(0)
    rv=ret.rolling(20).std()*np.sqrt(ann)
    pos=(sig*(VT/rv.replace(0,np.nan)).clip(upper=CAP).fillna(0)).shift(1).fillna(0)
    trn=pos.diff().abs().fillna(pos.abs())
    net=(pos*ret-trn*(h/p)).fillna(0)
    n=len(net); sp=int(n*SF)
    si,so=net.iloc[:sp],net.iloc[sp:]
    shi=si.mean()/si.std()*np.sqrt(ann) if si.std()>0 else 0
    sho=so.mean()/so.std()*np.sqrt(ann) if so.std()>0 else 0
    sh=net.mean()/net.std()*np.sqrt(ann) if net.std()>0 else 0
    return dict(name=name,net=net,sh=sh,shi=shi,sho=sho,score=shi+2*sho)

# ── Build best signal per instrument ──────────────────────────────────────────
all_csvs=sorted(glob.glob("*.csv")); inst={}
for f in all_csvs:
    m=re.match(r"([A-Z0-9]+)_(M5|M15|H1)_",Path(f).name)
    if not m: continue
    sym,tf=m.group(1),m.group(2); rank={"M5":0,"M15":1,"H1":2}
    if sym not in inst or rank[tf]>rank[inst[sym][1]]: inst[sym]=(f,tf)

survivors={}
for sym,(fp,tf) in sorted(inst.items()):
    d1=load_d1(fp)
    if d1 is None or len(d1)<300: continue
    p=d1["close"]; h=hs(sym,p); maxL=min(200,int(len(p)*0.20)); res=[]
    for L in [5,10,15,20,30,50,75,100,150,200]:
        if L<=maxL: res.append(run(p,h,tsmom(p,L),f"MOM{L}"))
    for n in [10,15,20,30,40]:
        for e in [1.5,2.0,2.5]: res.append(run(p,h,mr_sig(p,n,e),f"MR{n}e{e}"))
    for L in [1,2,3,5,7,10]: res.append(run(p,h,rev(p,L),f"REV{L}"))
    for fa,sl in [(5,20),(10,50),(20,100),(5,50)]:
        if sl<=maxL: res.append(run(p,h,ema(p,fa,sl),f"EMA{fa}x{sl}"))
    if len(p)>=252: res.append(run(p,h,valsig(p,252),"VAL1y"))
    res.sort(key=lambda x:-x["score"]); best=res[0]
    if best["shi"]>0.25 and best["sho"]>0.35:
        survivors[sym]=best

syms=list(survivors.keys())
print(f"  {len(syms)} survivors: {syms}")

# ── Greedy low-correlation selection ──────────────────────────────────────────
common=survivors[syms[0]]["net"].index
for s in syms[1:]: common=common.intersection(survivors[s]["net"].index)
ND=pd.DataFrame({s:survivors[s]["net"].reindex(common).fillna(0) for s in syms})
corr=ND.corr()

# start from highest-score, greedily add the stream that keeps avg corr lowest
order=sorted(syms,key=lambda s:-survivors[s]["score"])
selected=[order[0]]
remaining=[s for s in order if s!=order[0]]
while remaining:
    # pick remaining stream with lowest avg |corr| to current selection
    best_s=min(remaining,key=lambda s:np.mean([abs(corr.loc[s,x]) for x in selected]))
    selected.append(best_s); remaining.remove(best_s)
print(f"  Greedy diversification order: {selected}")

# ── No-time-limit challenge MC on a given basket ──────────────────────────────
def challenge(port_d, target=0.08, max_loss=0.06, daily_lim=0.05,
              base_risk=2.0, max_risk=6.0, brake_at=0.05,
              max_horizon=400, iters=8000, seed=7):
    rng=np.random.default_rng(seed); r=port_d.values; N=len(r)
    floor=1.0-max_loss; bad=abs(np.percentile(r,1))
    P=B=NR=0; dtp=[]
    for _ in range(iters):
        s=rng.integers(0,max(1,N-max_horizon)); w=r[s:s+max_horizon]
        bal=1.0; out=None
        for k,x in enumerate(w,1):
            prog=bal-1.0
            buf=base_risk+(max_risk-base_risk)*min(max(prog,0)/target,1.0)
            dist=bal-floor; fb=min(1.0,max(0.0,dist/brake_at))
            rs=buf*fb
            if bad>0: rs=min(rs,0.8*daily_lim/bad)
            xr=x*rs
            if xr<=-daily_lim: out="B"; break
            bal*=(1+xr)
            if bal<=floor: out="B"; break
            if bal-1>=target: out="P"; dtp.append(k); break
        if out=="P": P+=1
        elif out=="B": B+=1
        else: NR+=1
    return dict(passp=P/iters*100,bustp=B/iters*100,nrp=NR/iters*100,
                med=(np.median(dtp) if dtp else np.nan))

def basket_port(basket):
    c=survivors[basket[0]]["net"].index
    for s in basket[1:]: c=c.intersection(survivors[s]["net"].index)
    return sum((1/len(basket))*survivors[s]["net"].reindex(c).fillna(0) for s in basket), c

print("\n"+"="*78)
print("  VARIETY vs 5ers PASS RATE  (no time limit, daily-safe sizing)")
print("="*78)
print(f"  {'N':>3}{'basket':<46}{'mean|corr|':>10}{'Sh':>6}{'pass%':>7}{'bust%':>7}")
rows=[]
for N in [3,4,5,7,10,len(selected)]:
    if N>len(selected): continue
    basket=selected[:N]
    port,c=basket_port(basket)
    sh=port.mean()/port.std()*np.sqrt(ann) if port.std()>0 else 0
    sub=ND[basket]; cc=sub.corr().values
    mcorr=np.abs(cc[np.triu_indices(len(basket),k=1)]).mean()
    mc=challenge(port)
    rows.append((N,mcorr,sh,mc))
    label=",".join(basket) if N<=7 else ",".join(basket[:6])+f",+{N-6}"
    print(f"  {N:>3}{label:<46}{mcorr:>10.3f}{sh:>6.2f}{mc['passp']:>6.1f}%{mc['bustp']:>6.1f}%")

# Best basket = highest pass with bust<15
best=max(rows,key=lambda r:r[3]["passp"]-2*r[3]["bustp"])
Nb=best[0]
print(f"\n  ★ Best variety: N={Nb}  pass {best[3]['passp']:.1f}%  bust {best[3]['bustp']:.1f}%  "
      f"median {best[3]['med']:.0f}d  meanCorr {best[1]:.3f}")
print(f"     Basket: {selected[:Nb]}")

# ── Plot pass/bust vs N ───────────────────────────────────────────────────────
Ns=[r[0] for r in rows]; pp=[r[3]["passp"] for r in rows]; bb=[r[3]["bustp"] for r in rows]
cc=[r[1] for r in rows]
fig,ax=plt.subplots(1,2,figsize=(14,5))
ax[0].plot(Ns,pp,"o-",color="green",label="pass %")
ax[0].plot(Ns,bb,"o-",color="red",label="bust %")
ax[0].set_xlabel("# instruments (variety)"); ax[0].set_ylabel("%")
ax[0].axhline(70,ls="--",color="gray",alpha=0.5); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[0].set_title("5ers pass/bust vs variety")
ax[1].plot(Ns,cc,"o-",color="navy")
ax[1].set_xlabel("# instruments"); ax[1].set_ylabel("mean |corr|")
ax[1].grid(alpha=0.3); ax[1].set_title("Diversification (lower=better)")
plt.tight_layout(); plt.savefig("ensemble_expanded.png",dpi=130)
print("\n  Saved → ensemble_expanded.png")
