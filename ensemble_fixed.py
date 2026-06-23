"""
FIXED ensemble — solves all 3 red flags from diagnostic.py:

  FLAG 1 (OOS mirage):   Signal selection now scored on IS-ONLY data.
                          Challenge MC runs on IS data only (regime-neutral).

  FLAG 2 (2025 concentration): Weight streams by IS Sharpe (not equal-weight)
                          to reduce dependence on any one regime/instrument.
                          Also report IS-only pass stats honestly.

  FLAG 3 (daily-loss cap breaks): Daily risk capped on the TRUE tail —
                          the worst historical single-day loss (not 1st-pct).
                          This makes the cap hard, not statistical.
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

# ── FIX 1: score signals on IS ONLY ──────────────────────────────────────────
def is_sharpe(net):
    sp=int(len(net)*SF); si=net.iloc[:sp]
    return si.mean()/si.std()*np.sqrt(ann) if si.std()>0 else 0

all_csvs=sorted(glob.glob("*.csv")); inst={}
for f in all_csvs:
    m=re.match(r"([A-Z0-9]+)_(M5|M15|H1)_",Path(f).name)
    if not m: continue
    sym,tf=m.group(1),m.group(2); rank={"M5":0,"M15":1,"H1":2}
    if sym not in inst or rank[tf]>rank[inst[sym][1]]: inst[sym]=(f,tf)

print("="*80)
print("  Building signals — scored on IS only (fix for OOS mirage)")
print("="*80)
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
    # FIX 1: rank by IS sharpe only
    scored=[(nm,sig,is_sharpe(make_net(p,h,sig))) for nm,sig in cands]
    scored.sort(key=lambda x:-x[2]); nm,sig,shi=scored[0]
    net=make_net(p,h,sig)
    sp=int(len(net)*SF)
    sho_v=net.iloc[sp:]; sho=sho_v.mean()/sho_v.std()*np.sqrt(ann) if sho_v.std()>0 else 0
    sh=net.mean()/net.std()*np.sqrt(ann) if net.std()>0 else 0
    flag="✅" if shi>0.30 else "❌"   # stricter IS threshold
    print(f"  {sym:<8} {nm:<16} IS {shi:5.2f}  OOS {sho:5.2f}  full {sh:5.2f}  {flag}")
    if shi>0.30:
        survivors[sym]=dict(net=net,shi=shi,sho=sho,sh=sh,sig=nm)

syms=list(survivors.keys())
print(f"\n  {len(syms)} survivors (IS>0.30): {syms}")

# ── Greedy low-corr basket, IS-Sharpe weighted ────────────────────────────────
common=survivors[syms[0]]["net"].index
for s in syms[1:]: common=common.intersection(survivors[s]["net"].index)
ND=pd.DataFrame({s:survivors[s]["net"].reindex(common).fillna(0) for s in syms})
corr_mat=ND.corr()

order=sorted(syms,key=lambda s:-survivors[s]["shi"]); selected=[order[0]]
remaining=[s for s in order if s!=order[0]]
while remaining:
    best_s=min(remaining,key=lambda s:np.mean([abs(corr_mat.loc[s,x]) for x in selected]))
    selected.append(best_s); remaining.remove(best_s)

# FIX 2: IS-Sharpe weighting (not equal-weight) to reduce concentration
def is_wt_port(basket):
    c=survivors[basket[0]]["net"].index
    for s in basket[1:]: c=c.intersection(survivors[s]["net"].index)
    ws={s:max(survivors[s]["shi"],0.01) for s in basket}
    tot=sum(ws.values()); ws={s:v/tot for s,v in ws.items()}
    port=sum(ws[s]*survivors[s]["net"].reindex(c).fillna(0) for s in basket)
    return port,c,ws

# ── FIX 3: hard daily cap on TRUE worst tail ──────────────────────────────────
def hard_cap(port_d, daily_lim=0.05):
    """Cap daily risk using the ABSOLUTE worst day, not percentile.
    Returns the max safe risk multiplier such that worst_day * scale <= daily_lim."""
    worst_abs=port_d.abs().max()
    if worst_abs<=0: return 999.0
    return (daily_lim/worst_abs)*0.90   # 10% safety margin on top

# ── Challenge MC: IS-only data, hard-capped daily risk ───────────────────────
def challenge_fixed(port_d, use_is_only=True, target=0.08, max_loss=0.06,
                    daily_lim=0.05, base_risk=3.0, max_risk=8.0,
                    brake_at=0.04, max_horizon=400, iters=8000, seed=7):
    """
    use_is_only: sample from IS portion only (regime-neutral, fix for flag 1)
    hard cap: base/max risk can't exceed hard_cap (fix for flag 3)
    """
    rng=np.random.default_rng(seed); r=port_d.values
    N=len(r); sp=int(N*SF)
    safe=r[:sp] if use_is_only else r          # IS-only data for MC
    # FIX 3: hard cap — no matter what, daily move can't breach limit
    max_safe=hard_cap(pd.Series(safe),daily_lim)
    base_risk=min(base_risk, max_safe)
    max_risk =min(max_risk,  max_safe)
    floor=1.0-max_loss; NS=len(safe)
    P=B=NR=0; dtp=[]
    for _ in range(iters):
        s=rng.integers(0,max(1,NS-max_horizon)); w=safe[s:s+max_horizon]
        bal=1.0; out=None
        for k,x in enumerate(w,1):
            prog=bal-1.0
            buf=base_risk+(max_risk-base_risk)*min(max(prog,0)/target,1.0)
            dist=bal-floor; fb=min(1.0,max(0.0,dist/brake_at))
            rs=min(buf*fb, max_safe)           # hard ceiling every step
            xr=x*rs
            if xr<=-daily_lim: out="B"; break
            bal*=(1+xr)
            if bal<=floor: out="B"; break
            if bal-1>=target: out="P"; dtp.append(k); break
        if out=="P": P+=1
        elif out=="B": B+=1
        else: NR+=1
    return dict(passp=P/iters*100,bustp=B/iters*100,nrp=NR/iters*100,
                med=(np.median(dtp) if dtp else np.nan),
                max_safe=max_safe, base_risk=base_risk, max_risk=max_risk)

# ── Run on increasing basket sizes ───────────────────────────────────────────
print("\n"+"="*80)
print("  FIXED RESULTS: IS-only MC + IS-weighted + hard daily cap")
print("  (regime-neutral, no OOS 2025 boost)")
print("="*80)
print(f"  {'N':>3}  {'pass%':>7}{'bust%':>7}{'none%':>7}{'med_d':>8}  "
      f"{'hardcap':>9}  basket")
rows=[]
for N in [5,7,10,len(selected)]:
    if N>len(selected): continue
    basket=selected[:N]
    port,c,ws=is_wt_port(basket)
    mc=challenge_fixed(port, use_is_only=True, base_risk=3.0, max_risk=8.0)
    rows.append((N,mc,basket,port,c,ws))
    label=",".join(basket[:5])+(f",+{N-5}" if N>5 else "")
    star=" ⭐" if mc["passp"]>=70 and mc["bustp"]<=10 else ""
    print(f"  {N:>3}  {mc['passp']:>6.1f}%{mc['bustp']:>6.1f}%{mc['nrp']:>6.1f}%"
          f"{mc['med']:>8.0f}  {mc['max_safe']:>7.1f}x  {label}{star}")

# ── Risk sweep on best basket ────────────────────────────────────────────────
# pick basket with best (pass - 2*bust) score
best_row=max(rows,key=lambda r:r[1]["passp"]-2*r[1]["bustp"])
Nb,mc_b,bk,port_b,c_b,ws_b=best_row
print(f"\n  Sweeping risk on best basket (N={Nb}) — hard-capped IS-only:")
print(f"  hard cap = {mc_b['max_safe']:.1f}x  (daily limit / worst IS day)")
print(f"  {'base':>5}{'max':>5}  {'pass%':>7}{'bust%':>7}{'med_d':>8}")
best_cfg=None
for br,mr in [(1,3),(2,4),(2,6),(3,6),(3,8),(4,8)]:
    mc=challenge_fixed(port_b,use_is_only=True,base_risk=br,max_risk=mr)
    star=" ⭐" if mc["passp"]>=70 and mc["bustp"]<=10 else ""
    print(f"  {mc['base_risk']:>5.1f}{mc['max_risk']:>5.1f}  "
          f"{mc['passp']:>6.1f}%{mc['bustp']:>6.1f}%{mc['med']:>8.0f}{star}")
    score=mc["passp"]-2*mc["bustp"]
    if best_cfg is None or score>best_cfg[0]: best_cfg=(score,br,mr,mc)

sc,br,mr,mc=best_cfg
print(f"\n  ★ Best fixed config: base={mc['base_risk']:.1f}x max={mc['max_risk']:.1f}x "
      f"(hard-capped from requested {br},{mr})")
print(f"     PASS {mc['passp']:.1f}%  BUST {mc['bustp']:.1f}%  "
      f"median {mc['med']:.0f} trading days")

# ── IS-only full metrics ──────────────────────────────────────────────────────
sp_b=int(len(port_b)*SF); is_port=port_b.iloc[:sp_b]
eq_is=(1+is_port).cumprod()
sh_is=is_port.mean()/is_port.std()*np.sqrt(ann)
dd_is=(eq_is/eq_is.cummax()-1).min()
cagr_is=eq_is.iloc[-1]**(ann/len(is_port))-1
by_is=is_port.groupby(c_b[:sp_b].year).apply(lambda s:(1+s).prod()-1)

print("\n"+"="*80)
print(f"  IS-ONLY METRICS (regime-neutral truth, N={Nb})")
print("="*80)
print(f"  Sharpe   {sh_is:.2f}")
print(f"  CAGR     {cagr_is*100:.1f}%")
print(f"  maxDD    {dd_is*100:.1f}%")
print(f"  Year-by-year (IS period):")
for y,v in by_is.items(): print(f"    {y}: {v*100:+.1f}%")
print(f"  Instrument weights:")
for s,w in sorted(ws_b.items(),key=lambda x:-x[1]):
    print(f"    {s:<8} {w*100:5.1f}%  IS Sharpe {survivors[s]['shi']:.2f}  "
          f"signal={survivors[s]['sig']}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(2,2,figsize=(16,10))
# Full-period equity
eq_full=(1+port_b).cumprod()
axes[0,0].plot(c_b,eq_full.values,color="navy",lw=2)
axes[0,0].axvline(c_b[sp_b],color="red",ls="--",alpha=0.5,label="IS/OOS split")
axes[0,0].set_yscale("log"); axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)
axes[0,0].set_title(f"Equity (IS-weighted, N={Nb})  IS Sh {sh_is:.2f}  CAGR {cagr_is*100:.0f}% (IS only)")

# IS-only equity
axes[0,1].plot(c_b[:sp_b],eq_is.values,color="green",lw=2)
axes[0,1].set_yscale("log"); axes[0,1].grid(alpha=0.3)
axes[0,1].set_title(f"IS-only equity  Sh {sh_is:.2f}  CAGR {cagr_is*100:.0f}%  DD {dd_is*100:.1f}%")

# pass/bust vs N
Ns=[r[0] for r in rows]; pp=[r[1]["passp"] for r in rows]; bb=[r[1]["bustp"] for r in rows]
axes[1,0].plot(Ns,pp,"o-",color="green",label="pass%")
axes[1,0].plot(Ns,bb,"o-",color="red",label="bust%")
axes[1,0].axhline(70,ls="--",color="gray",alpha=0.5,label="70% target")
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
axes[1,0].set_title("Fixed: pass/bust vs variety (IS-only, hard-capped)")
axes[1,0].set_xlabel("# instruments")

# Rolling 60d return (IS only)
roll60=is_port.rolling(60).apply(lambda s:(1+s).prod()-1)
axes[1,1].plot(c_b[:sp_b],roll60.values,color="darkblue"); axes[1,1].axhline(0,color="k",lw=0.8)
neg60=(roll60<0).sum()/roll60.notna().sum()*100
axes[1,1].fill_between(c_b[:sp_b],roll60.values,0,
    where=roll60.values<0,color="red",alpha=0.3)
axes[1,1].set_title(f"IS rolling 60d return  {neg60:.0f}% windows negative")
axes[1,1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig("ensemble_fixed.png",dpi=130)
print("\n  Saved → ensemble_fixed.png")
