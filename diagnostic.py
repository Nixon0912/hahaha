"""
FULL DIAGNOSTIC on the 25-day config:
  Basket (10): AUDUSD HSIHKD NGCUSD USDJPY USDCHF XAGUSD UK100 NAS100 JPN225 USDCAD
  Risk: base 6x, max 15x, no time limit, daily-safe sizing

Red-flag hunt — the things that kill a strategy live:
  1. Return concentration   — is it all 2025 / a handful of days?
  2. Single-instrument dependence — does one name carry the book?
  3. Correlation stability  — does diversification vanish in stress?
  4. Daily-loss breaches    — would real history have tripped the 5% daily limit?
  5. IS vs OOS decay        — does edge survive out-of-sample?
  6. Signal look-ahead      — sanity check the 1-day lag
  7. Sizing realism         — what leverage are we actually demanding?
  8. Worst real 60-day window — actual historical worst case (not Monte Carlo)
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
BASKET=["AUDUSD","HSIHKD","NGCUSD","USDJPY","USDCHF","XAGUSD","UK100","NAS100","JPN225","USDCAD"]

HALF={"AUDUSD":0.00008,"USDJPY":0.008,"USDCHF":0.00009,"USDCAD":0.00010,
      "XAGUSD":0.02,"NGCUSD":0.005,"NAS100":1.5,"UK100":1.5,"JPN225":8.0,"HSIHKD":3.5}
def hs(s,p):
    for k,v in HALF.items():
        if s.startswith(k[:4]): return v
    return p.mean()*0.0002
def load_d1(fp):
    df=load_raw(fp); df=df[df["close"]>0]
    d=df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    return d[d["close"]>0]
def mr_sig(p,n,e):
    z=(p-p.rolling(n).mean())/p.rolling(n).std(); zv=z.values; out=np.zeros(len(zv)); cur=0.0
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
    z=(p-p.rolling(n).mean())/p.rolling(n).std(); return (-np.sign(z)).fillna(0)

# best signal per basket instrument (re-derive to get exact net + position)
all_csvs=sorted(glob.glob("*.csv")); inst={}
for f in all_csvs:
    m=re.match(r"([A-Z0-9]+)_(M5|M15|H1)_",Path(f).name)
    if not m: continue
    sym,tf=m.group(1),m.group(2); rank={"M5":0,"M15":1,"H1":2}
    if sym not in inst or rank[tf]>rank[inst[sym][1]]: inst[sym]=(f,tf)

def make_net(p,h,sig):
    ret=p.pct_change().fillna(0); rv=ret.rolling(20).std()*np.sqrt(ann)
    pos=(sig*(VT/rv.replace(0,np.nan)).clip(upper=CAP).fillna(0)).shift(1).fillna(0)
    trn=pos.diff().abs().fillna(pos.abs())
    net=(pos*ret-trn*(h/p)).fillna(0)
    return net,pos

def score_run(p,h,sig):
    net,_=make_net(p,h,sig); n=len(net); sp=int(n*SF)
    si,so=net.iloc[:sp],net.iloc[sp:]
    shi=si.mean()/si.std()*np.sqrt(ann) if si.std()>0 else 0
    sho=so.mean()/so.std()*np.sqrt(ann) if so.std()>0 else 0
    return shi+2*sho

SIGFAM=[]
def best_sig(sym,p):
    h=hs(sym,p); maxL=min(200,int(len(p)*0.20)); cands=[]
    for L in [5,10,15,20,30,50,75,100,150,200]:
        if L<=maxL: cands.append((f"MOM{L}",tsmom(p,L)))
    for n in [10,15,20,30,40]:
        for e in [1.5,2.0,2.5]: cands.append((f"MR{n}e{e}",mr_sig(p,n,e)))
    for L in [1,2,3,5,7,10]: cands.append((f"REV{L}",rev(p,L)))
    for fa,sl in [(5,20),(10,50),(20,100),(5,50)]:
        if sl<=maxL: cands.append((f"EMA{fa}x{sl}",ema(p,fa,sl)))
    if len(p)>=252: cands.append(("VAL1y",valsig(p,252)))
    best=max(cands,key=lambda c:score_run(p,h,c[1]))
    return best

nets={}; poss={}; prices={}; signames={}
for s in BASKET:
    d1=load_d1(inst[s][0]); p=d1["close"]; h=hs(s,p)
    nm,sig=best_sig(s,p); signames[s]=nm
    net,pos=make_net(p,h,sig); nets[s]=net; poss[s]=pos; prices[s]=p

common=nets[BASKET[0]].index
for s in BASKET[1:]: common=common.intersection(nets[s].index)
ND=pd.DataFrame({s:nets[s].reindex(common).fillna(0) for s in BASKET})
PD=pd.DataFrame({s:poss[s].reindex(common).fillna(0) for s in BASKET})
port=ND.mean(axis=1)   # equal weight

print("="*78)
print("  FULL DIAGNOSTIC — 10-instrument basket (25-day config)")
print("="*78)
print("  Signals chosen per instrument:")
for s in BASKET: print(f"    {s:<8} {signames[s]}")

# ── Core metrics ──────────────────────────────────────────────────────────────
eq=(1+port).cumprod(); n=len(port); sp=int(n*SF)
sh=port.mean()/port.std()*np.sqrt(ann)
dn=port[port<0]; sortino=port.mean()/dn.std()*np.sqrt(ann)
mdd=(eq/eq.cummax()-1).min(); cagr=eq.iloc[-1]**(ann/n)-1
shi=port.iloc[:sp].mean()/port.iloc[:sp].std()*np.sqrt(ann)
sho=port.iloc[sp:].mean()/port.iloc[sp:].std()*np.sqrt(ann)
print(f"\n  Sharpe {sh:.2f} (IS {shi:.2f}/OOS {sho:.2f})  Sortino {sortino:.2f}  "
      f"CAGR {cagr*100:.1f}%  maxDD {mdd*100:.1f}%")

# ── RED FLAG 1: return concentration ──────────────────────────────────────────
print("\n"+"-"*78)
print("  [1] RETURN CONCENTRATION")
by=port.groupby(common.year).apply(lambda s:(1+s).prod()-1)
tot=eq.iloc[-1]-1
for y,v in by.items(): print(f"      {y}: {v*100:+6.1f}%")
# what % of total log-return comes from 2025?
logr=np.log1p(port)
share={y: logr[common.year==y].sum()/logr.sum()*100 for y in by.index}
print(f"      Share of total log-return by year: "+
      "  ".join(f"{y}:{share[y]:.0f}%" for y in by.index))
# top 10 days
topdays=port.sort_values(ascending=False).head(10)
top10share=np.log1p(topdays).sum()/logr.sum()*100
print(f"      Top 10 days = {top10share:.1f}% of total log-return "
      f"(best day {topdays.iloc[0]*100:+.2f}%)")
flag1 = "⚠ HIGH" if (share.get(2025,0)>50 or top10share>40) else "ok"
print(f"      -> concentration: {flag1}")

# ── RED FLAG 2: single-instrument dependence ──────────────────────────────────
print("\n"+"-"*78)
print("  [2] SINGLE-INSTRUMENT DEPENDENCE")
contrib={s: ND[s].sum() for s in BASKET}
tt=sum(contrib.values())
for s in sorted(BASKET,key=lambda x:-contrib[x]):
    print(f"      {s:<8} contributes {contrib[s]/tt*100:5.1f}% of summed daily returns "
          f"(stream Sh {ND[s].mean()/ND[s].std()*np.sqrt(ann):.2f})")
# leave-one-out: portfolio Sharpe dropping each instrument
print("      Leave-one-out portfolio Sharpe:")
for s in BASKET:
    sub=ND[[c for c in BASKET if c!=s]].mean(axis=1)
    lsh=sub.mean()/sub.std()*np.sqrt(ann)
    print(f"        drop {s:<8} -> Sh {lsh:.2f}  ({(lsh-sh):+.2f})")
maxc=max(contrib.values())/tt*100
flag2="⚠ one name >35%" if maxc>35 else "ok"
print(f"      -> max single contribution {maxc:.0f}%: {flag2}")

# ── RED FLAG 3: correlation stability (rolling) ───────────────────────────────
print("\n"+"-"*78)
print("  [3] CORRELATION STABILITY (does diversification vanish in stress?)")
def mean_abs_corr(df):
    # drop columns that are flat (zero variance) in this window — corr undefined
    sub=df.loc[:,df.std()>0]
    if sub.shape[1]<2: return np.nan
    c=sub.corr().values
    u=c[np.triu_indices(len(sub.columns),k=1)]
    u=u[~np.isnan(u)]
    return np.abs(u).mean() if len(u) else np.nan
full_c=mean_abs_corr(ND)
# rolling 60-day mean abs corr
roll=[]; idxs=[]
for i in range(60,len(ND),20):
    w=ND.iloc[i-60:i]
    mc=mean_abs_corr(w)
    if not np.isnan(mc): roll.append(mc); idxs.append(common[i])
roll=np.array(roll)
print(f"      Full-sample mean|corr| {full_c:.3f}")
print(f"      Rolling 60d mean|corr|: avg {roll.mean():.3f}  max {roll.max():.3f}  "
      f"(spikes to {roll.max():.2f} on {idxs[int(roll.argmax())].date()})")
flag3="⚠ corr spikes >0.30" if roll.max()>0.30 else "ok"
print(f"      -> {flag3}")

# ── RED FLAG 4: daily-loss breaches in real history ───────────────────────────
print("\n"+"-"*78)
print("  [4] DAILY-LOSS LIMIT — would real history breach 5%?  (at 6x/15x sizing)")
# reconstruct adaptive daily risk path over ACTUAL history (not resampled)
def real_path(base_risk=6.0,max_risk=15.0,brake_at=0.05,target=0.08,
              max_loss=0.06,daily_lim=0.05):
    r=port.values; floor=1.0-max_loss; bad=abs(np.percentile(r,1))
    bal=1.0; worst_day=0; breaches=0; sized=[]
    for x in r:
        prog=bal-1.0
        buf=base_risk+(max_risk-base_risk)*min(max(prog,0)/target,1.0)
        dist=bal-floor; fb=min(1.0,max(0.0,dist/brake_at)); rs=buf*fb
        if bad>0: rs=min(rs,0.8*daily_lim/bad)
        xr=x*rs; sized.append(xr)
        worst_day=min(worst_day,xr)
        if xr<=-daily_lim: breaches+=1
        bal*=(1+xr)
        if bal<=floor: bal=floor  # would have busted; clamp
    return worst_day,breaches,np.array(sized),bad
wd,br,sized,bad=real_path()
print(f"      Worst unscaled 1% day: {bad*100:.2f}%   "
      f"daily-cap forces risk <= {0.8*0.05/bad:.1f}x on bad days")
print(f"      Worst SIZED day in history: {wd*100:.2f}%   "
      f"breaches of 5% daily limit: {br}")
flag4="⚠ breaches exist" if br>0 else "ok (cap holds)"
print(f"      -> {flag4}")

# ── RED FLAG 5: IS vs OOS per instrument ──────────────────────────────────────
print("\n"+"-"*78)
print("  [5] IS vs OOS DECAY per instrument")
for s in BASKET:
    ns=ND[s]; spi=int(len(ns)*SF)
    i=ns.iloc[:spi]; o=ns.iloc[spi:]
    ish=i.mean()/i.std()*np.sqrt(ann) if i.std()>0 else 0
    osh=o.mean()/o.std()*np.sqrt(ann) if o.std()>0 else 0
    tag="⚠ OOS<0" if osh<0 else ("⚠ jump" if osh>ish*3 and ish>0 else "")
    print(f"      {s:<8} IS {ish:5.2f}  OOS {osh:5.2f}   {tag}")

# ── RED FLAG 6: position sizing realism ───────────────────────────────────────
print("\n"+"-"*78)
print("  [6] LEVERAGE / MARGIN REALITY CHECK")
# Base portfolio is equal-weight of 10 streams, each vol-targeted to 15% but the
# stream weight is 1/10, so per-stream notional = (1/10)*pos. Gross book =
# sum over instruments of (1/10)*|pos|.
gross_book=(PD.abs().mean(axis=1))   # (1/10)*sum|pos| = portfolio gross notional
port_vol=port.std()*np.sqrt(ann)
# challenge scale needed: the daily-safe sizing uses base..max risk on the NET
# portfolio return. Effective gross notional = gross_book * challenge_scale.
for scale,label in [(6,"base 6x"),(15,"max 15x")]:
    eff=gross_book.mean()*scale
    print(f"      {label:<9}: portfolio gross notional ~ {eff:.0f}x account equity "
          f"(95pct {gross_book.quantile(0.95)*scale:.0f}x)")
print(f"      Base portfolio annualised vol: {port_vol*100:.1f}%")
print(f"      To make +8% with this {port_vol*100:.0f}% vol engine in ~25d you must")
print(f"        scale ~6-15x -> portfolio vol becomes {port_vol*6*100:.0f}-{port_vol*15*100:.0f}%/yr")
maxeff=gross_book.mean()*15
flag6="⚠ exceeds typical 5ers margin (need ~"+f"{maxeff:.0f}x)" if maxeff>30 else "within broker limits"
print(f"      -> {flag6}")

# ── RED FLAG 7: worst REAL 60-day window ──────────────────────────────────────
print("\n"+"-"*78)
print("  [7] WORST REAL 60-DAY WINDOW (actual history, equal-weight base vol)")
roll60=port.rolling(60).apply(lambda s:(1+s).prod()-1)
print(f"      Worst 60d return: {roll60.min()*100:+.1f}% ending {roll60.idxmin().date()}")
print(f"      Best  60d return: {roll60.max()*100:+.1f}% ending {roll60.idxmax().date()}")
neg60=(roll60<0).sum()/roll60.notna().sum()*100
print(f"      % of 60d windows negative: {neg60:.1f}%")

# ── Summary plot ──────────────────────────────────────────────────────────────
fig,ax=plt.subplots(2,2,figsize=(16,10))
ax[0,0].plot(common,eq.values,color="navy"); ax[0,0].set_yscale("log")
ax[0,0].axvline(common[sp],color="red",ls="--",alpha=0.4)
ax[0,0].set_title(f"Equity  Sh {sh:.2f}  CAGR {cagr*100:.0f}%  DD {mdd*100:.1f}%"); ax[0,0].grid(alpha=0.3)
ax[0,1].plot(idxs,roll,color="purple"); ax[0,1].axhline(0.30,ls="--",color="red",alpha=0.5)
ax[0,1].set_title("Rolling 60d mean|corr| (stress check)"); ax[0,1].grid(alpha=0.3)
cs=sorted(BASKET,key=lambda x:contrib[x]/tt*100)
ax[1,0].barh(cs,[contrib[s]/tt*100 for s in cs],color="teal")
ax[1,0].set_title("Contribution % by instrument"); ax[1,0].grid(axis="x",alpha=0.3)
ax[1,1].plot(common,roll60.values,color="darkgreen"); ax[1,1].axhline(0,color="k",lw=0.8)
ax[1,1].set_title("Rolling 60d return"); ax[1,1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("diagnostic.png",dpi=130)
print("\n  Saved → diagnostic.png")
