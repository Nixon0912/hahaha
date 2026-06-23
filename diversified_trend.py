"""
DIVERSIFIED TREND — the honest, real strategy. CTA / managed-futures style.

Principle (what actually survives live):
  - ONE fixed trend rule, applied IDENTICALLY to every instrument. No tuning.
  - Breadth across all ~25 markets -> sqrt(N) lifts Sharpe, cuts drawdown.
  - Proper portfolio vol-targeting to a chosen annual vol.
  - Honest IS/OOS split + per-year walk-forward (rule is FIXED, so every year
    after warmup is genuinely out-of-sample for that year's data).

Trend rule tested: multi-timeframe trend = average sign of several lookbacks
(robust, standard CTA construction — not a single fragile parameter).
"""
import sys, warnings, glob, re
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

ann=252; CAP=2.0

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
def sharpe(x): return x.mean()/x.std()*np.sqrt(ann) if x.std()>0 else 0

# ── THE FIXED TREND RULE (identical for every instrument) ─────────────────────
def trend_signal(p):
    """Multi-timeframe trend: average of sign(EMA_fast - EMA_slow) over several
    standard CTA horizons. Continuous in [-1,1]. No per-asset tuning."""
    pairs=[(8,24),(16,48),(32,96),(64,192)]   # ~classic CTA filter bank
    s=sum(np.sign(p.ewm(span=f).mean()-p.ewm(span=sl).mean()) for f,sl in pairs)/len(pairs)
    return s.fillna(0)

def vol_target_net(p,h,sig,target_vol=0.15):
    ret=p.pct_change().fillna(0)
    rv=ret.rolling(50).std()*np.sqrt(ann)
    pos=(sig*(target_vol/rv.replace(0,np.nan)).clip(upper=CAP).fillna(0)).shift(1).fillna(0)
    trn=pos.diff().abs().fillna(pos.abs())
    return (pos*ret-trn*(h/p)).fillna(0)

# ── Load universe ─────────────────────────────────────────────────────────────
all_csvs=sorted(glob.glob("*.csv")); inst={}
for f in all_csvs:
    m=re.match(r"([A-Z0-9]+)_(M5|M15|H1)_",Path(f).name)
    if not m: continue
    sym,tf=m.group(1),m.group(2); rank={"M5":0,"M15":1,"H1":2}
    if sym not in inst or rank[tf]>rank[inst[sym][1]]: inst[sym]=(f,tf)

nets={}
for sym,(fp,tf) in sorted(inst.items()):
    d1=load_d1(fp)
    if d1 is None or len(d1)<400: continue
    p=d1["close"]
    nets[sym]=vol_target_net(p,hs(sym,p),trend_signal(p))
syms=list(nets.keys())
print(f"  Universe: {len(syms)} instruments, ONE fixed trend rule each")

common=None
for s in syms:
    common=nets[s].index if common is None else common.intersection(nets[s].index)
ND=pd.DataFrame({s:nets[s].reindex(common).fillna(0) for s in syms})
port_raw=ND.mean(axis=1)   # equal weight

# ── Portfolio vol-target to a clean annual vol ────────────────────────────────
def vol_scale(port,target=0.15,cap=3.0):
    rv=port.rolling(30).std()*np.sqrt(ann)
    sc=(target/rv.replace(0,np.nan)).clip(upper=cap).fillna(1.0).shift(1).fillna(1.0)
    return port*sc

port=vol_scale(port_raw,target=0.15)

# ── Metrics ───────────────────────────────────────────────────────────────────
sp=int(len(port)*0.70)
eq=(1+port).cumprod(); dd=(eq/eq.cummax()-1).min()
cagr=eq.iloc[-1]**(ann/len(port))-1
sh=sharpe(port); shi=sharpe(port.iloc[:sp]); sho=sharpe(port.iloc[sp:])
sortino=port.mean()/port[port<0].std()*np.sqrt(ann)
by=port.groupby(common.year).apply(lambda s:(1+s).prod()-1)
mcorr=np.abs(ND.corr().values[np.triu_indices(len(syms),k=1)]).mean()

print("\n"+"="*78)
print("  DIVERSIFIED TREND — fixed rule, equal weight, 15% vol target")
print("="*78)
print(f"  Sharpe {sh:.2f}  (IS {shi:.2f} / OOS {sho:.2f})   Sortino {sortino:.2f}")
print(f"  CAGR {cagr*100:.1f}%   maxDD {dd*100:.1f}%   mean|corr| {mcorr:.3f}")
print(f"  Positive years: {(by>0).sum()}/{len(by)}")
print("  "+"  ".join(f"{y}:{v*100:+.0f}%" for y,v in by.items()))

# ── Per-year walk-forward sanity (rule is fixed -> each year is OOS) ───────────
print("\n  Each year is genuinely OOS (rule never changes):")
yr_sh={}
for y in by.index:
    seg=port[common.year==y]
    yr_sh[y]=sharpe(seg)
print("  "+"  ".join(f"{y}:Sh{yr_sh[y]:+.1f}" for y in by.index))
pos_yr_sh=sum(1 for v in yr_sh.values() if v>0)
print(f"  Years with positive Sharpe: {pos_yr_sh}/{len(yr_sh)}")

# ── Per-instrument trend contribution (is it broad or concentrated?) ──────────
print("\n  Per-instrument standalone trend Sharpe (full sample):")
inst_sh={s:sharpe(ND[s]) for s in syms}
for s in sorted(syms,key=lambda x:-inst_sh[x]):
    bar="#"*int(max(inst_sh[s],0)*15)
    print(f"    {s:<8} {inst_sh[s]:+.2f}  {bar}")
pos_inst=sum(1 for v in inst_sh.values() if v>0)
print(f"  Instruments with positive standalone trend: {pos_inst}/{len(syms)}")

# ── Compare vol targets (return dial) ─────────────────────────────────────────
print("\n  Return dial (same strategy, different vol target):")
print(f"  {'target_vol':>10}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}")
for tv in [0.10,0.15,0.25,0.40]:
    pt=vol_scale(port_raw,target=tv)
    et=(1+pt).cumprod()
    print(f"  {tv*100:>9.0f}%{(et.iloc[-1]**(ann/len(pt))-1)*100:>7.1f}%"
          f"{sharpe(pt):>8.2f}{(et/et.cummax()-1).min()*100:>7.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(2,1,figsize=(15,9),gridspec_kw={"height_ratios":[3,1]})
ax[0].plot(common,eq.values,color="darkgreen",lw=2,
           label=f"Diversified Trend  Sh {sh:.2f} (IS {shi:.2f}/OOS {sho:.2f})  CAGR {cagr*100:.0f}%")
ax[0].axvline(common[sp],color="red",ls="--",alpha=0.5,label="IS/OOS split")
ax[0].set_yscale("log"); ax[0].legend(fontsize=10); ax[0].grid(alpha=0.3)
ax[0].set_title(f"Diversified Trend-Following — {len(syms)} markets, ONE fixed rule, no tuning\n"
                f"Sharpe {sh:.2f}  CAGR {cagr*100:.1f}%  maxDD {dd*100:.1f}%  "
                f"{(by>0).sum()}/{len(by)} positive years")
uw=(eq/eq.cummax()-1)*100
ax[1].fill_between(common,uw.values,0,color="darkgreen",alpha=0.35)
ax[1].axvline(common[sp],color="red",ls="--",alpha=0.5)
ax[1].set_title(f"Drawdown  maxDD {dd*100:.1f}%"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("diversified_trend.png",dpi=130)
print("\n  Saved → diversified_trend.png")
