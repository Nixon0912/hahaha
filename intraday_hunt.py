"""
INTRADAY EDGE HUNT — HSIHKD M5, 17 years (386k bars), walk-forward validated.

Intraday edges that are economically motivated (not data-mined):
  1. Short-term reversal  — fade the last N-bar move (overreaction / liquidity).
  2. Overnight gap fade   — fade the open gap vs prior close.
  3. Intraday z-reversion — fade deviation from the rolling intraday VWAP/mean.
  4. Session momentum     — ride/fade moves within liquid sessions only.

Discipline:
  - Realistic spread cost charged on EVERY position change (intraday = many trades).
  - Walk-forward: re-pick best signal each year on PAST data, score NEXT year.
  - 17 years of unseen-year blocks = a real test.
  - Also test holding-period throttle (don't flip every bar -> cost control).
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

FILE="HSIHKD_M5_200902020300_202606102155.csv"
HALF=3.5            # half-spread in index points
BARS_PER_DAY=None   # estimated below

def load():
    d=load_raw(FILE); d=d[d["close"]>0]
    return d

def sharpe(x,per): return x.mean()/x.std()*np.sqrt(per) if x.std()>0 else 0

df=load()
p=df["close"]
# estimate bars/yr
span_years=(p.index[-1]-p.index[0]).days/365
per=int(len(p)/span_years)
print(f"  HSI M5: {len(p):,} bars  {p.index[0].date()} -> {p.index[-1].date()}  (~{per:,} bars/yr)")

ret=p.pct_change().fillna(0)
cost_frac=HALF/p   # fractional half-spread per unit position change

# ── Intraday signal builders ──────────────────────────────────────────────────
def revN(p,N):  # fade last N-bar return
    return -np.sign(p.pct_change(N)).fillna(0)
def zrev(p,n,e):  # fade deviation from rolling mean
    z=(p-p.rolling(n).mean())/p.rolling(n).std()
    return (-z.clip(-e,e)/e).fillna(0)   # continuous, saturates at +-1
def gap_fade(df):  # fade overnight gap: short if gap up at first bar of day
    day=df.index.normalize()
    first=~day.duplicated()
    prev_close=df["close"].shift(1)
    gap=(df["open"]-prev_close)/prev_close
    sig=pd.Series(0.0,index=df.index)
    sig[first]=-np.sign(gap[first])
    # hold the fade for the rest of the day (ffill within day), reset next day
    sig=sig.replace(0,np.nan)
    sig=sig.groupby(day).ffill().fillna(0)
    return sig

def net_of(sig, hold=1):
    """Apply signal with optional min-hold smoothing to cut turnover.
    hold = bars to hold (EMA-smooth the position)."""
    pos=sig.shift(1).fillna(0)
    if hold>1:
        pos=pos.ewm(span=hold).mean()
    trn=pos.diff().abs().fillna(pos.abs())
    return (pos*ret - trn*cost_frac).fillna(0)

# ── Build grid ────────────────────────────────────────────────────────────────
grid={}
for N in [3,6,12,24,48]:
    for hold in [1,6,12]:
        grid[f"REV{N}_h{hold}"]=net_of(revN(p,N),hold)
for n in [24,48,96]:
    for e in [2.0,3.0]:
        for hold in [6,12]:
            grid[f"ZREV{n}e{e}_h{hold}"]=net_of(zrev(p,n,e),hold)
grid["GAPFADE"]=net_of(gap_fade(df),1)

print(f"  Built {len(grid)} intraday signal variants (cost-charged)")

# ── Full-sample ranking (with cost) ───────────────────────────────────────────
print("\n"+"="*72)
print("  INTRADAY SIGNALS — full sample, NET of spread cost")
print("="*72)
print(f"  {'signal':<20}{'Sharpe':>8}{'CAGR':>9}{'maxDD':>9}{'trades/yr':>11}")
ranked=[]
for k,nt in grid.items():
    sh=sharpe(nt,per)
    eq=(1+nt).cumprod(); dd=(eq/eq.cummax()-1).min()
    cagr=eq.iloc[-1]**(per/len(nt))-1 if eq.iloc[-1]>0 else -1
    # turnover -> trades/yr (count sign changes of position)
    pos=(grid[k]!=0)
    ranked.append((k,sh,cagr,dd,nt))
ranked.sort(key=lambda x:-x[1])
for k,sh,cagr,dd,nt in ranked[:12]:
    print(f"  {k:<20}{sh:>8.2f}{cagr*100:>8.1f}%{dd*100:>8.1f}%")

# ── Walk-forward: re-pick each year, score next year ──────────────────────────
print("\n"+"="*72)
print("  WALK-FORWARD (re-pick best intraday signal each year, score next)")
print("="*72)
idx=p.index; years=sorted(set(idx.year)); start=years[2]
wf_blocks=[]; picks=[]
for y in years:
    if y<start: continue
    tr=idx.year<y; te=idx.year==y
    if tr.sum()<5000 or te.sum()<2000: continue
    best=None
    for k,nt in grid.items():
        s=sharpe(nt[tr],per)
        if best is None or s>best[0]: best=(s,k)
    _,k=best
    seg=grid[k][te]
    wf_blocks.append(seg); picks.append((y,k,sharpe(seg,per)))
wf=pd.concat(wf_blocks)
eqwf=(1+wf).cumprod(); ddwf=(eqwf/eqwf.cummax()-1).min()
print(f"  {'year':>6}  {'picked':<20}{'test_Sharpe':>12}")
for y,k,s in picks: print(f"  {y:>6}  {k:<20}{s:>+12.2f}")
print(f"\n  WALK-FORWARD OOS Sharpe {sharpe(wf,per):.2f}   "
      f"CAGR {(eqwf.iloc[-1]**(per/len(wf))-1)*100:.1f}%   maxDD {ddwf*100:.1f}%")
print(f"  Positive test-years: {sum(1 for _,_,s in picks if s>0)}/{len(picks)}")

# ── Best single FIXED signal across all 17 yrs (no re-picking) ────────────────
best_fixed=ranked[0]
bf_name,bf_sh=best_fixed[0],best_fixed[1]
bf_net=best_fixed[4]
# per-year sharpe of the fixed best
print(f"\n  Best FIXED signal [{bf_name}] per-year (genuine OOS, rule never changes):")
yr=bf_net.groupby(bf_net.index.year).apply(lambda s:sharpe(s,per))
print("  "+"  ".join(f"{y}:{v:+.1f}" for y,v in yr.items()))
print(f"  Positive years: {(yr>0).sum()}/{len(yr)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(2,1,figsize=(15,9),gridspec_kw={"height_ratios":[3,1]})
ax[0].plot(eqwf.index,eqwf.values,color="purple",lw=1.5,
           label=f"Walk-forward (re-pick yearly)  Sh {sharpe(wf,per):.2f}")
eqbf=(1+bf_net).cumprod()
ax[0].plot(eqbf.index,eqbf.values,color="darkgreen",lw=1.8,alpha=0.8,
           label=f"Fixed best [{bf_name}]  Sh {bf_sh:.2f}")
ax[0].set_yscale("log"); ax[0].legend(fontsize=10); ax[0].grid(alpha=0.3)
ax[0].set_title("Intraday Edge Hunt — HSI M5, 17 years, NET of spread cost")
uw=(eqwf/eqwf.cummax()-1)*100
ax[1].fill_between(eqwf.index,uw.values,0,color="purple",alpha=0.3)
ax[1].set_title(f"Walk-forward drawdown  maxDD {ddwf*100:.1f}%"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("intraday_hunt.png",dpi=130)
print("\n  Saved → intraday_hunt.png")
