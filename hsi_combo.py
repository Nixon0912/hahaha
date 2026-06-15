"""
HSI combined system — mean-reversion + trend-following, vol-targeted.

Findings so far:
  intraday = momentum (fading loses) ; daily swing = weak mean-reversion (31% DD)
Classic fix for a STEADY curve: blend MR (wins in ranges) with trend-following
(wins in the crashes that wreck MR). They are regime-anticorrelated, so the
blend's drawdown is far smaller than either alone. Then vol-target to smooth.

Daily bars from 17yr M5. Cost = half-spread/turnover. No-lookahead. IS70/OOS30.
Outputs equity curves for: MR, Trend, and the vol-targeted blend.
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
d1["ret"] = d1["close"].pct_change().fillna(0)
HALF = 3.5
cost_unit = HALF/d1["close"]
SPLIT = int(len(d1)*0.70)
ann = 252

def perf(net, name):
    net=net.fillna(0); eq=(1+net).cumprod()
    sh=net.mean()/net.std()*np.sqrt(ann) if net.std()>0 else 0
    iss=net.iloc[:SPLIT]; oos=net.iloc[SPLIT:]
    shi=iss.mean()/iss.std()*np.sqrt(ann) if iss.std()>0 else 0
    sho=oos.mean()/oos.std()*np.sqrt(ann) if oos.std()>0 else 0
    dd=(eq/eq.cummax()-1).min()
    cagr=eq.iloc[-1]**(ann/len(net))-1 if eq.iloc[-1]>0 else -1
    return dict(name=name,eq=eq,net=net,sh=sh,shi=shi,sho=sho,dd=dd,cagr=cagr)

def netfrom(pos, lbl):
    pos=pos.shift(1).fillna(0)
    turn=pos.diff().abs().fillna(pos.abs())
    return perf(pos*d1["ret"]-turn*cost_unit, lbl)

# ── MR component: z-score n30 fade (held until cross 0) ─────────────────────
n=30; ma=d1["close"].rolling(n).mean(); sd=d1["close"].rolling(n).std()
z=(d1["close"]-ma)/sd; zv=z.values; mr=np.zeros(len(d1)); cur=0.0
for i in range(len(d1)):
    if np.isnan(zv[i]): mr[i]=cur; continue
    if cur==0:
        if zv[i]>=2: cur=-1
        elif zv[i]<=-2: cur=1
    else:
        if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
    mr[i]=cur
mr=pd.Series(mr,index=d1.index)

# ── Trend component: time-series momentum (sign of past L-day return) ────────
trend_results=[]
for L in [50,100,150,200]:
    sig=np.sign(d1["close"].pct_change(L))
    trend_results.append((L, netfrom(sig, f"TSMOM {L}d")))
print("="*70)
print("  Trend-following (TSMOM) on HSI daily")
print(f"  {'lookback':<12}{'CAGR':>8}{'Sharpe':>8}{'IS':>7}{'OOS':>7}{'maxDD':>8}")
for L,r in trend_results:
    print(f"  {L}d{'':<8}{r['cagr']*100:>7.1f}%{r['sh']:>8.2f}{r['shi']:>7.2f}{r['sho']:>7.2f}{r['dd']*100:>7.1f}%")
bestL,bestTrend = max(trend_results, key=lambda x: min(x[1]["shi"],x[1]["sho"]))
trend_sig = np.sign(d1["close"].pct_change(bestL))

mrP = netfrom(mr, "MeanRev z30")
print(f"\n  MeanRev z30 : CAGR {mrP['cagr']*100:.1f}%  Sharpe {mrP['sh']:.2f}  "
      f"IS {mrP['shi']:.2f} OOS {mrP['sho']:.2f}  maxDD {mrP['dd']*100:.1f}%")
print(f"  Best trend  : {bestL}d  Sharpe {bestTrend['sh']:.2f}  maxDD {bestTrend['dd']*100:.1f}%")
print(f"  Correlation(MR, Trend) daily net: {mrP['net'].corr(bestTrend['net']):+.3f}")

# ── Blend: equal-risk MR + Trend, then vol-target to 10% annual ─────────────
realvol = d1["ret"].rolling(20).std()*np.sqrt(ann)
def voltarget(pos, target=0.10, cap=3.0):
    scale=(target/realvol.replace(0,np.nan)).clip(upper=cap).fillna(0)
    return pos*scale

raw_blend = 0.5*mr + 0.5*trend_sig.fillna(0)
blend_vt = voltarget(raw_blend)
mr_vt    = voltarget(mr)
blendP = netfrom(blend_vt, "Blend (MR+Trend) volTgt")
mrvtP  = netfrom(mr_vt,    "MR only volTgt")

print("\n" + "="*70)
print("  Vol-targeted (10% ann) results")
print(f"  {'system':<28}{'CAGR':>8}{'Sharpe':>8}{'IS':>7}{'OOS':>7}{'maxDD':>8}")
for r in [mrvtP, blendP]:
    print(f"  {r['name']:<28}{r['cagr']*100:>7.1f}%{r['sh']:>8.2f}{r['shi']:>7.2f}{r['sho']:>7.2f}{r['dd']*100:>7.1f}%")

# ── Plot ────────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(figsize=(12,6.5))
for r,c in [(mrP,"#1f77b4"),(bestTrend,"#ff7f0e"),(blendP,"#2ca02c")]:
    ax.plot(r["eq"].index, r["eq"].values, lw=1.3, label=f"{r['name']}  (Sh {r['sh']:.2f}, DD {r['dd']*100:.0f}%)", color=c)
ax.axvline(d1.index[SPLIT], color="red", ls="--", alpha=0.7, label="IS/OOS split")
ax.set_yscale("log")
ax.set_title("HSI: Mean-Reversion vs Trend vs Vol-Targeted Blend (daily, 2009-2026)")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("hsi_combo_equity.png", dpi=120)
print("\n  saved -> hsi_combo_equity.png")
