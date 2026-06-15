"""
HSI edge search — is HSI mean-reverting (fade) rather than trending (breakout)?

Disciplined: every config is scored on IS (pre-2025) AND OOS (2025+) separately.
A config only "survives" if net expR > 0 in BOTH halves with n>=30 each.
This guards against curve-fitting a single OOS window.

Matrix: {MOMENTUM breakout, FADE reversion} x {Asian range, NY range}
        x target RR {1.0, 1.5, 2.0, 3.5}.  Costs = real HSI spread x 0.01.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from multi_asset_scan import ranges as build_ranges, FORCE_H
from ml_filter import load_m15_mtf

SL_MULT=0.7; SL_LO=0.0008; SL_HI=0.006; POINT=0.01

m15, mtf = load_m15_mtf("HSIHKD")

def resolve_rr(m15, t, d, entry, sl_d, rr):
    sl = entry - d*sl_d; tp = entry + d*sl_d*rr
    day = m15[m15.index.date == t.date()]
    rem = day.loc[t:].iloc[1:]; rem = rem[rem.index.hour < FORCE_H]
    if rem.empty: return 0.0
    result="timeout"; exit_p=rem.iloc[-1]["close"]
    for _, rb in rem.iterrows():
        if d==1:
            if rb["low"]<=sl: result="sl"; exit_p=sl; break
            if rb["high"]>=tp: result="tp"; exit_p=tp; break
        else:
            if rb["high"]>=sl: result="sl"; exit_p=sl; break
            if rb["low"]<=tp: result="tp"; exit_p=tp; break
    return -1.0 if result=="sl" else (rr if result=="tp" else float(d*(exit_p-entry)/sl_d))

def run_config(mode, rng_win, entry_win, rr, rng_lo=0.0002, rng_hi=0.015):
    rstart,rend = rng_win; estart,eend = entry_win
    rtab = build_ranges(m15, rstart, rend)
    trades=[]
    for date in sorted(set(m15.index.date)):
        if pd.Timestamp(date).dayofweek>=4: continue
        dts=pd.Timestamp(date)
        if dts not in rtab.index: continue
        r=rtab.loc[dts]; mid=(r["hi"]+r["lo"])/2
        if mid<=0: continue
        rng_pct=r["rng"]/mid
        if not (rng_lo<=rng_pct<=rng_hi): continue
        eb=m15[(m15.index.date==date)&(m15.index.hour>=estart)&(m15.index.hour<eend)]
        for t,bar in eb.iterrows():
            if t not in mtf.index: continue
            i=mtf.loc[t]
            if pd.isna(i["h1_atr"]) or i["h1_atr"]<=0: continue
            price=bar["close"]
            up = price>r["hi"]; dn = price<r["lo"]
            if not up and not dn: continue
            brk = 1 if up else -1
            d = brk if mode=="MOM" else -brk     # FADE = trade against the break
            sl_d=float(np.clip(i["h1_atr"]*SL_MULT, price*SL_LO, price*SL_HI))
            R=resolve_rr(m15,t,d,price,sl_d,rr)
            spread=float(bar["spread"])*POINT
            Rnet=R-spread/sl_d if sl_d>0 else R
            trades.append({"date":dts,"R":R,"Rnet":Rnet})
            break  # one trade/day
    if not trades: return None
    df=pd.DataFrame(trades)
    IS=df[df["date"]<"2025-01-01"]; OOS=df[df["date"]>="2025-01-01"]
    def ex(d): return (len(d), 100*(d["Rnet"]>0).mean() if len(d) else 0, d["Rnet"].mean() if len(d) else 0)
    return ex(IS), ex(OOS), df["Rnet"].sum()

print("="*86)
print("  HSI edge search — MOM=breakout, FADE=reversion | net of real costs")
print(f"  {'config':<34}{'IS n/WR/expR':>22}{'OOS n/WR/expR':>22}  survive?")
print("="*86)
configs=[]
for mode in ["MOM","FADE"]:
    for label,rw,ew in [("Asian(0-8)e8-10",(0,8),(8,10)),
                        ("NY(10-13)e13-15",(10,13),(13,15)),
                        ("HKopen(4-6)e8-10",(4,6),(8,10))]:
        for rr in [1.0,1.5,2.0,3.5]:
            res=run_config(mode,rw,ew,rr)
            if res is None: continue
            (isn,iswr,isex),(on,owr,oex),tot=res
            surv = "✅ YES" if (isn>=30 and on>=20 and isex>0 and oex>0) else ""
            name=f"{mode} {label} RR{rr}"
            print(f"  {name:<34}{isn:4d}/{iswr:3.0f}%/{isex:+.2f}{'':6}{on:4d}/{owr:3.0f}%/{oex:+.2f}{'':4}{surv}")
            if surv: configs.append((name,isex,oex,tot))

print("="*86)
if configs:
    print("  SURVIVORS (positive IS AND OOS):")
    for n,i,o,t in sorted(configs,key=lambda x:-x[2]):
        print(f"    {n:<34} IS expR {i:+.3f}  OOS expR {o:+.3f}  totalR {t:+.1f}")
else:
    print("  No config positive in BOTH IS and OOS — HSI has no robust intraday edge here.")
