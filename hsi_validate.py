"""
HSI alpha — phase 3: is the intraday-momentum edge ECONOMICALLY tradeable?

Candidate (from phase 2): enter at 09:00 in the direction of the
(session-open -> 09:00) move when |move|/dailyATR >= thresh; ATR stop; force-close.

This phase answers the questions that decide deployability, not just significance:
  - Year-by-year net (is it broad, or one lucky year?)
  - Long vs short split (is it just long-bias?)
  - Annualized Sharpe (OOS)
  - R-based Monte Carlo on The5ers ($10k->$10.8k, $9k bust): pass/bust per risk
"""
import sys, warnings, glob
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from multi_asset_scan import load_raw

m5 = load_raw(glob.glob("HSIHKD_M5_*.csv")[0]); m5 = m5[m5["close"]>0].copy()
COST_PTS = 7.0
T_HOUR   = 9
EXIT_H   = 20
K_STOP   = 1.5   # stop distance = K_STOP * dailyATR

d1 = m5.resample("1D").agg({"high":"max","low":"min","close":"last"}).dropna()
d1["pc"]=d1["close"].shift(1)
d1["tr"]=np.maximum(d1["high"]-d1["low"],np.maximum((d1["high"]-d1["pc"]).abs(),(d1["low"]-d1["pc"]).abs()))
d1["atr"]=d1["tr"].ewm(span=14,adjust=False).mean().shift(1)
atr_map=d1["atr"].to_dict()

def build(thresh):
    """Return per-trade records with R (risk = K_STOP*ATR), intraday stop+force-close."""
    recs=[]
    for date, day in m5.groupby(m5.index.date):
        day=day[day.index.hour<=EXIT_H+1]
        if len(day)<30: continue
        early=day[day.index.hour<T_HOUR]; rest=day[(day.index.hour>=T_HOUR)&(day.index.hour<=EXIT_H)]
        if len(early)<6 or len(rest)<6: continue
        atr=atr_map.get(pd.Timestamp(date),np.nan)
        if not atr or np.isnan(atr) or atr<=0: continue
        entry=early["close"].iloc[-1]; strength=(entry-early["close"].iloc[0])/atr
        if abs(strength)<thresh: continue
        d=1 if strength>0 else -1
        stop_d=K_STOP*atr
        sl=entry-d*stop_d
        # walk intraday bars; stop if hit, else exit at force-close
        exit_px=rest["close"].iloc[-1]; stopped=False
        for _,b in rest.iterrows():
            if d==1 and b["low"]<=sl: exit_px=sl; stopped=True; break
            if d==-1 and b["high"]>=sl: exit_px=sl; stopped=True; break
        gross_pts=d*(exit_px-entry)
        net_pts=gross_pts-COST_PTS
        R=net_pts/stop_d
        recs.append({"date":pd.Timestamp(date),"d":d,"R":R,"net_pts":net_pts,
                     "entry":entry,"stopped":stopped,"yr":pd.Timestamp(date).year})
    return pd.DataFrame(recs)

def block_mc(R, risk, n=5000, init=10000, target=10800, floor=9000, horizon=400, blk=5):
    rng=np.random.default_rng(7); R=np.asarray(R); P=B=0; times=[]
    nb=max(len(R)//blk,1)
    for _ in range(n):
        bal=init; t=0; done=False
        while t<horizon and not done:
            i=rng.integers(len(R)-blk) if len(R)>blk else 0
            for r in R[i:i+blk]:
                bal+=bal*risk*r; t+=1
                if bal>=target: P+=1; times.append(t); done=True; break
                if bal<=floor: B+=1; done=True; break
    return P/n*100, B/n*100, (np.median(times) if times else None)

print("="*78)
print(f"  HSI intraday momentum @T={T_HOUR}:00, ATR stop x{K_STOP}, force-close — validation")
print(f"  cost {COST_PTS}pts | IS=2018-21  OOS=2022-26")
print("="*78)

for thresh in [0.2, 0.3, 0.4]:
    df=build(thresh)
    IS=df[(df.yr>=2018)&(df.yr<=2021)]; OOS=df[(df.yr>=2022)&(df.yr<=2026)]
    if len(OOS)<60: continue
    def line(s,nm):
        wr=100*(s.R>0).mean(); ex=s.R.mean(); sh=ex/s.R.std()*np.sqrt(len(s)/(len(s)/ (s.date.dt.year.nunique() or 1)))
        return wr,ex,s.R.sum()
    print(f"\n  === thresh={thresh}  (IS n={len(IS)}, OOS n={len(OOS)}) ===")
    for nm,s in [("IS 2018-21",IS),("OOS 2022-26",OOS)]:
        wr=100*(s.R>0).mean(); ex=s.R.mean()
        tr_per_yr=len(s)/max(s.date.dt.year.nunique(),1)
        sharpe=ex/s.R.std()*np.sqrt(tr_per_yr) if s.R.std()>0 else 0
        print(f"    {nm}: n={len(s):4d}  WR={wr:4.1f}%  expR={ex:+.3f}  "
              f"totalR={s.R.sum():+6.1f}  ~{tr_per_yr:.0f} tr/yr  annSharpe≈{sharpe:+.2f}")
    # year by year OOS
    print("    OOS year-by-year expR:",
          "  ".join(f"{y}:{OOS[OOS.yr==y].R.mean():+.2f}(n{len(OOS[OOS.yr==y])})"
                    for y in sorted(OOS.yr.unique())))
    # long/short split OOS
    lo=OOS[OOS.d==1]; sh=OOS[OOS.d==-1]
    print(f"    OOS long: n={len(lo)} expR={lo.R.mean():+.3f} | short: n={len(sh)} expR={sh.R.mean():+.3f}")
    # bootstrap CI on OOS expR
    rs=OOS.R.values; boot=[np.random.choice(rs,len(rs),replace=True).mean() for _ in range(3000)]
    print(f"    OOS expR 95% CI: [{np.percentile(boot,2.5):+.3f}, {np.percentile(boot,97.5):+.3f}]")
    # MC challenge sim on OOS pool
    print("    MC (OOS pool):", end="")
    for risk in [0.005,0.0075,0.01,0.0125]:
        p,b,med=block_mc(OOS.R.values,risk)
        print(f"  {risk*100:.2f}%→pass{p:.0f}%/bust{b:.0f}%", end="")
    print()
