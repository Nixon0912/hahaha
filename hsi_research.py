"""
HSI alpha research — phase 1: EDA on engineered timeframes from M5.

Pure statistics (no ML). Goal: locate systematic, era-stable structure before
designing any strategy. Every metric is split into two eras
(2009-2017 vs 2018-2026); only structure that holds in BOTH is worth trading.

Costs: HSI spread ~6-7 index pts round trip (we apply at strategy stage).
"""
import sys, warnings, glob
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from multi_asset_scan import load_raw

m5 = load_raw(glob.glob("HSIHKD_M5_*.csv")[0])
m5 = m5[m5["close"] > 0].copy()
m5["ret"] = m5["close"].pct_change()
m5["era"] = np.where(m5.index.year <= 2017, "2009-17", "2018-26")

def tf(freq):
    o = m5.resample(freq, label="left", closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    return o

print("="*78)
print("  1) Intraday drift — mean 5m return by HOUR (bps), per era")
print("="*78)
g = m5.groupby([m5.index.hour, "era"])["ret"].agg(["mean","count"]).unstack()
print(f"  {'hr':>3} {'2009-17 bps':>12} {'2018-26 bps':>12} {'n09':>7} {'n18':>7} {'stable?':>8}")
for h in sorted(m5.index.hour.unique()):
    try:
        a = g.loc[h, ("mean","2009-17")]*1e4; b = g.loc[h, ("mean","2018-26")]*1e4
        na = g.loc[h, ("count","2009-17")]; nb = g.loc[h, ("count","2018-26")]
        stable = "✅" if (a*b > 0 and abs(a) > 0.3 and abs(b) > 0.3) else ""
        print(f"  {h:>3} {a:>12.2f} {b:>12.2f} {na:>7.0f} {nb:>7.0f} {stable:>8}")
    except Exception:
        pass

print("\n" + "="*78)
print("  2) Day-of-week — mean DAILY return (bps), per era")
print("="*78)
d1 = tf("1D"); d1["ret"] = d1["close"].pct_change()
d1["dow"] = d1.index.dayofweek; d1["era"] = np.where(d1.index.year<=2017,"09-17","18-26")
gd = d1.groupby(["dow","era"])["ret"].mean().unstack()*1e4
print(gd.round(1))

print("\n" + "="*78)
print("  3) First-session-hour momentum: does early direction predict rest-of-day?")
print("="*78)
# Build a 'session' = one trading date. Compare sign of return in an early
# window vs return over the remainder (close-able intraday, before 21:00).
def session_momentum(early_end_h, label):
    rows=[]
    for date, day in m5.groupby(m5.index.date):
        day = day[day.index.hour < 21]
        if len(day) < 20: continue
        early = day[day.index.hour < early_end_h]
        rest  = day[day.index.hour >= early_end_h]
        if len(early) < 3 or len(rest) < 3: continue
        er = early["close"].iloc[-1]/early["close"].iloc[0]-1
        rr = rest["close"].iloc[-1]/rest["close"].iloc[0]-1
        rows.append({"date":pd.Timestamp(date),"er":er,"rr":rr})
    df=pd.DataFrame(rows)
    df["era"]=np.where(df["date"].dt.year<=2017,"09-17","18-26")
    out=[]
    for era,sub in df.groupby("era"):
        # if early up, go long rest; if early down, go short rest
        sig=np.sign(sub["er"]); pnl=sig*sub["rr"]
        corr=sub["er"].corr(sub["rr"])
        out.append((era,len(sub),pnl.mean()*1e4,(pnl>0).mean()*100,corr))
    print(f"  early<{early_end_h}h ({label}):")
    for era,n,bps,wr,corr in out:
        print(f"    {era}: n={n:5d}  follow-thru pnl={bps:+7.2f}bps  WR={wr:4.1f}%  corr={corr:+.3f}")

for h in [5,6,9,10,13]:
    session_momentum(h, f"enter at {h}:00, exit ~20:55")

print("\n" + "="*78)
print("  4) Overnight gap behaviour: gap-and-go vs gap-fill")
print("="*78)
# session open vs prior session close; does the day continue or fade the gap?
opens=[]
prev_close=None; prev_date=None
for date, day in m5.groupby(m5.index.date):
    day=day[day.index.hour<21]
    if len(day)<20: continue
    o=day["close"].iloc[0]; c=day["close"].iloc[-1]
    if prev_close is not None:
        gap=o/prev_close-1; intraday=c/o-1
        opens.append({"date":pd.Timestamp(date),"gap":gap,"intra":intraday})
    prev_close=c
od=pd.DataFrame(opens); od["era"]=np.where(od["date"].dt.year<=2017,"09-17","18-26")
for era,sub in od.groupby("era"):
    # gap-fill: gap up -> short; correlation of gap vs intraday
    corr=sub["gap"].corr(sub["intra"])
    fade=(-np.sign(sub["gap"])*sub["intra"]).mean()*1e4
    go=(np.sign(sub["gap"])*sub["intra"]).mean()*1e4
    print(f"  {era}: n={len(sub):5d}  corr(gap,intra)={corr:+.3f}  "
          f"FADE pnl={fade:+.2f}bps  GO pnl={go:+.2f}bps")
