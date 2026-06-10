"""
The Monte Carlo showed ~43% bust at 0.1 lot — the edge is thin and DD sits
near the -10% wall. The5ers has NO time limit, so the cleanest lever to cut
bust risk is SMALLER POSITION SIZE: it shrinks DD faster than it shrinks the
path to +8% (you just take more trades to get there).

This re-runs the worst-case-friction trade pool at several lot sizes and
reports the Monte Carlo bust probability for each.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stress_test import df_full, arb_r, ind, run, PERIODS, BALANCE

# Build worst-case-friction trade pool ONCE (pnl values at 0.1 lot)
pool=[]
for lbl,st,en in PERIODS:
    r=run(df_full,arb_r,ind,st,en,slip=2,spread_x=1.5,comm=0.70)
    if r is not None: pool.extend(r["log"]["pnl"].values)
pool=np.array(pool)
print(f"Pool: {len(pool)} trades, total ${pool.sum():+.0f} at 0.10 lot, "
      f"win rate {(pool>0).mean()*100:.0f}%\n")

rng=np.random.default_rng(7)
print(f"{'Lot':>6} {'scale':>6} {'+8% first':>10} {'BUST':>7} {'slow':>7} "
      f"{'medDD%':>8} {'p95DD%':>8} {'worstDD%':>9}")
for lot in [0.10,0.07,0.05,0.04,0.03,0.02]:
    scale=lot/0.10
    pnl=pool*scale
    passes=fails=slow=0; dds=[]
    for _ in range(8000):
        perm=rng.permutation(pnl)
        bal=BALANCE;pk=BALANCE;mdd=0;hit=None
        for x in perm:
            bal+=x;pk=max(pk,bal);mdd=min(mdd,bal-pk)
            if hit is None and bal>=BALANCE*1.08:hit="PASS";break
            if bal<=pk-BALANCE*0.10:hit="FAIL";break
        # continue accumulating DD only matters until terminal; fine
        dds.append(mdd)
        if hit=="PASS":passes+=1
        elif hit=="FAIL":fails+=1
        else:slow+=1
    dds=np.array(dds)
    print(f"{lot:>6.2f} {scale:>5.0%} {passes/8000*100:>9.1f}% "
          f"{fails/8000*100:>6.1f}% {slow/8000*100:>6.1f}% "
          f"{np.median(dds)/BALANCE*100:>7.2f}% "
          f"{np.percentile(dds,5)/BALANCE*100:>7.2f}% "
          f"{dds.min()/BALANCE*100:>8.2f}%")

print("\nNote: 'slow' = reached neither +8% nor -10% within the 78-trade pool.")
print("With no time limit, 'slow' outcomes simply keep trading — they are NOT")
print("failures. Only BUST is a real failure. Lower lot = lower BUST.")
