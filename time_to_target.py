"""
Time-to-+8% Monte Carlo across lot sizes.
Trade frequency from real data: 78 trades over the active span (excl. Rocket Bull).
Reports trades-to-target and calendar months, plus bust risk.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from stress_test import df_full, arb_r, ind, run, PERIODS, BALANCE

# worst-case-friction pool at 0.10 lot
pool=[]
spans_days=0
for lbl,st,en in PERIODS:
    r=run(df_full,arb_r,ind,st,en,slip=2,spread_x=1.5,comm=0.70)
    if r is not None:
        pool.extend(r["log"]["pnl"].values)
    spans_days+=(np.datetime64(en)-np.datetime64(st)).astype(int)
pool=np.array(pool)
n=len(pool)
active_months=spans_days/30.0
tr_per_month=n/active_months
print(f"Pool: {n} trades over {active_months:.1f} active months "
      f"= {tr_per_month:.1f} trades/month (~{tr_per_month/4.3:.1f}/week)\n")

rng=np.random.default_rng(11)
print(f"{'Lot':>6} {'bust%':>6} {'--- trades to +8% ---':>26} {'--- months to +8% ---':>24}")
print(f"{'':>6} {'':>6} {'median':>9}{'p50-p90':>17} {'median':>9}{'p90':>8}{'worst':>8}")
for lot in [0.10,0.07,0.05,0.04,0.03]:
    pnl=pool*(lot/0.10)
    trades_needed=[]; busts=0
    for _ in range(10000):
        perm=rng.permutation(pnl)
        bal=BALANCE;pk=BALANCE;done=False
        for k,x in enumerate(perm,1):
            bal+=x;pk=max(pk,bal)
            if pk-bal>=BALANCE*0.10:busts+=1;done=True;break
            if bal>=BALANCE*1.08:trades_needed.append(k);done=True;break
        # if neither within pool, needs more than n trades — extrapolate
        if not done:
            # average remaining rate
            per_tr=pnl.mean()
            remain=(BALANCE*1.08-bal)/per_tr if per_tr>0 else 9999
            trades_needed.append(n+remain)
    tn=np.array(trades_needed)
    med=np.median(tn); p90=np.percentile(tn,90)
    print(f"{lot:>6.2f} {busts/10000*100:>5.1f}% "
          f"{med:>9.0f}{('  ('+str(int(med))+'-'+str(int(p90))+')'):>17} "
          f"{med/tr_per_month:>8.1f} {p90/tr_per_month:>7.1f} {tn.max()/tr_per_month:>7.1f}")

print(f"\n  Trade frequency assumes ~{tr_per_month:.1f} trades/month (ARB, Mon-Thu, ~1 setup/day).")
print(f"  'months to +8%' = trades-to-target / trade-rate. No time limit on The5ers,")
print(f"  so larger months just means more patience — only bust% is a real failure.")
