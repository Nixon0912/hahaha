"""
Honest ML experiment: can a model predict breakout-setup quality and improve
expectancy out-of-sample?

Design choices that fight overfitting:
  - POOL all trustworthy assets (≈4-5k setups) so the model learns general
    setup quality, not per-asset noise (too little data per asset alone).
  - TIME-ORDERED walk-forward: 5 expanding folds; always predict the future
    from the past. Accumulate strictly out-of-sample predictions.
  - Target = trade R (regression). Take a trade only if predicted R > 0.
  - Compare OOS: all-trades expR vs ML-selected expR. Also report the
    TRAIN-set lift to show the over-fit gap explicitly.
"""
import warnings; warnings.filterwarnings("ignore")
import os, glob
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SETUPS=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","setups")
EXCLUDE={"HSIHKD","JPN225","US30","NGCUSD"}   # tick-uncalibrated
FEATURES=["trend","stack","atr_pct","atr_exp","atr_rank","range_pct",
          "dir_with_trend","hour","dow","dir"]
RISK=0.005

# Load + pool
frames=[]
for f in sorted(glob.glob(os.path.join(SETUPS,"*.csv"))):
    sym=os.path.basename(f)[:-4]
    if sym in EXCLUDE: continue
    d=pd.read_csv(f, parse_dates=["date"]); d["sym"]=sym
    frames.append(d)
df=pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
print(f"Pooled setups: {len(df)} across {df['sym'].nunique()} assets "
      f"({df['date'].min().date()} → {df['date'].max().date()})\n")

X=df[FEATURES].values; y=df["R"].values; dates=df["date"].values

# 5 expanding walk-forward folds (predict future from past)
n=len(df); folds=5
bounds=[int(n*k/(folds+1)) for k in range(1,folds+2)]  # train grows, test = next block
oos_pred=np.full(n, np.nan)
for k in range(folds):
    tr_end=bounds[k]; te_end=bounds[k+1]
    if tr_end<200: continue
    Xtr,ytr=X[:tr_end],y[:tr_end]
    Xte=X[tr_end:te_end]
    m=HistGradientBoostingRegressor(max_depth=3, max_iter=200,
        learning_rate=0.05, min_samples_leaf=40, l2_regularization=1.0)
    m.fit(Xtr,ytr)
    oos_pred[tr_end:te_end]=m.predict(Xte)

mask=~np.isnan(oos_pred)
oos=df[mask].copy(); oos["pred"]=oos_pred[mask]
print(f"Out-of-sample evaluation on {len(oos)} setups (folds 2-5):\n")

def stats(r):
    r=np.asarray(r)
    if len(r)==0: return (0,0,0,0)
    pf=(r[r>0].sum()/-r[r<0].sum()) if (r<0).any() else 99
    return (len(r), r.mean(), (r>0).mean()*100, pf)

# Baseline: take all OOS trades
n0,e0,w0,pf0=stats(oos["R"].values)
print(f"  ALL OOS trades        : n={n0}  expR={e0:+.3f}  WR={w0:.1f}%  PF={pf0:.2f}  "
      f"net={oos['R'].sum()*RISK*100:+.1f}%")

# ML-selected: predicted R above various thresholds
for thr in [0.0, 0.05, 0.10, 0.20]:
    sel=oos[oos["pred"]>thr]
    if len(sel)<20:
        print(f"  pred>{thr:<4}             : (only {len(sel)} trades)")
        continue
    nn,ee,ww,pff=stats(sel["R"].values)
    keep=len(sel)/len(oos)*100
    print(f"  ML pred R > {thr:<4}        : n={nn}  expR={ee:+.3f}  WR={ww:.1f}%  "
          f"PF={pff:.2f}  net={sel['R'].sum()*RISK*100:+.1f}%  (kept {keep:.0f}%)")

# Per-asset OOS lift from the pooled model (pred>0)
print(f"\n  Per-asset OOS expR: all vs ML(pred>0):")
print(f"  {'Symbol':<8}{'n_all':>6}{'expR_all':>10}{'n_ML':>6}{'expR_ML':>10}  Lift")
lift_pos=0; total=0
for sym,g in oos.groupby("sym"):
    sel=g[g["pred"]>0]
    if len(g)<15: continue
    total+=1
    ea=g["R"].mean(); em=sel["R"].mean() if len(sel)>=8 else float("nan")
    lift = (em-ea) if not np.isnan(em) else float("nan")
    if not np.isnan(lift) and lift>0 and em>0: lift_pos+=1
    fl="✅" if (not np.isnan(em) and em>0 and lift>0) else ("~" if (not np.isnan(em) and lift>0) else "")
    print(f"  {sym:<8}{len(g):>6}{ea:>+10.3f}{len(sel):>6}"
          f"{em:>+10.3f}  {fl}")
print(f"\n  Assets where ML improves AND turns positive OOS: {lift_pos}/{total}")
print(f"\n  (Compare OOS expR to a naive +EV bar of 0.000; honest walk-forward.)")
