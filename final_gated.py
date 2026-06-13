"""
Consolidated regime-gated OOS validation + Monte Carlo (auditor reproducibility).

Reproduces SYSTEM_REPORT.md v3.0 headline numbers:
  - Gated OOS (model trained pre-2025): n=43, WR 53.5%, expR +0.885, totalR +38.0
  - Wilson + bootstrap CIs, per-regime split
  - Weekly block-bootstrap Monte Carlo (pass/bust/median per risk level)

Run from repo root:  python final_gated.py
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from multi_asset_scan import extract_arb, extract_nyo, extract_mom
from ml_filter import load_m15_mtf, extract_features, FEAT_COLS
from ea.regime import classify_regime
from ea.config import REGIME_SYMS

SL_MULT=0.7; SL_LO=0.0008; SL_HI=0.006
POINT = {"ASXAUD":0.1,"DAX40":0.01,"ESXEUR":0.01,"SP500":0.01,"UK100":0.01,
         "USDCAD":0.00001,"USDJPY":0.001,"XAGUSD":0.001}
COMM  = {"USDCAD":0.00008,"USDJPY":0.00008,"XAGUSD":0.00002}
ALL9 = [("ASXAUD","NYO"),("SP500","MOM"),("USDCAD","MOM"),("USDJPY","NYO"),
        ("XAGUSD","ARB"),("DAX40","ARB"),("ESXEUR","NYO"),("UK100","ARB"),("USDJPY","ARB")]

# regime per date (cross-asset, lagged)
m15reg = {s: load_m15_mtf(s)[0] for s in REGIME_SYMS}
def reg_for(d):
    return classify_regime(m15reg, pd.Timestamp(d))["chop"]

recs_all=[]
for sym,arch in ALL9:
    m15,mtf = load_m15_mtf(sym)
    fn={"ARB":extract_arb,"NYO":extract_nyo,"MOM":extract_mom}[arch]
    recs = extract_features(m15,mtf,fn(m15,mtf),sym,arch)
    pt=POINT[sym]; cm=COMM.get(sym,0.0)
    for r in recs:
        et=r["entry_t"]; price=float(m15.loc[et,"close"])
        spread=float(m15.loc[et,"spread"])*pt
        h1=mtf.loc[et,"h1_atr"]; h1=float(h1) if not pd.isna(h1) else 0.0
        sl=float(np.clip(h1*SL_MULT,price*SL_LO,price*SL_HI))
        r["R_net"]=r["R"]-(spread+cm*price)/sl if sl>0 else r["R"]
        r["label"]=int(r["R_net"]>0)
    recs_all.extend(recs)

df=pd.DataFrame(recs_all).sort_values("entry_t").reset_index(drop=True)
df["date"]=pd.to_datetime(df["date"])
df["rolling_wr"]=df["label"].shift(1).rolling(10,min_periods=3).mean().fillna(0.5)
FE=FEAT_COLS+["rolling_wr"]

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
tr_=df[df["date"]<"2025-01-01"]; y=tr_["label"].values
spos=max((y==0).sum()/(y==1).sum(),0.1)
xgb=XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.04,subsample=0.8,
                  colsample_bytree=0.7,scale_pos_weight=spos,eval_metric="logloss",
                  random_state=42,verbosity=0)
m=CalibratedClassifierCV(xgb,cv=5,method="isotonic"); m.fit(tr_[FE].fillna(0).values,y)
df["prob"]=m.predict_proba(df[FE].fillna(0).values)[:,1]
df["filt"]=df["prob"]>=0.35

# unique date chop map
udates = sorted(df[df["filt"]]["date"].dt.date.unique())
chopmap = {d: reg_for(d) for d in udates}
df["chop"]=df["date"].dt.date.map(chopmap).fillna(False)

oos=df[(df["date"]>="2025-01-01")&df["filt"]].copy()
# Gate: drop MOM on chop days
gated=oos[~((oos["chop"])&(oos["arch"]=="MOM"))].copy().sort_values("entry_t")

def stats(d,name):
    wr=100*(d["R_net"]>0).mean(); ex=d["R_net"].mean()
    print(f"{name:18s}: n={len(d):3d}  WR={wr:.1f}%  expR={ex:+.3f}  totalR={d['R_net'].sum():+.1f}")
    return wr,ex
print("=== OOS 2025-01 → 2026-06 (model trained pre-2025) ===")
stats(oos,"Ungated (9 stream)")
stats(gated,"Gated (regime)")

# Wilson CI for gated WR
from math import sqrt
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h)*100,(c+h)*100
k=(gated["R_net"]>0).sum(); n=len(gated)
lo,hi=wilson(k,n); print(f"  Gated WR Wilson 95% CI: [{lo:.0f}%, {hi:.0f}%]")
# bootstrap expR CI
rs=gated["R_net"].values
boot=[np.random.choice(rs,len(rs),replace=True).mean() for _ in range(5000)]
print(f"  Gated expR bootstrap 95% CI: [{np.percentile(boot,2.5):+.2f}, {np.percentile(boot,97.5):+.2f}]")

# trades/month
months=(gated["date"].max()-gated["date"].min()).days/30.4
print(f"  Trade frequency: {len(gated)/months:.2f}/month over {months:.1f} months")

# ===== Monte Carlo on gated OOS pool =====
def mc(pool, risk, n_paths=5000, init=10000, target=10800, floor=9000, horizon=200):
    rng=np.random.default_rng(42)
    passes=busts=0; times=[]
    # weekly block bootstrap
    g=gated.copy(); g["wk"]=g["date"].dt.isocalendar().week.astype(str)+g["date"].dt.year.astype(str)
    blocks=[v["R_net"].values for _,v in g.groupby("wk")]
    for _ in range(n_paths):
        bal=init; done=False; t=0
        while t<horizon and not done:
            blk=blocks[rng.integers(len(blocks))]
            for R in blk:
                bal+=bal*risk*R; t+=1
                if bal>=target: passes+=1; times.append(t); done=True; break
                if bal<=floor: busts+=1; done=True; break
        if not done: pass
    return passes/n_paths*100, busts/n_paths*100, (np.median(times) if times else None)

print("\n=== Monte Carlo (weekly block bootstrap, gated pool, 5000 paths) ===")
for risk in [0.0075,0.01,0.0125,0.015]:
    p,b,med=mc(gated,risk)
    # convert median trades to months at freq
    fpm=len(gated)/months
    medmo=med/fpm if med else None
    print(f"  risk {risk*100:.2f}%: pass={p:.1f}%  bust={b:.1f}%  median≈{medmo:.1f}mo" if med else
          f"  risk {risk*100:.2f}%: pass={p:.1f}%  bust={b:.1f}%")

# chop vs trend in gated set
print("\n=== Gated, split by regime ===")
for c in [False,True]:
    s=gated[gated["chop"]==c]; tag="CHOP" if c else "TREND"
    if len(s): print(f"  {tag}: n={len(s)}  WR={100*(s['R_net']>0).mean():.0f}%  expR={s['R_net'].mean():+.3f}")
