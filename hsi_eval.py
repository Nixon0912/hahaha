"""
HSI evaluation — does HSIHKD add edge to the validated system?

HSI was in multi_asset_scan.EXCLUDE only because tick/spread was miscalibrated
in earlier research. With the real broker spec now known:
  digits=2, point=0.01, tick_value=0.001, NO commission (index CFD),
  swap irrelevant (force-close before 21:00), spread floating ~6-7 index pts.

This script:
  1. Runs all 3 archetypes on HSI M15 with REAL costs (spread from data x point)
  2. Reports full-period and OOS (2025-01+) raw vs net expR per archetype
  3. Trains the production model on the 9 streams + HSI pooled (pre-2025),
     applies to HSI OOS at P>=0.35, measures net edge
  4. Adds the best HSI archetype to the 9 -> re-runs gated OOS + Monte Carlo
     to see if the PORTFOLIO improves

Run:  python hsi_eval.py
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
         "USDCAD":0.00001,"USDJPY":0.001,"XAGUSD":0.001,"HSIHKD":0.01}
COMM  = {"USDCAD":0.00008,"USDJPY":0.00008,"XAGUSD":0.00002}  # HSI: none
ALL9 = [("ASXAUD","NYO"),("SP500","MOM"),("USDCAD","MOM"),("USDJPY","NYO"),
        ("XAGUSD","ARB"),("DAX40","ARB"),("ESXEUR","NYO"),("UK100","ARB"),("USDJPY","ARB")]
EXTRACT = {"ARB":extract_arb,"NYO":extract_nyo,"MOM":extract_mom}


def cost_net(recs, m15, sym):
    """Attach R_net (after spread+commission) and label to each trade record."""
    pt = POINT[sym]; cm = COMM.get(sym, 0.0)
    for r in recs:
        et = r["entry_t"]; price = float(m15.loc[et, "close"])
        spread = float(m15.loc[et, "spread"]) * pt
        # reconstruct SL distance the same way the extractor did
        from ml_filter import load_m15_mtf  # noqa
        sl = r.get("_sl")
        if sl is None:
            sl = price * 0.003
        r["R_net"] = r["R"] - (spread + cm*price)/sl if sl > 0 else r["R"]
        r["label"] = int(r["R_net"] > 0)
    return recs


def build_recs(sym, arch, m15, mtf):
    """Extract trades + features + net R for one stream."""
    trades = EXTRACT[arch](m15, mtf)
    # recompute SL per trade for cost (clip h1_atr*mult)
    for t in trades:
        et = t["entry_t"]
        price = float(m15.loc[et, "close"])
        h1 = mtf.loc[et, "h1_atr"]; h1 = float(h1) if not pd.isna(h1) else 0.0
        t["_sl"] = float(np.clip(h1*SL_MULT, price*SL_LO, price*SL_HI))
    recs = extract_features(m15, mtf, trades, sym, arch)
    # carry _sl into feature recs (extract_features copies **t)
    return recs


def stats(d, name):
    if len(d) == 0:
        print(f"  {name:14s}: n=0"); return
    wr = 100*(np.array([x['R_net'] for x in d])>0).mean()
    rawR = np.mean([x['R'] for x in d]); netR = np.mean([x['R_net'] for x in d])
    tot = np.sum([x['R_net'] for x in d])
    print(f"  {name:14s}: n={len(d):4d}  WR={wr:4.1f}%  rawExpR={rawR:+.3f}  "
          f"netExpR={netR:+.3f}  totalR={tot:+6.1f}")


print("="*72)
print("  HSI standalone — 3 archetypes, real costs (spread from data x 0.01)")
print("="*72)
m15h, mtfh = load_m15_mtf("HSIHKD")
hsi_by_arch = {}
for arch in ["ARB","NYO","MOM"]:
    recs = cost_net(build_recs("HSIHKD", arch, m15h, mtfh), m15h, "HSIHKD")
    for r in recs: r["date"] = pd.Timestamp(r["date"])
    hsi_by_arch[arch] = recs
    full = recs
    oos = [r for r in recs if r["date"] >= pd.Timestamp("2025-01-01")]
    print(f"\nHSI-{arch}:")
    stats(full, "full 2020-26")
    stats(oos,  "OOS 2025-26")

# ── ML filter test: train on 9 streams + HSI pooled (pre-2025) ──────────────
print("\n" + "="*72)
print("  HSI through the ML filter (model trained on 9 streams + HSI, pre-2025)")
print("="*72)
pool = []
for sym, arch in ALL9:
    m15, mtf = load_m15_mtf(sym)
    recs = cost_net(build_recs(sym, arch, m15, mtf), m15, sym)
    for r in recs: r["date"] = pd.Timestamp(r["date"])
    pool.extend(recs)
for arch in ["ARB","NYO","MOM"]:
    pool.extend(hsi_by_arch[arch])

df = pd.DataFrame(pool).sort_values("entry_t").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])
df["rolling_wr"] = df["label"].shift(1).rolling(10,min_periods=3).mean().fillna(0.5)
FE = FEAT_COLS + ["rolling_wr"]

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
tr = df[df["date"]<"2025-01-01"]; y = tr["label"].values
spos = max((y==0).sum()/(y==1).sum(), 0.1)
xgb = XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.04,subsample=0.8,
                    colsample_bytree=0.7,scale_pos_weight=spos,eval_metric="logloss",
                    random_state=42,verbosity=0)
mdl = CalibratedClassifierCV(xgb,cv=5,method="isotonic")
mdl.fit(tr[FE].fillna(0).values, y)
df["prob"] = mdl.predict_proba(df[FE].fillna(0).values)[:,1]
df["filt"] = df["prob"]>=0.35

hsi = df[(df["sym"]=="HSIHKD")&(df["date"]>="2025-01-01")&df["filt"]]
print(f"\nHSI OOS filtered (P>=35%): n={len(hsi)}")
for arch in ["ARB","NYO","MOM"]:
    s = hsi[hsi["arch"]==arch]
    if len(s):
        print(f"  HSI-{arch}: n={len(s)}  WR={100*(s['R_net']>0).mean():.0f}%  "
              f"netExpR={s['R_net'].mean():+.3f}  totalR={s['R_net'].sum():+.1f}")
if len(hsi):
    print(f"  HSI ALL: n={len(hsi)}  WR={100*(hsi['R_net']>0).mean():.0f}%  "
          f"netExpR={hsi['R_net'].mean():+.3f}  totalR={hsi['R_net'].sum():+.1f}")

df.to_pickle("/tmp/hsi_pool.pkl")
print("\n(pool saved to /tmp/hsi_pool.pkl for portfolio test)")
