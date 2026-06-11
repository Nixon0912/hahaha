"""
FINAL ALL-COSTS-IN VALIDATION
  - Spread: actual <SPREAD> column at the entry bar (points -> price units),
            charged once per trade (cross the spread on one side).
  - Commission: USDJPY/USDCAD $4/lot/side = $8 round-trip per 100k notional
                = 0.008% of notional. XAGUSD 0.001%/side = 0.002% round-trip.
                Index CFDs: zero (spread-only).
  - Swap: zero — all trades force-close 21:00 server, rollover ~00:00.
  R_net = R_gross - (spread_px + commission_px) / sl_dist
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from multi_asset_scan import extract_arb, extract_nyo, extract_mom, monte_carlo
from ml_filter import load_m15_mtf, extract_features, FEAT_COLS

SL_MULT, SL_LO, SL_HI = 0.7, 0.0008, 0.006

POINT = {"ASXAUD":0.1, "DAX40":0.01, "ESXEUR":0.01, "SP500":0.01,
         "UK100":0.01, "USDCAD":0.00001, "USDJPY":0.001, "XAGUSD":0.001}
COMM_PCT = {"USDCAD":0.00008, "USDJPY":0.00008, "XAGUSD":0.00002}  # round-trip, frac of notional

ALL9 = [("ASXAUD","NYO"),("SP500","MOM"),("USDCAD","MOM"),("USDJPY","NYO"),
        ("XAGUSD","ARB"),("DAX40","ARB"),("ESXEUR","NYO"),("UK100","ARB"),("USDJPY","ARB")]

print("Loading + applying costs …")
all_records = []
cost_stats = {}
for sym, arch in ALL9:
    m15, mtf = load_m15_mtf(sym)
    fn = {"ARB":extract_arb,"NYO":extract_nyo,"MOM":extract_mom}[arch]
    trades = fn(m15, mtf)
    recs = extract_features(m15, mtf, trades, sym, arch)
    pt = POINT[sym]; comm = COMM_PCT.get(sym, 0.0)
    costs_R = []
    for r in recs:
        et = r["entry_t"]
        price  = float(m15.loc[et, "close"])
        spread = float(m15.loc[et, "spread"]) * pt
        i = mtf.loc[et]
        h1_atr = float(i["h1_atr"]) if not pd.isna(i["h1_atr"]) else 0.0
        sl_d = float(np.clip(h1_atr*SL_MULT, price*SL_LO, price*SL_HI))
        cost_px = spread + comm*price
        cost_R = cost_px / sl_d if sl_d > 0 else 0.0
        r["R_gross"] = r["R"]
        r["R"] = r["R"] - cost_R          # net
        r["label"] = int(r["R"] > 0)      # relabel on net basis
        costs_R.append(cost_R)
    all_records.extend(recs)
    cost_stats[f"{sym}-{arch}"] = (np.mean(costs_R), np.mean([x["R_gross"] for x in recs]),
                                   np.mean([x["R"] for x in recs]))

print(f"\n{'Stream':<14} {'avg cost(R)':>11} {'gross expR':>11} {'net expR':>9}")
for k,(c,g,n) in sorted(cost_stats.items()):
    print(f"{k:<14} {c:>11.3f} {g:>+11.3f} {n:>+9.3f}")

FEAT_COLS_EXT = FEAT_COLS + ["rolling_wr"]
df = pd.DataFrame(all_records).sort_values("entry_t").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])
df["rolling_wr"] = df["label"].shift(1).rolling(10,min_periods=3).mean().fillna(0.5)

dates = sorted(df["date"].unique())
cut = dates[int(len(dates)*0.70)]
IS  = df[df["date"] < cut]
OOS = df[df["date"] >= cut].copy()

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
X_is = IS[FEAT_COLS_EXT].fillna(0).values; y_is = IS["label"].values
spos = max((y_is==0).sum()/(y_is==1).sum(), 0.1)
xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.7,
                    scale_pos_weight=spos, eval_metric="logloss",
                    random_state=42, verbosity=0)
model = CalibratedClassifierCV(xgb, cv=5, method="isotonic")
model.fit(X_is, y_is)
OOS["prob"] = model.predict_proba(OOS[FEAT_COLS_EXT].fillna(0).values)[:,1]

print(f"\n{'='*74}")
print("  FINAL OOS RESULTS — NET OF SPREAD + COMMISSION (swap = 0, verified)")
print(f"{'='*74}")
print(f"  {'Thresh':>7}  {'n':>4}  {'WR':>6}  {'net expR':>9}  {'t/mo':>5}  "
      f"{'Risk%':>6}  {'Pass%':>6}  {'Bust%':>6}  {'Med.Mo':>7}")
for thr in [0.33, 0.35, 0.36, 0.37, 0.38]:
    f = OOS[OOS["prob"] >= thr]
    if len(f) < 8: print(f"  >{thr:.0%}    {len(f):>4}  — too few"); continue
    Rs = f["R"].values
    d0, d1 = f["date"].min(), f["date"].max()
    tpm = len(f)/max((d1-d0).days/30.44, 0.1)
    best = None
    for risk in [0.0075, 0.0100, 0.0125, 0.0150]:
        mc = monte_carlo(f.to_dict("records"), risk)
        if mc["bust_pct"] <= 5.0 and (best is None or mc["med_mo"] < best[1]["med_mo"]):
            best = (risk, mc)
    if best:
        risk, mc = best
        flag = "  ***" if mc["bust_pct"]<=5 and mc["med_mo"]<=3.5 else ""
        print(f"  >{thr:.0%}    {len(f):>4}  {(Rs>0).mean()*100:>5.1f}%  {Rs.mean():>+9.3f}  "
              f"{tpm:>4.1f}  {risk*100:>5.2f}%  {mc['pass_pct']:>5.1f}%  "
              f"{mc['bust_pct']:>5.1f}%  {mc['med_mo']:>6.1f}{flag}")
    else:
        print(f"  >{thr:.0%}    {len(f):>4}  {(Rs>0).mean()*100:>5.1f}%  {Rs.mean():>+9.3f}  no safe risk")

# gross-vs-net comparison at the chosen config
f35 = OOS[OOS["prob"] >= 0.35]
print(f"\n  At >35%: gross expR={f35['R_gross'].mean():+.3f}  →  net expR={f35['R'].mean():+.3f}  "
      f"(cost = {f35['R_gross'].mean()-f35['R'].mean():.3f}R/trade)")
