"""
ML-Enhanced Trade Filter — XGBoost classifier on trade entry features.

Pipeline:
  1. Rebuild all 9 OOS-survivor streams with rich feature vectors at entry
  2. Chronological 70/30 split (no lookahead)
  3. Train XGBoost on IS (class-balanced)
  4. Apply to OOS at confidence thresholds
  5. Monte Carlo on filtered OOS trades
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings, glob
warnings.filterwarnings("ignore")

from multi_asset_scan import (
    load_raw, build_mtf,
    extract_arb, extract_nyo, extract_mom,
    monte_carlo, ranges as build_ranges
)

RAW_DIR = Path(__file__).parent
SL_MULT = 0.7; SL_LO = 0.0008; SL_HI = 0.006; FORCE_H = 21; TP_RR = 3.5

def load_m15_mtf(sym):
    files = sorted(glob.glob(str(RAW_DIR / f"{sym}_M15_*.csv")))
    fpath = max(files, key=lambda f: Path(f).stat().st_size)
    m15 = load_raw(fpath); mtf = build_mtf(m15)
    return m15, mtf

def extract_features(m15, mtf, trades, sym, arch):
    ar_ranges  = build_ranges(m15, 0, 8)
    nyo_ranges = build_ranges(m15, 10, 13)
    result = []
    for t in trades:
        et = t["entry_t"]
        if et not in mtf.index: continue
        i = mtf.loc[et]
        h4_adx   = float(i["h4_adx"])   if not pd.isna(i["h4_adx"])   else 0.0
        d1_atr_r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
        h1_trend = int(i["h1_trend"]); h4_trend = int(i["h4_trend"]); d1_trend = int(i["d1_trend"])
        h4_ema20 = float(i["h4_ema20"]) if not pd.isna(i["h4_ema20"]) else 0.0
        h1_atr   = float(i["h1_atr"])   if not pd.isna(i["h1_atr"])   else 0.0
        h4_atr   = float(i["h4_atr"])   if not pd.isna(i["h4_atr"])   else 0.0
        price    = float(m15.loc[et, "close"]) if et in m15.index else 0.0
        hour = et.hour; dow = et.dayofweek
        range_pct = 0.0
        dts = pd.Timestamp(t["date"])
        if arch == "ARB" and dts in ar_ranges.index:
            r = ar_ranges.loc[dts]; mid = (r["hi"]+r["lo"])/2
            range_pct = float(r["rng"]/mid) if mid > 0 else 0.0
        elif arch == "NYO" and dts in nyo_ranges.index:
            r = nyo_ranges.loc[dts]; mid = (r["hi"]+r["lo"])/2
            range_pct = float(r["rng"]/mid) if mid > 0 else 0.0
        ema_dist = float((price-h4_ema20)/price*100) if price > 0 else 0.0
        h1_atr_pct = float(h1_atr/price*100) if price > 0 else 0.0
        h4_to_h1   = float(h4_atr/h1_atr) if h1_atr > 0 else 1.0
        feat = dict(h4_adx=h4_adx, h4_to_h1=h4_to_h1, d1_atr_ratio=d1_atr_r,
                    h1_trend=h1_trend, h4_trend=h4_trend, d1_trend=d1_trend,
                    range_pct=range_pct*100, ema_dist=ema_dist, h1_atr_pct=h1_atr_pct,
                    hour=hour, dow=dow, direction=t["d"],
                    arch_arb=int(arch=="ARB"), arch_nyo=int(arch=="NYO"), arch_mom=int(arch=="MOM"))
        result.append({**t, "sym": sym, "arch": arch, **feat, "label": int(t["R"] > 0)})
    return result

FEAT_COLS = ["h4_adx","h4_to_h1","d1_atr_ratio","h1_trend","h4_trend","d1_trend",
             "range_pct","ema_dist","h1_atr_pct","hour","dow","direction",
             "arch_arb","arch_nyo","arch_mom"]

def run():
    ALL9 = [("ASXAUD","NYO"),("SP500","MOM"),("USDCAD","MOM"),("USDJPY","NYO"),
            ("XAGUSD","ARB"),("DAX40","ARB"),("ESXEUR","NYO"),("UK100","ARB"),("USDJPY","ARB")]

    print("Extracting features from all 9 streams …")
    all_records = []
    for sym, arch in ALL9:
        m15, mtf = load_m15_mtf(sym)
        fn = {"ARB":extract_arb,"NYO":extract_nyo,"MOM":extract_mom}[arch]
        trades = fn(m15, mtf)
        recs = extract_features(m15, mtf, trades, sym, arch)
        all_records.extend(recs)
        print(f"  {sym}-{arch}: {len(trades)} trades")

    df = pd.DataFrame(all_records).sort_values("entry_t").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["rolling_wr"] = df["label"].shift(1).rolling(10, min_periods=3).mean().fillna(0.5)
    FEAT_COLS_EXT = FEAT_COLS + ["rolling_wr"]

    print(f"\nTotal: {len(df)} trades  WR={df['label'].mean()*100:.1f}%  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    dates = sorted(df["date"].unique())
    cut   = dates[int(len(dates)*0.70)]
    IS    = df[df["date"] <  cut]
    OOS   = df[df["date"] >= cut]
    print(f"IS:  {len(IS):>4} trades  WR={IS['label'].mean()*100:.1f}%  "
          f"({IS['date'].min().date()} → {IS['date'].max().date()})")
    print(f"OOS: {len(OOS):>4} trades  WR={OOS['label'].mean()*100:.1f}%  "
          f"({OOS['date'].min().date()} → {OOS['date'].max().date()})")

    X_is  = IS[FEAT_COLS_EXT].fillna(0).values;  y_is  = IS["label"].values
    X_oos = OOS[FEAT_COLS_EXT].fillna(0).values; y_oos = OOS["label"].values

    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    scale_pos = (y_is==0).sum()/(y_is==1).sum()
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
                        subsample=0.8, colsample_bytree=0.7,
                        scale_pos_weight=scale_pos, eval_metric="logloss",
                        random_state=42, verbosity=0)
    model = CalibratedClassifierCV(xgb, cv=5, method="isotonic")
    print("\nTraining XGBoost (5-fold calibrated) …")
    model.fit(X_is, y_is)

    prob_oos = model.predict_proba(X_oos)[:,1]
    print(f"OOS AUC: {roc_auc_score(y_oos, prob_oos):.4f}")

    try:
        imps = np.mean([e.estimator.feature_importances_
                        for e in model.calibrated_classifiers_], axis=0)
        print("Top features:")
        for idx in np.argsort(imps)[::-1][:6]:
            print(f"  {FEAT_COLS_EXT[idx]:<20} {imps[idx]:.4f}")
    except Exception: pass

    OOS = OOS.copy(); OOS["prob"] = prob_oos

    print(f"\n{'='*78}")
    print(f"  ML Filter OOS Threshold Scan  (Top-5 streams only)")
    print(f"{'='*78}")
    print(f"  {'Thresh':>7}  {'Kept':>5}  {'WR':>6}  {'expR':>7}  "
          f"{'t/mo':>5}  {'Risk%':>6}  {'Pass%':>6}  {'Bust%':>6}  {'Med.Mo':>7}")
    print(f"  {'-'*76}")

    TOP5_KEYS = {("ASXAUD","NYO"),("SP500","MOM"),("USDCAD","MOM"),("USDJPY","NYO"),("XAGUSD","ARB")}
    OOS_T5 = OOS[OOS.apply(lambda r: (r["sym"],r["arch"]) in TOP5_KEYS, axis=1)]

    sweet = None
    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        filt = OOS_T5[OOS_T5["prob"] >= thresh]
        if len(filt) < 15:
            print(f"  >{thresh:.0%}        {len(filt):>5}  — too few"); continue
        trades_f = filt.to_dict("records")
        Rs = np.array([t["R"] for t in trades_f])
        wr = (Rs>0).mean()*100; expR = Rs.mean()
        d0 = filt["date"].min(); d1 = filt["date"].max()
        tpm = len(filt)/max((d1-d0).days/30.44, 0.1)
        best = None
        for risk in [0.0020,0.0025,0.0030,0.0035,0.0040,0.0050,0.0060,0.0080,0.0100]:
            mc = monte_carlo(trades_f, risk)
            if mc["bust_pct"] <= 5.0:
                if best is None or mc["med_mo"] < best[1]:
                    best = (risk, mc["med_mo"], mc["pass_pct"], mc["bust_pct"])
        if best:
            risk, mm, pp, bp = best
            flag = "  *** TARGET ***" if bp<=5.0 and mm<=3.5 else (
                   "  ** close **"    if bp<=5.0 and mm<=5.0 else "")
            if bp<=5.0 and mm<=3.5 and sweet is None:
                sweet = (thresh, filt, trades_f, risk)
            print(f"  >{thresh:.0%}        {len(filt):>5}  {wr:>5.1f}%  {expR:>+7.3f}  "
                  f"{tpm:>4.1f}  {risk*100:>5.2f}%  {pp:>5.1f}%  {bp:>5.1f}%  {mm:>6.1f}{flag}")
        else:
            print(f"  >{thresh:.0%}        {len(filt):>5}  {wr:>5.1f}%  {expR:>+7.3f}  "
                  f"{tpm:>4.1f}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>7}")

    print(f"\n  Baseline OOS Top-5 (no filter):")
    base = OOS_T5.to_dict("records")
    d0b = OOS_T5["date"].min(); d1b = OOS_T5["date"].max()
    for risk in [0.0020,0.0025,0.0030]:
        mc = monte_carlo(base, risk)
        tpm_b = len(base)/max((d1b-d0b).days/30.44,0.1)
        print(f"    {risk*100:.2f}%  pass={mc['pass_pct']:.1f}%  "
              f"bust={mc['bust_pct']:.1f}%  med={mc['med_mo']:.1f}mo  {tpm_b:.1f}t/mo")

    if sweet:
        thresh, filt, trades_f, best_risk = sweet
        Rs = np.array([t["R"] for t in trades_f])
        tpm = len(trades_f)/max((filt["date"].max()-filt["date"].min()).days/30.44,0.1)
        print(f"\n{'='*65}")
        print(f"  *** TARGET HIT: ML threshold >{thresh:.0%} ***")
        print(f"  {len(trades_f)} trades  expR={Rs.mean():+.3f}  WR={(Rs>0).mean()*100:.0f}%  {tpm:.1f} t/mo")
        print(f"{'='*65}")
        print(f"  {'Risk%':>6}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}")
        for risk in [0.0020,0.0025,0.0030,0.0035,0.0040,0.0050,0.0060,0.0080]:
            mc = monte_carlo(trades_f, risk)
            flag = " ***" if mc["bust_pct"]<=5.0 and mc["med_mo"]<=3.5 else ""
            print(f"  {risk*100:>5.2f}%  {mc['pass_pct']:>6.1f}%  {mc['bust_pct']:>5.1f}%  "
                  f"{mc['med_mo']:>6.1f}{flag}")
    print()

if __name__ == "__main__":
    run()

def run_all9_filter():
    """Apply trained ML model to all 9 streams, check if frequency is enough."""
    ALL9 = [("ASXAUD","NYO"),("SP500","MOM"),("USDCAD","MOM"),("USDJPY","NYO"),
            ("XAGUSD","ARB"),("DAX40","ARB"),("ESXEUR","NYO"),("UK100","ARB"),("USDJPY","ARB")]

    print("\n" + "="*70)
    print("  ALL-9 streams with ML filter (>35% confidence)")
    print("="*70)

    all_records = []
    for sym, arch in ALL9:
        m15, mtf = load_m15_mtf(sym)
        fn = {"ARB":extract_arb,"NYO":extract_nyo,"MOM":extract_mom}[arch]
        trades = fn(m15, mtf)
        recs = extract_features(m15, mtf, trades, sym, arch)
        all_records.extend(recs)

    df = pd.DataFrame(all_records).sort_values("entry_t").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["rolling_wr"] = df["label"].shift(1).rolling(10,min_periods=3).mean().fillna(0.5)
    FEAT_COLS_EXT = FEAT_COLS + ["rolling_wr"]

    dates = sorted(df["date"].unique())
    cut   = dates[int(len(dates)*0.70)]
    IS    = df[df["date"] <  cut]
    OOS   = df[df["date"] >= cut].copy()

    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    X_is = IS[FEAT_COLS_EXT].fillna(0).values; y_is = IS["label"].values
    scale_pos = (y_is==0).sum()/(y_is==1).sum()
    xgb = XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.04,
                        subsample=0.8,colsample_bytree=0.7,
                        scale_pos_weight=scale_pos,eval_metric="logloss",
                        random_state=42,verbosity=0)
    model = CalibratedClassifierCV(xgb,cv=5,method="isotonic")
    print("Retraining on all-9 IS …")
    model.fit(X_is, y_is)
    OOS["prob"] = model.predict_proba(OOS[FEAT_COLS_EXT].fillna(0).values)[:,1]

    for thresh in [0.33, 0.35, 0.38, 0.40]:
        filt = OOS[OOS["prob"] >= thresh]
        if len(filt) < 5: print(f"  >{thresh:.0%}: too few ({len(filt)})"); continue
        trades_f = filt.to_dict("records")
        Rs = np.array([t["R"] for t in trades_f])
        wr = (Rs>0).mean()*100; expR = Rs.mean()
        d0 = filt["date"].min(); d1 = filt["date"].max()
        tpm = len(filt)/max((d1-d0).days/30.44,0.1)
        print(f"\n  >{thresh:.0%}: {len(filt)} trades  WR={wr:.0f}%  expR={expR:+.3f}  {tpm:.1f}t/mo")
        print(f"  {'Risk%':>6}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}")
        for risk in [0.0050,0.0075,0.0100,0.0125,0.0150,0.0200]:
            mc = monte_carlo(trades_f, risk)
            flag = " ***" if mc["bust_pct"]<=5.0 and mc["med_mo"]<=3.5 else ""
            print(f"  {risk*100:>5.2f}%  {mc['pass_pct']:>6.1f}%  {mc['bust_pct']:>5.1f}%  "
                  f"{mc['med_mo']:>6.1f}{flag}")

if __name__ == "__main__":
    run_all9_filter()
