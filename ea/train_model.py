"""
One-time model training script.
Run ONCE before deploying the EA. Saves frozen model to ea/model.joblib.

Usage:
    cd /path/to/hahaha
    python ea/train_model.py
"""
import numpy as np
import pandas as pd
import joblib
import glob
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

from multi_asset_scan import load_raw, build_mtf, extract_arb, extract_nyo, extract_mom
from ml_filter import load_m15_mtf, extract_features, FEAT_COLS
from ea.config import (MODEL_PATH, RAW_DATA_DIR, ML_THRESHOLD,
                       SL_MULT, SL_LO, SL_HI)

POINT = {"ASXAUD":0.1,"DAX40":0.01,"ESXEUR":0.01,"SP500":0.01,"UK100":0.01,
         "USDCAD":0.00001,"USDJPY":0.001,"XAGUSD":0.001}
COMM  = {"USDCAD":0.00008,"USDJPY":0.00008,"XAGUSD":0.00002}

ALL9 = [("ASXAUD","NYO"),("SP500","MOM"),("USDCAD","MOM"),("USDJPY","NYO"),
        ("XAGUSD","ARB"),("DAX40","ARB"),("ESXEUR","NYO"),("UK100","ARB"),("USDJPY","ARB")]

FEAT_COLS_EXT = FEAT_COLS + ["rolling_wr"]


def build_dataset():
    print("Building training dataset from historical data …")
    all_records = []
    for sym, arch in ALL9:
        m15, mtf = load_m15_mtf(sym)
        fn = {"ARB": extract_arb, "NYO": extract_nyo, "MOM": extract_mom}[arch]
        trades = fn(m15, mtf)
        recs = extract_features(m15, mtf, trades, sym, arch)
        pt = POINT[sym]; cm = COMM.get(sym, 0.0)
        for r in recs:
            et = r["entry_t"]; price = float(m15.loc[et, "close"])
            spread = float(m15.loc[et, "spread"]) * pt
            h1 = mtf.loc[et, "h1_atr"]
            h1 = float(h1) if not pd.isna(h1) else 0.0
            sl = float(np.clip(h1 * SL_MULT, price * SL_LO, price * SL_HI))
            r["R"] = r["R"] - (spread + cm * price) / sl if sl > 0 else r["R"]
            r["label"] = int(r["R"] > 0)
        all_records.extend(recs)
        print(f"  {sym}-{arch}: {len(recs)} trades")

    df = pd.DataFrame(all_records).sort_values("entry_t").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["rolling_wr"] = df["label"].shift(1).rolling(10, min_periods=3).mean().fillna(0.5)
    return df


def train(df):
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    # Train on ALL data (full history — this is the frozen production model)
    X = df[FEAT_COLS_EXT].fillna(0).values
    y = df["label"].values
    spos = max((y == 0).sum() / (y == 1).sum(), 0.1)

    print(f"\nTraining on full dataset: {len(df)} trades …")
    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=spos, eval_metric="logloss",
        random_state=42, verbosity=0
    )
    model = CalibratedClassifierCV(xgb, cv=5, method="isotonic")
    model.fit(X, y)

    # Sanity check on training data
    prob = model.predict_proba(X)[:, 1]
    filt = prob >= ML_THRESHOLD
    print(f"Training AUC (in-sample): {roc_auc_score(y, prob):.4f}")
    print(f"Filtered trades (>{ML_THRESHOLD:.0%}): {filt.sum()} / {len(df)}")
    print(f"Filtered WR: {y[filt].mean()*100:.1f}%")
    return model


def save(model, df):
    payload = {
        "model": model,
        "feat_cols": FEAT_COLS_EXT,
        "train_end": str(df["date"].max().date()),
        "n_train": len(df),
        "threshold": ML_THRESHOLD,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Train period: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Threshold locked: P(win) > {ML_THRESHOLD:.0%}")
    print(f"Risk locked: 1.25% per trade")


if __name__ == "__main__":
    df = build_dataset()
    model = train(df)
    save(model, df)
    print("\nDone. Model is frozen for live deployment. Do not retrain without full revalidation.")
