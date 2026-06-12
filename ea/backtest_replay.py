"""
Backtest Replay — verifies live runner logic matches the Python backtest numbers.

Runs the same pipeline as final_costs.py but via the production code path:
  - Signal extraction: multi_asset_scan extract_arb/nyo/mom (same as backtest)
  - Feature extraction: ml_filter.extract_features (same as backtest)
  - ML scoring: ea/model.joblib (frozen production model)
  - Cost model: spread from CSV + commission per symbol
  - Outcome resolution: same SL/TP/FC logic as backtest

This lets you verify:
  1. Python numbers match backtest before going live
  2. Model.joblib is correctly trained and loaded
  3. OOS WR and expR are in the expected range

Usage (from hahaha/ root):
    python ea/backtest_replay.py

Expected OOS (filtered) output:
    n ≥ 40   WR ≥ 45%   net expR ≥ +0.50
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

from multi_asset_scan import load_raw, build_mtf, extract_arb, extract_nyo, extract_mom
from ml_filter import extract_features, load_m15_mtf, FEAT_COLS
from ea.config import MODEL_PATH, ML_THRESHOLD, SL_MULT, SL_LO, SL_HI

FEAT_COLS_EXT = FEAT_COLS + ["rolling_wr"]

POINT = {"ASXAUD":0.1,"DAX40":0.01,"ESXEUR":0.01,"SP500":0.01,"UK100":0.01,
         "USDCAD":0.00001,"USDJPY":0.001,"XAGUSD":0.001}
COMM  = {"USDCAD":0.00008,"USDJPY":0.00008,"XAGUSD":0.00002}

ALL9 = [
    ("ASXAUD", "NYO"), ("SP500",  "MOM"), ("USDCAD", "MOM"),
    ("USDJPY", "NYO"), ("XAGUSD", "ARB"), ("DAX40",  "ARB"),
    ("ESXEUR", "NYO"), ("UK100",  "ARB"), ("USDJPY", "ARB"),
]


def add_costs(recs, m15, sym):
    """Add net R (after spread + commission) to each record."""
    pt = POINT[sym]
    cm = COMM.get(sym, 0.0)
    for r in recs:
        et    = r["entry_t"]
        price = float(m15.loc[et, "close"]) if et in m15.index else r.get("entry", 0)
        spread_px = float(m15.loc[et, "spread"]) * pt if et in m15.index else 0.0
        comm_px   = cm * price
        sl_d = float(np.clip(
            r.get("h1_atr", 0) * SL_MULT if r.get("h1_atr", 0) > 0 else 0,
            price * SL_LO, price * SL_HI
        ))
        cost_R = (spread_px + comm_px) / sl_d if sl_d > 0 else 0.0
        r["R_gross"] = r["R"]
        r["R_net"]   = round(r["R"] - cost_R, 4)
        r["cost_R"]  = round(cost_R, 4)
    return recs


def run():
    # ── Load model ────────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        print("Run: python ea/train_model.py")
        sys.exit(1)

    payload   = joblib.load(MODEL_PATH)
    model     = payload["model"]
    feat_cols = payload["feat_cols"]
    print(f"Model loaded: trained to {payload['train_end']}  "
          f"n_train={payload['n_train']}  threshold={payload['threshold']:.0%}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("\nExtracting trades from historical CSVs …")
    all_recs = []
    for sym, arch in ALL9:
        try:
            m15, mtf = load_m15_mtf(sym)
        except Exception as e:
            print(f"  SKIP {sym}-{arch}: {e}")
            continue
        fn     = {"ARB": extract_arb, "NYO": extract_nyo, "MOM": extract_mom}[arch]
        trades = fn(m15, mtf)
        recs   = extract_features(m15, mtf, trades, sym, arch)

        # Attach H1 ATR for cost calculation (already in mtf)
        for r in recs:
            et = r["entry_t"]
            r["h1_atr"] = float(mtf.loc[et, "h1_atr"]) if et in mtf.index and not pd.isna(mtf.loc[et, "h1_atr"]) else 0.0

        recs = add_costs(recs, m15, sym)
        all_recs.extend(recs)
        print(f"  {sym}-{arch}: {len(recs)} raw trades")

    df = pd.DataFrame(all_recs).sort_values("entry_t").reset_index(drop=True)
    df["date_ts"] = pd.to_datetime(df["date"])

    # ── IS/OOS split ──────────────────────────────────────────────────────────
    unique_dates = sorted(df["date_ts"].dt.date.unique())
    cutoff = unique_dates[int(len(unique_dates) * 0.70)]
    df["period"] = np.where(df["date_ts"].dt.date >= cutoff, "OOS", "IS")
    print(f"\nIS/OOS split: cutoff={cutoff}  IS={sum(df.period=='IS')}  OOS={sum(df.period=='OOS')}")

    # ── Add rolling_wr (chronological, no lookahead) ──────────────────────────
    df["rolling_wr"] = df["label"].shift(1).rolling(10, min_periods=3).mean().fillna(0.5)

    # ── ML scoring ────────────────────────────────────────────────────────────
    X    = df[feat_cols].fillna(0).values
    prob = model.predict_proba(X)[:, 1]
    df["prob"]     = prob
    df["filtered"] = prob >= ML_THRESHOLD

    # ── Results summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("REPLAY RESULTS (production model vs historical data)")
    print("=" * 65)

    for period in ["IS", "OOS"]:
        sub  = df[df["period"] == period]
        filt = sub[sub["filtered"]]
        if filt.empty:
            continue
        wr     = (filt["R_net"] > 0).mean()
        expr   = filt["R_net"].mean()
        tp_ct  = (filt["result"] == "tp").sum()
        sl_ct  = (filt["result"] == "sl").sum()
        fc_ct  = (~filt["result"].isin(["tp", "sl"])).sum()
        print(f"\n{period}: n={len(filt):3d}  WR={wr*100:.1f}%  "
              f"net expR={expr:+.3f}  [TP={tp_ct} SL={sl_ct} FC/TO={fc_ct}]")

        per_stream = filt.groupby(["sym", "arch"]).agg(
            n=("R_net", "count"),
            wr=("R_net", lambda x: (x>0).mean()),
            expr=("R_net", "mean"),
        ).reset_index()
        for _, row in per_stream.iterrows():
            flag = "✅" if row["expr"] > 0 else "⚠️ "
            print(f"  {flag} {row['sym']}-{row['arch']:3s}: "
                  f"n={int(row['n']):2d}  WR={row['wr']*100:.0f}%  expR={row['expr']:+.3f}")

    # ── OOS gate check ────────────────────────────────────────────────────────
    oos_f = df[(df["period"] == "OOS") & df["filtered"]]
    print("\n" + "-" * 65)
    print("OOS GATE CHECKS (must all pass before going live):")
    checks = [
        ("n ≥ 30 trades",     len(oos_f) >= 30,        f"n={len(oos_f)}"),
        ("WR ≥ 45%",          (oos_f["R_net"]>0).mean() >= 0.45,
                               f"WR={100*(oos_f['R_net']>0).mean():.1f}%"),
        ("net expR ≥ +0.50",  oos_f["R_net"].mean() >= 0.50,
                               f"expR={oos_f['R_net'].mean():+.4f}"),
        ("avg cost ≤ 0.20R",  oos_f["cost_R"].mean() <= 0.20,
                               f"avg cost={oos_f['cost_R'].mean():.3f}R"),
    ]
    all_pass = True
    for name, ok, val in checks:
        mark = "PASS ✅" if ok else "FAIL ❌"
        print(f"  {mark}  {name}  ({val})")
        if not ok:
            all_pass = False

    if all_pass:
        print("\n✅  All gates passed — production model is verified.")
    else:
        print("\n❌  Gate failure — retrain or review before live deployment.")

    # ── Save full trade log ───────────────────────────────────────────────────
    out_cols = ["date", "entry_t", "sym", "arch", "d", "result",
                "R_gross", "cost_R", "R_net", "prob", "filtered", "period"]
    out_cols = [c for c in out_cols if c in df.columns]
    out_path = Path(__file__).parent / "replay_trades.csv"
    df[out_cols].to_csv(out_path, index=False)
    print(f"\nFull trade log saved → {out_path}")
    print("(Compare this with final_costs.py output to verify sync)")

    return df


if __name__ == "__main__":
    run()
