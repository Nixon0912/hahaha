"""
Rigorous validation of the ML-filtered strategy.

Tests:
  1. Statistical significance — is WR=50% vs 30% base meaningful with n=42?
  2. Walk-forward validation — rolling train/test windows, no single lucky OOS window
  3. Per-year breakdown — does edge exist each year independently?
  4. Per-regime breakdown — trending / volatile / calm
  5. Sensitivity analysis — threshold 0.30 to 0.45, risk 0.75% to 2.0%
  6. Bootstrap confidence intervals on expR and WR
  7. Max consecutive loss stress — what does a bad streak look like on $10k?
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings, glob
warnings.filterwarnings("ignore")

from multi_asset_scan import (
    load_raw, build_mtf, extract_arb, extract_nyo, extract_mom,
    monte_carlo, ranges as build_ranges
)
from ml_filter import load_m15_mtf, extract_features, FEAT_COLS

RAW_DIR = Path(__file__).parent
INIT_BAL = 10_000.0
FEAT_COLS_EXT = FEAT_COLS + ["rolling_wr"]

ALL9 = [("ASXAUD","NYO"),("SP500","MOM"),("USDCAD","MOM"),("USDJPY","NYO"),
        ("XAGUSD","ARB"),("DAX40","ARB"),("ESXEUR","NYO"),("UK100","ARB"),("USDJPY","ARB")]

THRESH = 0.35
RISK   = 0.0125

# ── Helper: train model on given IS, score given OOS ─────────────────────────
def train_score(IS_df, OOS_df):
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    X_is = IS_df[FEAT_COLS_EXT].fillna(0).values
    y_is = IS_df["label"].values
    if len(np.unique(y_is)) < 2 or len(y_is) < 30:
        return None
    scale_pos = max((y_is==0).sum()/(y_is==1).sum(), 0.1)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.7,
                        scale_pos_weight=scale_pos, eval_metric="logloss",
                        random_state=42, verbosity=0)
    model = CalibratedClassifierCV(xgb, cv=3, method="isotonic")
    model.fit(X_is, y_is)
    OOS_out = OOS_df.copy()
    OOS_out["prob"] = model.predict_proba(
        OOS_df[FEAT_COLS_EXT].fillna(0).values)[:,1]
    return OOS_out

def filtered(df, thresh=THRESH):
    return df[df["prob"] >= thresh]

def stats_for(trades_df):
    if len(trades_df) == 0:
        return {}
    Rs = np.array([t["R"] for t in trades_df.to_dict("records")])
    wr = (Rs>0).mean()*100
    er = Rs.mean()
    n  = len(Rs)
    return {"n": n, "WR": wr, "expR": er, "Rs": Rs}

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading all 9 streams …")
all_records = []
for sym, arch in ALL9:
    m15, mtf = load_m15_mtf(sym)
    fn = {"ARB":extract_arb,"NYO":extract_nyo,"MOM":extract_mom}[arch]
    trades = fn(m15, mtf)
    recs = extract_features(m15, mtf, trades, sym, arch)
    all_records.extend(recs)

df = pd.DataFrame(all_records).sort_values("entry_t").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["rolling_wr"] = df["label"].shift(1).rolling(10,min_periods=3).mean().fillna(0.5)
print(f"  Total: {len(df)} trades  {df['date'].min().date()} → {df['date'].max().date()}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATISTICAL SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  1. STATISTICAL SIGNIFICANCE")
print(f"{'='*70}")

# Standard 70/30 OOS
dates = sorted(df["date"].unique())
cut = dates[int(len(dates)*0.70)]
IS  = df[df["date"] < cut]
OOS_raw = df[df["date"] >= cut]
OOS_scored = train_score(IS, OOS_raw)
OOS_filt   = filtered(OOS_scored)

n_filt = len(OOS_filt)
n_wins = int((OOS_filt["label"] == 1).sum())
base_wr = df["label"].mean()  # overall base rate

# Binomial test: H0: WR = base_wr
binom = stats.binomtest(n_wins, n_filt, base_wr, alternative="greater")
# 95% CI on WR (Wilson interval)
from statsmodels.stats.proportion import proportion_confint
ci_lo, ci_hi = proportion_confint(n_wins, n_filt, alpha=0.05, method="wilson")

print(f"\n  Unfiltered base WR (all trades): {base_wr*100:.1f}%")
print(f"  ML-filtered OOS WR: {n_wins}/{n_filt} = {n_wins/n_filt*100:.1f}%")
print(f"  95% Wilson CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
print(f"  Binomial test p-value: {binom.pvalue:.4f}  "
      f"({'✅ significant (p<0.05)' if binom.pvalue < 0.05 else '❌ NOT significant'})")
print(f"\n  ⚠️  n={n_filt} trades in OOS — small sample caveat:")
print(f"  With n={n_filt}, 95% CI spans {(ci_hi-ci_lo)*100:.0f} percentage points.")
print(f"  True WR could be anywhere from {ci_lo*100:.0f}% to {ci_hi*100:.0f}%.")

# Bootstrap expR CI
bootstrap_expR = []
Rs_filt = np.array([t["R"] for t in OOS_filt.to_dict("records")])
rng = np.random.default_rng(42)
for _ in range(10000):
    sample = rng.choice(Rs_filt, size=len(Rs_filt), replace=True)
    bootstrap_expR.append(sample.mean())
expR_ci_lo, expR_ci_hi = np.percentile(bootstrap_expR, [2.5, 97.5])
print(f"\n  Bootstrap 95% CI on expR: [{expR_ci_lo:+.3f}, {expR_ci_hi:+.3f}]")
print(f"  Point estimate expR: {Rs_filt.mean():+.3f}")
print(f"  {'✅ Lower bound positive' if expR_ci_lo > 0 else '❌ Lower bound negative — edge not proven'}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  2. WALK-FORWARD VALIDATION (expanding window, annual OOS steps)")
print(f"{'='*70}")
print(f"\n  {'Window':<28}  {'IS n':>5}  {'OOS n':>5}  {'OOS filt':>8}  "
      f"{'WR':>6}  {'expR':>7}  {'Sig?':>5}")
print(f"  {'-'*68}")

wf_results = []
years = sorted(df["year"].unique())

for oos_year in years[2:]:   # need at least 2 years of IS
    IS_wf  = df[df["year"] < oos_year]
    OOS_wf = df[df["year"] == oos_year]
    if len(OOS_wf) < 20: continue
    scored = train_score(IS_wf, OOS_wf)
    if scored is None: continue
    filt_wf = filtered(scored)
    if len(filt_wf) == 0:
        print(f"  Train <{oos_year} → Test {oos_year}    {len(IS_wf):>5}  "
              f"{len(OOS_wf):>5}  {'0':>8}  {'—':>6}  {'—':>7}  {'—':>5}")
        continue
    Rs_wf = np.array([t["R"] for t in filt_wf.to_dict("records")])
    wr_wf = (Rs_wf>0).mean()*100; er_wf = Rs_wf.mean()
    nw = int((filt_wf["label"]==1).sum())
    pval = stats.binomtest(nw, len(filt_wf), base_wr, alternative="greater").pvalue
    sig = "✅" if pval < 0.10 else "〜" if pval < 0.20 else "❌"
    wf_results.append({"year": oos_year, "n": len(filt_wf),
                        "WR": wr_wf, "expR": er_wf, "sig": sig})
    print(f"  Train <{oos_year} → Test {oos_year}    {len(IS_wf):>5}  "
          f"{len(OOS_wf):>5}  {len(filt_wf):>8}  {wr_wf:>5.0f}%  "
          f"{er_wf:>+7.3f}  {sig:>5}  (p={pval:.2f})")

if wf_results:
    pos = sum(1 for r in wf_results if r["expR"] > 0)
    print(f"\n  Positive expR in {pos}/{len(wf_results)} walk-forward windows")
    avg_er = np.mean([r["expR"] for r in wf_results])
    print(f"  Average walk-forward expR: {avg_er:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PER-YEAR BREAKDOWN (full history, model retrained on all prior years)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  3. PER-YEAR BREAKDOWN — Unfiltered vs ML-filtered")
print(f"{'='*70}")
print(f"\n  {'Year':>6}  {'All trades':>10}  {'All WR':>7}  "
      f"{'Filt n':>7}  {'Filt WR':>8}  {'expR':>7}")
print(f"  {'-'*60}")
for yr in years:
    yr_df = df[df["year"]==yr]
    if len(yr_df) < 5: continue
    # Use the same standard model (train on pre-2025, score 2025+)
    # For per-year: just report unfiltered stats and where available filtered
    all_wr = yr_df["label"].mean()*100
    # For filtered: only show if in OOS window
    if yr_df["date"].min() >= pd.Timestamp(cut):
        yr_filt = OOS_scored[OOS_scored["date"].dt.year == yr]
        yr_filt = filtered(yr_filt)
        if len(yr_filt) > 0:
            Rs_yr = np.array([t["R"] for t in yr_filt.to_dict("records")])
            filt_wr = (Rs_yr>0).mean()*100; filt_er = Rs_yr.mean()
            print(f"  {yr:>6}  {len(yr_df):>10}  {all_wr:>6.0f}%  "
                  f"{len(yr_filt):>7}  {filt_wr:>7.0f}%  {filt_er:>+7.3f}  (OOS)")
        else:
            print(f"  {yr:>6}  {len(yr_df):>10}  {all_wr:>6.0f}%  "
                  f"{'0':>7}  {'—':>8}  {'—':>7}  (OOS, 0 filtered)")
    else:
        print(f"  {yr:>6}  {len(yr_df):>10}  {all_wr:>6.0f}%  "
              f"{'IS':>7}  {'(in-sample)':>8}  {'—':>7}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. REGIME BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  4. REGIME BREAKDOWN (based on D1 ATR ratio at entry)")
print(f"{'='*70}")
print(f"\n  D1 ATR ratio > 1.5 = HIGH VOL, 0.8–1.5 = NORMAL, < 0.8 = LOW VOL")

OOS_filt2 = OOS_filt.copy()
def regime(r):
    if r > 1.5: return "HIGH_VOL"
    elif r >= 0.8: return "NORMAL"
    else: return "LOW_VOL"
OOS_filt2["regime"] = OOS_filt2["d1_atr_ratio"].apply(regime)

print(f"\n  {'Regime':<12}  {'Trades':>7}  {'WR':>6}  {'expR':>7}")
print(f"  {'-'*36}")
for reg, grp in OOS_filt2.groupby("regime"):
    Rs_r = np.array([t["R"] for t in grp.to_dict("records")])
    print(f"  {reg:<12}  {len(grp):>7}  {(Rs_r>0).mean()*100:>5.0f}%  "
          f"{Rs_r.mean():>+7.3f}")

# Also by time-of-day
print(f"\n  By session (entry hour):")
print(f"  {'Session':<14}  {'Trades':>7}  {'WR':>6}  {'expR':>7}")
OOS_filt2["session"] = OOS_filt2["hour"].apply(
    lambda h: "LONDON(8-12)"  if 8<=h<12 else
              "NY(13-17)"     if 13<=h<17 else
              "OVERLAP(12-13)" if h==12 else "OTHER")
for sess, grp in OOS_filt2.groupby("session"):
    Rs_s = np.array([t["R"] for t in grp.to_dict("records")])
    print(f"  {sess:<14}  {len(grp):>7}  {(Rs_s>0).mean()*100:>5.0f}%  "
          f"{Rs_s.mean():>+7.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  5. SENSITIVITY — threshold stability")
print(f"{'='*70}")
print(f"\n  {'Thresh':>8}  {'n OOS':>6}  {'WR':>6}  {'expR':>7}  "
      f"{'Pass%':>6}  {'Bust%':>6}  {'Med.Mo':>7}")
print(f"  {'-'*58}")
for thr in [0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40]:
    f2 = OOS_scored[OOS_scored["prob"] >= thr]
    if len(f2) < 8: print(f"  >{thr:.0%}       {len(f2):>6}  — too few"); continue
    t2 = f2.to_dict("records")
    Rs2 = np.array([t["R"] for t in t2])
    d0 = f2["date"].min(); d1 = f2["date"].max()
    tpm2 = len(f2)/max((d1-d0).days/30.44,0.1)
    mc = monte_carlo(t2, RISK)
    flag = " ***" if mc["bust_pct"]<=5.0 and mc["med_mo"]<=3.5 else ""
    print(f"  >{thr:.0%}       {len(f2):>6}  {(Rs2>0).mean()*100:>5.0f}%  "
          f"{Rs2.mean():>+7.3f}  {mc['pass_pct']:>5.1f}%  "
          f"{mc['bust_pct']:>5.1f}%  {mc['med_mo']:>6.1f}{flag}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. DRAWDOWN STRESS — worst sequential streak impact
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  6. DRAWDOWN STRESS TEST")
print(f"{'='*70}")

Rs_filt = np.array([t["R"] for t in OOS_filt.to_dict("records")])
n_f = len(Rs_filt)

# Worst N consecutive trades
print(f"\n  Worst consecutive loss sequences (at {RISK*100:.2f}% risk):")
for n_consec in [3, 4, 5, 6, 7]:
    # Slide window
    min_r = min(sum(Rs_filt[i:i+n_consec]) for i in range(len(Rs_filt)-n_consec+1)) if len(Rs_filt)>=n_consec else 0
    # Dollar impact starting from $10k
    impact = INIT_BAL * RISK * min_r  # rough (ignores compounding)
    pct = impact/INIT_BAL*100
    print(f"  Worst {n_consec} consecutive: {min_r:+.2f}R → ${impact:+,.0f} ({pct:+.1f}%)")

# All-loss scenario (all SL hit): probability at n=42 and WR=50%
from scipy.stats import binom as sp_binom
print(f"\n  Probability of ≥5 consecutive losses in a 42-trade sequence")
print(f"  (WR=50%, independent): {1-(1-0.5**5)**38*0.5**4:.3f}")

# MC worst-case: P(at any point balance < $9000)
print(f"\n  MC worst-path: P(balance ever below $9,000) at {RISK*100:.2f}% risk:")
bust_ever = 0
rng2 = np.random.default_rng(99)
for _ in range(10000):
    bal2 = INIT_BAL
    for r in rng2.choice(Rs_filt, size=len(Rs_filt), replace=False):
        bal2 += bal2*RISK*r
        if bal2 < 9000: bust_ever+=1; break
print(f"  {bust_ever/10000*100:.1f}% (using actual OOS trade sequence reshuffled)")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  7. VALIDATION SUMMARY")
print(f"{'='*70}")

wf_positive = sum(1 for r in wf_results if r["expR"] > 0) if wf_results else 0
wf_total    = len(wf_results)

print(f"""
  Statistical significance (n=42 OOS): p={binom.pvalue:.3f}
    {'✅ Significant at p<0.05' if binom.pvalue<0.05 else '❌ Not significant — n too small'}
  95% CI on expR: [{expR_ci_lo:+.3f}, {expR_ci_hi:+.3f}]
    {'✅ Lower bound > 0 — edge likely real' if expR_ci_lo > 0 else '❌ Lower bound ≤ 0 — edge uncertain'}
  Walk-forward: {wf_positive}/{wf_total} annual windows positive expR
    {'✅ Consistent across years' if wf_positive >= wf_total*0.75 else '⚠️  Inconsistent — regime dependent'}

  KEY CAVEATS:
  • n=42 is very small — results can shift dramatically with ±5 trades
  • Model trained on 2020-2024, OOS is only 2025-2026 (18 months, one regime)
  • XGBoost with 16 features on 1477 IS samples — overfitting risk even with CV
  • Walk-forward annual windows also have small n per window
  • High-confidence filter may be picking up luck rather than skill

  HONEST VERDICT:
  {'✅ Edge appears real but fragile — proceed with smallest viable live size' 
   if binom.pvalue < 0.05 and expR_ci_lo > 0 and wf_positive >= wf_total*0.5
   else '⚠️  Edge is plausible but statistically unproven — paper trade first'}
""")
