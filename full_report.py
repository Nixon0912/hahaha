"""
Full Performance Report — ML-filtered strategy (>35% confidence, all-9 streams)
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
from ml_filter import load_m15_mtf, extract_features, FEAT_COLS

RAW_DIR = Path(__file__).parent
INIT_BAL = 10_000.0
RISK_PCT  = 0.0100   # 1.00% per trade (sweet spot)
TP_RR     = 3.5

ALL9 = [("ASXAUD","NYO"),("SP500","MOM"),("USDCAD","MOM"),("USDJPY","NYO"),
        ("XAGUSD","ARB"),("DAX40","ARB"),("ESXEUR","NYO"),("UK100","ARB"),("USDJPY","ARB")]

FEAT_COLS_EXT = FEAT_COLS + ["rolling_wr"]

# ── Rebuild dataset ────────────────────────────────────────────────────────────
print("Loading all 9 streams …")
all_records = []
for sym, arch in ALL9:
    m15, mtf = load_m15_mtf(sym)
    fn = {"ARB":extract_arb,"NYO":extract_nyo,"MOM":extract_mom}[arch]
    trades = fn(m15, mtf)
    recs = extract_features(m15, mtf, trades, sym, arch)
    all_records.extend(recs)
    print(f"  {sym}-{arch}: {len(trades)}")

df = pd.DataFrame(all_records).sort_values("entry_t").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])
df["rolling_wr"] = df["label"].shift(1).rolling(10,min_periods=3).mean().fillna(0.5)

# ── Train/OOS split ────────────────────────────────────────────────────────────
dates = sorted(df["date"].unique())
cut   = dates[int(len(dates)*0.70)]
IS    = df[df["date"] <  cut]
OOS   = df[df["date"] >= cut].copy()

# ── Train model ───────────────────────────────────────────────────────────────
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
X_is = IS[FEAT_COLS_EXT].fillna(0).values; y_is = IS["label"].values
scale_pos = (y_is==0).sum()/(y_is==1).sum()
xgb = XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.04,
                    subsample=0.8,colsample_bytree=0.7,
                    scale_pos_weight=scale_pos,eval_metric="logloss",
                    random_state=42,verbosity=0)
model = CalibratedClassifierCV(xgb,cv=5,method="isotonic")
print("Training model …")
model.fit(X_is, y_is)
OOS["prob"] = model.predict_proba(OOS[FEAT_COLS_EXT].fillna(0).values)[:,1]

THRESH = 0.35
filt = OOS[OOS["prob"] >= THRESH].copy()
trades_f = filt.to_dict("records")
Rs  = np.array([t["R"] for t in trades_f])
n   = len(trades_f)
d0  = filt["date"].min(); d1  = filt["date"].max()
span_mo = max((d1-d0).days/30.44, 0.1)
tpm = n / span_mo

# ── Simulate sequential equity curve ──────────────────────────────────────────
bal = INIT_BAL; peak = INIT_BAL
bal_curve = [INIT_BAL]
daily_pnl = {}
monthly_pnl = {}
daily_dd_max = 0.0

for t in trades_f:
    risk_amt = bal * RISK_PCT
    pnl_r    = risk_amt * t["R"]
    bal     += pnl_r
    peak     = max(peak, bal)
    date_key = pd.Timestamp(t["date"])
    mo_key   = date_key.to_period("M")
    daily_pnl[date_key]   = daily_pnl.get(date_key, 0) + pnl_r
    monthly_pnl[mo_key]   = monthly_pnl.get(mo_key, 0) + pnl_r
    bal_curve.append(bal)

bal_arr   = np.array(bal_curve)
peak_arr  = np.maximum.accumulate(bal_arr)
dd_arr    = (peak_arr - bal_arr) / INIT_BAL * 100
max_dd    = dd_arr.max()

wins  = Rs[Rs > 0]; losses = Rs[Rs < 0]; timeouts = Rs[(Rs >= -0.99) & (Rs < 0.01)]
wr    = (Rs > 0).mean() * 100
avg_w = wins.mean()   if len(wins)   > 0 else 0
avg_l = losses.mean() if len(losses) > 0 else 0
pf    = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else np.inf

# Sharpe (annualised, assuming 252 trading days, ~2.6 trades/mo)
monthly_arr = np.array(list(monthly_pnl.values()))
ann_ret  = monthly_arr.mean() * 12 / INIT_BAL * 100
ann_vol  = monthly_arr.std()  * np.sqrt(12) / INIT_BAL * 100
sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0

calmar   = ann_ret / max_dd if max_dd > 0 else 0
exp_R    = Rs.mean()

# Consecutive wins/losses
consec_w = consec_l = cur_w = cur_l = 0
for r in Rs:
    if r > 0: cur_w += 1; cur_l = 0; consec_w = max(consec_w, cur_w)
    else:     cur_l += 1; cur_w = 0; consec_l = max(consec_l, cur_l)

# ── Print full report ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  FULL PERFORMANCE REPORT — ML-Filtered Strategy")
print(f"  Config: XGBoost confidence >{THRESH:.0%}, risk {RISK_PCT*100:.2f}%/trade")
print(f"{'='*70}")

print(f"\n  ── Dataset ──────────────────────────────────────────────────────")
print(f"  Train period : {IS['date'].min().date()} → {IS['date'].max().date()}")
print(f"  Test period  : {d0.date()} → {d1.date()}  ({span_mo:.1f} months)")
print(f"  Instruments  : {filt['sym'].nunique()}  |  Archetypes: {filt['arch'].nunique()}")
print(f"  Total trades : {n}  ({tpm:.1f}/month)")

print(f"\n  ── Trade Statistics ─────────────────────────────────────────────")
print(f"  Win rate         : {wr:.1f}%  ({(Rs>0).sum()} W / {(Rs<0).sum()} L / {((Rs>=-0.99)&(Rs<0.01)).sum()} T)")
print(f"  Avg win (R)      : +{avg_w:.3f}")
print(f"  Avg loss (R)     : {avg_l:.3f}")
print(f"  Profit factor    : {pf:.2f}")
print(f"  Expected R/trade : {exp_R:+.4f}")
print(f"  TP hit rate      : {(filt['result']=='tp').mean()*100:.1f}%")
print(f"  SL hit rate      : {(filt['result']=='sl').mean()*100:.1f}%")
print(f"  Timeout rate     : {(filt['result']=='timeout').mean()*100:.1f}%")
print(f"  Max consec wins  : {consec_w}")
print(f"  Max consec losses: {consec_l}")

print(f"\n  ── Equity Curve (1% risk, $10k start) ──────────────────────────")
final_bal = bal_curve[-1]
total_pnl = final_bal - INIT_BAL
print(f"  Starting balance : ${INIT_BAL:>10,.2f}")
print(f"  Final balance    : ${final_bal:>10,.2f}  ({total_pnl/INIT_BAL*100:+.2f}%)")
print(f"  Total P&L        : ${total_pnl:>+10,.2f}")
print(f"  Max drawdown     : {max_dd:.2f}%  (${max_dd/100*INIT_BAL:.0f})")
print(f"  Avg monthly P&L  : ${monthly_arr.mean():>+8,.2f}")
print(f"  Best month       : ${max(monthly_pnl.values()):>+8,.2f}")
print(f"  Worst month      : ${min(monthly_pnl.values()):>+8,.2f}")
print(f"  Profitable months: {sum(v>0 for v in monthly_pnl.values())}/{len(monthly_pnl)}")

print(f"\n  ── Risk-Adjusted Metrics ───────────────────────────────────────")
print(f"  Annualised return: {ann_ret:+.1f}%")
print(f"  Annualised vol   : {ann_vol:.1f}%")
print(f"  Sharpe ratio     : {sharpe:.2f}")
print(f"  Calmar ratio     : {calmar:.2f}")

print(f"\n  ── Monthly Breakdown ────────────────────────────────────────────")
print(f"  {'Month':<10}  {'Trades':>6}  {'WR':>6}  {'P&L':>10}  {'Balance':>11}  {'DD%':>6}")
bal_mo = INIT_BAL
for mo, pnl in sorted(monthly_pnl.items()):
    mo_t = filt[filt["date"].dt.to_period("M") == mo]
    mo_wr = (np.array([t["R"] for t in mo_t.to_dict("records")]) > 0).mean()*100 if len(mo_t) else 0
    bal_mo += pnl
    dd_pct = (INIT_BAL - bal_mo)/INIT_BAL*100 if bal_mo < INIT_BAL else 0
    bar = "▓"*int(abs(pnl)/50) if pnl >= 0 else "░"*int(abs(pnl)/50)
    sign = "+" if pnl >= 0 else ""
    print(f"  {str(mo):<10}  {len(mo_t):>6}  {mo_wr:>5.0f}%  "
          f"${pnl:>+8,.0f}  ${bal_mo:>9,.0f}  {dd_pct:>5.1f}%  {bar[:20]}")

print(f"\n  ── Per-Stream Breakdown ────────────────────────────────────────")
print(f"  {'Stream':<14}  {'Trades':>6}  {'WR':>6}  {'expR':>7}  {'Contrib $':>10}")
for (sym,arch), grp in filt.groupby(["sym","arch"]):
    Rs_g = np.array([t["R"] for t in grp.to_dict("records")])
    contrib = sum(INIT_BAL * RISK_PCT * r for r in Rs_g)
    print(f"  {sym+'-'+arch:<14}  {len(grp):>6}  {(Rs_g>0).mean()*100:>5.0f}%  "
          f"{Rs_g.mean():>+7.3f}  ${contrib:>+9,.0f}")

print(f"\n  ── Monte Carlo Challenge Simulation ────────────────────────────")
print(f"  {'Risk%':>6}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}  {'P10.Mo':>7}  {'P90.Mo':>7}")
for risk in [0.0075, 0.0100, 0.0125, 0.0150, 0.0200]:
    Rs_arr = np.array([t["R"] for t in trades_f])
    rng = np.random.default_rng(42)
    draw = max(n*5, 400); n_sim = 8000
    pc=bc=0; t2p=[]
    for _ in range(n_sim):
        seq = rng.choice(Rs_arr, size=draw, replace=True)
        bal2=peak2=INIT_BAL
        for k,r in enumerate(seq):
            bal2 += bal2*risk*r; peak2=max(peak2,bal2)
            if bal2>=INIT_BAL*1.08: pc+=1; t2p.append(k+1); break
            if bal2<=INIT_BAL*0.90 or peak2-bal2>=INIT_BAL*0.10: bc+=1; break
    med = float(np.median(t2p))/tpm if t2p else np.nan
    p10 = float(np.percentile(t2p,10))/tpm if t2p else np.nan
    p90 = float(np.percentile(t2p,90))/tpm if t2p else np.nan
    flag = " ***" if bc/n_sim*100<=5.0 and med<=3.5 else ""
    print(f"  {risk*100:>5.2f}%  {pc/n_sim*100:>6.1f}%  {bc/n_sim*100:>5.1f}%  "
          f"{med:>6.1f}  {p10:>6.1f}  {p90:>6.1f}{flag}")

print(f"\n  ── The5ers Challenge Simulation (sequential, actual order) ────")
bal=INIT_BAL; peak=INIT_BAL; target=INIT_BAL*1.08; bust=INIT_BAL*0.90
passed=False; busted=False; dd_breach=False
for k, t in enumerate(trades_f):
    risk_amt = bal*RISK_PCT; pnl_r = risk_amt*t["R"]
    bal+=pnl_r; peak=max(peak,bal)
    daily_dd = (INIT_BAL - bal)/INIT_BAL
    if bal>=target and not passed:
        print(f"  ✅ Target hit on trade #{k+1}  ({t['date']})  Balance: ${bal:,.0f}")
        passed=True; break
    if bal<=bust:
        print(f"  ❌ Balance bust on trade #{k+1}  ({t['date']})  Balance: ${bal:,.0f}")
        busted=True; break
    if peak-bal>=INIT_BAL*0.10:
        print(f"  ❌ Max DD bust on trade #{k+1}  ({t['date']})  Balance: ${bal:,.0f}")
        busted=True; break
if not passed and not busted:
    print(f"  ⏳ Not resolved — final balance ${bal:,.0f}  ({(bal/INIT_BAL-1)*100:+.2f}%)")

print(f"\n{'='*70}\n")
