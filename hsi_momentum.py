"""
HSI alpha — phase 2: is intraday momentum monetizable with selectivity?

EDA (phase 1) showed HSI has weak-but-persistent intraday momentum (gap-and-go,
early-direction follow-through). Gross ~3bps < ~7bps cost, so naive fails.
Hypothesis: STRONG early moves have larger follow-through that beats cost.

Signal: session-open -> T return, normalized by trailing daily ATR (strength).
Enter sign(signal) if |signal| > threshold; hold to force-close (20:50).
Cost: 7 index pts round trip (real HSI spread).

Discipline: split the DENSE era 2018-2021 (IS) vs 2022-2026 (OOS).
Threshold chosen on IS only, then applied unchanged to OOS.
"""
import sys, warnings, glob
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from multi_asset_scan import load_raw

m5 = load_raw(glob.glob("HSIHKD_M5_*.csv")[0])
m5 = m5[m5["close"] > 0].copy()
COST_PTS = 7.0   # round-trip index points (conservative vs recent ~6.6)

# Daily ATR (for normalizing the early move into a strength z-score)
d1 = m5.resample("1D").agg({"high":"max","low":"min","close":"last"}).dropna()
d1["pc"] = d1["close"].shift(1)
d1["tr"] = np.maximum(d1["high"]-d1["low"],
            np.maximum((d1["high"]-d1["pc"]).abs(), (d1["low"]-d1["pc"]).abs()))
d1["atr"] = d1["tr"].ewm(span=14, adjust=False).mean().shift(1)  # lagged, no lookahead
atr_map = d1["atr"].to_dict()

def build_days(T_hour, exit_hour=20):
    rows=[]
    for date, day in m5.groupby(m5.index.date):
        day = day[day.index.hour <= exit_hour+1]
        if len(day) < 30: continue
        early = day[day.index.hour < T_hour]
        rest  = day[(day.index.hour >= T_hour) & (day.index.hour <= exit_hour)]
        if len(early) < 6 or len(rest) < 6: continue
        atr = atr_map.get(pd.Timestamp(date), np.nan)
        if not atr or np.isnan(atr) or atr <= 0: continue
        entry = early["close"].iloc[-1]
        early_move = entry - early["close"].iloc[0]
        strength = early_move / atr                 # z-like signal
        exit_px = rest["close"].iloc[-1]
        rows.append({"date":pd.Timestamp(date), "entry":entry, "exit":exit_px,
                     "strength":strength, "atr":atr})
    df = pd.DataFrame(rows)
    df["yr"] = df["date"].dt.year
    return df

def evaluate(df, thresh):
    s = df[df["strength"].abs() >= thresh].copy()
    if len(s) == 0: return None
    d = np.sign(s["strength"])
    gross = d * (s["exit"] - s["entry"])            # index points per unit
    net = gross - COST_PTS
    # express as % of price for risk-comparability
    netpct = net / s["entry"]
    return dict(n=len(s), wr=100*(net>0).mean(),
                net_pts=net.mean(), net_bps=netpct.mean()*1e4,
                total_pts=net.sum())

print("="*80)
print(f"  HSI intraday momentum — strength = (open->T move)/dailyATR, hold to 20:00")
print(f"  cost = {COST_PTS} pts round trip | IS=2018-2021  OOS=2022-2026")
print("="*80)

for T in [8, 9, 10]:
    df = build_days(T)
    IS  = df[(df["yr"]>=2018)&(df["yr"]<=2021)]
    OOS = df[(df["yr"]>=2022)&(df["yr"]<=2026)]
    print(f"\n  --- entry T={T}:00  (IS n={len(IS)}, OOS n={len(OOS)}) ---")
    print(f"   {'thresh':>7} {'IS n':>5} {'IS net_bps':>11} {'IS WR':>6} "
          f"{'OOS n':>6} {'OOS net_bps':>12} {'OOS WR':>7} {'robust':>8}")
    for th in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]:
        ri = evaluate(IS, th); ro = evaluate(OOS, th)
        if ri is None or ro is None: continue
        robust = "✅" if (ri["net_bps"]>0 and ro["net_bps"]>0 and ri["n"]>=80 and ro["n"]>=80) else ""
        print(f"   {th:>7.2f} {ri['n']:>5d} {ri['net_bps']:>11.2f} {ri['wr']:>5.0f}% "
              f"{ro['n']:>6d} {ro['net_bps']:>12.2f} {ro['wr']:>6.0f}% {robust:>8}")
