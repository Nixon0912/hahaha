"""
Money Printer Hunt — comprehensive alpha search across all available data.

Instruments tested:
  1. XAUUSD H1  (2009-2026, 17yr) — trend + MR archetypes
  2. HSI Weekly  (resampled from M5, 2009-2026) — weekly momentum + MR
  3. Multi-asset daily blend (HSI + XAUUSD) — diversification lift

Signals tested per instrument:
  A. TSMOM (time-series momentum) — lookbacks 20/60/120/250 bars
  B. Z-score MR — n20/30/40, fade |z|>=1.5/2.0
  C. Dual-timeframe: weekly trend filter + daily/H4 MR entries
  D. Gap-day: fade >0.3% overnight gap, hold 1 day
  E. Volatility regime switch: MR when vol<median, trend when vol>median

All: no-lookahead, cost-adjusted, IS70/OOS30 split, vol-targeted to 15%.
Best curve plotted and saved.
"""
import sys, warnings, glob
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

ann_xau = 252 * 24   # H1 bars per year approx (not exactly, but for Sharpe scaling)
ann_d1  = 252
HALF_XAU = 0.20      # ~20 cents per side on gold (spread)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")

# XAUUSD H1
h1_file = glob.glob("XAUUSD_H1_*.csv")[0]
xau_h1 = load_raw(h1_file); xau_h1 = xau_h1[xau_h1["close"]>0].copy()
xau_h1["ret"] = xau_h1["close"].pct_change().fillna(0)
print(f"  XAUUSD H1: {len(xau_h1)} bars  {xau_h1.index[0].date()} → {xau_h1.index[-1].date()}")

# HSI daily (from M5)
m5_file = glob.glob("HSIHKD_M5_*.csv")[0]
m5 = load_raw(m5_file); m5 = m5[m5["close"]>0]
hsi_d1 = m5.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
hsi_d1["ret"] = hsi_d1["close"].pct_change().fillna(0)
print(f"  HSI daily: {len(hsi_d1)} bars  {hsi_d1.index[0].date()} → {hsi_d1.index[-1].date()}")

# HSI weekly
hsi_w = m5.resample("1W").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
hsi_w["ret"] = hsi_w["close"].pct_change().fillna(0)
print(f"  HSI weekly: {len(hsi_w)} bars  {hsi_w.index[0].date()} → {hsi_w.index[-1].date()}")

SPLIT_FRAC = 0.70

# ── Generic backtest engine ──────────────────────────────────────────────────
def backtest(prices, sig, cost_per_unit, name, ann_periods, vol_target=0.15, vol_cap=4.0):
    """
    prices: Series (close prices), sig: Series (raw position signal, same index)
    cost_per_unit: Series (half-spread / price, already fractional)
    """
    ret = prices.pct_change().fillna(0)
    # vol-target
    rv = ret.rolling(max(int(ann_periods/12), 5)).std() * np.sqrt(ann_periods)
    scale = (vol_target / rv.replace(0, np.nan)).clip(upper=vol_cap).fillna(0)
    pos = sig * scale
    pos = pos.shift(1).fillna(0)
    turn = pos.diff().abs().fillna(pos.abs())
    net = (pos * ret - turn * cost_per_unit).fillna(0)

    n = len(net); split = int(n * SPLIT_FRAC)
    eq = (1 + net).cumprod()
    sh_all = net.mean() / net.std() * np.sqrt(ann_periods) if net.std() > 0 else 0
    net_is = net.iloc[:split]; net_oos = net.iloc[split:]
    sh_is  = net_is.mean()  / net_is.std()  * np.sqrt(ann_periods) if net_is.std()  > 0 else 0
    sh_oos = net_oos.mean() / net_oos.std() * np.sqrt(ann_periods) if net_oos.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    yrs = n / ann_periods
    cagr = eq.iloc[-1] ** (1/max(yrs, 0.5)) - 1 if eq.iloc[-1] > 0 else -1
    return dict(name=name, eq=eq, net=net, sh=sh_all, sh_is=sh_is, sh_oos=sh_oos,
                dd=dd, cagr=cagr, split=split, ann=ann_periods)

# ── XAUUSD H1 tests ──────────────────────────────────────────────────────────
print("\n=== XAUUSD H1 ===")
results_xau = []
prices_xau = xau_h1["close"]
cost_xau = HALF_XAU / prices_xau   # fractional half-spread

# A. TSMOM at various lookbacks
for L in [24, 48, 120, 240, 504, 1008]:
    sig = np.sign(prices_xau.pct_change(L)).fillna(0)
    r = backtest(prices_xau, sig, cost_xau, f"TSMOM {L}h", ann_xau)
    results_xau.append(r)

# B. Z-score MR
for n in [20, 48, 120]:
    ma = prices_xau.rolling(n).mean(); sd = prices_xau.rolling(n).std()
    z = (prices_xau - ma) / sd
    for entry in [1.5, 2.0, 2.5]:
        zv = z.values; out = np.zeros(len(zv)); cur = 0.0
        for i in range(len(zv)):
            if np.isnan(zv[i]): out[i]=cur; continue
            if cur == 0:
                if zv[i] >= entry: cur = -1
                elif zv[i] <= -entry: cur = 1
            else:
                if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur = 0
            out[i] = cur
        sig = pd.Series(out, index=prices_xau.index)
        r = backtest(prices_xau, sig, cost_xau, f"MR_z n{n} e{entry}", ann_xau)
        results_xau.append(r)

# C. Dual-timeframe: weekly trend filter on H1 MR
xau_weekly_trend = np.sign(prices_xau.resample("1W").last().pct_change(4)).reindex(
    prices_xau.index, method="ffill").fillna(0)
for n in [24, 48]:
    ma = prices_xau.rolling(n).mean(); sd = prices_xau.rolling(n).std()
    z = (prices_xau - ma) / sd
    zv = z.values; out = np.zeros(len(zv)); cur = 0.0
    for i in range(len(zv)):
        if np.isnan(zv[i]): out[i]=cur; continue
        if cur == 0:
            if zv[i] >= 1.5: cur = -1
            elif zv[i] <= -1.5: cur = 1
        else:
            if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur = 0
        out[i] = cur
    mr_sig = pd.Series(out, index=prices_xau.index)
    # blend: MR when trend agrees or neutral, trend always
    sig = 0.5*mr_sig + 0.5*xau_weekly_trend
    r = backtest(prices_xau, sig, cost_xau, f"Dual_w4_mr{n}", ann_xau)
    results_xau.append(r)

# D. Vol regime switch: trend when 20-bar vol > rolling median, MR otherwise
rv20 = prices_xau.pct_change().rolling(20).std() * np.sqrt(ann_xau)
vol_med = rv20.rolling(252*24).median()
high_vol = (rv20 > vol_med).astype(float)
tsmom_120 = np.sign(prices_xau.pct_change(120)).fillna(0)
ma48 = prices_xau.rolling(48).mean(); sd48 = prices_xau.rolling(48).std()
z48 = (prices_xau - ma48) / sd48
mr48_arr = np.zeros(len(z48)); cur = 0.0; zv=z48.values
for i in range(len(zv)):
    if np.isnan(zv[i]): mr48_arr[i]=cur; continue
    if cur==0:
        if zv[i]>=2: cur=-1
        elif zv[i]<=-2: cur=1
    else:
        if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
    mr48_arr[i]=cur
mr48 = pd.Series(mr48_arr, index=prices_xau.index)
sig_regime = high_vol * tsmom_120 + (1 - high_vol) * mr48
r = backtest(prices_xau, sig_regime, cost_xau, "VolRegime_switch", ann_xau)
results_xau.append(r)

# E. Pure carry-style: always long (buy-and-hold vol-targeted) as baseline
sig_long = pd.Series(1.0, index=prices_xau.index)
r = backtest(prices_xau, sig_long, cost_xau, "BuyHold_VT", ann_xau)
results_xau.append(r)

results_xau.sort(key=lambda x: -min(x["sh_is"], x["sh_oos"]))

print(f"  {'strategy':<22}{'CAGR':>8}{'Sharpe':>8}{'IS Sh':>8}{'OOS Sh':>8}{'maxDD':>8}")
print("  " + "-"*62)
for r in results_xau[:15]:
    flag = " ✅" if r["sh_is"] > 0.4 and r["sh_oos"] > 0.4 else ""
    print(f"  {r['name']:<22}{r['cagr']*100:>7.1f}%{r['sh']:>8.2f}{r['sh_is']:>8.2f}{r['sh_oos']:>8.2f}{r['dd']*100:>7.1f}%{flag}")

# ── HSI Weekly tests ──────────────────────────────────────────────────────────
print("\n=== HSI Weekly ===")
results_hsiw = []
prices_w = hsi_w["close"]
HALF_HSI = 3.5
cost_w = HALF_HSI / prices_w
ann_w = 52

for L in [4, 8, 13, 26, 52]:
    sig = np.sign(prices_w.pct_change(L)).fillna(0)
    r = backtest(prices_w, sig, cost_w, f"TSMOM {L}wk", ann_w)
    results_hsiw.append(r)

for n in [4, 8, 13]:
    ma = prices_w.rolling(n).mean(); sd = prices_w.rolling(n).std()
    z = (prices_w - ma) / sd
    for entry in [1.5, 2.0]:
        zv = z.values; out = np.zeros(len(zv)); cur = 0.0
        for i in range(len(zv)):
            if np.isnan(zv[i]): out[i]=cur; continue
            if cur==0:
                if zv[i]>=entry: cur=-1
                elif zv[i]<=-entry: cur=1
            else:
                if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
            out[i]=cur
        sig = pd.Series(out, index=prices_w.index)
        r = backtest(prices_w, sig, cost_w, f"MR n{n} e{entry}", ann_w)
        results_hsiw.append(r)

# blend best weekly TSMOM + MR
sig_tsmom8 = np.sign(prices_w.pct_change(8)).fillna(0)
ma8=prices_w.rolling(8).mean(); sd8=prices_w.rolling(8).std(); z8=(prices_w-ma8)/sd8
mr8_arr=np.zeros(len(z8)); cur=0.0; zv=z8.values
for i in range(len(zv)):
    if np.isnan(zv[i]): mr8_arr[i]=cur; continue
    if cur==0:
        if zv[i]>=2: cur=-1
        elif zv[i]<=-2: cur=1
    else:
        if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
    mr8_arr[i]=cur
mr8=pd.Series(mr8_arr,index=prices_w.index)
blend_w = 0.5*sig_tsmom8 + 0.5*mr8
r = backtest(prices_w, blend_w, cost_w, "Blend_8wk", ann_w)
results_hsiw.append(r)

results_hsiw.sort(key=lambda x: -min(x["sh_is"], x["sh_oos"]))
print(f"  {'strategy':<22}{'CAGR':>8}{'Sharpe':>8}{'IS Sh':>8}{'OOS Sh':>8}{'maxDD':>8}")
print("  " + "-"*62)
for r in results_hsiw[:10]:
    flag = " ✅" if r["sh_is"] > 0.3 and r["sh_oos"] > 0.3 else ""
    print(f"  {r['name']:<22}{r['cagr']*100:>7.1f}%{r['sh']:>8.2f}{r['sh_is']:>8.2f}{r['sh_oos']:>8.2f}{r['dd']*100:>7.1f}%{flag}")

# ── HSI Gap-day fade (daily) ─────────────────────────────────────────────────
print("\n=== HSI Gap-day strategies (daily) ===")
results_gap = []
prices_d = hsi_d1["close"]
cost_d = HALF_HSI / prices_d
gap = hsi_d1["open"] / hsi_d1["close"].shift(1) - 1

for thresh in [0.003, 0.005, 0.008, 0.01]:
    # fade gaps above thresh
    sig = pd.Series(0.0, index=hsi_d1.index)
    sig[gap > thresh] = -1.0
    sig[gap < -thresh] = 1.0
    r = backtest(prices_d, sig, cost_d, f"GapFade >{thresh*100:.1f}%", ann_d1)
    results_gap.append(r)
    # follow gaps above thresh
    sig2 = -sig
    r2 = backtest(prices_d, sig2, cost_d, f"GapFollow >{thresh*100:.1f}%", ann_d1)
    results_gap.append(r2)

results_gap.sort(key=lambda x: -min(x["sh_is"], x["sh_oos"]))
print(f"  {'strategy':<26}{'CAGR':>8}{'Sharpe':>8}{'IS Sh':>8}{'OOS Sh':>8}{'maxDD':>8}")
print("  " + "-"*66)
for r in results_gap[:10]:
    flag = " ✅" if r["sh_is"] > 0.3 and r["sh_oos"] > 0.3 else ""
    print(f"  {r['name']:<26}{r['cagr']*100:>7.1f}%{r['sh']:>8.2f}{r['sh_is']:>8.2f}{r['sh_oos']:>8.2f}{r['dd']*100:>7.1f}%{flag}")

# ── XAUUSD daily from H1 resampled ──────────────────────────────────────────
print("\n=== XAUUSD Daily (from H1) ===")
xau_d1 = xau_h1.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
xau_d1 = xau_d1[xau_d1["close"] > 0]
prices_xaud = xau_d1["close"]
cost_xaud = HALF_XAU / prices_xaud
results_xaud = []

# multi-day TSMOM
for L in [10, 20, 50, 100, 200]:
    sig = np.sign(prices_xaud.pct_change(L)).fillna(0)
    r = backtest(prices_xaud, sig, cost_xaud, f"TSMOM {L}d", ann_d1)
    results_xaud.append(r)

# z-score MR daily
for n in [10, 20, 30]:
    ma = prices_xaud.rolling(n).mean(); sd = prices_xaud.rolling(n).std()
    z = (prices_xaud - ma) / sd
    for entry in [1.5, 2.0]:
        zv=z.values; out=np.zeros(len(zv)); cur=0.0
        for i in range(len(zv)):
            if np.isnan(zv[i]): out[i]=cur; continue
            if cur==0:
                if zv[i]>=entry: cur=-1
                elif zv[i]<=-entry: cur=1
            else:
                if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
            out[i]=cur
        sig=pd.Series(out,index=prices_xaud.index)
        r = backtest(prices_xaud, sig, cost_xaud, f"MR n{n} e{entry}", ann_d1)
        results_xaud.append(r)

# blend MR + TSMOM daily
sig_t50 = np.sign(prices_xaud.pct_change(50)).fillna(0)
ma20=prices_xaud.rolling(20).mean(); sd20=prices_xaud.rolling(20).std(); z20=(prices_xaud-ma20)/sd20
mr20_arr=np.zeros(len(z20)); cur=0.0; zv=z20.values
for i in range(len(zv)):
    if np.isnan(zv[i]): mr20_arr[i]=cur; continue
    if cur==0:
        if zv[i]>=2: cur=-1
        elif zv[i]<=-2: cur=1
    else:
        if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
    mr20_arr[i]=cur
mr20=pd.Series(mr20_arr,index=prices_xaud.index)
for w in [0.3, 0.5, 0.7]:
    sig_b = w*mr20 + (1-w)*sig_t50
    r = backtest(prices_xaud, sig_b, cost_xaud, f"Blend MR{w:.0%}+T{1-w:.0%}", ann_d1)
    results_xaud.append(r)

results_xaud.sort(key=lambda x: -min(x["sh_is"], x["sh_oos"]))
print(f"  {'strategy':<26}{'CAGR':>8}{'Sharpe':>8}{'IS Sh':>8}{'OOS Sh':>8}{'maxDD':>8}")
print("  " + "-"*66)
for r in results_xaud[:12]:
    flag = " ✅" if r["sh_is"] > 0.4 and r["sh_oos"] > 0.4 else ""
    print(f"  {r['name']:<26}{r['cagr']*100:>7.1f}%{r['sh']:>8.2f}{r['sh_is']:>8.2f}{r['sh_oos']:>8.2f}{r['dd']*100:>7.1f}%{flag}")

# ── Multi-asset: XAUUSD daily + HSI daily blend ───────────────────────────────
print("\n=== Multi-asset daily (XAUUSD + HSI) ===")
# align to common dates
common_idx = xau_d1.index.intersection(hsi_d1.index)
xau_c = xau_d1.loc[common_idx, "close"]
hsi_c = hsi_d1.loc[common_idx, "close"]
xau_ret = xau_c.pct_change().fillna(0)
hsi_ret = hsi_c.pct_change().fillna(0)
print(f"  Common dates: {len(common_idx)}  {common_idx[0].date()} → {common_idx[-1].date()}")

# vol-target each individually, then blend
rv_xau = xau_ret.rolling(20).std() * np.sqrt(ann_d1)
rv_hsi = hsi_ret.rolling(20).std() * np.sqrt(ann_d1)

def vt_series(sig, rv, target=0.15, cap=4.0):
    return sig * (target / rv.replace(0, np.nan)).clip(upper=cap).fillna(0)

# best XAUUSD signal (TSMOM 50d or whatever wins above)
best_xau_sig = np.sign(xau_c.pct_change(50)).fillna(0)
# best HSI signal (MR+trend blend from hsi_final.py)
nn=30; ma_hsi=hsi_c.rolling(nn).mean(); sd_hsi=hsi_c.rolling(nn).std()
z_hsi=(hsi_c-ma_hsi)/sd_hsi; zv=z_hsi.values; mr_arr=np.zeros(len(z_hsi)); cur=0.0
for i in range(len(zv)):
    if np.isnan(zv[i]): mr_arr[i]=cur; continue
    if cur==0:
        if zv[i]>=2: cur=-1
        elif zv[i]<=-2: cur=1
    else:
        if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
    mr_arr[i]=cur
mr_hsi=pd.Series(mr_arr,index=hsi_c.index)
trend_hsi=np.sign(hsi_c.pct_change(100)).fillna(0)
blend_hsi=0.5*mr_hsi+0.5*trend_hsi

results_ma = []
for xau_w in [0.0, 0.25, 0.5, 0.75, 1.0]:
    hsi_w_v = 1.0 - xau_w
    # vol-target each leg separately
    xau_pos_vt = vt_series(best_xau_sig, rv_xau)
    hsi_pos_vt = vt_series(blend_hsi, rv_hsi)
    # combined P&L
    xau_pos_vt_sh = xau_pos_vt.shift(1).fillna(0)
    hsi_pos_vt_sh = hsi_pos_vt.shift(1).fillna(0)
    xau_turn = xau_pos_vt_sh.diff().abs().fillna(xau_pos_vt_sh.abs())
    hsi_turn = hsi_pos_vt_sh.diff().abs().fillna(hsi_pos_vt_sh.abs())
    xau_cost_frac = HALF_XAU / xau_c
    hsi_cost_frac = HALF_HSI / hsi_c
    net = xau_w*(xau_pos_vt_sh*xau_ret - xau_turn*xau_cost_frac) + \
          hsi_w_v*(hsi_pos_vt_sh*hsi_ret - hsi_turn*hsi_cost_frac)
    net = net.fillna(0)
    split = int(len(net) * SPLIT_FRAC)
    eq = (1+net).cumprod()
    sh = net.mean()/net.std()*np.sqrt(ann_d1) if net.std()>0 else 0
    sh_is = net.iloc[:split].mean()/net.iloc[:split].std()*np.sqrt(ann_d1) if net.iloc[:split].std()>0 else 0
    sh_oos = net.iloc[split:].mean()/net.iloc[split:].std()*np.sqrt(ann_d1) if net.iloc[split:].std()>0 else 0
    dd = (eq/eq.cummax()-1).min()
    yrs = len(net)/ann_d1
    cagr = eq.iloc[-1]**(1/max(yrs,0.5))-1 if eq.iloc[-1]>0 else -1
    name = f"XAU{xau_w:.0%}+HSI{hsi_w_v:.0%}"
    results_ma.append(dict(name=name,eq=eq,net=net,sh=sh,sh_is=sh_is,sh_oos=sh_oos,dd=dd,cagr=cagr,split=split))
    flag = " ✅" if sh_is>0.4 and sh_oos>0.4 else ""
    print(f"  {name:<20}{cagr*100:>7.1f}%{sh:>8.2f}{sh_is:>8.2f}{sh_oos:>8.2f}{dd*100:>7.1f}%{flag}")

# ── Pick global winners and plot ──────────────────────────────────────────────
print("\n" + "="*70)
print("  GLOBAL TOP PICKS (by min(IS,OOS) Sharpe)")
print("="*70)
all_results = []
for r in results_xau:   r["group"]="XAU H1";  all_results.append(r)
for r in results_hsiw:  r["group"]="HSI Wkly"; all_results.append(r)
for r in results_gap:   r["group"]="HSI Gap";  all_results.append(r)
for r in results_xaud:  r["group"]="XAU D1";   all_results.append(r)
for r in results_ma:    r["group"]="MultiAst";  all_results.append(r)

all_results.sort(key=lambda x: -min(x["sh_is"], x["sh_oos"]))
print(f"  {'group':<10}{'strategy':<26}{'CAGR':>8}{'Sharpe':>8}{'IS Sh':>8}{'OOS Sh':>8}{'maxDD':>8}")
print("  " + "-"*78)
for r in all_results[:20]:
    flag = " ✅" if r["sh_is"]>0.4 and r["sh_oos"]>0.4 else ""
    print(f"  {r['group']:<10}{r['name']:<26}{r['cagr']*100:>7.1f}%{r['sh']:>8.2f}{r['sh_is']:>8.2f}{r['sh_oos']:>8.2f}{r['dd']*100:>7.1f}%{flag}")

# ── Plot top 4 + multi-asset ──────────────────────────────────────────────────
top4 = all_results[:4]
best_ma = max(results_ma, key=lambda x: min(x["sh_is"], x["sh_oos"]))

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

ax = axes[0]
colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
for i, r in enumerate(top4):
    ax.plot(r["eq"].index, r["eq"].values, lw=1.4, color=colors[i],
            label=f"[{r['group']}] {r['name']}  Sh {r['sh']:.2f} (OOS {r['sh_oos']:.2f}) DD {r['dd']*100:.0f}%")
    if "split" in r:
        ax.axvline(r["eq"].index[r["split"]], color=colors[i], ls=":", alpha=0.4)
ax.set_yscale("log"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
ax.set_title("Money Printer Hunt — Top 4 vol-targeted strategies (15% ann vol target)")

ax2 = axes[1]
ax2.plot(best_ma["eq"].index, best_ma["eq"].values, lw=1.5, color="#2ca02c",
         label=f"Best multi-asset: {best_ma['name']}  Sh {best_ma['sh']:.2f} (OOS {best_ma['sh_oos']:.2f}) DD {best_ma['dd']*100:.0f}%")
if "split" in best_ma:
    ax2.axvline(best_ma["eq"].index[best_ma["split"]], color="red", ls="--", alpha=0.6, label="IS/OOS split")
ax2.set_yscale("log"); ax2.legend(loc="upper left", fontsize=9); ax2.grid(alpha=0.3)
ax2.set_title("Best multi-asset blend (XAUUSD + HSI, vol-targeted independently)")

plt.tight_layout()
plt.savefig("money_printer_equity.png", dpi=120)
print("\n  saved -> money_printer_equity.png")

# ── Year-by-year for the global best ─────────────────────────────────────────
best = all_results[0]
print(f"\n  Year-by-year [{best['group']}] {best['name']}:")
yby = best["net"].groupby(best["eq"].index.year).apply(lambda s: (1+s).prod()-1)
print("  " + "  ".join(f"{y}:{v*100:+.1f}%" for y,v in yby.items()))
pos = (yby > 0).sum()
print(f"  Positive years: {pos}/{len(yby)}")
