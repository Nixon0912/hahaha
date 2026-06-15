"""
XAUUSD Money Printer — daily TSMOM + MR blend, vol-targeted.

Winner: XAUUSD daily, 70% TSMOM-50d + 30% MR-z20, vol-targeted to 15%.
Result: CAGR 6.9%, Sharpe 0.72, IS 0.62, OOS 0.95, maxDD -18.9%.
Compares to HSI blend (CAGR 1.9%, Sharpe 0.38).

Also shows: pure TSMOM-50d (CAGR 11.4%, Sharpe 0.73, DD -29.7%)
and multi-asset combo XAU+HSI (CAGR 4.7-7.3% range, OOS Sharpe 0.72-0.79).
"""
import sys, warnings, glob
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

# ── Data ─────────────────────────────────────────────────────────────────────
h1 = load_raw(glob.glob("XAUUSD_H1_*.csv")[0]); h1 = h1[h1["close"]>0]
d1 = h1.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
d1 = d1[d1["close"]>0].copy()
d1["ret"] = d1["close"].pct_change().fillna(0)

m5 = load_raw(glob.glob("HSIHKD_M5_*.csv")[0]); m5 = m5[m5["close"]>0]
hsi = m5.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
hsi["ret"] = hsi["close"].pct_change().fillna(0)

HALF_XAU = 0.20   # ~$0.20 half-spread per side
HALF_HSI = 3.5
ann = 252
SPLIT = int(len(d1) * 0.70)
TARGET_VOL = 0.15
VOL_CAP = 4.0

realvol_xau = d1["ret"].rolling(20).std() * np.sqrt(ann)
realvol_hsi = hsi["ret"].rolling(20).std() * np.sqrt(ann)

def vt(pos, rv, target=TARGET_VOL, cap=VOL_CAP):
    return pos * (target / rv.replace(0, np.nan)).clip(upper=cap).fillna(0)

def net_of(pos, prices, half_spread, rv):
    pos_vt = vt(pos, rv)
    pos_s = pos_vt.shift(1).fillna(0)
    turn = pos_s.diff().abs().fillna(pos_s.abs())
    cost = turn * (half_spread / prices)
    return (pos_s * prices.pct_change().fillna(0) - cost).fillna(0)

def stats(net):
    eq = (1+net).cumprod()
    sh = net.mean()/net.std()*np.sqrt(ann) if net.std()>0 else 0
    sh_is = net.iloc[:SPLIT].mean()/net.iloc[:SPLIT].std()*np.sqrt(ann) if net.iloc[:SPLIT].std()>0 else 0
    sh_oos = net.iloc[SPLIT:].mean()/net.iloc[SPLIT:].std()*np.sqrt(ann) if net.iloc[SPLIT:].std()>0 else 0
    dd = (eq/eq.cummax()-1).min()
    cagr = eq.iloc[-1]**(ann/len(net))-1 if eq.iloc[-1]>0 else -1
    return eq, sh, sh_is, sh_oos, dd, cagr

# ── XAUUSD signals ───────────────────────────────────────────────────────────
# TSMOM 50d
tsmom50 = np.sign(d1["close"].pct_change(50)).fillna(0)

# MR z20
n=20; ma=d1["close"].rolling(n).mean(); sd=d1["close"].rolling(n).std(); z=(d1["close"]-ma)/sd
zv=z.values; mr_arr=np.zeros(len(zv)); cur=0.0
for i in range(len(zv)):
    if np.isnan(zv[i]): mr_arr[i]=cur; continue
    if cur==0:
        if zv[i]>=2: cur=-1
        elif zv[i]<=-2: cur=1
    else:
        if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
    mr_arr[i]=cur
mr20 = pd.Series(mr_arr, index=d1.index)

blend_xau = 0.70 * tsmom50 + 0.30 * mr20

# ── HSI blend (from hsi_final.py) ────────────────────────────────────────────
nn=30; mh=hsi["close"].rolling(nn).mean(); sh=hsi["close"].rolling(nn).std(); zh=(hsi["close"]-mh)/sh
zv2=zh.values; mr_h=np.zeros(len(zv2)); cur=0.0
for i in range(len(zv2)):
    if np.isnan(zv2[i]): mr_h[i]=cur; continue
    if cur==0:
        if zv2[i]>=2: cur=-1
        elif zv2[i]<=-2: cur=1
    else:
        if (cur==-1 and zv2[i]<=0) or (cur==1 and zv2[i]>=0): cur=0
    mr_h[i]=cur
mr_hsi = pd.Series(mr_h, index=hsi.index)
trend_hsi = np.sign(hsi["close"].pct_change(100)).fillna(0)
blend_hsi = 0.5*mr_hsi + 0.5*trend_hsi

# ── Compute nets ──────────────────────────────────────────────────────────────
net_tsmom  = net_of(tsmom50,   d1["close"], HALF_XAU, realvol_xau)
net_blend  = net_of(blend_xau, d1["close"], HALF_XAU, realvol_xau)
net_hsi    = net_of(blend_hsi, hsi["close"], HALF_HSI, realvol_hsi)

# multi-asset XAU25+HSI75 (vol-target each independently)
common = d1.index.intersection(hsi.index)
xau_c = d1.loc[common]
hsi_c = hsi.loc[common]
rv_xau_c = realvol_xau.reindex(common)
rv_hsi_c = realvol_hsi.reindex(common)
bx = blend_xau.reindex(common)
bh = blend_hsi.reindex(common)
px = d1["close"].reindex(common)
ph = hsi["close"].reindex(common)

def net_ma(sig_x, sig_h, w_x, w_h):
    vt_x = vt(sig_x, rv_xau_c).shift(1).fillna(0)
    vt_h = vt(sig_h, rv_hsi_c).shift(1).fillna(0)
    tx = vt_x.diff().abs().fillna(vt_x.abs())
    th = vt_h.diff().abs().fillna(vt_h.abs())
    nx = vt_x * px.pct_change().fillna(0) - tx*(HALF_XAU/px)
    nh = vt_h * ph.pct_change().fillna(0) - th*(HALF_HSI/ph)
    return (w_x*nx + w_h*nh).fillna(0)

net_ma25 = net_ma(bx, bh, 0.25, 0.75)
net_ma50 = net_ma(bx, bh, 0.50, 0.50)

systems = {
    "XAU TSMOM-50d (pure)":      (net_tsmom,  d1.index, SPLIT),
    "XAU Blend T70+MR30":        (net_blend,  d1.index, SPLIT),
    "XAU25+HSI75 multi-asset":   (net_ma25,   common,   int(len(common)*0.70)),
    "XAU50+HSI50 multi-asset":   (net_ma50,   common,   int(len(common)*0.70)),
    "HSI Blend MR+Trend":        (net_hsi,    hsi.index, int(len(hsi)*0.70)),
}

print("="*82)
print(f"  {'System':<30}{'CAGR':>8}{'Sharpe':>8}{'IS Sh':>8}{'OOS Sh':>8}{'maxDD':>8}")
print("="*82)
for name, (net, idx, split) in systems.items():
    net = net.fillna(0)
    eq = (1+net).cumprod()
    sh = net.mean()/net.std()*np.sqrt(ann) if net.std()>0 else 0
    sh_is = net.iloc[:split].mean()/net.iloc[:split].std()*np.sqrt(ann) if net.iloc[:split].std()>0 else 0
    sh_oos = net.iloc[split:].mean()/net.iloc[split:].std()*np.sqrt(ann) if net.iloc[split:].std()>0 else 0
    dd = (eq/eq.cummax()-1).min()
    cagr = eq.iloc[-1]**(ann/len(net))-1 if eq.iloc[-1]>0 else -1
    print(f"  {name:<30}{cagr*100:>7.1f}%{sh:>8.2f}{sh_is:>8.2f}{sh_oos:>8.2f}{dd*100:>7.1f}%")

# ── Year-by-year for best ─────────────────────────────────────────────────────
print("\n  XAU Blend T70+MR30 year-by-year (vol-tgt 15%):")
net_b = net_blend.fillna(0)
eq_b = (1+net_b).cumprod()
by = net_b.groupby(d1.index.year).apply(lambda s: (1+s).prod()-1)
print("  " + "  ".join(f"{y}:{v*100:+.1f}%" for y,v in by.items()))
print(f"  Positive years: {(by>0).sum()}/{len(by)}")

print("\n  XAU TSMOM-50d year-by-year (vol-tgt 15%):")
net_t = net_tsmom.fillna(0)
by2 = net_t.groupby(d1.index.year).apply(lambda s: (1+s).prod()-1)
print("  " + "  ".join(f"{y}:{v*100:+.1f}%" for y,v in by2.items()))
print(f"  Positive years: {(by2>0).sum()}/{len(by2)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False,
                                 gridspec_kw={"height_ratios":[3,1]})

# equity curves (all on same base index where possible)
colors = {"XAU TSMOM-50d (pure)":"#ff7f0e",
          "XAU Blend T70+MR30":"#2ca02c",
          "XAU25+HSI75 multi-asset":"#9467bd",
          "XAU50+HSI50 multi-asset":"#17becf",
          "HSI Blend MR+Trend":"#aec7e8"}
lws    = {"XAU TSMOM-50d (pure)":1.0,
          "XAU Blend T70+MR30":1.8,
          "XAU25+HSI75 multi-asset":1.4,
          "XAU50+HSI50 multi-asset":1.0,
          "HSI Blend MR+Trend":0.9}
alphas = {"XAU TSMOM-50d (pure)":0.8,
          "XAU Blend T70+MR30":1.0,
          "XAU25+HSI75 multi-asset":0.85,
          "XAU50+HSI50 multi-asset":0.7,
          "HSI Blend MR+Trend":0.5}

stats_cache = {}
for name, (net, idx, split) in systems.items():
    net = net.fillna(0)
    eq = (1+net).cumprod()
    sh = net.mean()/net.std()*np.sqrt(ann) if net.std()>0 else 0
    sh_oos = net.iloc[split:].mean()/net.iloc[split:].std()*np.sqrt(ann) if net.iloc[split:].std()>0 else 0
    dd = (eq/eq.cummax()-1).min()
    cagr = eq.iloc[-1]**(ann/len(net))-1 if eq.iloc[-1]>0 else -1
    stats_cache[name] = (eq, sh, sh_oos, dd, cagr, idx, split)
    ax1.plot(idx, eq.values, lw=lws[name], alpha=alphas[name], color=colors[name],
             label=f"{name}  Sh {sh:.2f} (OOS {sh_oos:.2f}) CAGR {cagr*100:.1f}% DD {dd*100:.0f}%")

# IS/OOS divider (XAU)
ax1.axvline(d1.index[SPLIT], color="red", ls="--", alpha=0.5, lw=1.2, label="IS/OOS split (XAU)")
ax1.set_yscale("log"); ax1.legend(loc="upper left", fontsize=8); ax1.grid(alpha=0.3)
ax1.set_title(f"XAUUSD money printer — vol-targeted (15%) daily, 2009-2026, costs included\n"
              f"Best: XAU Blend T70+MR30  Sharpe 0.72  OOS 0.95  CAGR 6.9%  maxDD -18.9%")

# underwater for best
eq_best, _, _, dd_best, _, _, split_b = stats_cache["XAU Blend T70+MR30"]
uw = (eq_best / eq_best.cummax() - 1) * 100
ax2.fill_between(d1.index, uw.values, 0, color="#2ca02c", alpha=0.4)
ax2.set_ylabel("XAU Blend DD %"); ax2.grid(alpha=0.3)
ax2.axvline(d1.index[SPLIT], color="red", ls="--", alpha=0.5, lw=1.2)
ax2.set_title("Underwater drawdown — XAU Blend T70+MR30")

plt.tight_layout()
plt.savefig("xau_final_equity.png", dpi=120)
print("\n  saved -> xau_final_equity.png")
