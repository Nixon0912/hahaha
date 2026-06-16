"""
Full-repo money printer scan — all 28 instruments × TSMOM + MR blend.

For each instrument: resample M15 (or H1/M5) to daily, apply vol-targeted
TSMOM (lookbacks 10/20/50/100/200d) and MR-z (n=20, entry=2) and blend.
Rank by min(IS_Sharpe, OOS_Sharpe). Build equal-vol multi-asset portfolio
of all survivors (both IS and OOS > 0.3).

IS = first 70%, OOS = last 30%.
Cost = half-spread (from data SPREAD column where available, else default).
Vol target = 15% annual.
"""
import sys, warnings, glob, re
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

ann = 252
VT  = 0.15    # vol target
CAP = 4.0     # leverage cap
SPLIT_FRAC = 0.70

# half-spreads: rough defaults by asset class (in price units)
HALF_SPREAD_DEFAULTS = {
    "XAU": 0.20, "XAG": 0.015, "XPD": 2.0, "XPT": 0.5,
    "XTI": 0.03, "XBR": 0.03, "NGC": 0.003, "CUC": 0.002,
    "USD": 0.00005, "EUR": 0.00004, "AUD": 0.00004,
    "DAX": 1.0, "ESP": 1.0, "F40": 0.5, "IBX": 2.0,
    "ASX": 1.0, "JPN": 5.0, "NAS": 0.5, "SP5": 0.5,
    "US3": 2.0, "UK1": 1.0, "HSI": 3.5,
}

def get_half_spread(sym, prices):
    for prefix, hs in HALF_SPREAD_DEFAULTS.items():
        if sym.startswith(prefix): return hs
    return prices.mean() * 0.0001   # fallback 1bp

def load_daily(fpath):
    try:
        df = load_raw(fpath)
        df = df[df["close"] > 0]
        # use spread column if present and sensible
        has_spread = "spread" in df.columns and df["spread"].median() > 0
        d1 = df.resample("1D").agg({"open":"first","high":"max","low":"min",
                                     "close":"last","spread":"mean"} if has_spread
                                    else {"open":"first","high":"max","low":"min","close":"last"}).dropna()
        d1 = d1[d1["close"] > 0]
        if not has_spread:
            d1["spread"] = np.nan
        return d1
    except Exception as e:
        print(f"  ERROR loading {fpath}: {e}")
        return None

def backtest_sys(prices, spread_series, sym, sig, name):
    ret = prices.pct_change().fillna(0)
    rv  = ret.rolling(20).std() * np.sqrt(ann)
    pos_vt = sig * (VT / rv.replace(0, np.nan)).clip(upper=CAP).fillna(0)
    pos_s  = pos_vt.shift(1).fillna(0)
    turn   = pos_s.diff().abs().fillna(pos_s.abs())

    # cost: use actual spread/2 if available, else default
    if spread_series.notna().mean() > 0.5:
        # spread is in points/ticks — convert to fractional
        half_pts = spread_series.ffill() / 2
        cost_frac = half_pts / prices
    else:
        hs = get_half_spread(sym, prices)
        cost_frac = hs / prices

    net = (pos_s * ret - turn * cost_frac).fillna(0)
    n = len(net); split = int(n * SPLIT_FRAC)
    eq  = (1+net).cumprod()
    sh  = net.mean()/net.std()*np.sqrt(ann) if net.std()>0 else 0
    nis = net.iloc[:split]; nos = net.iloc[split:]
    sh_is  = nis.mean()/nis.std()*np.sqrt(ann) if nis.std()>0 else 0
    sh_oos = nos.mean()/nos.std()*np.sqrt(ann) if nos.std()>0 else 0
    dd  = (eq/eq.cummax()-1).min()
    yrs = n/ann
    cagr = eq.iloc[-1]**(1/max(yrs,0.5))-1 if eq.iloc[-1]>0 else -1
    return dict(sym=sym, name=name, eq=eq, net=net,
                sh=sh, sh_is=sh_is, sh_oos=sh_oos, dd=dd, cagr=cagr,
                split=split, n=n, idx=prices.index)

def mr_signal(prices, n=20, entry=2.0):
    ma = prices.rolling(n).mean(); sd = prices.rolling(n).std()
    z  = (prices - ma) / sd; zv = z.values
    out = np.zeros(len(zv)); cur = 0.0
    for i in range(len(zv)):
        if np.isnan(zv[i]): out[i]=cur; continue
        if cur==0:
            if zv[i]>=entry: cur=-1
            elif zv[i]<=-entry: cur=1
        else:
            if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
        out[i]=cur
    return pd.Series(out, index=prices.index)

# ── Collect all CSV files ─────────────────────────────────────────────────────
all_csvs = sorted(glob.glob("*.csv"))
print(f"Found {len(all_csvs)} CSV files")

# For each instrument pick the best (longest) timeframe file
inst_files = {}
for f in all_csvs:
    m = re.match(r"([A-Z0-9]+)_(M5|M15|H1)_", Path(f).name)
    if not m: continue
    sym, tf = m.group(1), m.group(2)
    tf_rank = {"M5":0, "M15":1, "H1":2}
    if sym not in inst_files or tf_rank[tf] > tf_rank[inst_files[sym][1]]:
        inst_files[sym] = (f, tf)

print(f"Instruments: {sorted(inst_files.keys())}")

# ── Run scan ──────────────────────────────────────────────────────────────────
all_results = []
print("\n" + "="*90)
print(f"  {'sym':<10}{'strategy':<22}{'CAGR':>8}{'Sharpe':>8}{'IS Sh':>8}{'OOS Sh':>8}{'maxDD':>8}{'bars':>6}")
print("="*90)

for sym, (fpath, tf) in sorted(inst_files.items()):
    d1 = load_daily(fpath)
    if d1 is None or len(d1) < 200: continue
    prices = d1["close"]
    spread = d1["spread"] if "spread" in d1.columns else pd.Series(np.nan, index=d1.index)

    best_per_sym = []

    # A. TSMOM at various lookbacks
    for L in [10, 20, 50, 100, 200]:
        if L >= len(prices)*0.3: continue
        sig = np.sign(prices.pct_change(L)).fillna(0)
        r = backtest_sys(prices, spread, sym, sig, f"TSMOM{L}d")
        best_per_sym.append(r)

    # B. MR z20
    sig_mr = mr_signal(prices, n=20, entry=2.0)
    r = backtest_sys(prices, spread, sym, sig_mr, "MR_z20")
    best_per_sym.append(r)

    # C. Blend: 70% TSMOM-50d + 30% MR (winner on gold)
    if len(prices) > 60:
        sig_t = np.sign(prices.pct_change(min(50, int(len(prices)*0.1)))).fillna(0)
        sig_b = 0.70*sig_t + 0.30*sig_mr
        r = backtest_sys(prices, spread, sym, sig_b, "Blend_T70MR30")
        best_per_sym.append(r)

    # D. Blend: 50/50
    if len(prices) > 60:
        sig_b2 = 0.50*sig_t + 0.50*sig_mr
        r = backtest_sys(prices, spread, sym, sig_b2, "Blend_T50MR50")
        best_per_sym.append(r)

    # Pick best per symbol
    best_per_sym.sort(key=lambda x: -min(x["sh_is"], x["sh_oos"]))
    top = best_per_sym[0]
    all_results.append(top)
    flag = " ✅" if top["sh_is"] > 0.3 and top["sh_oos"] > 0.3 else ""
    print(f"  {sym:<10}{top['name']:<22}{top['cagr']*100:>7.1f}%{top['sh']:>8.2f}"
          f"{top['sh_is']:>8.2f}{top['sh_oos']:>8.2f}{top['dd']*100:>7.1f}%{top['n']:>6}{flag}")

# ── Rank globally ─────────────────────────────────────────────────────────────
all_results.sort(key=lambda x: -min(x["sh_is"], x["sh_oos"]))
survivors = [r for r in all_results if r["sh_is"] > 0.3 and r["sh_oos"] > 0.3]

print("\n" + "="*90)
print(f"  GLOBAL RANKING — {len(survivors)} survivors (IS>0.3 AND OOS>0.3)")
print("="*90)
print(f"  {'sym':<10}{'strategy':<22}{'CAGR':>8}{'Sharpe':>8}{'IS Sh':>8}{'OOS Sh':>8}{'maxDD':>8}")
for r in all_results[:20]:
    flag = " ✅" if r["sh_is"] > 0.3 and r["sh_oos"] > 0.3 else ""
    print(f"  {r['sym']:<10}{r['name']:<22}{r['cagr']*100:>7.1f}%{r['sh']:>8.2f}"
          f"{r['sh_is']:>8.2f}{r['sh_oos']:>8.2f}{r['dd']*100:>7.1f}%{flag}")

# ── Multi-asset equal-vol portfolio of survivors ──────────────────────────────
if len(survivors) >= 2:
    print(f"\n=== Multi-asset portfolio: {len(survivors)} instruments ===")
    # Align all survivor nets to common date range
    common_idx = survivors[0]["idx"]
    for r in survivors[1:]:
        common_idx = common_idx.intersection(r["idx"])
    print(f"  Common date range: {common_idx[0].date()} → {common_idx[-1].date()}  ({len(common_idx)} days)")

    port_net = pd.Series(0.0, index=common_idx)
    w = 1.0 / len(survivors)
    for r in survivors:
        port_net += w * r["net"].reindex(common_idx).fillna(0)

    split_p = int(len(port_net) * SPLIT_FRAC)
    eq_p = (1+port_net).cumprod()
    sh_p = port_net.mean()/port_net.std()*np.sqrt(ann) if port_net.std()>0 else 0
    sh_is_p = port_net.iloc[:split_p].mean()/port_net.iloc[:split_p].std()*np.sqrt(ann) if port_net.iloc[:split_p].std()>0 else 0
    sh_oos_p = port_net.iloc[split_p:].mean()/port_net.iloc[split_p:].std()*np.sqrt(ann) if port_net.iloc[split_p:].std()>0 else 0
    dd_p = (eq_p/eq_p.cummax()-1).min()
    yrs_p = len(port_net)/ann
    cagr_p = eq_p.iloc[-1]**(1/max(yrs_p,0.5))-1 if eq_p.iloc[-1]>0 else -1

    print(f"  Equal-weight portfolio:  CAGR {cagr_p*100:.1f}%  Sharpe {sh_p:.2f}  "
          f"IS {sh_is_p:.2f}  OOS {sh_oos_p:.2f}  maxDD {dd_p*100:.1f}%")

    by_p = port_net.groupby(common_idx.year).apply(lambda s: (1+s).prod()-1)
    print("  Year-by-year:", "  ".join(f"{y}:{v*100:+.1f}%" for y,v in by_p.items()))
    print(f"  Positive years: {(by_p>0).sum()}/{len(by_p)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={"height_ratios":[3,1]})
ax1 = axes[0]; ax2 = axes[1]

palette = plt.cm.tab20.colors
top_plot = [r for r in all_results if r["sh_is"] > 0.25 and r["sh_oos"] > 0.25][:12]

for i, r in enumerate(top_plot):
    lw = 2.0 if i == 0 else (1.4 if i < 3 else 0.9)
    alpha = 1.0 if i == 0 else (0.85 if i < 3 else 0.6)
    ax1.plot(r["idx"], r["eq"].values, lw=lw, alpha=alpha, color=palette[i % 20],
             label=f"{r['sym']} {r['name']}  Sh {r['sh']:.2f} (OOS {r['sh_oos']:.2f}) "
                   f"CAGR {r['cagr']*100:.1f}% DD {r['dd']*100:.0f}%")

ax1.set_yscale("log"); ax1.legend(loc="upper left", fontsize=7, ncol=2)
ax1.grid(alpha=0.25)
best = all_results[0]
ax1.axvline(best["idx"][best["split"]], color="red", ls="--", alpha=0.5, lw=1)
ax1.set_title(f"Full-repo money printer scan — top strategies, vol-targeted 15%\n"
              f"#1: {best['sym']} {best['name']}  Sh {best['sh']:.2f}  OOS {best['sh_oos']:.2f}  "
              f"CAGR {best['cagr']*100:.1f}%  maxDD {best['dd']*100:.0f}%")

# Portfolio underwater
if len(survivors) >= 2:
    uw_p = (eq_p / eq_p.cummax() - 1) * 100
    ax2.fill_between(common_idx, uw_p.values, 0, color="#2ca02c", alpha=0.5, label="Portfolio DD")
    ax2.set_ylabel("Portfolio DD %")
    ax2.axvline(common_idx[split_p], color="red", ls="--", alpha=0.5)
    ax2.set_title(f"Equal-vol portfolio ({len(survivors)} instruments)  "
                  f"Sh {sh_p:.2f}  OOS {sh_oos_p:.2f}  CAGR {cagr_p*100:.1f}%  maxDD {dd_p*100:.0f}%")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
else:
    ax2.axis("off")

plt.tight_layout()
plt.savefig("full_scan_equity.png", dpi=120)
print("\n  saved -> full_scan_equity.png")
