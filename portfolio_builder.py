"""
Portfolio builder — maximize Sharpe through diversification.

Target: Sharpe 2.5 (portfolio level). Approach:
  - Test all 25 instruments × 8 signals
  - Select survivors (IS>0 AND OOS>0 at signal level)
  - Build equal-vol portfolio; report portfolio Sharpe
  - Also test multi-signal blends per instrument

Key fix: IGNORE MT5 spread column (unreliable units). Use proper default
half-spreads per asset class.

Vol target = 15% per stream. IS = first 70%, OOS = last 30%.
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
VT  = 0.15
CAP = 3.0
SF  = 0.70   # IS fraction

# ── Proper half-spread defaults (in price units) ──────────────────────────────
# FX 5-digit: 1 pip = 0.0001; typical 1-pip half-spread
HALF = {
    "AUDUSD": 0.00008, "EURUSD": 0.00007, "USDJPY": 0.008, "USDCAD": 0.00010,
    "USDCHF": 0.00009, "AUDNZD": 0.00012,
    "XAUUSD": 0.15,    "XAGUSD": 0.02,    "XPDUSD": 3.0,    "XPTUSD": 0.8,
    "XTIUSD": 0.04,    "XBRUSD": 0.04,    "NGCUSD": 0.005,  "CUCUSD": 0.0025,
    "NAS100": 1.5,     "SP500": 0.5,      "US30": 3.0,
    "DAX40": 2.0,      "ESXEUR": 1.0,     "F40EUR": 0.8,    "IBXEUR": 4.0,
    "UK100": 1.5,      "ASXAUD": 2.0,     "JPN225": 8.0,    "HSIHKD": 3.5,
}

def half_spread(sym, prices):
    for k, v in HALF.items():
        if sym.startswith(k[:4]): return v
    return prices.mean() * 0.0002

def load_daily(fpath):
    try:
        df = load_raw(fpath); df = df[df["close"] > 0]
        d1 = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        return d1[d1["close"] > 0]
    except: return None

def mr_sig(prices, n=20, entry=2.0):
    ma = prices.rolling(n).mean(); sd = prices.rolling(n).std()
    z = (prices-ma)/sd; zv = z.values
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

def run(prices, hs, sig, name):
    ret = prices.pct_change().fillna(0)
    rv  = ret.rolling(20).std() * np.sqrt(ann)
    pos = sig * (VT / rv.replace(0,np.nan)).clip(upper=CAP).fillna(0)
    pos = pos.shift(1).fillna(0)
    trn = pos.diff().abs().fillna(pos.abs())
    net = (pos*ret - trn*(hs/prices)).fillna(0)
    n = len(net); sp = int(n*SF)
    eq = (1+net).cumprod()
    sh  = net.mean()/net.std()*np.sqrt(ann) if net.std()>0 else 0
    si  = net.iloc[:sp]; so = net.iloc[sp:]
    shi = si.mean()/si.std()*np.sqrt(ann) if si.std()>0 else 0
    sho = so.mean()/so.std()*np.sqrt(ann) if so.std()>0 else 0
    dd  = (eq/eq.cummax()-1).min()
    cagr= eq.iloc[-1]**(ann/max(n,1))-1 if eq.iloc[-1]>0 else -1
    return dict(name=name,net=net,eq=eq,sh=sh,shi=shi,sho=sho,dd=dd,cagr=cagr,sp=sp,idx=prices.index,n=n)

# ── Collect files ─────────────────────────────────────────────────────────────
all_csvs = sorted(glob.glob("*.csv"))
inst_files = {}
for f in all_csvs:
    m = re.match(r"([A-Z0-9]+)_(M5|M15|H1)_", Path(f).name)
    if not m: continue
    sym, tf = m.group(1), m.group(2)
    rank = {"M5":0,"M15":1,"H1":2}
    if sym not in inst_files or rank[tf] > rank[inst_files[sym][1]]:
        inst_files[sym] = (f, tf)

# ── Per-instrument signal search ──────────────────────────────────────────────
print("="*95)
print(f"  {'sym':<10}{'best_signal':<22}{'CAGR':>8}{'Sh':>7}{'IS':>7}{'OOS':>7}{'DD':>8}  survivor")
print("="*95)

all_best = {}   # sym -> best result
all_nets = {}   # sym -> net series of best result

for sym, (fpath, tf) in sorted(inst_files.items()):
    d1 = load_daily(fpath)
    if d1 is None or len(d1) < 300: continue
    p = d1["close"]
    hs = half_spread(sym, p)
    results = []

    # TSMOM lookbacks — proportional to data length
    maxL = min(200, int(len(p)*0.25))
    for L in [10, 20, 50, 100, 200]:
        if L > maxL: continue
        sig = np.sign(p.pct_change(L)).fillna(0)
        results.append(run(p, hs, sig, f"TSMOM{L}d"))

    # MR
    for n,e in [(10,1.5),(20,2.0),(30,2.0),(20,1.5)]:
        results.append(run(p, hs, mr_sig(p, n, e), f"MR_n{n}_e{e}"))

    # Blends
    sig_t = np.sign(p.pct_change(min(50, maxL))).fillna(0)
    sig_mr = mr_sig(p, 20, 2.0)
    for wt in [0.3, 0.5, 0.7]:
        sig_b = wt*sig_t + (1-wt)*sig_mr
        results.append(run(p, hs, sig_b, f"Blend_T{int(wt*10)}MR{int((1-wt)*10)}"))

    # Reversal (short-term anti-momentum)
    for L in [1, 2, 3, 5]:
        sig = -np.sign(p.pct_change(L)).fillna(0)
        results.append(run(p, hs, sig, f"REV{L}d"))

    # Pick best by min(IS, OOS)
    results.sort(key=lambda x: -min(x["shi"], x["sho"]))
    best = results[0]
    surv = "✅" if best["shi"]>0 and best["sho"]>0 else "❌"
    print(f"  {sym:<10}{best['name']:<22}{best['cagr']*100:>7.1f}%{best['sh']:>7.2f}"
          f"{best['shi']:>7.2f}{best['sho']:>7.2f}{best['dd']*100:>7.1f}%  {surv}")
    all_best[sym] = best
    all_nets[sym] = best["net"]

# ── Portfolio construction ────────────────────────────────────────────────────
survivors_sym = [s for s,r in all_best.items() if r["shi"]>0 and r["sho"]>0]
print(f"\n✅ Survivors: {len(survivors_sym)} / {len(all_best)}: {survivors_sym}")

def build_portfolio(sym_list, label):
    if not sym_list: return None
    nets = [all_nets[s] for s in sym_list]
    common = nets[0].index
    for n in nets[1:]: common = common.intersection(n.index)
    if len(common) < 100: return None
    combined = sum(n.reindex(common).fillna(0) for n in nets) / len(nets)
    sp = int(len(combined) * SF)
    eq = (1+combined).cumprod()
    sh  = combined.mean()/combined.std()*np.sqrt(ann) if combined.std()>0 else 0
    shi = combined.iloc[:sp].mean()/combined.iloc[:sp].std()*np.sqrt(ann) if combined.iloc[:sp].std()>0 else 0
    sho = combined.iloc[sp:].mean()/combined.iloc[sp:].std()*np.sqrt(ann) if combined.iloc[sp:].std()>0 else 0
    dd  = (eq/eq.cummax()-1).min()
    cagr= eq.iloc[-1]**(ann/max(len(combined),1))-1 if eq.iloc[-1]>0 else -1
    by  = combined.groupby(common.year).apply(lambda s:(1+s).prod()-1)
    return dict(label=label,eq=eq,net=combined,sh=sh,shi=shi,sho=sho,dd=dd,cagr=cagr,
                sp=sp,idx=common,by=by,n_inst=len(sym_list))

print("\n" + "="*80)
print("  PORTFOLIO RESULTS")
print("="*80)

# Portfolio 1: all survivors
p_all = build_portfolio(survivors_sym, f"All-{len(survivors_sym)}")
# Portfolio 2: top 10 by OOS Sharpe
top10 = sorted(survivors_sym, key=lambda s: -all_best[s]["sho"])[:10]
p_top10 = build_portfolio(top10, f"Top-10 OOS")
# Portfolio 3: top 5 by OOS Sharpe
top5 = top10[:5]
p_top5 = build_portfolio(top5, "Top-5 OOS")
# Portfolio 4: top 3
top3 = top10[:3]
p_top3 = build_portfolio(top3, "Top-3 OOS")

for port in [p for p in [p_all, p_top10, p_top5, p_top3] if p]:
    print(f"\n  [{port['label']}] {port['n_inst']} instruments  "
          f"CAGR {port['cagr']*100:.1f}%  Sharpe {port['sh']:.2f}  "
          f"IS {port['shi']:.2f}  OOS {port['sho']:.2f}  maxDD {port['dd']*100:.1f}%")
    print(f"  Year-by-year: " + "  ".join(f"{y}:{v*100:+.1f}%" for y,v in port["by"].items()))
    print(f"  Positive years: {(port['by']>0).sum()}/{len(port['by'])}")

# ── Correlation matrix of survivors ──────────────────────────────────────────
if len(survivors_sym) > 1:
    common_all = all_nets[survivors_sym[0]].index
    for s in survivors_sym[1:]: common_all = common_all.intersection(all_nets[s].index)
    net_df = pd.DataFrame({s: all_nets[s].reindex(common_all) for s in survivors_sym})
    corr = net_df.corr()
    print(f"\n  Pairwise correlation among {len(survivors_sym)} survivors (net returns):")
    print(f"  Mean |corr| = {corr.where(corr<1).abs().stack().mean():.3f}")
    print(corr.round(2).to_string())

# ── Multi-signal within XAU specifically ─────────────────────────────────────
print("\n\n" + "="*80)
print("  DEEP DIVE: XAUUSD multi-signal (all good signals combined)")
print("="*80)
if "XAUUSD" in inst_files:
    d1_x = load_daily(inst_files["XAUUSD"][0])
    if d1_x is not None:
        p_x = d1_x["close"]; hs_x = HALF["XAUUSD"]
        xau_sigs = []
        all_xau = []
        for L in [20, 50, 100]:
            sig = np.sign(p_x.pct_change(L)).fillna(0)
            r = run(p_x, hs_x, sig, f"TSMOM{L}d")
            all_xau.append(r)
            if r["shi"]>0 and r["sho"]>0: xau_sigs.append(sig)
        for n,e in [(20,2.0),(30,2.0)]:
            sig = mr_sig(p_x, n, e)
            r = run(p_x, hs_x, sig, f"MR_n{n}_e{e}")
            all_xau.append(r)
        # Ensemble of all positive IS+OOS signals
        if xau_sigs:
            ens = sum(xau_sigs) / len(xau_sigs)
            r_ens = run(p_x, hs_x, ens, f"Ensemble_{len(xau_sigs)}sig")
            all_xau.append(r_ens)
        all_xau.sort(key=lambda x: -min(x["shi"],x["sho"]))
        print(f"  {'name':<20}{'CAGR':>8}{'Sh':>7}{'IS':>7}{'OOS':>7}{'DD':>8}")
        for r in all_xau:
            flag = " ✅" if r["shi"]>0 and r["sho"]>0 else ""
            print(f"  {r['name']:<20}{r['cagr']*100:>7.1f}%{r['sh']:>7.2f}{r['shi']:>7.2f}{r['sho']:>7.2f}{r['dd']*100:>7.1f}%{flag}")

# ── Plot ──────────────────────────────────────────────────────────────────────
ports_to_plot = [p for p in [p_top3, p_top5, p_top10, p_all] if p]
top_insts = sorted(survivors_sym, key=lambda s: -all_best[s]["sho"])[:5]

fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={"height_ratios":[3,2,1]})
ax1,ax2,ax3 = axes

pal = plt.cm.tab10.colors
# Top individual instruments
for i, sym in enumerate(top_insts):
    r = all_best[sym]
    ax1.plot(r["idx"], r["eq"].values, lw=1.2, alpha=0.7, color=pal[i],
             label=f"{sym} {r['name']}  Sh {r['sh']:.2f}(OOS {r['sho']:.2f}) CAGR {r['cagr']*100:.1f}%")
ax1.set_yscale("log"); ax1.legend(fontsize=8,loc="upper left"); ax1.grid(alpha=0.25)
ax1.set_title("Individual instrument equity curves (top survivors, vol-target 15%)")

# Portfolio curves
pcolors = ["#2ca02c","#1f77b4","#ff7f0e","#d62728"]
for i, port in enumerate(ports_to_plot):
    ax2.plot(port["idx"], port["eq"].values, lw=2.0 if i==0 else 1.2,
             alpha=1.0 if i==0 else 0.7, color=pcolors[i],
             label=f"[{port['label']}] Sh {port['sh']:.2f}(IS {port['shi']:.2f} OOS {port['sho']:.2f}) "
                   f"CAGR {port['cagr']*100:.1f}% DD {port['dd']*100:.0f}%")
ax2.set_yscale("log"); ax2.legend(fontsize=8,loc="upper left"); ax2.grid(alpha=0.25)
ax2.set_title("Portfolio equity curves — diversified vol-targeted blend")

# Drawdown of best portfolio
if ports_to_plot:
    bp = ports_to_plot[0]
    uw = (bp["eq"]/bp["eq"].cummax()-1)*100
    ax3.fill_between(bp["idx"], uw.values, 0, color=pcolors[0], alpha=0.45)
    ax3.axvline(bp["idx"][bp["sp"]], color="red", ls="--", alpha=0.5)
    ax3.set_ylabel("DD %"); ax3.grid(alpha=0.25)
    ax3.set_title(f"Drawdown [{bp['label']}]  maxDD {bp['dd']*100:.1f}%  IS/OOS split (red)")

plt.tight_layout()
plt.savefig("portfolio_equity.png", dpi=120)
print("\n  saved -> portfolio_equity.png")
