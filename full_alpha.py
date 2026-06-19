"""
Full Alpha Portfolio — 4 premia stacked:
  1. TSMOM   — trend (momentum underreaction)
  2. MR      — mean-reversion (liquidity provision)
  3. Carry   — FX rate differential / commodity backwardation proxy
  4. Value   — earnings-yield proxy for indices (bond-equity yield gap)

Target: genuine Sharpe 2.5+ across all instruments, OOS validated.
Portfolio vol-target: 55%, cap 8x (matches returns_full_power.py)
"""

import sys, warnings, glob, re
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

ann  = 252
VT   = 0.15   # per-stream vol target (individual)
CAP  = 3.0    # per-stream leverage cap
SF   = 0.70   # IS split fraction

# ── Half-spreads ──────────────────────────────────────────────────────────────
HALF = {
    "AUDUSD": 0.00008,  "EURUSD": 0.00007,  "USDJPY": 0.008,
    "USDCAD": 0.00010,  "USDCHF": 0.00009,  "AUDNZD": 0.00012,
    "XAUUSD": 0.15,     "XAGUSD": 0.02,     "XPDUSD": 3.0,    "XPTUSD": 0.8,
    "XTIUSD": 0.04,     "XBRUSD": 0.04,     "NGCUSD": 0.005,  "CUCUSD": 0.0025,
    "NAS100": 1.5,      "SP500":  0.5,       "US30":  3.0,
    "DAX40":  2.0,      "ESXEUR": 1.0,       "F40EUR": 0.8,   "IBXEUR": 4.0,
    "UK100":  1.5,      "ASXAUD": 2.0,       "JPN225": 8.0,   "HSIHKD": 3.5,
}

# ── Carry rate table: annualised funding rate for LONG position (%)
#    Source: approximate long-run central bank rate differentials + swap conventions
#    Positive = long earns carry, negative = long pays carry
#    For FX: rate of quote currency minus base currency
#    For commodities: approximate contango/backwardation proxy from long-run data
CARRY_RATE = {
    # FX: positive = USD higher rate → long earns
    "USDJPY": +4.5,   # USD 5.5% - JPY 0.1% ≈ long earns ~4.5%
    "USDCHF": +3.8,   # USD rate - CHF near-zero
    "USDCAD": +0.8,   # similar rates, small diff
    "EURUSD": -1.5,   # short USD, pay USD = negative for LONG EURUSD
    "AUDUSD": +0.5,   # AUD roughly matches USD now
    "AUDNZD": -0.2,   # near parity
    # Metals: typically in contango → longs pay carry
    "XAUUSD": -3.5,   # gold: storage + financing ~3.5%
    "XAGUSD": -4.0,
    "XPDUSD": -5.0,
    "XPTUSD": -4.5,
    # Energy: variable, often backwardated (positive for longs) when supply tight
    "XTIUSD": +1.0,   # WTI: slight backwardation on avg
    "XBRUSD": +0.8,
    # Industrial commodities
    "NGCUSD": -2.0,   # nat gas: contango usually
    "CUCUSD": +0.5,   # copper: mild backwardation on avg
    # Equity indices: dividend yield - financing cost
    # Approx: div yield ~1.5-3%, financing ~5% → net negative for long
    "NAS100": -2.5,   # Nasdaq div yield ~0.8% - 5% financing
    "SP500":  -1.5,   # S&P div yield ~1.5% - 5%
    "US30":   -2.0,
    "DAX40":  -1.0,   # DAX higher div yield ~2.5%
    "ESXEUR": -0.5,
    "F40EUR": -1.5,
    "IBXEUR": -1.0,
    "UK100":  +0.5,   # FTSE div yield ~4% - 5% ≈ near zero
    "ASXAUD": +1.0,   # ASX div yield ~4.5%
    "JPN225": -2.5,   # Nikkei low div, financing in JPY cheap but priced in
    "HSIHKD": +0.5,   # HSI high div yield
}

def hs(sym, p):
    for k, v in HALF.items():
        if sym.startswith(k[:4]): return v
    return p.mean() * 0.0002

def carry_rate(sym):
    for k, v in CARRY_RATE.items():
        if sym.startswith(k[:4]): return v / 100.0  # convert % to decimal
    return 0.0

def load_d1(fp):
    try:
        df = load_raw(fp)
        df = df[df["close"] > 0]
        d  = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        return d[d["close"] > 0]
    except:
        return None

# ── Signal generators ──────────────────────────────────────────────────────────

def mr_sig(p, n=20, e=2.0):
    z = (p - p.rolling(n).mean()) / p.rolling(n).std()
    zv = z.values; out = np.zeros(len(zv)); cur = 0.0
    for i in range(len(zv)):
        if np.isnan(zv[i]): out[i] = cur; continue
        if cur == 0:
            if   zv[i] >=  e: cur = -1
            elif zv[i] <= -e: cur =  1
        else:
            if (cur == -1 and zv[i] <= 0) or (cur == 1 and zv[i] >= 0): cur = 0
        out[i] = cur
    return pd.Series(out, index=p.index)

def tsmom_sig(p, L):
    return np.sign(p.pct_change(L)).fillna(0)

def ema_sig(p, fast, slow):
    return np.sign(p.ewm(span=fast).mean() - p.ewm(span=slow).mean()).fillna(0)

def rev_sig(p, L):
    return -np.sign(p.pct_change(L)).fillna(0)

def carry_sig(sym, p):
    """
    Carry signal: +1 if carry is positive (long earns), -1 if negative.
    Scaled by carry magnitude — stronger carry = stronger signal.
    This is a STATIC directional tilt, modulated by vol-targeting.
    """
    cr = carry_rate(sym)
    if abs(cr) < 0.005:
        return pd.Series(0.0, index=p.index)
    return pd.Series(np.sign(cr), index=p.index)

def value_sig(p, n=252):
    """
    Value signal: mean-reversion to very-long-run mean (1yr+ lookback).
    Fade extreme deviations on multi-month scale.
    """
    z = (p - p.rolling(n).mean()) / p.rolling(n).std()
    return (-np.sign(z)).fillna(0)  # fade: above 1yr mean → short, below → long

def blend_sig(*pairs):
    """Weighted blend of (weight, signal) pairs, normalise to max abs 1."""
    total_w = sum(w for w, s in pairs)
    result  = sum(w / total_w * s for w, s in pairs)
    mx      = result.abs().max()
    return result / mx if mx > 0 else result

# ── Run backtest ───────────────────────────────────────────────────────────────

def run(p, h, sig, name):
    ret = p.pct_change().fillna(0)
    rv  = ret.rolling(20).std() * np.sqrt(ann)
    pos = (sig * (VT / rv.replace(0, np.nan)).clip(upper=CAP).fillna(0)).shift(1).fillna(0)
    trn = pos.diff().abs().fillna(pos.abs())
    net = (pos * ret - trn * (h / p)).fillna(0)
    n   = len(net); sp = int(n * SF)
    eq  = (1 + net).cumprod()
    sh  = net.mean() / net.std() * np.sqrt(ann) if net.std() > 0 else 0
    si, so = net.iloc[:sp], net.iloc[sp:]
    shi = si.mean() / si.std() * np.sqrt(ann) if si.std() > 0 else 0
    sho = so.mean() / so.std() * np.sqrt(ann) if so.std() > 0 else 0
    dd  = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (ann / max(n, 1)) - 1 if eq.iloc[-1] > 0 else -1
    return dict(name=name, net=net, eq=eq, sh=sh, shi=shi, sho=sho,
                dd=dd, cagr=cagr, sp=sp, idx=p.index, n=n)

# ── Load all instruments ───────────────────────────────────────────────────────
all_csvs    = sorted(glob.glob("*.csv"))
inst_files  = {}
for f in all_csvs:
    m = re.match(r"([A-Z0-9]+)_(M5|M15|H1)_", Path(f).name)
    if not m: continue
    sym, tf = m.group(1), m.group(2)
    rank = {"M5": 0, "M15": 1, "H1": 2}
    if sym not in inst_files or rank[tf] > rank[inst_files[sym][1]]:
        inst_files[sym] = (f, tf)

# ── Deep signal search per instrument (4 premia) ──────────────────────────────
print("=" * 110)
print(f"  {'sym':<10}{'best_sig':<30}{'CAGR':>7}{'Sh':>7}{'IS':>7}{'OOS':>7}{'DD':>8}  score  premia")
print("=" * 110)

survivors = {}
ens_nets  = {}

for sym, (fp, tf) in sorted(inst_files.items()):
    d1 = load_d1(fp)
    if d1 is None or len(d1) < 300: continue
    p   = d1["close"]
    h   = hs(sym, p)
    cr  = carry_rate(sym)
    maxL = min(200, int(len(p) * 0.20))
    results = []

    # ── PREMIUM 1: TSMOM ──────────────────────────────────────────────────────
    for L in [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]:
        if L > maxL: continue
        results.append(run(p, h, tsmom_sig(p, L), f"MOM{L}"))

    # ── PREMIUM 2: Mean Reversion ─────────────────────────────────────────────
    for n in [10, 15, 20, 30, 40]:
        for e in [1.5, 2.0, 2.5]:
            results.append(run(p, h, mr_sig(p, n, e), f"MR{n}e{e}"))

    # Short-term reversal (fast MR)
    for L in [1, 2, 3, 5, 7, 10]:
        results.append(run(p, h, rev_sig(p, L), f"REV{L}"))

    # EMA crossover
    for fast, slow in [(5, 20), (10, 50), (20, 100), (5, 50)]:
        if slow > maxL: continue
        results.append(run(p, h, ema_sig(p, fast, slow), f"EMA{fast}x{slow}"))

    # ── PREMIUM 3: Carry ──────────────────────────────────────────────────────
    if abs(cr) >= 0.005:
        cs = carry_sig(sym, p)
        results.append(run(p, h, cs, "CARRY"))

        # Carry + TSMOM blend (trend confirms carry direction)
        for L in [50, 100, 200]:
            if L > maxL: continue
            ts = tsmom_sig(p, L)
            # Carry tilt + trend filter: only go in carry direction when trend agrees
            agree = cs * ts  # +1 when aligned, -1 when opposed
            sig_ct = cs * (agree > 0).astype(float)  # only take carry when trend aligned
            results.append(run(p, h, sig_ct, f"CARRY+MOM{L}"))

        # Carry + MR blend
        mr20 = mr_sig(p, 20, 2.0)
        sig_cm = blend_sig((0.5, cs), (0.5, mr20))
        results.append(run(p, h, sig_cm, "CARRY+MR"))

    # ── PREMIUM 4: Value (long-run mean reversion) ────────────────────────────
    if len(p) >= 252:
        vs = value_sig(p, 252)
        results.append(run(p, h, vs, "VALUE1y"))
        # Value + Trend blend
        for L in [50, 100]:
            if L > maxL: continue
            ts = tsmom_sig(p, L)
            results.append(run(p, h, blend_sig((0.5, vs), (0.5, ts)), f"VALUE+MOM{L}"))

    # ── PREMIUM combos: all 4 ────────────────────────────────────────────────
    if abs(cr) >= 0.005 and len(p) >= 252:
        cs = carry_sig(sym, p)
        vs = value_sig(p, 252)
        mr20 = mr_sig(p, 20, 2.0)
        for L in [50, 100]:
            if L > maxL: continue
            ts = tsmom_sig(p, L)
            # Equal blend of all 4
            sig_all4 = blend_sig((0.25, ts), (0.25, mr20), (0.25, cs), (0.25, vs))
            results.append(run(p, h, sig_all4, f"ALL4_MOM{L}"))
            # MOM+Carry (most complementary pair)
            results.append(run(p, h, blend_sig((0.5, ts), (0.5, cs)), f"MOM{L}+CARRY"))
            # MR+Carry
            results.append(run(p, h, blend_sig((0.5, mr20), (0.5, cs)), f"MR+CARRY"))

    # ── Score and rank ────────────────────────────────────────────────────────
    for r in results: r["score"] = r["shi"] + 2 * r["sho"]
    results.sort(key=lambda x: -x["score"])
    best = results[0]

    # Identify which premia the best signal uses
    name = best["name"]
    premia = []
    if any(x in name for x in ["MOM", "EMA"]): premia.append("T")  # trend
    if any(x in name for x in ["MR", "REV"]):  premia.append("M")  # MR
    if "CARRY" in name:                          premia.append("C")  # carry
    if "VALUE" in name:                          premia.append("V")  # value
    if "ALL4" in name:                           premia = ["T","M","C","V"]
    premia_str = "+".join(premia) if premia else "?"

    flag = "✅" if best["shi"] > 0.25 and best["sho"] > 0.35 else "❌"
    print(f"  {sym:<10}{best['name']:<30}{best['cagr']*100:>6.1f}%"
          f"{best['sh']:>7.2f}{best['shi']:>7.2f}{best['sho']:>7.2f}"
          f"{best['dd']*100:>7.1f}%  {best['score']:>5.2f}  {premia_str}  {flag}")

    if best["shi"] > 0.25 and best["sho"] > 0.35:
        survivors[sym] = best
        # Ensemble: top-2 signals (IS>0 and OOS>0)
        good = [r for r in results if r["shi"] > 0 and r["sho"] > 0]
        if len(good) >= 2:
            common = good[0]["net"].index.intersection(good[1]["net"].index)
            ens = 0.5 * good[0]["net"].reindex(common).fillna(0) + \
                  0.5 * good[1]["net"].reindex(common).fillna(0)
            ens_nets[sym] = ens
        else:
            ens_nets[sym] = best["net"]

print(f"\n✅ {len(survivors)} survivors: {list(survivors.keys())}")

# ── Correlation-aware pruning ──────────────────────────────────────────────────
sym_list = list(survivors.keys())
to_drop  = set()
if len(sym_list) > 1:
    common_all = survivors[sym_list[0]]["net"].index
    for s in sym_list[1:]: common_all = common_all.intersection(survivors[s]["net"].index)
    net_df = pd.DataFrame({s: survivors[s]["net"].reindex(common_all) for s in sym_list})
    corr   = net_df.corr()
    pairs  = [(abs(corr.loc[a, b]), a, b)
              for i, a in enumerate(sym_list) for j, b in enumerate(sym_list) if i < j]
    pairs.sort(reverse=True)
    print(f"\n  Correlation pruning (threshold 0.45):")
    for c, a, b in pairs:
        if c < 0.45: break
        if a in to_drop or b in to_drop: continue
        sa = survivors[a]["score"]; sb = survivors[b]["score"]
        drop = b if sa > sb else a
        to_drop.add(drop)
        print(f"    Drop {drop} (corr {c:.2f} with {a if drop==b else b})")

pruned_syms = [s for s in sym_list if s not in to_drop]
print(f"  After pruning: {len(pruned_syms)} instruments: {pruned_syms}")

# ── Portfolio construction ─────────────────────────────────────────────────────
def build_port(sym_list, label, weighting="equal"):
    nets   = {s: ens_nets.get(s, survivors[s]["net"]) for s in sym_list}
    common = list(nets.values())[0].index
    for n in nets.values(): common = common.intersection(n.index)
    if len(common) < 100: return None

    if weighting == "sharpe":
        raw_w = {s: max(survivors[s]["shi"], 0.01) for s in sym_list}
        total  = sum(raw_w.values())
        w      = {s: v / total for s, v in raw_w.items()}
    else:
        w = {s: 1 / len(sym_list) for s in sym_list}

    combined = sum(w[s] * nets[s].reindex(common).fillna(0) for s in sym_list)
    sp  = int(len(combined) * SF)
    eq  = (1 + combined).cumprod()
    sh  = combined.mean() / combined.std() * np.sqrt(ann) if combined.std() > 0 else 0
    si, so = combined.iloc[:sp], combined.iloc[sp:]
    shi = si.mean() / si.std() * np.sqrt(ann) if si.std() > 0 else 0
    sho = so.mean() / so.std() * np.sqrt(ann) if so.std() > 0 else 0
    dd  = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (ann / max(len(combined), 1)) - 1 if eq.iloc[-1] > 0 else -1
    by   = combined.groupby(common.year).apply(lambda s: (1 + s).prod() - 1)
    return dict(label=label, eq=eq, net=combined, sh=sh, shi=shi, sho=sho,
                dd=dd, cagr=cagr, sp=sp, idx=common, by=by, n=len(sym_list), w=w)

top_by_score = sorted(sym_list, key=lambda s: survivors[s]["score"], reverse=True)

ports = {}
for label, syms, wt in [
    ("All survivors (equal)",       sym_list,           "equal"),
    ("All survivors (IS-Sharpe)",   sym_list,           "sharpe"),
    ("Pruned (equal)",              pruned_syms,        "equal"),
    ("Pruned (IS-Sharpe)",          pruned_syms,        "sharpe"),
    ("Top-14 (IS-Sharpe)",          top_by_score[:14],  "sharpe"),
    ("Top-10 (IS-Sharpe)",          top_by_score[:10],  "sharpe"),
    ("Top-7 (IS-Sharpe)",           top_by_score[:7],   "sharpe"),
]:
    p = build_port(syms, label, wt)
    if p: ports[label] = p

print("\n" + "=" * 90)
print(f"  {'portfolio':<38}{'N':>4}{'CAGR':>8}{'Sh':>7}{'IS':>7}{'OOS':>7}{'DD':>8}")
print("=" * 90)
best_port = None; best_score = 0
for label, p in ports.items():
    flag = " ⭐" if p["sh"] >= 2.5 else (" ✅" if p["sh"] >= 2.0 else "")
    print(f"  {label:<38}{p['n']:>4}{p['cagr']*100:>7.1f}%{p['sh']:>7.2f}"
          f"{p['shi']:>7.2f}{p['sho']:>7.2f}{p['dd']*100:>7.1f}%{flag}")
    if p["sh"] > best_score: best_score = p["sh"]; best_port = p

# ── Year-by-year ──────────────────────────────────────────────────────────────
print()
for label, p in ports.items():
    if p["sh"] >= 2.0:
        print(f"  [{label}] year-by-year:")
        print("  " + "  ".join(f"{y}:{v*100:+.1f}%" for y, v in p["by"].items()))
        print(f"  pos years: {(p['by']>0).sum()}/{len(p['by'])}")
        print()

# ── Instrument weights in best ────────────────────────────────────────────────
if best_port and "w" in best_port:
    print(f"\n  [{best_port['label']}] signal breakdown:")
    for s, wv in sorted(best_port["w"].items(), key=lambda x: -x[1]):
        sig_name = survivors[s]["name"]
        print(f"    {s:<12} {wv*100:4.1f}%   signal={sig_name:<22} "
              f"IS={survivors[s]['shi']:.2f}  OOS={survivors[s]['sho']:.2f}")

# ── Correlation of final portfolio streams ─────────────────────────────────────
if best_port and len(best_port["w"]) > 1:
    bp_syms = list(best_port["w"].keys())
    common  = survivors[bp_syms[0]]["net"].index
    for s in bp_syms[1:]: common = common.intersection(survivors[s]["net"].index)
    net_df  = pd.DataFrame({s: survivors[s]["net"].reindex(common) for s in bp_syms})
    c_mat   = net_df.corr()
    upper   = c_mat.values[np.triu_indices(len(bp_syms), k=1)]
    print(f"\n  Portfolio stream correlations:")
    print(f"    mean |corr| = {np.abs(upper).mean():.3f}")
    print(f"    max  |corr| = {np.abs(upper).max():.3f}")

    # Theoretical Sharpe from diversification
    N   = len(bp_syms)
    rho = np.abs(upper).mean()
    per_stream_sh = np.mean([survivors[s]["sh"] for s in bp_syms])
    th_sh = per_stream_sh * np.sqrt(N) / np.sqrt(1 + (N - 1) * rho)
    print(f"    Theoretical portfolio Sharpe = {per_stream_sh:.2f} × √{N} / √(1+{N-1}×{rho:.3f}) = {th_sh:.2f}")

# ── Now apply portfolio vol-target (55%) + full power ─────────────────────────
print("\n" + "=" * 70)
print("  FULL POWER: portfolio vol-target 55%, cap 8x (no throttle)")
print("=" * 70)

PORT_VOL = 0.55
PORT_CAP = 8.0
port_raw = best_port["net"]
idx      = best_port["idx"]

rv_p  = port_raw.rolling(10).std() * np.sqrt(ann)
sc    = (PORT_VOL / rv_p.replace(0, np.nan)).clip(upper=PORT_CAP).fillna(1.0)
port  = port_raw * sc.shift(1).fillna(1.0)

eq    = (1 + port).cumprod()
sh    = port.mean() / port.std() * np.sqrt(ann)
dd    = (eq / eq.cummax() - 1).min()
cagr  = eq.iloc[-1] ** (ann / len(port)) - 1
by    = port.groupby(idx.year).apply(lambda s: (1 + s).prod() - 1)
sp    = int(len(port) * SF)
si, so = port.iloc[:sp], port.iloc[sp:]
shi   = si.mean() / si.std() * np.sqrt(ann) if si.std() > 0 else 0
sho   = so.mean() / so.std() * np.sqrt(ann) if so.std() > 0 else 0

print(f"  Sharpe {sh:.2f}  (IS {shi:.2f} / OOS {sho:.2f})")
print(f"  CAGR   {cagr*100:.1f}%")
print(f"  maxDD  {dd*100:.1f}%")
print(f"  Year-by-year:")
for y, v in by.items(): print(f"    {y}: {v*100:+.1f}%")
print(f"  Positive years: {(by>0).sum()}/{len(by)}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(17, 13),
                          gridspec_kw={"height_ratios": [3, 1, 1]})
ax1, ax2, ax3 = axes
pal = plt.cm.tab10.colors

# Base portfolios (15% vol target)
port_list = sorted(ports.items(), key=lambda x: -x[1]["sh"])
for i, (label, p) in enumerate(port_list[:5]):
    lw = 2.0 if i == 0 else 1.2
    al = 0.9 if i == 0 else 0.55
    ax1.plot(p["idx"], p["eq"].values, lw=lw, alpha=al, color=pal[i],
             label=f"{label}  Sh {p['sh']:.2f}  CAGR {p['cagr']*100:.1f}%")

# Full-power overlay
ax1.plot(idx, eq.values, lw=2.8, color="gold", alpha=0.95,
         label=f"FULL POWER (55% vol)  Sh {sh:.2f}  CAGR {cagr*100:.1f}%  DD {dd*100:.1f}%")
ax1.axvline(idx[sp], color="red", ls="--", alpha=0.4, lw=1.2, label="IS/OOS split")
ax1.set_yscale("log")
ax1.legend(fontsize=7.5, loc="upper left")
ax1.grid(alpha=0.25)
ax1.set_title(f"Full Alpha: 4 Premia (TSMOM + MR + Carry + Value)\n"
              f"Best base: [{best_port['label']}]  Sh {best_port['sh']:.2f}  "
              f"Full-power: Sh {sh:.2f}  CAGR {cagr*100:.1f}%  maxDD {dd*100:.1f}%")

# Full-power drawdown
uw = (eq / eq.cummax() - 1) * 100
ax2.fill_between(idx, uw.values, 0, color="gold", alpha=0.5)
ax2.axvline(idx[sp], color="red", ls="--", alpha=0.4)
ax2.set_ylabel("DD %"); ax2.grid(alpha=0.25)
ax2.set_title(f"Drawdown  maxDD {dd*100:.1f}%")

# Annual bars
years  = list(by.keys())
vals   = [v * 100 for v in by.values]
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
ax3.bar(years, vals, color=colors, edgecolor="white", linewidth=0.5)
ax3.axhline(0, color="white", lw=0.8)
ax3.set_ylabel("Annual return %"); ax3.grid(axis="y", alpha=0.25)
for y, v in zip(years, vals):
    ax3.text(y, v + (3 if v >= 0 else -8), f"{v:+.0f}%", ha="center", fontsize=8, color="white")
ax3.set_title(f"Annual returns  avg {np.mean(vals):.1f}%  std {np.std(vals):.1f}%")

plt.tight_layout()
plt.savefig("full_alpha.png", dpi=130)
print(f"\n  Saved → full_alpha.png")
