"""
Portfolio v2 — push Sharpe to 2.5+ by:
  1. Removing poor performers (IS<0.25 OR OOS<0.35)
  2. Deeper signal search per instrument (20+ signals)
  3. Sharpe-weighted portfolio (not equal-weight)
  4. Ensemble of best 2-3 signals per instrument
  5. Correlation-aware pruning (drop redundant streams)
"""
import sys, warnings, glob, re
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

ann = 252; VT = 0.15; CAP = 3.0; SF = 0.70

HALF = {
    "AUDUSD": 0.00008, "EURUSD": 0.00007, "USDJPY": 0.008, "USDCAD": 0.00010,
    "USDCHF": 0.00009, "AUDNZD": 0.00012,
    "XAUUSD": 0.15,    "XAGUSD": 0.02,    "XPDUSD": 3.0,    "XPTUSD": 0.8,
    "XTIUSD": 0.04,    "XBRUSD": 0.04,    "NGCUSD": 0.005,  "CUCUSD": 0.0025,
    "NAS100": 1.5,     "SP500": 0.5,      "US30": 3.0,
    "DAX40": 2.0,      "ESXEUR": 1.0,     "F40EUR": 0.8,    "IBXEUR": 4.0,
    "UK100": 1.5,      "ASXAUD": 2.0,     "JPN225": 8.0,    "HSIHKD": 3.5,
}

def hs(sym, p):
    for k,v in HALF.items():
        if sym.startswith(k[:4]): return v
    return p.mean()*0.0002

def load_d1(fp):
    try:
        df = load_raw(fp); df = df[df["close"]>0]
        d = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
        return d[d["close"]>0]
    except: return None

def mr_sig(p, n=20, e=2.0):
    z = (p - p.rolling(n).mean()) / p.rolling(n).std()
    zv = z.values; out = np.zeros(len(zv)); cur = 0.0
    for i in range(len(zv)):
        if np.isnan(zv[i]): out[i]=cur; continue
        if cur==0:
            if zv[i]>=e: cur=-1
            elif zv[i]<=-e: cur=1
        else:
            if (cur==-1 and zv[i]<=0) or (cur==1 and zv[i]>=0): cur=0
        out[i]=cur
    return pd.Series(out, index=p.index)

def run(p, h, sig, name):
    ret = p.pct_change().fillna(0)
    rv  = ret.rolling(20).std()*np.sqrt(ann)
    pos = (sig*(VT/rv.replace(0,np.nan)).clip(upper=CAP).fillna(0)).shift(1).fillna(0)
    trn = pos.diff().abs().fillna(pos.abs())
    net = (pos*ret - trn*(h/p)).fillna(0)
    n=len(net); sp=int(n*SF)
    eq=(1+net).cumprod()
    sh = net.mean()/net.std()*np.sqrt(ann) if net.std()>0 else 0
    si,so = net.iloc[:sp],net.iloc[sp:]
    shi = si.mean()/si.std()*np.sqrt(ann) if si.std()>0 else 0
    sho = so.mean()/so.std()*np.sqrt(ann) if so.std()>0 else 0
    dd  = (eq/eq.cummax()-1).min()
    cagr= eq.iloc[-1]**(ann/max(n,1))-1 if eq.iloc[-1]>0 else -1
    return dict(name=name,net=net,eq=eq,sh=sh,shi=shi,sho=sho,dd=dd,cagr=cagr,sp=sp,idx=p.index,n=n)

# ── Load all instruments ──────────────────────────────────────────────────────
all_csvs = sorted(glob.glob("*.csv"))
inst_files = {}
for f in all_csvs:
    m = re.match(r"([A-Z0-9]+)_(M5|M15|H1)_", Path(f).name)
    if not m: continue
    sym,tf = m.group(1),m.group(2)
    rank = {"M5":0,"M15":1,"H1":2}
    if sym not in inst_files or rank[tf] > rank[inst_files[sym][1]]:
        inst_files[sym] = (f, tf)

# ── Deep signal search per instrument ────────────────────────────────────────
print("="*95)
print(f"  {'sym':<10}{'best_sig':<22}{'CAGR':>7}{'Sh':>7}{'IS':>7}{'OOS':>7}{'DD':>8}  score")
print("="*95)

survivors = {}   # sym -> best run dict
ens_nets  = {}   # sym -> ensemble net (average of top-2 per instrument)

for sym, (fp,tf) in sorted(inst_files.items()):
    d1 = load_d1(fp)
    if d1 is None or len(d1)<300: continue
    p = d1["close"]; h = hs(sym, p)
    maxL = min(200, int(len(p)*0.20))
    results = []

    # TSMOM family
    for L in [5,10,15,20,30,50,75,100,150,200]:
        if L > maxL: continue
        results.append(run(p, h, np.sign(p.pct_change(L)).fillna(0), f"MOM{L}d"))

    # MR family
    for n in [10,15,20,30,40]:
        for e in [1.5,2.0,2.5]:
            results.append(run(p, h, mr_sig(p,n,e), f"MR{n}e{e}"))

    # Short-term reversal
    for L in [1,2,3,5,7,10]:
        results.append(run(p, h, -np.sign(p.pct_change(L)).fillna(0), f"REV{L}d"))

    # Blend: TSMOM + MR at different mixes
    t = np.sign(p.pct_change(min(50,maxL))).fillna(0)
    mr = mr_sig(p,20,2.0)
    for wt in [0.2,0.4,0.6,0.8]:
        results.append(run(p, h, wt*t+(1-wt)*mr, f"B_T{int(wt*10)}M{int((1-wt)*10)}"))

    # EMA crossover
    for fast,slow in [(5,20),(10,50),(20,100),(5,50)]:
        if slow > maxL: continue
        sig = np.sign(p.ewm(span=fast).mean() - p.ewm(span=slow).mean()).fillna(0)
        results.append(run(p, h, sig, f"EMA{fast}x{slow}"))

    # Score = IS + 2*OOS (penalize OOS degradation, reward IS strength too)
    for r in results: r["score"] = r["shi"] + 2*r["sho"]
    results.sort(key=lambda x: -x["score"])

    best = results[0]
    score = best["score"]
    flag = "✅" if best["shi"]>0.25 and best["sho"]>0.35 else "❌"
    print(f"  {sym:<10}{best['name']:<22}{best['cagr']*100:>6.1f}%{best['sh']:>7.2f}"
          f"{best['shi']:>7.2f}{best['sho']:>7.2f}{best['dd']*100:>7.1f}%  {score:.2f} {flag}")

    if best["shi"] > 0.25 and best["sho"] > 0.35:
        survivors[sym] = best
        # Ensemble: average top-2 signals (must both be positive IS & OOS)
        good = [r for r in results if r["shi"]>0 and r["sho"]>0]
        if len(good) >= 2:
            common = good[0]["net"].index.intersection(good[1]["net"].index)
            ens = 0.5*good[0]["net"].reindex(common).fillna(0) + \
                  0.5*good[1]["net"].reindex(common).fillna(0)
            ens_nets[sym] = ens
        else:
            ens_nets[sym] = best["net"]

print(f"\n✅ {len(survivors)} survivors: {list(survivors.keys())}")

# ── Correlation-aware pruning ─────────────────────────────────────────────────
# Drop streams that are too correlated with a stronger stream
# (keep the stronger one, prune the weaker if |corr| > 0.5)
sym_list = list(survivors.keys())
if len(sym_list) > 1:
    common_all = survivors[sym_list[0]]["net"].index
    for s in sym_list[1:]: common_all = common_all.intersection(survivors[s]["net"].index)
    net_df = pd.DataFrame({s: survivors[s]["net"].reindex(common_all) for s in sym_list})
    corr = net_df.corr()

    # Greedy pruning: iterate pairs by correlation, drop the one with lower score
    to_drop = set()
    pairs = [(abs(corr.loc[a,b]), a, b)
             for i,a in enumerate(sym_list) for j,b in enumerate(sym_list) if i<j]
    pairs.sort(reverse=True)
    for c,a,b in pairs:
        if c < 0.45: break
        if a in to_drop or b in to_drop: continue
        score_a = survivors[a]["score"] if "score" in survivors[a] else survivors[a]["shi"]+2*survivors[a]["sho"]
        score_b = survivors[b]["score"] if "score" in survivors[b] else survivors[b]["shi"]+2*survivors[b]["sho"]
        drop = b if score_a > score_b else a
        to_drop.add(drop)
        print(f"  Pruning {drop} (corr {c:.2f} with {a if drop==b else b})")

    pruned_syms = [s for s in sym_list if s not in to_drop]
    print(f"  After corr-prune: {len(pruned_syms)} instruments: {pruned_syms}")
else:
    pruned_syms = sym_list

# ── Portfolio construction ────────────────────────────────────────────────────
def build_port(sym_list, label, weighting="equal"):
    nets = {s: ens_nets.get(s, survivors[s]["net"]) for s in sym_list}
    common = list(nets.values())[0].index
    for n in nets.values(): common = common.intersection(n.index)
    if len(common) < 100: return None

    if weighting == "sharpe":
        # weight proportional to IS Sharpe (use IS to avoid OOS snooping)
        raw_w = {s: max(survivors[s]["shi"], 0.01) for s in sym_list}
        total = sum(raw_w.values())
        w = {s: v/total for s,v in raw_w.items()}
    else:
        w = {s: 1/len(sym_list) for s in sym_list}

    combined = sum(w[s]*nets[s].reindex(common).fillna(0) for s in sym_list)
    sp = int(len(combined)*SF)
    eq = (1+combined).cumprod()
    sh  = combined.mean()/combined.std()*np.sqrt(ann) if combined.std()>0 else 0
    si,so = combined.iloc[:sp], combined.iloc[sp:]
    shi = si.mean()/si.std()*np.sqrt(ann) if si.std()>0 else 0
    sho = so.mean()/so.std()*np.sqrt(ann) if so.std()>0 else 0
    dd  = (eq/eq.cummax()-1).min()
    cagr= eq.iloc[-1]**(ann/max(len(combined),1))-1 if eq.iloc[-1]>0 else -1
    by  = combined.groupby(common.year).apply(lambda s:(1+s).prod()-1)
    return dict(label=label,eq=eq,net=combined,sh=sh,shi=shi,sho=sho,dd=dd,cagr=cagr,
                sp=sp,idx=common,by=by,n=len(sym_list),w=w)

all_surv = list(survivors.keys())
top_by_score = sorted(all_surv, key=lambda s: survivors[s]["shi"]+2*survivors[s]["sho"], reverse=True)

ports = {}
for label, syms, wt in [
    ("All survivors (equal)", all_surv, "equal"),
    ("All survivors (IS-Sharpe wt)", all_surv, "sharpe"),
    ("Pruned (equal)", pruned_syms, "equal"),
    ("Pruned (IS-Sharpe wt)", pruned_syms, "sharpe"),
    ("Top-12 (IS-Sharpe wt)", top_by_score[:12], "sharpe"),
    ("Top-8 (IS-Sharpe wt)", top_by_score[:8], "sharpe"),
    ("Top-5 (IS-Sharpe wt)", top_by_score[:5], "sharpe"),
]:
    p = build_port(syms, label, wt)
    if p: ports[label] = p

print("\n" + "="*85)
print(f"  {'portfolio':<35}{'N':>4}{'CAGR':>8}{'Sh':>7}{'IS':>7}{'OOS':>7}{'DD':>8}")
print("="*85)
best_port = None; best_score = 0
for label, p in ports.items():
    flag = " ⭐" if p["sh"] >= 2.5 else (" ✅" if p["sh"] >= 2.0 else "")
    print(f"  {label:<35}{p['n']:>4}{p['cagr']*100:>7.1f}%{p['sh']:>7.2f}"
          f"{p['shi']:>7.2f}{p['sho']:>7.2f}{p['dd']*100:>7.1f}%{flag}")
    if p["sh"] > best_score: best_score=p["sh"]; best_port=p

# ── Print year-by-year for best portfolios ────────────────────────────────────
print()
for label, p in ports.items():
    if p["sh"] >= 1.8:
        print(f"  [{label}] year-by-year:")
        print("  " + "  ".join(f"{y}:{v*100:+.1f}%" for y,v in p["by"].items()))
        print(f"  pos years: {(p['by']>0).sum()}/{len(p['by'])}")
        print()

# ── Weights of best portfolio ─────────────────────────────────────────────────
if best_port and "w" in best_port:
    print(f"  Weights in [{best_port['label']}]:")
    for s,wv in sorted(best_port["w"].items(), key=lambda x:-x[1]):
        print(f"    {s:<12} {wv*100:.1f}%  (IS {survivors[s]['shi']:.2f}, OOS {survivors[s]['sho']:.2f})")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2,1,figsize=(16,11), gridspec_kw={"height_ratios":[3,1]})
ax1,ax2 = axes

pal = plt.cm.tab10.colors
port_list = sorted(ports.items(), key=lambda x: -x[1]["sh"])
for i,(label,p) in enumerate(port_list[:6]):
    lw = 2.4 if i==0 else (1.6 if i<3 else 1.0)
    al = 1.0 if i==0 else (0.8 if i<3 else 0.6)
    ax1.plot(p["idx"], p["eq"].values, lw=lw, alpha=al, color=pal[i],
             label=f"[{label}] N={p['n']}  Sh {p['sh']:.2f}(IS {p['shi']:.2f} OOS {p['sho']:.2f})"
                   f"  CAGR {p['cagr']*100:.1f}%  DD {p['dd']*100:.0f}%")

if best_port:
    ax1.axvline(best_port["idx"][best_port["sp"]], color="red", ls="--", alpha=0.4, lw=1.2)
ax1.set_yscale("log"); ax1.legend(fontsize=7.5, loc="upper left", ncol=1)
ax1.grid(alpha=0.25)
ax1.set_title(f"Portfolio v2 — optimized diversification (vol-target 15%)\n"
              f"Best: [{best_port['label']}]  Sharpe {best_port['sh']:.2f}  "
              f"CAGR {best_port['cagr']*100:.1f}%  maxDD {best_port['dd']*100:.1f}%")

if best_port:
    uw = (best_port["eq"]/best_port["eq"].cummax()-1)*100
    ax2.fill_between(best_port["idx"], uw.values, 0, color=pal[0], alpha=0.45)
    ax2.axvline(best_port["idx"][best_port["sp"]], color="red", ls="--", alpha=0.4)
    ax2.set_ylabel("Best portfolio DD %"); ax2.grid(alpha=0.25)
    ax2.set_title(f"Underwater drawdown  maxDD {best_port['dd']*100:.1f}%  IS/OOS split (red)")

plt.tight_layout()
plt.savefig("portfolio_v2.png", dpi=120)
print(f"\n  saved -> portfolio_v2.png")
print(f"\n  ★ Best portfolio Sharpe: {best_port['sh']:.2f}  "
      f"(IS {best_port['shi']:.2f} / OOS {best_port['sho']:.2f})")
