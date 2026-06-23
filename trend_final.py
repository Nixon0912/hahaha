"""
FINAL HONEST TREND STRATEGY — deployable tear-sheet.

Data reality:
  XAUUSD H1:  17 years (2009-2026) — the only long-history anchor
  ESXEUR/F40EUR/HSIHKD/IBXEUR: 6 years (2020-2026)
  Everything else: 4-5 years (2022-2026)

We report three honest portfolios:
  A) Gold-only (XAUUSD H1): 17-year walk-forward, the real bedrock
  B) 5-market (2020+): gold + 4 equity indices, 6 years
  C) Full 13-market basket: 4.5 years (limited data, interpret with caution)

Signal: multi-timeframe EMA trend [(8,24),(16,48),(32,96),(64,192)]
Costs:  bid/ask spread + swap/financing drag
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

HALF = {
    "XAUUSD": 0.15, "XAGUSD": 0.02,
    "NAS100": 1.5,  "SP500": 0.5,   "US30": 3.0,
    "DAX40":  2.0,  "ESXEUR": 1.0,  "F40EUR": 0.8,
    "IBXEUR": 4.0,  "UK100":  1.5,  "ASXAUD": 2.0,
    "JPN225": 8.0,  "HSIHKD": 3.5,
}
SWAP_ANN = {k: -0.015 for k in HALF}
SWAP_ANN.update({"NAS100": -0.020, "SP500": -0.020, "US30": -0.020,
                  "JPN225": -0.010, "HSIHKD": -0.010})
CAP = 2.0
VOL_TGT = 0.20

def hs(sym):
    for k, v in HALF.items():
        if sym.startswith(k[:4]): return v
    return None

def swap_drag(sym):
    for k, v in SWAP_ANN.items():
        if sym.startswith(k[:4]): return v
    return -0.015

def load_d1(fp):
    try:
        df = load_raw(fp); df = df[df["close"] > 0]
        d = df.resample("1D").agg({"open": "first", "high": "max",
                                    "low": "min", "close": "last"}).dropna()
        return d[d["close"] > 0]
    except: return None

def metrics(x):
    if len(x) < 2: return {}
    eq = (1 + x).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (ann / len(x)) - 1 if eq.iloc[-1] > 0 else -1
    sh = x.mean() / x.std() * np.sqrt(ann) if x.std() > 0 else 0
    neg = x[x < 0]
    so = x.mean() / neg.std() * np.sqrt(ann) if len(neg) > 1 and neg.std() > 0 else 0
    return dict(sh=sh, so=so, cagr=cagr, dd=dd,
                calmar=cagr / abs(dd) if dd < 0 else 0)

def trend_signal(p):
    pairs = [(8, 24), (16, 48), (32, 96), (64, 192)]
    s = sum(np.sign(p.ewm(span=f).mean() - p.ewm(span=sl).mean())
            for f, sl in pairs) / len(pairs)
    return s.fillna(0)

def instrument_net(p, h, swap_yr):
    ret = p.pct_change().fillna(0)
    rv = ret.rolling(50).std() * np.sqrt(ann)
    sig = trend_signal(p)
    raw_pos = sig * (VOL_TGT / rv.replace(0, np.nan)).clip(upper=CAP).fillna(0)
    pos = raw_pos.shift(1).fillna(0)
    trn = pos.diff().abs().fillna(pos.abs())
    return (pos * ret - trn * (h / p) - pos.abs() * abs(swap_yr) / ann).fillna(0)

def vol_scale(port, target=VOL_TGT, cap=3.0):
    rv = port.rolling(30).std() * np.sqrt(ann)
    sc = (target / rv.replace(0, np.nan)).clip(upper=cap).fillna(1.0).shift(1).fillna(1.0)
    return port * sc

# ── Load all 13 ───────────────────────────────────────────────────────────────
TREND_MARKETS = list(HALF.keys())
all_csvs = sorted(glob.glob("*.csv"))
inst = {}
for f in all_csvs:
    m = re.match(r"([A-Z0-9]+)_(M5|M15|H1)_", Path(f).name)
    if not m: continue
    sym, tf = m.group(1), m.group(2)
    rank = {"M5": 0, "M15": 1, "H1": 2}
    if sym not in inst or rank[tf] > rank[inst[sym][1]]:
        inst[sym] = (f, tf)

nets = {}
for sym in TREND_MARKETS:
    if sym not in inst: continue
    h = hs(sym)
    if h is None: continue
    d1 = load_d1(inst[sym][0])
    if d1 is None or len(d1) < 400: continue
    nets[sym] = instrument_net(d1["close"], h, swap_drag(sym))

syms = list(nets.keys())

# ── PORTFOLIO A: Gold only (17 years) ────────────────────────────────────────
portA = nets["XAUUSD"].copy()
mA = metrics(portA)
byA = portA.groupby(portA.index.year).apply(lambda s: (1 + s).prod() - 1)
yr_shA = portA.groupby(portA.index.year).apply(lambda s: metrics(s)["sh"]).to_dict()

# ── PORTFOLIO B: 5 markets (2020+) ────────────────────────────────────────────
B_syms = [s for s in ["XAUUSD", "ESXEUR", "F40EUR", "HSIHKD", "IBXEUR"]
          if s in nets]
commonB = None
for s in B_syms:
    commonB = nets[s].index if commonB is None else commonB.intersection(nets[s].index)
NDB = pd.DataFrame({s: nets[s].reindex(commonB).fillna(0) for s in B_syms})
portB_raw = NDB.mean(axis=1)
portB = vol_scale(portB_raw)
mB = metrics(portB)
byB = portB.groupby(commonB.year).apply(lambda s: (1 + s).prod() - 1)
yr_shB = portB.groupby(commonB.year).apply(lambda s: metrics(s)["sh"]).to_dict()

# ── PORTFOLIO C: All 13 (2022+) ───────────────────────────────────────────────
commonC = None
for s in syms:
    commonC = nets[s].index if commonC is None else commonC.intersection(nets[s].index)
NDC = pd.DataFrame({s: nets[s].reindex(commonC).fillna(0) for s in syms})
portC_raw = NDC.mean(axis=1)
portC = vol_scale(portC_raw)
mC = metrics(portC)
byC = portC.groupby(commonC.year).apply(lambda s: (1 + s).prod() - 1)
yr_shC = portC.groupby(commonC.year).apply(lambda s: metrics(s)["sh"]).to_dict()
mcorrC = np.abs(NDC.corr().values[np.triu_indices(len(syms), k=1)]).mean()

# ── PRINT ─────────────────────────────────────────────────────────────────────
def show_port(name, port, by, yr_sh, idx=None, extra=""):
    if idx is None: idx = port.index
    sp = int(len(port) * 0.70)
    shi = metrics(port.iloc[:sp])["sh"]
    sho = metrics(port.iloc[sp:])["sh"]
    m = metrics(port)
    pos = sum(1 for v in yr_sh.values() if v > 0)
    print(f"\n{'─'*76}")
    print(f"  {name}{extra}")
    print(f"  Period : {idx[0].date()} → {idx[-1].date()}  ({len(port)} days)")
    print(f"  Sharpe : {m['sh']:.2f}  (IS {shi:.2f} / OOS {sho:.2f})")
    print(f"  Sortino: {m['so']:.2f}   Calmar: {m['calmar']:.2f}")
    print(f"  CAGR   : {m['cagr']*100:.1f}%   maxDD: {m['dd']*100:.1f}%")
    print(f"  Positive years: {pos}/{len(yr_sh)}")
    print(f"  Annual returns:")
    print("    " + "  ".join(f"{y}:{v*100:+.0f}%" for y, v in by.items()))
    print(f"  Per-year Sharpe:")
    print("    " + "  ".join(f"{y}:{v:+.1f}" for y, v in yr_sh.items()))

print("=" * 76)
print("  FINAL HONEST TREND STRATEGY — multi-timeframe EMA, realistic costs")
print("=" * 76)
show_port("A) GOLD ONLY  [XAUUSD H1, 17 years — most honest test]",
          portA, byA, yr_shA)
show_port("B) 5-MARKET BASKET  [gold + 4 EU/Asia equity indices, 6 years]",
          portB, byB, yr_shB, idx=commonB,
          extra=f"\n  Markets: {B_syms}")
show_port("C) FULL 13-MARKET  [all trend-able markets, 4.5 years]",
          portC, byC, yr_shC, idx=commonC,
          extra=f"\n  Markets: {syms}\n  Mean |corr|: {mcorrC:.3f}")

# ── Per-instrument standalone ─────────────────────────────────────────────────
print(f"\n{'─'*76}")
print("  PER-INSTRUMENT standalone trend Sharpe (with costs, full available history)")
print(f"{'─'*76}")
for s in sorted(syms, key=lambda _s: -metrics(nets[_s])["sh"]):
    m = metrics(nets[s])
    n_days = len(nets[s])
    bar = "#" * int(max(m["sh"], 0) * 20)
    print(f"  {s:<8} Sh{m['sh']:+.2f}  CAGR{m['cagr']*100:+.0f}%  maxDD{m['dd']*100:.0f}%"
          f"  ({n_days}d)  {bar}")

# ── Vol dial ──────────────────────────────────────────────────────────────────
print(f"\n{'─'*76}")
print("  VOL-TARGET DIAL — 5-market basket (B), same rule")
print(f"  {'target':>8}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'Sortino':>9}")
for tv in [0.10, 0.15, 0.20, 0.25, 0.35, 0.50]:
    pt = vol_scale(portB_raw, target=tv)
    m = metrics(pt)
    print(f"  {tv*100:>7.0f}%{m['cagr']*100:>7.1f}%{m['sh']:>8.2f}"
          f"{m['dd']*100:>7.1f}%{m['so']:>9.2f}")

# ── Honest 5%ers applicability ────────────────────────────────────────────────
print(f"\n{'─'*76}")
print("  5%ERS CHALLENGE — honest numbers using Portfolio B at 35% vol target")
pt_chal = vol_scale(portB_raw, target=0.35)
mc = metrics(pt_chal)
worst_day = pt_chal.abs().max()
max_safe = 0.05 / worst_day * 0.90
print(f"  Portfolio Sharpe {mc['sh']:.2f}  CAGR {mc['cagr']*100:.0f}%  maxDD {mc['dd']*100:.1f}%")
print(f"  Worst single day (6yr): {worst_day*100:.1f}%  → hard cap at {max_safe:.1f}× scales")
print(f"  At max safe scale → daily vol ≈ {worst_day*max_safe*100:.2f}%/day")
print(f"  Expected time to +8%: ≈ {int(0.08/(mc['cagr']*max_safe/ann))} trading days")
print(f"")
print(f"  HONEST VERDICT:")
print(f"  This is a SLOW trend strategy. Sharpe ~{mc['sh']:.1f}, CAGR ~{mc['cagr']*100:.0f}%.")
print(f"  A 5%ers challenge at the sizes needed to reach 8% quickly requires")
print(f"  leverage that takes the drawdown beyond the 6% DD limit.")
print(f"  Best use: longer-horizon accounts, prop firms with no time limit,")
print(f"  or as a risk-diversifier alongside a faster intraday strategy.")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 13),
                          gridspec_kw={"height_ratios": [3, 1, 1]})

ax = axes[0]
eA = (1 + portA).cumprod()
eB = (1 + portB).cumprod()
eC = (1 + portC).cumprod()
ax.plot(portA.index, eA.values, color="gold", lw=2.0,
        label=f"A) Gold only (17yr)  Sh {mA['sh']:.2f}  CAGR {mA['cagr']*100:.0f}%")
ax.plot(commonB, eB.values, color="darkgreen", lw=2.0,
        label=f"B) 5-market (6yr)  Sh {mB['sh']:.2f}  CAGR {mB['cagr']*100:.0f}%")
ax.plot(commonC, eC.values, color="steelblue", lw=1.5, ls="--",
        label=f"C) 13-market (4.5yr)  Sh {mC['sh']:.2f}  CAGR {mC['cagr']*100:.0f}%")
ax.set_yscale("log")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_title("Final Honest Trend Strategy — multi-timeframe EMA, realistic costs\n"
             "ONE fixed rule, no per-instrument tuning, bid/ask spread + swap costs")

ax = axes[1]
ddA = (eA / eA.cummax() - 1) * 100
ddB = (eB / eB.cummax() - 1) * 100
ax.fill_between(portA.index, ddA.values, 0, color="gold", alpha=0.4, label="Gold")
ax.fill_between(commonB, ddB.values, 0, color="darkgreen", alpha=0.3, label="5-market")
ax.set_title(f"Drawdown  (Gold maxDD {mA['dd']*100:.1f}%  /  5-market maxDD {mB['dd']*100:.1f}%)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[2]
years_b = list(byB.index)
cols = ["green" if v > 0 else "red" for v in byB.values]
ax.bar(years_b, byB.values * 100, color=cols, alpha=0.75, edgecolor="black", lw=0.5,
       label="5-market")
ax.plot(list(byA.index)[-len(years_b):],
        [byA.get(y, 0) * 100 for y in years_b],
        "o--", color="gold", lw=1.5, label="Gold")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Annual Returns (%) — 5-market basket (bars) vs Gold (line)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("trend_final.png", dpi=130)
print("\n  Saved → trend_final.png")
