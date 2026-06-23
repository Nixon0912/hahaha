"""
3-Instrument Ensemble for The5ers challenge
  SP500  — MR  (mean-reversion, z>=2 on 10d)
  USDJPY — MR  (mean-reversion, z>=2.5 on 15d)
  XAGUSD — TSMOM 200d (trend)

Full metrics + INTRADAY drawdown (from M15 bars, not daily close) +
The5ers challenge Monte Carlo (pass probability).
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multi_asset_scan import load_raw

ann = 252
SF  = 0.70

FILES = {
    "SP500":  "SP500_M15_202202230100_202606101815.csv",
    "USDJPY": "USDJPY_M15_202205230000_202606101815.csv",
    "XAGUSD": "XAGUSD_M15_202203171815_202606101815.csv",
}
HALF = {"SP500": 0.5, "USDJPY": 0.008, "XAGUSD": 0.02}

def load_m15(fp):
    df = load_raw(fp); df = df[df["close"] > 0]
    return df

def to_d1(m15):
    d = m15.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    return d[d["close"] > 0]

def mr_sig(p, n, e):
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

def tsmom(p, L):
    return np.sign(p.pct_change(L)).fillna(0)

# ── Build per-instrument daily net returns at a chosen vol target ──────────────
def stream_net(sym, m15, sig_d1, vt, cap):
    d1  = to_d1(m15)
    p   = d1["close"]
    h   = HALF[sym]
    ret = p.pct_change().fillna(0)
    rv  = ret.rolling(20).std() * np.sqrt(ann)
    sig = sig_d1.reindex(p.index).fillna(0)
    pos = (sig * (vt / rv.replace(0, np.nan)).clip(upper=cap).fillna(0)).shift(1).fillna(0)
    trn = pos.diff().abs().fillna(pos.abs())
    net = (pos * ret - trn * (h / p)).fillna(0)
    return net, pos, p

# Load
data = {s: load_m15(f) for s, f in FILES.items()}
d1s  = {s: to_d1(m) for s, m in data.items()}

# Signals (computed on D1 close)
sigs = {
    "SP500":  mr_sig(d1s["SP500"]["close"], 10, 2.0),
    "USDJPY": mr_sig(d1s["USDJPY"]["close"], 15, 2.5),
    "XAGUSD": tsmom(d1s["XAGUSD"]["close"], 200),
}

VT, CAP = 0.15, 3.0   # base per-stream vol target
nets, poss, prices = {}, {}, {}
for s in FILES:
    n, pos, p = stream_net(s, data[s], sigs[s], VT, CAP)
    nets[s], poss[s], prices[s] = n, pos, p

# Common daily index
common = nets["SP500"].index
for s in FILES: common = common.intersection(nets[s].index)
port_d = sum((1/3) * nets[s].reindex(common).fillna(0) for s in FILES)

# ── Full metrics helper ────────────────────────────────────────────────────────
def metrics(net, idx, label):
    eq   = (1 + net).cumprod()
    n    = len(net)
    sh   = net.mean() / net.std() * np.sqrt(ann) if net.std() > 0 else 0
    down = net[net < 0]
    sortino = net.mean() / down.std() * np.sqrt(ann) if len(down) and down.std() > 0 else np.nan
    dd_s = (eq / eq.cummax() - 1)
    mdd  = dd_s.min()
    cagr = eq.iloc[-1] ** (ann / n) - 1 if eq.iloc[-1] > 0 else -1
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    sp   = int(n * SF)
    si, so = net.iloc[:sp], net.iloc[sp:]
    shi  = si.mean() / si.std() * np.sqrt(ann) if si.std() > 0 else 0
    sho  = so.mean() / so.std() * np.sqrt(ann) if so.std() > 0 else 0
    pos_days = (net > 0).sum(); neg_days = (net < 0).sum()
    win_rate = pos_days / (pos_days + neg_days) if (pos_days+neg_days) else 0
    gross_win = net[net > 0].sum(); gross_loss = -net[net < 0].sum()
    pf   = gross_win / gross_loss if gross_loss > 0 else np.nan
    # monthly
    mo   = net.groupby([idx.year, idx.month]).apply(lambda s: (1+s).prod()-1)
    by   = net.groupby(idx.year).apply(lambda s: (1+s).prod()-1)
    best_mo, worst_mo = mo.max(), mo.min()
    pos_mo = (mo > 0).sum() / len(mo)
    avg_dd_dur = None
    return dict(label=label, eq=eq, sh=sh, sortino=sortino, mdd=mdd, cagr=cagr,
                calmar=calmar, shi=shi, sho=sho, win_rate=win_rate, pf=pf,
                by=by, mo=mo, best_mo=best_mo, worst_mo=worst_mo, pos_mo=pos_mo,
                n=n, idx=idx)

# ── Intraday equity for true trailing DD ──────────────────────────────────────
# Reconstruct intraday equity by applying each day's position to the M15 returns
def intraday_equity(vt, cap):
    """Build a fine-grained equity curve using M15 bars within each day so the
    trailing drawdown reflects intraday swings, not just daily closes."""
    # Per-instrument daily positions (already lagged 1 day = known at day open)
    pieces = []
    for s in FILES:
        m15 = data[s].copy()
        d1  = to_d1(m15)
        p   = d1["close"]
        ret = p.pct_change().fillna(0)
        rv  = ret.rolling(20).std() * np.sqrt(ann)
        sig = sigs[s].reindex(p.index).fillna(0)
        pos = (sig * (vt / rv.replace(0, np.nan)).clip(upper=cap).fillna(0)).shift(1).fillna(0)
        # map daily position onto M15 bars of that day
        m15 = m15.copy()
        m15["day"] = m15.index.normalize()
        posmap = pos.copy(); posmap.index = posmap.index.normalize()
        m15["pos"] = m15["day"].map(posmap).fillna(0).values
        m15["r15"] = m15["close"].pct_change().fillna(0)
        m15["pnl15"] = (1/3) * m15["pos"] * m15["r15"]   # equal weight 1/3
        pieces.append(m15["pnl15"].rename(s))
    allpnl = pd.concat(pieces, axis=1).fillna(0)
    port15 = allpnl.sum(axis=1)
    eq15   = (1 + port15).cumprod()
    trail_dd = (eq15 / eq15.cummax() - 1)
    return eq15, trail_dd.min()

# ── The5ers challenge Monte Carlo ─────────────────────────────────────────────
# Rules (High Stakes / Bootcamp typical): target +8%, max trailing DD 6% intraday,
# daily loss 5%, 60-day window. We use daily returns scaled to a chosen risk vol.
def challenge_mc(daily_net, target=0.08, max_dd=0.06, daily_lim=0.05,
                 win_days=60, risk_scale=1.0, iters=4000, seed=7):
    rng = np.random.default_rng(seed)
    r   = daily_net.values * risk_scale
    N   = len(r)
    passes = busts = neither = 0
    days_to_pass = []
    for _ in range(iters):
        start = rng.integers(0, max(1, N - win_days))
        window = r[start:start+win_days]
        bal = 1.0; peak = 1.0; outcome = None
        for k, x in enumerate(window, 1):
            if x <= -daily_lim:           # daily loss breach
                outcome = "B"; break
            bal *= (1 + x); peak = max(peak, bal)
            if (peak - bal) / peak >= max_dd:   # trailing DD breach
                outcome = "B"; break
            if bal - 1.0 >= target:             # target hit
                outcome = "P"; days_to_pass.append(k); break
        if outcome == "P": passes += 1
        elif outcome == "B": busts += 1
        else: neither += 1
    med = np.median(days_to_pass) if days_to_pass else float("nan")
    return dict(passp=passes/iters*100, bustp=busts/iters*100,
                noresult=neither/iters*100, med_days=med)

# ── Compute base-vol metrics ──────────────────────────────────────────────────
M = metrics(port_d, common, "Ensemble (base 15% vol)")

print("="*70)
print("  3-INSTRUMENT ENSEMBLE — SP500 + USDJPY + XAGUSD")
print("  signals: SP500 MR10/2.0 | USDJPY MR15/2.5 | XAGUSD TSMOM200")
print("="*70)

# Individual stream metrics
print("\n  Per-stream (base 15% vol target):")
print(f"  {'sym':<8}{'Sh':>7}{'IS':>7}{'OOS':>7}{'maxDD':>9}{'CAGR':>8}")
for s in FILES:
    m = metrics(nets[s].reindex(common).fillna(0), common, s)
    print(f"  {s:<8}{m['sh']:>7.2f}{m['shi']:>7.2f}{m['sho']:>7.2f}"
          f"{m['mdd']*100:>8.1f}%{m['cagr']*100:>7.1f}%")

# Stream correlation
cdf = pd.DataFrame({s: nets[s].reindex(common).fillna(0) for s in FILES})
cm  = cdf.corr()
upper = cm.values[np.triu_indices(3, k=1)]
print(f"\n  Mean |corr| between streams: {np.abs(upper).mean():.3f}")
print(f"  Pairwise: SP500-USDJPY {abs(cm.loc['SP500','USDJPY']):.2f}  "
      f"SP500-XAGUSD {abs(cm.loc['SP500','XAGUSD']):.2f}  "
      f"USDJPY-XAGUSD {abs(cm.loc['USDJPY','XAGUSD']):.2f}")

print("\n" + "="*70)
print("  FULL METRICS — Ensemble @ base 15% vol target")
print("="*70)
print(f"  Sharpe ratio        {M['sh']:.2f}   (IS {M['shi']:.2f} / OOS {M['sho']:.2f})")
print(f"  Sortino ratio       {M['sortino']:.2f}")
print(f"  Calmar ratio        {M['calmar']:.2f}")
print(f"  CAGR                {M['cagr']*100:.1f}%")
print(f"  maxDD (daily close) {M['mdd']*100:.1f}%")
print(f"  Win rate (days)     {M['win_rate']*100:.1f}%")
print(f"  Profit factor       {M['pf']:.2f}")
print(f"  Best month          {M['best_mo']*100:+.1f}%")
print(f"  Worst month         {M['worst_mo']*100:+.1f}%")
print(f"  Positive months     {M['pos_mo']*100:.0f}%")
print(f"  Year-by-year:")
for y, v in M["by"].items(): print(f"    {y}: {v*100:+.1f}%")

# Intraday DD at base vol
eq15, trail_mdd = intraday_equity(VT, CAP)
print(f"\n  ⚠ maxDD INTRADAY (M15)  {trail_mdd*100:.1f}%   "
      f"(vs daily-close {M['mdd']*100:.1f}%)")
print(f"  Intraday/close DD ratio: {trail_mdd/M['mdd']:.2f}x")

# ── Challenge sizing sweep ────────────────────────────────────────────────────
print("\n" + "="*70)
print("  THE5ERS CHALLENGE  (target +8%, trailing DD 6%, daily 5%, 60 days)")
print("="*70)
print(f"  {'risk_scale':>10}{'~vol':>8}{'pass%':>8}{'bust%':>8}{'none%':>8}{'med_days':>10}")
base_daily_vol = port_d.std()
best_cfg = None
for rs in [3, 5, 7, 9, 12, 15]:
    mc = challenge_mc(port_d, risk_scale=rs)
    approx_vol = base_daily_vol * rs * np.sqrt(ann) * 100
    print(f"  {rs:>10}{approx_vol:>7.0f}%{mc['passp']:>7.1f}%{mc['bustp']:>7.1f}%"
          f"{mc['noresult']:>7.1f}%{mc['med_days']:>10.1f}")
    # pick best by pass% with bust% < 25
    if mc["bustp"] < 25 and (best_cfg is None or mc["passp"] > best_cfg[1]["passp"]):
        best_cfg = (rs, mc)

if best_cfg:
    rs, mc = best_cfg
    print(f"\n  ★ Best risk_scale = {rs}x  →  pass {mc['passp']:.0f}%  "
          f"bust {mc['bustp']:.0f}%  median {mc['med_days']:.0f} days to pass")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios":[3,1]})
ax[0].plot(common, M["eq"].values, lw=2, color="navy", label=f"Ensemble  Sh {M['sh']:.2f}  CAGR {M['cagr']*100:.0f}%")
for s in FILES:
    e = (1 + nets[s].reindex(common).fillna(0)).cumprod()
    ax[0].plot(common, e.values, lw=0.9, alpha=0.5, label=s)
ax[0].axvline(common[int(len(common)*SF)], color="red", ls="--", alpha=0.4)
ax[0].set_yscale("log"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
ax[0].set_title(f"3-Instrument Ensemble (SP500+USDJPY+XAGUSD)  Sh {M['sh']:.2f}  "
                f"CAGR {M['cagr']*100:.1f}%  closeDD {M['mdd']*100:.1f}%  intradayDD {trail_mdd*100:.1f}%")
uw = (M["eq"]/M["eq"].cummax()-1)*100
ax[1].fill_between(common, uw.values, 0, color="navy", alpha=0.4)
ax[1].set_ylabel("DD %"); ax[1].grid(alpha=0.25); ax[1].set_title("Drawdown (daily close)")
plt.tight_layout(); plt.savefig("ensemble_5ers.png", dpi=130)
print("\n  Saved → ensemble_5ers.png")
