"""
3-Instrument Ensemble for The5ers — ADAPTIVE risk management to lift pass rate.

Same edge (SP500 MR + USDJPY MR + XAGUSD TSMOM) but the CHALLENGE is run with
smart money-management instead of flat sizing:

  1. Buffer-scaled risk  — risk small at the start, scale up only after building
                           a profit cushion above the bust floor. Early stumbles
                           no longer bust you.
  2. DD circuit breaker  — when trailing DD from peak exceeds a soft threshold,
                           cut risk hard to avoid touching the 6% hard limit.
  3. Profit lock / coast — once within reach of +8%, shrink size to glide in
                           rather than risk giving it back.

Goal: pass rate >= 70% on a single attempt.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from multi_asset_scan import load_raw

ann = 252
FILES = {
    "SP500":  "SP500_M15_202202230100_202606101815.csv",
    "USDJPY": "USDJPY_M15_202205230000_202606101815.csv",
    "XAGUSD": "XAGUSD_M15_202203171815_202606101815.csv",
}
HALF = {"SP500": 0.5, "USDJPY": 0.008, "XAGUSD": 0.02}

def load_m15(fp):
    df = load_raw(fp); return df[df["close"] > 0]
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
def tsmom(p, L): return np.sign(p.pct_change(L)).fillna(0)

data = {s: load_m15(f) for s, f in FILES.items()}
d1s  = {s: to_d1(m) for s, m in data.items()}
sigs = {
    "SP500":  mr_sig(d1s["SP500"]["close"], 10, 2.0),
    "USDJPY": mr_sig(d1s["USDJPY"]["close"], 15, 2.5),
    "XAGUSD": tsmom(d1s["XAGUSD"]["close"], 200),
}
VT, CAP = 0.15, 3.0
def stream_net(sym):
    d1 = to_d1(data[sym]); p = d1["close"]; h = HALF[sym]
    ret = p.pct_change().fillna(0)
    rv  = ret.rolling(20).std() * np.sqrt(ann)
    sig = sigs[sym].reindex(p.index).fillna(0)
    pos = (sig * (VT / rv.replace(0, np.nan)).clip(upper=CAP).fillna(0)).shift(1).fillna(0)
    trn = pos.diff().abs().fillna(pos.abs())
    return (pos * ret - trn * (h / p)).fillna(0)
nets = {s: stream_net(s) for s in FILES}
common = nets["SP500"].index
for s in FILES: common = common.intersection(nets[s].index)
port_d = sum((1/3) * nets[s].reindex(common).fillna(0) for s in FILES)

# ── FLAT sizing baseline ──────────────────────────────────────────────────────
def challenge_flat(r_base, target=0.08, max_dd=0.06, daily_lim=0.05,
                   win_days=60, risk_scale=1.0, iters=8000, seed=7):
    rng = np.random.default_rng(seed); r = port_d.values * risk_scale; N = len(r)
    P = B = 0; dtp = []
    for _ in range(iters):
        s = rng.integers(0, max(1, N - win_days)); w = r[s:s+win_days]
        bal = 1.0; peak = 1.0; out = None
        for k, x in enumerate(w, 1):
            if x <= -daily_lim: out = "B"; break
            bal *= (1 + x); peak = max(peak, bal)
            if (peak - bal)/peak >= max_dd: out = "B"; break
            if bal - 1 >= target: out = "P"; dtp.append(k); break
        if out == "P": P += 1
        elif out == "B": B += 1
    return P/iters*100, B/iters*100, (np.median(dtp) if dtp else np.nan)

# ── ADAPTIVE sizing ───────────────────────────────────────────────────────────
def challenge_adaptive(target=0.08, max_dd=0.06, daily_lim=0.05, win_days=60,
                       base_risk=8.0, max_risk=18.0,
                       soft_dd=0.045, coast_at=0.90, iters=8000, seed=7):
    """
    base_risk : starting risk scale (cushion-builder, kept reasonably high so we
                actually make progress toward target)
    max_risk  : ceiling once a profit buffer exists
    soft_dd   : trailing DD beyond which we cut risk; brake stays ~1.0 until
                HALF of soft_dd, then ramps down (late-activating, not early)
    coast_at  : fraction of target after which we shrink to glide in
    """
    rng = np.random.default_rng(seed); r = port_d.values; N = len(r)
    P = B = NR = 0; dtp = []
    for _ in range(iters):
        s = rng.integers(0, max(1, N - win_days)); w = r[s:s+win_days]
        bal = 1.0; peak = 1.0; out = None
        for k, x in enumerate(w, 1):
            prog = bal - 1.0                      # current profit
            tdd  = (peak - bal)/peak              # trailing DD now

            # buffer factor: scale risk up as profit grows
            buf_f = base_risk + (max_risk - base_risk) * min(max(prog,0)/target, 1.0)
            # DD brake: full risk until half soft_dd, then ramp to 0 at soft_dd
            half = soft_dd * 0.5
            if tdd <= half:
                dd_brake = 1.0
            elif tdd >= soft_dd:
                dd_brake = 0.0
            else:
                dd_brake = 1.0 - (tdd - half) / (soft_dd - half)
            # coast brake: once past coast_at*target, halve risk to glide in
            coast = 0.5 if prog >= coast_at * target else 1.0

            rs = buf_f * dd_brake * coast
            xr = x * rs
            if xr <= -daily_lim: out = "B"; break
            bal *= (1 + xr); peak = max(peak, bal)
            if (peak - bal)/peak >= max_dd: out = "B"; break
            if bal - 1 >= target: out = "P"; dtp.append(k); break
        if out == "P": P += 1
        elif out == "B": B += 1
        else: NR += 1
    return dict(passp=P/iters*100, bustp=B/iters*100, nrp=NR/iters*100,
                med=(np.median(dtp) if dtp else np.nan))

print("="*72)
print("  THE5ERS — ADAPTIVE money management to lift pass rate")
print("  (target +8%, trailing DD 6%, daily 5%, 60 days)")
print("="*72)

print("\n  FLAT sizing baseline:")
print(f"  {'scale':>6}{'pass%':>8}{'bust%':>8}{'med_days':>10}")
for rs in [3, 5, 7, 9]:
    p, b, m = challenge_flat(None, risk_scale=rs)
    print(f"  {rs:>6}{p:>7.1f}%{b:>7.1f}%{m:>10.1f}")

print("\n  ADAPTIVE sizing sweep (base_risk, max_risk, soft_dd):")
print(f"  {'base':>5}{'max':>5}{'softDD':>8}{'pass%':>8}{'bust%':>8}{'none%':>8}{'med':>7}")
best = None
for base_risk in [6, 8, 10, 12]:
    for max_risk in [14, 18, 24]:
        for soft_dd in [0.035, 0.045, 0.050]:
            d = challenge_adaptive(base_risk=base_risk, max_risk=max_risk, soft_dd=soft_dd)
            star = " ⭐" if d["passp"] >= 70 and d["bustp"] <= 25 else ""
            if d["passp"] >= 60:
                print(f"  {base_risk:>5}{max_risk:>5}{soft_dd:>8.3f}"
                      f"{d['passp']:>7.1f}%{d['bustp']:>7.1f}%{d['nrp']:>7.1f}%{d['med']:>7.0f}{star}")
            # score: maximize pass, penalize bust
            score = d["passp"] - 1.5 * d["bustp"]
            if best is None or score > best[0]:
                best = (score, base_risk, max_risk, soft_dd, d)

print("\n" + "="*72)
sc, br, mr, sd, d = best
print(f"  ★ BEST (trailing-DD assumption):  base={br} max={mr} soft_dd={sd}")
print(f"     PASS {d['passp']:.1f}%  BUST {d['bustp']:.1f}%  "
      f"none {d['nrp']:.1f}%  median {d['med']:.0f}d")
print("="*72)


# ── STATIC max-loss variant (the actual 5ers rule: loss from INITIAL balance) ──
def challenge_static(target=0.08, max_loss=0.06, daily_lim=0.05, win_days=60,
                     base_risk=8.0, max_risk=18.0, brake_at=0.04,
                     coast_at=0.90, iters=8000, seed=7):
    """
    max_loss : bust if balance falls below (1 - max_loss) of INITIAL (static floor).
    brake_at : when balance is within this distance of the static floor, cut risk.
    Buffer-scaled risk + static-floor brake + coast.
    """
    rng = np.random.default_rng(seed); r = port_d.values; N = len(r)
    floor = 1.0 - max_loss
    P = B = NR = 0; dtp = []
    for _ in range(iters):
        s = rng.integers(0, max(1, N - win_days)); w = r[s:s+win_days]
        bal = 1.0; out = None
        for k, x in enumerate(w, 1):
            prog = bal - 1.0
            # buffer factor: scale up risk as profit grows
            buf_f = base_risk + (max_risk - base_risk) * min(max(prog,0)/target, 1.0)
            # static-floor brake: cut risk as balance approaches floor
            dist = bal - floor                       # distance above bust floor
            fl_brake = min(1.0, max(0.0, dist / brake_at))
            coast = 0.5 if prog >= coast_at * target else 1.0
            rs = buf_f * fl_brake * coast
            xr = x * rs
            if xr <= -daily_lim: out = "B"; break
            bal *= (1 + xr)
            if bal <= floor: out = "B"; break
            if bal - 1 >= target: out = "P"; dtp.append(k); break
        if out == "P": P += 1
        elif out == "B": B += 1
        else: NR += 1
    return dict(passp=P/iters*100, bustp=B/iters*100, nrp=NR/iters*100,
                med=(np.median(dtp) if dtp else np.nan))

print("\n" + "="*72)
print("  STATIC max-loss variant (6% from INITIAL balance — actual 5ers rule)")
print("="*72)
print(f"  {'base':>5}{'max':>5}{'brake':>7}{'pass%':>8}{'bust%':>8}{'none%':>8}{'med':>7}")
best_s = None
for base_risk in [6, 8, 10, 12]:
    for max_risk in [14, 18, 24]:
        for brake_at in [0.03, 0.04, 0.05]:
            d = challenge_static(base_risk=base_risk, max_risk=max_risk, brake_at=brake_at)
            star = " ⭐" if d["passp"] >= 70 and d["bustp"] <= 25 else ""
            if d["passp"] >= 65:
                print(f"  {base_risk:>5}{max_risk:>5}{brake_at:>7.3f}"
                      f"{d['passp']:>7.1f}%{d['bustp']:>7.1f}%{d['nrp']:>7.1f}%{d['med']:>7.0f}{star}")
            score = d["passp"] - 1.5 * d["bustp"]
            if best_s is None or score > best_s[0]:
                best_s = (score, base_risk, max_risk, brake_at, d)

sc, br, mr, ba, d = best_s
print("\n" + "="*72)
print(f"  ★ BEST (static-loss, real 5ers):  base={br} max={mr} brake_at={ba}")
print(f"     PASS {d['passp']:.1f}%  BUST {d['bustp']:.1f}%  "
      f"none {d['nrp']:.1f}%  median {d['med']:.0f}d")
print("="*72)


# ── NO TIME LIMIT variant (the actual 5ers selling point) ─────────────────────
# Daily-loss-aware: cap per-day risk so a bad day can't breach the 5% daily limit.
def challenge_notimelimit(target=0.08, max_loss=0.06, daily_lim=0.05,
                          base_risk=4.0, max_risk=10.0, brake_at=0.04,
                          day_vol_cap=True, max_horizon=400,
                          iters=8000, seed=7):
    """No 60-day clock. Trade until pass or bust (or run out of sampled history).
       Per-day risk capped so worst-case daily move stays within daily_lim."""
    rng = np.random.default_rng(seed); r = port_d.values; N = len(r)
    floor = 1.0 - max_loss
    # estimate a 'bad day' as the 1st-percentile daily return magnitude
    bad_day = abs(np.percentile(r, 1))            # ~worst 1% daily loss (unscaled)
    P = B = NR = 0; dtp = []
    for _ in range(iters):
        s = rng.integers(0, max(1, N - max_horizon)); w = r[s:s+max_horizon]
        bal = 1.0; out = None
        for k, x in enumerate(w, 1):
            prog = bal - 1.0
            buf_f = base_risk + (max_risk - base_risk) * min(max(prog,0)/target, 1.0)
            dist = bal - floor
            fl_brake = min(1.0, max(0.0, dist / brake_at))
            rs = buf_f * fl_brake
            # cap so a bad day stays inside daily limit (with margin)
            if day_vol_cap and bad_day > 0:
                rs = min(rs, 0.8 * daily_lim / bad_day)
            xr = x * rs
            if xr <= -daily_lim: out = "B"; break
            bal *= (1 + xr)
            if bal <= floor: out = "B"; break
            if bal - 1 >= target: out = "P"; dtp.append(k); break
        if out == "P": P += 1
        elif out == "B": B += 1
        else: NR += 1
    return dict(passp=P/iters*100, bustp=B/iters*100, nrp=NR/iters*100,
                med=(np.median(dtp) if dtp else np.nan))

print("\n" + "="*72)
print("  NO-TIME-LIMIT variant (5ers has no clock) + daily-loss-safe sizing")
print("="*72)
print(f"  {'base':>5}{'max':>5}{'brake':>7}{'pass%':>8}{'bust%':>8}{'none%':>8}{'med':>7}")
best_n = None
for base_risk in [2, 3, 4, 5]:
    for max_risk in [6, 8, 10]:
        for brake_at in [0.03, 0.04, 0.05]:
            d = challenge_notimelimit(base_risk=base_risk, max_risk=max_risk, brake_at=brake_at)
            star = " ⭐" if d["passp"] >= 70 and d["bustp"] <= 25 else ""
            if d["passp"] >= 65:
                print(f"  {base_risk:>5}{max_risk:>5}{brake_at:>7.3f}"
                      f"{d['passp']:>7.1f}%{d['bustp']:>7.1f}%{d['nrp']:>7.1f}%{d['med']:>7.0f}{star}")
            score = d["passp"] - 2.0 * d["bustp"]
            if best_n is None or score > best_n[0]:
                best_n = (score, base_risk, max_risk, brake_at, d)

sc, br, mr, ba, d = best_n
print("\n" + "="*72)
print(f"  ★ BEST (no-time-limit, real 5ers):  base={br} max={mr} brake_at={ba}")
print(f"     PASS {d['passp']:.1f}%  BUST {d['bustp']:.1f}%  "
      f"none {d['nrp']:.1f}%  median {d['med']:.0f}d")
print("="*72)
