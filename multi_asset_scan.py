"""
Multi-Asset HTF Scanner — 25 instruments × 3 archetypes × OOS validation

Archetypes:
  ARB — Asian Range Breakout (00:00–08:00 range, entry 08:00–10:00)
  NYO — NY Open Breakout   (10:00–13:00 range, entry 13:00–15:00)
  MOM — H4 EMA20 Pullback Momentum (entry 08:00–20:00)

OOS discipline: first 70% of dates = in-sample, last 30% = out-of-sample.
Only archetypes with positive expR in BOTH IS and OOS count toward portfolio.

Goal: ≤5% bust AND ≤3 months median time-to-pass on The5ers challenge.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings, glob, re, os
warnings.filterwarnings("ignore")

RAW_DIR  = Path(__file__).parent
TP_RR    = 3.5
SL_MULT  = 0.7
SL_LO    = 0.0008   # min SL = 0.08% of price
SL_HI    = 0.006    # max SL = 0.6% of price
FORCE_H  = 21
INIT_BAL = 10_000.0
TARGET   = 0.08
MAX_DD   = 0.10

# Instruments to exclude (tick/spread miscalibrated in earlier research)
EXCLUDE = {"HSIHKD", "JPN225", "US30", "NGCUSD"}


# ── Instrument metadata ───────────────────────────────────────────────────────

def sym_from_path(p: str) -> str:
    return re.match(r"([A-Z0-9]+)_M15_", Path(p).name).group(1)


# ── Data loader (MT5 tab-sep format) ─────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S"
    )
    df.set_index("datetime", inplace=True)
    df.drop(columns=["date", "time"], errors="ignore", inplace=True)
    df.rename(columns={"tickvol": "tick_vol", "vol": "real_vol"}, inplace=True)
    df = df[["open", "high", "low", "close", "tick_vol", "spread"]].copy()
    df = df.sort_index()
    # drop flat/zero bars
    df = df[(df["high"] > df["low"]) & (df["close"] > 0)]
    return df


# ── Technical indicators ──────────────────────────────────────────────────────

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr_s(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-pc).abs(),
                    (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def adx_s(df, n=14):
    a = atr_s(df, n).replace(0, np.nan)
    up = (df["high"]-df["high"].shift(1)).clip(lower=0)
    dn = (df["low"].shift(1)-df["low"]).clip(lower=0)
    dip = 100*up.where(up>=dn,0).ewm(span=n,adjust=False).mean()/a
    dim = 100*dn.where(dn>up,0).ewm(span=n,adjust=False).mean()/a
    dx  = 100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.ewm(span=n, adjust=False).mean()


def build_mtf(m15: pd.DataFrame) -> pd.DataFrame:
    """Build H1, H4, D1 indicators aligned to M15 index (1-bar lag, no lookahead)."""
    def rs(freq):
        return m15.resample(freq, label="left", closed="left").agg(
            {"open":"first","high":"max","low":"min","close":"last","tick_vol":"sum"}
        ).dropna()

    h1 = rs("1h"); h4 = rs("4h"); d1 = rs("1D")

    # H1
    h1["atr"] = atr_s(h1)
    h1["ema50"] = ema(h1["close"], 50)
    h1["trend"] = np.where(h1["close"] > h1["ema50"], 1, -1)

    # H4
    h4["atr"]  = atr_s(h4)
    h4["ema20"] = ema(h4["close"], 20)
    h4["ema50"] = ema(h4["close"], 50)
    h4["adx"]   = adx_s(h4)
    h4["trend"] = np.where(
        (h4["close"] > h4["ema20"]) & (h4["ema20"] >= h4["ema50"]), 1,
        np.where((h4["close"] < h4["ema20"]) & (h4["ema20"] <= h4["ema50"]), -1, 0)
    )

    # D1
    d1["atr"]      = atr_s(d1)
    d1["ema20"]    = ema(d1["close"], 20)
    d1["atr_ma"]   = d1["atr"].rolling(20).mean()
    d1["atr_r"]    = d1["atr"] / d1["atr_ma"].replace(0, np.nan)
    d1["trend"]    = np.where(d1["close"] > d1["ema20"], 1, -1)

    idx = m15.index
    def ff(s): return s.shift(1).reindex(idx, method="ffill")

    R = pd.DataFrame(index=idx)
    R["h1_atr"]   = ff(h1["atr"])
    R["h1_trend"] = ff(h1["trend"]).fillna(0).astype(int)
    R["h4_atr"]   = ff(h4["atr"])
    R["h4_ema20"] = ff(h4["ema20"])
    R["h4_adx"]   = ff(h4["adx"])
    R["h4_trend"] = ff(h4["trend"]).fillna(0).astype(int)
    R["d1_atr_r"] = ff(d1["atr_r"])
    R["d1_trend"] = ff(d1["trend"]).fillna(0).astype(int)
    return R


# ── Range builder ─────────────────────────────────────────────────────────────

def ranges(m15, h_start, h_end):
    mask = (m15.index.hour >= h_start) & (m15.index.hour < h_end)
    g = m15[mask].groupby(m15[mask].index.date)
    r = pd.DataFrame({"hi": g["high"].max(), "lo": g["low"].min()})
    r.index = pd.to_datetime(r.index)
    r["rng"] = r["hi"] - r["lo"]
    return r


# ── Trade resolver ────────────────────────────────────────────────────────────

def resolve(m15, t, d, entry, sl, tp):
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return {"R": 0.0, "result": "skip"}
    day   = m15[m15.index.date == t.date()]
    rem   = day.loc[t:].iloc[1:]
    rem   = rem[rem.index.hour < FORCE_H]
    result = "timeout"
    exit_p = rem.iloc[-1]["close"] if not rem.empty else entry
    for _, rb in rem.iterrows():
        if d == 1:
            if rb["low"]  <= sl: result="sl"; exit_p=sl;  break
            if rb["high"] >= tp: result="tp"; exit_p=tp;  break
        else:
            if rb["high"] >= sl: result="sl"; exit_p=sl;  break
            if rb["low"]  <= tp: result="tp"; exit_p=tp;  break
    R = -1.0 if result=="sl" else (TP_RR if result=="tp" else
        float(d*(exit_p-entry)/sl_dist))
    return {"R": R, "result": result}


# ── Archetype extractors ──────────────────────────────────────────────────────

def extract_arb(m15, mtf, rng_lo=0.0003, rng_hi=0.015,
                h4_adx_min=15, d1_atr_lo=0.6, d1_atr_hi=2.8):
    """Asian Range Breakout: 00:00–08:00 range, entry 08:00–10:00."""
    ar = ranges(m15, 0, 8)
    trades = []
    for date in sorted(set(m15.index.date)):
        if pd.Timestamp(date).dayofweek >= 4: continue
        dts = pd.Timestamp(date)
        if dts not in ar.index: continue
        r = ar.loc[dts]
        # range as fraction of price
        mid = (r["hi"] + r["lo"]) / 2
        rng_pct = r["rng"] / mid if mid > 0 else 0
        if not (rng_lo <= rng_pct <= rng_hi): continue

        eb = m15[(m15.index.date==date) & (m15.index.hour>=8) & (m15.index.hour<10)]
        traded = False
        for t, bar in eb.iterrows():
            if traded or t not in mtf.index: continue
            i = mtf.loc[t]
            if pd.isna(i["h1_atr"]) or i["h1_atr"]<=0: continue
            d1r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
            if not (d1_atr_lo <= d1r <= d1_atr_hi): continue
            if not pd.isna(i["h4_adx"]) and float(i["h4_adx"]) < h4_adx_min: continue

            price = bar["close"]
            sl_d  = float(np.clip(i["h1_atr"]*SL_MULT, price*SL_LO, price*SL_HI))
            long  = price > r["hi"]; short = price < r["lo"]
            if not long and not short: continue
            d = 1 if long else -1
            if i["h1_trend"] * d < 0: continue
            if i["h4_trend"] != 0 and i["h4_trend"] * d < 0: continue

            res = resolve(m15, t, d, price, price-d*sl_d, price+d*sl_d*TP_RR)
            trades.append({"date":date,"entry_t":t,"arch":"ARB","d":d,**res})
            traded = True
    return trades


def extract_nyo(m15, mtf, rng_lo=0.0002, rng_hi=0.012,
                h4_adx_min=15, d1_atr_lo=0.6, d1_atr_hi=2.8):
    """NY Open Breakout: 10:00–13:00 range, entry 13:00–15:00."""
    nr = ranges(m15, 10, 13)
    trades = []
    for date in sorted(set(m15.index.date)):
        if pd.Timestamp(date).dayofweek >= 4: continue
        dts = pd.Timestamp(date)
        if dts not in nr.index: continue
        r = nr.loc[dts]
        mid = (r["hi"] + r["lo"]) / 2
        rng_pct = r["rng"] / mid if mid > 0 else 0
        if not (rng_lo <= rng_pct <= rng_hi): continue

        eb = m15[(m15.index.date==date) & (m15.index.hour>=13) & (m15.index.hour<15)]
        traded = False
        for t, bar in eb.iterrows():
            if traded or t not in mtf.index: continue
            i = mtf.loc[t]
            if pd.isna(i["h1_atr"]) or i["h1_atr"]<=0: continue
            d1r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
            if not (d1_atr_lo <= d1r <= d1_atr_hi): continue
            if not pd.isna(i["h4_adx"]) and float(i["h4_adx"]) < h4_adx_min: continue

            price = bar["close"]
            sl_d  = float(np.clip(i["h1_atr"]*SL_MULT, price*SL_LO, price*SL_HI))
            long  = price > r["hi"]; short = price < r["lo"]
            if not long and not short: continue
            d = 1 if long else -1
            if i["h1_trend"] * d < 0: continue
            if i["h4_trend"] != 0 and i["h4_trend"] * d < 0: continue

            res = resolve(m15, t, d, price, price-d*sl_d, price+d*sl_d*TP_RR)
            trades.append({"date":date,"entry_t":t,"arch":"NYO","d":d,**res})
            traded = True
    return trades


def extract_mom(m15, mtf, h4_adx_min=20, d1_atr_lo=0.6, d1_atr_hi=2.8):
    """H4 EMA20 pullback momentum: entry when M15 crosses back through H4 EMA20."""
    trades = []
    for date in sorted(set(m15.index.date)):
        if pd.Timestamp(date).dayofweek >= 4: continue
        eb = m15[(m15.index.date==date) & (m15.index.hour>=8) & (m15.index.hour<20)]
        if eb.empty: continue

        prev_above = None; traded = False
        for t, bar in eb.iterrows():
            if traded or t not in mtf.index: continue
            i = mtf.loc[t]
            if pd.isna(i["h4_ema20"]) or int(i["h4_trend"])==0: continue
            if not pd.isna(i["h4_adx"]) and float(i["h4_adx"]) < h4_adx_min: continue
            d1r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
            if not (d1_atr_lo <= d1r <= d1_atr_hi): continue
            if int(i["d1_trend"]) * int(i["h4_trend"]) < 0: continue

            h4t   = int(i["h4_trend"])
            price = bar["close"]
            ema20 = float(i["h4_ema20"])
            above = price > ema20

            if prev_above is None:
                prev_above = above; continue

            d = None
            if h4t==1 and not prev_above and above:   d = 1
            elif h4t==-1 and prev_above and not above: d = -1

            prev_above = above
            if d is None: continue

            h1_atr = float(i["h1_atr"]) if not pd.isna(i["h1_atr"]) else 0
            if h1_atr <= 0: continue
            sl_d = float(np.clip(h1_atr*SL_MULT, price*SL_LO, price*SL_HI))

            res = resolve(m15, t, d, price, price-d*sl_d, price+d*sl_d*TP_RR)
            trades.append({"date":date,"entry_t":t,"arch":"MOM","d":d,**res})
            traded = True
    return trades


# ── OOS validation ────────────────────────────────────────────────────────────

def oos_split(trades: list, train_frac=0.70):
    if not trades:
        return [], []
    dates = sorted(set(t["date"] for t in trades))
    cut   = dates[int(len(dates) * train_frac)]
    IS  = [t for t in trades if t["date"] <  cut]
    OOS = [t for t in trades if t["date"] >= cut]
    return IS, OOS


def edge_ok(trades: list, min_trades=8) -> bool:
    if len(trades) < min_trades: return False
    return float(np.mean([t["R"] for t in trades])) > 0.0


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def monte_carlo(trades: list, risk_pct: float, n_sim=5000) -> dict:
    if not trades:
        return {"pass_pct":0,"bust_pct":100,"med_mo":np.nan,"tpm":0,"expR":0}
    Rs  = np.array([t["R"] for t in trades])
    n   = len(Rs)
    d0  = pd.Timestamp(trades[0]["date"])
    d1  = pd.Timestamp(trades[-1]["date"])
    tpm = n / max((d1-d0).days/30.44, 0.1)
    rng = np.random.default_rng(42)
    draw= max(n*4, 300)
    pc=bc=0; t2p=[]
    for _ in range(n_sim):
        seq = rng.choice(Rs, size=draw, replace=True)
        bal=peak=INIT_BAL; done=False
        for k,r in enumerate(seq):
            bal += bal*risk_pct*r
            peak = max(peak,bal)
            if bal >= INIT_BAL*(1+TARGET):
                pc+=1; t2p.append(k+1); done=True; break
            if bal <= INIT_BAL*(1-MAX_DD) or peak-bal >= INIT_BAL*MAX_DD:
                bc+=1; done=True; break
        # inconclusive: neither pass nor bust in draw
    med = float(np.median(t2p)) if t2p else np.nan
    return {
        "pass_pct": pc/n_sim*100, "bust_pct": bc/n_sim*100,
        "med_mo":   med/tpm if not np.isnan(med) else np.nan,
        "tpm": tpm, "expR": float(Rs.mean()),
        "win_rate": float((Rs>0).mean()), "n": n,
    }


# ── Main scan ─────────────────────────────────────────────────────────────────

def scan_all():
    files = sorted(glob.glob(str(RAW_DIR / "*_M15_*.csv")))
    # Prefer longer files (4yr) over shorter data dir files
    seen = {}
    for f in files:
        sym = sym_from_path(f)
        if sym not in seen or os.path.getsize(f) > os.path.getsize(seen[sym]):
            seen[sym] = f

    print(f"\n{'='*80}")
    print(f"  Multi-Asset HTF Scanner — {len(seen)} instruments")
    print(f"{'='*80}\n")

    survivors = []   # (sym, arch, OOS_trades)

    for sym, fpath in sorted(seen.items()):
        if sym in EXCLUDE:
            continue
        try:
            m15 = load_raw(fpath)
        except Exception as e:
            print(f"  {sym}: load error — {e}"); continue

        if len(m15) < 5000:
            print(f"  {sym}: too few bars ({len(m15)}), skip"); continue

        try:
            mtf = build_mtf(m15)
        except Exception as e:
            print(f"  {sym}: indicator error — {e}"); continue

        results = []
        for arch, fn in [("ARB", extract_arb), ("NYO", extract_nyo), ("MOM", extract_mom)]:
            try:
                trades = fn(m15, mtf)
            except Exception as e:
                continue
            if not trades:
                continue
            IS, OOS = oos_split(trades)
            is_ok  = edge_ok(IS)
            oos_ok = edge_ok(OOS)
            Rs_all = [t["R"] for t in trades]
            tag = "✅" if (is_ok and oos_ok) else ("〜" if is_ok else "❌")
            results.append(
                f"{arch}: n={len(trades):>3} IS={'✓' if is_ok else '✗'} "
                f"OOS={'✓' if oos_ok else '✗'} expR={np.mean(Rs_all):+.3f} {tag}"
            )
            if is_ok and oos_ok:
                # Use full history for portfolio (OOS-validated)
                survivors.append({"sym": sym, "arch": arch, "trades": trades})

        spans = f"{m15.index[0].date()} → {m15.index[-1].date()}"
        print(f"  {sym:<10} {spans}  bars={len(m15):>6}")
        for r in results:
            print(f"    {r}")
        if not results:
            print(f"    (no setups)")
        print()

    # ── Portfolio construction ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  OOS Survivors: {len(survivors)} streams")
    print(f"{'='*80}")
    for s in survivors:
        Rs = [t["R"] for t in s["trades"]]
        print(f"  {s['sym']:<10} {s['arch']}  n={len(s['trades']):>3}  "
              f"expR={np.mean(Rs):+.3f}  WR={sum(r>0 for r in Rs)/len(Rs)*100:.0f}%")

    if not survivors:
        print("  No survivors — nothing to combine.")
        return

    all_trades = sorted(
        [t for s in survivors for t in s["trades"]],
        key=lambda t: t["entry_t"]
    )
    Rs_all = [t["R"] for t in all_trades]
    d0 = pd.Timestamp(all_trades[0]["date"])
    d1 = pd.Timestamp(all_trades[-1]["date"])
    tpm = len(all_trades) / max((d1-d0).days/30.44, 0.1)

    print(f"\n  Combined pool: {len(all_trades)} trades  "
          f"expR={np.mean(Rs_all):+.3f}  WR={sum(r>0 for r in Rs_all)/len(Rs_all)*100:.0f}%  "
          f"{tpm:.1f} trades/mo")

    print(f"\n{'='*80}")
    print(f"  Monte Carlo Challenge Simulation")
    print(f"{'='*80}")
    print(f"  {'Risk%':>6}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}  {'t/mo':>5}")
    sweet = None
    for risk in [0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050]:
        mc = monte_carlo(all_trades, risk)
        flag = ""
        if mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 3.0:
            flag = "  *** SWEET SPOT ***"
            if sweet is None: sweet = (risk, mc)
        elif mc["bust_pct"] <= 5.0 and mc["med_mo"] <= 4.0:
            flag = "  ** close **"
        print(f"  {risk*100:>5.2f}%  {mc['pass_pct']:>6.1f}%  {mc['bust_pct']:>5.1f}%  "
              f"{mc['med_mo']:>6.1f}  {mc['tpm']:>4.1f}{flag}")

    # ── Per-symbol contribution ────────────────────────────────────────────
    print(f"\n  Per-stream contribution:")
    print(f"  {'Symbol':<10}  {'Arch':<4}  {'Trades':>6}  {'t/mo':>5}  "
          f"{'expR':>6}  {'WR':>6}")
    for s in sorted(survivors, key=lambda x: -len(x["trades"])):
        Rs = [t["R"] for t in s["trades"]]
        d0s = pd.Timestamp(s["trades"][0]["date"])
        d1s = pd.Timestamp(s["trades"][-1]["date"])
        tpms = len(s["trades"]) / max((d1s-d0s).days/30.44, 0.1)
        print(f"  {s['sym']:<10}  {s['arch']:<4}  {len(s['trades']):>6}  "
              f"{tpms:>4.1f}  {np.mean(Rs):>+6.3f}  "
              f"{sum(r>0 for r in Rs)/len(Rs)*100:>5.0f}%")

    if sweet:
        risk, mc = sweet
        print(f"\n  ✅ TARGET ACHIEVED at {risk*100:.2f}% risk/trade")
        print(f"     Pass: {mc['pass_pct']:.1f}%  Bust: {mc['bust_pct']:.1f}%  "
              f"Median: {mc['med_mo']:.1f} months")
    else:
        print(f"\n  ⚠️  No single risk level hits both ≤5% bust AND ≤3 months.")
        print(f"     Consider adding more instruments or loosening OOS threshold.")

    print(f"\n{'='*80}\n")
    return survivors, all_trades


if __name__ == "__main__":
    scan_all()
