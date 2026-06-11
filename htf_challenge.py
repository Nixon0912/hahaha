"""
HTF Multi-Archetype Challenge Runner — XAUUSD

Derives H4 and D1 from M15 data.
Three entry archetypes:
  ARB  — Asian Range Breakout (08:00–10:00) with H1 trend gate
  NYO  — NY Open Breakout (13:00–15:00) with H4 range + trend gate
  MOM  — H4 EMA20 Pullback Momentum (enter on M15 when H4 confirms)

Goal: ≤5% bust probability AND ≤3 months median time-to-pass on The5ers challenge.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
PT = 0.01       # 1 point = $0.01 for XAUUSD
INIT_BAL     = 10_000.0
TARGET_PCT   = 0.08
MAX_DD_PCT   = 0.10   # bust if balance falls below $9,000
DAILY_DD_PCT = 0.05
TP_RR        = 3.5
FORCE_CLOSE_H = 21
SL_ATR_MULT  = 0.7
SL_MIN_PTS   = 400
SL_MAX_PTS   = 1600


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_m15() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "XAUUSD_M15.csv",
                     parse_dates=["datetime"], index_col="datetime")
    return df.sort_index()


def resample(m15: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "tick_vol": "sum"}
    return m15.resample(freq, label="left", closed="left").agg(agg).dropna()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr   = pd.concat([df["high"] - df["low"],
                      (df["high"] - prev).abs(),
                      (df["low"]  - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def adx_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    atr_s = atr_series(df, n).replace(0, np.nan)
    up  = (df["high"] - df["high"].shift(1)).clip(lower=0)
    dn  = (df["low"].shift(1) - df["low"]).clip(lower=0)
    dmp = up.where(up >= dn, 0.0)
    dmm = dn.where(dn > up, 0.0)
    dip = 100 * dmp.ewm(span=n, adjust=False).mean() / atr_s
    dim = 100 * dmm.ewm(span=n, adjust=False).mean() / atr_s
    dx  = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(span=n, adjust=False).mean()


def _ffill(s: pd.Series, idx) -> pd.Series:
    """Shift 1 bar (no look-ahead), reindex to target index, forward-fill."""
    return s.shift(1).reindex(idx, method="ffill")


# ── Build multi-timeframe indicator frame ─────────────────────────────────────

def build_mtf(m15: pd.DataFrame) -> pd.DataFrame:
    h1 = resample(m15, "1h")
    h4 = resample(m15, "4h")
    d1 = resample(m15, "1D")

    # H1
    h1["atr14"] = atr_series(h1)
    h1["ema50"] = ema(h1["close"], 50)
    h1["trend"] = np.where(h1["close"] > h1["ema50"], 1, -1)
    h1["atr_ma20"] = h1["atr14"].rolling(20).mean()

    # H4
    h4["atr14"]  = atr_series(h4)
    h4["ema20"]  = ema(h4["close"], 20)
    h4["ema50"]  = ema(h4["close"], 50)
    h4["adx14"]  = adx_series(h4)
    h4["trend"]  = np.where(
        (h4["close"] > h4["ema20"]) & (h4["ema20"] >= h4["ema50"]), 1,
        np.where(
            (h4["close"] < h4["ema20"]) & (h4["ema20"] <= h4["ema50"]), -1, 0
        )
    )
    # momentum: close vs 3-bar-ago close
    h4["mom3"]   = h4["close"] - h4["close"].shift(3)
    h4["atr_rank50"] = h4["atr14"].rolling(50).rank(pct=True)

    # D1
    d1["atr14"]    = atr_series(d1)
    d1["ema20"]    = ema(d1["close"], 20)
    d1["ema50"]    = ema(d1["close"], 50)
    d1["atr_ma20"] = d1["atr14"].rolling(20).mean()
    d1["atr_ratio"] = d1["atr14"] / d1["atr_ma20"].replace(0, np.nan)
    d1["trend"]    = np.where(d1["close"] > d1["ema20"], 1, -1)
    # prev-day high/low for breakout reference
    d1["prev_high"] = d1["high"].shift(1)
    d1["prev_low"]  = d1["low"].shift(1)

    idx = m15.index
    R = pd.DataFrame(index=idx)

    R["h1_atr"]     = _ffill(h1["atr14"], idx)
    R["h1_trend"]   = _ffill(h1["trend"], idx).fillna(0).astype(int)
    R["h1_atr_ma20"]= _ffill(h1["atr_ma20"], idx)

    R["h4_trend"]   = _ffill(h4["trend"], idx).fillna(0).astype(int)
    R["h4_adx"]     = _ffill(h4["adx14"], idx)
    R["h4_ema20"]   = _ffill(h4["ema20"], idx)
    R["h4_mom3"]    = _ffill(h4["mom3"], idx)
    R["h4_atr"]     = _ffill(h4["atr14"], idx)
    R["h4_atr_rank"]= _ffill(h4["atr_rank50"], idx)

    R["d1_trend"]    = _ffill(d1["trend"], idx).fillna(0).astype(int)
    R["d1_atr_ratio"]= _ffill(d1["atr_ratio"], idx)
    R["d1_prev_high"]= _ffill(d1["prev_high"], idx)
    R["d1_prev_low"] = _ffill(d1["prev_low"], idx)

    return R


# ── Asian range helper ────────────────────────────────────────────────────────

def build_ranges(m15: pd.DataFrame, start_h: int, end_h: int) -> pd.DataFrame:
    mask  = (m15.index.hour >= start_h) & (m15.index.hour < end_h)
    grp   = m15[mask].groupby(m15[mask].index.date)
    rng   = pd.DataFrame({"high": grp["high"].max(), "low": grp["low"].min()})
    rng.index = pd.to_datetime(rng.index)
    rng["range_pts"] = ((rng["high"] - rng["low"]) / PT).astype(int)
    return rng


# ── Trade resolution (shared) ─────────────────────────────────────────────────

def resolve_trade(m15: pd.DataFrame, entry_t, d: int,
                  entry: float, sl: float, tp: float) -> dict:
    """Resolve trade from bar AFTER entry_t. Returns result dict with R."""
    sl_dist = abs(entry - sl)
    day_bars = m15[m15.index.date == entry_t.date()]
    rem = day_bars.loc[entry_t:].iloc[1:]          # skip entry bar
    rem = rem[rem.index.hour < FORCE_CLOSE_H]

    result = "timeout"
    exit_p = rem.iloc[-1]["close"] if not rem.empty else entry

    for _, rb in rem.iterrows():
        if d == 1:
            if rb["low"]  <= sl: result = "sl"; exit_p = sl;  break
            if rb["high"] >= tp: result = "tp"; exit_p = tp;  break
        else:
            if rb["high"] >= sl: result = "sl"; exit_p = sl;  break
            if rb["low"]  <= tp: result = "tp"; exit_p = tp;  break

    if result == "sl":
        R = -1.0
    elif result == "tp":
        R = TP_RR
    else:
        R = d * (exit_p - entry) / sl_dist if sl_dist > 0 else 0.0

    return {"result": result, "exit_p": exit_p, "R": float(R)}


# ── ARB: Asian Range Breakout (08:00–10:00) ───────────────────────────────────

def extract_ARB(m15: pd.DataFrame, mtf: pd.DataFrame,
                require_h4: bool = True, require_d1: bool = True,
                h4_adx_min: float = 15.0,
                d1_atr_lo: float = 0.7, d1_atr_hi: float = 2.5,
                ar_min: int = 500, ar_max: int = 3000) -> list:
    ar_ranges = build_ranges(m15, 0, 8)
    trades = []
    dates  = sorted(set(m15.index.date))

    for date in dates:
        dow = pd.Timestamp(date).dayofweek
        if dow >= 4:
            continue
        date_ts = pd.Timestamp(date)
        if date_ts not in ar_ranges.index:
            continue
        ar = ar_ranges.loc[date_ts]
        if not (ar_min <= ar["range_pts"] <= ar_max):
            continue

        entry_mask = (m15.index.date == date) & \
                     (m15.index.hour >= 8) & (m15.index.hour < 10)
        entry_bars = m15[entry_mask]
        if entry_bars.empty:
            continue

        traded = False
        for t, bar in entry_bars.iterrows():
            if traded or t not in mtf.index:
                continue
            i = mtf.loc[t]
            if pd.isna(i["h1_atr"]) or i["h1_atr"] <= 0:
                continue

            # D1 ATR volatility gate
            d1_r = float(i["d1_atr_ratio"]) if not pd.isna(i["d1_atr_ratio"]) else 1.0
            if not (d1_atr_lo <= d1_r <= d1_atr_hi):
                continue

            # H4 ADX
            if not pd.isna(i["h4_adx"]) and i["h4_adx"] < h4_adx_min:
                continue

            price   = bar["close"]
            sl_dist = float(np.clip(i["h1_atr"] * SL_ATR_MULT,
                                    SL_MIN_PTS * PT, SL_MAX_PTS * PT))

            long_ok  = price > ar["high"]
            short_ok = price < ar["low"]
            if not long_ok and not short_ok:
                continue

            d = 1 if long_ok else -1

            # Trend gates
            if i["h1_trend"] * d < 0:
                continue
            if require_h4 and i["h4_trend"] != 0 and i["h4_trend"] * d < 0:
                continue
            if require_d1 and i["d1_trend"] * d < 0:
                continue

            sl = price - d * sl_dist
            tp = price + d * sl_dist * TP_RR

            res = resolve_trade(m15, t, d, price, sl, tp)
            trades.append({"date": date, "entry_t": t, "archetype": "ARB",
                           "dir": d, **res})
            traded = True

    return trades


# ── NYO: NY Open Breakout (13:00–15:00) ──────────────────────────────────────

def extract_NYO(m15: pd.DataFrame, mtf: pd.DataFrame,
                require_h4: bool = True,
                h4_adx_min: float = 15.0,
                d1_atr_lo: float = 0.7, d1_atr_hi: float = 2.5,
                nyo_min: int = 400, nyo_max: int = 2500) -> list:
    """NY morning range (10:00–13:00) breakout, entered 13:00–15:00."""
    nyo_ranges = build_ranges(m15, 10, 13)
    trades = []
    dates  = sorted(set(m15.index.date))

    for date in dates:
        dow = pd.Timestamp(date).dayofweek
        if dow >= 4:
            continue
        date_ts = pd.Timestamp(date)
        if date_ts not in nyo_ranges.index:
            continue
        nr = nyo_ranges.loc[date_ts]
        if not (nyo_min <= nr["range_pts"] <= nyo_max):
            continue

        entry_mask = (m15.index.date == date) & \
                     (m15.index.hour >= 13) & (m15.index.hour < 15)
        entry_bars = m15[entry_mask]
        if entry_bars.empty:
            continue

        traded = False
        for t, bar in entry_bars.iterrows():
            if traded or t not in mtf.index:
                continue
            i = mtf.loc[t]
            if pd.isna(i["h1_atr"]) or i["h1_atr"] <= 0:
                continue

            d1_r = float(i["d1_atr_ratio"]) if not pd.isna(i["d1_atr_ratio"]) else 1.0
            if not (d1_atr_lo <= d1_r <= d1_atr_hi):
                continue
            if not pd.isna(i["h4_adx"]) and i["h4_adx"] < h4_adx_min:
                continue

            price   = bar["close"]
            sl_dist = float(np.clip(i["h1_atr"] * SL_ATR_MULT,
                                    SL_MIN_PTS * PT, SL_MAX_PTS * PT))

            long_ok  = price > nr["high"]
            short_ok = price < nr["low"]
            if not long_ok and not short_ok:
                continue

            d = 1 if long_ok else -1

            if i["h1_trend"] * d < 0:
                continue
            if require_h4 and i["h4_trend"] != 0 and i["h4_trend"] * d < 0:
                continue

            sl = price - d * sl_dist
            tp = price + d * sl_dist * TP_RR

            res = resolve_trade(m15, t, d, price, sl, tp)
            trades.append({"date": date, "entry_t": t, "archetype": "NYO",
                           "dir": d, **res})
            traded = True

    return trades


# ── MOM: H4 EMA20 Pullback Momentum ──────────────────────────────────────────

def extract_MOM(m15: pd.DataFrame, mtf: pd.DataFrame,
                require_d1: bool = True,
                h4_adx_min: float = 20.0,
                d1_atr_lo: float = 0.7, d1_atr_hi: float = 2.5) -> list:
    """
    Enter when M15 price touches H4 EMA20 from the trend side,
    during London/NY session (08:00–20:00), H4 trend confirmed.
    SL = 1× H1 ATR below EMA20; TP = SL × TP_RR.
    """
    trades = []
    dates  = sorted(set(m15.index.date))
    prev_above: dict = {}   # date → last bar above/below ema

    for date in dates:
        dow = pd.Timestamp(date).dayofweek
        if dow >= 4:
            continue

        day_mask = (m15.index.date == date) & \
                   (m15.index.hour >= 8) & (m15.index.hour < 20)
        day_bars = m15[day_mask]
        if day_bars.empty:
            continue

        traded = False
        prev_side = None   # track crossback

        for t, bar in day_bars.iterrows():
            if traded or t not in mtf.index:
                continue
            i = mtf.loc[t]

            if pd.isna(i["h4_ema20"]) or pd.isna(i["h4_trend"]):
                continue
            if int(i["h4_trend"]) == 0:
                continue  # need clear H4 trend
            if not pd.isna(i["h4_adx"]) and i["h4_adx"] < h4_adx_min:
                continue

            d1_r = float(i["d1_atr_ratio"]) if not pd.isna(i["d1_atr_ratio"]) else 1.0
            if not (d1_atr_lo <= d1_r <= d1_atr_hi):
                continue

            if require_d1 and int(i["d1_trend"]) * int(i["h4_trend"]) < 0:
                continue

            h4t   = int(i["h4_trend"])
            ema20 = float(i["h4_ema20"])
            price = bar["close"]
            h1_atr = float(i["h1_atr"]) if not pd.isna(i["h1_atr"]) else None
            if h1_atr is None or h1_atr <= 0:
                continue

            # Pullback: in uptrend, price dips to EMA20 from above
            # Detect: prev bar was "near or below" EMA20, current bar closes above
            cur_above  = price > ema20
            side_label = cur_above

            if prev_side is None:
                prev_side = side_label
                continue

            # Uptrend pullback: was at/below EMA20, now back above → long
            if h4t == 1 and (not prev_side) and cur_above:
                d = 1
            # Downtrend pullback: was at/above EMA20, now back below → short
            elif h4t == -1 and prev_side and (not cur_above):
                d = -1
            else:
                prev_side = side_label
                continue

            # Entry
            sl_dist = float(np.clip(h1_atr * SL_ATR_MULT,
                                    SL_MIN_PTS * PT, SL_MAX_PTS * PT))
            sl = price - d * sl_dist
            tp = price + d * sl_dist * TP_RR

            res = resolve_trade(m15, t, d, price, sl, tp)
            trades.append({"date": date, "entry_t": t, "archetype": "MOM",
                           "dir": d, **res})
            traded = True
            prev_side = side_label

    return trades


# ── Monte Carlo challenge sim ─────────────────────────────────────────────────

def monte_carlo(trades: list, risk_pct: float, n_sim: int = 5000) -> dict:
    if not trades:
        return {"pass_pct": 0, "bust_pct": 100,
                "median_trades": np.nan, "median_months": np.nan,
                "trades_per_mo": 0, "exp_R": 0}

    Rs  = np.array([t["R"] for t in trades])
    n   = len(Rs)
    d0  = pd.Timestamp(trades[0]["date"])
    d1  = pd.Timestamp(trades[-1]["date"])
    span_mo = max((d1 - d0).days / 30.44, 0.1)
    tpm = n / span_mo

    target    = INIT_BAL * (1 + TARGET_PCT)
    bust_lvl  = INIT_BAL * (1 - MAX_DD_PCT)
    rng       = np.random.default_rng(42)
    n_draw    = max(n * 4, 200)  # enough to reach target even in lucky run

    pass_cnt = bust_cnt = 0
    t2pass = []

    for _ in range(n_sim):
        seq = rng.choice(Rs, size=n_draw, replace=True)
        bal = peak = INIT_BAL
        passed = busted = False
        for k, r in enumerate(seq):
            bal  += bal * risk_pct * r
            peak  = max(peak, bal)
            if bal >= target:
                passed = True; t2pass.append(k + 1); break
            if bal <= bust_lvl or (peak - bal) >= INIT_BAL * MAX_DD_PCT:
                busted = True; break
        pass_cnt += passed
        bust_cnt += busted

    med_t = float(np.median(t2pass)) if t2pass else np.nan
    med_m = med_t / tpm if not np.isnan(med_t) else np.nan

    return {
        "pass_pct":      pass_cnt / n_sim * 100,
        "bust_pct":      bust_cnt / n_sim * 100,
        "median_trades": med_t,
        "median_months": med_m,
        "trades_per_mo": tpm,
        "exp_R":         float(Rs.mean()),
        "win_rate":      float((Rs > 0).mean()),
        "total_trades":  n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading M15 data …")
    m15 = load_m15()
    print(f"  {len(m15)} bars  {m15.index[0]} → {m15.index[-1]}")

    print("Computing H1/H4/D1 indicators …")
    mtf = build_mtf(m15)

    # ── Extract all archetypes ─────────────────────────────────────────────
    print("Extracting ARB setups …")
    arb = extract_ARB(m15, mtf,
                      require_h4=True, require_d1=True,
                      h4_adx_min=15, d1_atr_lo=0.7, d1_atr_hi=2.5,
                      ar_min=500, ar_max=3000)

    print("Extracting NYO setups …")
    nyo = extract_NYO(m15, mtf,
                      require_h4=True, h4_adx_min=15,
                      d1_atr_lo=0.7, d1_atr_hi=2.5,
                      nyo_min=300, nyo_max=2500)

    print("Extracting MOM setups …")
    mom = extract_MOM(m15, mtf,
                      require_d1=True, h4_adx_min=20,
                      d1_atr_lo=0.7, d1_atr_hi=2.5)

    all_trades = sorted(arb + nyo + mom, key=lambda t: t["entry_t"])

    print(f"\n{'='*75}")
    print(f"  Setup pools:")
    for label, pool in [("ARB", arb), ("NYO", nyo), ("MOM", mom)]:
        if not pool:
            print(f"    {label}: 0 trades")
            continue
        Rs = [t["R"] for t in pool]
        d0 = pd.Timestamp(pool[0]["date"])
        d1 = pd.Timestamp(pool[-1]["date"])
        mo = max((d1 - d0).days / 30.44, 0.1)
        print(f"    {label}: {len(pool):>3} trades  "
              f"WR={sum(r>0 for r in Rs)/len(Rs)*100:.0f}%  "
              f"expR={np.mean(Rs):+.3f}  {len(pool)/mo:.1f}/mo")
    print()

    # ── Portfolio combinations ─────────────────────────────────────────────
    combos = [
        ("ARB only",           arb),
        ("NYO only",           nyo),
        ("MOM only",           mom),
        ("ARB + NYO",          arb + nyo),
        ("ARB + MOM",          arb + mom),
        ("NYO + MOM",          nyo + mom),
        ("ARB + NYO + MOM",    all_trades),
    ]

    print(f"{'='*90}")
    print(f"  Portfolio scan — Monte Carlo (5000 sims)")
    print(f"{'='*90}")
    print(f"  {'Combo':<22}  {'Trades':>6}  {'WR':>6}  {'expR':>6}  "
          f"{'Pass%':>6}  {'Bust%':>6}  {'Mo(med)':>8}  {'t/mo':>5}")
    print(f"  {'-'*86}")

    best = None
    for label, pool in combos:
        if not pool:
            continue
        pool_s = sorted(pool, key=lambda t: t["entry_t"])
        Rs     = [t["R"] for t in pool_s]
        mc025  = monte_carlo(pool_s, risk_pct=0.0025)
        flag   = ""
        if mc025["bust_pct"] <= 5.0 and mc025["median_months"] <= 3.5:
            flag = "  *** TARGET ***"
            if best is None:
                best = (label, pool_s, mc025)
        elif mc025["bust_pct"] <= 8.0 and mc025["median_months"] <= 5.0:
            flag = "  ** good **"

        print(f"  {label:<22}  {len(pool_s):>6}  "
              f"{sum(r>0 for r in Rs)/len(Rs)*100:>5.1f}%  "
              f"{np.mean(Rs):>+6.3f}  "
              f"{mc025['pass_pct']:>5.1f}%  {mc025['bust_pct']:>5.1f}%  "
              f"{mc025['median_months']:>7.1f}  {mc025['trades_per_mo']:>4.1f}{flag}")

    print(f"\n  Risk level detail for ARB+NYO+MOM portfolio:")
    print(f"  {'Risk%':>6}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}  {'t/mo':>5}")
    for risk in [0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050]:
        mc = monte_carlo(all_trades, risk_pct=risk)
        flag = "  *** SWEET SPOT ***" if mc["bust_pct"] <= 5.0 and mc["median_months"] <= 3.0 else ""
        print(f"  {risk*100:>5.2f}%  {mc['pass_pct']:>6.1f}%  {mc['bust_pct']:>5.1f}%  "
              f"{mc['median_months']:>6.1f}  {mc['trades_per_mo']:>4.1f}{flag}")

    print(f"\n{'='*90}")

    # ── Monthly breakdown ──────────────────────────────────────────────────
    if all_trades:
        df = pd.DataFrame(all_trades)
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month").agg(
            trades=("R", "count"),
            expR=("R", "mean"),
            sumR=("R", "sum"),
            wins=("R", lambda x: (x > 0).sum()),
        )
        print(f"\n  Monthly breakdown (ARB+NYO+MOM):")
        print(f"  {'Month':<10}  {'Trades':>6}  {'WR':>6}  {'avgR':>7}  {'sumR':>7}  "
              f"{'By archetype'}")

        for m, row in monthly.iterrows():
            month_trades = df[df["month"] == m]
            by_arch = " ".join(
                f"{a}:{int(v)}" for a, v in
                month_trades.groupby("archetype")["R"].count().items()
            )
            print(f"  {str(m):<10}  {int(row['trades']):>6}  "
                  f"{row['wins']/row['trades']*100:>5.0f}%  "
                  f"{row['expR']:>+7.3f}  {row['sumR']:>+7.3f}  {by_arch}")
