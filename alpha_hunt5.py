"""
Alpha Hunt v5 — truly vectorised inner loop, warnings suppressed.

Changes from v4:
  - Pre-convert opportunity table to numpy arrays before scanning
  - No pandas ops inside the 150k-config loop → 10-50× faster
  - warnings suppressed
  - Progress counter every 5000 configs
"""

import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
from backtester import Backtester, Strategy, load
from strategy_combined import compute_ranges

PT      = 0.01
LOTS    = 0.1
LOT_VAL = PT * 100 * LOTS   # $0.10 per point
COMM    = 0.40
BALANCE = 10_000.0

PERIODS = [
    ("Steady Uptrend",   "2025-01-07", "2025-03-31"),
    ("Post-ATH Correct", "2025-06-01", "2025-08-31"),
    ("Range Recovery",   "2025-09-01", "2025-12-31"),
    ("High-Vol Chop",    "2026-01-01", "2026-03-31"),
    ("Current",          "2026-04-01", "2026-06-09"),
]

ALPHA_PF        = 1.10
ALPHA_MONTHLY   = 0.0
ALPHA_MAX_DD    = -1200.0
ALPHA_WORST_DAY = -4.0
ALPHA_MIN_TR    = 8
ALPHA_PERIODS   = 4


# ── H1 indicators ─────────────────────────────────────────────────────────────

def build_indicators(df_full):
    h1 = load("H1")
    prev_c = h1["close"].shift(1)
    tr = pd.concat([h1["high"]-h1["low"],
                    (h1["high"]-prev_c).abs(),
                    (h1["low"]-prev_c).abs()], axis=1).max(axis=1)
    h1_atr    = tr.ewm(span=14, adjust=False).mean()
    h1_atr_ma = h1_atr.rolling(50, min_periods=25).mean()
    h1_ema50  = h1["close"].ewm(span=50, adjust=False).mean()
    h1_trend  = (h1["close"] > h1_ema50).astype(int)*2 - 1

    hi=h1["high"]; lo=h1["low"]; cl=h1["close"]
    ph=hi.shift(1); pl=lo.shift(1)
    dm_p = np.where((hi-ph)>(pl-lo), np.maximum(hi-ph,0), 0)
    dm_m = np.where((pl-lo)>(hi-ph), np.maximum(pl-lo,0), 0)
    def _wma(arr, n):
        out = np.full(len(arr), np.nan); a = np.asarray(arr, float)
        out[n-1] = np.nanmean(a[:n])
        for i in range(n, len(a)):
            out[i] = out[i-1] - out[i-1]/n + a[i]
        return pd.Series(out, index=h1.index)
    atr14 = _wma(tr, 14)
    di_p  = 100*_wma(pd.Series(dm_p, index=h1.index), 14) / atr14
    di_m  = 100*_wma(pd.Series(dm_m, index=h1.index), 14) / atr14
    dx    = 100*(di_p-di_m).abs()/(di_p+di_m).replace(0, np.nan)
    h1_adx= _wma(dx.fillna(0), 14)

    def _ff(s): return s.shift(1).reindex(df_full.index, method="ffill")
    out = pd.DataFrame(index=df_full.index)
    out["atr"]     = _ff(h1_atr)
    out["atr_ma"]  = _ff(h1_atr_ma)
    out["trend"]   = _ff(h1_trend).fillna(0).astype(int)
    out["adx"]     = _ff(h1_adx).fillna(0)
    out["atr_exp"] = (out["atr"] >= out["atr_ma"]).astype(int)
    return out


# ── Opportunity table ─────────────────────────────────────────────────────────

def build_opps(df_full, ind):
    print("  Building opportunity table …")
    arb_r, nyo_r = compute_ranges(df_full.copy())
    buf = 30 * PT
    records = []
    for date, day_df in df_full.groupby(df_full.index.date):
        dow = day_df.index[0].dayofweek
        for sess, ranges, h0, h1_, rlo, rhi in [
            ("ARB", arb_r, 8, 10, 500, 9000),
            ("NYO", nyo_r, 13, 15, 300, 7000),
        ]:
            if date not in ranges.index: continue
            r   = ranges.loc[date]
            rng = float(r["range_pts"])
            if not (rlo <= rng <= rhi): continue
            win = day_df[(day_df.index.hour >= h0) & (day_df.index.hour < h1_)]
            for t, bar in win.iterrows():
                close = bar["close"]; spread = bar["spread"]
                iv    = ind.loc[t] if t in ind.index else None
                if iv is None: continue
                atr=float(iv["atr"]); trend=int(iv["trend"])
                adx=float(iv["adx"]); atr_exp=int(iv["atr_exp"])
                direction = None
                if close > r["high"] + buf:
                    direction=1;  entry=close+(spread/2)*PT
                elif close < r["low"] - buf:
                    direction=-1; entry=close-(spread/2)*PT
                if direction is not None:
                    rem = day_df.loc[t:]
                    records.append((date, sess, direction, entry, spread,
                                    atr, trend, rng, rem["high"].max(),
                                    rem["low"].min(), dow, adx, atr_exp))
                    break

    cols=["date","session","direction","entry","spread","h1_atr","h1_trend",
          "range_pts","day_high","day_low","dow","h1_adx","atr_exp"]
    opps = pd.DataFrame(records, columns=cols)
    opps["date_dt"]  = pd.to_datetime(opps["date"].astype(str))
    opps["date_int"] = (opps["date_dt"].dt.year*10000
                       +opps["date_dt"].dt.month*100
                       +opps["date_dt"].dt.day)
    n = len(opps)
    print(f"  {n:,} opportunities  "
          f"({(opps.session=='ARB').sum()} ARB, {(opps.session=='NYO').sum()} NYO)")
    return opps


# ── Pre-packaged numpy arrays for fast scanning ───────────────────────────────

class OppArrays:
    """All opp columns as float/int numpy arrays + period masks."""
    def __init__(self, opps):
        self.n         = len(opps)
        self.sess_arb  = (opps["session"]=="ARB").values   # bool
        self.dir       = opps["direction"].values.astype(np.int8)
        self.entry     = opps["entry"].values
        self.spread    = opps["spread"].values
        self.atr       = opps["h1_atr"].values
        self.trend     = opps["h1_trend"].values.astype(np.int8)
        self.rng_pts   = opps["range_pts"].values
        self.day_high  = opps["day_high"].values
        self.day_low   = opps["day_low"].values
        self.dow       = opps["dow"].values.astype(np.int8)
        self.adx       = opps["h1_adx"].values
        self.atr_exp   = opps["atr_exp"].values.astype(np.int8)
        self.date_dt   = opps["date_dt"].values          # datetime64
        self.date_int  = opps["date_int"].values

        # Period masks (datetime64 comparison)
        self.period_masks = []
        for label, start, end in PERIODS:
            s = np.datetime64(start); e = np.datetime64(end)
            mask = (self.date_dt >= s) & (self.date_dt <= e)
            self.period_masks.append((label, start, end, mask))

        # Month float for per-month avg calculation
        dt = opps["date_dt"]
        self.month_id = (dt.dt.year*12 + dt.dt.month).values  # int


def score_fast_np(oa: OppArrays, cfg: dict):
    """Vectorised scan — no pandas in this function."""
    n = oa.n

    # Build base mask
    m = np.ones(n, dtype=bool)
    if not cfg["use_arb"]:   m &= ~oa.sess_arb
    if not cfg["use_nyo"]:   m &= oa.sess_arb
    if cfg["dow"] == "mon-thu": m &= (oa.dow <= 3)
    elif cfg["dow"] == "no-fri": m &= (oa.dow != 4)
    min_atr = cfg["min_atr"]
    if min_atr > 0: m &= (oa.atr >= min_atr)
    min_adx = cfg["min_adx"]
    if min_adx > 0: m &= (oa.adx >= min_adx)
    if cfg["atr_exp"]: m &= (oa.atr_exp == 1)
    if cfg["trend"]:   m &= (oa.dir == oa.trend)

    if m.sum() < ALPHA_MIN_TR * 2:
        return None

    # Compute SL/TP pts under mask
    sl_mode = cfg["sl_mode"]
    sl_min  = cfg["sl_min"]; sl_max = cfg["sl_max"]
    if sl_mode == "fixed":
        sl_pts = np.full(n, float(cfg["sl_fixed"]))
    elif sl_mode == "atr":
        sl_pts = np.clip(oa.atr * cfg["sl_atr_m"] / PT, sl_min, sl_max)
    else:  # range
        sl_pts = np.clip(oa.rng_pts * 0.5, sl_min, sl_max)
    tp_pts = sl_pts * cfg["tp_mult"]

    dirs   = oa.dir.astype(float)
    tp_p   = oa.entry + dirs * tp_pts * PT
    sl_p   = oa.entry - dirs * sl_pts * PT

    sl_hit = np.where(dirs == 1, oa.day_low  <= sl_p,
                                  oa.day_high >= sl_p)
    tp_hit = np.where(dirs == 1, oa.day_high >= tp_p,
                                  oa.day_low  <= tp_p)
    win    = tp_hit & (~sl_hit)
    pnl    = np.where(win, tp_pts, -sl_pts) * LOT_VAL
    pnl   -= oa.spread * 0.5 * LOT_VAL + COMM

    results = {}
    passes  = 0
    for label, start, end, pmask in oa.period_masks:
        idx = m & pmask
        if idx.sum() < ALPHA_MIN_TR:
            continue
        p_pnl = pnl[idx]
        p_win = win[idx]
        p_di  = oa.date_int[idx]

        gross_w = p_pnl[p_win].sum()
        gross_l = p_pnl[~p_win].sum()
        pf = gross_w / (-gross_l) if gross_l < 0 else 99.0

        # Monthly avg
        mo_len = (np.datetime64(end) - np.datetime64(start)) / np.timedelta64(30, 'D')
        mo_avg = p_pnl.sum() / float(mo_len)

        # Max DD
        cum   = np.cumsum(p_pnl)
        peak  = np.maximum.accumulate(cum)
        max_dd = float((cum - peak).min())

        # Worst day
        uniq_d, inv = np.unique(p_di, return_inverse=True)
        day_sums = np.bincount(inv, weights=p_pnl)
        worst_day = float(day_sums.min()) / BALANCE * 100.0

        ok = (pf >= ALPHA_PF and mo_avg >= ALPHA_MONTHLY
              and max_dd >= ALPHA_MAX_DD and worst_day >= ALPHA_WORST_DAY)
        if ok: passes += 1
        results[label] = {"trades": int(idx.sum()), "pf": round(pf, 3),
                          "monthly_avg": round(mo_avg, 2),
                          "max_dd": round(max_dd, 2),
                          "worst_day": round(worst_day, 2), "ok": ok}
    results["_passes"] = passes
    return results


# ── Full backtester validation (unchanged from v4) ────────────────────────────

class ValidateStrat(Strategy):
    def __init__(self, cfg, bal, arb_r, nyo_r, ind):
        self.cfg=cfg; self.initial_bal=self.balance=self.peak_bal=bal
        self._day=None; self._day_start=bal
        self._arb_done=self._nyo_done=False
        self._in_trade=False; self._dir=self._sl=self._tp=None
        self.arb_ranges=arb_r; self.nyo_ranges=nyo_r; self.ind=ind

    def _new_day(self, d):
        self._day=d; self._day_start=self.balance
        self._arb_done=self._nyo_done=False

    def next(self, i, df):
        bar=df.iloc[i]; t=df.index[i]; today=t.date(); hour=t.hour
        if today != self._day: self._new_day(today)
        dow=t.dayofweek
        if self.cfg["dow"]=="mon-thu" and dow>=4: return None
        if self.cfg["dow"]=="no-fri"  and dow==4: return None

        iv   = self.ind.iloc[i] if i < len(self.ind) else None
        atr  = float(iv["atr"])     if iv is not None and not pd.isna(iv["atr"])  else 0
        adx  = float(iv["adx"])     if iv is not None and not pd.isna(iv["adx"])  else 0
        trnd = int(iv["trend"])     if iv is not None else 0
        aexp = int(iv["atr_exp"])   if iv is not None else 0

        if self._in_trade:
            self.peak_bal = max(self.peak_bal, self.balance)
            if hour >= 21 or (self.peak_bal-self.balance)/self.initial_bal >= 0.085:
                return self._c()
            if self._dir == "buy":
                if bar["low"]  <= self._sl: return self._c()
                if bar["high"] >= self._tp: return self._c()
            else:
                if bar["high"] >= self._sl: return self._c()
                if bar["low"]  <= self._tp: return self._c()
            return None

        self.peak_bal = max(self.peak_bal, self.balance)
        if (self._day_start-self.balance)/self.initial_bal >= 0.04:  return None
        if (self.peak_bal  -self.balance)/self.initial_bal >= 0.085: return None
        if self.cfg["min_atr"] > 0 and atr < self.cfg["min_atr"]: return None
        if self.cfg["min_adx"] > 0 and adx < self.cfg["min_adx"]: return None
        if self.cfg["atr_exp"] and not aexp: return None
        if self.cfg["trend"]   and trnd == 0: return None

        close=bar["close"]; sp=bar["spread"]

        def _slp():
            m=self.cfg["sl_mode"]
            if m=="fixed": return int(np.clip(self.cfg["sl_fixed"], self.cfg["sl_min"], self.cfg["sl_max"]))
            elif m=="atr":  return int(np.clip(atr*self.cfg["sl_atr_m"]/PT, self.cfg["sl_min"], self.cfg["sl_max"]))
            else: return self.cfg["sl_min"]

        if self.cfg["use_arb"] and 8<=hour<10 and not self._arb_done and today in self.arb_ranges.index:
            r=self.arb_ranges.loc[today]; rng=float(r["range_pts"])
            if 500<=rng<=9000:
                sl=_slp(); tp=int(sl*self.cfg["tp_mult"])
                if (not self.cfg["trend"] or trnd>=0) and close>r["high"]+30*PT:
                    e=close+(sp/2)*PT; self._sl=e-sl*PT; self._tp=e+tp*PT
                    self._in_trade=True; self._dir="buy"; self._arb_done=True; return "buy"
                if (not self.cfg["trend"] or trnd<=0) and close<r["low"]-30*PT:
                    e=close-(sp/2)*PT; self._sl=e+sl*PT; self._tp=e-tp*PT
                    self._in_trade=True; self._dir="sell"; self._arb_done=True; return "sell"

        if self.cfg["use_nyo"] and 13<=hour<15 and not self._nyo_done and today in self.nyo_ranges.index:
            r=self.nyo_ranges.loc[today]; rng=float(r["range_pts"])
            if 300<=rng<=7000:
                sl=_slp(); tp=int(sl*self.cfg["tp_mult"])
                if (not self.cfg["trend"] or trnd>=0) and close>r["high"]+30*PT:
                    e=close+(sp/2)*PT; self._sl=e-sl*PT; self._tp=e+tp*PT
                    self._in_trade=True; self._dir="buy"; self._nyo_done=True; return "buy"
                if (not self.cfg["trend"] or trnd<=0) and close<r["low"]-30*PT:
                    e=close-(sp/2)*PT; self._sl=e+sl*PT; self._tp=e-tp*PT
                    self._in_trade=True; self._dir="sell"; self._nyo_done=True; return "sell"
        return None

    def _c(self):
        self._in_trade=False; self._dir=self._sl=self._tp=None; return "close"


def validate_full(cfg, df_full, arb_r, nyo_r, ind):
    results = {}
    for label, start, end in PERIODS:
        df = df_full.loc[start:end].copy()
        iv = ind.reindex(df.index, method="ffill")
        strat = ValidateStrat(cfg, BALANCE, arb_r, nyo_r, iv)
        bt    = Backtester(df, strat, lots=LOTS, initial_balance=BALANCE)
        rep   = bt.run()
        log   = rep.trade_log()
        if log.empty: results[label]=None; continue

        s   = rep.summary()
        log["date"]  = log["exit_time"].dt.date
        log["month"] = log["exit_time"].dt.to_period("M")
        log["hold_min"] = (log["exit_time"]-log["entry_time"]).dt.total_seconds()/60
        daily   = log.groupby("date")["pnl"].sum()
        monthly = log.groupby("month")["pnl"].sum()
        wins=log[log["pnl"]>0]; losses=log[log["pnl"]<0]

        net = float(str(s["net_pnl"]).replace("$","").replace(",",""))
        mdd = float(str(s["max_drawdown"]).replace("$","").replace(",",""))
        pf  = wins["pnl"].sum()/-losses["pnl"].sum() if len(losses)>0 else 99

        n_days=len(daily); n_yrs=n_days/252
        dr=daily/BALANCE
        sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
        ann_ret = (net/BALANCE)/n_yrs if n_yrs>0 else 0
        calmar  = ann_ret/(abs(mdd)/BALANCE) if mdd!=0 else 0

        bal=BALANCE; pk=BALANCE; ch="⏳"; chd=None
        for _,t in log.iterrows():
            bal+=t["pnl"]; pk=max(pk,bal)
            if bal>=BALANCE*1.08: ch="✅ PASS"; chd=t["exit_time"].date(); break
            if pk-bal>=BALANCE*0.10: ch="❌ FAIL"; chd=t["exit_time"].date(); break

        results[label]={
            "trades":len(log),"wr":len(wins)/len(log)*100,"pf":round(pf,2),
            "net":net,"mdd":mdd,"monthly":monthly.mean(),
            "worst_day":daily.min()/BALANCE*100,
            "avg_win":wins["pnl"].mean() if len(wins)>0 else 0,
            "avg_loss":losses["pnl"].mean() if len(losses)>0 else 0,
            "hold":log["hold_min"].mean(),
            "sharpe":round(sharpe,2),"calmar":round(calmar,2),
            "costs":log["commission"].sum()+log["spread_cost"].sum(),
            "challenge":ch,"ch_date":chd,
        }
    return results


def print_validation(cfg, results):
    sm = (f"SL={cfg['sl_mode']}×{cfg['sl_atr_m']}[{cfg['sl_min']}-{cfg['sl_max']}]"
          if cfg["sl_mode"]!="fixed" else f"SL={cfg['sl_fixed']}pt")
    sess = ("ARB+NYO" if cfg["use_arb"] and cfg["use_nyo"]
            else ("ARB" if cfg["use_arb"] else "NYO"))
    extras=[]
    if cfg["trend"]:       extras.append("trend")
    if cfg["dow"]!="all":  extras.append(cfg["dow"])
    if cfg["min_adx"]>0:   extras.append(f"adx≥{cfg['min_adx']}")
    if cfg["atr_exp"]:     extras.append("atr-exp")
    print(f"\n  {'─'*63}")
    print(f"  {sm}  TP×{cfg['tp_mult']}  {sess}  {' '.join(extras)}")
    print(f"  {'─'*63}")
    print(f"  {'Period':<22} {'Tr':>4} {'WR%':>5} {'PF':>5} {'Net%':>6} "
          f"{'MaxDD%':>7} {'Shrp':>5} {'Clmr':>6} {'The5ers'}")
    passes=0
    for label,_,_ in PERIODS:
        r=results.get(label)
        if r is None: print(f"  {label:<22}  — no trades"); continue
        ok=(r["pf"]>=ALPHA_PF and r["monthly"]>=0 and r["mdd"]>=ALPHA_MAX_DD
            and r["worst_day"]>=ALPHA_WORST_DAY and r["trades"]>=ALPHA_MIN_TR)
        if ok: passes+=1
        flag="✅" if ok else "❌"
        print(f"  {flag} {label:<21} {r['trades']:>4} {r['wr']:>5.1f} {r['pf']:>5.2f} "
              f"{r['net']/BALANCE*100:>5.1f}% {r['mdd']/BALANCE*100:>6.2f}% "
              f"{r['sharpe']:>5.2f} {r['calmar']:>6.2f} {r['challenge']}"
              + (f" ({r['ch_date']})" if r["ch_date"] else ""))
    print(f"  {'─'*63}")
    if any(results.values()):
        print(f"  Passes: {passes}/5  |  Avg hold: "
              f"{np.mean([r['hold'] for r in results.values() if r]):>3.0f} min  |  "
              f"Total costs: ${sum(r['costs'] for r in results.values() if r):,.0f}")


# ── Grid ──────────────────────────────────────────────────────────────────────

def build_grid():
    cfgs = []
    for sl_mode in ["atr", "range", "fixed"]:
        sl_fixeds = [600, 800, 1000] if sl_mode=="fixed" else [800]
        sl_atr_ms = [0.3, 0.4, 0.5, 0.6, 0.7] if sl_mode!="fixed" else [0.5]
        sl_mins   = [200, 300, 400] if sl_mode!="fixed" else [300]
        sl_maxs   = [800, 1200, 1600] if sl_mode!="fixed" else [1200]
        for sl_fixed in sl_fixeds:
            for tp_mult in [2.5, 3.0, 3.5, 4.0, 5.0]:
                for sl_atr_m in sl_atr_ms:
                    for sl_min in sl_mins:
                        for sl_max in sl_maxs:
                            if sl_min >= sl_max: continue
                            for use_arb in [True, False]:
                                for use_nyo in [True, False]:
                                    if not use_arb and not use_nyo: continue
                                    for min_atr in [0.0, 6.0, 10.0]:
                                        for min_adx in [0.0, 20.0, 28.0]:
                                            for atr_exp in [False, True]:
                                                for trend in [True, False]:
                                                    for dow in ["all","mon-thu","no-fri"]:
                                                        cfgs.append(dict(
                                                            sl_mode=sl_mode, sl_fixed=sl_fixed,
                                                            tp_mult=tp_mult, sl_atr_m=sl_atr_m,
                                                            sl_min=sl_min, sl_max=sl_max,
                                                            use_arb=use_arb, use_nyo=use_nyo,
                                                            min_atr=min_atr, min_adx=min_adx,
                                                            atr_exp=atr_exp, trend=trend, dow=dow,
                                                        ))
    return cfgs


# ── Main ──────────────────────────────────────────────────────────────────────

def scan(top_n=15):
    print("Loading M5 data …")
    df_full = load("M5")
    print("Computing H1 indicators …")
    ind = build_indicators(df_full)
    opps = build_opps(df_full, ind)
    arb_r, nyo_r = compute_ranges(df_full.copy())

    for label, start, end in PERIODS:
        dt = pd.to_datetime(opps["date_dt"])
        n  = ((dt>=start)&(dt<=end)).sum()
        print(f"  {label}: {n} opps")

    oa   = OppArrays(opps)
    cfgs = build_grid()
    print(f"\nPhase 1 — scanning {len(cfgs):,} configs (numpy vectorised) …")
    t0 = time.time()

    scored = []
    for i, cfg in enumerate(cfgs):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time()-t0
            rate    = i/elapsed
            remain  = (len(cfgs)-i)/rate
            print(f"  {i:>7,}/{len(cfgs):,}  "
                  f"{elapsed:.0f}s elapsed  ~{remain:.0f}s left  "
                  f"alpha-candidates: {sum(1 for s in scored if s['passes']>=ALPHA_PERIODS)}")

        r = score_fast_np(oa, cfg)
        if r is None: continue
        passes  = r["_passes"]
        metrics = [v for v in r.values() if isinstance(v, dict) and "monthly_avg" in v]
        if not metrics: continue
        avg_mo  = float(np.mean([v["monthly_avg"] for v in metrics]))
        min_pf  = float(min(v["pf"] for v in metrics))
        scored.append({"cfg":cfg,"passes":passes,"avg_mo":avg_mo,
                       "min_pf":min_pf,"detail":r})

    elapsed = time.time()-t0
    scored.sort(key=lambda x:(x["passes"],-x["avg_mo"]), reverse=True)
    alphas  = [s for s in scored if s["passes"] >= ALPHA_PERIODS]
    print(f"\nPhase 1 done in {elapsed:.1f}s — {len(scored):,} scored, "
          f"{len(alphas)} pass {ALPHA_PERIODS}+ periods\n")

    if not scored:
        print("No configs passed minimum trade count. Exiting.")
        return

    # Print Phase 1 top-10 summary
    print("Top 10 Phase-1 candidates:")
    print(f"  {'#':>3} {'P':>2} {'avg_mo':>8} {'min_pf':>7}  Config")
    for j, s in enumerate(scored[:10]):
        cfg=s["cfg"]
        sm=(f"SL={cfg['sl_mode']}×{cfg['sl_atr_m']}[{cfg['sl_min']}-{cfg['sl_max']}]"
            if cfg['sl_mode']!='fixed' else f"SL={cfg['sl_fixed']}pt")
        sess=("ARB+NYO" if cfg["use_arb"] and cfg["use_nyo"]
              else ("ARB" if cfg["use_arb"] else "NYO"))
        ex=[]
        if cfg["trend"]: ex.append("trend")
        if cfg["dow"]!="all": ex.append(cfg["dow"])
        if cfg["min_adx"]>0: ex.append(f"adx≥{cfg['min_adx']}")
        if cfg["atr_exp"]: ex.append("atr-exp")
        print(f"  {j+1:>3} {s['passes']:>2}  ${s['avg_mo']:>7,.0f}  {s['min_pf']:>6.3f}  "
              f"TP×{cfg['tp_mult']} {sm} {sess} {' '.join(ex)}")

    # Deduplicate for Phase 2
    seen=set(); top=[]
    for s in scored:
        key=(s["passes"], round(s["avg_mo"],0), round(s["min_pf"],2))
        if key not in seen:
            seen.add(key); top.append(s)
        if len(top) >= top_n*3: break

    print(f"\nPhase 2 — full backtester validation on top {min(top_n,len(top))} configs …\n")
    validated=[]
    for j, s in enumerate(top[:top_n]):
        print(f"  [{j+1}/{min(top_n,len(top))}] validating …", flush=True)
        vr = validate_full(s["cfg"], df_full, arb_r, nyo_r, ind)
        full_passes = sum(1 for lbl,_,_ in PERIODS
                          if vr.get(lbl) and vr[lbl]["pf"]>=ALPHA_PF
                          and vr[lbl]["monthly"]>=0 and vr[lbl]["mdd"]>=ALPHA_MAX_DD
                          and vr[lbl]["worst_day"]>=ALPHA_WORST_DAY
                          and vr[lbl]["trades"]>=ALPHA_MIN_TR)
        vals = [vr[lbl]["monthly"] for lbl,_,_ in PERIODS if vr.get(lbl) and vr[lbl]]
        full_avg_mo = float(np.mean(vals)) if vals else 0.0
        validated.append({"cfg":s["cfg"],"passes":full_passes,"avg_mo":full_avg_mo,"vr":vr})

    validated.sort(key=lambda x:(x["passes"],-x["avg_mo"]), reverse=True)

    print(f"\n{'='*65}")
    print(f"  ALPHA HUNT v5 — FULL VALIDATION RESULTS")
    print(f"  Criteria: PF≥{ALPHA_PF}, monthly≥$0, DD≥${ALPHA_MAX_DD}, "
          f"wd≥{ALPHA_WORST_DAY}%, trades≥{ALPHA_MIN_TR}")
    print(f"{'='*65}")

    for v in validated:
        print_validation(v["cfg"], v["vr"])

    if validated:
        best=validated[0]
        print(f"\n{'='*65}")
        print(f"  WINNER: {best['passes']}/5 periods  avg_monthly=${best['avg_mo']:,.0f}")
        print(f"{'='*65}")


if __name__ == "__main__":
    scan()
