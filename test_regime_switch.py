"""
Test regime-switching ARB strategy.

Core idea: Use the best ARB config (SL=ATR×0.7[400-1600], TP×3.5, Mon-Thu)
but gate breakout direction by regime:
  - ADX >= thresh (trending)  → only trade WITH H1 EMA50 trend
  - ADX <  thresh (ranging)   → trade both BUY and SELL breakouts

Test ADX thresholds: 20, 25, 30, 35 — find sweet spot where
Current (strong bear trend, high ADX) gets trend-filtered
while Range Recovery + High-Vol Chop (choppy, lower ADX) stay unfiltered.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from backtester import Backtester, Strategy, load
from strategy_combined import compute_ranges

PT      = 0.01
LOTS    = 0.1
LOT_VAL = PT * 100 * LOTS
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


def build_h1_indicators(df_full):
    h1 = load("H1")
    prev_c = h1["close"].shift(1)
    tr = pd.concat([h1["high"]-h1["low"],
                    (h1["high"]-prev_c).abs(),
                    (h1["low"]-prev_c).abs()], axis=1).max(axis=1)
    h1_atr   = tr.ewm(span=14, adjust=False).mean()
    h1_atr_ma= h1_atr.rolling(50, min_periods=25).mean()
    h1_ema50 = h1["close"].ewm(span=50, adjust=False).mean()
    h1_trend = (h1["close"] > h1_ema50).astype(int)*2 - 1

    # Wilder ADX
    hi=h1["high"]; lo=h1["low"]
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


class RegimeSwitchARB(Strategy):
    """
    ARB-only, Mon-Thu, SL=ATR×0.7[400-1600], TP×3.5.
    Regime switch: if ADX >= adx_thresh → apply H1 EMA trend filter.
    """
    def __init__(self, bal, arb_r, ind, adx_thresh,
                 sl_atr_m=0.7, sl_min=400, sl_max=1600, tp_mult=3.5):
        self.initial_bal = self.balance = self.peak_bal = bal
        self.arb_ranges  = arb_r
        self.ind         = ind
        self.adx_thresh  = adx_thresh
        self.sl_atr_m    = sl_atr_m
        self.sl_min      = sl_min
        self.sl_max      = sl_max
        self.tp_mult     = tp_mult
        self._day        = None
        self._day_start  = bal
        self._arb_done   = False
        self._in_trade   = False
        self._dir = self._sl = self._tp = None

    def _new_day(self, d):
        self._day = d; self._day_start = self.balance; self._arb_done = False

    def next(self, i, df):
        bar=df.iloc[i]; t=df.index[i]; today=t.date(); hour=t.hour
        if today != self._day: self._new_day(today)
        if t.dayofweek >= 4: return None  # Mon-Thu only

        iv   = self.ind.iloc[i] if i < len(self.ind) else None
        atr  = float(iv["atr"]) if iv is not None and not pd.isna(iv["atr"]) else 0
        adx  = float(iv["adx"]) if iv is not None and not pd.isna(iv["adx"]) else 0
        trnd = int(iv["trend"]) if iv is not None else 0

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
        if (self._day_start - self.balance)/self.initial_bal >= 0.04:  return None
        if (self.peak_bal   - self.balance)/self.initial_bal >= 0.085: return None

        if not (8 <= hour < 10) or self._arb_done: return None
        if today not in self.arb_ranges.index: return None
        r   = self.arb_ranges.loc[today]
        rng = float(r["range_pts"])
        if not (500 <= rng <= 9000): return None

        sl  = int(np.clip(atr * self.sl_atr_m / PT, self.sl_min, self.sl_max))
        tp  = int(sl * self.tp_mult)
        close = bar["close"]; sp = bar["spread"]

        # Regime gate: trending → trend-filter, ranging → both directions
        trending = (adx >= self.adx_thresh)
        allow_buy  = (not trending) or (trnd >= 0)
        allow_sell = (not trending) or (trnd <= 0)

        if allow_buy and close > r["high"] + 30*PT:
            e = close + (sp/2)*PT
            self._sl=e-sl*PT; self._tp=e+tp*PT
            self._in_trade=True; self._dir="buy"; self._arb_done=True; return "buy"
        if allow_sell and close < r["low"] - 30*PT:
            e = close - (sp/2)*PT
            self._sl=e+sl*PT; self._tp=e-tp*PT
            self._in_trade=True; self._dir="sell"; self._arb_done=True; return "sell"
        return None

    def _c(self):
        self._in_trade=False; self._dir=self._sl=self._tp=None; return "close"


def run_period(label, start, end, df_full, arb_r, ind, adx_thresh):
    df = df_full.loc[start:end].copy()
    iv = ind.reindex(df.index, method="ffill")
    strat = RegimeSwitchARB(BALANCE, arb_r, iv, adx_thresh)
    bt    = Backtester(df, strat, lots=LOTS, initial_balance=BALANCE)
    rep   = bt.run()
    log   = rep.trade_log()
    if log.empty:
        return None

    s   = rep.summary()
    log["date"]  = log["exit_time"].dt.date
    log["month"] = log["exit_time"].dt.to_period("M")
    daily   = log.groupby("date")["pnl"].sum()
    monthly = log.groupby("month")["pnl"].sum()
    wins    = log[log["pnl"]>0]; losses=log[log["pnl"]<0]

    net = float(str(s["net_pnl"]).replace("$","").replace(",",""))
    mdd = float(str(s["max_drawdown"]).replace("$","").replace(",",""))
    pf  = wins["pnl"].sum()/-losses["pnl"].sum() if len(losses)>0 else 99

    n_days=len(daily); n_yrs=n_days/252
    dr=daily/BALANCE
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    ann_ret = (net/BALANCE)/n_yrs if n_yrs>0 else 0
    calmar  = ann_ret/(abs(mdd)/BALANCE) if mdd!=0 else 0

    # Challenge sim
    bal=BALANCE; pk=BALANCE; ch="⏳"; chd=None
    for _,t in log.iterrows():
        bal+=t["pnl"]; pk=max(pk,bal)
        if bal>=BALANCE*1.08: ch="✅ PASS"; chd=t["exit_time"].date(); break
        if pk-bal>=BALANCE*0.10: ch="❌ FAIL"; chd=t["exit_time"].date(); break

    # Direction breakdown
    buys  = log[log["direction"]=="buy"]
    sells = log[log["direction"]=="sell"]

    ok = (pf>=ALPHA_PF and monthly.mean()>=ALPHA_MONTHLY
          and mdd>=ALPHA_MAX_DD and daily.min()/BALANCE*100>=ALPHA_WORST_DAY
          and len(log)>=ALPHA_MIN_TR)

    return {
        "trades":len(log),"wr":len(wins)/len(log)*100,"pf":round(pf,2),
        "net":net,"mdd":mdd,"monthly":monthly.mean(),
        "worst_day":daily.min()/BALANCE*100,
        "sharpe":round(sharpe,2),"calmar":round(calmar,2),
        "costs":log["commission"].sum()+log["spread_cost"].sum(),
        "challenge":ch,"ch_date":chd,
        "n_buy":len(buys),"n_sell":len(sells),
        "buy_pnl":buys["pnl"].sum(),"sell_pnl":sells["pnl"].sum(),
        "ok":ok,
    }


def test_threshold(adx_thresh, df_full, arb_r, ind):
    results = {}
    for label, start, end in PERIODS:
        results[label] = run_period(label, start, end, df_full, arb_r, ind, adx_thresh)
    passes = sum(1 for r in results.values() if r and r["ok"])
    return results, passes


def print_results(adx_thresh, results, passes):
    print(f"\n{'='*70}")
    print(f"  ADX regime threshold = {adx_thresh}  ({passes}/5 periods pass)")
    print(f"{'='*70}")
    print(f"  {'Period':<22} {'Tr':>4} {'B/S':>5} {'WR%':>5} {'PF':>5} {'Net%':>6} "
          f"{'MaxDD%':>7} {'Shrp':>5} {'The5ers'}")
    for label,_,_ in PERIODS:
        r = results.get(label)
        if r is None:
            print(f"  ❌ {label:<21}  — no trades"); continue
        flag = "✅" if r["ok"] else "❌"
        bs = f"{r['n_buy']}B/{r['n_sell']}S"
        print(f"  {flag} {label:<21} {r['trades']:>4} {bs:>5} {r['wr']:>5.1f} {r['pf']:>5.2f} "
              f"{r['net']/BALANCE*100:>5.1f}% {r['mdd']/BALANCE*100:>6.2f}% "
              f"{r['sharpe']:>5.2f} {r['challenge']}"
              + (f" ({r['ch_date']})" if r["ch_date"] else ""))
    print(f"  {'─'*68}")
    avg_costs = sum(r["costs"] for r in results.values() if r)
    print(f"  Total costs ${avg_costs:,.0f}  |  "
          f"Avg monthly ${np.mean([r['monthly'] for r in results.values() if r]):,.0f}")


def main():
    print("Loading data …")
    df_full = load("M5")
    ind     = build_h1_indicators(df_full)
    arb_r, _ = compute_ranges(df_full.copy())

    # Print ADX distribution per period to understand regime coverage
    print("\nH1 ADX distribution per period (percentiles 25/50/75/90):")
    for label, start, end in PERIODS:
        mask = (df_full.index >= start) & (df_full.index <= end)
        adx_vals = ind.loc[mask, "adx"].dropna()
        p = np.percentile(adx_vals, [25, 50, 75, 90])
        print(f"  {label:<22}: p25={p[0]:.0f}  p50={p[1]:.0f}  p75={p[2]:.0f}  p90={p[3]:.0f}")

    # Test regime thresholds
    best_passes = 0; best_thresh = None; best_results = None
    for thresh in [20, 25, 28, 30, 35, 40]:
        results, passes = test_threshold(thresh, df_full, arb_r, ind)
        print_results(thresh, results, passes)
        if passes > best_passes or (passes == best_passes and thresh == 25):
            best_passes = passes; best_thresh = thresh; best_results = results

    print(f"\n{'#'*70}")
    print(f"  BEST: ADX threshold = {best_thresh}  ({best_passes}/5 periods)")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
