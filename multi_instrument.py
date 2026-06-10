"""
Cross-instrument test of the ARB-trend breakout across all 25 M15 instruments.

Goal 1 (robustness): does the edge generalise, or is it gold-only (overfit)?
Goal 2 (diversification): would trading several uncorrelated symbols on one
        account give a more robust aggregate equity curve than gold alone?

Method: the strategy is scale-free in PRICE terms because SL/TP are ATR-based
and we measure everything in % of entry price (R-multiples), NOT dollars.
That lets us compare a $4500 gold bar against a 1.08 EURUSD bar fairly.
Each trade's result is expressed as % account risked-return assuming a fixed
fractional risk per trade (risk_pct of balance on the SL distance).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, glob, re; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
RISK_PCT = 0.005   # risk 0.5% of balance per trade (fixed-fractional, scale-free)

def load_raw(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"]+" "+df["time"], format="%Y.%m.%d %H:%M:%S")
    df = df.rename(columns={"tickvol":"tick_vol"}).set_index("datetime").sort_index()
    return df[["open","high","low","close","tick_vol","spread"]]

def h1_from(df):
    return df.resample("1h").agg({"open":"first","high":"max","low":"min",
                                  "close":"last"}).dropna()

def indicators(df, h1):
    prev=h1["close"].shift(1)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-prev).abs(),(h1["low"]-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(span=14,adjust=False).mean()
    ema50=h1["close"].ewm(span=50,adjust=False).mean()
    trend=np.where(h1["close"]>ema50,1,-1)
    def _ff(s):return pd.Series(s,index=h1.index).shift(1).reindex(df.index,method="ffill")
    out=pd.DataFrame(index=df.index)
    out["atr"]=_ff(atr); out["trend"]=_ff(trend).fillna(0).astype(int)
    return out

def asian_ranges(df):
    """ARB range 00:00-07:55, returns dict date->(hi,lo)."""
    sub=df[(df.index.hour<8)]
    g=sub.groupby(sub.index.date)
    return {d:(x["high"].max(),x["low"].min()) for d,x in g}

def backtest(df, ind, ranges):
    """ARB-trend on M15. Returns list of R-multiples (pnl / risk)."""
    results=[]  # each = R multiple (TP hit=+3.5, SL=-1, minus costs in R)
    SL_M=0.7; TPx=3.5
    by_day=df.groupby(df.index.date)
    for date, day in by_day:
        if pd.Timestamp(date).dayofweek>=4: continue
        if date not in ranges: continue
        hi,lo=ranges[date]
        rng=hi-lo
        # entry window 08:00-10:00
        win=day[(day.index.hour>=8)&(day.index.hour<10)]
        if win.empty: continue
        # need atr+trend at entry
        for t,bar in win.iterrows():
            iv=ind.loc[t] if t in ind.index else None
            if iv is None or pd.isna(iv["atr"]) or iv["atr"]<=0: continue
            atr=float(iv["atr"]); trnd=int(iv["trend"])
            price=bar["close"]
            buf=atr*0.05
            # SL = ATR×0.7, clipped to [0.10%, 0.50%] of price (scale-free
            # equivalent of gold's 400-1600pt clip: 4-16 on $4500 ≈ 0.09-0.36%)
            sl_dist=atr*SL_M
            sl_dist=float(np.clip(sl_dist, price*0.0010, price*0.0050))
            if sl_dist<=0: continue
            allow_buy=(trnd>=0); allow_sell=(trnd<=0)
            direction=None
            if allow_buy and price>hi+buf: direction=1
            elif allow_sell and price<lo-buf: direction=-1
            if direction is None: continue
            entry=price
            tp=entry+direction*sl_dist*TPx
            sl=entry-direction*sl_dist
            # resolve over remainder of day (intraday, close by 21:00)
            rem=day.loc[t:]
            rem=rem[rem.index.hour<21]
            hit=0.0; resolved=False
            for _,b in rem.iterrows():
                if direction==1:
                    if b["low"]<=sl: hit=-1.0; resolved=True; break
                    if b["high"]>=tp: hit=TPx; resolved=True; break
                else:
                    if b["high"]>=sl: hit=-1.0; resolved=True; break
                    if b["low"]<=tp: hit=TPx; resolved=True; break
            if not resolved:
                # close at last bar price → partial R (capped to [-1,TPx])
                last=rem["close"].iloc[-1] if len(rem) else entry
                hit=float(np.clip(direction*(last-entry)/sl_dist, -1.0, TPx))
            # flat small cost in R (spread+commission), scale-free approximation
            cost_R=0.05
            results.append((date, hit-cost_R))
            break
    return results

def metrics(rmults):
    if not rmults: return None
    r=np.array([x[1] for x in rmults])
    # account return per trade = R * RISK_PCT
    ret=r*RISK_PCT
    wins=r[r>0]; losses=r[r<0]
    pf=wins.sum()/-losses.sum() if losses.sum()<0 else 99
    cum=np.cumsum(ret); peak=np.maximum.accumulate(cum); mdd=(cum-peak).min()
    sharpe=ret.mean()/ret.std()*np.sqrt(252*1) if ret.std()>0 else 0  # rough
    return dict(n=len(r), wr=(r>0).mean()*100, pf=pf, expR=r.mean(),
                totR=r.sum(), net=cum[-1]*100, mdd=mdd*100, sharpe=sharpe)

files=sorted(glob.glob(os.path.join(RAW,"*.csv")))
print(f"Testing ARB-trend across {len(files)} instruments "
      f"(risk {RISK_PCT*100:.1f}%/trade, R-multiple basis)\n")
print(f"  {'Instrument':<12}{'N':>5}{'WR%':>6}{'PF':>6}{'ExpR':>7}{'TotR':>7}"
      f"{'Net%':>7}{'MaxDD%':>8}{'Shrp':>6}  Edge")
rows=[]; per_day_returns={}
for f in files:
    sym=re.match(r"([A-Z0-9]+)_M15", os.path.basename(f)).group(1)
    try:
        df=load_raw(f); h1=h1_from(df); ind=indicators(df,h1); rg=asian_ranges(df)
        res=backtest(df,ind,rg); m=metrics(res)
    except Exception as e:
        print(f"  {sym:<12}  ERROR {e}"); continue
    if m is None: continue
    edge="✅" if (m["pf"]>=1.10 and m["expR"]>0) else ("~" if m["pf"]>=1.0 else "❌")
    print(f"  {sym:<12}{m['n']:>5}{m['wr']:>6.1f}{m['pf']:>6.2f}{m['expR']:>+7.3f}"
          f"{m['totR']:>+7.1f}{m['net']:>+6.1f}%{m['mdd']:>7.2f}%{m['sharpe']:>6.2f}  {edge}")
    rows.append((sym,m))
    # store daily R for portfolio aggregation
    dd={}
    for date,rv in res: dd[date]=dd.get(date,0)+rv
    per_day_returns[sym]=dd

# ── Portfolio: equal-weight across instruments with positive edge ─────────────
edged=[s for s,m in rows if m["pf"]>=1.10 and m["expR"]>0]
print(f"\n  Instruments with edge (PF>=1.10 & ExpR>0): {len(edged)}/{len(rows)}")
print(f"  {', '.join(edged)}")

if len(edged)>=2:
    all_dates=sorted(set().union(*[set(per_day_returns[s]) for s in edged]))
    port=[]
    for d in all_dates:
        day_R=[per_day_returns[s][d] for s in edged if d in per_day_returns[s]]
        if day_R: port.append(np.mean(day_R)*RISK_PCT)  # equal weight that day
    port=np.array(port)
    cum=np.cumsum(port); peak=np.maximum.accumulate(cum); mdd=(cum-peak).min()
    sharpe=port.mean()/port.std()*np.sqrt(252) if port.std()>0 else 0
    print(f"\n  ── DIVERSIFIED PORTFOLIO (equal-weight, edged instruments) ──")
    print(f"     Trading days   : {len(port)}")
    print(f"     Net return     : {cum[-1]*100:+.1f}%  over {(all_dates[-1]-all_dates[0]).days/365:.1f} yrs")
    print(f"     Max drawdown   : {mdd*100:.2f}%")
    print(f"     Daily Sharpe   : {sharpe:.2f}")
    print(f"     Worst day      : {port.min()*100:+.2f}%")
