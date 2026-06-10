"""
17-year out-of-sample validation of ARB-trend on XAUUSD H1 (2009-2026).
Base timeframe = H1, so 'intraday' entry bars are hourly. Coarser SL/TP
resolution than M5 (assume SL-first when both in a bar = conservative),
but the directional edge test across 17 years of regimes is what matters.

SL is %-of-price clipped (scale-free) so $900 gold (2009) and $4500 gold
(2026) are treated fairly.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from backtester import load

PT=0.01; RISK=0.005; BALANCE=10_000.0
SL_M=0.7; TPx=3.5; COMM_R=0.02   # commission ~2% of risk
SL_LO_PCT=0.0010; SL_HI_PCT=0.0050

def indicators(h1):
    prev=h1["close"].shift(1)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-prev).abs(),(h1["low"]-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(span=14,adjust=False).mean()
    ema50=h1["close"].ewm(span=50,adjust=False).mean()
    trend=np.where(h1["close"]>ema50,1,-1)
    out=pd.DataFrame(index=h1.index)
    out["atr"]=atr.shift(1)
    out["trend"]=pd.Series(trend,index=h1.index).shift(1).fillna(0).astype(int)
    return out

def backtest(h1, ind):
    """ARB-trend on H1. Returns list of (date, R-multiple)."""
    res=[]
    for date, day in h1.groupby(h1.index.date):
        if pd.Timestamp(date).dayofweek>=4: continue
        asia=day[day.index.hour<8]
        if len(asia)<3: continue
        hi=asia["high"].max(); lo=asia["low"].min(); rng=hi-lo
        win=day[(day.index.hour>=8)&(day.index.hour<10)]
        if win.empty: continue
        for t,bar in win.iterrows():
            if t not in ind.index: continue
            iv=ind.loc[t]
            if pd.isna(iv["atr"]) or iv["atr"]<=0: continue
            atr=float(iv["atr"]); trnd=int(iv["trend"]); price=float(bar["close"])
            # range filter: skip tiny/huge ranges (0.1%-2% of price)
            if not (price*0.001 <= rng <= price*0.03):
                continue
            buf=atr*0.05
            sl_dist=float(np.clip(atr*SL_M, price*SL_LO_PCT, price*SL_HI_PCT))
            if sl_dist<=0: continue
            allow_buy=(trnd>=0); allow_sell=(trnd<=0)
            d=None
            if allow_buy and price>hi+buf: d=1
            elif allow_sell and price<lo-buf: d=-1
            if d is None: continue
            entry=price; tp=entry+d*sl_dist*TPx; sl=entry-d*sl_dist
            rem=day.loc[t:]; rem=rem[rem.index.hour<21]
            hit=None
            for _,b in rem.iterrows():
                if d==1:
                    if b["low"]<=sl: hit=-1.0; break
                    if b["high"]>=tp: hit=TPx; break
                else:
                    if b["high"]>=sl: hit=-1.0; break
                    if b["low"]<=tp: hit=TPx; break
            if hit is None:
                last=float(rem["close"].iloc[-1]) if len(rem) else entry
                hit=float(np.clip(d*(last-entry)/sl_dist,-1.0,TPx))
            spread_R=(float(bar["spread"])*PT)/sl_dist
            res.append((date, hit-spread_R-COMM_R))
            break
    return res

def yr_metrics(rows):
    if not rows: return None
    r=np.array([x[1] for x in rows]); ret=r*RISK
    wins=r[r>0]; losses=r[r<0]
    pf=wins.sum()/-losses.sum() if losses.sum()<0 else 99
    cum=np.cumsum(ret); peak=np.maximum.accumulate(cum); mdd=(cum-peak).min()
    sh=ret.mean()/ret.std()*np.sqrt(252) if ret.std()>0 else 0
    return dict(n=len(r),wr=(r>0).mean()*100,pf=pf,expR=r.mean(),
                net=cum[-1]*100,mdd=mdd*100,sharpe=sh)

print("Loading 17-year H1 …")
h1=load("H1"); ind=indicators(h1)
allres=backtest(h1, ind)
print(f"Total trades 2009-2026: {len(allres)}\n")

# Per year
print(f"  {'Year':<6}{'N':>5}{'WR%':>6}{'PF':>6}{'ExpR':>7}{'Net%':>8}{'MaxDD%':>8}{'Shrp':>6}  Edge")
byyr={}
for date,rv in allres: byyr.setdefault(date.year,[]).append((date,rv))
ok=0; tot_net=0; pos_years=0
for y in sorted(byyr):
    m=yr_metrics(byyr[y])
    edge="✅" if (m["pf"]>=1.10 and m["expR"]>0) else ("~" if m["pf"]>=1.0 else "❌")
    if m["pf"]>=1.10 and m["expR"]>0: ok+=1
    if m["net"]>0: pos_years+=1
    tot_net+=m["net"]
    print(f"  {y:<6}{m['n']:>5}{m['wr']:>6.1f}{m['pf']:>6.2f}{m['expR']:>+7.3f}"
          f"{m['net']:>+7.1f}%{m['mdd']:>7.2f}%{m['sharpe']:>6.2f}  {edge}")

# Aggregate
allm=yr_metrics(allres)
print(f"  {'-'*60}")
print(f"  {'ALL':<6}{allm['n']:>5}{allm['wr']:>6.1f}{allm['pf']:>6.2f}{allm['expR']:>+7.3f}"
      f"{allm['net']:>+7.1f}%{allm['mdd']:>7.2f}%{allm['sharpe']:>6.2f}")
print(f"\n  Profitable years: {pos_years}/{len(byyr)}   "
      f"Edge years (PF>=1.1): {ok}/{len(byyr)}   "
      f"17yr Sharpe {allm['sharpe']:.2f}")
print(f"  Note: H1-resolution backtest (coarser than M5). %-of-price SL, "
      f"risk {RISK*100:.1f}%/trade.")
