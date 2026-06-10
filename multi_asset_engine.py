"""
Faithful multi-asset setup-extraction engine (M15 resolution).

Mirrors the validated RegimeSwitchARB mechanics but in scale-free terms so it
works on any instrument:
  - Asian range  : bars hour<8  (high/low)
  - Range filter : 0.10%-2.0% of price
  - Entry window : 08:00-10:00, first M15 bar breaking range +- buffer
  - Trend filter : H1 EMA50 (price>EMA50 -> only longs, etc.)  [regime switch]
  - SL           : ATR(H1)x0.7 clipped to [0.10%,0.50%] of price
  - TP           : 3.5 x SL
  - Resolve      : intraday M15 bars until 21:00, SL-checked-first
  - Risk         : fixed-fractional (each trade risks RISK of balance)

Output per setup: outcome in R, plus a rich feature row for stats/ML.

SANITY GATE: run_one('XAUUSD') must be positive in 2025-2026 with WR ~25-40%,
else the engine (or params) is wrong.
"""
import warnings; warnings.filterwarnings("ignore")
import os, glob, re
import numpy as np, pandas as pd
from assets import tick_for, CLASS

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
SL_M=0.7; TPx=3.5; COMM_R=0.02
SL_LO=0.0010; SL_HI=0.0050
RANGE_LO=0.0010; RANGE_HI=0.020
BUF_ATR=0.05

def _load_raw(path):
    df=pd.read_csv(path, sep="\t")
    df.columns=[c.strip("<>").lower() for c in df.columns]
    df["datetime"]=pd.to_datetime(df["date"]+" "+df["time"], format="%Y.%m.%d %H:%M:%S")
    df=df.rename(columns={"tickvol":"tick_vol"}).set_index("datetime").sort_index()
    df=df[["open","high","low","close","tick_vol","spread"]]
    return df[~df.index.duplicated(keep="first")]

def _h1_ind(df):
    h1=df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    prev=h1["close"].shift(1)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-prev).abs(),(h1["low"]-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(span=14,adjust=False).mean()
    atr_ma=atr.rolling(50,min_periods=25).mean()
    ema50=h1["close"].ewm(span=50,adjust=False).mean()
    ema200=h1["close"].ewm(span=200,adjust=False).mean()
    trend=np.where(h1["close"]>ema50,1,-1)
    stack=np.where(ema50>ema200,1,-1)
    atr_pct=atr/h1["close"]
    atr_rank=atr.rolling(500,min_periods=100).rank(pct=True)
    def _ff(s):return pd.Series(s,index=h1.index).shift(1).reindex(df.index,method="ffill")
    o=pd.DataFrame(index=df.index)
    o["atr"]=_ff(atr); o["atr_pct"]=_ff(atr_pct); o["atr_exp"]=_ff((atr>=atr_ma).astype(int))
    o["trend"]=_ff(trend).fillna(0).astype(int); o["stack"]=_ff(stack).fillna(0).astype(int)
    o["atr_rank"]=_ff(atr_rank).fillna(0.5)
    return o

def extract_setups(sym, path):
    """Return DataFrame of breakout setups: features + outcome R."""
    df=_load_raw(path); tick=tick_for(sym)
    ind=_h1_ind(df)
    prev_close_by_day=df["close"].resample("1D").last()
    recs=[]
    for date, day in df.groupby(df.index.date):
        dow=pd.Timestamp(date).dayofweek
        if dow>=4: continue
        asia=day[day.index.hour<8]
        if len(asia)<4: continue
        hi=asia["high"].max(); lo=asia["low"].min(); rng=hi-lo
        win=day[(day.index.hour>=8)&(day.index.hour<10)]
        if win.empty: continue
        for t,bar in win.iterrows():
            if t not in ind.index: continue
            iv=ind.loc[t]
            if pd.isna(iv["atr"]) or iv["atr"]<=0: continue
            atr=float(iv["atr"]); price=float(bar["close"])
            if price<=0: continue
            rpct=rng/price
            if not (RANGE_LO<=rpct<=RANGE_HI): continue
            trnd=int(iv["trend"])
            buf=atr*BUF_ATR
            sl_dist=float(np.clip(atr*SL_M, price*SL_LO, price*SL_HI))
            if sl_dist<=0: continue
            allow_buy=(trnd>=0); allow_sell=(trnd<=0)
            d=None
            if allow_buy and price>hi+buf: d=1
            elif allow_sell and price<lo-buf: d=-1
            if d is None: continue
            entry=price; tp=entry+d*sl_dist*TPx; sl=entry-d*sl_dist
            # resolve from the NEXT bar (entry bar's own range must not trigger exits)
            rem=day.loc[t:]; rem=rem.iloc[1:]; rem=rem[rem.index.hour<21]
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
            spread_R=(float(bar["spread"])*tick)/sl_dist
            R=hit-spread_R-COMM_R
            # prior-day return feature
            pdt=pd.Timestamp(date)-pd.Timedelta(days=1)
            pc=prev_close_by_day.get(pd.Timestamp(date).normalize(),np.nan)
            recs.append(dict(
                sym=sym, date=pd.Timestamp(date), hour=t.hour, dow=dow, dir=d,
                R=R, win=int(hit==TPx),
                trend=trnd, stack=int(iv["stack"]), atr_pct=float(iv["atr_pct"]),
                atr_exp=int(iv["atr_exp"]), atr_rank=float(iv["atr_rank"]),
                range_pct=rpct, dir_with_trend=int(d==trnd),
            ))
            break
    return pd.DataFrame(recs)

def summarise(setups, label=""):
    if len(setups)==0: return None
    r=setups["R"].values; ret=r*0.005
    wins=r[r>0]; losses=r[r<0]
    pf=wins.sum()/-losses.sum() if losses.sum()<0 else 99
    cum=np.cumsum(ret); mdd=(cum-np.maximum.accumulate(cum)).min()
    sh=ret.mean()/ret.std()*np.sqrt(252) if ret.std()>0 else 0
    return dict(n=len(r),wr=setups["win"].mean()*100,pf=pf,expR=r.mean(),
                totR=r.sum(),net=cum[-1]*100,mdd=mdd*100,sharpe=sh)

def find_files():
    out={}
    for f in sorted(glob.glob(os.path.join(RAW,"*_M15_*.csv"))):
        m=re.match(r"([A-Z0-9]+)_M15",os.path.basename(f))
        if m: out[m.group(1)]=f
    return out

if __name__=="__main__":
    files=find_files()
    print(f"Found {len(files)} M15 instruments\n")
    # SANITY GATE on gold
    print("=== SANITY GATE: XAUUSD ===")
    g=extract_setups("XAUUSD", files["XAUUSD"])
    gm=summarise(g)
    print(f"  all years : n={gm['n']}  WR={gm['wr']:.1f}%  PF={gm['pf']:.2f}  "
          f"expR={gm['expR']:+.3f}  net={gm['net']:+.1f}%")
    recent=g[g["date"]>="2025-06-01"]
    rm=summarise(recent)
    print(f"  2025-H2+  : n={rm['n']}  WR={rm['wr']:.1f}%  PF={rm['pf']:.2f}  "
          f"expR={rm['expR']:+.3f}  net={rm['net']:+.1f}%")
    cur=g[g["date"]>="2026-01-01"]
    cm=summarise(cur)
    print(f"  2026      : n={cm['n']}  WR={cm['wr']:.1f}%  PF={cm['pf']:.2f}  "
          f"expR={cm['expR']:+.3f}  net={cm['net']:+.1f}%")
    gate = rm["pf"]>=1.10 and 22<=rm["wr"]<=45
    print(f"\n  GATE {'PASSED ✅' if gate else 'FAILED ❌'} "
          f"(need recent PF>=1.10 and WR 22-45%)")
    if not gate:
        print("  Engine/params do not reproduce validated gold edge — stopping.")
