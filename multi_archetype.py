"""
Multi-archetype edge discovery across assets, with OOS validation.

Three structurally different strategies, each extracted as R-multiple setups:
  BRK  breakout      : Asian-range break + H1 trend filter   (known: gold)
  MRV  mean-reversion: fade M15 Z-score >= 2σ back to the mean
  MOM  momentum      : H1 uptrend + M15 pullback to EMA20, enter on resume

Discipline: minimal fixed params (no per-asset tuning), chronological 70/30
split, report TRAIN and OOS expectancy. An edge counts only if positive in
BOTH train and OOS with enough trades.
"""
import warnings; warnings.filterwarnings("ignore")
import os, glob, re
import numpy as np, pandas as pd
from assets import tick_for, CLASS

RAW=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","raw")
EXCLUDE={"HSIHKD","JPN225","US30","NGCUSD"}
RISK=0.005; COMM_R=0.02

def load_raw(path):
    df=pd.read_csv(path,sep="\t"); df.columns=[c.strip("<>").lower() for c in df.columns]
    df["datetime"]=pd.to_datetime(df["date"]+" "+df["time"],format="%Y.%m.%d %H:%M:%S")
    df=df.rename(columns={"tickvol":"tick_vol"}).set_index("datetime").sort_index()
    return df[~df.index.duplicated(keep="first")][["open","high","low","close","spread"]]

def resolve(day, t, entry, d, sl, tp, end_h=21):
    """SL-first intraday resolution from the bar AFTER t. Returns R in {-1..(tp-dist)}."""
    rem=day.loc[t:].iloc[1:]; rem=rem[rem.index.hour<end_h]
    sld=abs(entry-sl); tpd=abs(tp-entry)/sld if sld>0 else 0
    for _,b in rem.iterrows():
        if d==1:
            if b["low"]<=sl: return -1.0
            if b["high"]>=tp: return tpd
        else:
            if b["high"]>=sl: return -1.0
            if b["low"]<=tp: return tpd
    last=float(rem["close"].iloc[-1]) if len(rem) else entry
    return float(np.clip(d*(last-entry)/sld, -1.0, tpd)) if sld>0 else 0.0

# ── H1 helpers ────────────────────────────────────────────────────────────────
def h1_trend_atr(df):
    h1=df.resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    prev=h1["close"].shift(1)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-prev).abs(),(h1["low"]-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(span=14,adjust=False).mean()
    ema50=h1["close"].ewm(span=50,adjust=False).mean()
    trend=np.where(h1["close"]>ema50,1,-1)
    def _ff(s):return pd.Series(s,index=h1.index).shift(1).reindex(df.index,method="ffill")
    return _ff(atr), _ff(trend).fillna(0)

# ── Archetype extractors ──────────────────────────────────────────────────────
def setups_BRK(df, sym):
    atr,trend=h1_trend_atr(df); tick=tick_for(sym); recs=[]
    for date,day in df.groupby(df.index.date):
        if pd.Timestamp(date).dayofweek>=4: continue
        asia=day[day.index.hour<8]
        if len(asia)<4: continue
        hi=asia["high"].max(); lo=asia["low"].min(); rng=hi-lo
        win=day[(day.index.hour>=8)&(day.index.hour<10)]
        for t,bar in win.iterrows():
            if t not in atr.index or pd.isna(atr[t]) or atr[t]<=0: continue
            a=float(atr[t]); tr=int(trend[t]); p=float(bar["close"])
            if not (p*0.001<=rng<=p*0.02): continue
            sld=float(np.clip(a*0.7,p*0.001,p*0.005)); buf=a*0.05
            d=1 if (tr>=0 and p>hi+buf) else (-1 if (tr<=0 and p<lo-buf) else 0)
            if d==0: continue
            R=resolve(day,t,p,d,p-d*sld,p+d*sld*3.5)
            recs.append(dict(date=pd.Timestamp(date),R=R-float(bar["spread"])*tick/sld-COMM_R,
                             win=int(R>=3.4))); break
    return pd.DataFrame(recs)

def setups_MRV(df, sym):
    """Fade M15 Z-score>=2σ; target mean, stop 1.5σ beyond entry."""
    tick=tick_for(sym); recs=[]
    c=df["close"]; ma=c.rolling(20).mean(); sd=c.rolling(20).std()
    z=(c-ma)/sd
    atr,_=h1_trend_atr(df)
    for date,day in df.groupby(df.index.date):
        if pd.Timestamp(date).dayofweek>=4: continue
        win=day[(day.index.hour>=7)&(day.index.hour<18)]
        done=False
        for t,bar in win.iterrows():
            if done or t not in z.index or pd.isna(z[t]) or pd.isna(sd[t]) or sd[t]<=0: continue
            zz=float(z[t]); p=float(bar["close"]); m=float(ma[t]); s=float(sd[t])
            if zz>=2.0:    d=-1
            elif zz<=-2.0: d=1
            else: continue
            tp=m; sl=p - d*1.5*s      # stop 1.5σ further against
            if abs(p-sl)<=0: continue
            R=resolve(day,t,p,d,sl,tp)
            sld=abs(p-sl)
            recs.append(dict(date=pd.Timestamp(date),R=R-float(bar["spread"])*tick/sld-COMM_R,
                             win=int(R>0))); done=True
    return pd.DataFrame(recs)

def setups_MOM(df, sym):
    """H1 uptrend + M15 pullback to EMA20 then resume; trend-following continuation."""
    tick=tick_for(sym); recs=[]
    atr,trend=h1_trend_atr(df)
    ema20=df["close"].ewm(span=20,adjust=False).mean()
    for date,day in df.groupby(df.index.date):
        if pd.Timestamp(date).dayofweek>=4: continue
        win=day[(day.index.hour>=7)&(day.index.hour<18)]
        done=False; prev_below=False; prev_above=False
        for t,bar in win.iterrows():
            if done or t not in trend.index or pd.isna(atr[t]) or atr[t]<=0: continue
            tr=int(trend[t]); p=float(bar["close"]); e=float(ema20[t]); a=float(atr[t])
            lo=float(bar["low"]); hi=float(bar["high"])
            if tr>0:  # uptrend: want pullback touching ema then close back above
                if lo<=e and p>e and prev_below:
                    sld=float(np.clip(a*0.7,p*0.001,p*0.006))
                    R=resolve(day,t,p,1,p-sld,p+sld*2.5)
                    recs.append(dict(date=pd.Timestamp(date),R=R-float(bar["spread"])*tick/sld-COMM_R,
                                     win=int(R>=2.4))); done=True
                prev_below=(p<e)
            elif tr<0:
                if hi>=e and p<e and prev_above:
                    sld=float(np.clip(a*0.7,p*0.001,p*0.006))
                    R=resolve(day,t,p,-1,p+sld,p-sld*2.5)
                    recs.append(dict(date=pd.Timestamp(date),R=R-float(bar["spread"])*tick/sld-COMM_R,
                                     win=int(R>=2.4))); done=True
                prev_above=(p>e)
    return pd.DataFrame(recs)

def oos_eval(s):
    if len(s)<60: return None
    s=s.sort_values("date"); cut=s["date"].quantile(0.70)
    tr=s[s["date"]<=cut]; te=s[s["date"]>cut]
    if len(te)<15 or len(tr)<25: return None
    def st(x):
        r=x["R"].values
        return dict(n=len(r),expR=r.mean(),wr=x["win"].mean()*100,
                    pf=(r[r>0].sum()/-r[r<0].sum()) if (r<0).any() else 99)
    return dict(tr=st(tr), te=st(te), full=st(s))

files={re.match(r"([A-Z0-9]+)_M15",os.path.basename(f)).group(1):f
       for f in sorted(glob.glob(os.path.join(RAW,"*_M15_*.csv")))}

print(f"Multi-archetype OOS scan ({len(files)-len(EXCLUDE)} assets, 70/30 split)\n")
print(f"  {'Symbol':<8}{'Class':<7}"
      f"{'BRK tr/oos expR':>20}{'MRV tr/oos expR':>20}{'MOM tr/oos expR':>20}")
winners=[]
for sym,path in files.items():
    if sym in EXCLUDE: continue
    df=load_raw(path)
    row=f"  {sym:<8}{CLASS.get(sym,'?'):<7}"
    for tag,fn in [("BRK",setups_BRK),("MRV",setups_MRV),("MOM",setups_MOM)]:
        try: r=oos_eval(fn(df,sym))
        except Exception: r=None
        if r is None: row+=f"{'—':>20}"; continue
        surv=(r['tr']['expR']>0 and r['te']['expR']>0 and r['te']['pf']>=1.10 and r['te']['n']>=15)
        if surv: winners.append((sym,tag,r))
        mark="✅" if surv else (" " if r['te']['expR']>0 else "·")
        row+=f"{r['tr']['expR']:>+7.2f}/{r['te']['expR']:>+6.2f}{mark:>3}"
    print(row)

print(f"\n  Edges surviving OOS (tr>0 & oos>0 & oos PF>=1.10 & n>=15): {len(winners)}")
for sym,tag,r in sorted(winners,key=lambda x:-x[2]['te']['expR']):
    print(f"    ✅ {sym:<8} {tag}  OOS: n={r['te']['n']} WR={r['te']['wr']:.0f}% "
          f"PF={r['te']['pf']:.2f} expR={r['te']['expR']:+.3f}")
