"""
Deeper drill: compare trend-filter definitions across all 5 periods.

Variants:
  A. price_vs_ema50  (current)  : trend = +1 if close>EMA50 else -1
  B. ema50_slope                : trend = sign(EMA50 - EMA50.shift(N))
  C. ema_stack       (50 vs 200): trend = +1 if EMA50>EMA200 else -1
  D. dual (slope AND price)     : both must agree, else trend=0 (block both? no—)

For each, BUY allowed if trend>=0, SELL allowed if trend<=0 (regime always trending).
Goal: kill losing BUYs in bear (Current) without killing winning BUYs in bull (Steady Uptrend).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from backtester import Backtester, Strategy, load
from strategy_combined import compute_ranges

PT=0.01; LOTS=0.1; LOT_VAL=PT*100*LOTS; BALANCE=10_000.0
PERIODS=[("Steady Uptrend","2025-01-07","2025-03-31"),
         ("Post-ATH Correct","2025-06-01","2025-08-31"),
         ("Range Recovery","2025-09-01","2025-12-31"),
         ("High-Vol Chop","2026-01-01","2026-03-31"),
         ("Current","2026-04-01","2026-06-09")]

def build_trends(df_full):
    """Return DataFrame indexed to M5 with atr + 4 trend variants."""
    h1=load("H1")
    prev=h1["close"].shift(1)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-prev).abs(),(h1["low"]-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(span=14,adjust=False).mean()
    ema50=h1["close"].ewm(span=50,adjust=False).mean()
    ema200=h1["close"].ewm(span=200,adjust=False).mean()

    t_price = np.where(h1["close"]>ema50,1,-1)
    t_slope = np.sign(ema50 - ema50.shift(6)).fillna(0).astype(int)   # 6 H1 bars
    t_stack = np.where(ema50>ema200,1,-1)

    def _ff(s): return pd.Series(s,index=h1.index).shift(1).reindex(df_full.index,method="ffill")
    out=pd.DataFrame(index=df_full.index)
    out["atr"]=_ff(atr)
    out["t_price"]=_ff(t_price).fillna(0).astype(int)
    out["t_slope"]=_ff(t_slope).fillna(0).astype(int)
    out["t_stack"]=_ff(t_stack).fillna(0).astype(int)
    return out

class ARB(Strategy):
    def __init__(self,bal,arb_r,ind,tcol):
        self.initial_bal=self.balance=self.peak_bal=bal
        self.arb_r=arb_r; self.ind=ind; self.tcol=tcol
        self._day=None;self._ds=bal;self._done=False
        self._in=False;self._dir=self._sl=self._tp=None
    def _nd(self,d): self._day=d;self._ds=self.balance;self._done=False
    def next(self,i,df):
        bar=df.iloc[i];t=df.index[i];today=t.date();h=t.hour
        if today!=self._day:self._nd(today)
        if t.dayofweek>=4:return None
        iv=self.ind.iloc[i] if i<len(self.ind) else None
        atr=float(iv["atr"]) if iv is not None and not pd.isna(iv["atr"]) else 0
        trnd=int(iv[self.tcol]) if iv is not None else 0
        if self._in:
            self.peak_bal=max(self.peak_bal,self.balance)
            if h>=21 or (self.peak_bal-self.balance)/self.initial_bal>=0.085:return self._c()
            if self._dir=="buy":
                if bar["low"]<=self._sl:return self._c()
                if bar["high"]>=self._tp:return self._c()
            else:
                if bar["high"]>=self._sl:return self._c()
                if bar["low"]<=self._tp:return self._c()
            return None
        self.peak_bal=max(self.peak_bal,self.balance)
        if (self._ds-self.balance)/self.initial_bal>=0.04:return None
        if (self.peak_bal-self.balance)/self.initial_bal>=0.085:return None
        if not(8<=h<10) or self._done:return None
        if today not in self.arb_r.index:return None
        r=self.arb_r.loc[today];rng=float(r["range_pts"])
        if not(500<=rng<=9000):return None
        sl=int(np.clip(atr*0.7/PT,400,1600));tp=int(sl*3.5)
        c=bar["close"];sp=bar["spread"]
        allow_buy=(trnd>=0);allow_sell=(trnd<=0)
        if allow_buy and c>r["high"]+30*PT:
            e=c+(sp/2)*PT;self._sl=e-sl*PT;self._tp=e+tp*PT
            self._in=True;self._dir="buy";self._done=True;return "buy"
        if allow_sell and c<r["low"]-30*PT:
            e=c-(sp/2)*PT;self._sl=e+sl*PT;self._tp=e-tp*PT
            self._in=True;self._dir="sell";self._done=True;return "sell"
        return None
    def _c(self):self._in=False;self._dir=self._sl=self._tp=None;return "close"

def run(df_full,arb_r,ind,tcol,start,end):
    df=df_full.loc[start:end].copy()
    iv=ind.reindex(df.index,method="ffill")
    bt=Backtester(df,ARB(BALANCE,arb_r,iv,tcol),lots=LOTS,initial_balance=BALANCE)
    log=bt.run().trade_log()
    if log.empty:return None
    w=log[log.pnl>0];l=log[log.pnl<0]
    pf=w.pnl.sum()/-l.pnl.sum() if len(l)>0 else 99
    log["date"]=log["exit_time"].dt.date
    daily=log.groupby("date")["pnl"].sum()
    cum=log["pnl"].cumsum();peak=cum.cummax();mdd=(cum-peak).min()
    b=log[log.direction=="buy"];s=log[log.direction=="sell"]
    ok=(pf>=1.10 and log.pnl.sum()>=0 and mdd>=-1200 and daily.min()/BALANCE*100>=-4 and len(log)>=8)
    return dict(n=len(log),pf=pf,net=log.pnl.sum(),mdd=mdd,
                wd=daily.min()/BALANCE*100,
                nb=len(b),bn=b.pnl.sum(),ns=len(s),sn=s.pnl.sum(),ok=ok)

df_full=load("M5");ind=build_trends(df_full);arb_r,_=compute_ranges(df_full.copy())

for tcol,name in [("t_price","A. price vs EMA50 (current)"),
                  ("t_slope","B. EMA50 slope (6 H1 bars)"),
                  ("t_stack","C. EMA50 vs EMA200 stack")]:
    print(f"\n{'='*86}\n  {name}\n{'='*86}")
    print(f"  {'Period':<20}{'N':>4}{'PF':>6}{'Net%':>7}{'MaxDD%':>8}{'WD%':>6}  "
          f"{'BUY':>14}  {'SELL':>14}")
    p=0
    for lbl,st,en in PERIODS:
        r=run(df_full,arb_r,ind,tcol,st,en)
        if r is None:print(f"  {lbl:<20} no trades");continue
        if r["ok"]:p+=1
        fl="✅"if r["ok"]else"❌"
        print(f"  {fl}{lbl:<18}{r['n']:>4}{r['pf']:>6.2f}{r['net']/BALANCE*100:>6.1f}%"
              f"{r['mdd']/BALANCE*100:>7.2f}%{r['wd']:>5.1f}%  "
              f"{r['nb']:>2}t ${r['bn']:>+7.0f}  {r['ns']:>2}t ${r['sn']:>+7.0f}")
    print(f"  → {p}/5 pass")
