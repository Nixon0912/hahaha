"""
Test volatility gates to make the strategy SIT OUT calm regimes instead of
losing in them. Gates are self-adjusting (relative to price/own history) so
they work across $1800 (2022) and $4500 (2026) gold alike.

Gate variants:
  none      : trade always (baseline)
  atr_exp   : only trade if H1 ATR >= its own 50-bar MA (expansion)
  atr_pct_X : only trade if H1 ATR/price >= X%  (relative vol floor)
  combo     : atr_exp AND atr_pct>=0.15%
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from backtester import Backtester, Strategy, load
from strategy_combined import compute_ranges

PT=0.01; LOT=0.04; BALANCE=10_000.0
PERIODS=[("2022 Bear","2022-03-18","2022-10-31"),
         ("2022-23 Recov","2022-11-01","2023-04-30"),
         ("2023 Range","2023-05-01","2023-09-30"),
         ("2023-24 Brk","2023-10-01","2024-03-31"),
         ("2024 Bull","2024-04-01","2024-10-31"),
         ("2024 Corr","2024-11-01","2025-01-31"),
         ("2025 H1 Bull","2025-02-01","2025-05-31"),
         ("2025 H2 Bull","2025-06-01","2025-12-31"),
         ("2026 Para+Crash","2026-01-01","2026-06-10")]

def build_ind(df):
    h1=load("H1"); prev=h1["close"].shift(1)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-prev).abs(),(h1["low"]-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(span=14,adjust=False).mean()
    atr_ma=atr.rolling(50,min_periods=25).mean()
    ema50=h1["close"].ewm(span=50,adjust=False).mean()
    trend=np.where(h1["close"]>ema50,1,-1)
    atr_pct=atr/h1["close"]*100
    def _ff(s):return pd.Series(s,index=h1.index).shift(1).reindex(df.index,method="ffill")
    o=pd.DataFrame(index=df.index)
    o["atr"]=_ff(atr);o["atr_ma"]=_ff(atr_ma);o["atr_pct"]=_ff(atr_pct)
    o["trend"]=_ff(trend).fillna(0).astype(int)
    return o

class GatedARB(Strategy):
    def __init__(self,bal,arb_r,ind,gate):
        self.initial_bal=self.balance=self.peak_bal=bal
        self.arb_r=arb_r;self.ind=ind;self.gate=gate
        self._day=None;self._ds=bal;self._done=False
        self._in=False;self._dir=self._sl=self._tp=None
    def _nd(self,d):self._day=d;self._ds=self.balance;self._done=False
    def _pass_gate(self,atr,atr_ma,atr_pct):
        g=self.gate
        if g=="none":return True
        if g=="atr_exp":return atr>=atr_ma
        if g.startswith("atr_pct"):return atr_pct>=float(g.split("_")[-1])
        if g=="combo":return (atr>=atr_ma) and (atr_pct>=0.15)
        return True
    def next(self,i,df):
        bar=df.iloc[i];t=df.index[i];today=t.date();h=t.hour
        if today!=self._day:self._nd(today)
        if t.dayofweek>=4:return None
        iv=self.ind.iloc[i] if i<len(self.ind) else None
        atr=float(iv["atr"]) if iv is not None and not pd.isna(iv["atr"]) else 0
        atr_ma=float(iv["atr_ma"]) if iv is not None and not pd.isna(iv["atr_ma"]) else 0
        atr_pct=float(iv["atr_pct"]) if iv is not None and not pd.isna(iv["atr_pct"]) else 0
        trnd=int(iv["trend"]) if iv is not None else 0
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
        if not self._pass_gate(atr,atr_ma,atr_pct):return None
        r=self.arb_r.loc[today];rng=float(r["range_pts"])
        if not(500<=rng<=9000):return None
        sl=int(np.clip(atr*0.7/PT,400,1600));tp=int(sl*3.5)
        c=bar["close"];sp=bar["spread"]
        ab=(trnd>=0);asl=(trnd<=0)
        if ab and c>r["high"]+30*PT:
            e=c+(sp/2)*PT;self._sl=e-sl*PT;self._tp=e+tp*PT
            self._in=True;self._dir="buy";self._done=True;return "buy"
        if asl and c<r["low"]-30*PT:
            e=c-(sp/2)*PT;self._sl=e+sl*PT;self._tp=e-tp*PT
            self._in=True;self._dir="sell";self._done=True;return "sell"
        return None
    def _c(self):self._in=False;self._dir=self._sl=self._tp=None;return "close"

def run(m15,ind,arb_r,gate,st,en):
    df=m15.loc[st:en].copy();iv=ind.reindex(df.index,method="ffill")
    bt=Backtester(df,GatedARB(BALANCE,arb_r,iv,gate),lots=LOT,initial_balance=BALANCE)
    log=bt.run().trade_log()
    if log.empty:return dict(n=0,ret=0.0,pf=0,ok=True,flat=True)  # flat = no loss
    w=log[log.pnl>0];l=log[log.pnl<0]
    pf=w.pnl.sum()/-l.pnl.sum() if len(l)>0 else 99
    log["date"]=log["exit_time"].dt.date;daily=log.groupby("date")["pnl"].sum()
    cum=log["pnl"].cumsum();mdd=(cum-cum.cummax()).min()
    net=log.pnl.sum()
    # acceptable = positive OR flat; not losing
    ok=(net>=0 and mdd>=-1200*(LOT/0.10) and daily.min()/BALANCE*100>=-4)
    return dict(n=len(log),ret=net/BALANCE*100,pf=pf,ok=ok,flat=False)

print("Loading …")
m15=load("M15");ind=build_ind(m15);arb_r,_=compute_ranges(m15.copy())

gates=["none","atr_exp","atr_pct_0.15","atr_pct_0.20","atr_pct_0.25","combo"]
print(f"\n{'Period':<17}"+"".join(f"{g.replace('atr_pct_','vp'):>13}" for g in gates))
print("-"*(17+13*len(gates)))
tot={g:[0,0,0] for g in gates}  # ok_count, total_ret, total_trades
for lbl,st,en in PERIODS:
    line=f"{lbl:<17}"
    for g in gates:
        r=run(m15,ind,arb_r,g,st,en)
        if r["ok"]:tot[g][0]+=1
        tot[g][1]+=r["ret"];tot[g][2]+=r["n"]
        mark="·" if r["n"]==0 else ("✓" if r["ok"] else "✗")
        line+=f"{mark}{r['ret']:>+6.1f}%({r['n']:>2})"
    print(line)
print("-"*(17+13*len(gates)))
print(f"{'OK / 9 periods':<17}"+"".join(f"{tot[g][0]:>10}/9 " for g in gates))
print(f"{'total ret %':<17}"+"".join(f"{tot[g][1]:>+11.1f} " for g in gates))
print(f"{'total trades':<17}"+"".join(f"{tot[g][2]:>11} " for g in gates))
print("\nLegend: ✓=positive/pass  ✗=losing  ·=flat(no trades, no loss)")
print("'OK' counts periods that are positive OR flat (not losing).")
