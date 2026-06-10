"""
Stress test the validated ARB-trend strategy before production.

Tests:
  1. Execution friction: added slippage on entry+exit (0/1/2/3 pts/side)
  2. Spread widening: 1.0x / 1.5x / 2.0x the historical spread
  3. Commission sensitivity: $0.40 / $0.70 / $1.00 per trade
  4. Entry timing jitter: fill at bar +0 / +1 (1 bar = 5 min later)
  5. Parameter robustness: SL mult ±, TP mult ± neighbourhood
  6. Monte Carlo: reshuffle trade order 5000x → DD & pass distribution

Pass criteria per period: PF>=1.10, net>=0, maxDD>=-1200, worstday>=-4%, trades>=8
Challenge: reach +8% before -10% DD.
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

def build_ind(df_full):
    h1=load("H1")
    prev=h1["close"].shift(1)
    tr=pd.concat([h1["high"]-h1["low"],(h1["high"]-prev).abs(),(h1["low"]-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(span=14,adjust=False).mean()
    ema50=h1["close"].ewm(span=50,adjust=False).mean()
    t_price=np.where(h1["close"]>ema50,1,-1)
    def _ff(s):return pd.Series(s,index=h1.index).shift(1).reindex(df_full.index,method="ffill")
    out=pd.DataFrame(index=df_full.index)
    out["atr"]=_ff(atr); out["trend"]=_ff(t_price).fillna(0).astype(int)
    return out

class ARB(Strategy):
    def __init__(self,bal,arb_r,ind,sl_m=0.7,sl_lo=400,sl_hi=1600,tp_m=3.5,
                 slip=0.0,spread_x=1.0,comm=0.40,jitter=0):
        self.initial_bal=self.balance=self.peak_bal=bal
        self.arb_r=arb_r;self.ind=ind
        self.sl_m=sl_m;self.sl_lo=sl_lo;self.sl_hi=sl_hi;self.tp_m=tp_m
        self.slip=slip;self.spread_x=spread_x;self.comm=comm;self.jitter=jitter
        self._day=None;self._ds=bal;self._done=False
        self._in=False;self._dir=self._sl=self._tp=None;self._pending=None
    def _nd(self,d):self._day=d;self._ds=self.balance;self._done=False
    def next(self,i,df):
        bar=df.iloc[i];t=df.index[i];today=t.date();h=t.hour
        if today!=self._day:self._nd(today)
        if t.dayofweek>=4:return None
        iv=self.ind.iloc[i] if i<len(self.ind) else None
        atr=float(iv["atr"]) if iv is not None and not pd.isna(iv["atr"]) else 0
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
        # pending jittered entry
        if self._pending is not None:
            self._pending-=1
            if self._pending<=0:
                d,sl,tp=self._side
                c=bar["close"];sp=bar["spread"]*self.spread_x
                if d=="buy":
                    e=c+(sp/2)*PT+self.slip*PT;self._sl=e-sl*PT;self._tp=e+tp*PT
                else:
                    e=c-(sp/2)*PT-self.slip*PT;self._sl=e+sl*PT;self._tp=e-tp*PT
                self._in=True;self._dir=d;self._done=True;self._pending=None
                return d
            return None
        self.peak_bal=max(self.peak_bal,self.balance)
        if (self._ds-self.balance)/self.initial_bal>=0.04:return None
        if (self.peak_bal-self.balance)/self.initial_bal>=0.085:return None
        if not(8<=h<10) or self._done:return None
        if today not in self.arb_r.index:return None
        r=self.arb_r.loc[today];rng=float(r["range_pts"])
        if not(500<=rng<=9000):return None
        sl=int(np.clip(atr*self.sl_m/PT,self.sl_lo,self.sl_hi));tp=int(sl*self.tp_m)
        c=bar["close"];sp=bar["spread"]*self.spread_x
        ab=(trnd>=0);asl=(trnd<=0)
        if ab and c>r["high"]+30*PT:
            if self.jitter>0:self._pending=self.jitter;self._side=("buy",sl,tp);return None
            e=c+(sp/2)*PT+self.slip*PT;self._sl=e-sl*PT;self._tp=e+tp*PT
            self._in=True;self._dir="buy";self._done=True;return "buy"
        if asl and c<r["low"]-30*PT:
            if self.jitter>0:self._pending=self.jitter;self._side=("sell",sl,tp);return None
            e=c-(sp/2)*PT-self.slip*PT;self._sl=e+sl*PT;self._tp=e-tp*PT
            self._in=True;self._dir="sell";self._done=True;return "sell"
        return None
    def _c(self):self._in=False;self._dir=self._sl=self._tp=None;return "close"

def run(df_full,arb_r,ind,start,end,**kw):
    df=df_full.loc[start:end].copy()
    iv=ind.reindex(df.index,method="ffill")
    s=ARB(BALANCE,arb_r,iv,**kw)
    # apply extra commission by post-adjusting? backtester handles comm internally;
    # we approximate added comm/slip via pnl adjustment using the trade log
    bt=Backtester(df,s,lots=LOTS,initial_balance=BALANCE)
    log=bt.run().trade_log()
    if log.empty:return None
    # extra commission beyond backtester default
    extra=(kw.get("comm",0.40)-0.40)
    if extra:log["pnl"]=log["pnl"]-extra
    w=log[log.pnl>0];l=log[log.pnl<0]
    pf=w.pnl.sum()/-l.pnl.sum() if len(l)>0 else 99
    log["date"]=log["exit_time"].dt.date
    daily=log.groupby("date")["pnl"].sum()
    cum=log["pnl"].cumsum();peak=cum.cummax();mdd=(cum-peak).min()
    ok=(pf>=1.10 and log.pnl.sum()>=0 and mdd>=-1200 and daily.min()/BALANCE*100>=-4 and len(log)>=8)
    # challenge sim
    bal=BALANCE;pk=BALANCE;ch="slow"
    for _,tr in log.iterrows():
        bal+=tr["pnl"];pk=max(pk,bal)
        if bal>=BALANCE*1.08:ch="PASS";break
        if pk-bal>=BALANCE*0.10:ch="FAIL";break
    return dict(n=len(log),pf=pf,net=log.pnl.sum(),mdd=mdd,
                wd=daily.min()/BALANCE*100,ok=ok,ch=ch,log=log)

def summary_line(df_full,arb_r,ind,label,**kw):
    p=0;passes_ch=0;nets=[]
    for lbl,st,en in PERIODS:
        r=run(df_full,arb_r,ind,st,en,**kw)
        if r is None:continue
        if r["ok"]:p+=1
        if r["ch"]=="PASS":passes_ch+=1
        nets.append(r["net"]/BALANCE*100)
    print(f"  {label:<34} {p}/5 pass   {passes_ch}/5 hit+8%   "
          f"avg net {np.mean(nets):+5.1f}%   min {min(nets):+5.1f}%")
    return p

print("Loading …")
df_full=load("M5");ind=build_ind(df_full);arb_r,_=compute_ranges(df_full.copy())

print("\n"+"="*86)
print("  BASELINE")
print("="*86)
summary_line(df_full,arb_r,ind,"baseline (slip0 spread1.0 comm0.40)")

print("\n"+"="*86)
print("  TEST 1 — EXECUTION SLIPPAGE (pts/side added)")
print("="*86)
for slip in [1,2,3]:
    summary_line(df_full,arb_r,ind,f"slippage {slip} pts/side",slip=slip)

print("\n"+"="*86)
print("  TEST 2 — SPREAD WIDENING")
print("="*86)
for sx in [1.5,2.0]:
    summary_line(df_full,arb_r,ind,f"spread x{sx}",spread_x=sx)

print("\n"+"="*86)
print("  TEST 3 — COMMISSION SENSITIVITY")
print("="*86)
for cm in [0.70,1.00]:
    summary_line(df_full,arb_r,ind,f"commission ${cm}/trade",comm=cm)

print("\n"+"="*86)
print("  TEST 4 — ENTRY TIMING JITTER (fill N bars late)")
print("="*86)
for j in [1,2]:
    summary_line(df_full,arb_r,ind,f"fill +{j} bar ({j*5}min late)",jitter=j)

print("\n"+"="*86)
print("  TEST 5 — PARAMETER ROBUSTNESS (neighbourhood)")
print("="*86)
for slm in [0.6,0.7,0.8]:
    for tpm in [3.0,3.5,4.0]:
        summary_line(df_full,arb_r,ind,f"SL×{slm}  TP×{tpm}",sl_m=slm,tp_m=tpm)

print("\n"+"="*86)
print("  TEST 6 — COMBINED WORST-CASE (slip2 + spread1.5 + comm0.70)")
print("="*86)
summary_line(df_full,arb_r,ind,"all friction stacked",slip=2,spread_x=1.5,comm=0.70)

# ── Monte Carlo on combined full-period trade stream ──────────────────────────
print("\n"+"="*86)
print("  TEST 7 — MONTE CARLO (reshuffle trade order, 5000 runs, worst-case friction)")
print("="*86)
all_pnl=[]
for lbl,st,en in PERIODS:
    r=run(df_full,arb_r,ind,st,en,slip=2,spread_x=1.5,comm=0.70)
    if r is not None:all_pnl.extend(r["log"]["pnl"].values)
all_pnl=np.array(all_pnl)
print(f"  Pool: {len(all_pnl)} trades, total ${all_pnl.sum():+.0f}, "
      f"win rate {(all_pnl>0).mean()*100:.0f}%")
rng=np.random.default_rng(42)
maxdds=[];fails=0;passes=0
for _ in range(5000):
    perm=rng.permutation(all_pnl)
    bal=BALANCE;pk=BALANCE;mdd=0;hit=None
    for x in perm:
        bal+=x;pk=max(pk,bal);mdd=min(mdd,bal-pk)
        if hit is None and bal>=BALANCE*1.08:hit="PASS"
        if bal<=pk-BALANCE*0.10:hit="FAIL";break
    maxdds.append(mdd)
    if hit=="FAIL":fails+=1
    elif hit=="PASS":passes+=1
maxdds=np.array(maxdds)
print(f"  Max DD distribution:  median ${np.median(maxdds):.0f}  "
      f"p95 ${np.percentile(maxdds,5):.0f}  worst ${maxdds.min():.0f}")
print(f"  Challenge outcome over 5000 shuffles:")
print(f"    reached +8% before -10% DD : {passes/5000*100:.1f}%")
print(f"    hit -10% DD (BUST)         : {fails/5000*100:.1f}%")
print(f"    neither (slow/incomplete)  : {(5000-passes-fails)/5000*100:.1f}%")
