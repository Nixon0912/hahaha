"""
Gold + USDJPY portfolio — The5ers challenge Monte Carlo.

Combines the three OOS-validated edges into one $10k account:
   GOLD  breakout
   USDJPY breakout
   USDJPY momentum

Fixed-fractional risk: each trade moves balance by R * RISK * 10_000.
Challenge: reach +8% (bal>=10_800) before -10% trailing DD (peak-bal>=1_000).

Reports, per RISK level:
  - historical sequence outcome + max DD
  - Monte Carlo (reshuffle trade order) bust% / pass% / median trades-to-pass
Compared head-to-head: GOLD-only vs GOLD+USDJPY portfolio.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multi_archetype import load_raw, setups_BRK, setups_MOM, files

BAL=10_000.0; rng=np.random.default_rng(20)

def get_trades():
    g=load_raw(files["XAUUSD"]); j=load_raw(files["USDJPY"])
    gb=setups_BRK(g,"XAUUSD");  gb["src"]="GOLD-BRK"
    jb=setups_BRK(j,"USDJPY");  jb["src"]="JPY-BRK"
    jm=setups_MOM(j,"USDJPY");  jm["src"]="JPY-MOM"
    return gb, jb, jm

def stream(frames):
    df=pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return df

def historical(df, risk):
    bal=BAL; peak=BAL; mdd=0; passed=None; busted=None
    for k,row in enumerate(df.itertuples(),1):
        bal+=row.R*risk*BAL; peak=max(peak,bal); mdd=min(mdd,bal-peak)
        if passed is None and bal>=BAL*1.08: passed=k
        if busted is None and peak-bal>=BAL*0.10: busted=k; break
    return dict(end=bal, mdd=mdd/BAL*100, passed=passed, busted=busted, n=len(df))

def montecarlo(R, risk, iters=8000, dd="static"):
    """dd='static' -> bust if bal<=9000 (The5ers High Stakes: 10% of initial).
       dd='trailing' -> bust if peak-bal>=1000 (harder, conservative)."""
    passes=busts=0; trades_to_pass=[]
    for _ in range(iters):
        perm=rng.permutation(R)
        bal=BAL; peak=BAL; outcome=None
        for k,r in enumerate(perm,1):
            bal+=r*risk*BAL; peak=max(peak,bal)
            if bal>=BAL*1.08: outcome="P"; trades_to_pass.append(k); break
            bust = (bal<=BAL*0.90) if dd=="static" else (peak-bal>=BAL*0.10)
            if bust: outcome="B"; break
        if outcome=="P": passes+=1
        elif outcome=="B": busts+=1
    med=np.median(trades_to_pass) if trades_to_pass else float("nan")
    return passes/iters*100, busts/iters*100, med

gb,jb,jm=get_trades()
gold=stream([gb])
port=stream([gb,jb,jm])

# trade frequency for time estimate
span_days=(port["date"].max()-port["date"].min()).days
gold_tpm=len(gold)/(span_days/30); port_tpm=len(port)/(span_days/30)

print(f"Trade pools (full history, {span_days/365:.1f} yrs):")
print(f"  GOLD-only      : {len(gold)} trades  ({gold_tpm:.1f}/mo)  "
      f"expR={gold['R'].mean():+.3f}")
for tag,fr in [("GOLD-BRK",gb),("JPY-BRK",jb),("JPY-MOM",jm)]:
    print(f"     {tag:<9}: {len(fr):>4} trades  expR={fr['R'].mean():+.3f}  "
          f"WR={fr['win'].mean()*100:.0f}%")
print(f"  PORTFOLIO      : {len(port)} trades  ({port_tpm:.1f}/mo)  "
      f"expR={port['R'].mean():+.3f}")

# correlation check: daily R, gold vs jpy
gd=gb.groupby("date")["R"].sum()
jd=pd.concat([jb,jm]).groupby("date")["R"].sum()
common=gd.index.intersection(jd.index)
corr=np.corrcoef(gd.loc[common], jd.loc[common])[0,1] if len(common)>20 else float("nan")
print(f"\n  Daily-return correlation GOLD vs USDJPY: {corr:+.2f}  "
      f"({len(common)} overlapping days)")

Rg=gold["R"].values; Rp=port["R"].values
# recent-regime portfolio (forward-looking: the regime we'll actually trade)
recent=port[port["date"]>="2024-06-01"]
Rpr=recent["R"].values
print(f"\n  Recent portfolio (2024-06+): {len(recent)} trades  expR={recent['R'].mean():+.3f}")

for dd in ["static","trailing"]:
    rule = "-10% of INITIAL (The5ers High Stakes)" if dd=="static" else "-10% TRAILING from peak (conservative)"
    print(f"\n{'='*82}")
    print(f"  CHALLENGE MONTE CARLO — DD rule: {rule}")
    print(f"{'='*82}")
    print(f"  {'Risk/tr':>8} │ {'GOLD-ONLY':^20} │ {'PORTFOLIO (full 4yr)':^22} │ {'PORTFOLIO (recent)':^22}")
    print(f"  {'':>8} │ {'pass%':>7}{'bust%':>7} │ {'pass%':>7}{'bust%':>7}{'mo':>6} │ {'pass%':>7}{'bust%':>7}{'mo':>6}")
    print(f"  {'-'*8}─┼─{'-'*20}─┼─{'-'*22}─┼─{'-'*22}")
    for risk in [0.0015,0.0025,0.005,0.0075,0.010]:
        gp,gb_,_=montecarlo(Rg,risk,dd=dd)
        pp,pb,pm=montecarlo(Rp,risk,dd=dd)
        rp,rb,rm=montecarlo(Rpr,risk,dd=dd)
        mo  = pm/port_tpm if not np.isnan(pm) else float("nan")
        rmo = rm/port_tpm if not np.isnan(rm) else float("nan")
        print(f"  {risk*100:>6.2f}% │ {gp:>7.1f}{gb_:>7.1f} │ "
              f"{pp:>7.1f}{pb:>7.1f}{mo:>6.1f} │ {rp:>7.1f}{rb:>7.1f}{rmo:>6.1f}")

print(f"\n  Historical sequence (risk 0.5%, static -10% DD):")
hg=historical(gold,0.005); hp=historical(port,0.005)
print(f"    GOLD-only : end ${hg['end']:,.0f}  maxDD {hg['mdd']:.1f}%")
print(f"    PORTFOLIO : end ${hp['end']:,.0f}  maxDD {hp['mdd']:.1f}%  "
      f"{'PASS@'+str(hp['passed']) if hp['passed'] else 'no pass'}")
