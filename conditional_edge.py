"""
Per-asset conditional-edge discovery with out-of-sample validation.

For each instrument:
  1. Chronological split: train = oldest 70%, test = newest 30%.
  2. Search candidate feature filters; rank by train expectancy (with a
     minimum trade count so we don't chase noise).
  3. Report the best filter's OUT-OF-SAMPLE expectancy. An edge only counts
     if it is positive in BOTH train and test (survives OOS).

Honest about overfitting: searching many filters inflates train numbers;
the OOS column is the truth. We report how many assets keep an edge OOS.
"""
import warnings; warnings.filterwarnings("ignore")
import os, glob
import numpy as np, pandas as pd
from assets import CLASS

SETUPS=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","setups")
MIN_TR=25          # minimum trades for a filter to be considered
RISK=0.005

# Instruments excluded: tick/spread mis-calibration (impossible cost). Flag, fix later.
EXCLUDE={"HSIHKD","JPN225","US30","NGCUSD"}

def candidate_filters(df):
    """Yield (name, boolean mask) for interpretable single & paired conditions."""
    yield ("baseline (all)", np.ones(len(df),bool))
    yield ("dir_with_trend",  df["dir_with_trend"]==1)
    yield ("against_trend",   df["dir_with_trend"]==0)
    yield ("stack_aligned",   df["dir"]==df["stack"])
    yield ("atr_expanding",   df["atr_exp"]==1)
    yield ("atr_rank>0.5",    df["atr_rank"]>0.5)
    yield ("atr_rank>0.7",    df["atr_rank"]>0.7)
    yield ("atr_pct>median",  df["atr_pct"]>df["atr_pct"].median())
    yield ("range>median",    df["range_pct"]>df["range_pct"].median())
    yield ("range<median",    df["range_pct"]<df["range_pct"].median())
    yield ("longs_only",      df["dir"]==1)
    yield ("shorts_only",     df["dir"]==-1)
    yield ("early(<9h)",      df["hour"]<9)
    # paired combos with trend alignment
    base=df["dir_with_trend"]==1
    yield ("trend+atr_exp",   base&(df["atr_exp"]==1))
    yield ("trend+atr_rank>.5",base&(df["atr_rank"]>0.5))
    yield ("trend+range>med", base&(df["range_pct"]>df["range_pct"].median()))
    yield ("trend+stack",     base&(df["dir"]==df["stack"]))

def expR(df, mask):
    sub=df[mask]
    if len(sub)<1: return None
    r=sub["R"].values
    return dict(n=len(r), expR=r.mean(), wr=sub["win"].mean()*100,
                pf=(r[r>0].sum()/-r[r<0].sum()) if (r<0).any() else 99.0,
                net=r.sum()*RISK*100)

def analyse(sym):
    df=pd.read_csv(os.path.join(SETUPS,f"{sym}.csv"), parse_dates=["date"]).sort_values("date")
    n=len(df)
    if n<60: return None
    cut=df["date"].quantile(0.70)
    tr=df[df["date"]<=cut].reset_index(drop=True)
    te=df[df["date"]> cut].reset_index(drop=True)
    if len(te)<15: return None

    # evaluate filters on train
    best=None
    for name,mask_tr in candidate_filters(tr):
        m=expR(tr, mask_tr.values if hasattr(mask_tr,"values") else mask_tr)
        if m is None or m["n"]<MIN_TR: continue
        if best is None or m["expR"]>best[1]["expR"]:
            best=(name,m)
    if best is None: return None
    name=best[0]

    # recompute the SAME filter on test (OOS)
    # rebuild mask on test by name
    fmap=dict(candidate_filters(te))
    if name not in fmap: return None
    te_m=expR(te, fmap[name].values if hasattr(fmap[name],"values") else fmap[name])
    full=expR(df, dict(candidate_filters(df))[name].values
              if hasattr(dict(candidate_filters(df))[name],"values")
              else dict(candidate_filters(df))[name])
    return dict(sym=sym, filt=name, tr=best[1], te=te_m, full=full)

files=sorted(glob.glob(os.path.join(SETUPS,"*.csv")))
print(f"Per-asset conditional edge (train=oldest70%, test=newest30%, min {MIN_TR} tr)\n")
print(f"  {'Symbol':<8}{'Best filter (train-selected)':<22}"
      f"{'TRAIN expR/PF':>16}{'OOS expR/PF':>15}{'OOS net':>9}  Survives")
results=[]
for f in files:
    sym=os.path.basename(f)[:-4]
    if sym in EXCLUDE: continue
    r=analyse(sym)
    if r is None: continue
    results.append(r)

# rank by OOS expR
results.sort(key=lambda x:-x["te"]["expR"])
survivors=[]
for r in results:
    tr,te=r["tr"],r["te"]
    surv = (tr["expR"]>0 and te["expR"]>0 and te["n"]>=10 and te["pf"]>=1.05)
    if surv: survivors.append(r)
    flag="✅ OOS" if surv else ("~" if te["expR"]>0 else "❌")
    print(f"  {r['sym']:<8}{r['filt']:<22}"
          f"{tr['expR']:>+7.3f}/{tr['pf']:>4.2f}{te['expR']:>+8.3f}/{te['pf']:>4.2f}"
          f"{te['net']:>+8.1f}%  {flag}")

print(f"\n  {len(survivors)}/{len(results)} assets keep a positive edge OUT-OF-SAMPLE:")
for r in survivors:
    print(f"    ✅ {r['sym']:<8} [{r['filt']}]  "
          f"OOS: {r['te']['n']} tr, WR {r['te']['wr']:.0f}%, PF {r['te']['pf']:.2f}, "
          f"expR {r['te']['expR']:+.3f}")
