"""
Run the faithful engine across all 25 instruments.
Caches every setup table to data/setups/ for the stats/ML phase.
Reports per-instrument expectancy (full history + recent 12 months).
"""
import warnings; warnings.filterwarnings("ignore")
import os, time
import numpy as np, pandas as pd
from multi_asset_engine import extract_setups, summarise, find_files
from assets import CLASS

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","setups")
os.makedirs(OUT, exist_ok=True)

files=find_files()
print(f"Extracting setups for {len(files)} instruments …\n")
print(f"  {'Symbol':<9}{'Class':<7}{'N':>5}{'WR%':>6}{'PF':>6}{'ExpR':>7}{'Net%':>8}"
      f"{'Shrp':>6}   {'recentPF':>9}{'recentNet':>10}  Edge")

rows=[]
all_setups={}
t0=time.time()
for sym,path in files.items():
    try:
        s=extract_setups(sym,path)
    except Exception as e:
        print(f"  {sym:<9} ERROR {e}"); continue
    if len(s)==0: continue
    s.to_csv(os.path.join(OUT,f"{sym}.csv"),index=False)
    all_setups[sym]=s
    m=summarise(s)
    recent=s[s["date"]>="2025-06-01"]
    rm=summarise(recent) if len(recent)>=10 else None
    rpf = rm["pf"] if rm else float("nan")
    rnet= rm["net"] if rm else float("nan")
    # edge = positive expectancy on FULL history AND recent
    edge = "✅" if (m["pf"]>=1.05 and m["expR"]>0 and (rm and rm["expR"]>0)) else \
           ("~" if m["pf"]>=1.0 else "❌")
    rows.append((sym,m,rm))
    print(f"  {sym:<9}{CLASS.get(sym,'?'):<7}{m['n']:>5}{m['wr']:>6.1f}{m['pf']:>6.2f}"
          f"{m['expR']:>+7.3f}{m['net']:>+7.1f}%{m['sharpe']:>6.2f}   "
          f"{rpf:>9.2f}{rnet:>+9.1f}%  {edge}")

print(f"\n  ({time.time()-t0:.0f}s)  setups cached to data/setups/")

# Rank by recent expectancy
print(f"\n{'='*70}")
print("  RANKED by recent (2025-H2+) profit factor:")
ranked=sorted([r for r in rows if r[2]], key=lambda x:-x[2]["pf"])
for sym,m,rm in ranked[:12]:
    flag="✅" if (m["expR"]>0 and rm["expR"]>0 and m["pf"]>=1.05) else "  "
    print(f"   {flag} {sym:<8} {CLASS.get(sym,'?'):<7} "
          f"recent PF {rm['pf']:.2f}  expR {rm['expR']:+.3f}  net {rm['net']:+.1f}%  "
          f"(full PF {m['pf']:.2f})")

edged=[r[0] for r in rows if r[1]["pf"]>=1.05 and r[1]["expR"]>0 and r[2] and r[2]["expR"]>0]
print(f"\n  Instruments with edge (full PF>=1.05 & expR>0 & recent expR>0): "
      f"{len(edged)}/{len(rows)}")
print(f"  {', '.join(edged) if edged else '(none)'}")
