"""
TP Ratio Scan — find the optimal TP_RR for the Top-5 survivor streams.

Current: TP_RR=3.5 → WR=31%, expR=+0.164
Hypothesis: lower TP (1.5–2.5) raises WR toward 50%, lowers variance,
            allows higher risk per trade at same bust %, compressing time.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings, glob
warnings.filterwarnings("ignore")

from multi_asset_scan import (
    load_raw, build_mtf,
    extract_arb, extract_nyo, extract_mom,
    monte_carlo
)

RAW_DIR = Path(__file__).parent

FORCE_H  = 21
SL_MULT  = 0.7
SL_LO    = 0.0008
SL_HI    = 0.006


def resolve_with_tp(m15, t, d, entry, sl, tp_rr):
    """Resolve trade with a custom TP_RR (overrides the module-level constant)."""
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return {"R": 0.0, "result": "skip"}
    tp = entry + d * sl_dist * tp_rr
    day  = m15[m15.index.date == t.date()]
    rem  = day.loc[t:].iloc[1:]
    rem  = rem[rem.index.hour < FORCE_H]
    result = "timeout"
    exit_p = rem.iloc[-1]["close"] if not rem.empty else entry
    for _, rb in rem.iterrows():
        if d == 1:
            if rb["low"]  <= sl: result="sl"; exit_p=sl;  break
            if rb["high"] >= tp: result="tp"; exit_p=tp;  break
        else:
            if rb["high"] >= sl: result="sl"; exit_p=sl;  break
            if rb["low"]  <= tp: result="tp"; exit_p=tp;  break
    if result == "sl":
        R = -1.0
    elif result == "tp":
        R = tp_rr
    else:
        R = float(d * (exit_p - entry) / sl_dist)
    return {"R": R, "result": result}


def extract_stream_custom_tp(m15, mtf, arch, tp_rr):
    """Re-extract a stream with a custom TP ratio."""
    from multi_asset_scan import ranges

    trades = []

    if arch in ("ARB", "NYO"):
        rng_start, rng_end, entry_start, entry_end = {
            "ARB": (0, 8, 8, 10),
            "NYO": (10, 13, 13, 15),
        }[arch]
        rng_table = ranges(m15, rng_start, rng_end)

        for date in sorted(set(m15.index.date)):
            if pd.Timestamp(date).dayofweek >= 4: continue
            dts = pd.Timestamp(date)
            if dts not in rng_table.index: continue
            r = rng_table.loc[dts]
            mid = (r["hi"] + r["lo"]) / 2
            rng_pct = r["rng"] / mid if mid > 0 else 0
            if not (0.0002 <= rng_pct <= 0.015): continue

            eb = m15[(m15.index.date==date) &
                     (m15.index.hour>=entry_start) &
                     (m15.index.hour<entry_end)]
            traded = False
            for t, bar in eb.iterrows():
                if traded or t not in mtf.index: continue
                i = mtf.loc[t]
                if pd.isna(i["h1_atr"]) or i["h1_atr"] <= 0: continue
                d1r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
                if not (0.6 <= d1r <= 2.8): continue
                if not pd.isna(i["h4_adx"]) and float(i["h4_adx"]) < 15: continue

                price = bar["close"]
                sl_d  = float(np.clip(i["h1_atr"]*SL_MULT, price*SL_LO, price*SL_HI))
                long  = price > r["hi"]; short = price < r["lo"]
                if not long and not short: continue
                d = 1 if long else -1
                if i["h1_trend"] * d < 0: continue
                if i["h4_trend"] != 0 and i["h4_trend"] * d < 0: continue

                res = resolve_with_tp(m15, t, d, price, price-d*sl_d, tp_rr)
                trades.append({"date": date, "entry_t": t, "arch": arch,
                               "d": d, **res})
                traded = True

    elif arch == "MOM":
        for date in sorted(set(m15.index.date)):
            if pd.Timestamp(date).dayofweek >= 4: continue
            eb = m15[(m15.index.date==date) &
                     (m15.index.hour>=8) & (m15.index.hour<20)]
            if eb.empty: continue
            prev_above = None; traded = False
            for t, bar in eb.iterrows():
                if traded or t not in mtf.index: continue
                i = mtf.loc[t]
                if pd.isna(i["h4_ema20"]) or int(i["h4_trend"])==0: continue
                if not pd.isna(i["h4_adx"]) and float(i["h4_adx"]) < 20: continue
                d1r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
                if not (0.6 <= d1r <= 2.8): continue
                if int(i["d1_trend"]) * int(i["h4_trend"]) < 0: continue

                h4t   = int(i["h4_trend"])
                price = bar["close"]
                ema20 = float(i["h4_ema20"])
                above = price > ema20
                if prev_above is None:
                    prev_above = above; continue
                d = None
                if h4t==1 and not prev_above and above:    d = 1
                elif h4t==-1 and prev_above and not above: d = -1
                prev_above = above
                if d is None: continue

                h1_atr = float(i["h1_atr"]) if not pd.isna(i["h1_atr"]) else 0
                if h1_atr <= 0: continue
                sl_d = float(np.clip(h1_atr*SL_MULT, price*SL_LO, price*SL_HI))
                res = resolve_with_tp(m15, t, d, price, price-d*sl_d, tp_rr)
                trades.append({"date": date, "entry_t": t, "arch": arch,
                               "d": d, **res})
                traded = True

    return trades


def load_m15_mtf(sym):
    files = sorted(glob.glob(str(RAW_DIR / f"{sym}_M15_*.csv")))
    fpath = max(files, key=lambda f: Path(f).stat().st_size)
    m15   = load_raw(fpath)
    mtf   = build_mtf(m15)
    return m15, mtf


def run():
    TOP5 = [
        ("ASXAUD", "NYO"),
        ("SP500",  "MOM"),
        ("USDCAD", "MOM"),
        ("USDJPY", "NYO"),
        ("XAGUSD", "ARB"),
    ]

    TP_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    print("Loading m15/mtf data for Top-5 streams …")
    data = {}
    for sym, arch in TOP5:
        if sym not in data:
            print(f"  {sym} …")
            data[sym] = load_m15_mtf(sym)

    print(f"\n{'='*85}")
    print(f"  TP Ratio Scan — Top-5 Portfolio")
    print(f"{'='*85}")
    print(f"  {'TP_RR':>6}  {'Trades':>7}  {'WR':>6}  {'expR':>7}  "
          f"{'t/mo':>5}  {'Risk%':>6}  {'Pass%':>6}  {'Bust%':>6}  {'Med.Mo':>7}")
    print(f"  {'-'*83}")

    sweet_spot = None

    for tp_rr in TP_RATIOS:
        all_trades = []
        for sym, arch in TOP5:
            m15, mtf = data[sym]
            t = extract_stream_custom_tp(m15, mtf, arch, tp_rr)
            all_trades.extend(t)

        if not all_trades:
            continue

        all_trades.sort(key=lambda x: x["entry_t"])
        Rs  = np.array([t["R"] for t in all_trades])
        d0  = pd.Timestamp(all_trades[0]["date"])
        d1  = pd.Timestamp(all_trades[-1]["date"])
        tpm = len(all_trades) / max((d1-d0).days/30.44, 0.1)
        wr  = (Rs > 0).mean() * 100
        exp = Rs.mean()

        # Find best risk level for this TP
        best_row = None
        for risk in [0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050,
                     0.0060, 0.0070, 0.0080, 0.0100]:
            mc = monte_carlo(all_trades, risk)
            if mc["bust_pct"] <= 5.0:
                if best_row is None or mc["med_mo"] < best_row[2]:
                    best_row = (risk, mc["pass_pct"], mc["med_mo"],
                                mc["bust_pct"], mc)

        if best_row:
            risk, pp, mm, bp, mc = best_row
            flag = ""
            if bp <= 5.0 and mm <= 3.5:
                flag = "  *** TARGET ***"
                sweet_spot = (tp_rr, risk, mc, all_trades)
            elif bp <= 5.0 and mm <= 5.0:
                flag = "  ** close **"
            print(f"  {tp_rr:>6.1f}  {len(all_trades):>7}  {wr:>5.1f}%  "
                  f"{exp:>+7.3f}  {tpm:>4.1f}  {risk*100:>5.2f}%  "
                  f"{pp:>5.1f}%  {bp:>5.1f}%  {mm:>6.1f}{flag}")
        else:
            print(f"  {tp_rr:>6.1f}  {len(all_trades):>7}  {wr:>5.1f}%  "
                  f"{exp:>+7.3f}  {tpm:>4.1f}  {'—':>6}  {'—':>6}  {'—':>6}  {'—':>7}")

    print(f"  {'='*83}\n")

    # ── Full risk scan for best TP ────────────────────────────────────────
    if sweet_spot:
        tp_rr, best_risk, mc, trades = sweet_spot
        Rs  = np.array([t["R"] for t in trades])
        d0  = pd.Timestamp(trades[0]["date"])
        d1  = pd.Timestamp(trades[-1]["date"])
        tpm = len(trades) / max((d1-d0).days/30.44, 0.1)

        print(f"  *** TARGET HIT at TP={tp_rr} ***")
        print(f"  {len(trades)} trades  expR={Rs.mean():+.3f}  "
              f"WR={(Rs>0).mean()*100:.0f}%  {tpm:.1f} t/mo\n")

        print(f"  Full risk scan at TP={tp_rr}:")
        print(f"  {'Risk%':>6}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}")
        for risk in [0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050,
                     0.0060, 0.0070, 0.0080, 0.0100]:
            mc2 = monte_carlo(trades, risk)
            flag = " ***" if mc2["bust_pct"]<=5.0 and mc2["med_mo"]<=3.5 else ""
            print(f"  {risk*100:>5.2f}%  {mc2['pass_pct']:>6.1f}%  "
                  f"{mc2['bust_pct']:>5.1f}%  {mc2['med_mo']:>6.1f}{flag}")
    else:
        # Show best TP regardless
        print("  No single TP hit target. Best TP=1.5 full scan:")
        all_trades = []
        for sym, arch in TOP5:
            m15, mtf = data[sym]
            all_trades.extend(extract_stream_custom_tp(m15, mtf, arch, 1.5))
        all_trades.sort(key=lambda x: x["entry_t"])
        Rs  = np.array([t["R"] for t in all_trades])
        d0  = pd.Timestamp(all_trades[0]["date"])
        d1  = pd.Timestamp(all_trades[-1]["date"])
        tpm = len(all_trades) / max((d1-d0).days/30.44, 0.1)
        print(f"  {len(all_trades)} trades  expR={Rs.mean():+.3f}  "
              f"WR={(Rs>0).mean()*100:.0f}%  {tpm:.1f} t/mo\n")
        print(f"  {'Risk%':>6}  {'Pass%':>7}  {'Bust%':>6}  {'Med.Mo':>7}")
        for risk in [0.0030, 0.0040, 0.0050, 0.0060, 0.0070, 0.0080, 0.0100]:
            mc = monte_carlo(all_trades, risk)
            flag = " ***" if mc["bust_pct"]<=5.0 and mc["med_mo"]<=3.5 else ""
            print(f"  {risk*100:>5.2f}%  {mc['pass_pct']:>6.1f}%  "
                  f"{mc['bust_pct']:>5.1f}%  {mc['med_mo']:>6.1f}{flag}")


if __name__ == "__main__":
    run()
