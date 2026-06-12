"""
Signal detection — ARB, NYO, MOM archetypes.
Mirrors backtest entry logic exactly.
"""
import numpy as np
import pandas as pd
from multi_asset_scan import ranges as build_ranges, build_mtf
from ea.config import SL_MULT, SL_LO, SL_HI, ARB_RANGE_H, NYO_RANGE_H


def _sl_dist(m15: pd.DataFrame, mtf: pd.DataFrame,
             entry_t: pd.Timestamp, price: float) -> float:
    i = mtf.loc[entry_t]
    h1_atr = float(i["h1_atr"]) if not pd.isna(i["h1_atr"]) else 0.0
    return float(np.clip(h1_atr * SL_MULT, price * SL_LO, price * SL_HI))


def check_arb(m15: pd.DataFrame, entry_t: pd.Timestamp) -> dict | None:
    """
    Asian Range Breakout check at a given M15 bar.
    Returns signal dict or None.
    """
    mtf = build_mtf(m15)
    if entry_t not in mtf.index:
        return None

    i = mtf.loc[entry_t]
    if pd.isna(i["h1_atr"]) or i["h1_atr"] <= 0:
        return None

    d1r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
    if not (0.6 <= d1r <= 2.8):
        return None
    if not pd.isna(i["h4_adx"]) and float(i["h4_adx"]) < 15:
        return None

    rng_table = build_ranges(m15, ARB_RANGE_H[0], ARB_RANGE_H[1])
    dts = pd.Timestamp(entry_t.date())
    if dts not in rng_table.index:
        return None

    r = rng_table.loc[dts]
    mid = (r["hi"] + r["lo"]) / 2
    rng_pct = r["rng"] / mid if mid > 0 else 0
    if not (0.0002 <= rng_pct <= 0.015):
        return None

    price = float(m15.loc[entry_t, "close"])
    long  = price > r["hi"]
    short = price < r["lo"]
    if not long and not short:
        return None

    d = 1 if long else -1
    if i["h1_trend"] * d < 0:
        return None
    if i["h4_trend"] != 0 and i["h4_trend"] * d < 0:
        return None

    sl_d = _sl_dist(m15, mtf, entry_t, price)
    return {
        "arch": "ARB", "direction": d,
        "entry": price, "sl": price - d * sl_d, "tp": price + d * sl_d * 3.5,
        "sl_dist": sl_d,
    }


def check_nyo(m15: pd.DataFrame, entry_t: pd.Timestamp) -> dict | None:
    """NY Open Breakout check."""
    mtf = build_mtf(m15)
    if entry_t not in mtf.index:
        return None

    i = mtf.loc[entry_t]
    if pd.isna(i["h1_atr"]) or i["h1_atr"] <= 0:
        return None

    d1r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
    if not (0.6 <= d1r <= 2.8):
        return None
    if not pd.isna(i["h4_adx"]) and float(i["h4_adx"]) < 15:
        return None

    rng_table = build_ranges(m15, NYO_RANGE_H[0], NYO_RANGE_H[1])
    dts = pd.Timestamp(entry_t.date())
    if dts not in rng_table.index:
        return None

    r = rng_table.loc[dts]
    mid = (r["hi"] + r["lo"]) / 2
    rng_pct = r["rng"] / mid if mid > 0 else 0
    if not (0.0002 <= rng_pct <= 0.015):
        return None

    price = float(m15.loc[entry_t, "close"])
    long  = price > r["hi"]
    short = price < r["lo"]
    if not long and not short:
        return None

    d = 1 if long else -1
    if i["h1_trend"] * d < 0:
        return None
    if i["h4_trend"] != 0 and i["h4_trend"] * d < 0:
        return None

    sl_d = _sl_dist(m15, mtf, entry_t, price)
    return {
        "arch": "NYO", "direction": d,
        "entry": price, "sl": price - d * sl_d, "tp": price + d * sl_d * 3.5,
        "sl_dist": sl_d,
    }


def check_mom(m15: pd.DataFrame, entry_t: pd.Timestamp,
              prev_above_ema: bool | None) -> tuple[dict | None, bool | None]:
    """
    H4 EMA20 Pullback Momentum check.
    Returns (signal_or_None, new_prev_above_ema).
    Caller must maintain prev_above_ema state across bars.
    """
    mtf = build_mtf(m15)
    if entry_t not in mtf.index:
        return None, prev_above_ema

    i = mtf.loc[entry_t]
    if pd.isna(i["h4_ema20"]) or int(i["h4_trend"]) == 0:
        return None, prev_above_ema
    if not pd.isna(i["h4_adx"]) and float(i["h4_adx"]) < 20:
        return None, prev_above_ema

    d1r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
    if not (0.6 <= d1r <= 2.8):
        return None, prev_above_ema
    if int(i["d1_trend"]) * int(i["h4_trend"]) < 0:
        return None, prev_above_ema

    h4t   = int(i["h4_trend"])
    price = float(m15.loc[entry_t, "close"])
    ema20 = float(i["h4_ema20"])
    above = price > ema20

    if prev_above_ema is None:
        return None, above

    d = None
    if h4t == 1  and not prev_above_ema and above: d = 1
    elif h4t == -1 and prev_above_ema and not above: d = -1

    new_prev = above

    if d is None:
        return None, new_prev

    h1_atr = float(i["h1_atr"]) if not pd.isna(i["h1_atr"]) else 0.0
    if h1_atr <= 0:
        return None, new_prev

    sl_d = _sl_dist(m15, mtf, entry_t, price)
    return ({
        "arch": "MOM", "direction": d,
        "entry": price, "sl": price - d * sl_d, "tp": price + d * sl_d * 3.5,
        "sl_dist": sl_d,
    }, new_prev)
