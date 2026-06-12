"""
Feature computation for live signals.
Mirrors the backtest feature pipeline exactly — same code, same lag logic.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml_filter import FEAT_COLS
from multi_asset_scan import build_mtf, ranges as build_ranges


FEAT_COLS_EXT = FEAT_COLS + ["rolling_wr"]


def compute_features_live(m15: pd.DataFrame, sym: str, arch: str,
                           entry_t: pd.Timestamp, direction: int,
                           rolling_wr: float) -> dict:
    """
    Compute the 16+1 features for a candidate entry bar.
    Identical to extract_features() in ml_filter.py — verified no leakage.

    Args:
        m15:        M15 OHLCV dataframe indexed by datetime (server time)
        sym:        instrument symbol
        arch:       "ARB", "NYO", or "MOM"
        entry_t:    entry bar timestamp (must already exist in m15.index)
        direction:  1 for long, -1 for short
        rolling_wr: rolling win rate of last 10 closed trades (pre-computed)

    Returns:
        dict of feature values keyed by FEAT_COLS_EXT
    """
    mtf = build_mtf(m15)

    if entry_t not in mtf.index:
        return None

    i = mtf.loc[entry_t]

    h4_adx   = float(i["h4_adx"])   if not pd.isna(i["h4_adx"])   else 0.0
    d1_atr_r = float(i["d1_atr_r"]) if not pd.isna(i["d1_atr_r"]) else 1.0
    h1_trend = int(i["h1_trend"])
    h4_trend = int(i["h4_trend"])
    d1_trend = int(i["d1_trend"])
    h4_ema20 = float(i["h4_ema20"]) if not pd.isna(i["h4_ema20"]) else 0.0
    h1_atr   = float(i["h1_atr"])   if not pd.isna(i["h1_atr"])   else 0.0
    h4_atr   = float(i["h4_atr"])   if not pd.isna(i["h4_atr"])   else 0.0
    price    = float(m15.loc[entry_t, "close"])

    ar_ranges  = build_ranges(m15, 0, 8)
    nyo_ranges = build_ranges(m15, 10, 13)

    range_pct = 0.0
    dts = pd.Timestamp(entry_t.date())
    if arch == "ARB" and dts in ar_ranges.index:
        r = ar_ranges.loc[dts]; mid = (r["hi"] + r["lo"]) / 2
        range_pct = float(r["rng"] / mid) if mid > 0 else 0.0
    elif arch == "NYO" and dts in nyo_ranges.index:
        r = nyo_ranges.loc[dts]; mid = (r["hi"] + r["lo"]) / 2
        range_pct = float(r["rng"] / mid) if mid > 0 else 0.0

    ema_dist   = float((price - h4_ema20) / price * 100) if price > 0 else 0.0
    h1_atr_pct = float(h1_atr / price * 100) if price > 0 else 0.0
    h4_to_h1   = float(h4_atr / h1_atr)      if h1_atr > 0 else 1.0

    return {
        "h4_adx":      h4_adx,
        "h4_to_h1":    h4_to_h1,
        "d1_atr_ratio": d1_atr_r,
        "h1_trend":    h1_trend,
        "h4_trend":    h4_trend,
        "d1_trend":    d1_trend,
        "range_pct":   range_pct * 100,
        "ema_dist":    ema_dist,
        "h1_atr_pct":  h1_atr_pct,
        "hour":        entry_t.hour,
        "dow":         entry_t.dayofweek,
        "direction":   direction,
        "arch_arb":    int(arch == "ARB"),
        "arch_nyo":    int(arch == "NYO"),
        "arch_mom":    int(arch == "MOM"),
        "rolling_wr":  rolling_wr,
    }
