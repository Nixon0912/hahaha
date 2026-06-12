"""
Regime detection — classifies market as CHOP or TREND each day.

Rule (validated OOS 2025-01 → 2026-06):
  CHOP = cross-asset 60d realized vol > 18% (annualized)
         AND cross-asset trendiness < 0.30
  where trendiness = |60d return| / 60d realized vol.

In CHOP regime, MOM streams are benched (ARB and NYO keep trading):
  OOS chop expR by archetype: ARB +2.08, NYO +0.47, MOM -0.74

All metrics are lagged 1 day — only information available at the start of
the trading day is used. No lookahead.
"""
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ea.config import (REGIME_SYMS, REGIME_VOL_THRESH, REGIME_TREND_THRESH,
                       REGIME_LOOKBACK_D, REGIME_BENCH_ARCHS)

log = logging.getLogger("apex9.regime")


def daily_metrics(m15: pd.DataFrame) -> pd.DataFrame:
    """60d realized vol (ann. %) and trendiness from M15 data, lagged 1 day."""
    d1 = m15.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    ret = d1["close"].pct_change()
    rv = ret.rolling(REGIME_LOOKBACK_D).std() * np.sqrt(252) * 100
    tr = (d1["close"].pct_change(REGIME_LOOKBACK_D) * 100).abs() / rv.replace(0, np.nan)
    # shift(1): today's regime decision uses data through yesterday's close
    return pd.DataFrame({"rv": rv, "tr": tr}).shift(1)


def classify_regime(m15_by_sym: dict[str, pd.DataFrame],
                    on_date: pd.Timestamp) -> dict:
    """
    Classify the regime for a given date from a dict of {symbol: m15 df}.
    Uses only REGIME_SYMS that are present. Returns dict with rv, tr, chop.
    """
    rvs, trs = [], []
    for sym in REGIME_SYMS:
        m15 = m15_by_sym.get(sym)
        if m15 is None or len(m15) < REGIME_LOOKBACK_D * 30:
            continue
        met = daily_metrics(m15)
        # last row at or before on_date
        met = met[met.index <= on_date]
        if met.empty:
            continue
        row = met.iloc[-1]
        if not pd.isna(row["rv"]):
            rvs.append(float(row["rv"]))
        if not pd.isna(row["tr"]):
            trs.append(float(row["tr"]))

    if not rvs or not trs:
        # Insufficient data — default to TREND (no benching) so the system
        # degrades to the original validated behavior rather than silently
        # disabling streams.
        return {"rv": np.nan, "tr": np.nan, "chop": False, "n_syms": 0}

    rv = float(np.mean(rvs))
    tr = float(np.mean(trs))
    chop = (rv > REGIME_VOL_THRESH) and (tr < REGIME_TREND_THRESH)
    return {"rv": rv, "tr": tr, "chop": chop, "n_syms": len(rvs)}


def stream_allowed(arch: str, regime: dict) -> bool:
    """True if this archetype may trade in the current regime."""
    if regime.get("chop") and arch in REGIME_BENCH_ARCHS:
        return False
    return True
