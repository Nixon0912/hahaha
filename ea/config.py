"""
APEX-9 — Frozen configuration (parameters locked post-audit)
Do NOT modify thresholds or risk without re-running full validation.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
EA_DIR       = Path(__file__).parent
MODEL_PATH   = EA_DIR / "model.joblib"
LOG_FILE     = EA_DIR / "ea.log"
STATE_FILE   = EA_DIR / "state.json"     # persists daily PnL, trade history

# MT5 on Mac writes/reads files from its sandbox (FILE_COMMON flag in MQL5).
# Python must write signals.json to this exact path so the EA can find it.
# Adjust the profile folder name if yours differs (check ~/Library/Application Support/).
MT5_FILES_DIR = Path.home() / "Library/Application Support/MetaTrader 5/MQL5/Files"
SIGNAL_FILE   = MT5_FILES_DIR / "apex9_signals.json"

# ── Locked strategy parameters (audit-frozen) ─────────────────────────────
ML_THRESHOLD  = 0.35     # P(win) cutoff — DO NOT TUNE
RISK_PCT      = 0.0125   # 1.25% risk per trade — DO NOT TUNE
TP_RR         = 3.5
SL_MULT       = 0.7
SL_LO         = 0.0008   # min SL as fraction of price
SL_HI         = 0.006    # max SL as fraction of price
FORCE_CLOSE_H = 21       # server hour — all positions closed before this

# Range-size acceptance bands (fraction of mid price).
# MUST match the extractor defaults in multi_asset_scan.py exactly:
#   extract_arb: rng_lo=0.0003, rng_hi=0.015
#   extract_nyo: rng_lo=0.0002, rng_hi=0.012
# These are the validated backtest values — the source of truth. ea/signals.py
# imports them so the live path can never drift from the backtest (audit F13).
ARB_RNG_LO    = 0.0003
ARB_RNG_HI    = 0.015
NYO_RNG_LO    = 0.0002
NYO_RNG_HI    = 0.012

# ── Streams ────────────────────────────────────────────────────────────────
STREAMS = [
    # (MT5 symbol,  archetype,  entry_start_h, entry_end_h)
    ("ASXAUD",  "NYO", 13, 15),
    ("DAX40",   "ARB",  8, 10),
    ("ESXEUR",  "NYO", 13, 15),
    ("SP500",   "MOM",  8, 20),
    ("UK100",   "ARB",  8, 10),
    ("USDCAD",  "MOM",  8, 20),
    ("USDJPY",  "ARB",  8, 10),
    ("USDJPY",  "NYO", 13, 15),
    ("XAGUSD",  "ARB",  8, 10),
]

ARB_RANGE_H  = (0,  8)
NYO_RANGE_H  = (10, 13)

# ── Risk guards ────────────────────────────────────────────────────────────
DAILY_STOP_PCT     = 0.03    # halt day at -3% equity
FLOOR_GUARD_BAL    = 9200.0  # halt all trading below $9,200
SPREAD_MAX_MULT    = 1.5     # skip if live spread > 1.5x historical norm
INIT_BALANCE       = 10_000.0
TARGET_BALANCE     = 10_800.0
BUST_FLOOR         = 9_000.0

# ── Circuit breaker (3 independent triggers) ──────────────────────────────
CB_T1_TRADES       = 5       # last N trades
CB_T1_MIN_R        = -3.0    # total R threshold
CB_T2_DRAWDOWN     = -0.04   # -4% from starting balance
CB_T3_TRADES       = 10      # last N trades
CB_T3_MIN_EXPR     = 0.0     # expR threshold
CB_T3_MIN_DAYS     = 10      # minimum calendar days elapsed

# ── Regime gate (validated OOS 2025-01 → 2026-06) ─────────────────────────
# CHOP = cross-asset 60d vol > 18% ann. AND trendiness < 0.30 (lagged 1 day).
# In CHOP, MOM streams are benched: OOS chop expR ARB +2.08, NYO +0.47, MOM -0.74.
REGIME_SYMS         = ["SP500", "DAX40", "USDJPY", "XAGUSD", "USDCAD", "ESXEUR"]
REGIME_LOOKBACK_D   = 60
REGIME_VOL_THRESH   = 18.0    # annualized %, cross-asset mean
REGIME_TREND_THRESH = 0.30    # |60d ret| / 60d vol, cross-asset mean
REGIME_BENCH_ARCHS  = {"MOM"}  # archetypes disabled in CHOP

# ── Inactivity alerts (days since last closed trade) ──────────────────────
INACTIVITY_ALERT_DAYS = [14, 21, 25, 28]

# ── Model training data location ──────────────────────────────────────────
RAW_DATA_DIR = EA_DIR.parent  # root hahaha/ dir with CSV files
