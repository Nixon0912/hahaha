"""
APEX-9 — Frozen configuration (parameters locked post-audit)
Do NOT modify thresholds or risk without re-running full validation.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
EA_DIR       = Path(__file__).parent
MODEL_PATH   = EA_DIR / "model.joblib"
SIGNAL_FILE  = EA_DIR / "signals.json"   # Python writes, MQL5 reads
LOG_FILE     = EA_DIR / "ea.log"
STATE_FILE   = EA_DIR / "state.json"     # persists daily PnL, trade history

# ── Locked strategy parameters (audit-frozen) ─────────────────────────────
ML_THRESHOLD  = 0.35     # P(win) cutoff — DO NOT TUNE
RISK_PCT      = 0.0125   # 1.25% risk per trade — DO NOT TUNE
TP_RR         = 3.5
SL_MULT       = 0.7
SL_LO         = 0.0008   # min SL as fraction of price
SL_HI         = 0.006    # max SL as fraction of price
FORCE_CLOSE_H = 21       # server hour — all positions closed before this

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

# ── Inactivity alerts (days since last closed trade) ──────────────────────
INACTIVITY_ALERT_DAYS = [14, 21, 25, 28]

# ── Model training data location ──────────────────────────────────────────
RAW_DATA_DIR = EA_DIR.parent  # root hahaha/ dir with CSV files
