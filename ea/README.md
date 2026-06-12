# APEX-9 EA — Deployment Guide

## Architecture

```
Python runner (runner.py)
  ├── Loads M15 data from MT5 or CSV
  ├── Computes 16 features (same as backtest)
  ├── Scores with frozen XGBoost model
  ├── Applies all guards (floor, daily stop, spread, circuit breaker)
  ├── Calculates lot size (audit-compliant, rounds DOWN)
  └── Places orders via MT5 Python package  ← preferred (Windows + MT5)
       OR writes to signals.json            ← fallback (Mac/Linux)
              ↓
      APEX9_EA.mq5 (attached to any MT5 chart)
        └── Polls signals.json every 10s and executes
```

---

## Quick Start (Windows VPS with MT5)

### Step 1 — Install dependencies
```
pip install xgboost lightgbm scikit-learn joblib pandas numpy MetaTrader5
```

### Step 2 — Train and freeze the model (run ONCE)
```
cd C:\apex9
python ea\train_model.py
```
This saves `ea/model.joblib`. Never retrain without full revalidation.

### Step 3 — Start the runner
```
python -m ea.runner
```

The runner connects to MT5 directly. No MQL5 EA needed in this mode.

---

## Signal-File Mode (Mac or if MT5 Python package unavailable)

### Step 1 — Install APEX9_EA.mq5 in MT5
- Copy `APEX9_EA.mq5` to `MT5/MQL5/Experts/`
- Compile in MetaEditor
- Set `SIGNAL_FILE` input to the full path of `ea/signals.json`
- Attach to any chart (e.g. EURUSD M1)

### Step 2 — Run Python in signal-file mode
```
python -m ea.runner --signal-file-mode
```

Python writes signals to `ea/signals.json`. MQL5 EA polls and executes.

---

## Acceptance Tests (must pass before live challenge)

### F10 — Lot sizing unit test
```python
from ea.risk import calculate_lots
# USDJPY: $125 risk, SL 0.100, tick_size=0.001, tick_value≈0.65/lot
lots = calculate_lots("USDJPY", 0.100, 10000,
                      tick_size=0.001, tick_value=0.65,
                      volume_min=0.01, volume_max=100, volume_step=0.01)
actual_risk = lots * (0.100/0.001) * 0.65
assert abs(actual_risk - 125) < 5, f"Risk error: ${actual_risk:.2f}"
print(f"USDJPY: {lots} lots, actual risk ${actual_risk:.2f} ✅")
```
Run the same test for all 8 symbols before deployment.

### F12 — Force-close test
- Open a manual position on demo
- Wait for 20:55 server time
- Verify EA closes it automatically
- Check log for "FORCE CLOSE" message

### F13 — Reconciliation test
- Pick any week from 2025 data
- Run: `python ea/reconcile.py --start 2025-03-01 --end 2025-03-07`
- Verify signals match backtest signals exactly

---

## Daily Operating Procedure

| Time (server) | Check |
|---|---|
| 07:00 | Runner active, connected |
| 08:00–10:00 | ARB entry window |
| 13:00–15:00 | NYO entry window |
| 08:00–20:00 | MOM entry window |
| 20:55 | Force-close begins |
| 21:00 | All positions closed, no overnight exposure |

---

## Circuit Breaker — Manual Resume

If the circuit breaker fires, the system stops trading. Before resuming:
1. Check `ea/state.json` for the trigger reason
2. Review recent trades in the log
3. Manually reset if you're satisfied the regime has normalised:

```python
import json
with open("ea/state.json") as f: state = json.load(f)
state["circuit_breaker_active"] = False
state["circuit_breaker_reason"] = None
with open("ea/state.json", "w") as f: json.dump(state, f, indent=2)
```

Do not reset during a losing streak — that defeats the purpose.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Locked parameters — do not edit |
| `train_model.py` | One-time model training |
| `features.py` | Feature computation (mirrors backtest) |
| `signals.py` | ARB/NYO/MOM entry logic |
| `risk.py` | Lot sizing, guards, circuit breaker |
| `executor.py` | MT5 order placement |
| `runner.py` | Main loop |
| `APEX9_EA.mq5` | MQL5 signal-file executor |
| `model.joblib` | Frozen model (created by train_model.py) |
| `signals.json` | Live signal queue (signal-file mode) |
| `state.json` | Persisted trade history and guard state |
| `ea.log` | Full execution log |
