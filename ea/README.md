# APEX-9 EA — Mac Deployment Guide

## Architecture (Mac)

```
Mac (Python)                         Mac (MT5 native)
─────────────────────────────────    ──────────────────────────────
runner.py                            APEX9_EA.mq5
  ├── loads M15 data from CSV  →→→   (MT5 bars, same source)
  ├── computes features                
  ├── scores XGBoost model            
  ├── applies all guards              
  ├── calculates lot size             
  └── writes apex9_signals.json  →→→ polls every 10s, executes orders
       │                                  ├── place_order()
       │                                  ├── close_position()
       └─────────────── reads back ←←← └── force_close_all()

Shared file location (MT5 common sandbox):
  ~/Library/Application Support/MetaTrader 5/MQL5/Files/apex9_signals.json
```

---

## One-Time Setup

### 1 — Install Python dependencies
```bash
pip install xgboost scikit-learn pandas numpy joblib
```

### 2 — Train and freeze the model (run once, from the hahaha/ root)
```bash
cd ~/path/to/hahaha
python ea/train_model.py
```
Creates `ea/model.joblib`. This is the frozen production model. Do not retrain without full revalidation.

### 3 — Install MQL5 EA in MT5
- Open MT5 on Mac
- Go to **File → Open Data Folder → MQL5 → Experts**
- Copy `APEX9_EA.mq5` there
- Open MetaEditor (F4), open the file, press **Compile** (F7)
- Attach to any chart (e.g. EURUSD M15) — the chart instrument doesn't matter
- In EA inputs, confirm `SIGNAL_FILENAME = apex9_signals.json`
- Enable **AutoTrading** (the green button in the toolbar)

### 4 — Verify the shared file path
Python writes to:
```
~/Library/Application Support/MetaTrader 5/MQL5/Files/apex9_signals.json
```
Check your MT5 data folder: **File → Open Data Folder** in MT5. The `MQL5/Files/` subfolder must match.

> If your MT5 data folder has a different name (e.g. `MetaTrader 5 - The5ers`), update `MT5_FILES_DIR` in `ea/config.py` to match exactly.

---

## Running the Signal Engine

```bash
# Keep Mac awake (important — Mac sleep kills the process)
caffeinate -i python -m ea.runner
```

The `caffeinate` command prevents Mac from sleeping while the runner is active. Run this in a dedicated Terminal window.

The runner:
- Detects Mac automatically, uses signal-file mode
- Polls every 30 seconds for new M15 bars
- Active hours: 07:00–21:00 server time
- Writes force-close signals at 20:55 / 20:57 / 20:59 / 21:00 (MQL5 EA also has independent force-close)

---

## Acceptance Tests (run before live challenge)

### F10 — Lot sizing check
```python
# Run from hahaha/ root
python -c "
from ea.risk import calculate_lots
# USDJPY example: $125 risk, 10 pip SL
lots = calculate_lots('USDJPY', 0.100, 10000,
    tick_size=0.001, tick_value=0.65,
    volume_min=0.01, volume_max=100, volume_step=0.01)
risk = lots * (0.100/0.001) * 0.65
print(f'USDJPY: {lots} lots → actual risk \${risk:.2f} (target \$125)')
assert abs(risk - 125) < 10
print('PASS ✅')
"
```

### F12 — Force-close test (demo account)
1. Manually open a small position in MT5 demo
2. Wait until 20:55 server time
3. Check MT5 journal — should log `FORCE CLOSE`
4. Verify no open positions remain after 21:00

### F11 — Signal file handshake test
```bash
# Write a test signal and verify EA picks it up on demo
python -c "
import json
from ea.config import SIGNAL_FILE, MT5_FILES_DIR
MT5_FILES_DIR.mkdir(parents=True, exist_ok=True)
test = [{'status':'pending','symbol':'EURUSD','direction':1,
         'lots':0.01,'sl':1.0800,'tp':1.1200,'comment':'TEST'}]
with open(SIGNAL_FILE,'w') as f: json.dump(test, f)
print(f'Test signal written to: {SIGNAL_FILE}')
print('Check MT5 journal for order execution within 10 seconds.')
"
```

---

## 30-Day Demo Run Checklist

Before switching to the live challenge account, run on **demo** for 30 days:

- [ ] Signals generate at expected times (ARB 08–10, NYO 13–15, MOM 08–20)
- [ ] Force-close fires before 21:00 every day (check `ea.log`)
- [ ] No open positions at 22:00 on any day
- [ ] Daily stop guard triggers correctly (test by temporarily setting `DAILY_STOP_PCT=0.001`)
- [ ] Circuit breaker logs activate correctly
- [ ] Lot sizes match intended risk within 5% for each symbol
- [ ] Live spreads stay within `SPREAD_MAX_MULT=1.5×` historical median
- [ ] At least one full trade cycle observed (entry → TP or SL)

---

## Circuit Breaker — Manual Resume

If CB fires, trading stops. Check `ea/state.json` for the reason:
```bash
cat ea/state.json | python -m json.tool | grep -A3 circuit
```

Reset only after confirming regime has normalised:
```python
import json
with open("ea/state.json") as f: s = json.load(f)
s["circuit_breaker_active"] = False
s["circuit_breaker_reason"] = None
with open("ea/state.json","w") as f: json.dump(s, f, indent=2)
```

---

## Daily Procedure (live challenge)

| Server time | Action |
|---|---|
| Before 07:00 | Start `caffeinate -i python -m ea.runner` if not running |
| 08:00–10:00 | ARB entry window — signals may fire |
| 13:00–15:00 | NYO entry window — signals may fire |
| 08:00–20:00 | MOM entry window — signals may fire |
| 20:55 | Force-close begins (Python + EA both trigger) |
| 21:00 | Verify no open positions in MT5 |
| End of day | Check `ea.log` for any warnings or errors |

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Locked parameters — do not edit thresholds or risk |
| `train_model.py` | One-time model training |
| `features.py` | Live feature computation (mirrors backtest) |
| `signals.py` | ARB/NYO/MOM entry logic |
| `risk.py` | Lot sizing, all 6 guards, circuit breaker |
| `executor.py` | Signal file writer |
| `runner.py` | Main loop |
| `APEX9_EA.mq5` | MQL5 EA — polls signals.json, executes orders |
| `model.joblib` | Frozen XGBoost model (created by train_model.py) |
| `state.json` | Trade history, daily guard state, CB state |
| `ea.log` | Full execution log |
| `apex9_signals.json` | Live signal queue (in MT5 Files folder) |
