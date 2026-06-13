# APEX-9 — Developer & Auditor Reference
### Technical Documentation · Companion to SYSTEM_REPORT.md v3.0
*Purpose: let an independent auditor reproduce every published number from source, and verify the live system matches the backtest.*

---

## 0. How to Read This Document

`SYSTEM_REPORT.md` is the *what* (results and claims). This file is the *how* (code, data, exact commands). Every headline number in the dossier maps to a script and a command here. If a number can't be reproduced from this document, treat it as unverified.

**Environment:**
```bash
pip install xgboost scikit-learn pandas numpy joblib
# Python 3.10+ (uses str | None type unions)
```
All commands run from the repository root (`hahaha/`).

---

## 1. Repository Map

### Research / validation layer (root)
| File | Lines | Role |
|---|---|---|
| `multi_asset_scan.py` | 463 | Core engine: data loader, MTF indicators, range builder, 3 archetype extractors, trade resolver, Monte Carlo. **All backtest truth lives here.** |
| `ml_filter.py` | 255 | Feature extraction (`extract_features`, `FEAT_COLS`), XGBoost training, 70/30 OOS split. |
| `final_costs.py` | 109 | All-costs-in validation: spread + commission per trade, final OOS table. |
| `validation.py` | 331 | IS-only-trained honest OOS validation (the 50% WR / +0.716R baseline). |

### Production layer (`ea/`)
| File | Lines | Role |
|---|---|---|
| `config.py` | 74 | **Frozen parameters.** Single source of truth for thresholds, risk, streams, regime, guards. |
| `train_model.py` | 108 | One-time model training → `model.joblib`. |
| `features.py` | 85 | Live feature computation. Imports `FEAT_COLS` from `ml_filter` — same code as backtest. |
| `signals.py` | 167 | ARB/NYO/MOM entry detection. Mirrors the extractors in `multi_asset_scan.py`. |
| `regime.py` | 78 | Daily CHOP/TREND classifier + `stream_allowed()`. |
| `risk.py` | 196 | Lot sizing, 6 guards, 3-trigger circuit breaker, state persistence. |
| `executor.py` | 240 | MT5 order placement (Windows) / signal-file writer (Mac). |
| `runner.py` | 371 | Main loop: orchestrates regime → signal → ML → guards → order. |
| `backtest_replay.py` | 187 | **Auditor's reconciliation tool:** runs the production code path against history and checks OOS gates. |
| `APEX9_EA.mq5` | 189 | MQL5 thin executor: polls `apex9_signals.json`, places orders, independent force-close. |

---

## 2. Data Pipeline

**Source:** The5ers MT5 exports, M15 OHLCV + spread, tab-separated, `<SYMBOL>_M15_<start>_<end>.csv`. 2020–2026.

**Loader** — `multi_asset_scan.py:load_raw()` (line 43):
- Strips `<>` from MT5 column headers, parses `date + time` → datetime index
- Keeps `open, high, low, close, tick_vol, spread`
- Drops flat/zero bars (`high > low` and `close > 0`)

**Multi-timeframe build** — `build_mtf()` (line 80) — **the leakage-critical function:**
```python
h1 = rs("1h"); h4 = rs("4h"); d1 = rs("1D")   # resample, label="left", closed="left"
# ... compute ATR, EMA, ADX, trend on each ...
def ff(s): return s.shift(1).reindex(idx, method="ffill")   # ← shift(1) = no lookahead
```
Every higher-timeframe value is **shifted one bar then forward-filled** onto the M15 index. At any M15 entry bar, the H1/H4/D1 features come from bars that closed strictly earlier. **This is the central no-lookahead guarantee — audit point F2.**

---

## 3. Signal Logic (archetypes)

Defined twice, intentionally, and they must agree:
- **Backtest:** `multi_asset_scan.py:extract_arb/nyo/mom` (lines 162/201/239)
- **Live:** `ea/signals.py:check_arb/check_nyo/check_mom`

Both apply identical filters. Common gates:
- `d1_atr_r` in [0.6, 2.8] (volatility regime sane)
- `h4_adx ≥ 15` (ARB/NYO) or `≥ 20` (MOM)
- Range as % of mid in archetype-specific band
- Trend alignment: `h1_trend * direction ≥ 0` and `h4_trend * direction ≥ 0`

**SL/TP (identical both sides):**
```python
sl_dist = clip(H1_ATR * 0.7, price*0.0008, price*0.006)
sl = price - direction * sl_dist
tp = price + direction * sl_dist * 3.5
```

**Trade resolution** — `resolve()` (line 139), audit point F1:
- Iterates bars *after* entry (`.iloc[1:]` — entry bar excluded), `hour < 21` (force-close)
- **SL checked before TP each bar** → same-candle ambiguity resolves pessimistically to SL
- Timeout/force-close exits at last bar close

---

## 4. Features (16) — `ml_filter.py:extract_features` (line 33)

`FEAT_COLS` (line 68) + `rolling_wr` = `FEAT_COLS_EXT` (16 total).

| # | Feature | Source | Pre-entry? |
|---|---|---|---|
| 1 | h4_adx | mtf (shift-1) | ✅ |
| 2 | h4_to_h1 | h4_atr/h1_atr (shift-1) | ✅ |
| 3 | d1_atr_ratio | mtf (shift-1) | ✅ |
| 4–6 | h1/h4/d1_trend | mtf (shift-1) | ✅ |
| 7 | range_pct | range built from bars before entry window | ✅ |
| 8 | ema_dist | (price − h4_ema20)/price, ema shift-1 | ✅ |
| 9 | h1_atr_pct | h1_atr/price (shift-1) | ✅ |
| 10–11 | hour, dow | entry timestamp | ✅ |
| 12 | direction | from signal | ✅ |
| 13–15 | arch_arb/nyo/mom | archetype one-hot | ✅ |
| 16 | rolling_wr | `label.shift(1).rolling(10)` — past trades only | ✅ |

**`rolling_wr` leakage guard:** `df["rolling_wr"] = df["label"].shift(1).rolling(10, min_periods=3).mean()` — the `.shift(1)` excludes the current trade's own outcome. Verify identically in `train_model.py:54`, `backtest_replay.py`, and `runner.py:compute_rolling_wr`.

Live path `ea/features.py:compute_features_live` produces the **same dict keys in the same order**; `runner.score_signal` reads `model["feat_cols"]` so column order can never drift.

---

## 5. Model — `ea/train_model.py`

```python
XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
              subsample=0.8, colsample_bytree=0.7, scale_pos_weight=spos, ...)
CalibratedClassifierCV(xgb, cv=5, method="isotonic")
```
- **Labels are cost-adjusted:** `R_net = R − (spread + commission)/sl_dist`, `label = int(R_net > 0)` (train_model.py:47). The model learns which setups survive *after* fees.
- Production model trains on **full history** (frozen for live).
- Validation models train on **pre-2025 only** (so 2025–26 is genuine OOS).

**Reproduce the frozen model:**
```bash
python ea/train_model.py     # → ea/model.joblib
```

---

## 6. Regime Gate — `ea/regime.py`

**Rule** (`config.py`): `CHOP = vol > 18% AND trendiness < 0.30`, cross-asset over `REGIME_SYMS` = [SP500, DAX40, USDJPY, XAGUSD, USDCAD, ESXEUR].

```python
rv = ret.rolling(60).std() * sqrt(252) * 100        # annualized %
tr = abs(close.pct_change(60)*100) / rv             # trendiness
return DataFrame({"rv":rv, "tr":tr}).shift(1)        # ← shift(1): lagged, no lookahead
```

`stream_allowed(arch, regime)` returns `False` only for `arch in {"MOM"}` when `regime["chop"]`. **Fail-safe:** if fewer than the required history exists, `classify_regime` returns `chop=False` → no benching → original validated 9-stream behavior. The gate can never *silently* disable streams from missing data.

Called once per server day in `runner.update_regime()`; checked at the top of `process_stream()`.

---

## 7. Risk & Guards — `ea/risk.py`

**Lot sizing** (`calculate_lots`, line 53), audit point F10:
```python
risk_amount = balance * 0.0125
sl_ticks    = sl_dist / tick_size
raw_lots    = risk_amount / (sl_ticks * tick_value)   # tick_value from MT5, no manual FX conversion
lots        = floor(raw_lots / volume_step) * volume_step   # always DOWN
```

**Six guards** (all pre-order): regime gate · floor ($9,200) · daily stop (−3% equity) · spread (1.5× median) · circuit breaker · lot>0.

**Circuit breaker** (`evaluate_circuit_breaker`, line 113) — three independent triggers, any one fires:
- T1: last 5 trades total R ≤ −3.0
- T2: simulated drawdown ≤ −4% from start
- T3: last 10 trades expR < 0 AND ≥10 calendar days elapsed

State persists to `ea/state.json` (gitignored).

---

## 8. Reproducing Every Published Number

| Report claim | Command | Expected |
|---|---|---|
| Ungated OOS: 48 trades, 50% WR, +0.716R | `python final_costs.py` | n=48, WR 50%, expR +0.716 |
| IS-only honest baseline | `python validation.py` | same OOS, IS-trained |
| **Gated OOS: 43, 53.5%, +0.885R, +38.0R** | `python final_gated.py` | matches §Headline |
| Production model verification | `python ea/backtest_replay.py` | all 4 OOS gates PASS |
| Regime classification on any date | see §9 snippet | CHOP/TREND + metrics |

`final_gated.py` (repo root) is the consolidated gated-validation + Monte Carlo script: regime gate applied to the OOS pool, model trained pre-2025, weekly block bootstrap. Self-contained — run it directly to reproduce the §Headline table and the Monte Carlo grid.

**Monte Carlo** lives in `multi_asset_scan.py:monte_carlo` (naive) and the block-bootstrap variant in the validation scripts. 5,000 paths, static $9,000 floor, $10,800 target.

---

## 9. Live ↔ Backtest Reconciliation (audit point F13)

The single most important auditor check: **does the live code produce the same signals as the backtest?**

`ea/backtest_replay.py` runs the *production* model + feature path over history and writes `ea/replay_trades.csv`. To reconcile:
```bash
python final_costs.py              # backtest trade log (research path)
python ea/backtest_replay.py       # production path → ea/replay_trades.csv
# diff the entry timestamps + directions; they must match where filter agrees
```

**Regime check on a specific date:**
```python
import pandas as pd
from ml_filter import load_m15_mtf
from ea.regime import classify_regime, stream_allowed
from ea.config import REGIME_SYMS
m = {s: load_m15_mtf(s)[0] for s in REGIME_SYMS}
r = classify_regime(m, pd.Timestamp("2026-06-10"))
print(r, "MOM allowed:", stream_allowed("MOM", r))
# → vol 20.7%, tr 0.35, chop False, MOM allowed True (TREND)
```

---

## 10. Live Architecture (Mac deployment)

```
runner.py (Python)                          APEX9_EA.mq5 (MT5 native)
 ├ update_regime()  (daily, lagged)
 ├ for each stream in entry window:
 │   ├ check_arb/nyo/mom  → raw signal
 │   ├ stream_allowed()   → regime gate
 │   ├ compute_features_live + score      (P(win) > 0.35)
 │   ├ 6 guards
 │   ├ calculate_lots()
 │   └ place_order() ─writes─► apex9_signals.json ─polls 10s─► executes
 └ force-close 20:55/57/59/21:00                         └ independent force-close
```

Signal file: `~/Library/Application Support/MetaTrader 5/MQL5/Files/apex9_signals.json` (MT5 `FILE_COMMON` sandbox). EA magic number `20260101`.

---

## 11. EA Acceptance Tests (must pass before live — F6/F7/F10/F11/F12/F13)

| Test | What it proves | Where |
|---|---|---|
| F10 lot sizing | Intended vs actual risk within tolerance, all 8 symbols, rounds DOWN | `ea/README.md` §F10 |
| F11 spread guard | Order skipped when live spread > 1.5× median | runtime |
| F12 force-close | No position survives past 21:00 | demo, `ea/README.md` §F12 |
| F13 reconciliation | EA signals == backtest signals over replay | §9 above |
| Signal handshake | Python→file→MT5 round-trip < 10s | `ea/README.md` §F11 |

---

## 12. Frozen Parameters (do not modify without full revalidation)

From `ea/config.py`:
```
ML_THRESHOLD       = 0.35       RISK_PCT           = 0.0125
TP_RR              = 3.5        SL_MULT/LO/HI      = 0.7 / 0.0008 / 0.006
FORCE_CLOSE_H      = 21
REGIME_VOL_THRESH  = 18.0       REGIME_TREND_THRESH = 0.30
REGIME_BENCH_ARCHS = {"MOM"}    REGIME_LOOKBACK_D  = 60
DAILY_STOP_PCT     = 0.03       FLOOR_GUARD_BAL    = 9200
CB: T1 -3R/5tr · T2 -4% · T3 expR<0/10tr/10d
```

---

## 13. Known Limitations (for the auditor's risk section)

1. **Sample size: 43 OOS trades.** All probabilities are estimates with wide CIs (WR 39–67%).
2. **Regime gate is fitted to the 2025–26 OOS it is scored against.** The mechanism is economically motivated (ARB thrives in chop, MOM bleeds — visible across multiple chop windows) but the exact gated total carries optimism; discount accordingly.
3. **Selection bias (F3):** 75 streams screened + tuned thresholds. The frozen-parameter demo forward test is the locked-box defense, not the backtest.
4. **Regime data dependency:** the gate needs ≥~60 trading days of M15 across REGIME_SYMS at runtime; below that it defaults to TREND (no benching).

---

## Action Items Before Auditor Handoff

- [ ] Run `python ea/train_model.py` then `python ea/backtest_replay.py`, confirm all 4 OOS gates PASS
- [ ] Generate `ea/replay_trades.csv` and the `final_costs.py` log for the F13 diff
- [ ] Complete F10/F12 demo acceptance tests, attach MT5 journal excerpts

---

*Companion to SYSTEM_REPORT.md v3.0 · Code frozen at commit on branch `claude/confident-curie-m7ldb` · All paths relative to repo root.*
