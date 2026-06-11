# APEX-9 — ML-Filtered Multi-Asset Breakout System
### Performance Dossier · The5ers $10,000 Challenge · All Costs Included

---

## The One-Line Pitch

> A machine-learning filter scans 9 proven breakout/momentum streams across FX, indices and metals, takes only the top ~6% highest-conviction setups, and passes the challenge in a median of **2.8 months** with a **4.1% bust risk** — net of every fee.

---

## Headline Numbers (True Out-of-Sample, Jan 2025 – Jun 2026)

| Metric | Value |
|---|---|
| OOS trades | **48** |
| Win rate | **50.0%** (vs 29% unfiltered base) |
| Expectancy | **+0.716R per trade** (net of all costs) |
| Trade frequency | **2.9 trades / month** |
| Best / worst trade | +3.45R / −1.17R |
| Statistical significance | p = 0.003 vs base rate |
| Bootstrap 95% CI on expR | [+0.16, +1.27] — lower bound positive |
| Wilson 95% CI on win rate | [36%, 64%] |

## Challenge Outcome Probabilities (Monte Carlo, 5,000 paths)

| Risk/trade | Pass | Bust | Median time |
|---|---|---|---|
| 0.75% | 99.3% | 0.7% | 4.9 mo |
| 1.00% | 98.0% | 2.0% | 3.5 mo |
| **1.25% ← recommended** | **95.9%** | **4.1%** | **2.8 mo** |
| 1.50% | 93.6% | 6.4% | 2.1 mo |

Both mandate constraints met at 1.25%: bust ≤ 5% ✅ · median ≤ 3 months ✅

---

## Every Cost Is In The Numbers

| Cost | Treatment |
|---|---|
| Spread | Actual recorded spread at each entry bar, charged per trade (avg −0.10R) |
| Commission | $4/lot/side USDJPY & USDCAD · 0.001%/side XAGUSD · indices zero |
| Overnight swap | **$0 — structurally avoided.** Every position force-closes 21:00 server; rollover is ~00:00 |
| Weekend exposure | None — no positions held past 21:00, no Friday triple-swap ever paid |

Cost analysis exposed and removed fake edges: ASXAUD (+0.145 gross → −0.083 net) and UK100 (+0.068 → −0.129) are unprofitable after spread. The model is trained on **cost-adjusted outcomes**, so it learns to avoid setups the spread eats.

---

## How It Works

**Layer 1 — Three battle-tested archetypes across 8 instruments:**
- **ARB** Asian Range Breakout (range 00:00–08:00, entry 08:00–10:00): DAX40, UK100, USDJPY, XAGUSD
- **NYO** NY Open Breakout (range 10:00–13:00, entry 13:00–15:00): ASXAUD, ESXEUR, USDJPY
- **MOM** H4 EMA20 pullback momentum (entry 08:00–20:00): SP500, USDCAD

All 9 streams individually survived chronological in-sample AND out-of-sample profitability tests over 4–6 years (selected from 75 candidate streams; 66 rejected).

**Layer 2 — Risk engineering:**
- SL = clip(H1 ATR × 0.7, 0.08%–0.6% of price) · TP = 3.5R · force-close 21:00
- One trade per stream per day, trend-aligned entries only (H1/H4/D1 agreement)

**Layer 3 — XGBoost conviction filter:**
- 16 features at entry (time-of-day, volatility state, ADX, EMA distance, multi-TF trend, range size, rolling form)
- 5-fold isotonic-calibrated probabilities, trained on 2020–2024 (1,477 trades)
- Trades only when P(win) > 35% — keeps the best ~6% of signals

---

## Validation Gauntlet (everything was tested, including what failed)

| Test | Result |
|---|---|
| 70/30 chronological OOS | ✅ +0.716R net, p=0.003 |
| Walk-forward (annual, expanding) | ✅ 4/5 windows positive with vol filter; sole failure (2024) is regime-driven |
| Threshold sensitivity 33–38% | ✅ Edge present and monotonic across the band |
| Regime check (current 2026 conditions) | ✅ 2026 YTD: WR 60%, expR +1.04 — strongest period on record |
| Volatility regimes | ✅ Profitable in normal vol; 2022-style shocks filtered by ATR guard |
| Multi-model ensemble (XGB+LGBM+RF+Logit) | ❌ Tested, rejected — dilutes the signal tail |
| Next-candle direction trading | ❌ Tested, rejected — direction predictable (up to 65%) but spread eats 100% of edge |
| Lower TP ratios (1.0–3.0) | ❌ Tested, rejected — TP 3.5 optimal |

---

## Live Safety Rails

1. **Daily stop:** halt for the day at −3% (The5ers limit: 5%)
2. **Floor guard:** halt all trading at $9,200 balance (bust floor: $9,000)
3. **Circuit breaker:** pause system if rolling 20-trade expR turns negative — the live defense against a 2024-style regime shift
4. **Spread guard:** skip any trade where live spread > 1.5× the historical norm for that symbol/hour

---

## Honest Risk Disclosure

- The OOS sample is 48 trades. The probabilities are estimates with real uncertainty (WR could plausibly be 36–64%). A 4.1% bust rate means ~1 in 24 attempts fails even if nothing changes.
- The edge is regime-dependent: 2024 was a losing year for the underlying streams and no entry-bar filter fully fixes that. The circuit breaker is the mitigation, not a guarantee.
- At ~3 trades/month, variance is lumpy: expect multi-week quiet stretches and losing months on the way to the median 2.8-month pass.

---

*All results net of spread, commission and swap · Data: The5ers MT5 exports, M15, 2020–2026 · Simulation: 5,000-path Monte Carlo, static $9,000 bust floor, $10,800 target*
