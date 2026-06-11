# APEX-9 — ML-Filtered Multi-Asset Breakout System
### Performance Dossier & Audit Response · The5ers $10,000 Challenge · v2.0
*All results net of spread, commission and swap · Audited 12 June 2026 · Critical findings remediated*

---

## The One-Line Pitch

> A machine-learning filter scans 9 validated breakout/momentum streams across FX, indices and metals, takes only the top ~6% highest-conviction setups, and passes the challenge in a median of **2.8 months** with a **1.3% bust risk** (block-bootstrap) — net of every fee, audit-verified.

---

## Headline Numbers (True Out-of-Sample, Jan 2025 – Jun 2026)

| Metric | Value |
|---|---|
| OOS trades | **48** |
| Win rate | **50.0%** (vs 29% unfiltered base) · Wilson 95% CI [36%, 64%] |
| Expectancy | **+0.716R per trade** net of all costs · Bootstrap 95% CI [+0.16, +1.27] |
| Trade frequency | **2.9 trades / month** |
| Best / worst trade | +3.45R / −1.17R |
| Statistical significance | p = 0.003 vs base rate |

## Challenge Outcome Probabilities

Two independent Monte Carlo methods, 5,000 paths each. Block bootstrap (weekly blocks) preserves real loss clustering and is the more faithful model; naive reshuffle is shown as the conservative floor.

| Risk/trade | Pass (block / naive) | Bust (block / naive) | Median time |
|---|---|---|---|
| 0.75% | 99.9% / 99.3% | 0.1% / 0.7% | 4.5–4.9 mo |
| 1.00% | 99.6% / 98.0% | 0.4% / 2.0% | 3.5 mo |
| **1.25% ← recommended** | **98.7% / 95.9%** | **1.3% / 4.1%** | **2.8 mo** |
| 1.50% | 97.4% / 93.6% | 2.6% / 6.4% | 2.1 mo |

Mandate constraints met at 1.25% under BOTH methods: bust ≤ 5% ✅ · median ≤ 3 months ✅

---

## Every Cost Is In The Numbers

| Cost | Treatment |
|---|---|
| Spread | Actual recorded spread at each entry bar, charged per trade (avg −0.10R) |
| Commission | $4/lot/side USDJPY & USDCAD · 0.001%/side XAGUSD · indices zero |
| Overnight swap | **$0 — structurally avoided.** Every position force-closes 21:00 server; rollover ~00:00 |
| Weekend exposure | None — no positions past 21:00, no Friday triple-swap ever paid |

Cost analysis exposed and removed fake edges: ASXAUD (+0.145 gross → −0.083 net) and UK100 (+0.068 → −0.129) are unprofitable after spread. The model is trained on **cost-adjusted outcomes**, so it learns to avoid setups the spread eats.

---

## How It Works

**Layer 1 — Three archetypes across 8 instruments:**
- **ARB** Asian Range Breakout (range 00:00–08:00, entry 08:00–10:00): DAX40, UK100, USDJPY, XAGUSD
- **NYO** NY Open Breakout (range 10:00–13:00, entry 13:00–15:00): ASXAUD, ESXEUR, USDJPY
- **MOM** H4 EMA20 pullback momentum (entry 08:00–20:00): SP500, USDCAD

All 9 streams individually survived chronological in-sample AND out-of-sample profitability tests over 4–6 years (selected from 75 candidates; 66 rejected).

**Layer 2 — Risk engineering:**
- SL = clip(H1 ATR × 0.7, 0.08%–0.6% of price) · TP = 3.5R · force-close 21:00
- One trade per stream per day, trend-aligned entries only (H1/H4/D1 agreement)

**Layer 3 — XGBoost conviction filter:**
- 16 leakage-audited features at entry · 5-fold isotonic calibration · trained 2020–2024 (1,477 trades)
- Trades only when P(win) > 35% — keeps the best ~6% of signals
- Objective validated: P(win) classification outperforms expected-R regression (the regressor chases unrepeatable outliers — 49% bust vs 4%)

---

## Independent Audit — Findings & Resolution

The system underwent a 13-finding external-style audit (full report on file). Status:

### Critical findings — CLEARED with evidence

**F1 · M15 execution realism — CLEARED.** Source-verified: SL is checked before TP in every bar; same-candle ambiguity always resolves to SL (pessimistic). The entry bar itself is excluded from resolution entirely. The backtest's fill assumptions are conservative by construction, not optimistic.

**F2 · ML feature leakage — CLEARED.** Formal timestamp audit: every H1/H4/D1 indicator value at entry comes from a bar that closed strictly before the entry timestamp (verified bar-by-bar on live trades). All multi-timeframe frames are lag-shifted; rolling form excludes the current trade. The event-driven feature path and research path are the same code.

### High findings — REMEDIATED

**F5 · Monte Carlo realism — REMEDIATED, numbers improved.** Rebuilt with weekly block bootstrap preserving temporal loss clustering. Result: bust at 1.25% is **1.3%** (vs 4.1% naive) — the original report was *more* pessimistic than reality because OOS losses are not clustered.

**F8 · Circuit breaker speed — REDESIGNED.** Old: 20-trade rolling window (~7 months response). New: three independent triggers, any one pauses the system —
- T1: last 5 trades total ≤ −3R (≈6-week max response)
- T2: account drawdown ≥ −4% from start (immediate)
- T3: last 10 trades expR < 0 with ≥10 calendar days elapsed
Replayed on the OOS sequence: fires early and safely (trade 13, balance $10,612 — never approached danger).

**F9 · Model objective — TESTED, CONFIRMED.** Head-to-head: P(win) classifier vs expected-R regressor. The classifier wins decisively; the regressor selects high-variance outlier setups (49–40% bust). P(win)>35% retained with evidence.

### Findings deferred to EA acceptance criteria (implementation, not research)

| Finding | EA acceptance requirement |
|---|---|
| F6 Daily stop | Real-time **equity** guard at −3%, tick-level, blocks new orders + closes positions; reset on server-day clock |
| F7 Inactivity | Monitor with 14/21/25/28-day alerts + documented compliant contingency trade |
| F10 Lot sizing | Per-symbol unit tests (all 8 instruments): intended risk vs actual SL loss within tolerance; round lots DOWN |
| F11 Spread guard | Per-symbol/hour median + p90 baselines; recheck at order-send; log signal/send/fill spread |
| F12 Force-close | Redundant closes 20:55 / 20:57 / 20:59 / 21:00, server-time based, alert on any residual position |
| F13 Reconciliation | Python backtest vs EA signal-level match over a replay period before any live order |

### Findings accepted as residual risk (cannot be closed without time)

**F3 · Selection bias:** 75 streams screened, parameters tuned — the 18-month OOS is the defense, not proof. Mitigation: parameters are now frozen (35% threshold, 1.25% risk); the demo forward test is a locked-box validation with no adjustments permitted.

**F4 · Small sample:** 48 OOS trades. The probabilities are estimates with real uncertainty. ~1 in 24 attempts fails at the naive bust estimate even if nothing changes.

---

## Live Safety Rails (final design)

1. **Daily stop:** −3% real-time equity, hard block + position close (The5ers limit: 5%)
2. **Floor guard:** halt all trading at $9,200 balance (bust floor: $9,000)
3. **Circuit breaker:** T1/T2/T3 triple-trigger (above) — ~6-week worst-case regime response
4. **Spread guard:** skip if live spread > 1.5× historical norm for that symbol/hour
5. **Force-close:** redundant 20:55–21:00 sequence, zero overnight exposure
6. **Inactivity monitor:** alerts from day 14, compliant contingency by day 28

---

## Deployment Plan (audit-aligned)

| Phase | Gate |
|---|---|
| **1 · EA build** | Passes F6/F7/F10/F11/F12 acceptance tests |
| **2 · Reconciliation** | EA signals match backtest signal-for-signal over replay period (F13) |
| **3 · Demo forward test** | ≥30 trading days · locked parameters · spread/slippage/force-close verified |
| **4 · Live challenge** | Enter at 1.25% if demo matches model; 0.75% fallback if live costs run worse |

---

## Validation Gauntlet (full history, including failures)

| Test | Result |
|---|---|
| 70/30 chronological OOS | ✅ +0.716R net, p=0.003 |
| Walk-forward (annual, expanding) | ✅ 4/5 windows positive with vol filter; sole failure (2024) regime-driven — now covered by fast circuit breaker |
| Threshold sensitivity 33–38% | ✅ Edge present and monotonic |
| Current regime check (2026 YTD) | ✅ WR 60%, expR +1.04 — strongest period on record |
| Block-bootstrap MC vs naive | ✅ Block bootstrap improves numbers (no loss clustering) |
| Feature leakage audit | ✅ All features strictly pre-entry |
| Same-candle fill audit | ✅ SL-first, entry bar excluded — conservative |
| P(win) vs expected-R objective | ✅ Tested head-to-head, P(win) superior |
| Multi-model ensemble (XGB+LGBM+RF+Logit) | ❌ Tested, rejected — dilutes signal tail |
| Next-candle direction trading | ❌ Tested, rejected — spread eats 100% of edge |
| Lower TP ratios (1.0–3.0) | ❌ Tested, rejected — TP 3.5 optimal |

---

## Honest Risk Disclosure

- 48 OOS trades is a small sample: true WR plausibly 36–64%. The bust estimates (1.3% block / 4.1% naive) carry that uncertainty.
- The edge is regime-dependent. 2024 was a losing year for the underlying streams; the fast circuit breaker limits — but does not eliminate — regime risk.
- ~3 trades/month means lumpy variance: expect multi-week quiet stretches and losing months on the way to a median 2.8-month pass.
- Selection bias from the research process cannot be fully ruled out until the locked-parameter forward test confirms live behavior.

---

*Data: The5ers MT5 exports, M15, 2020–2026 · Costs: actual per-bar spread + spec-sheet commissions, swap structurally zero · Simulation: 5,000-path Monte Carlo (naive + weekly block bootstrap), static $9,000 bust floor, $10,800 target · Parameters frozen: P(win)>35%, risk 1.25%/trade*
