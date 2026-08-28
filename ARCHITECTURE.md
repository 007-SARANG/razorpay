# Architecture

This is the "architecture" deliverable for Razorpay AI Buildathon Track 04. It explains
what Trikon does, why each component is built the way it is, and — where a decision could
reasonably have gone the other way — what the alternative was and why it lost.

---

## 1. The loop being closed

**Period-end settlement close for a Razorpay merchant**, as four questions:

| # | Question | Tier | Mechanism |
|---|---|---|---|
| Q1 | Did every paid order reach the gateway? | order ↔ payment row | blocking → scoring → optimal assignment |
| Q2 | Does each settlement equal its members, net of fee and GST? | settlement ↔ Σ(rows) | pure arithmetic; no matching needed |
| Q3 | Did each settlement land in the bank? | settlement ↔ bank credit | assignment + **bidirectional subset-sum** |
| Q4 | What is still outstanding? | unsettled / on-hold / rolled over | arithmetic over known-unsettled rows |

Q1–Q3 are "the books". Q4 is "the cash position". The track title names both.

One loop, closed end to end — not four loops half-built. Where the other example directions
(settlement Q&A, forward forecaster, tax-line matcher) genuinely belong, they appear as
**narrow slices of this loop** rather than as separate features:

- **Tax-line matching** *is* Q2: fee and 18% GST are recomputed per row and asserted against
  what the gateway reported.
- **Forward cash** *is* Q4, as arithmetic over unsettled rows rather than a statistical
  forecast. On synthetic data a forecast model would be predicting a process we wrote —
  circular and unfalsifiable. Summing what is demonstrably still in flight is checkable
  line by line.
- **Q&A** is deliberately *not* built. It would have been the weakest possible use of the
  remaining time: a commodity text-to-SQL layer, graded by an LLM judge, on a track whose
  bar is hard measurement.

## 2. Data flow

```
  ┌─────────────┐   ┌──────────────────────┐   ┌──────────────┐
  │ order       │   │ settlement recon rows│   │ bank credits │
  │ ledger      │   │ + settlement headers │   │ (narration)  │
  │ (merchant)  │   │ (Razorpay schema)    │   │              │
  └──────┬──────┘   └───────────┬──────────┘   └──────┬───────┘
         │                      │                     │
         ▼                      ▼                     ▼
  ╔══════════════════════════════════════════════════════════════╗
  ║ normalize.py    integer paise · IST working-day index ·       ║
  ║                 canonical reference forms (strict + loose)    ║
  ╚══════════════════════════════════════════════════════════════╝
         │  all three sources now share ONE comparable shape
         ▼
  ╔══════════════════════════════════════════════════════════════╗
  ║ block.py        inverted index on reference | amount |        ║
  ║                 fee-adjusted amount | (conditional) date      ║
  ║                 → 99.98% of the pair space never scored       ║
  ╚══════════════════════════════════════════════════════════════╝
         ▼
  ╔══════════════════════════════════════════════════════════════╗
  ║ score.py        pairwise feature vector → provisional rule    ║
  ║                 + a reviewable evidence chain                 ║
  ╚══════════════════════════════════════════════════════════════╝
         ▼
  ╔══════════════════════════════════════════════════════════════╗
  ║ assign.py       Hungarian per connected cluster               ║
  ║                 + subset-sum, merge AND split directions      ║
  ║                 + tie detection → escalate, never guess       ║
  ╚══════════════════════════════════════════════════════════════╝
         │                                    │
    auto-accepted (≥0.90)          ambiguous residue (R6/R7)
         │                                    │
         │                                    ▼
         │                     ╔════════════════════════════════╗
         │                     ║ llm/adjudicate.py  (OPTIONAL)  ║
         │                     ║ clamped below auto-accept;     ║
         │                     ║ failure ⇒ escalate             ║
         │                     ╚════════════════════════════════╝
         ▼                                    ▼
  ╔══════════════════════════════════════════════════════════════╗
  ║ classify.py     tier-2 arithmetic · presence · duplicates ·   ║
  ║                 lifecycle · ₹ at risk · review-case grouping  ║
  ╚══════════════════════════════════════════════════════════════╝
         ▼
  ╔══════════════════════════════════════════════════════════════╗
  ║ evaluate.py     vs ground truth (loaded ONLY here):           ║
  ║                 P / R / F1 · resolvable recall · calibration  ║
  ║                 · ECE · throughput · exception detection      ║
  ╚══════════════════════════════════════════════════════════════╝
         ▼
  report.py ──► CLI report · JSON · FastAPI ──► dashboard
```

## 3. Why a fixed DAG and not an agent loop

Trikon is an orchestrated controller with deterministic tools, plus a narrowly scoped model
call. It is not an LLM planning its own steps, and that is a decision rather than a
shortcut.

The stage order above is **forced by data dependencies**, not by judgement: tier-2
arithmetic needs settlement membership; presence exceptions need to know what matching left
over; the cash position needs settled flags. A model asked to sequence these would either
reproduce this order or get it wrong — and its choice would vary between runs.

**Reproducibility is the product.** A reconciliation whose output depends on a sampling
temperature cannot be re-run by an auditor, cannot be regression-tested, and cannot support
the metrics table in the README. So orchestration is code, and the model is confined to the
one sub-problem where language judgement genuinely helps.

*What was given up:* a more impressive-sounding "multi-agent system". Seven agents that are
each one LLM call would demo well and defend badly. One honest orchestrator that can explain
every decision defends well under questioning, which is what this submission is actually for.

## 4. Component decisions

### 4.1 Money — `money.py`

**Integer paise everywhere. No floats, ever.**

Reconciliation *is* the assertion that two independently-computed sums are equal. Floating
point makes that assertion unsound (`0.1 + 0.2 != 0.3`), so a float-based reconciler emits
phantom one-paise breaks that a human then triages. Integers make equality exact and every
reported variance real.

Rounding is explicit half-up in one function. Python's `round()` is half-to-even, which
would disagree with published fee tables on exactly the boundary cases a large batch
surfaces.

### 4.2 Calendar — `calendar_ist.py`

T+2 **working** days, over a calendar that closes on Sundays, listed holidays, and the
**2nd/4th Saturday** (RBI convention — the 1st, 3rd and 5th are working days). Timing
features are scored in working-day deltas, not calendar deltas, so a settlement that looks
"3 days late" across a long weekend is correctly recognised as on time.

### 4.3 Normalisation — `normalize.py`

Three dissimilar sources are projected onto one `NormRecord` shape, so blocking, scoring and
assignment are written once instead of once per source pair.

Two reference forms, with different jobs:
- `ref_strict` — uppercased, non-alphanumerics stripped. Safe for **equality**.
- `ref_loose` — additionally folds `O`→`0`, `I`/`L`→`1`. Used for **blocking only**, never
  for asserting a match, because folding is lossy and can collide two genuinely different
  references.

UTR extraction returns `(value, is_strict)` — the parse reports its own confidence rather
than letting a guess from an unreadable narration masquerade as a reliable reference.

### 4.4 Blocking — `block.py`

Scoring all pairs is O(n·m): 19.3M comparisons at 5,000 orders per side. Blocking scores a
pair only if it shares a cheap key: exact reference, loose reference, exact amount,
**fee-adjusted amount** (so a gross-versus-net comparison still blocks together), or — as a
narrow conditional fallback — value date.

The date key is deliberately conditional. Applying it unconditionally over a ±9-day window
pruned only **47%** of the pair space in an early version, which would have made the
throughput claim hollow. Restricting it to records with no usable reference, plus indexing
reference-less right-hand records separately so they stay reachable, took pruning to
**99.98%** with no loss of reachable links.

**No amount tolerance in the index.** Tolerance is what lets the one-paise adversary
through, so where a band is genuinely needed (FX reconstruction) it is applied visibly in a
detector, never silently in the index.

The blocking recall ceiling is measured and published: `coverage_of` reports how many true
links were never even generated as candidates. A recall ceiling you have measured is
engineering; one you have not is luck.

### 4.5 Scoring — `score.py`

Split into "what is true about this pair" (pure, local) and "what rule does that justify"
(provisional). The verdict is provisional because two rungs depend on **uniqueness**, which
is a property of the whole candidate set: a bank credit with an illegible narration matching
exactly one settlement on amount and date is nearly certain, and the identical pair becomes a
coin flip the moment a second settlement shares that amount and date. Same pairwise
features; opposite correct verdicts.

Every rule carries its evidence, **including items that oppose the match**. An evidence
chain listing only reasons to agree is marketing; the opposing lines are what let a reviewer
find the weak point fast.

### 4.6 Assignment — `assign.py`

**Optimal assignment, not greedy first-best.** Greedy is order-dependent: record A,
processed first, claims the only partner record B could have matched. `scipy`'s
`linear_sum_assignment` maximises total evidence per connected cluster, which is
order-invariant by construction. `test_assignment_is_order_invariant` guards it — and
**caught a real bug**: the cost matrix was originally built in input-index order, so
shuffling the input changed which duplicate row won. Rows and columns are now ordered by
record id.

**Ties escalate on every rung.** Fixing the ordering exposed a worse problem: a tie was
being resolved arbitrarily, producing a false positive. When two candidates score
identically, nothing in the data identifies the right one, so Trikon reports the ambiguity
and matches neither. Cost: ~2pp recall on the smallest batch. Benefit: zero false positives
at every scale.

**Bidirectional subset-sum** for N:M payouts. `merge` = several settlements arrive as one
credit; `split` = one settlement arrives as several credits. Implementing only one direction
silently fails on the other — the split case was missing initially and showed up as four
unexplained false negatives. If **more than one distinct subset** sums to the target, the
result is ambiguous and no match is produced.

### 4.7 Exception detection — `classify.py`

16 codes, every one raised by arithmetic or set logic. A model may later attach prose; the
code, subject records, rupees at risk and evidence are fixed before any model runs.

Four decisions here are about **not double-counting**, which is the difference between an
exception list a controller trusts and one they audit:

1. **A wrong fee is one finding, not two.** GST is levied on the fee, so a corrupted fee
   necessarily breaks the tax line too. That is one root cause with a derived consequence:
   `FEE_MISMATCH` absorbs the consequent tax error into its exposure, and `GST_MISMATCH` is
   reserved for rows whose fee is correct. Before this fix, 3 injected fee defects produced
   5 exceptions.
2. **A failed settlement is not also "missing in bank".** Its absence from the bank is
   explained by the failure.
3. **An amount-corrupted settlement did not "never arrive".** `pair_residue_by_variance`
   pairs leftover settlements and credits that are evidently the same payout (shared UTR, or
   near amount on a near date) and reports one `AMOUNT_MISMATCH_UNEXPLAINED` naming both.
   Critically it **does not create a match link** — pairing for diagnosis and matching for
   reconciliation carry different burdens of proof.
4. **A duplicate row that lost 1:1 assignment is not an orphan.** `MISSING_IN_BOOKS` fires
   only when the `order_id` appears nowhere in the ledger.

**Review cases** group findings by connected component over shared record ids, then combine
exposure by kind: several *principal* findings on one record describe the same rupees (take
the maximum), while *delta* findings — a fee over-charge, a GST error — are true increments
(sum them). On one batch this reduced reported exposure from an inflated ₹1,23,286 to a
correct ₹81,123.

### 4.8 LLM adjudication — `llm/`

Five fences, each answering a specific way a model can be wrong. Full detail in the README;
the load-bearing one is: **decisions are clamped to the rule's ceiling (0.70 / 0.50), and
auto-accept needs 0.90.** No model output can produce an auto-accepted match. Verified in
`tests/test_llm_safety.py` against a model that accepts everything at 0.999.

Responses are cached on disk by evidence hash. This is a reproducibility mechanism, not a
speed optimisation: it makes reported metrics checkable, the demo offline-safe, and the
submission survivable if the provider disappears before the deadline.

**Transport is verified against a local stub, not a live provider.**
`tests/test_llm_transport.py` runs the real client over a real socket against an
OpenAI-compatible stub whose behaviour each test scripts — covering bearer auth, JSON
salvage from markdown-fenced and prose-wrapped output, `429` retry honouring `Retry-After`,
`5xx` fallback to the next model, `400` without retry burn, `401` degrading to escalation
with the deterministic results untouched, the budget cap, and the cache suppressing a second
call. A stub is the stronger choice: one successful live request proves nothing repeatable
and stops proving anything the moment the provider changes. The gap this leaves is
deliberate and stated — adjudication *quality* is unmeasured, since scoring it needs a live
model and a labelled set (§8).

### 4.9 Evaluation — `evaluate.py`

Ground truth is loaded **here and nowhere else**.
`test_ground_truth_is_never_read_by_the_pipeline` proves the separation structurally by
mutilating the truth object and asserting identical pipeline output.

Three methodological choices:
- **Two recall figures** (see README) — refusal and failure are different things.
- **False positives are priced**, not just counted. Mismatching two 40-lakh settlements is
  not "one error" in any sense a controller cares about.
- **Calibration groups by distinct stated confidence**, not fixed deciles. The ladder emits
  a handful of exact values, so a decile histogram collapses to one point and hides whether
  each individual rung is honest. An earlier version also compared accuracy to the bucket
  *midpoint*, reporting ECE 0.1000 for a system that was in fact well calibrated; standard
  ECE uses mean predicted confidence, and the corrected figure is 0.0012.

## 5. Stack, and what it isn't

| Choice | Why | Rejected alternative |
|---|---|---|
| Python + Pydantic v2 | frozen models make source records immutable evidence | — |
| SQLite-free, in-memory | a reviewer must run this in two commands | Postgres — setup cost, no benefit at this scale |
| `scipy.linear_sum_assignment` | order-invariant optimal matching in one call | hand-rolled greedy — order-dependent |
| `rapidfuzz` | fast, well-tested edit similarity | hand-rolled Levenshtein |
| FastAPI | typed request models, free OpenAPI docs | Flask |
| **Single-file HTML dashboard** | zero build step; `make dashboard` cannot fail on npm | React + Vite — a toolchain risk with no payoff on an 11-day deadline |
| **No LangGraph / agent framework** | the DAG is fixed; a framework adds indirection over ~150 lines and would obscure the audit trail | LangGraph |

## 6. Security posture

- **No key has a default.** `Settings.llm_api_key` defaults to `None`; blank and placeholder
  values resolve to "absent" so a half-filled `.env` cannot produce doomed calls reported as
  model failures.
- **`redacted()` is the only way settings are printed**, so `trikon doctor` and
  `/api/config` are safe on a screen share. A test asserts the key never appears.
- **Prompts carry derived evidence only** — no raw source rows, no card or FX fields. Tested.
- **Read-and-report only.** There is no endpoint that mutates data or moves money. The most
  Trikon does is tell a human what to look at, which is the correct bounded-autonomy posture
  for a system reasoning about settlements.
- Synthetic data throughout, per the track's requirement, so nothing real transits a
  third-party model provider.

## 7. Failure modes and what happens

| Failure | Behaviour |
|---|---|
| No API key | Deterministic mode; ambiguous cases escalate; all metrics still produced |
| Provider down / rate-limited | Retry with backoff honouring `Retry-After`, then fallback models, then escalate |
| Budget exhausted mid-run | Remaining cases stay escalated; the run is partial but honest |
| Unparseable model output | Response discarded; escalate |
| Model invents a case id or evidence | Whole response rejected; escalate |
| Two subsets sum to one credit | No match; `AMBIGUOUS_MULTI_CANDIDATE` |
| Subset pool exceeds bounds | Search abandoned; escalate rather than hang |
| Corrupt LLM cache file | Treated as a miss; costs one API call |
| Shuffled input | Identical output (asserted by test) |

## 8. What I would build next

In priority order, honestly assessed:

1. **Real-data validation.** Everything here is measured on synthetic data with known truth.
   The next real signal comes from a merchant's actual recon report, where reference quality
   and defect distribution will differ from my generator's.
2. **Learned scoring on the residue.** The rule ladder is deliberately hand-written and
   auditable. Once labelled escalation outcomes exist, a calibrated classifier on the
   ambiguous residue could reduce review volume — trained on human decisions, kept below the
   auto-accept ceiling.
3. **Incremental reconciliation.** Currently a batch is reconciled whole. A production
   system reconciles a rolling window and must handle a record settling after it was already
   reported as unsettled.
4. **Fee-schedule verification** against a contracted rate card, turning Q2 from "internally
   consistent" into "correctly priced".
