# Trikon

**Three-way settlement reconciliation for Razorpay merchants.** Trikon reconciles a
merchant's order ledger against Razorpay settlement recon rows against bank statement
credits in one pass, reports a measured match rate, and produces an honest list of the
records it refused to resolve.

Built for **Razorpay AI Buildathon 2026, Track 04 — AI Finance Controller**.

तriकोण / *trikon* — "triangle". Three sources, one triangle: orders ↔ settlements ↔ bank.

---

## The headline

Measured on a 19,708-record batch with 41 deliberately injected defects, no tuning to the
test set, and **no LLM involved**:

| | |
|---|---|
| **False matches** | **0** |
| Precision | **1.000** |
| Match rate (recall) | 0.999 |
| Must-report defects detected | **100%** (29/29) |
| False alarms on absorbable defects | **0%** (0/12) |
| Throughput | 11,269 records/sec |
| Expected calibration error | 0.0001 |

The full scaling table, and the reasons the smallest batch scores *lower* recall than the
largest, are in [Results](#results).

## Why this exists

Every Razorpay merchant must prove three things at each close:

1. Every order the books call paid actually reached the gateway.
2. Every settlement equals its member transactions, net of fee and 18% GST.
3. Every settlement actually landed in the bank.

Today that is a spreadsheet. Razorpay ships a settlement recon report *because* this is
painful — but the report says what happened, not what fails to add up. Four things make it
genuinely hard, and all four are modelled here:

- **It is N:M, not 1:1.** Many payments net into one settlement, which lands as one bank
  credit. `VLOOKUP` cannot express that.
- **Net ≠ gross.** Each row carries its own fee plus GST, so matching on gross fails by
  construction.
- **Partial settlements are real.** Razorpay documents that when the amount due exceeds
  live balance, only the subset summing to live balance settles and the rest rolls over. A
  naive matcher reads this as a shortfall.
- **T+2 lands on working days.** Over weekends, bank holidays, and the 2nd/4th Saturday
  that Indian banks close, naive date arithmetic breaks.

## Quick start

No API key required. The entire pipeline, and every number above, runs deterministically
offline.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python -m trikon.cli run --orders 1000 --exceptions 5 --cash
```

Then the dashboard:

```bash
make dashboard          # or: .venv/bin/python -m uvicorn api.main:app --port 8000
# open http://localhost:8000
```

### Commands

| Command | What it does |
|---|---|
| `trikon doctor` | Show configuration; probe the LLM provider. Never prints a secret. |
| `trikon generate --orders 1000` | Write a synthetic batch plus its ground truth to `data/batches/`. |
| `trikon run --orders 1000` | Reconcile and print the full report. `--exceptions N`, `--cash`, `--json-out`. |
| `trikon stress --sizes 120 1000 10000` | Accuracy and throughput across batch sizes. |
| `trikon sweep` | Vary the auto-accept threshold; show the precision/recall/review tradeoff. |
| `make test` | 69 tests, including the adversarial refusal cases. |

## How it works

```
                    orders          settlement recon rows          bank credits
                  (merchant OMS)      (Razorpay schema)          (bank statement)
                        │                     │                         │
                        └──────────┬──────────┴────────────┬────────────┘
                                   ▼                       ▼
                            normalize (integer paise, IST working-day index,
                                       canonical references)
                                   │
                            block   (hash buckets; 99.98% of the pair space
                                       never scored)
                                   │
                            score   (feature vector → R1..R8 rule ladder)
                                   │
                            assign  (Hungarian per cluster + bidirectional
                                       subset-sum for N:M payouts)
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
            auto-accepted                  ambiguous residue  ──► optional LLM
            (arithmetic-backed)                                   adjudication
                   │                               │              (capped below
                   └───────────────┬───────────────┘               auto-accept)
                                   ▼
                            classify (16-code exception taxonomy,
                                      ₹ at risk, evidence chains)
                                   │
                                   ▼
                            evaluate (vs ground truth: P/R/F1,
                                      calibration, throughput)
```

Full detail, and the reasoning behind each choice, is in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

### The three ideas that matter

**1. Confidence comes from a rule, never from a model.**

| Rule | Ceiling | Fires when |
|---|---|---|
| `R1_EXACT` | 1.00 | reference exact + amount exact + in window |
| `R2_FEE_EXPLAINED` | 0.99 | gap equals recomputed fee + 18% GST, **exact to the paise** |
| `R5_SUBSET_SUM` | 0.95 | a unique exact subset explains the counterpart |
| `R3_TIMING_SHIFTED` | 0.92 | exact match, but outside the settlement window |
| `R4_UNIQUE_AMOUNT` | 0.90 | no usable reference; amount+date exact **and unique** |
| `R6_FUZZY_REF` | 0.70 | reference damaged but amount corroborates → *LLM may adjudicate* |
| `R7_AMBIGUOUS` | 0.50 | two candidates fit equally well → *LLM may adjudicate* |
| `R8_NO_MATCH` | 0.00 | exception |

Auto-accept requires 0.90. Both adjudicable rungs sit **below** it, and a test asserts that
invariant holds. Therefore **no model output can ever produce an auto-accepted match** — a
hallucination can cost a reviewer attention, never money.

**2. Ties escalate. Always.**

When two candidates explain a record equally well, nothing in the data says which is right.
Trikon reports the ambiguity instead of picking. On the smallest batch that costs about 2
percentage points of recall and buys **zero false positives** — the trade this codebase
makes everywhere, stated in numbers rather than asserted.

**3. Two recall figures, both reported.**

A defect that corrupts an amount leaves the underlying correspondence intact, but a correct
reconciler must refuse to bless it. Scoring that refusal as a miss punishes right
behaviour; dropping those links from the denominator inflates the metric. So both are
published: `recall` over every true link, and `resolvable_recall` over links carrying no
must-report defect.

## Results

`make stress` — seed 42, no LLM, nothing tuned to the test set:

| Orders | Records | Precision | Recall | Resolvable recall | FP | Detection | False alarms | Time | Throughput | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 120 | 300 | 1.000 | 0.939 | 0.995 | 0 | 1.000 | 0.000 | 22 ms | 13,362/s | 0.0088 |
| 500 | 1,036 | 1.000 | 0.983 | 0.999 | 0 | 1.000 | 0.000 | 64 ms | 16,215/s | 0.0025 |
| 1,000 | 2,017 | 1.000 | 0.991 | 0.999 | 0 | 1.000 | 0.000 | 127 ms | 15,837/s | 0.0012 |
| 5,000 | 9,893 | 1.000 | 0.998 | 1.000 | 0 | 1.000 | 0.000 | 789 ms | 12,545/s | 0.0003 |
| 10,000 | 19,708 | 1.000 | 0.999 | 1.000 | 0 | 1.000 | 0.000 | 1,749 ms | 11,269/s | 0.0001 |

**Recall is lowest on the smallest batch, and that is expected.** The defect plan injects a
fixed count of defects regardless of size, so a 300-record batch is ~14% defective while a
19,708-record batch is ~0.2%. The small batch is the hard one. We report it first rather
than leading with the flattering number.

### Calibration

Grouped by *distinct stated confidence*, because the rule ladder emits a handful of exact
values and a decile histogram would collapse them into one uninformative point:

| Stated | n | Correct | Observed | Gap | Direction |
|---|---|---|---|---|---|
| 1.00 | 872 | 872 | 1.000 | 0.000 | exact |
| 0.99 | 2 | 2 | 1.000 | 0.010 | under-confident |
| 0.95 | 8 | 8 | 1.000 | 0.050 | under-confident |
| 0.92 | 2 | 2 | 1.000 | 0.080 | under-confident |
| 0.90 | 5 | 5 | 1.000 | 0.100 | under-confident |

**No rung is over-confident.** Every gap is the ladder claiming less than it delivered,
which is the safe direction.

### Threshold sweep

| Threshold | Precision | Recall | Straight-through | FP |
|---|---|---|---|---|
| 0.50 – 0.90 | 1.000 | 0.991 | 0.991 | 0 |
| 0.95 | 1.000 | 0.983 | 0.983 | 0 |
| 1.00 | 1.000 | 0.981 | 0.981 | 0 |

Honestly, this is flatter than expected: precision is 1.000 everywhere because the ladder
only emits high confidence on arithmetic-backed evidence. The usable finding is at the top
— tightening past 0.90 costs recall for **zero** precision gain, which is why 0.90 is the
operating point.

### Negative control

A defect-free batch produces **precision 1.000, recall 1.000, zero exceptions of any hard
type**. The only findings are `DISPUTE_HOLD` and `UNSETTLED_AGED` — money genuinely in
flight, which a real clean month also has.

## The synthetic dataset

The generator builds an internally consistent merchant-month, then injects defects and
records each one as ground truth. **The reconciliation pipeline never sees ground truth**;
a test asserts that corrupting it does not change the pipeline's output.

Modelled from Razorpay's published behaviour: real recon-report field names
(`entity_id`, `settlement_utr`, `on_hold`, `dispute_id`, …), T+2 over an Indian banking
calendar, live-balance-constrained partial settlement, INR conversion at the
payment-creation rate, and zero-MDR UPI.

**19 defect types.** 29 must be reported (false-negative traps); 12 must be *absorbed*
without raising anything (false-positive traps):

| Must be reported | Must be absorbed |
|---|---|
| missing in gateway / books / bank | mutated receipt reference |
| amount, fee, GST, FX variance | split bank credit (one → two) |
| duplicate row, double charge | merged bank credit (two → one) |
| twin-amount ambiguity | illegible UTR in narration |
| timing breach, aged unsettled | one-paise twin *(must resolve correctly)* |
| failed settlement, dispute hold | |

Two are adversarial controls. **`ONE_PAISE_TWIN`** creates two settlements one paise apart
with unreadable UTRs: any amount tolerance wider than zero turns this into an ambiguity or a
crossed pair. **`TWIN_AMOUNT_AMBIGUITY`** is unresolvable by construction, and the system
must escalate rather than guess. Both are asserted in the test suite.

## LLM adjudication (optional)

Disabled by default and never required. Configure via `.env` (see `.env.example`) — any
OpenAI-compatible endpoint works:

```bash
TRIKON_LLM_BASE_URL=https://openrouter.ai/api/v1
TRIKON_LLM_API_KEY=...
TRIKON_LLM_MODEL=...
```

Then `trikon run --llm`. On a 2,000-record batch this issues roughly **8 requests**, because
only the ambiguous residue reaches a model and cases are batched — **LLM cost scales with
ambiguity, not with volume.** That is what makes it viable on a free tier capped at 50
requests/day. Responses are cached to `data/cache/` by evidence hash, so re-runs are free,
reproducible, and work offline.

Five fences, each tested offline against a hostile model that accepts everything at 0.999
confidence:

1. Only R6/R7 candidates are ever sent.
2. Decisions are clamped to the rule ceiling — **never auto-accepted**.
3. All arithmetic is precomputed; the model is never asked to add.
4. A response citing an unknown case id or evidence feature is discarded entirely.
5. Any failure — bad JSON, timeout, exhausted budget — escalates to human review.

### The wire path is verified, not assumed

`tests/test_llm_transport.py` runs the real `LLMProvider` against a local
OpenAI-compatible stub server over an actual socket — no mocked transport. It asserts, on
every run:

| Behaviour | Verified |
|---|---|
| Bearer auth, correct model, `temperature=0`, JSON schema sent | ✅ |
| An HTTP `MATCH` becomes a **review-flagged, never auto-accepted** link | ✅ |
| JSON salvaged from markdown fences, prose wrappers, padding | ✅ (5 shapes) |
| Unparseable output → escalate | ✅ |
| Fabricated `case_id` arriving over the wire → whole response discarded | ✅ |
| `429` retried, honouring `Retry-After` | ✅ |
| `5xx` → falls through to the fallback model | ✅ |
| `400` → no retry burn, move on | ✅ |
| **`401` → escalates; deterministic results untouched** | ✅ |
| Budget cap stops calls mid-batch, result stays honest | ✅ |
| Cache prevents a second HTTP call entirely | ✅ |

A stub is deliberately preferred over a one-off live call: a single successful request to a
third-party proxy proves nothing repeatable and stops proving anything the moment that
provider rate-limits or disappears. These tests exercise every branch, in CI, with no key.

**What this does not claim:** nothing here measures how *well* a given model adjudicates.
That needs a live model and a labelled set, and is listed as future work in
[ARCHITECTURE.md](ARCHITECTURE.md) §8.

## Project layout

```
trikon/
  money.py          integer paise, fee + 18% GST, explicit half-up rounding
  calendar_ist.py   T+2, holidays, 2nd/4th Saturday
  models.py         Razorpay-shaped entities, 16-code taxonomy, rule ceilings
  normalize.py      project three sources onto one comparable shape
  block.py          hash-bucket candidate generation + measured recall ceiling
  score.py          pairwise features → provisional rule + evidence chain
  assign.py         Hungarian assignment + bidirectional subset-sum
  classify.py       exception detectors, ₹ at risk, review-case grouping
  pipeline.py       the deterministic orchestrator
  evaluate.py       metrics with precise definitions, calibration, ECE
  report.py         one JSON serialiser shared by CLI and API
  generate/         synthetic world + 19 defect injectors with ground truth
  llm/              provider, disk cache, bounded adjudicator
api/main.py         FastAPI: /api/reconcile, /api/config, serves the dashboard
web/index.html      dashboard, single self-contained file, no build step
tests/              69 tests
```

## Limitations

Stated plainly, because a limitation found by a reviewer costs more than one you declared:

- **Synthetic data cannot cover every real pathology.** The 19 defect types are drawn from
  documented Razorpay behaviour and ordinary accounting failure, not from a production
  incident log.
- **Calibration is only valid on this generator's distribution.** ECE of 0.0001 says the
  ladder is well calibrated *here*; a real merchant's reference-quality would differ.
- **Subset-sum is capped** at 5 members from a pool of 24, then escalates. Beyond that the
  search is exponential and a bounded refusal beats an unbounded wait.
- **Tier 2 verifies arithmetic, it does not verify the fee schedule.** The rate table is
  synthetic and shared by generator and verifier, so it detects rows inconsistent *with the
  contracted rate*, not a mis-negotiated rate.
- **The fee table is not Razorpay's published price list.** It is representative.
- **The dashboard has not been visually regression-tested** — the JS is syntax-checked and
  its data contract is asserted, but there is no browser in the build environment.
- **Free-tier models are weaker adjudicators** than frontier models. The design makes this
  safe rather than invisible: a weak model produces more escalations, never a wrong match.
- **Adjudication quality is unmeasured.** The transport path is fully tested and the safety
  fences are proven, but no live model has been scored on a labelled set of ambiguous cases.
  Every metric in this README comes from the deterministic path, so none of them depend on
  a model's judgement.

## Licence

MIT.
