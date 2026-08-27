# Demo & pitch package

Everything needed to record the 5-minute video and answer a panel. All numbers here are
reproducible with `make stress` and `make run`.

---

## Before you record

```bash
make install
make test          # 52 pass -- shows this on camera if you want
make report        # warms data/reports/report.json
make dashboard     # http://localhost:8000
```

Set the dashboard to **1000 orders, seed 42**. Have a terminal ready in a second window.

> **Do this once before recording:** open the dashboard and click through all six tabs.
> The JS is syntax-checked and its data contract is asserted by a test, but there is no
> browser in the build environment, so its *visual* layout is unverified. Check for label
> collisions and overflow, and toggle the theme button once.

---

## The 5-minute demo

**Timing target: 4:40, leaving buffer.** The through-line is the track's own bar:
*throughput + measured accuracy + an honest exception list.* Say those three words early
and hit them in order.

### 0:00–0:35 — The problem, concretely

> "Every Razorpay merchant has to prove three things at close: every paid order reached the
> gateway, every settlement equals its transactions net of fee and GST, and every settlement
> actually hit the bank. Today that's a spreadsheet.
>
> It's hard for a specific reason — it's **not one-to-one**. Many payments net into one
> settlement, which lands as one bank credit, minus fees, on T+2 working days over Indian
> bank holidays. And Razorpay's own docs say that when the amount due exceeds live balance,
> only *part* of it settles and the rest rolls over. A VLOOKUP can't express any of that."

*On screen: the three-source diagram from the README.*

### 0:35–1:10 — What it does, and the honest headline

Terminal:

```bash
make run
```

> "Two thousand records across three sources. 127 milliseconds. Precision **1.000** —
> **zero false matches**. Match rate 99.1%.
>
> And here's the number I care about most: **it detected 100% of the defects I deliberately
> planted, and raised zero false alarms** on the ones designed to trick it into escalating
> something it should have absorbed."

*Point at the CALIBRATION block: "and no confidence tier is over-confident."*

### 1:10–1:55 — Throughput is real, not a toy batch

```bash
make stress
```

> "120 records to nearly twenty thousand. Precision stays 1.000 and false positives stay
> zero the whole way. Eleven thousand records a second.
>
> Notice recall is **lowest** on the *smallest* batch — 0.939. That's not a bug. I inject a
> fixed number of defects regardless of size, so the 300-record batch is 14% defective and
> the 20,000-record one is 0.2%. **The small batch is the hard one, and I lead with it.**
>
> That throughput comes from blocking — 19.3 million possible pairs reduced to under five
> thousand actually scored. **99.98% pruned**, and I measure the recall ceiling that costs
> me, so it's not a free lunch I'm hiding."

### 1:55–2:50 — The honest exception list *(the heart of the demo)*

Dashboard → **Exceptions** tab.

> "49 review cases, ordered by rupees at risk — because that's how a controller triages.
> Top case: a settlement Razorpay says it processed, ₹3.78 lakh, with a UTR, and **no bank
> credit matches it, alone or in any combination**. That's unreceived cash.
>
> Every case opens into an evidence chain — and it shows the evidence **against** too, so a
> reviewer can argue with a specific line instead of the verdict."

Expand a `FEE_MISMATCH` case:

> "This one recomputes the fee from the method and amount, and the 18% GST on it, and says
> exactly how much was over-charged. That's the tax-line matching from the track brief,
> living inside the recon loop."

### 2:50–3:35 — Where it refuses, and why that's the point

Dashboard → **Reconciliation** tab, then Exceptions filtered to ambiguity.

> "Two things in this batch are **unresolvable by construction**, and I built them on
> purpose.
>
> First: two settlements with identical amounts on the same day, two identical bank credits,
> no legible UTR. There is **no fact of the matter** about which belongs to which. A system
> that picks one gets it right half the time and reports a better match rate than it earned.
> Mine escalates it — and there's a **test that fails if it ever guesses**.
>
> Second, the opposite trap: two settlements exactly **one paise apart**. That *is*
> resolvable — by exact amount. Any system with a sloppy rounding tolerance turns this into
> an ambiguity or crosses the pair. Mine resolves both correctly, and that's tested too.
>
> Money is integer paise everywhere in this codebase. Never a float."

### 3:35–4:15 — Where the LLM is allowed to act

Dashboard → **Pipeline** tab, rule ceiling table.

> "Deterministic rules do the work. The LLM only sees candidates the rules genuinely can't
> settle — a damaged invoice reference where the amount agrees exactly. On 2,000 records
> that's about **eight API calls**, because cost scales with *ambiguity*, not with volume.
> That's why it runs on a free tier capped at 50 requests a day.
>
> And here's the fence: an adjudicated decision is **clamped to its rule's ceiling** — 0.70
> or 0.50. Auto-accept needs 0.90. So **no model output can ever produce an auto-accepted
> match.** I test that against a hostile model that accepts everything at 0.999 confidence:
> zero ceiling violations, zero auto-accepts.
>
> A hallucination here can cost a reviewer attention. It cannot cost money."

*Then:* `make test` → 52 passing.

### 4:15–4:40 — Close on measurement

Dashboard → **Evaluation** tab.

> "Every number I've shown is measured against a ground-truth file the pipeline never reads —
> and there's a test that mutilates the truth object and proves the output doesn't change.
>
> Calibration groups by stated confidence: expected calibration error **0.0012**, and no tier
> over-confident. Precision 1.000, recall 0.999, 100% detection, 0% false alarms, zero false
> matches across twenty thousand records.
>
> Three bugs in this system were caught only *because* I measured — including one where my own
> test fixture was wrong and the reconciler was right. That's the difference between a demo
> and a system you'd let near a settlement account."

---

## 5-minute pitch structure (if you present without a screen)

| Beat | Seconds | Content |
|---|---|---|
| Hook | 20 | "Reconciliation is not a matching problem, it's a *refusing to match* problem." |
| Problem | 40 | N:M, net≠gross, partial settlements, T+2 over bank holidays |
| What it is | 30 | Three-way recon controller: orders ↔ settlements ↔ bank |
| Throughput | 30 | 11k rec/s, 99.98% pruned, measured recall ceiling |
| Accuracy | 60 | P 1.000, 0 FP at 20k records, 100% detection, 0% false alarms, ECE 0.0012 |
| Honest exceptions | 60 | 49 cases by ₹ at risk; evidence chains; the two unresolvable-by-design cases |
| The LLM fence | 40 | ambiguity-scaled cost; clamped below auto-accept; hostile-model test |
| Close | 20 | "Three bugs found by measurement, one of them in my own fixture." |

---

## Likely judge questions, and answers

**"Your recall isn't 100%. What's missing?"**
> Deliberate refusals, mostly. Three of the tier-3 misses at 1,000 orders are settlements
> whose bank credit had a corrupted amount. Trikon *finds* the pair — it reports
> `AMOUNT_MISMATCH_UNEXPLAINED` naming both records — but declines to record a match, because
> the amounts disagree. That's why I publish two figures: recall 0.991 counts those against
> me; resolvable recall 0.999 excludes them. I'd rather show both than pick the flattering one.

**"How do I know you didn't tune to your own test set?"**
> You can't, from the numbers alone — so check the structure instead. Ground truth is written
> by the generator *before* the run, loaded only in `evaluate.py`, and
> `test_ground_truth_is_never_read_by_the_pipeline` mutilates the truth object and asserts the
> pipeline's output is unchanged. Also: change the seed. `--seed 7`, `--seed 99` — the numbers
> hold.

**"Isn't this just a matching script with a rules engine?"**
> The parts that aren't: optimal bipartite assignment rather than greedy, so output doesn't
> depend on row order — I have a test that caught exactly that bug. Bidirectional subset-sum
> for N:M payouts, which pairwise matching structurally cannot reach. Uniqueness-conditional
> confidence, where identical pairwise features produce opposite correct verdicts. And a
> measured calibration curve. A script doesn't know when to refuse.

**"Why so little LLM? Isn't this an AI buildathon?"**
> Because the track's stated thesis is that verification capacity is the bottleneck, and
> verification is where LLMs are weakest and arithmetic is strongest. I use the model where
> language judgement genuinely helps — deciding whether `INV-202607-00421` and
> `INV20260700421` are the same reference — and nowhere near the money arithmetic. The
> constraint also made the system better: because cost scales with ambiguity, it runs on a
> free tier and stays reproducible.

**"What if the LLM hallucinates?"**
> It structurally cannot create a false match. Decisions are clamped to the rule's ceiling —
> 0.70 or 0.50 — and auto-accept needs 0.90, so every adjudicated link stays flagged for
> human review. `tests/test_llm_safety.py` runs a model that accepts everything at 0.999 and
> asserts zero ceiling violations and an unchanged auto-accepted set. A response citing an
> invented case id or evidence feature is discarded whole.

**"Why no LangGraph / multi-agent architecture?"**
> The stage order is forced by data dependencies, not judgement — tier-2 arithmetic needs
> settlement membership, presence exceptions need to know what matching left over. A model
> sequencing that would either reproduce my order or get it wrong, and would vary between
> runs. Reproducibility is the product; every metric in my README depends on it. Seven agents
> that are each one LLM call would demo better and defend worse.

**"Would this work on real Razorpay data?"**
> The schema is real — I built against the published `settlements/recon/combined` field names,
> the T+2 working-day rule, and the documented partial-settlement behaviour. What I can't
> claim is that my *defect distribution* matches production; reference quality in particular
> would differ. That's the first thing I'd validate with real data, and it's the top item in
> ARCHITECTURE.md §8.

**"What's the weakest part?"**
> Two things. The threshold sweep is flatter than I expected — precision is 1.000 at every
> threshold, so it doesn't demonstrate a real tradeoff, only that 0.90 is the right operating
> point. And calibration is only valid on my generator's distribution; ECE 0.0012 says the
> ladder is honest *here*, not that it would be on a merchant's real reference quality.

**"How long did this take, and what would you do with more time?"**
> Real-data validation first. Then a learned classifier on the ambiguous residue only —
> trained on human escalation outcomes, kept below the auto-accept ceiling so it can reduce
> review volume without ever being able to create a false match. Then incremental
> reconciliation over a rolling window, which is what production actually needs.

---

## Known weaknesses and how to frame them

| Weakness | How to address it before it's asked |
|---|---|
| Synthetic data only | Say it first. The schema is real and cited; the defect distribution is mine. |
| Threshold sweep is flat | Present it as "0.90 is the operating point, here's the evidence" rather than a tradeoff story. |
| Dashboard not visually regression-tested | Click through it before recording. Don't claim test coverage it doesn't have. |
| Fee table isn't Razorpay's real price list | Stated in the README. Q2 verifies internal consistency, not contracted pricing. |
| Recall < 1.0 | Lead with *why*, using the two-recall framing, before anyone reads it as a failure. |

---

## Submission checklist

Official Track 04 deliverables are exactly three: **a public repo, a 5-minute pitch video,
and the architecture.**

- [x] **Public repo** — <https://github.com/007-SARANG/razorpay> (public, 34 files, no secrets)
- [x] **Architecture** — `ARCHITECTURE.md` ✅
- [ ] **5-minute video** — script above; record with dashboard + terminal
- [ ] `README.md` with reproducible numbers ✅
- [ ] Tests pass (`make test` → 52) ✅
- [ ] Runs with **no API key** ✅
- [ ] No hardcoded secrets — `.env` gitignored, `.env.example` committed ✅
- [ ] `data/cache/` committed so metrics reproduce offline (only if you get `--llm` working)

**Provider status (be accurate about this if asked):** AgentRouter returns
`unauthorized_client_error` on every endpoint including `GET /v1/models`, so it gates
free-quota access to recognised coding-tool clients rather than to arbitrary API callers.
The LLM path is therefore exercised only against simulated adjudicators. The safety fences
are fully tested; the provider *transport* is not. Every published metric comes from the
deterministic path, so nothing in the results depends on this being resolved.
- [ ] Apply via the form: <https://forms.gle/d9r2gvxp8cmoZhon9>
- [ ] **Deadline: 5 September 2026**

### Before pushing

```bash
grep -rIn --exclude-dir=.venv --exclude-dir=.git -E "(ak-|sk-or-|sk-)[A-Za-z0-9]{12}" . || echo "no keys found"
make test && make stress
```
