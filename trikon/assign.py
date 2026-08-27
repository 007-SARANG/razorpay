"""Global assignment and subset-sum resolution.

Two algorithmic choices here are the difference between this and a reconciliation script.

**Assignment, not greedy first-best.** The obvious implementation walks the candidate
list and takes the best partner for each record in turn. That is order-dependent: record
A, processed first, can claim the only partner that record B could have matched, leaving
B unmatched and producing a false negative that disappears if you shuffle the input. A
reconciler whose output depends on row order cannot be trusted or reproduced. We solve
each connected cluster as an optimal bipartite assignment instead
(:func:`scipy.optimize.linear_sum_assignment`), maximising total evidence across the
cluster, which is order-invariant by construction.

**Subset-sum for N:M.** A bank credit that consolidates several settlements equals no
single settlement, so pairwise matching cannot reach it at all. We search for subsets
that sum to the credit *exactly*. Crucially, if **more than one distinct subset** sums to
the target, the result is ambiguous and we return no match -- picking one would be a
guess dressed as a reconciliation. That refusal is the behaviour the exception list is
for.

Both routines are bounded: cluster size and subset cardinality are capped so a
pathological batch degrades into escalations rather than hanging.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Final, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from trikon.models import (
    ADJUDICABLE_RULES,
    RULE_CONFIDENCE_CEILING,
    Evidence,
    MatchRule,
    Tier,
)
from trikon.normalize import NormRecord
from trikon.score import PairFeatures, build_evidence, compute_features, provisional_rule

#: Two candidates whose base scores differ by less than this are treated as equally
#: good, which makes the winner a coin flip rather than a decision.
SCORE_TIE_EPSILON: Final[float] = 1e-9

#: Maximum number of records that may participate in one subset-sum search. Beyond this
#: the search is abandoned and the record is escalated, rather than allowed to run away.
MAX_SUBSET_POOL: Final[int] = 24

#: Maximum cardinality of a subset considered as a single consolidated payout.
MAX_SUBSET_SIZE: Final[int] = 5


@dataclass(frozen=True, slots=True)
class ScoredPair:
    """A candidate pair with its features, provisional rule and evidence."""

    left_index: int
    right_index: int
    rule: MatchRule
    score: float
    features: PairFeatures


@dataclass
class AssignmentResult:
    """Outcome of matching one tier."""

    tier: Tier
    #: (left_index, right_index, rule, confidence, evidence)
    accepted: list[tuple[int, int, MatchRule, float, tuple[Evidence, ...]]] = field(
        default_factory=list
    )
    #: Pairs needing adjudication before they can be accepted or rejected.
    for_adjudication: list[tuple[int, int, MatchRule, float, tuple[Evidence, ...]]] = field(
        default_factory=list
    )
    #: Left indices with no viable partner.
    unmatched_left: list[int] = field(default_factory=list)
    #: Right indices no left record claimed.
    unmatched_right: list[int] = field(default_factory=list)
    #: Ambiguity groups: (left_index, [competing right_indices]).
    ambiguities: list[tuple[int, list[int]]] = field(default_factory=list)
    #: N:M groups recovered by subset-sum: (right_index, [left_indices]).
    subset_groups: list[tuple[int, list[int]]] = field(default_factory=list)


def score_candidates(
    left: Sequence[NormRecord],
    right: Sequence[NormRecord],
    pairs: Iterable[tuple[int, int]],
    *,
    tier: Tier,
) -> list[ScoredPair]:
    """Score every candidate pair, discarding those the ladder rejects outright."""
    scored: list[ScoredPair] = []
    for i, j in pairs:
        features = compute_features(left[i], right[j], tier=tier)
        rule, score = provisional_rule(features)
        if rule is MatchRule.R8_NO_MATCH:
            continue
        scored.append(ScoredPair(i, j, rule, score, features))
    return scored


def _clusters(
    scored: Sequence[ScoredPair],
    left: Sequence[NormRecord],
    right: Sequence[NormRecord],
) -> list[tuple[list[int], list[int], list[ScoredPair]]]:
    """Partition candidates into connected components via union-find.

    Assignment is solved per component rather than over the whole batch: the full cost
    matrix would be mostly forbidden entries, and components are typically tiny, so this
    turns one large dense solve into many trivial ones.

    Within a component, rows and columns are ordered by **record id, not by input
    index**. This matters more than it looks. When two records score identically against
    the same counterpart -- which happens whenever a batch contains a duplicated gateway
    row -- the assignment solver has no basis to prefer either, and returns whichever the
    matrix happened to list first. Ordering the matrix by index makes that depend on the
    order rows arrived in, so shuffling the input silently changes which link is produced.
    A reconciler whose output depends on row order cannot be reproduced or trusted.
    Canonical id ordering makes the matrix -- and therefore the result -- identical for any
    input permutation. (The duplication itself is still reported, by
    :func:`trikon.classify.detect_duplicates`; this only fixes *which* twin gets credit.)
    """
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[str, int], b: tuple[str, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in scored:
        union(("L", pair.left_index), ("R", pair.right_index))

    groups: dict[tuple[str, int], list[ScoredPair]] = defaultdict(list)
    for pair in scored:
        groups[find(("L", pair.left_index))].append(pair)

    out: list[tuple[list[int], list[int], list[ScoredPair]]] = []
    for members in groups.values():
        lefts = sorted({p.left_index for p in members}, key=lambda i: left[i].record_id)
        rights = sorted({p.right_index for p in members}, key=lambda j: right[j].record_id)
        ordered_members = sorted(
            members,
            key=lambda p: (left[p.left_index].record_id, right[p.right_index].record_id),
        )
        out.append((lefts, rights, ordered_members))
    # Solve components in a canonical order too, so that any per-cluster side effects
    # (ambiguity reporting, claimed sets) are emitted identically across permutations.
    out.sort(key=lambda item: left[item[0][0]].record_id if item[0] else "")
    return out


def solve_assignment(
    left: Sequence[NormRecord],
    right: Sequence[NormRecord],
    scored: Sequence[ScoredPair],
    *,
    tier: Tier,
    auto_accept_threshold: float,
) -> AssignmentResult:
    """Assign left records to right records optimally, cluster by cluster.

    A rung that depends on uniqueness (R4) is confirmed only if the assigned pair has no
    equally-scoring rival for either endpoint; otherwise it is demoted to
    ``R7_AMBIGUOUS`` and routed to adjudication rather than accepted.
    """
    result = AssignmentResult(tier=tier)
    claimed_left: set[int] = set()
    claimed_right: set[int] = set()

    for lefts, rights, members in _clusters(scored, left, right):
        best: dict[tuple[int, int], ScoredPair] = {}
        for pair in members:
            key = (pair.left_index, pair.right_index)
            if key not in best or pair.score > best[key].score:
                best[key] = pair

        li = {idx: k for k, idx in enumerate(lefts)}
        ri = {idx: k for k, idx in enumerate(rights)}

        # Maximise total score: negate for the minimising solver. Forbidden cells carry a
        # large finite penalty so the solver always returns a complete assignment, which
        # we then filter -- np.inf would make the problem infeasible.
        cost = np.full((len(lefts), len(rights)), 1e6, dtype=float)
        for (i, j), pair in best.items():
            cost[li[i], ri[j]] = -pair.score

        rows, cols = linear_sum_assignment(cost)

        for r, c in zip(rows, cols):
            if cost[r, c] >= 1e6:
                continue  # solver filled a forbidden cell; not a real candidate
            i, j = lefts[r], rights[c]
            pair = best[(i, j)]

            rivals = _equal_rivals(best, pair)
            rule, confidence = pair.rule, RULE_CONFIDENCE_CEILING[pair.rule]

            if rivals:
                # A tie means the evidence does not single out a counterpart, whatever rung
                # produced it. The commonest cause is a duplicated gateway row: two rows
                # with identical economics both explain one order perfectly, so R1 fires
                # twice at 1.00 and the solver must pick one arbitrarily.
                #
                # Picking is the wrong move even though one of the two is "right", because
                # nothing in the data says which. Asserting a link to an arbitrary twin
                # manufactures a false positive half the time; refusing costs a little
                # recall and reports the duplicate instead -- which is the finding a
                # reviewer actually needs. Consistent with the rest of the system: when the
                # evidence does not determine an answer, escalate rather than guess.
                rule = MatchRule.R7_AMBIGUOUS
                confidence = RULE_CONFIDENCE_CEILING[rule]
                result.ambiguities.append((i, sorted({j, *(p.right_index for p in rivals)})))

            evidence = build_evidence(left[i], right[j], pair.features, rule)

            if rule in ADJUDICABLE_RULES or confidence < auto_accept_threshold:
                result.for_adjudication.append((i, j, rule, confidence, evidence))
            else:
                result.accepted.append((i, j, rule, confidence, evidence))
            claimed_left.add(i)
            claimed_right.add(j)

    result.unmatched_left = [i for i in range(len(left)) if i not in claimed_left]
    result.unmatched_right = [j for j in range(len(right)) if j not in claimed_right]
    return result


def _equal_rivals(
    best: dict[tuple[int, int], ScoredPair], winner: ScoredPair
) -> list[ScoredPair]:
    """Candidates that explain either endpoint exactly as well as the winner does."""
    rivals: list[ScoredPair] = []
    for (i, j), pair in best.items():
        if (i, j) == (winner.left_index, winner.right_index):
            continue
        touches_endpoint = i == winner.left_index or j == winner.right_index
        if touches_endpoint and abs(pair.score - winner.score) <= SCORE_TIE_EPSILON:
            rivals.append(pair)
    return rivals


@dataclass(frozen=True, slots=True)
class SubsetSolution:
    """A consolidated or split payout resolved by subset-sum.

    ``direction`` distinguishes the two real-world shapes, which are transposes of each
    other and both occur:

    * ``"merge"`` -- several settlements arrived as one bank credit (the bank
      consolidated the day's payouts).
    * ``"split"`` -- one settlement arrived as several bank credits (the payout was
      broken up in transit).

    A matcher that implements only one direction silently fails on the other, which is
    exactly the gap this dataclass exists to close.
    """

    direction: str
    one_index: int
    many_indices: tuple[int, ...]
    total: int


def _search_subsets(
    targets: Sequence[tuple[int, int, object]],
    pool: Sequence[tuple[int, int, object]],
    *,
    date_window_days: int,
) -> tuple[list[tuple[int, tuple[int, ...], int]], list[tuple[int, int]]]:
    """Generic exact subset-sum over ``pool`` against each target amount.

    ``targets`` and ``pool`` are ``(index, amount, day)`` triples. Returns
    ``(solutions, ambiguous)`` where a solution is ``(target_index, member_indices,
    total)`` and ambiguity is ``(target_index, distinct_subset_count)``.

    Ambiguity is a refusal, not a tie-break: if two different subsets both sum exactly to
    the target, the evidence does not identify which one is real, and choosing would
    fabricate a reconciliation. The search stops as soon as a second solution appears,
    since enumerating further cannot change the verdict.
    """
    import datetime as _dt

    solutions: list[tuple[int, tuple[int, ...], int]] = []
    ambiguous: list[tuple[int, int]] = []
    consumed: set[int] = set()

    for t_idx, t_amount, t_day in targets:
        candidates = [
            (idx, amount)
            for idx, amount, day in pool
            if idx not in consumed
            and amount <= t_amount
            and abs((day - t_day).days) <= date_window_days  # type: ignore[operator]
        ]
        if len(candidates) < 2 or len(candidates) > MAX_SUBSET_POOL:
            continue

        found: list[tuple[int, ...]] = []
        max_size = min(MAX_SUBSET_SIZE, len(candidates))
        for size in range(2, max_size + 1):
            for combo in combinations(candidates, size):
                if sum(a for _, a in combo) == t_amount:
                    found.append(tuple(i for i, _ in combo))
                    if len(found) > 1:
                        break
            if len(found) > 1:
                break

        if not found:
            continue
        if len(found) > 1:
            ambiguous.append((t_idx, len(found)))
            continue

        solutions.append((t_idx, found[0], t_amount))
        consumed.update(found[0])

    return solutions, ambiguous


def solve_subset_sums(
    left: Sequence[NormRecord],
    right: Sequence[NormRecord],
    *,
    unmatched_left: Sequence[int],
    unmatched_right: Sequence[int],
    date_window_days: int = 4,
) -> tuple[list[SubsetSolution], list[tuple[str, int, int]]]:
    """Resolve N:M payouts in **both** directions.

    Runs the merge search (many settlements to one credit) and then the split search
    (one settlement to many credits) over whatever remains. Only unmatched records
    participate, so this operates on the residue after assignment -- typically a handful
    of records, which is what keeps a combinatorial search affordable.

    Returns ``(solutions, ambiguous)`` where each ambiguity is
    ``(direction, target_index, distinct_subset_count)``.
    """
    solutions: list[SubsetSolution] = []
    ambiguous: list[tuple[str, int, int]] = []

    # --- merge: several left records sum to one right record -------------------------
    merge_solutions, merge_amb = _search_subsets(
        [(j, right[j].amount, right[j].day) for j in unmatched_right],
        [(i, left[i].amount, left[i].day) for i in unmatched_left],
        date_window_days=date_window_days,
    )
    consumed_left: set[int] = set()
    consumed_right: set[int] = set()
    for j, members, total in merge_solutions:
        solutions.append(
            SubsetSolution(direction="merge", one_index=j, many_indices=members, total=total)
        )
        consumed_right.add(j)
        consumed_left.update(members)
    ambiguous.extend(("merge", j, n) for j, n in merge_amb)

    # --- split: one left record equals the sum of several right records ---------------
    split_solutions, split_amb = _search_subsets(
        [(i, left[i].amount, left[i].day) for i in unmatched_left if i not in consumed_left],
        [(j, right[j].amount, right[j].day) for j in unmatched_right if j not in consumed_right],
        date_window_days=date_window_days,
    )
    for i, members, total in split_solutions:
        solutions.append(
            SubsetSolution(direction="split", one_index=i, many_indices=members, total=total)
        )
    ambiguous.extend(("split", i, n) for i, n in split_amb)

    return solutions, ambiguous


def subset_links(
    solutions: Sequence[SubsetSolution],
    left: Sequence[NormRecord],
    right: Sequence[NormRecord],
) -> list[tuple[int, int, str]]:
    """Expand subset solutions into ``(left_index, right_index, direction)`` links.

    Direction determines which side the ``one_index`` refers to, so callers do not have
    to re-derive it and cannot get the orientation backwards.
    """
    out: list[tuple[int, int, str]] = []
    for sol in solutions:
        if sol.direction == "merge":
            out.extend((i, sol.one_index, "merge") for i in sol.many_indices)
        else:
            out.extend((sol.one_index, j, "split") for j in sol.many_indices)
    return out
