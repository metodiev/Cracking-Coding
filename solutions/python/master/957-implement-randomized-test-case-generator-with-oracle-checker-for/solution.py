"""Task 957 - Implement randomized test-case generator with oracle checker for each problem.
(Tier: Master, CTCI Ch 11 - Testing)

Problem family under test: merging touching/overlapping closed intervals.  The generator is a
fixed-seed splitmix64 that emits a batch of cases covering a declared edge policy (empty, single,
disjoint, nested, touching, duplicate, unsorted, random), the oracle is an independent
merge-until-fixpoint implementation, and the checker compares a candidate against the oracle on
every case and reports the first failures together with the case index and kind.  A candidate that
merges only strict overlaps is therefore caught by the touching cases, and one that skips the sort by
the unsorted case - which is exactly what the edge policy is for.

Verified: generator determinism and seed sensitivity, the declared size/coordinate invariants, edge
coverage, the correct candidate against the oracle, both buggy candidates being caught on the right
case kind, agreement between the interval oracle and an integer point-set oracle, the canonical-form
invariant, and the empty/single boundaries.
"""
EDGE_KINDS = ("empty", "single", "disjoint", "nested", "touching", "duplicate", "unsorted", "random")
LOW, HIGH = -20, 20
MAX_CASES = 12


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seed, so a generated batch is reproducible."""

    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return z ^ (z >> 31)

    def ranged(self, low, high):
        return low + self.next() % (high - low + 1)

    def interval(self):
        start = self.ranged(LOW, HIGH - 1)
        return (start, self.ranged(start + 1, HIGH))


def make_case(rng, kind):
    """One case of the requested kind; every interval satisfies start <= end and the span."""
    if kind == "empty":
        return ()
    if kind == "single":
        return (rng.interval(),)
    if kind == "disjoint":
        first, second = (LOW, LOW + 3), (LOW + 6, LOW + 9)
        return (first, second) if rng.ranged(0, 1) == 0 else (second, first)
    if kind == "nested":
        outer = (LOW, HIGH)
        middle = rng.interval()
        return (outer, middle, (middle[0] + 1, middle[1]))
    if kind == "touching":
        start = rng.ranged(LOW, 0)
        return ((start, start + 2), (start + 2, start + 4))
    if kind == "duplicate":
        item = rng.interval()
        return (item, item, (item[0], item[1]))
    if kind == "unsorted":
        return ((5, 7), (1, 3), (2, 4))
    return tuple(rng.interval() for _ in range(rng.ranged(1, 6)))


def batch(seed, count=len(EDGE_KINDS) * 2):
    """A reproducible batch: each edge kind first, then random cases, all size-bounded."""
    rng = Rng(seed)
    return [(kind, make_case(rng, kind)) for kind in (EDGE_KINDS * 2)[:count]]


def merge_oracle(intervals):
    """Independent oracle: merge any overlapping or touching pair until nothing changes."""
    work = [list(item) for item in intervals]
    changed = True
    while changed:
        changed = False
        for i in range(len(work)):
            for j in range(len(work)):
                if i != j and work[i][0] <= work[j][1] and work[j][0] <= work[i][1]:
                    merged = [min(work[i][0], work[j][0]), max(work[i][1], work[j][1])]
                    work = [row for index, row in enumerate(work) if index not in (i, j)] + [merged]
                    changed = True
                    break
            if changed:
                break
    return sorted(tuple(row) for row in work)


def merge_candidate(intervals):
    """The submitted solution: sort by start, then extend while the next start is reached."""
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def merge_buggy_strict(intervals):
    """Bug: only strictly overlapping intervals are merged, so touching ones stay apart."""
    result = []
    for start, end in sorted(intervals):
        if result and start < result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def merge_buggy_unsorted(intervals):
    """Bug: the intervals are never sorted before the single scan."""
    result = []
    for start, end in intervals:
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def coverage_points(intervals):
    """Point-set oracle: the integer positions covered, as an unordered set."""
    points = set()
    for start, end in intervals:
        points.update(range(start, end + 1))
    return points


def checker(candidate, cases):
    """Run a candidate against the oracle; report failures with their case index and kind."""
    failures, checked = [], 0
    for index, (kind, intervals) in enumerate(cases):
        expected = merge_oracle(intervals)
        produced = [tuple(item) for item in candidate(intervals)]
        checked += 1
        if produced != expected:
            failures.append({"case": index, "kind": kind, "input": intervals,
                             "expected": expected, "produced": produced})
    return {"ok": not failures, "failures": failures, "checked": checked}


def canonical(intervals):
    """A merged result must be sorted and internally disjoint (touching pairs merged)."""
    return (all(intervals[i][0] <= intervals[i][1] for i in range(len(intervals)))
            and all(intervals[i][1] < intervals[i + 1][0] for i in range(len(intervals) - 1)))


def main():
    cases = batch(20240517)
    kinds = [kind for kind, _ in cases]

    # 1) determinism: the same seed reproduces the batch, another seed does not.
    other = batch(20240517 + 1)
    check(batch(20240517) == cases and other != cases and len(cases) == 16
          and kinds[:8] == list(EDGE_KINDS),
          f"the generator must be reproducible and seed-sensitive: {len(cases)} cases")

    # 2) declared invariants: every case is size-bounded and every interval has start <= end.
    within_span = all(LOW <= start <= end <= HIGH for _, intervals in cases for start, end in
                      intervals)
    bounded = all(len(intervals) <= 6 for _, intervals in cases)
    check(within_span and bounded and len(cases) <= MAX_CASES * 2,
          "generated intervals must be span-bounded and non-inverted")

    # 3) the edge policy is actually covered by the batch.
    check(set(kinds) == set(EDGE_KINDS) and kinds.count("touching") == 2
          and any(intervals == () for kind, intervals in cases if kind == "empty"),
          f"edge coverage incomplete: {sorted(set(kinds))}")

    # 4) happy path: the submitted solution passes every case in the batch.
    verdict = checker(merge_candidate, cases)
    check(verdict["ok"] and verdict["checked"] == len(cases),
          f"the correct candidate must pass: {verdict['failures'][:1]}")

    # 5) the touching-interval bug is caught, and the first case it fails is a touching case.
    touching_index = kinds.index("touching")
    strict = checker(merge_buggy_strict, cases)
    failing_kinds = {failure["kind"] for failure in strict["failures"]}
    check(not strict["ok"] and "touching" in failing_kinds
          and strict["failures"][0]["case"] == touching_index
          and strict["failures"][0]["kind"] == "touching",
          f"the strict-overlap bug must be caught on touching cases: {failing_kinds}")

    # 6) the unsorted bug is caught, and the unsorted case is among the failures.
    unsorted = checker(merge_buggy_unsorted, cases)
    check(not unsorted["ok"] and any(failure["kind"] == "unsorted"
                                     for failure in unsorted["failures"])
          and len(unsorted["failures"]) >= 1,
          f"the missing-sort bug must be caught: {len(unsorted['failures'])} failures")

    # 7) independent oracle #2: the interval oracle agrees with the integer point-set oracle, and
    #    every merged result is in canonical form.
    mismatches = [index for index, (_, intervals) in enumerate(cases)
                  if coverage_points(merge_oracle(intervals)) != coverage_points(intervals)
                  or not canonical(merge_oracle(intervals))
                  or coverage_points(merge_candidate(intervals)) != coverage_points(intervals)]
    check(not mismatches,
          f"the two oracles must agree on coverage and canonical form: {mismatches}")

    # 8) boundaries: the empty case merges to nothing; a single interval is returned unchanged;
    #    a duplicate set collapses to one interval.
    empty_case = next(intervals for kind, intervals in cases if kind == "empty")
    single_case = next(intervals for kind, intervals in cases if kind == "single")
    duplicate_case = next(intervals for kind, intervals in cases if kind == "duplicate")
    check(merge_oracle(empty_case) == [] and merge_candidate(empty_case) == []
          and merge_oracle(single_case) == list(single_case)
          and merge_candidate(single_case) == list(single_case)
          and merge_oracle(duplicate_case) == [duplicate_case[0]]
          and merge_candidate(duplicate_case) == [duplicate_case[0]],
          f"empty/single/duplicate boundaries wrong: {merge_candidate(duplicate_case)}")

    # 9) purity: checking twice reports the same JSON-able structure, and the batch is not mutated
    #    by any candidate (buggy or correct).
    snapshot = list(cases)
    checker(merge_buggy_strict, cases)
    checker(merge_candidate, cases)
    check(cases == snapshot and checker(merge_candidate, cases) == verdict,
          "checking must not mutate the batch and must be repeatable")

    print(f"cases={len(cases)} kinds={sorted(set(kinds))} touching_failures="
          f"{len(strict['failures'])} unsorted_failures={len(unsorted['failures'])}")
    print(f"correct_verdict={'ok' if verdict['ok'] else 'failed'} "
          f"first_failure={strict['failures'][0]['input']} -> {strict['failures'][0]['produced']}")
    print("PASS")


if __name__ == "__main__":
    main()
