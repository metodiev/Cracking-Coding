"""Task 952 - Implement autograder that runs tests, measures complexity, and reports regression.
(Tier: Master, CTCI Ch 11 - Testing)

Grader model: a submission is a callable (input, step counter) -> answer.  Every case runs in
isolation (a crash is recorded, never aborted on); the complexity class is fitted by least squares
over the measured (n, steps) curve; the step count at a reference size is compared with a recorded
baseline to flag regressions.  Correctness and complexity are graded independently, so a
correct-but-quadratic solution passes every test and is still O(n^2).  Simplified versus a real
autograder: step counts stand in for timing and inputs come from a hand-written splitmix64.
Verified: per-case verdicts, exception isolation, outcome conservation, labels for
logarithmic/linear/quadratic submissions (cross-checked against doubling ratios), the brute-force
subarray oracle, regression labelling and the empty/single-case boundaries.
"""
LENGTHS = (8, 16, 32, 64, 128)
CLASSES = (
    ("O(1)", lambda n: 1.0),
    ("O(log n)", lambda n: max(1.0, n.bit_length() - 1)),
    ("O(n)", lambda n: float(n)),
    ("O(n log n)", lambda n: n * max(1.0, n.bit_length() - 1)),
    ("O(n^2)", lambda n: float(n) * n),
)
FIT_TOLERANCE, REGRESSION_RATIO = 0.15, 1.2


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seed, so the generated test inputs are reproducible."""

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


class Steps:
    """Operation counter handed to a submission."""
    count = 0

    def add(self, units=1):
        self.count += units


# The exercise (maximum subarray sum) and four submissions.
def kadane(values, steps):
    best = current = values[0]
    for value in values[1:]:
        steps.add(2)
        current = value if current < 0 else current + value
        best = max(best, current)
    return best


def quadratic(values, steps):
    best = values[0]
    for i in range(len(values)):
        total = 0
        for j in range(i, len(values)):
            steps.add(1)
            total += values[j]
            best = max(best, total)
    return best


def buggy_best_zero(values, steps):
    """Classic bug: the running best is seeded with 0, so an all-negative array returns 0."""
    best = current = 0
    for value in values:
        steps.add(2)
        current = max(0, current + value)
        best = max(best, current)
    return best


def crashing(values, steps):
    raise RuntimeError("student left a stub")

def brute_force(values, _steps=None):
    """Independent oracle: enumerate every subarray (empty input raises)."""
    return max(sum(values[i:j])
               for i in range(len(values)) for j in range(i + 1, len(values) + 1))


def search_first(sorted_values, steps):
    """Binary search for the smallest element - a logarithmic probe count."""
    target, low, high = sorted_values[0], 0, len(sorted_values) - 1
    while low <= high:
        steps.add(1)
        middle = (low + high) // 2
        if sorted_values[middle] == target:
            return middle
        if sorted_values[middle] < target:
            low = middle + 1
        else:
            high = middle - 1
    return -1


# Generation, execution and the grader.
def generated_cases(seed, sizes):
    """Random cases with one positive element forced in, so only the curated cases are edge."""
    rng = Rng(seed)
    cases = []
    for n in sizes:
        values = [rng.ranged(-9, 9) for _ in range(n)]
        values[rng.ranged(0, n - 1)] = rng.ranged(1, 9)
        cases.append((values, brute_force(values)))
    return cases


def run_tests(submission, tests):
    """One outcome row per case, in order; a crash is recorded, not propagated."""
    outcomes = []
    for index, (values, expected) in enumerate(tests):
        steps = Steps()
        try:
            produced = submission(values, steps)
            ok = produced == expected
            reason = None if ok else f"wrong answer {produced!r} != {expected!r}"
        except Exception as error:
            ok, reason = False, f"{type(error).__name__}: {error}"
        outcomes.append({"case": index, "ok": ok, "steps": steps.count, "reason": reason})
    return outcomes


def measure_curve(submission, sizes, build):
    """(n, steps) for one input per size."""
    curve = []
    for n in sizes:
        steps = Steps()
        submission(build(n), steps)
        curve.append((n, steps.count))
    return curve


def fit_complexity(curve):
    """Least-squares scale per model; the smallest worst-case relative residual wins."""
    best = None
    for label, model in CLASSES:
        shape = [model(n) for n, _ in curve]
        scale = sum(steps * value for (_, steps), value in zip(curve, shape)) / sum(
            value * value for value in shape)
        if scale <= 0:
            continue
        error = max(abs(steps - scale * value) / (scale * value)
                    for (_, steps), value in zip(curve, shape))
        if best is None or error < best[1] - 1e-12:
            best = (label, error, scale)
    if best is None:
        raise ValueError("cannot fit a degenerate curve")
    return best


def grade(submission, tests, curve, baseline=None, reference_n=None):
    """A verdict: correctness score, complexity label and - optionally - a regression flag."""
    outcomes = run_tests(submission, tests)
    passed = sum(1 for row in outcomes if row["ok"])
    label, error, _ = fit_complexity(curve)
    verdict = {"outcomes": outcomes, "passed": passed, "total": len(outcomes),
               "score": round(passed / len(outcomes), 2) if outcomes else 0.0,
               "status": "graded" if outcomes else "no_tests",
               "complexity": label, "fit_error": error, "regressed": False, "ratio": 1.0}
    if baseline is not None:
        before = dict(baseline[submission.__name__][1])[reference_n]
        ratio = dict(curve)[reference_n] / before if before else float("inf")
        verdict["regressed"], verdict["ratio"] = ratio > REGRESSION_RATIO, ratio
    return verdict


def main():
    seed = 20240517
    # The suite: three generated cases plus curated ones - single element, two all-negative.
    tests = generated_cases(seed, (1, 5, 9)) + [([-5], -5), ([-4, -3, -2], -2), ([3], 3)]
    check(len(tests) == 6 and all(brute_force(values) == expected for values, expected in tests),
          "every curated and generated case must agree with the brute-force oracle")
    # 1) happy path: both correct submissions score 1.0.
    correct = {fn.__name__: grade(fn, tests, measure_curve(fn, LENGTHS, lambda n: [1] * n))
               for fn in (kadane, quadratic)}
    check(all(v["passed"] == 6 and v["score"] == 1.0 and v["status"] == "graded"
              for v in correct.values()),
          f"correct submissions must score 1.0: {[v['score'] for v in correct.values()]}")

    # 2) the seed-with-zero bug is caught, and only on the two all-negative cases.
    buggy = grade(buggy_best_zero, tests,
                  measure_curve(buggy_best_zero, LENGTHS, lambda n: [-1] * n))
    failed = [row["case"] for row in buggy["outcomes"] if not row["ok"]]
    check(failed == [3, 4] and buggy["score"] == round(4 / 6, 2)
          and all(buggy["outcomes"][i]["reason"].startswith("wrong answer") for i in failed),
          f"only the all-negative cases may fail the buggy submission: {failed}")
    # 3) exception isolation: a crash is reported per case and the batch completes intact.
    crashed = grade(crashing, tests, [(8, 8)])
    check(crashed["passed"] == 0 and crashed["total"] == len(tests)
          and all(row["reason"].startswith("RuntimeError") for row in crashed["outcomes"])
          and [row["case"] for row in crashed["outcomes"]] == list(range(len(tests))),
          "a crashing submission must be reported without aborting the batch")

    # 4) complexity fit versus the doubling-ratio oracle, for log/linear/quadratic submissions.
    curves = {"kadane": measure_curve(kadane, LENGTHS, lambda n: [1] * n),
              "quadratic": measure_curve(quadratic, (8, 16, 32, 64), lambda n: [1] * n),
              "search": measure_curve(search_first, (16, 32, 64), lambda n: list(range(n)))}
    fits = {name: fit_complexity(curve) for name, curve in curves.items()}
    labels = {name: fit[0] for name, fit in fits.items()}
    linear_ratio = dict(curves["kadane"])[128] / dict(curves["kadane"])[64]
    quad_ratio = dict(curves["quadratic"])[64] / dict(curves["quadratic"])[32]
    check(labels == {"kadane": "O(n)", "quadratic": "O(n^2)", "search": "O(log n)"}
          and max(fit[1] for fit in fits.values()) < FIT_TOLERANCE
          and abs(linear_ratio - 2.0) < 0.05 and abs(quad_ratio - 4.0) < 0.3,
          f"labels {labels} ratios {linear_ratio:.2f}/{quad_ratio:.2f}")

    # 5) regression reporting: a 3x step increase is flagged, an unchanged curve is not.
    baseline = {name: (LENGTHS, dict(curve)) for name, curve in curves.items()}
    flagged = grade(kadane, tests, sorted((n, s * 3) for n, s in curves["kadane"]), baseline, 64)
    steady = grade(quadratic, tests, curves["quadratic"], baseline, 64)
    check(flagged["regressed"] and flagged["ratio"] == 3.0 and flagged["score"] == 1.0
          and not steady["regressed"] and steady["ratio"] == 1.0,
          f"only the slowed curve may regress: {flagged['ratio']} / {steady['ratio']}")
    # 6) boundaries: an empty suite grades nothing instead of crashing; one case is enough.
    empty = grade(kadane, [], curves["kadane"])
    single = grade(kadane, [([2, -1, 2], 3)], curves["kadane"])
    check(empty["status"] == "no_tests" and empty["score"] == 0.0 and empty["outcomes"] == []
          and single["status"] == "graded" and single["score"] == 1.0,
          f"empty and single-case suites must behave: {empty['status']}, {single['score']}")

    # 7) determinism, plus a second oracle sweep over unseen random inputs.
    repeat = grade(buggy_best_zero, tests,
                   measure_curve(buggy_best_zero, LENGTHS, lambda n: [-1] * n))
    disagreements = [values for values, _ in generated_cases(seed + 7, (2, 4, 8, 16))
                     if kadane(values, Steps()) != brute_force(values)]
    check(repeat == buggy and not disagreements,
          f"re-grading must be deterministic and Kadane must match the oracle ({disagreements})")

    print(f"cases={len(tests)} buggy_score={buggy['score']} failed_cases={failed} labels={labels}")
    print(f"fit_errors={[round(f[1], 4) for f in fits.values()]} ratios={linear_ratio:.2f}/"
          f"{quad_ratio:.2f} regressed={flagged['regressed']}")
    print("PASS")


if __name__ == "__main__":
    main()
