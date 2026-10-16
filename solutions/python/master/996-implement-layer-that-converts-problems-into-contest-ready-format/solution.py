"""Task 996 - Implement layer that converts problems into contest-ready format (time limits, scoring).
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Turns an editorial problem into a contest package.  A time limit is derived from the declared
complexity class and the maximum input size: the class fixes how many elementary operations the
worst case performs, the reference machine price per 100k operations and a safety factor of 3
turn that into milliseconds, and the result is clamped into the platform window so neither a
trivial nor a quadratic problem gets an unusable limit.  The 100 points of a problem are
apportioned over its subtasks with the largest-remainder method, so subtask points always sum
to exactly 100 while staying within one point of their exact quota.

Simplified versus production: timings are a closed-form cost model on a nominal reference
machine instead of measured runs, and a contest holds at most 26 problems (short names A-Z).
Verified: score conservation and the largest-remainder quota property, monotonicity of the
limit in complexity and input size with both clamps, byte-identical reproducibility under
permuted input, package validation rejections, and the single-problem/single-subtask boundary.
"""
import hashlib
import json
import math

UNIT_MS = 10               # milliseconds the reference machine needs per 100k operations
SAFETY = 3                 # headroom over the nominal worst case
MIN_LIMIT_MS = 1000
MAX_LIMIT_MS = 10000
TOTAL_POINTS = 100
COMPLEXITY = ("O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)")
SHORT_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class ContestFormatError(ValueError):
    """Raised when a problem cannot be turned into a fair contest package."""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle):
    try:
        action()
    except ContestFormatError as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected a ContestFormatError mentioning %r" % needle)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_of(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def operations(complexity, n):
    """Worst-case elementary operations of the declared class at input size n."""
    if complexity not in COMPLEXITY:
        raise ContestFormatError("unknown complexity class %r" % (complexity,))
    if n <= 0:
        raise ContestFormatError("max input size must be positive, got %r" % (n,))
    if complexity == "O(1)":
        return 1
    if complexity == "O(log n)":
        return max(1, int(math.log2(max(n, 2))))
    if complexity == "O(n)":
        return n
    if complexity == "O(n log n)":
        return int(n * max(1.0, math.log2(max(n, 2))))
    return n * n


def time_limit_ms(complexity, n):
    """Clamped limit that never punishes a fast problem and never flatters a slow one."""
    raw = SAFETY * UNIT_MS * operations(complexity, n) / 100_000
    return int(min(MAX_LIMIT_MS, max(MIN_LIMIT_MS, math.ceil(raw))))


def apportion(weights, total=TOTAL_POINTS):
    """Largest-remainder (Hamilton) apportionment: exact sum, quota error below one point."""
    if not weights:
        raise ContestFormatError("a problem needs at least one subtask")
    if any(weight <= 0 for weight in weights):
        raise ContestFormatError("subtask weights must be positive")
    exact = [weight * total / sum(weights) for weight in weights]
    points = [int(value) for value in exact]
    leftover = total - sum(points)
    order = sorted(range(len(weights)), key=lambda index: (-(exact[index] - points[index]), index))
    for index in order[:leftover]:
        points[index] += 1
    return points


def validate(problem, seen_ids):
    if not problem.get("id"):
        raise ContestFormatError("problem without an id")
    if problem["id"] in seen_ids:
        raise ContestFormatError("duplicate problem id %r" % (problem["id"],))
    if not 1 <= problem.get("difficulty", 0) <= 5:
        raise ContestFormatError("problem %r has difficulty outside 1..5" % (problem["id"],))
    if problem.get("complexity") not in COMPLEXITY:
        raise ContestFormatError("problem %r declares an unknown complexity class" % (problem["id"],))
    operations(problem["complexity"], problem.get("max_n", 0))
    subtasks = problem.get("subtasks") or []
    if not subtasks:
        raise ContestFormatError("problem %r needs at least one subtask" % (problem["id"],))
    names = set()
    for subtask in subtasks:
        if not subtask.get("id") or subtask["id"] in names:
            raise ContestFormatError("problem %r has a duplicate/empty subtask id" % (problem["id"],))
        names.add(subtask["id"])
        operations(problem["complexity"], subtask.get("max_n", 0))
    return problem["id"]


def build_package(problems, contest_id="open-practice-round"):
    if len(problems) > len(SHORT_NAMES):
        raise ContestFormatError("a round may hold at most %d problems" % len(SHORT_NAMES))
    seen, ordered = set(), []
    for problem in problems:
        validate(problem, seen)
        seen.add(problem["id"])
        ordered.append(problem)
    ordered.sort(key=lambda problem: (problem["difficulty"], problem["id"]))
    entries = []
    for position, problem in enumerate(ordered):
        weights = [subtask["weight"] for subtask in problem["subtasks"]]
        points = apportion(weights)
        entries.append({"id": problem["id"], "short_name": SHORT_NAMES[position],
                        "difficulty": problem["difficulty"],
                        "time_limit_ms": time_limit_ms(problem["complexity"], problem["max_n"]),
                        "points": TOTAL_POINTS,
                        "subtasks": [{"id": subtask["id"], "points": points[index],
                                      "max_n": subtask["max_n"]}
                                     for index, subtask in enumerate(problem["subtasks"])]})
    return {"contest_id": contest_id, "problems": entries,
            "total_points": sum(entry["points"] for entry in entries)}


def quota_violations(entry, weights):
    """Independent characterisation: every subtask point count must round its exact quota."""
    exact = [weight * TOTAL_POINTS / sum(weights) for weight in weights]
    return not all(points == int(value) or points == int(value) + 1
                   for points, value in zip([sub["points"] for sub in entry["subtasks"]], exact))


def problem(pid, difficulty, complexity, max_n, subtasks):
    return {"id": pid, "difficulty": difficulty, "complexity": complexity, "max_n": max_n,
            "subtasks": subtasks}


def sample_problems():
    return [problem("dijkstra", 4, "O(n log n)", 1_000_000,
                    [{"id": "small", "weight": 1, "max_n": 100},
                     {"id": "medium", "weight": 3, "max_n": 10_000},
                     {"id": "full", "weight": 6, "max_n": 1_000_000}]),
            problem("two-sum", 1, "O(n)", 1_000_000,
                    [{"id": "n-squared", "weight": 2, "max_n": 5_000},
                     {"id": "linear", "weight": 8, "max_n": 1_000_000}]),
            problem("matrix-multiply", 5, "O(n^2)", 2_000,
                    [{"id": "tiny", "weight": 1, "max_n": 50},
                     {"id": "full", "weight": 2, "max_n": 2_000}])]


def main():
    package = build_package(sample_problems())
    entries = {entry["id"]: entry for entry in package["problems"]}

    # Happy path: difficulty ordering fixes the short names, and every limit stays in the window.
    check([entry["id"] for entry in package["problems"]] == ["two-sum", "dijkstra", "matrix-multiply"],
          "problems must be ordered by difficulty then id")
    check([entry["short_name"] for entry in package["problems"]] == ["A", "B", "C"],
          "short names must follow the contest order")
    check(all(MIN_LIMIT_MS <= entry["time_limit_ms"] <= MAX_LIMIT_MS
              for entry in package["problems"]), "every limit must sit inside the platform window")
    check(package["total_points"] == 300 and all(entry["points"] == TOTAL_POINTS
                                                for entry in package["problems"]),
          "each problem is worth exactly 100 points")

    # Score conservation plus the independent largest-remainder quota characterisation.
    weights = {"dijkstra": [1, 3, 6], "two-sum": [2, 8], "matrix-multiply": [1, 2]}
    for pid, entry in entries.items():
        points = [subtask["points"] for subtask in entry["subtasks"]]
        check(sum(points) == TOTAL_POINTS, "%s subtask points must sum to 100" % pid)
        check(not quota_violations(entry, weights[pid]),
              "%s points must each be floor or ceil of their exact quota" % pid)
        check(all(point >= 1 for point in points), "%s must not starve a subtask" % pid)
    check([subtask["points"] for subtask in entries["two-sum"]["subtasks"]] == [20, 80],
          "the 2:8 weighting must apportion 20/80 exactly")
    check([subtask["points"] for subtask in entries["dijkstra"]["subtasks"]] == [10, 30, 60],
          "the 1:3:6 weighting must apportion 10/30/60 exactly")
    awkward = apportion([1, 1, 1])
    check(sum(awkward) == 100 and sorted(awkward) == [33, 33, 34],
          "an indivisible 1:1:1 split must still total 100 via largest remainder")

    # Time limits: monotone in complexity and in input size, with both clamps honoured.
    limits = [time_limit_ms(complexity, 1_000_000) for complexity in COMPLEXITY]
    check(limits == sorted(limits), "a heavier class must never get a tighter limit: %r" % (limits,))
    check(time_limit_ms("O(n)", 1_000) <= time_limit_ms("O(n)", 1_000_000),
          "a larger input size must not reduce the limit")
    check(time_limit_ms("O(n)", 10) == MIN_LIMIT_MS and time_limit_ms("O(n^2)", 1_000_000) == MAX_LIMIT_MS,
          "the small and huge cases must clamp to the platform window")
    expected_ms = math.ceil(SAFETY * UNIT_MS * int(1_000_000 * math.log2(1_000_000)) / 100_000)
    check(time_limit_ms("O(n log n)", 1_000_000) == expected_ms == 5980,
          "O(n log n) at 10^6 must match the closed-form cost model")

    # Reproducibility: permuting the problems cannot change the package.
    shuffled = build_package(list(reversed(sample_problems())))
    check(shuffled == package, "the package must not depend on the input order")
    check(digest_of(shuffled) == digest_of(package), "the package digest must be reproducible")
    check(digest_of(build_package(sample_problems(), "other-round")) != digest_of(package),
          "a different contest id must change the digest")

    # Validation rejections.
    bad_difficulty = sample_problems()
    bad_difficulty[0]["difficulty"] = 9
    expect_error(lambda: build_package(bad_difficulty), "difficulty outside")
    unknown_class = sample_problems()
    unknown_class[0]["complexity"] = "O(n!)"
    expect_error(lambda: build_package(unknown_class), "unknown complexity")
    duplicate_subtask = sample_problems()
    duplicate_subtask[0]["subtasks"][1]["id"] = "small"
    expect_error(lambda: build_package(duplicate_subtask), "duplicate/empty subtask")
    empty_subtasks = sample_problems()
    empty_subtasks[1]["subtasks"] = []
    expect_error(lambda: build_package(empty_subtasks), "at least one subtask")
    duplicated = sample_problems() + [sample_problems()[0]]
    expect_error(lambda: build_package(duplicated), "duplicate problem id")
    too_many = [problem("p%02d" % i, 3, "O(n)", 100, [{"id": "s", "weight": 1, "max_n": 100}])
                for i in range(27)]
    expect_error(lambda: build_package(too_many), "at most 26")
    expect_error(lambda: apportion([0, 0]), "weights must be positive")

    # Boundary: one problem with one subtask takes all 100 points and short name A.
    solo = build_package([problem("solo", 1, "O(1)", 10,
                                  [{"id": "only", "weight": 1, "max_n": 10}])])
    check(solo["problems"][0]["short_name"] == "A"
          and solo["problems"][0]["subtasks"][0]["points"] == TOTAL_POINTS,
          "a lone subtask must receive the whole problem score")
    check(len(build_package([])["problems"]) == 0, "an empty round must build an empty package")

    print("problems=%d total_points=%d limits=%s digest=%s"
          % (len(package["problems"]), package["total_points"],
             [entry["time_limit_ms"] for entry in package["problems"]], digest_of(package)[:16]))
    print("PASS")


if __name__ == "__main__":
    main()
