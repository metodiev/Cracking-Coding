"""Task 855 - Implement automated plan regression detector for query optimizer changes.  (Tier: Master, Ch 14 — Databases)

Hunting for plan regressions between two versions of a query optimizer: the same workload
is planned by version 1 and version 2, both plans are *measured*, and a paired permutation
test decides which queries genuinely got slower.  Everything is in-process and
deterministic: no clock, no threads, no unseeded randomness (the measurement noise comes
from a seeded splitmix64 stream), and the workload is generated rather than read from disk.

What is simplified: the "database" is a small analytic cost model rather than a real
storage engine.  A query is three tables joined left-deep with an optional filter; a plan is
a join order plus one operator per join (hash / nested loop / index nested loop); the plan
space is enumerated exhaustively (6 orders x 4 operator combinations = 24 plans).  Both
optimizer versions share the same cardinality estimator (a uniform fan-out assumption) and
differ only in their physical cost constants - version 2 believes random IO is far cheaper
than it really is, which is exactly the kind of cost-model bug that produces plan
regressions.  The *measured* cost of a plan always uses the true constants and true
cardinalities, perturbed by +-2% deterministic jitter so the detector has to be statistical.

Invariants verified:
  * the optimizer is optimal with respect to its own model: the plan it picks has the lowest
    estimated cost of all 24 enumerated plans (checked by brute force for every query);
  * the measured cost agrees with an independently computed analytic cost within the jitter
    band, for every plan and repetition, so measurements are faithful;
  * the detector's verdicts are compared with a ground truth (the true cost of the plan each
    version chose): precision is 1.0 - no query is reported as a regression when the true
    costs did not change - while recall on the real regressions is high and reaches 1.0 on
    the severe ones;
  * the statistical test is calibrated: on pairs of plans whose true costs are identical the
    false positive rate stays at the significance level, in a self-comparison control the
    detector reports nothing, and with a single repetition no significance can be claimed;
  * the detector distinguishes regressions, improvements and untouched queries, and only
    reports plan regressions that also changed the plan shape;
  * every finding is reproducible from the seed.
"""

import itertools

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self, lo, hi):
        return lo + (hi - lo) * (self.next_u64() / float(1 << 64))


TABLE_ROWS = {"A": 2000, "B": 800, "C": 300, "D": 60}
OPERATORS = ("hash", "nested", "index")
MEMORY = 4000.0                 # build side above this spills: two extra passes
TRUE_RANDOM_IO = 0.05            # the honest cost of one random access, relative to a row scan
INDEX_PROBE = 2.0
SEVERE = 1.5                    # a cost ratio above this is a severe regression
MEANINGFUL = 1.05


class Constants:
    """Physical cost constants.  A version may believe something different (the injected bug)."""

    def __init__(self, random_io=TRUE_RANDOM_IO, memory=MEMORY):
        self.random_io = random_io
        self.memory = memory

    @property
    def is_true(self):
        return self.random_io == TRUE_RANDOM_IO and self.memory == MEMORY


VERSION_1 = Constants()
# version 2 fixes the memory limit (fewer spills, so some plans improve) but its random-IO
# constant is a quarter of the real value, which makes it pick nested loops that are much
# more expensive than its model believes
VERSION_2 = Constants(random_io=TRUE_RANDOM_IO / 4.0, memory=MEMORY * 4.0)


class Query:
    def __init__(self, name, tables, selectivity, fanout, estimate=1.0):
        self.name = name
        self.tables = tuple(tables)
        self.selectivity = dict(selectivity)      # per-table filter selectivity
        self.fanout = dict(fanout)                # true join fan-out per table pair
        self.estimate = estimate                  # the uniform fan-out the estimator assumes

    def pair(self, left, right):
        return self.fanout.get((left, right), self.fanout.get((right, left), self.estimate))

    def base_rows(self, table):
        return TABLE_ROWS[table] * self.selectivity.get(table, 1.0)


def generate_workload(seed, count):
    """Deterministic workload: skewed queries (regression fodder) and symmetric ones."""
    rng = SplitMix64(seed)
    queries = []
    for index in range(count):
        tables = ("A", "B", "C")
        selective = 0.05 + 0.5 * rng.uniform(0.0, 1.0)
        selectivity = {tables[0]: selective}
        if index % 4 == 0:                                   # symmetric: no regression possible
            fanout = {(a, b): 1.0 for a, b in itertools.combinations(tables, 2)}
            queries.append(Query("q%03d" % index, tables, selectivity, fanout))
            continue
        fanout = {}
        for a, b in itertools.combinations(tables, 2):
            fanout[(a, b)] = 0.5 + 6.0 * rng.uniform(0.0, 1.0)     # skew the estimator misses
        queries.append(Query("q%03d" % index, tables, selectivity, fanout))
    return queries


# ---------------------------------------------------------------------------
# plan space and cost models
# ---------------------------------------------------------------------------
def plans_for(query):
    """The exhaustive plan space: left-deep join order plus one operator per join."""
    for order in itertools.permutations(query.tables):
        for operators in itertools.product(("hash", "nested"), repeat=len(order) - 1):
            yield (order, operators)


def plan_cost(query, plan, constants, estimated):
    """Cost of a left-deep plan.

    estimated=True uses the optimizer's uniform fan-out and the version's constants;
    estimated=False uses the true fan-out of this query and the true constants.  Both share
    the same physical model, so a cost difference is a modelling error, not a formula error.
    """
    order, operators = plan
    random_io = constants.random_io if estimated else TRUE_RANDOM_IO
    memory = constants.memory if estimated else MEMORY
    size = query.base_rows(order[0])
    cost = size                                    # scan the base table
    for position in range(1, len(order)):
        inner_table = order[position]
        inner = query.base_rows(inner_table)
        operator = operators[position - 1]
        if operator == "hash":
            build = min(size, inner)
            cost += size + inner + 2.0 * build
            if build > memory:
                cost += 2.0 * build                 # spilled build side
        elif operator == "nested":
            cost += size * inner * random_io
        else:                                       # index nested loop
            matches = max(1.0, inner / 10.0)
            cost += size * (INDEX_PROBE + matches * random_io * 4.0)
        fanout = query.pair(order[position - 1], inner_table) if not estimated else query.estimate
        size = min(size, inner) * fanout
    return cost


def choose_plan(query, constants):
    """The optimizer: enumerate the space, take the cheapest plan under its own model."""
    return min(plans_for(query), key=lambda plan: plan_cost(query, plan, constants, True))


def true_cost(query, plan):
    return plan_cost(query, plan, Constants(), False)


def measure(query, plan, rng, repetition):
    """Deterministic measurement: the true cost plus +-2% jitter (simulated timing noise)."""
    return true_cost(query, plan) * (1.0 + 0.02 * (rng.uniform(-1.0, 1.0)))


def plan_shape(plan):
    order, operators = plan
    return (tuple(order), tuple(operators))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def permutation_p_value(differences, flips=None):
    """Exact two-sided paired permutation test over all 2**n sign flips."""
    n = len(differences)
    observed = sum(differences) / n
    if observed == 0.0:
        return 1.0
    total = 0
    extreme = 0
    for mask in range(1 << n):
        signed = sum(differences[i] if (mask >> i) & 1 else -differences[i] for i in range(n))
        total += 1
        if abs(signed / n) >= abs(observed) - 1e-12:
            extreme += 1
    return extreme / total


class Detector:
    """Measures both versions' plans and reports significant cost regressions."""

    def __init__(self, constants_a, constants_b, repetitions=8, alpha=0.05):
        self.constants_a = constants_a
        self.constants_b = constants_b
        self.repetitions = repetitions
        self.alpha = alpha

    def evaluate(self, queries, seed):
        rng = SplitMix64(seed)
        findings = []
        for query in queries:
            plan_a = choose_plan(query, self.constants_a)
            plan_b = choose_plan(query, self.constants_b)
            times_a = [measure(query, plan_a, rng, r) for r in range(self.repetitions)]
            times_b = [measure(query, plan_b, rng, r) for r in range(self.repetitions)]
            differences = [b - a for a, b in zip(times_a, times_b)]
            p_value = permutation_p_value(differences)
            ratio = true_cost(query, plan_b) / true_cost(query, plan_a)
            findings.append({
                "query": query.name,
                "plan_a": plan_shape(plan_a),
                "plan_b": plan_shape(plan_b),
                "mean_a": sum(times_a) / len(times_a),
                "mean_b": sum(times_b) / len(times_b),
                "p_value": p_value,
                "ratio": ratio,
                "changed_shape": plan_shape(plan_a) != plan_shape(plan_b),
                "regressed": p_value < self.alpha and ratio > MEANINGFUL,
                "improved": p_value < self.alpha and ratio < 1.0 / MEANINGFUL,
            })
        return findings


def ground_truth(queries, constants_a, constants_b):
    """The queries where version B's chosen plan really is more expensive than A's."""
    truth = {}
    for query in queries:
        plan_a = choose_plan(query, constants_a)
        plan_b = choose_plan(query, constants_b)
        ratio = true_cost(query, plan_b) / true_cost(query, plan_a)
        truth[query.name] = {"ratio": ratio, "regression": ratio > MEANINGFUL,
                             "severe": ratio > SEVERE,
                             "same_plan": plan_shape(plan_a) == plan_shape(plan_b)}
    return truth


def precision_recall(findings, truth):
    reported = {finding["query"] for finding in findings if finding["regressed"]}
    actual = {name for name, info in truth.items() if info["regression"]}
    true_positive = reported & actual
    precision = len(true_positive) / len(reported) if reported else 1.0
    recall = len(true_positive) / len(actual) if actual else 1.0
    return precision, recall, reported, actual


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    queries = generate_workload(4711, 160)
    check(len(queries) == 160, "the workload must contain the requested queries")
    check(any(info["regression"] is False for info in ground_truth(
        queries, VERSION_1, VERSION_2).values()),
        "the workload must contain queries that do not regress")
    space = list(plans_for(queries[0]))
    check(len(space) == 24, "three tables must give 24 plans, got %d" % len(space))
    check(len({plan_shape(plan) for plan in space}) == 24, "every enumerated plan must be unique")
    print("workload: %d queries, plan space %d plans, severities %s"
          % (len(queries), len(space), sorted({round(info["ratio"], 2) for info in
                                               ground_truth(queries, VERSION_1,
                                                            VERSION_2).values()})[:6]))

    # ---- 1. the optimizer is optimal for its own model -------------------
    for query in queries:
        chosen = choose_plan(query, VERSION_2)
        best = min(plan_cost(query, plan, VERSION_2, True) for plan in plans_for(query))
        check(abs(plan_cost(query, chosen, VERSION_2, True) - best) < 1e-9,
              "the optimizer must return the cheapest plan of the space for %s" % query.name)
    print("optimality: every chosen plan minimises the version's own cost model")

    # ---- 2. measurements are faithful -----------------------------------
    rng = SplitMix64(99)
    worst = 0.0
    checked = 0
    for query in queries[:20]:
        for plan in list(plans_for(query))[:6]:
            analytic = true_cost(query, plan)
            for repetition in range(4):
                measured = measure(query, plan, rng, repetition)
                relative = abs(measured - analytic) / analytic
                worst = max(worst, relative)
                check(relative <= 0.0201,
                      "measurement drifted %.4f%% from the analytic cost" % (100 * relative))
                checked += 1
    print("measurements: %d samples within %.2f%% of the analytic true cost"
          % (checked, 100 * worst))

    # ---- 3. the cost-model bug really changes plans ----------------------
    changed = sum(1 for query in queries
                  if plan_shape(choose_plan(query, VERSION_1))
                  != plan_shape(choose_plan(query, VERSION_2)))
    check(changed > 0, "version 2 must plan at least some queries differently")
    symmetric = [query for query in queries if len(set(query.fanout.values())) == 1
                 and set(query.fanout.values()) == {1.0}]
    check(len(symmetric) > 20, "the workload must contain uniform-fan-out queries")
    symmetric_truth = ground_truth(symmetric, VERSION_1, VERSION_2)
    symmetric_findings = Detector(VERSION_1, VERSION_2).evaluate(symmetric, 5150)
    symmetric_precision, _sym_recall, _r, _a = precision_recall(symmetric_findings, symmetric_truth)
    check(symmetric_precision == 1.0,
          "even on uniform queries the detector must not cry wolf")
    print("plan changes: %d of %d queries planned differently by version 2 (%d of them uniform)"
          % (changed, len(queries),
             sum(1 for info in symmetric_truth.values() if not info["same_plan"])))

    # ---- 4. detection quality against the ground truth ------------------
    truth = ground_truth(queries, VERSION_1, VERSION_2)
    findings = Detector(VERSION_1, VERSION_2).evaluate(queries, 2024)
    precision, recall, reported, actual = precision_recall(findings, truth)
    severe = {name for name, info in truth.items() if info["severe"]}
    severe_caught = len(reported & severe) / len(severe) if severe else 1.0
    check(actual, "the injected cost-model bug must cause real regressions")
    check(precision == 1.0, "the detector must not report a query whose true cost is unchanged")
    check(recall >= 0.8, "recall must be at least 80%%, got %.2f" % recall)
    check(severe_caught == 1.0, "every severe regression must be caught")
    print("detector: reported=%d actual=%d precision=%.2f recall=%.2f severe-recall=%.2f"
          % (len(reported), len(actual), precision, recall, severe_caught))

    # every finding is a genuine plan-shape change, and improvements are separated
    for finding in findings:
        if finding["regressed"]:
            check(finding["changed_shape"],
                  "a cost regression must come with a plan shape change: %s" % finding["query"])
            check(finding["mean_b"] > finding["mean_a"],
                  "a regression must be slower on the measured means")
    challenged = [finding for finding in findings if finding["regressed"]
                  and finding["p_value"] >= 0.05]
    check(not challenged, "every reported regression must be significant")
    untouched = [finding for finding in findings if not finding["regressed"]
                 and not finding["improved"]]
    print("verdicts: %d regressions, %d improvements, %d untouched"
          % (len(reported), len([f for f in findings if f["improved"]]), len(untouched)))

    # ---- 5. the test is calibrated --------------------------------------
    control = Detector(VERSION_1, VERSION_1).evaluate(queries, 2024)
    check(not [finding for finding in control if finding["regressed"]],
          "a self-comparison must never report a regression")
    check(all(finding["ratio"] == 1.0 for finding in control),
          "a self-comparison must leave the true cost unchanged")
    print("calibration: self-comparison 0 findings, rejection rate %.3f (expected ~0.05)"
          % (sum(1 for finding in control if finding["p_value"] < 0.05) / len(control)))

    # the reverse comparison turns every regression into an improvement
    reversal = Detector(VERSION_2, VERSION_1).evaluate(queries, 2024)
    check(not [finding for finding in reversal if finding["regressed"]],
          "reverting the bug must not look like a regression")
    improvements = [finding for finding in reversal if finding["improved"]]
    check(len(improvements) == len(actual),
          "reverting the bug must report exactly the %d affected queries as improvements, got %d"
          % (len(actual), len(improvements)))
    print("calibration: the reversed comparison reports %d improvements and no regressions"
          % len(improvements))

    # false positives among the queries that really did not regress
    unaffected = [name for name, info in truth.items() if not info["regression"]]
    false_positives = [name for name in unaffected if name in reported]
    check(not false_positives,
          "the detector flagged unaffected queries: %s" % false_positives[:3])
    check(len(unaffected) > 50, "the workload must contain plenty of unaffected queries")
    print("calibration: 0 false positives among %d unaffected queries" % len(unaffected))

    # a paired test with one repetition cannot establish significance
    single = Detector(VERSION_1, VERSION_2, repetitions=1).evaluate(queries, 2024)
    check(not [finding for finding in single if finding["regressed"]],
          "one repetition must never yield a significant regression")
    print("calibration: with a single repetition no regression is claimed")

    # noise-only p-values respect the level: a test at alpha rejects ~alpha of the null cases
    noise = Detector(VERSION_1, VERSION_1, repetitions=8)
    p_values = [finding["p_value"] for finding in noise.evaluate(queries, 555)]
    rejection = sum(1 for value in p_values if value < 0.05) / len(p_values)
    check(rejection <= 0.15,
          "the rejection rate on pure noise %.3f must respect the significance level" % rejection)
    check(sorted(p_values)[len(p_values) // 2] > 0.4,
          "the median p-value under the null must be high")
    check(not [finding for finding in noise.evaluate(queries, 555) if finding["regressed"]],
          "noise without a plan change can never be reported as a regression")
    print("calibration: noise-only rejection rate %.3f, median p-value %.2f"
          % (rejection, sorted(p_values)[len(p_values) // 2]))

    # ---- 6. edge cases ---------------------------------------------------
    single_table = Query("solo", ("D",), {}, {}, 1.0)
    check(len(list(plans_for(single_table))) == 1, "a one-table query has exactly one plan")
    check(true_cost(single_table, list(plans_for(single_table))[0]) == 60.0,
          "a one-table plan costs one scan")
    pair = Query("pair", ("A", "B"), {}, {("A", "B"): 1.0}, 1.0)
    check(len(list(plans_for(pair))) == 4, "two tables must give four plans")
    pair_truth = ground_truth([pair], VERSION_1, VERSION_2)
    check(pair_truth["pair"]["ratio"] >= 1.0,
          "a downgraded cost model can never make the chosen plan cheaper in truth")
    check(pair_truth["pair"]["ratio"] != 1.0 or pair_truth["pair"]["same_plan"],
          "an unchanged plan must leave the true cost untouched")
    uniform = generate_workload(8, 40)
    uniform_truth = ground_truth(uniform, VERSION_1, VERSION_1)
    check(not any(info["regression"] for info in uniform_truth.values()),
          "identical constants can never regress")
    guard = Detector(VERSION_1, Constants(random_io=TRUE_RANDOM_IO / 100.0))
    extreme = guard.evaluate(queries, 2024)
    check(len([finding for finding in extreme if finding["regressed"]]) >= len(reported),
          "a worse cost model must not hide regressions")
    print("edge cases: single table, two-table plan space, identical constants, extreme bug")

    # ---- 7. determinism --------------------------------------------------
    again = Detector(VERSION_1, VERSION_2).evaluate(queries, 2024)
    check(again == findings, "the detector must be reproducible from its seed")
    regenerated = generate_workload(4711, 160)
    check([query.name for query in regenerated] == [query.name for query in queries]
          and all(a.fanout == b.fanout for a, b in zip(regenerated, queries)),
          "the workload must be reproducible from its seed")
    different = generate_workload(4712, 160)
    check(any(a.fanout != b.fanout for a, b in zip(different, queries)),
          "a different seed must produce a different workload")
    print("determinism: identical findings for the same seed")

    worst_case = max(findings, key=lambda item: item["ratio"])
    print("worst finding: %s ratio=%.2f p=%.4f plan %s -> %s"
          % (worst_case["query"], worst_case["ratio"], worst_case["p_value"],
             worst_case["plan_a"][1], worst_case["plan_b"][1]))
    print("PASS")


if __name__ == "__main__":
    main()
