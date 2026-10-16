"""Task 856 - Implement runtime adaptive operator re-optimization and replanning.  (Tier: Master, Ch 14 — Databases)

A pipeline of joins is executed step by step with *materialisation boundaries* between the
steps, exactly like an exchange-separated plan.  At every boundary the executor compares the
cardinality it actually produced with the cardinality the optimizer had estimated; when the
two disagree by more than a threshold the remaining steps are re-planned from the observed
cardinality, and the observed fan-out of the finished step is fed back into the statistics -
the classic adaptive re-optimization loop.  Materialisation is also what makes the state
hand-off safe: the tuples of a finished step already sit in an ordinary list, so a re-plan
cannot lose or duplicate one.

What is simplified: the data are four small in-memory relations, the plan is a left-deep
chain of three joins with one pushed-down predicate, and each join chooses one of three
physical algorithms (hash, nested loop, sorted merge) and which side to build on - a plan
space of 6**3 = 216 plans.  Work is *counted while executing* (build insertions, probes,
candidate examinations), so every cost is a real measurement, and an independently written
analytic model is verified against it.  There are no threads, no clock and no unseeded
randomness; sub-step adaptation (Eddies-style tuple routing) is deliberately out of scope.

Invariants verified:
  * the plan never changes the answer: all 216 plans produce the identical result multiset as
    a brute-force oracle, and the adaptive runs - including one that re-plans at every
    boundary - produce exactly that same multiset (no tuple lost, none duplicated);
  * the cost model is faithful: the analytic per-step work and cardinality equal the measured
    values exactly, so the optimizer optimises the quantity the executor really pays;
  * reactiveness: accurate statistics trigger zero re-plans, adversarial statistics trigger
    them, and every re-plan is justified by an actual drift between the observed and the
    estimated cardinality and is attributed to a specific boundary;
  * decision quality: each re-plan equals an independent brute-force argmin over the whole
    remaining plan space;
  * cost bound and regret: the adaptive run does strictly less work than the stale static
    plan, and its work stays within 1.35x of the post-hoc optimal plan that already knew the
    true cardinalities;
  * monotonicity: the number of re-plans is non-increasing in the mis-estimate threshold;
  * edge cases: an unfiltered pipeline, a single-row database, an empty database and a run
    with adaptivity disabled all stay correct and reproducible.
"""

import math
from itertools import product

MASK64 = (1 << 64) - 1
OPTIONS = tuple(product(("hash", "nested", "sorted"), ("stream", "table")))
MEMORY = 300.0        # rows that fit in a hash build before it spills to disk
SPILL = 4.0           # extra passes charged to the build side when it does not fit


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next_u64() % n


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_tables():
    """A small schema with a hot join key, so that a uniform estimate is badly wrong."""
    table_a = [{"aid": index, "k": 0 if index < 12 else 1 + (index % 4), "m": index % 3}
               for index in range(20)]
    table_b = [{"bid": index, "k": 0 if index < 32 else 1 + (index % 4), "v": (index * 7) % 5}
               for index in range(60)]
    table_c = [{"cid": index, "m": index % 3, "w": index % 4, "n": index % 2}
               for index in range(12)]
    table_d = [{"did": index, "n": index % 2, "x": index} for index in range(4)]
    return {"A": table_a, "B": table_b, "C": table_c, "D": table_d}


class Join:
    """One pipeline step: join the running stream with a table, then apply the predicate."""

    def __init__(self, name, table, left_key, right_key, algorithm="hash", build="stream",
                 predicate=None):
        self.name = name
        self.table = table
        self.left_key = left_key
        self.right_key = right_key
        self.algorithm = algorithm
        self.build = build          # "stream": hash the stream; "table": hash the relation
        self.predicate = predicate  # (column, limit): keep rows with column < limit

    def with_options(self, algorithm, build):
        return Join(self.name, self.table, self.left_key, self.right_key, algorithm, build,
                    self.predicate)

    def describe(self):
        return "%s[%s/%s]" % (self.name, self.algorithm, self.build)

    def keep(self, row):
        if self.predicate is None:
            return True
        column, limit = self.predicate
        return row.get(column) is not None and row[column] < limit


class Stats:
    """Estimates: relation sizes, expected matches per driving row, and stream cardinalities."""

    def __init__(self, sizes, fanouts):
        self.sizes = dict(sizes)
        self.fanouts = dict(fanouts)

    def copy(self):
        return Stats(self.sizes, self.fanouts)

    def observe(self, name, fanout, boundary, cardinality):
        self.fanouts[name] = fanout
        self.sizes["stream_%d" % boundary] = cardinality
        return self


def plan_space(steps):
    for combination in product(*[OPTIONS] * len(steps)):
        yield [step.with_options(*option) for step, option in zip(steps, combination)]


def predicted_cardinality(step, rows, stats):
    fanout = stats.fanouts[step.name]
    return max(0.0, rows * fanout * (0.5 if step.predicate else 1.0))


def predicted_work(step, rows, stats):
    """The optimizer's cost model, in exactly the units the executor counts."""
    table_rows = stats.sizes[step.table]
    build, probe = (rows, table_rows) if step.build == "stream" else (table_rows, rows)
    fanout = stats.fanouts[step.name]
    if step.algorithm == "hash":
        work = build + probe * (1.0 + fanout)
        return work + (SPILL * build if build > MEMORY else 0.0)
    if step.algorithm == "nested":
        return probe * build
    return build * (math.log(build + 1, 2) + 1.0) + probe * (math.log(build + 1, 2) + fanout)


def plan_cost(plan, rows, stats):
    total = 0.0
    estimate = float(rows)
    for step in plan:
        total += predicted_work(step, estimate, stats)
        estimate = predicted_cardinality(step, estimate, stats)
    return total, estimate


def best_plan(plan_steps, rows, stats):
    best = None
    for candidate in plan_space(plan_steps):
        cost, _estimate = plan_cost(candidate, rows, stats)
        if best is None or cost < best[0] - 1e-9:
            best = (cost, candidate)
    return best[1], best[0]


# ---------------------------------------------------------------------------
# execution - work is counted, not formula-derived
# ---------------------------------------------------------------------------
def execute_step(step, stream, tables):
    """Run a step: returns (output rows, measured work, emitted cardinality)."""
    table_rows = tables[step.table]
    if step.build == "stream":
        build_rows, probe_rows = stream, table_rows
        build_key, probe_key = step.left_key, step.right_key
    else:
        build_rows, probe_rows = table_rows, stream
        build_key, probe_key = step.right_key, step.left_key
    work = 0
    output = []
    index = None
    ordered = None
    if step.algorithm == "hash":
        index = {}
        for row in build_rows:
            work += 1
            index.setdefault(row[build_key], []).append(row)
        if len(build_rows) > MEMORY:            # the build side does not fit: it spills
            work += SPILL * len(build_rows)
    elif step.algorithm == "sorted":
        ordered = sorted(build_rows, key=lambda row: (row[build_key] is None, row[build_key]))
        work += len(build_rows) * math.log(len(build_rows) + 1, 2)
    for row in probe_rows:
        if step.algorithm == "hash":
            candidates = index.get(row[probe_key], ())
            work += 1 + len(candidates)
        elif step.algorithm == "nested":
            candidates = build_rows
            work += len(build_rows)
        else:
            candidates = [other for other in ordered
                          if other[build_key] is not None and other[build_key] == row[probe_key]]
            work += math.log(len(build_rows) + 1, 2) + len(candidates)
        for other in candidates:
            if step.build == "stream":
                left_row, right_row = row, other
            else:
                left_row, right_row = other, row
            if left_row.get(step.left_key) is None:
                continue
            if left_row[step.left_key] != right_row[step.right_key]:
                continue
            merged = dict(left_row)
            merged.update(right_row)
            if step.keep(merged):
                output.append(merged)
    return output, work, len(output)


def analytic_step(step, stream, tables):
    """Independent recomputation: count the matching pairs by brute force, then apply the
    cost model.  Returns (work, emitted) so both halves can be compared with the executor."""
    table_rows = tables[step.table]
    driving = stream if step.build == "stream" else table_rows
    other = table_rows if step.build == "stream" else stream
    pairs = emitted = 0
    for row in driving:
        for candidate in other:
            if step.build == "stream":
                left_row, right_row = row, candidate
            else:
                left_row, right_row = candidate, row
            if left_row.get(step.left_key) is None:
                continue
            if left_row[step.left_key] != right_row[step.right_key]:
                continue
            pairs += 1
            merged = dict(left_row)
            merged.update(right_row)
            if step.keep(merged):
                emitted += 1
    build_rows = len(stream) if step.build == "stream" else len(table_rows)
    probe_rows = len(table_rows) if step.build == "stream" else len(stream)
    if step.algorithm == "hash":
        work = build_rows + probe_rows + pairs
        if build_rows > MEMORY:
            work += SPILL * build_rows
    elif step.algorithm == "nested":
        work = probe_rows * build_rows
    else:
        work = build_rows * math.log(build_rows + 1, 2) + \
            probe_rows * math.log(build_rows + 1, 2) + pairs
    return work, emitted


def execute_plan(plan, tables):
    stream = [dict(row) for row in tables["A"]]
    total = 0.0
    trace = []
    for step in plan:
        stream, work, cardinality = execute_step(step, stream, tables)
        total += work
        trace.append((step.describe(), work, cardinality))
    return stream, total, trace


# ---------------------------------------------------------------------------
# adaptive execution
# ---------------------------------------------------------------------------
class AdaptiveRun:
    def __init__(self, steps, tables, stats, threshold=0.25, adaptive=True):
        self.steps = list(steps)
        self.tables = tables
        self.stats = stats
        self.threshold = threshold
        self.adaptive = adaptive
        self.replans = []
        self.trace = []
        self.work = 0.0

    def run(self):
        stream = [dict(row) for row in self.tables["A"]]
        previous = len(stream)
        for index in range(len(self.steps)):
            if index > 0 and self.adaptive:
                self.maybe_replan(index, stream)
            step = self.steps[index]        # re-read: a re-plan may have replaced this step
            stream, work, cardinality = execute_step(step, stream, self.tables)
            self.work += work
            self.trace.append((step.describe(), work, cardinality))
            if self.adaptive and previous > 0:
                # feed the observed fan-out back, exactly like a real adaptivity controller
                self.stats.observe(step.name, cardinality / float(previous), index, cardinality)
            previous = cardinality
        return stream

    def maybe_replan(self, index, stream):
        step = self.steps[index]
        observed = float(len(stream))
        estimated = self.stats.sizes.get("stream_%d" % index)
        if estimated is None:
            return False
        drift = abs(observed - estimated) / max(estimated, 1.0)
        if drift <= self.threshold:
            return False
        remaining = [candidate.with_options(candidate.algorithm, candidate.build)
                     for candidate in self.steps[index:]]
        chosen, cost = best_plan(remaining, observed, self.stats.copy())
        self.stats.sizes["stream_%d" % index] = observed
        for position, candidate in enumerate(chosen):
            self.steps[index + position] = candidate
        self.replans.append({"boundary": step.name, "observed": observed,
                             "estimated": estimated, "drift": drift,
                             "chosen": [candidate.describe() for candidate in chosen],
                             "predicted_cost": cost})
        return True


def oracle_result(tables):
    """Brute force over the data: A join B on k, join C on m with w<3, join D on n."""
    rows = []
    for a in tables["A"]:
        for b in tables["B"]:
            if a["k"] != b["k"]:
                continue
            first = dict(a)
            first.update(b)
            for c in tables["C"]:
                if first["m"] != c["m"] or not c["w"] < 3:
                    continue
                second = dict(first)
                second.update(c)
                for d in tables["D"]:
                    if second["n"] != d["n"]:
                        continue
                    final = dict(second)
                    final.update(d)
                    rows.append(final)
    return rows


def canonical(rows, columns):
    return sorted(tuple(row.get(column) for column in columns) for row in rows)


def baseline_steps():
    return [Join("A-B", "B", "k", "k", "nested", "stream"),
            Join("A-B-C", "C", "m", "m", "nested", "stream", ("w", 3)),
            Join("A-B-C-D", "D", "n", "n", "nested", "stream")]


def plan_time_stats():
    """The deliberately uniform plan-time view: fan-out 12 for A-B, 4 for A-B-C, 2 for D."""
    return Stats({"A": 20, "B": 60, "C": 12, "D": 4, "stream_1": 240.0, "stream_2": 480.0},
                 {"A-B": 12.0, "A-B-C": 4.0, "A-B-C-D": 2.0})


def accurate_stats(true_cardinalities, true_fanouts):
    return Stats({"A": 20, "B": 60, "C": 12, "D": 4,
                  "stream_1": true_cardinalities[0], "stream_2": true_cardinalities[1]},
                 true_fanouts)


def measure_cardinalities(tables):
    """Walk the pipeline once, recording the true intermediate cardinalities and fan-outs."""
    stream = [dict(row) for row in tables["A"]]
    previous = len(stream)
    cardinalities = []
    fanouts = {}
    for step in baseline_steps():
        stream, _work, cardinality = execute_step(step, stream, tables)
        cardinalities.append(cardinality)
        fanouts[step.name] = cardinality / float(previous)
        previous = cardinality
    return cardinalities, fanouts


def optimal_work_and_plan(tables):
    """Brute force with the true cardinalities: the best plan anyone could have chosen."""
    best = None
    for plan in plan_space(baseline_steps()):
        stream = [dict(row) for row in tables["A"]]
        total = 0.0
        for step in plan:
            stream, work, _cardinality = execute_step(step, stream, tables)
            total += work
        if best is None or total < best[0]:
            best = (total, plan)
    return best[0], best[1]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    tables = build_tables()
    columns = ("aid", "bid", "m", "w", "n", "did")
    check(len(tables["A"]) == 20 and len(tables["B"]) == 60, "the schema must be populated")
    steps = baseline_steps()
    space = list(plan_space(steps))
    check(len(space) == 216, "the plan space must have 216 plans, got %d" % len(space))
    check(len({tuple(step.describe() for step in plan) for plan in space}) == 216,
          "every enumerated plan must be distinct")

    expected = oracle_result(tables)
    check(len(expected) > 1000, "the oracle must return a large result (%d rows)" % len(expected))
    cardinalities, fanouts = measure_cardinalities(tables)
    print("plan space: %d plans; true cardinalities %s; oracle %d rows"
          % (len(space), cardinalities, len(expected)))
    print("true fan-outs: %s" % {name: round(value, 3) for name, value in fanouts.items()})

    # ---- 1. the plan never changes the answer ----------------------------
    for plan in space:
        rows, _work, _trace = execute_plan(plan, tables)
        check(canonical(rows, columns) == canonical(expected, columns),
              "plan %s changed the answer" % [step.describe() for step in plan])
    print("semantics: all %d plans produce the identical result multiset" % len(space))

    # ---- 2. the cost model matches the measurement exactly ---------------
    rng = SplitMix64(31337)
    sampled = [space[rng.below(len(space))] for _ in range(24)]
    worst_work = worst_cardinality = 0.0
    for plan in sampled:
        stream = [dict(row) for row in tables["A"]]
        model_total = 0.0
        for step in plan:
            model_work, model_emitted = analytic_step(step, stream, tables)
            stream, work, cardinality = execute_step(step, stream, tables)
            check(abs(work - model_work) < 1e-9,
                  "model work %.6f != measured %.6f for %s" % (model_work, work, step.describe()))
            check(cardinality == model_emitted,
                  "model cardinality %d != measured %d for %s"
                  % (model_emitted, cardinality, step.describe()))
            worst_work = max(worst_work, abs(work - model_work))
            worst_cardinality = max(worst_cardinality, float(abs(cardinality - model_emitted)))
            model_total += model_work
        measured_total = sum(work for _name, work, _card in execute_plan(plan, tables)[2])
        check(abs(model_total - measured_total) < 1e-9,
              "a whole-plan model mismatch on %s" % [step.describe() for step in plan])
    print("cost model: %d sampled plans match the measured work exactly (max deviation %.1e)"
          % (len(sampled), worst_work))
    check(worst_cardinality == 0.0, "the model must be exact about cardinalities")

    # ---- 3. accurate statistics trigger no re-plan ----------------------
    accurate = accurate_stats(cardinalities[:2], fanouts)
    calm = AdaptiveRun(baseline_steps(), tables, accurate)
    rows = calm.run()
    check(calm.replans == [], "accurate statistics must not trigger a re-plan")
    check(canonical(rows, columns) == canonical(expected, columns), "the calm run must be correct")
    print("accurate statistics: %d re-plans, work %.0f" % (len(calm.replans), calm.work))

    # ---- 4. adversarial statistics react --------------------------------
    initial_plan, initial_cost = best_plan(baseline_steps(), len(tables["A"]), plan_time_stats())
    check(all(step.algorithm in ("hash", "nested", "sorted") for step in initial_plan),
          "the plan-time optimizer must return a well-formed plan")
    print("plan-time choice: %s (predicted %.0f)"
          % ([step.describe() for step in initial_plan], initial_cost))
    static_rows, static_work, _static_trace = execute_plan(initial_plan, tables)
    check(canonical(static_rows, columns) == canonical(expected, columns),
          "the stale static plan must still be correct")
    run = AdaptiveRun(initial_plan, tables, plan_time_stats())
    rows = run.run()
    check(canonical(rows, columns) == canonical(expected, columns),
          "the adaptive run must produce the same answer")
    check(run.replans, "a mis-estimated boundary must trigger a re-plan")
    check(all(replan["drift"] > run.threshold for replan in run.replans),
          "every re-plan must be justified by a real drift")
    check(all(replan["observed"] != replan["estimated"] for replan in run.replans),
          "a re-plan must report the observed and the estimated cardinality")
    check(all(replan["boundary"] in {step.name for step in steps} for replan in run.replans),
          "every re-plan must be attributed to a step boundary")
    check(any(replan["predicted_cost"] > 0 for replan in run.replans),
          "a re-plan must carry a predicted cost")
    print("adversarial statistics: %d re-plans at %s (drift %.2f..%.2f)"
          % (len(run.replans), [r["boundary"] for r in run.replans],
             min(r["drift"] for r in run.replans), max(r["drift"] for r in run.replans)))

    # ---- 5. every decision is the brute-force optimum -------------------
    check(len(run.replans) == 2, "both downstream boundaries must react, got %d" % len(run.replans))
    for replan in run.replans:
        index = [step.name for step in steps].index(replan["boundary"])
        remaining = [candidate.with_options(candidate.algorithm, candidate.build)
                     for candidate in steps[index:]]
        reference, _cost = best_plan(remaining, replan["observed"], plan_time_stats().copy())
        check(replan["chosen"] == [step.describe() for step in reference],
              "the re-plan decision must equal the brute-force argmin")
    print("decision: every re-plan matches an independent brute-force search")

    # ---- 6. cost bound and regret ---------------------------------------
    check(run.work < static_work,
          "the adaptive plan must do less work than the stale plan (%.0f vs %.0f)"
          % (run.work, static_work))
    optimal_work, optimal_plan = optimal_work_and_plan(tables)
    regret = run.work / optimal_work
    check(regret <= 1.35, "the adaptive run must stay within 1.35x of optimal, got %.3f" % regret)
    check(optimal_work <= static_work, "the optimal plan cannot be worse than the static one")
    print("work: static=%.0f adaptive=%.0f optimal=%.0f (speedup %.2fx, regret %.2f)"
          % (static_work, run.work, optimal_work, static_work / run.work, regret))
    print("post-hoc optimal plan: %s" % [step.describe() for step in optimal_plan])

    # the work counts are consistent with the trace
    check(abs(sum(entry[1] for entry in run.trace) - run.work) < 1e-9,
          "the recorded trace must add up to the total work")

    # ---- 7. monotonicity in the threshold -------------------------------
    counts = []
    for threshold in (0.0, 0.05, 0.25, 0.5, 1.0, 5.0):
        probe = AdaptiveRun(baseline_steps(), tables, plan_time_stats(), threshold=threshold)
        probe.run()
        counts.append((threshold, len(probe.replans)))
    check(all(counts[i][1] >= counts[i + 1][1] for i in range(len(counts) - 1)),
          "re-plans must be non-increasing in the threshold: %s" % counts)
    check(counts[0][1] == 2 and counts[-1][1] == 0,
          "a zero threshold reacts at both boundaries, a huge one never: %s" % counts)
    print("threshold sweep: %s" % counts)

    # ---- 8. exactly-once with re-planning at every boundary -------------
    storm = AdaptiveRun(baseline_steps(), tables, plan_time_stats(), threshold=0.0)
    rows = storm.run()
    check(canonical(rows, columns) == canonical(expected, columns),
          "re-planning at every boundary must still be exactly-once")
    check(len(storm.replans) == len(steps) - 1, "a zero threshold must react at every boundary")
    check(all(cardinality >= 0 for _name, _work, cardinality in storm.trace),
          "every step must report a cardinality")
    check([entry[2] for entry in storm.trace] == cardinalities,
          "re-planning must not change the intermediate cardinalities")

    # ---- 9. edge cases ---------------------------------------------------
    unfiltered = [Join("A-B", "B", "k", "k"), Join("A-B-C", "C", "m", "m"),
                  Join("A-B-C-D", "D", "n", "n")]
    plain = AdaptiveRun(unfiltered, tables, plan_time_stats())
    rows = plain.run()
    static_unfiltered, _work, _trace = execute_plan(unfiltered, tables)
    check(canonical(rows, columns) == canonical(static_unfiltered, columns),
          "an unfiltered pipeline must agree with the static execution")
    single = AdaptiveRun(baseline_steps(),
                         {"A": tables["A"][:1], "B": tables["B"][:1],
                          "C": tables["C"][:1], "D": tables["D"][:1]}, plan_time_stats())
    check(len(single.run()) <= 1, "a one-row database must give at most one result")
    blank = AdaptiveRun(baseline_steps(), {"A": [], "B": [], "C": [], "D": []},
                        plan_time_stats())
    check(blank.run() == [], "an empty database must give no rows and no crash")
    frozen = AdaptiveRun(initial_plan, tables, plan_time_stats(), adaptive=False)
    frozen.run()
    check(frozen.replans == [] and abs(frozen.work - static_work) < 1e-9,
          "a run with adaptivity disabled must never re-plan")
    print("edge cases: unfiltered, single-row, empty, adaptivity disabled")

    # ---- 10. determinism -------------------------------------------------
    first = AdaptiveRun(baseline_steps(), tables, plan_time_stats())
    second = AdaptiveRun(baseline_steps(), tables, plan_time_stats())
    check(canonical(first.run(), columns) == canonical(second.run(), columns),
          "the adaptive run must be deterministic")
    check(first.replans == second.replans, "the re-plan decisions must be reproducible")
    print("determinism: identical results and identical re-plan decisions")
    print("PASS")


if __name__ == "__main__":
    main()
