"""Task 980 - Implement continuous feedback loop for curriculum improvement from user telemetry.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A closed loop over the curriculum.  Each cycle collects per-problem telemetry (attempts, solves, median
minutes, complaints, ratings), a priority-ordered rule engine turns those signals into at most one action per
problem - retire an unsolvable problem, simplify one that is too slow, add hints to one learners fight with,
promote one that is too easy, rewrite one learners rate badly - and the actions change the expected telemetry
parameters, which is what the next cycle observes.  The loop stops when a cycle produces no action, and the
KPI vector (solve rate, minutes, rating, complaints per hundred attempts) must improve monotonically on the
way to that fixpoint.

Simplified versus a live platform: the telemetry is simulated from expected parameters with a seeded PRNG
instead of collected over weeks, cohorts are one batch, and the effect of each action on the parameters is a
hand-written delta rather than a learned causal model.  Verified: every rule firing on its planted problem,
priority among overlapping rules, the documented parameter effects, convergence to an idempotent fixpoint,
monotone KPI improvement, retirement bookkeeping, the KPI against an independent scan oracle, zero-attempt
and empty-telemetry boundaries and reproducibility.
"""
from collections import namedtuple

Metrics = namedtuple("Metrics", "attempts solves minutes complaints rating")
PACE_BUDGET, MIN_ATTEMPTS, COHORT = 25.0, 10, 34
PLANTED = {
    "p01": {"solve_rate": 0.05, "minutes": 30.0, "complaints": 4.0, "rating": 2.2, "status": "active"},
    "p02": {"solve_rate": 0.50, "minutes": 40.0, "complaints": 3.0, "rating": 3.4, "status": "active"},
    "p03": {"solve_rate": 0.35, "minutes": 20.0, "complaints": 3.0, "rating": 3.8, "status": "active"},
    "p04": {"solve_rate": 0.98, "minutes": 6.0, "complaints": 0.0, "rating": 4.6, "status": "active"},
    "p05": {"solve_rate": 0.70, "minutes": 18.0, "complaints": 1.0, "rating": 2.0, "status": "active"},
    "p06": {"solve_rate": 0.70, "minutes": 15.0, "complaints": 1.0, "rating": 4.5, "status": "active"},
}
EXPECTED_FIRST_CYCLE = (("p01", "retire"), ("p02", "simplify"), ("p03", "add_hints"),
                        ("p04", "promote"), ("p05", "rewrite"))
# parameter deltas per action: solve-rate shift, minute factor, complaint shift, rating shift
EFFECTS = {"simplify": (0.15, 0.75, -0.5, 0.1), "add_hints": (0.10, 1.0, -2.0, 0.3),
           "promote": (-0.20, 1.0, 0.0, -0.1), "rewrite": (0.05, 1.0, -2.0, 1.0)}
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seeds make the simulated cohorts reproducible."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) % bound


def initial_problems():
    return {ident: dict(values) for ident, values in PLANTED.items()}


def simulate(problems, seed, cohort=COHORT):
    """One cohort of telemetry per active problem, jittered deterministically around the parameters."""
    rng = Rng(seed)
    telemetry = {}
    for ident in sorted(problems):
        parameters = problems[ident]
        if parameters["status"] != "active":
            continue
        attempts = cohort + rng.below(13)
        solves = int(round(attempts * min(max(parameters["solve_rate"], 0.0), 1.0)))
        minutes = round(parameters["minutes"] * (0.9 + rng.below(21) / 100.0), 2)
        complaints = max(0, int(round(parameters["complaints"])) + rng.below(3) - 1)
        rating = round(min(5.0, max(1.0, parameters["rating"] + (rng.below(5) - 2) / 10.0)), 2)
        telemetry[ident] = Metrics(attempts, solves, minutes, complaints, rating)
    return telemetry


def solve_rate(metrics):
    return metrics.solves / metrics.attempts if metrics.attempts else 0.0


def complaints_per_100(metrics):
    return 100.0 * metrics.complaints / metrics.attempts if metrics.attempts else 0.0


def act(metrics, pace=PACE_BUDGET, minimum=MIN_ATTEMPTS):
    """The single highest-priority action for one problem, or None when the signals are healthy."""
    if metrics.attempts < minimum:
        return None
    rate, complaints = solve_rate(metrics), complaints_per_100(metrics)
    if rate < 0.20 and complaints >= 5.0:
        return "retire"
    if metrics.minutes > pace and rate < 0.60:
        return "simplify"
    if rate < 0.45 and complaints >= 2.0:
        return "add_hints"
    if rate > 0.95 and metrics.minutes < 8.0:
        return "promote"
    return "rewrite" if metrics.rating <= 2.5 else None


def apply_action(problems, ident, action):
    """Mutate the problem's expected parameters - this is the only way the loop can change telemetry."""
    parameters = problems[ident]
    if action == "retire":
        parameters["status"] = "retired"
        return
    rate_shift, minute_factor, complaint_shift, rating_shift = EFFECTS[action]
    parameters["solve_rate"] = round(min(1.0, max(0.0, parameters["solve_rate"] + rate_shift)), 4)
    parameters["minutes"] = round(parameters["minutes"] * minute_factor, 4)
    parameters["complaints"] = round(max(0.0, parameters["complaints"] + complaint_shift), 4)
    parameters["rating"] = round(min(5.0, max(1.0, parameters["rating"] + rating_shift)), 4)
    parameters["applied"] = parameters.get("applied", 0) + 1


def kpi(problems, telemetry):
    """Learner-experience scores over the active curriculum of one cohort."""
    rows = [telemetry[ident] for ident in sorted(telemetry)]
    if not rows:
        return {"solve_rate": 0.0, "minutes": 0.0, "rating": 0.0, "complaints_per_100": 0.0}
    return {"solve_rate": sum(solve_rate(row) for row in rows) / len(rows),
            "minutes": sum(row.minutes for row in rows) / len(rows),
            "rating": sum(row.rating for row in rows) / len(rows),
            "complaints_per_100": sum(complaints_per_100(row) for row in rows) / len(rows)}


def kpi_oracle(problems, telemetry):
    """Independent oracle: accumulate the same scores with explicit running totals."""
    count, rates, minutes, ratings, complaints = 0, 0.0, 0.0, 0.0, 0.0
    for ident in sorted(telemetry):
        row = telemetry[ident]
        count += 1
        rates += row.solves / row.attempts if row.attempts else 0.0
        minutes += row.minutes
        ratings += row.rating
        complaints += (100.0 * row.complaints / row.attempts) if row.attempts else 0.0
    if not count:
        return {"solve_rate": 0.0, "minutes": 0.0, "rating": 0.0, "complaints_per_100": 0.0}
    return {"solve_rate": rates / count, "minutes": minutes / count, "rating": ratings / count,
            "complaints_per_100": complaints / count}


def run_loop(seed=7, cycles=6):
    """Collect, decide, apply - until a cycle produces no action (the curriculum fixpoint)."""
    problems, history = initial_problems(), []
    for number in range(cycles):
        telemetry = simulate(problems, seed + number)
        record = {"cycle": number + 1, "kpi": kpi(problems, telemetry), "actions": []}
        for ident in sorted(telemetry):
            action = act(telemetry[ident])
            if action:
                record["actions"].append((ident, action))
                apply_action(problems, ident, action)
        history.append(record)
        if not record["actions"]:
            break
    return problems, history


def main():
    problems, history = run_loop()
    # 1) every rule fires on its planted problem in the first cycle - an exact action list.
    check(tuple(history[0]["actions"]) == EXPECTED_FIRST_CYCLE and len(history) >= 2,
          f"the first cycle's actions are wrong: {history[0]['actions']}")
    # 2) priority among overlapping rules: the most severe match wins, never a later rule.
    crowded = Metrics(60, 3, 60.0, 20, 1.0)
    check(act(crowded) == "retire" and act(Metrics(60, 21, 60.0, 0, 4.0)) == "simplify"
          and act(Metrics(60, 24, 20.0, 3, 4.0)) == "add_hints"
          and act(Metrics(60, 58, 5.0, 0, 4.0)) == "promote" and act(Metrics(60, 40, 20.0, 0, 2.0)) == "rewrite"
          and act(Metrics(60, 40, 20.0, 0, 4.5)) is None,
          f"rule priority is wrong: {act(crowded)}")
    # 3) the documented parameter effects, checked on a fresh copy of one problem per action.
    sample = initial_problems()
    before = dict(sample["p02"])
    apply_action(sample, "p02", "simplify")
    check(sample["p02"]["solve_rate"] == round(before["solve_rate"] + 0.15, 4)
          and sample["p02"]["minutes"] == round(before["minutes"] * 0.75, 4)
          and sample["p02"]["complaints"] == round(before["complaints"] - 0.5, 4)
          and sample["p02"]["applied"] == 1 and before["solve_rate"] == PLANTED["p02"]["solve_rate"],
          f"the simplify effect is wrong: {sample['p02']}")
    retired = initial_problems()
    apply_action(retired, "p01", "retire")
    check(retired["p01"]["status"] == "retired"
          and "p01" not in simulate(retired, 1) and "p05" in simulate(retired, 1),
          "a retired problem must leave the curriculum")
    # 4) convergence: the loop ends with an actionless cycle, and re-running the rules there is idempotent.
    final_problems, final_history = run_loop()
    final_telemetry = simulate(final_problems, 7 + len(history) - 1)
    check(final_history[-1]["actions"] == [] and len(final_history) <= 6
          and all(act(row) is None for row in final_telemetry.values()) and len(final_history) == 3,
          f"the loop did not reach a fixpoint: {final_history[-1]['actions']}")
    # 5) KPI improvement: solve rate and rating rise, minutes and complaints fall, cycle over cycle.
    first, last = history[0]["kpi"], history[-1]["kpi"]
    check(last["solve_rate"] > first["solve_rate"] and last["rating"] > first["rating"]
          and last["minutes"] < first["minutes"] and last["complaints_per_100"] < first["complaints_per_100"]
          and all(history[index + 1]["kpi"]["solve_rate"] >= history[index]["kpi"]["solve_rate"]
                  for index in range(len(history) - 1)),
          f"the KPI did not improve: {first} -> {last}")
    # 6) retirement bookkeeping: exactly one problem retired, it is gone from every later cohort, and the
    #    curriculum is conserved as active plus retired.
    retired_ids = [ident for ident in final_problems if final_problems[ident]["status"] == "retired"]
    check(retired_ids == ["p01"] and len(final_problems) == len(PLANTED)
          and len(final_problems) == len(final_telemetry) + len(retired_ids)
          and not any(ident == "p01" for ident in final_telemetry),
          f"retirement bookkeeping is wrong: {retired_ids}")
    # 7) oracle: the KPI and the action list match an independent scan of the same telemetry.
    def matches_oracle(seed):
        mine, theirs = kpi(final_problems, simulate(final_problems, seed)), \
            kpi_oracle(final_problems, simulate(final_problems, seed))
        return all(abs(mine[key] - theirs[key]) < 1e-9 for key in mine)
    decisions = [ident for ident in sorted(final_telemetry) if act(final_telemetry[ident])]
    check(all(matches_oracle(seed) for seed in range(7, 12)) and decisions == []
          and kpi({}, {}) == kpi_oracle({}, {}),
          "the KPI or the action list disagrees with the oracle")
    # 8) boundaries: empty telemetry, a zero-attempt problem and a one-cycle loop.
    empty = Metrics(0, 0, 0.0, 0, 0.0)
    check(act(empty) is None and solve_rate(empty) == 0.0 and complaints_per_100(empty) == 0.0
          and act(Metrics(9, 0, 90.0, 9, 1.0)) is None and act(Metrics(0, 0, 0.0, 0, 4.0), minimum=0) is None
          and kpi({}, {}) == {"solve_rate": 0.0, "minutes": 0.0, "rating": 0.0, "complaints_per_100": 0.0}
          and len(run_loop(cycles=1)[1]) == 1 and simulate({}, 3) == {},
          "the zero-attempt and empty-telemetry boundaries are wrong")
    # 9) reproducibility: the whole loop, its actions and KPIs are byte-identical across runs.
    check(run_loop() == (problems, history) and [record["actions"] for record in final_history]
          == [record["actions"] for record in history] and simulate(initial_problems(), 7)
          == simulate(initial_problems(), 7),
          "the feedback loop must be reproducible")
    print(f"problems={len(PLANTED)} cycles={len(history)} actions="
          f"{[len(record['actions']) for record in history]} retired={retired_ids}")
    print(f"kpi_first={{{', '.join(f'{k}:{v:.2f}' for k, v in sorted(first.items()))}}} "
          f"kpi_final={{{', '.join(f'{k}:{v:.2f}' for k, v in sorted(last.items()))}}}")
    print("PASS")


if __name__ == "__main__":
    main()
