"""Task 669 - Implement earliest deadline first scheduler simulator and feasibility check.  (Tier: Expert, Beyond — Networking, OS & Real-Time)

Three independent feasibility tests for preemptive uniprocessor scheduling of periodic
task sets (C = worst-case execution, T = period, D = relative deadline, D <= T):

1. `edf_simulate` - the ground truth: a cycle-accurate EDF simulator. Jobs are released
   synchronously (every task ready at t = 0, the critical instant), the ready job with the
   earliest absolute deadline runs one unit per tick, and any job still unfinished when its
   deadline passes is a miss.
2. `utilization` - the exact test when D = T: a synchronous implicit-deadline set is
   EDF-schedulable iff U = sum(C_i / T_i) <= 1. Computed with `Fraction`, so the U == 1
   boundary is decided exactly rather than by floating point.
3. `deadline_demand_feasible` - Baruah's processor demand test, exact also for constrained
   deadlines (D < T): for every absolute deadline t the demand
   dh(t) = sum_i max(0, floor((t - D_i)/T_i) + 1) * C_i must satisfy dh(t) <= t.

The sweep cross-checks all three on deterministically generated small task sets whose
hyperperiod stays tiny. For D = T the three must agree exactly; for D < T the utilization
test is only necessary, so the sweep also counts the sets where U <= 1 yet EDF still
misses a deadline, and there both exact tests must agree.

Simplified: synchronous release only (the critical instant, which is the worst case for
EDF), no aperiodic work, no context-switch overhead; infeasible runs are cut off at the
end of the simulation window instead of running to the hyperperiod.
"""

from fractions import Fraction
from math import gcd


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class PeriodicTask:
    """One periodic task: worst-case cost C, period T, relative deadline D <= T."""

    def __init__(self, name, cost, period, deadline=None):
        self.name = name
        self.cost = cost
        self.period = period
        self.deadline = period if deadline is None else deadline
        if cost < 0 or period <= 0:
            raise ValueError("cost must be >= 0 and period > 0")
        if self.deadline <= 0 or self.deadline > period:
            raise ValueError("relative deadline must satisfy 0 < D <= T")

    def __repr__(self):
        return "PeriodicTask(%s, C=%d, T=%d, D=%d)" % (self.name, self.cost,
                                                       self.period, self.deadline)


def lcm(a, b):
    return a * b // gcd(a, b)


def hyperperiod(tasks):
    h = 1
    for task in tasks:
        h = lcm(h, task.period)
    return h


def utilization(tasks):
    return sum((Fraction(t.cost, t.period) for t in tasks), Fraction(0))


def releases_in(tasks, ticks):
    """Total execution released inside [0, ticks) - one job at every multiple of T."""
    return sum(t.cost * ((ticks + t.period - 1) // t.period) for t in tasks)


def decision_window(tasks):
    """(ticks to simulate, largest deadline the demand test checks).

    Both tests get exactly the same deadline set: the simulator reports a miss when the
    tick reaches a deadline, so it covers every deadline <= bound, and the demand test is
    asked about the very same range. That makes their verdicts directly comparable.
    """
    if not tasks:
        return 1, 1
    bound = hyperperiod(tasks) + max(t.period for t in tasks) + 1
    return bound + 1, bound


def edf_simulate(tasks, ticks):
    """Cycle-accurate EDF over [0, ticks). Returns the evidence, not just a verdict."""
    jobs = []                      # [abs_deadline, remaining, task_index, release]
    executed = [0] * len(tasks)
    misses = []
    idle = 0
    for tick in range(ticks):
        for index, task in enumerate(tasks):
            if tick % task.period == 0:
                jobs.append([tick + task.deadline, task.cost, index, tick])
        for job in jobs:
            if job[1] > 0 and job[0] <= tick:
                misses.append((job[2], job[3], job[0]))
        ready = [job for job in jobs if job[1] > 0]
        if not ready:
            idle += 1
            continue
        job = min(ready, key=lambda j: (j[0], j[2], j[3]))
        job[1] -= 1
        executed[job[2]] += 1
    return {"misses": misses, "idle": idle, "executed": executed, "feasible": not misses}


def deadline_demand_feasible(tasks, bound):
    """Baruah's exact test over every absolute deadline <= bound."""
    deadlines = set()
    for task in tasks:
        k = 0
        while k * task.period + task.deadline <= bound:
            deadlines.add(k * task.period + task.deadline)
            k += 1
    for t in sorted(deadlines):
        demand = sum(((t - task.deadline) // task.period + 1) * task.cost
                     for task in tasks if t >= task.deadline)
        if demand > t:
            return False
    return True


def splitmix64(state):
    """Deterministic PRNG (splitmix64) so the sweeps are reproducible."""
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_task_sets(count, seed, constrained=False, periods=(4, 6, 8, 12)):
    """Small task sets whose hyperperiod stays tiny, so the simulator is the oracle."""
    sets = []
    state = seed
    for n in range(count):
        state, raw = splitmix64(state)
        task_count = 2 + (raw % 3)
        tasks = []
        for i in range(task_count):
            state, raw = splitmix64(state)
            period = periods[raw % len(periods)]
            state, raw = splitmix64(state)
            cost = 1 + (raw % max(1, period - 1))
            deadline = period
            if constrained and period > 2:
                state, raw = splitmix64(state)
                deadline = 2 + (raw % (period - 1))
            tasks.append(PeriodicTask("t%d" % i, cost, period, deadline))
        sets.append(tasks)
    return sets


def main():
    check(lcm(4, 6) == 12 and lcm(3, 5) == 15, "lcm sanity")
    check(hyperperiod([PeriodicTask("a", 1, 4), PeriodicTask("b", 1, 6),
                       PeriodicTask("c", 1, 12)]) == 12, "hyperperiod of 4/6/12 is 12")
    check(utilization([PeriodicTask("a", 1, 2), PeriodicTask("b", 1, 3)]) == Fraction(5, 6),
          "utilization must be exact (5/6)")

    # --- a half-loaded set over exactly one hyperperiod: idle and work are exact -------
    balanced = [PeriodicTask("a", 1, 4), PeriodicTask("b", 1, 6)]
    h = hyperperiod(balanced)
    check(h == 12 and utilization(balanced) == Fraction(5, 12), "U = 5/12 over H = 12")
    bal = edf_simulate(balanced, h)
    check(bal["feasible"], "a set with U = 5/12 must be feasible")
    check(bal["idle"] == h - releases_in(balanced, h) == 7,
          "idle ticks must be H - U*H = 7 (got %d)" % bal["idle"])
    check(bal["executed"] == [h // 4, h // 6] == [3, 2],
          "each task must run exactly C per period: %r" % (bal["executed"],))

    # --- textbook sets ---------------------------------------------------------------
    feasible_set = [PeriodicTask("t1", 1, 2), PeriodicTask("t2", 2, 5)]
    ticks, bound = decision_window(feasible_set)
    run = edf_simulate(feasible_set, ticks)
    check(run["feasible"], "C=1/T=2 + C=2/T=5 is EDF-feasible (U = 9/10)")
    check(utilization(feasible_set) == Fraction(9, 10), "U = 9/10")
    check(all(units >= task.cost * ((bound - task.deadline) // task.period + 1)
              for units, task in zip(run["executed"], feasible_set)),
          "every job whose deadline lies inside the window must have completed: %r"
          % (run["executed"],))
    check(run["idle"] == ticks - sum(run["executed"]), "no tick is both idle and busy")
    check(deadline_demand_feasible(feasible_set, bound),
          "the demand test must accept the feasible set")

    saturated = [PeriodicTask("a", 1, 2), PeriodicTask("b", 2, 4)]
    sat_ticks, sat_bound = decision_window(saturated)
    sat_run = edf_simulate(saturated, sat_ticks)
    check(utilization(saturated) == 1, "U = 1 exactly")
    check(sat_run["feasible"] and sat_run["idle"] == 0,
          "U == 1 must still be schedulable, with zero idle time")
    check(deadline_demand_feasible(saturated, sat_bound), "U == 1 passes the demand test")

    overloaded = [PeriodicTask("a", 2, 3), PeriodicTask("b", 2, 5)]
    over_ticks, over_bound = decision_window(overloaded)
    over_run = edf_simulate(overloaded, over_ticks)
    check(utilization(overloaded) > 1, "U > 1 for the overloaded set")
    check(not over_run["feasible"] and over_run["misses"], "an overloaded set must miss")
    check(not deadline_demand_feasible(overloaded, over_bound),
          "the demand test must reject the overloaded set")
    check(min(m[2] for m in over_run["misses"]) <= over_bound,
          "every reported miss carries a real absolute deadline")
    check(sum(over_run["executed"]) + over_run["idle"] == over_ticks,
          "every tick is either idle or busy, never both")

    # A constrained-deadline set with U <= 1 that is nevertheless infeasible: proof that
    # the utilization test is not sufficient once D < T.
    tight = [PeriodicTask("a", 1, 2, 2), PeriodicTask("b", 2, 5, 2)]
    check(utilization(tight) == Fraction(9, 10), "the tight set has U = 9/10 <= 1")
    tight_ticks, tight_bound = decision_window(tight)
    tight_run = edf_simulate(tight, tight_ticks)
    check(not tight_run["feasible"], "D = 2 < T = 5 really does miss deadlines here")
    check(not deadline_demand_feasible(tight, tight_bound),
          "the demand test must also reject the tight set")
    check(edf_simulate([PeriodicTask("a", 1, 2, 2), PeriodicTask("b", 2, 5, 5)], 30)["feasible"],
          "loosening b's deadline to D = T makes the set feasible")

    # --- deterministic sweeps ---------------------------------------------------------
    implicit = random_task_sets(120, 0xC0FFEE)
    feasible_seen = 0
    for tasks in implicit:
        tasks_ticks, tasks_bound = decision_window(tasks)
        sim = edf_simulate(tasks, tasks_ticks)["feasible"]
        demand = deadline_demand_feasible(tasks, tasks_bound)
        u_ok = utilization(tasks) <= 1
        check(sim == demand, "simulator and demand test disagree on %r" % (tasks,))
        check(sim == u_ok, "for D = T the utilization test is exact: %r -> %s" % (tasks, sim))
        feasible_seen += sim
    check(0 < feasible_seen < len(implicit),
          "the sweep must contain both feasible and infeasible sets (%d of %d)"
          % (feasible_seen, len(implicit)))

    constrained = random_task_sets(120, 0xBEEF, constrained=True)
    u_not_sufficient = 0
    for tasks in constrained:
        tasks_ticks, tasks_bound = decision_window(tasks)
        sim = edf_simulate(tasks, tasks_ticks)["feasible"]
        check(sim == deadline_demand_feasible(tasks, tasks_bound),
              "the demand test must be exact for constrained deadlines: %r" % (tasks,))
        if utilization(tasks) > 1:
            check(not sim, "U > 1 is always infeasible: %r" % (tasks,))
        elif not sim:
            u_not_sufficient += 1
    check(u_not_sufficient > 0,
          "the sweep must contain D < T sets with U <= 1 that are still late (%d)"
          % u_not_sufficient)

    # --- boundary and error handling --------------------------------------------------
    check(edf_simulate([], 5) == {"misses": [], "idle": 5, "executed": [], "feasible": True},
          "an empty task set is trivially feasible and always idle")
    check(deadline_demand_feasible([], 5), "the demand test accepts the empty set")
    check(edf_simulate([PeriodicTask("only", 3, 3)], 9)["feasible"],
          "a single task with C == T is feasible")
    check(not edf_simulate([PeriodicTask("only", 4, 3)], 12)["feasible"],
          "a single task with C > T is rejected by the simulator too")
    for bad in (lambda: PeriodicTask("x", 1, 0), lambda: PeriodicTask("x", -1, 4),
                lambda: PeriodicTask("x", 1, 4, 5), lambda: PeriodicTask("x", 1, 4, 0)):
        rejected = False
        try:
            bad()
        except ValueError:
            rejected = True
        check(rejected, "invalid periodic task parameters must be rejected")

    # --- determinism ------------------------------------------------------------------
    check(edf_simulate(overloaded, over_ticks) == over_run, "the simulator must be deterministic")
    first = [(t.cost, t.period, t.deadline) for t in random_task_sets(1, 4242)[0]]
    second = [(t.cost, t.period, t.deadline) for t in random_task_sets(1, 4242)[0]]
    check(first == second and len(first) >= 2, "the task-set generator must be deterministic")

    print("sweep: %d implicit sets (%d feasible), %d constrained (%d with U<=1 but late)"
          % (len(implicit), feasible_seen, len(constrained), u_not_sufficient))
    print("balanced idle=%d over H=%d, saturated idle=%d, tight first miss at t=%d"
          % (bal["idle"], h, sat_run["idle"], min(m[2] for m in tight_run["misses"])))
    print("PASS")


if __name__ == "__main__":
    main()
