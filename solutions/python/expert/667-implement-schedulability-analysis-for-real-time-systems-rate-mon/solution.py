"""Task 667 - Implement schedulability analysis for real-time systems (rate-monotonic).  (Tier: Expert, Beyond - Networking, OS and Real-Time)

Three classical schedulability tests for a periodic task set (C, T, D) under preemptive fixed-priority
scheduling are implemented together with an exact scheduler simulation that cross-checks them:

* the Liu-Layland utilisation bound `U = sum C_i/T_i <= n(2^(1/n) - 1)`, a sufficient (not necessary)
  test for rate-monotonic priorities with implicit deadlines;
* the hyperbolic bound `prod(U_i + 1) <= 2`, a strictly tighter sufficient test that uses the same
  utilisation figures;
* the exact response-time analysis `R_i = C_i + sum_{hp(i)} ceil(R_i / T_j) * C_j`, iterated to its fixed
  point (or until it passes D_i), which is both necessary and sufficient for fixed-priority preemptive
  scheduling and also handles D < T (the recurrence uses the periods of the higher-priority tasks, the
  deadline is only used in the acceptance test), and
* the EDF test `U <= 1` for implicit deadlines, which is optimal for that model.

Every analytic result is checked against a tick-accurate simulation of the schedule over several
hyperperiods: for implicit deadlines, "RTA says schedulable" must be equivalent to "no job misses its
deadline", and - the strongest cross-check - the simulated worst-case response time of each task must
equal the fixed point of the response-time equation. Harmonic task sets (periods dividing each other)
must be schedulable exactly when U <= 1, and sets with U > 1 must fail both the analysis and the
simulation. Constrained deadlines, deadline misses, a single task, and an empty set are covered too.

Simplified: periodic tasks only (no aperiodic/sporadic arrivals, no blocking from shared resources, no
jitter or context-switch overhead) and the simulation measures response times in whole ticks with one
tick of execution per task per tick. Verified: the bound values and their monotonicity, the
analytic/simulation equivalence and the response-time match, the harmonic-set equivalence, U > 1
failure, constrained deadlines, EDF optimality on implicit deadlines, and determinism.
"""
import math


DEFAULT_CYCLES = 2
MAX_ITERATIONS = 1000


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


class Task:
    __slots__ = ("name", "cost", "period", "deadline")

    def __init__(self, name, cost, period, deadline=None):
        check(cost > 0 and period > 0, "a task needs a positive cost and period")
        self.name = name
        self.cost = cost
        self.period = period
        self.deadline = period if deadline is None else deadline

    @property
    def utilisation(self):
        return self.cost / self.period

    def __repr__(self):
        return f"{self.name}(C={self.cost},T={self.period},D={self.deadline})"


def utilisation(tasks):
    return sum(task.utilisation for task in tasks)


def liu_layland_bound(count):
    check(count >= 1, "the bound is only defined for a non-empty task set")
    return count * (2 ** (1 / count) - 1)


def hyperbolic_bound(tasks):
    product = 1.0
    for task in tasks:
        product *= task.utilisation + 1
    return product


def rate_monotonic_order(tasks):
    """Rate-monotonic priority order: shortest period first, ties by position."""
    return sorted(range(len(tasks)), key=lambda index: (tasks[index].period, index))


def response_times(tasks, order=None):
    """Exact response-time analysis; returns (schedulable, {index: R}) with R = None when it diverges."""
    order = rate_monotonic_order(tasks) if order is None else list(order)
    result = {}
    for position, index in enumerate(order):
        task = tasks[index]
        higher = order[:position]
        current = task.cost
        for _ in range(MAX_ITERATIONS):
            interference = sum(-(-current // tasks[j].period) * tasks[j].cost for j in higher)
            updated = task.cost + interference
            if updated == current:
                break
            current = updated
            if current > task.deadline:
                break
        result[index] = current if current <= task.deadline else None
    return all(value is not None for value in result.values()), result


def edf_schedulable(tasks):
    return utilisation(tasks) <= 1.0 + 1e-12


def hyperperiod(tasks):
    period = 1
    for task in tasks:
        period = period * task.period // math.gcd(period, task.period)
    return period


def simulate(tasks, cycles=DEFAULT_CYCLES, edf=False):
    """Tick-accurate preemptive schedule; returns (feasible, worst response times, missed jobs)."""
    if not tasks:
        return True, {}, 0
    horizon = hyperperiod(tasks) * cycles
    order = rate_monotonic_order(tasks)
    priority = {index: position for position, index in enumerate(order)}
    active = []
    worst = {}
    missed_jobs = 0
    for tick in range(horizon):
        for index, task in enumerate(tasks):
            if tick % task.period == 0:
                active.append({"task": index, "arrival": tick, "remaining": task.cost,
                               "deadline": tick + task.deadline, "missed": False})
        for job in active:
            if not job["missed"] and job["remaining"] > 0 and job["deadline"] <= tick:
                job["missed"] = True
                missed_jobs += 1
        runnable = [job for job in active if job["remaining"] > 0]
        if not runnable:
            continue
        if edf:
            job = min(runnable, key=lambda item: (item["deadline"], item["arrival"], item["task"]))
        else:
            job = min(runnable, key=lambda item: (priority[item["task"]], item["arrival"], item["task"]))
        job["remaining"] -= 1
        if job["remaining"] == 0:
            response = tick + 1 - job["arrival"]
            index = job["task"]
            worst[index] = max(worst.get(index, 0), response)
            active.remove(job)
    return missed_jobs == 0 and len(worst) == len(tasks), worst, missed_jobs


def random_task_set(seed, count, max_cost=4, max_period=16):
    tasks = []
    for index in range(count):
        seed, r1 = splitmix64(seed)
        seed, r2 = splitmix64(seed)
        period = 2 + r2 % max_period
        cost = 1 + r1 % min(max_cost, period)
        tasks.append(Task(f"t{index}", cost, period))
    return tasks, seed


def main():
    # 1. the utilisation bound: value, monotonicity and its limit
    check(abs(liu_layland_bound(1) - 1.0) < 1e-12, "a single task is schedulable up to U = 1")
    check(abs(liu_layland_bound(2) - 2 * (math.sqrt(2) - 1)) < 1e-12, "the bound for two tasks")
    bounds = [liu_layland_bound(count) for count in range(1, 21)]
    check(all(bounds[index] > bounds[index + 1] for index in range(len(bounds) - 1)),
          "the bound must decrease with the number of tasks")
    check(abs(bounds[-1] - math.log(2)) < 0.05, "the bound tends to ln 2")
    check(liu_layland_bound(3) < liu_layland_bound(2) < 1.0, "bounds stay below one")

    # 2. Liu-Layland is sufficient but not necessary: the classic example
    classical = [Task("a", 1, 4), Task("b", 2, 5), Task("c", 2, 10)]
    total = utilisation(classical)
    check(abs(total - 0.85) < 1e-12, "the example utilisation is 0.85")
    check(total > liu_layland_bound(3), "the example must fail the utilisation bound")
    check(total < 1.0, "and it is still below full utilisation")
    feasible, classical_times = response_times(classical)
    check(feasible, "the exact response-time analysis must show it schedulable")
    check(classical_times == {0: 1, 1: 3, 2: 8}, f"hand-computed response times, got {classical_times}")
    ok, worst, missed = simulate(classical)
    check(ok and missed == 0, "the simulation must not miss a deadline")
    check(worst == {0: 1, 1: 3, 2: 8},
          f"the simulated worst-case response times must match the analysis, got {worst}")
    check(hyperbolic_bound(classical) > 2, "the hyperbolic bound also rejects this example")

    # 3. an unschedulable set is rejected by both the analysis and the simulation
    overloaded = [Task("a", 1, 4), Task("b", 2, 5), Task("c", 2, 7)]
    check(utilisation(overloaded) < 1.0, "the example stays below full utilisation")
    feasible, times = response_times(overloaded)
    check(not feasible, "the response-time analysis must reject it")
    check(times[2] is None, "the lowest-priority task must miss its deadline in the analysis")
    ok, worst, missed = simulate(overloaded)
    check(not ok and missed > 0, "the simulation must observe the missed deadline")
    check(utilisation(overloaded) > liu_layland_bound(3),
          "the utilisation bound rejects it as well")

    # 4. rate-monotonic order and the EDF comparison
    check(rate_monotonic_order(classical) == [0, 1, 2], "shorter periods get higher priority")
    reordered = [Task("c", 2, 10), Task("a", 1, 4), Task("b", 2, 5)]
    check(rate_monotonic_order(reordered) == [1, 2, 0], "priority must follow the period, not the order")
    check(sorted(rate_monotonic_order(reordered)) == [0, 1, 2], "the order must be a permutation")
    check(edf_schedulable(classical), "EDF follows the utilisation test")
    overfull = [Task("a", 1, 2), Task("b", 1, 4), Task("c", 3, 8)]
    check(not edf_schedulable(overfull), "EDF rejects a set above full utilisation")
    check(edf_schedulable(overloaded) and simulate(overloaded, edf=True)[0],
          "EDF schedules the set that rate-monotonic priorities cannot, as theory predicts")
    ok_edf, worst_edf, missed_edf = simulate(classical, edf=True)
    check(ok_edf and missed_edf == 0, "EDF must schedule a set with U < 1")

    # 5. harmonic task sets: schedulable exactly when U <= 1
    harmonic = [Task("a", 1, 4), Task("b", 2, 8), Task("c", 4, 16)]
    check(abs(utilisation(harmonic) - 0.75) < 1e-12, "harmonic utilisation is 0.75")
    check(response_times(harmonic)[0], "a harmonic set with U <= 1 must be schedulable")
    check(simulate(harmonic)[0], "and the simulation must agree")
    full = [Task("a", 1, 2), Task("b", 1, 4), Task("c", 2, 8)]
    check(abs(utilisation(full) - 1.0) < 1e-12, "this harmonic set is fully utilised")
    check(response_times(full)[0] and simulate(full)[0],
          "a harmonic set at U = 1 is still schedulable")
    over = [Task("a", 1, 2), Task("b", 1, 4), Task("c", 3, 8)]
    check(utilisation(over) > 1.0, "this harmonic set exceeds full utilisation")
    check(not response_times(over)[0] and not simulate(over)[0],
          "a harmonic set above U = 1 must fail both ways")

    # 6. response times from the analysis match the simulated worst cases over random sets
    seed = 0x5EED
    agreeing = 0
    for trial in range(24):
        tasks, seed = random_task_set(seed, 1 + trial % 4, max_cost=3, max_period=12)
        feasible, times = response_times(tasks)
        ok, worst, missed = simulate(tasks)
        check(feasible == ok,
              f"analysis and simulation must agree for {tasks}: {feasible} vs {ok}")
        if feasible:
            agreeing += 1
            for index, value in times.items():
                check(worst[index] == value,
                      f"response time mismatch for {tasks[index]}: analysis {value}, "
                      f"simulation {worst[index]}")
        else:
            check(missed > 0, "an unschedulable set must show a missed job in the simulation")
        check(utilisation(tasks) <= 1.0 or not feasible,
              "a set with U > 1 can never be schedulable")
    check(agreeing > 0, "some random sets must be feasible so the match is exercised")

    # 7. the sufficient bounds never claim more than the exact analysis
    seed = 0x1234
    for trial in range(24):
        tasks, seed = random_task_set(seed, 2 + trial % 4, max_cost=2, max_period=10)
        feasible, _ = response_times(tasks)
        if utilisation(tasks) <= liu_layland_bound(len(tasks)) + 1e-12:
            check(feasible, "the utilisation bound must only pass schedulable sets")
        if hyperbolic_bound(tasks) <= 2 + 1e-12:
            check(feasible, "the hyperbolic bound must only pass schedulable sets")
        if feasible:
            check(utilisation(tasks) <= 1.0 + 1e-12,
                  "a schedulable fixed-priority set can never exceed full utilisation")

    # 8. constrained deadlines (D < T) are analysed with the periods, accepted against the deadline
    constrained = [Task("a", 1, 4, deadline=3), Task("b", 2, 8, deadline=6)]
    feasible, times = response_times(constrained)
    check(times[0] == 1 and times[1] == 3, f"response times under D < T, got {times}")
    check(feasible, "both tasks meet their constrained deadlines")
    check(simulate(constrained)[0], "the simulation agrees for D < T")
    tight = [Task("a", 2, 4, deadline=2), Task("b", 3, 8, deadline=6)]
    feasible, times = response_times(tight)
    check(times[0] == 2, "the first task finishes exactly at its deadline")
    check(times[1] is None, "the second task cannot complete inside its constrained deadline")
    ok, worst, missed = simulate(tight)
    check(not ok and missed > 0, "the simulation must see the constrained-deadline miss")

    # 9. degenerate cases
    check(utilisation([]) == 0.0, "an empty set has zero utilisation")
    check(response_times([])[0], "an empty set is trivially schedulable")
    check(simulate([]) == (True, {}, 0), "the simulation of an empty set is trivially feasible")
    single = [Task("only", 7, 7)]
    check(abs(utilisation(single) - 1.0) < 1e-12, "a task with C = T is fully utilised")
    feasible, times = response_times(single)
    check(feasible and times[0] == 7, "a task with C = T just meets its deadline")
    check(simulate(single)[0], "and the simulation agrees")
    impossible = [Task("bad", 5, 4)]
    check(utilisation(impossible) > 1.0, "C > T exceeds full utilisation")
    check(not response_times(impossible)[0] and not simulate(impossible)[0],
          "a task with C > T can never be scheduled")
    check(response_times([Task("x", 3, 10, deadline=3)])[0],
          "a response time of three meets a deadline of three")
    check(response_times([Task("x", 3, 10, deadline=9)])[0],
          "a deadline larger than the period is allowed when the response time fits")

    # 10. determinism of both the analysis and the simulation
    tasks, _ = random_task_set(0xABCD, 4, max_cost=3, max_period=12)
    check(response_times(tasks) == response_times(tasks), "the analysis must be deterministic")
    check(simulate(tasks) == simulate(tasks), "the simulation must be deterministic")
    first = simulate(tasks, cycles=1)
    second = simulate(tasks, cycles=3)
    check(first[0] == second[0], "the verdict must not depend on the number of hyperperiods")
    check(all(second[1].get(index, 0) >= first[1].get(index, 0) for index in range(len(tasks))),
          "a longer simulation cannot find smaller worst-case response times")

    print(f"667 classical example U={total:.4f} bound={liu_layland_bound(3):.4f} "
          f"R={sorted(classical_times.items())}")
    print(f"667 randomised cross-checks agreed on {agreeing} feasible sets")
    print("PASS")


if __name__ == "__main__":
    main()
