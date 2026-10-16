"""Task 700 - Implement work partitioning heuristics for heterogeneous clusters.  (Tier: Expert, CTCI Ch 9 — System Design and Scalability)

Scheduling indivisible tasks onto machines of different speeds, and comparing three strategies that a
real cluster scheduler actually uses:

* **list scheduling (LPT)** — sort the tasks longest-first and give each one to the machine that will
  finish it earliest.  `finish[j] + size/speed[j]` is the greedy criterion; on identical machines this
  is the classical LPT rule with its 4/3 - 1/(3m) guarantee;
* **divisible-work proportional split** — when the work can be cut anywhere, each machine gets the
  speed share `speed[j]/total_speed`, so every machine finishes at exactly `work / total_speed`; that
  is provably optimal, so it also gives the tightest lower bound for the indivisible case;
* **chunked split** — the same proportional split but quantised to fixed-size chunks, which is what a
  task-based runtime does in practice; the extra makespan over the optimum is bounded by one chunk on
  the slowest machine.

Because the instances are small, the heuristics are checked against the *true* optimum by exhaustive
enumeration of every assignment, and against the two information-theoretic lower bounds
`total_work / total_speed` and `max_task / max_speed` — no scheduler can ever beat those, and the
checks verify that nothing does.

Verified: every task is assigned exactly once and every schedule respects both lower bounds; the
exhaustive optimum is never beaten by a heuristic and is sometimes strictly better; the list schedule
stays inside the classical 4/3 - 1/(3m) ratio on identical machines and inside the speed-aware bound
on the heterogeneous ones; the proportional split attains the lower bound exactly and the chunked one
stays within one chunk per machine; and the fast machines really do receive the larger share.
"""

import math


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def make_prng(seed):
    """splitmix64: the instance generator is deterministic, so the reported ratios are stable."""
    state = seed & 0xFFFFFFFFFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    return nxt


def generate_instance(prng, tasks, machines, max_speed=8):
    task_sizes = [1 + prng() % 20 for _ in range(tasks)]
    speeds = [1 + prng() % max_speed for _ in range(machines)]
    return task_sizes, speeds


def makespan(assignment, tasks, speeds):
    load = [0.0] * len(speeds)
    for task, machine in zip(tasks, assignment):
        load[machine] += task
    return max(load[machine] / speeds[machine] for machine in range(len(speeds)))


def list_schedule(tasks, speeds):
    """LPT with the speed-aware greedy criterion: the machine that finishes earliest gets the task."""
    order = sorted(range(len(tasks)), key=lambda index: -tasks[index])
    finish = [0.0] * len(speeds)
    load = [0.0] * len(speeds)
    assignment = [None] * len(tasks)
    for index in order:
        machine = min(range(len(speeds)),
                      key=lambda candidate: (finish[candidate] + tasks[index] / speeds[candidate],
                                             candidate))
        finish[machine] += tasks[index] / speeds[machine]
        load[machine] += tasks[index]
        assignment[index] = machine
    return assignment, load, finish


def brute_force_optimum(tasks, speeds):
    """Every assignment of n tasks to m machines: m^n, feasible for the tiny instances used here."""
    best = None
    best_assignment = None
    total = len(speeds) ** len(tasks)
    for code in range(total):
        assignment, value = [], code
        for _ in tasks:
            assignment.append(value % len(speeds))
            value //= len(speeds)
        span = makespan(assignment, tasks, speeds)
        if best is None or span < best - 1e-12:
            best, best_assignment = span, assignment
    return best, best_assignment


def proportional_split(work, speeds):
    """Divisible work: the share is the speed share, so everybody finishes together."""
    total_speed = sum(speeds)
    shares = [work * speed / total_speed for speed in speeds]
    make = [share / speed for share, speed in zip(shares, speeds)]
    return shares, make


def chunked_split(work, speeds, chunk):
    """The proportional split quantised to whole chunks; the remainder is put on the fastest machine."""
    shares, _ = proportional_split(work, speeds)
    counts = [int(math.floor(share / chunk)) for share in shares]
    leftover = work - sum(counts) * chunk
    fastest = speeds.index(max(speeds))
    counts[fastest] += math.ceil(leftover / chunk)
    load = [count * chunk for count in counts]
    load[fastest] -= sum(load) - work              # the fastest machine's last chunk is partly full
    return load


def lower_bounds(tasks, speeds):
    total = sum(tasks)
    return total / sum(speeds), max(tasks) / max(speeds)


def main():
    check(makespan([0, 0, 1, 1], [4, 3, 4, 3], [1, 1]) == 7.0,
          "the hand-checked example takes seven time units")
    check(proportional_split(120, [1, 2, 3])[0] == [20.0, 40.0, 60.0],
          "the divisible split follows the speed shares")
    check(abs(proportional_split(120, [1, 2, 3])[1][0] - 20.0) < 1e-12,
          "and every machine finishes at the same instant")
    check(lower_bounds([4, 4, 3, 3], [1, 1])[0] == 7.0, "the work lower bound is the total over speed")
    check(lower_bounds([20, 1], [4, 1])[1] == 5.0, "the longest task bounds the makespan too")

    prng = make_prng(0xC0DE5)
    identical_ratios = []
    hetero_ratios = []
    strict_improvements = 0
    instances = 0
    for _ in range(40):
        task_count = 6 + prng() % 2
        machine_count = 3 + prng() % 2
        tasks, speeds = generate_instance(prng, task_count, machine_count)
        assignment, load, finish = list_schedule(tasks, speeds)
        span = max(finish)
        check(None not in assignment and len(assignment) == len(tasks),
              "every task is assigned exactly once")
        check(all(0 <= machine < len(speeds) for machine in assignment), "and to a real machine")
        expected_load = [sum(task for index, task in enumerate(tasks)
                             if assignment[index] == machine)
                         for machine in range(len(speeds))]
        check(load == expected_load, "the reported loads follow from the assignment")
        check(abs(sum(load) - sum(tasks)) < 1e-9, "no work is lost or duplicated")
        check(span >= 1e-9, "a schedule takes positive time")
        work_bound, task_bound = lower_bounds(tasks, speeds)
        check(span >= work_bound - 1e-9,
              f"the list schedule respects the work bound ({span:.3f} >= {work_bound:.3f})")
        check(span >= task_bound - 1e-9,
              f"and the longest-task bound ({span:.3f} >= {task_bound:.3f})")
        optimum, _ = brute_force_optimum(tasks, speeds)
        check(optimum >= max(work_bound, task_bound) - 1e-9,
              "so does the true optimum")
        check(span >= optimum - 1e-9, "the heuristic can never beat the optimum")
        check(optimum <= span + 1e-9, "and the optimum is never worse than the heuristic")
        if span > optimum + 1e-9:
            strict_improvements += 1
        ratio = span / optimum
        hetero_ratios.append(ratio)
        check(ratio <= 2.0, f"the speed-aware list schedule stays within 2x of optimal: {ratio:.3f}")
        instances += 1

    check(instances == 40, "forty heterogeneous instances were checked")
    check(strict_improvements > 0,
          f"in {strict_improvements} of them the exhaustive optimum is strictly better")

    for _ in range(30):
        tasks, _ = generate_instance(prng, 8, 3)
        speeds = [1, 1, 1]
        assignment, load, finish = list_schedule(tasks, speeds)
        span = max(finish)
        optimum, _ = brute_force_optimum(tasks, speeds)
        bound = (4.0 / 3.0 - 1.0 / (3.0 * len(speeds))) * optimum
        check(span <= bound + 1e-9,
              f"on identical machines the classical LPT bound holds ({span:.3f} <= {bound:.3f})")
        identical_ratios.append(span / optimum)

    heavy_tasks, heavy_speeds = [40, 1, 1, 1], [4, 2, 2, 1]
    assignment, load, finish = list_schedule(heavy_tasks, heavy_speeds)
    check(abs(max(finish) - 10.0) < 1e-9, "one dominant task fixes the makespan at size/speed")
    check(abs(max(finish) - lower_bounds(heavy_tasks, heavy_speeds)[1]) < 1e-9,
          "which is exactly the longest-task lower bound")
    check(load[0] == 40 and assignment[0] == 0, "the fast machine gets the giant task")

    speeds = [1, 2, 4, 8]
    tasks = [8, 8, 8, 8, 4, 4, 2, 2]
    assignment, load, finish = list_schedule(tasks, speeds)
    work_bound, task_bound = lower_bounds(tasks, speeds)
    check(load[3] > load[0], f"the fastest machine carries more work ({load[3]} vs {load[0]})")
    check(max(finish) - min(finish) < work_bound,
          "while the finish times stay within one lower bound of each other")
    check(max(finish) >= work_bound - 1e-9, "and the bound still holds here")

    work = 1000.0
    speeds = [1, 3, 4]
    shares, make = proportional_split(work, speeds)
    check(abs(sum(shares) - work) < 1e-9, "the divisible split allocates exactly the work")
    check(max(make) - min(make) < 1e-12, "and produces a perfectly balanced finish time")
    work_bound, _ = lower_bounds([work], speeds)
    check(abs(make[0] - work_bound) < 1e-9, "which equals the lower bound: it is optimal")

    chunk = 10.0
    for work in (1000.0, 1013.0, 17.0, 3.0):
        load = chunked_split(work, [1, 3, 4], chunk)
        check(abs(sum(load) - work) < 1e-9, f"the chunked split allocates exactly {work} units")
        span = max(value / speed for value, speed in zip(load, [1, 3, 4]))
        bound = work / 8.0 + 3 * chunk / 4.0
        check(span >= work / 8.0 - 1e-9, "the chunked split can never beat the work bound")
        check(span <= bound + 1e-9,
              f"and it stays within one chunk per machine of it ({span:.3f} <= {bound:.3f})")
    check(max(chunked_split(1000.0, [1, 3, 4], 250.0)) == 750.0,
          "with a coarse chunk the fastest machine absorbs the remainder")

    tasks, speeds = [6, 6, 5, 5, 5, 3], [2, 1]
    assignment, load, finish = list_schedule(tasks, speeds)
    optimum, optimum_assignment = brute_force_optimum(tasks, speeds)
    check(makespan(optimum_assignment, tasks, speeds) == optimum,
          "the brute force reports the makespan of the assignment it returns")
    check(math.isclose(max(finish), makespan(assignment, tasks, speeds)),
          "the list schedule's finish times and makespan agree")
    check(makespan(assignment, tasks, speeds) >= optimum - 1e-9,
          "the heuristic is never better than the optimum")

    check(list_schedule([], [1, 2])[2] == [0.0, 0.0], "an empty task list schedules nothing")
    check(list_schedule([5], [1])[2] == [5.0], "a single task on a single machine is one load")
    check(list_schedule([5], [1, 1, 1])[0] == [0], "and lands on the first machine by the tie-break")

    repeat_tasks, repeat_speeds = [7, 3, 9, 1, 4], [2, 3, 5]
    check(list_schedule(repeat_tasks, repeat_speeds)[2] == list_schedule(repeat_tasks, repeat_speeds)[2],
          "the scheduler is deterministic")

    print(f"heterogeneous instances: worst list-schedule ratio {max(hetero_ratios):.3f}, "
          f"mean {sum(hetero_ratios) / len(hetero_ratios):.3f}, "
          f"{strict_improvements} where exhaustive search wins")
    print(f"identical machines: worst LPT ratio {max(identical_ratios):.3f} "
          f"(classical bound {4 / 3 - 1 / 9:.3f} for three machines)")
    print("divisible split hits total_work/total_speed exactly; chunked split stays one chunk above")
    print("PASS")


if __name__ == "__main__":
    main()
