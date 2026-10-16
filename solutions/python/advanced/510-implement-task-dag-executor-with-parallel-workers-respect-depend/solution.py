"""Task 510 - Implement task DAG executor with parallel workers (respect dependencies). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

List scheduling of a task DAG on `workers` identical machines, in three independent formulations
that must agree: (1) an event-driven simulation that keeps a heap of ready tasks keyed by the
scheduling rule and a heap of running tasks keyed by finish time, assigning the best ready task to
the lowest-numbered free worker whenever one is available and otherwise jumping the clock to the
earliest completion; (2) a tick-by-tick simulation on integer durations that completes first,
releases workers and then greedily fills them; and (3) a real-thread executor where `workers`
threads pull ready tasks through a lock/condition protocol, which is used only to exercise the
dependency handshake (all assertions there - one execution per task, every prerequisite before its
dependant - hold for any interleaving, because the shared log is appended under the lock and a task
only becomes ready after its prerequisites were logged). The fixed rule is "smallest task id first"
an alternative "longest duration first" (LPT) rule is also simulated to show that the rule matters
(neither dominates) while all of them stay inside the classic bound.

Documented checks: the hand-computed example A(3), B(2), C(4) all ready with D(1) after A and B and
E(1) after C has makespan exactly 7 on 2 workers (A 0-3, B 0-2, C 2-6, D 3-4, E 6-7); the event and
tick simulations must produce element-wise identical start/finish/worker assignments and makespans
on every instance; one worker must give exactly the total work and `workers >= n` exactly the
critical path (both closed forms, with the critical path itself cross-checked against a brute-force
enumeration of all paths); every schedule must satisfy start >= finish of each prerequisite, each
worker must run at most one task at a time, and total work <= workers * makespan; Graham's bound
makespan <= (2 - 1/P) * OPT is asserted using the minimum makespan over all valid list orders as a
certified upper bound on OPT (the minimum over feasible schedules is always >= OPT, so the
inequality stays valid); random LCG instances with n = 6 and P = 2, 3 and the real-thread executor
over the same DAGs close the loop; and a cyclic graph must be rejected by both simulations.
"""

import heapq
import threading
from itertools import permutations


class Task:
    __slots__ = ("tid", "duration", "deps")

    def __init__(self, tid, duration, deps=()):
        self.tid = tid
        self.duration = duration
        self.deps = list(deps)

    def __repr__(self):
        return "Task(%r, %r, %r)" % (self.tid, self.duration, self.deps)


def by_id_rule(task):
    return (task.tid,)


def longest_first_rule(task):
    return (-task.duration, task.tid)


def _prepare(tasks):
    ids = [task.tid for task in tasks]
    assert len(set(ids)) == len(ids), "task ids must be unique"
    by_id = {task.tid: task for task in tasks}
    indegree = {task.tid: len(task.deps) for task in tasks}
    dependants = {task.tid: [] for task in tasks}
    for task in tasks:
        for dep in task.deps:
            if dep not in by_id:
                raise ValueError("unknown prerequisite %r" % (dep,))
            dependants[dep].append(task.tid)
    for tid in dependants:
        dependants[tid].sort()
    return by_id, indegree, dependants


def simulate_event(tasks, workers, rule=by_id_rule):
    """Event-driven list schedule. Returns (schedule, makespan) with schedule[tid] = (s, f, w)."""
    if workers <= 0:
        raise ValueError("need at least one worker")
    by_id, indegree, dependants = _prepare(tasks)
    free = list(range(workers))
    heapq.heapify(free)
    ready = []
    for task in tasks:
        if indegree[task.tid] == 0:
            heapq.heappush(ready, (rule(task), task.tid))
    busy = []
    schedule = {}
    now = 0
    remaining = len(tasks)
    while remaining > 0:
        while free and ready:
            _, tid = heapq.heappop(ready)
            worker = heapq.heappop(free)
            task = by_id[tid]
            schedule[tid] = (now, now + task.duration, worker)
            heapq.heappush(busy, (now + task.duration, worker, tid))
            remaining -= 1
        if not busy:
            raise ValueError("cyclic dependencies: nothing is runnable")
        now = busy[0][0]
        while busy and busy[0][0] == now:
            _, worker, tid = heapq.heappop(busy)
            heapq.heappush(free, worker)
            for dependant in dependants[tid]:
                indegree[dependant] -= 1
                if indegree[dependant] == 0:
                    heapq.heappush(ready, (rule(by_id[dependant]), dependant))
    makespan = max(finish for _, finish, _ in schedule.values()) if schedule else 0
    return schedule, makespan


def simulate_tick(tasks, workers, rule=by_id_rule):
    """Second, independent formulation: step the clock one tick at a time."""
    by_id, indegree, dependants = _prepare(tasks)
    current = [None] * workers
    finishes = [0] * workers
    assigned = set()
    schedule = {}
    now = 0
    while len(assigned) < len(tasks):
        # Quiesce inside every tick: completions first, then greedy filling, until nothing moves.
        # (Zero-duration tasks are the reason the tick formulation needs this inner loop.)
        progress = True
        while progress:
            progress = False
            for worker in range(workers):
                tid = current[worker]
                if tid is not None and finishes[worker] <= now:
                    current[worker] = None
                    for dependant in dependants[tid]:
                        indegree[dependant] -= 1
                    progress = True
            for worker in range(workers):
                if current[worker] is not None:
                    continue
                candidates = [task for task in tasks
                              if task.tid not in assigned and indegree[task.tid] == 0]
                if not candidates:
                    continue
                chosen = min(candidates, key=rule)
                assigned.add(chosen.tid)
                current[worker] = chosen.tid
                finishes[worker] = now + chosen.duration
                schedule[chosen.tid] = (now, finishes[worker], worker)
                progress = True
        now += 1
        if now > 10 ** 6:
            raise ValueError("cyclic dependencies: nothing is runnable")
    makespan = max(finish for _, finish, _ in schedule.values()) if schedule else 0
    return schedule, makespan


def critical_path(tasks):
    """Longest weighted path by DP over a topological order."""
    by_id, indegree, dependants = _prepare(tasks)
    order = []
    ready = sorted(tid for tid in indegree if indegree[tid] == 0)
    local = dict(indegree)
    while ready:
        tid = ready.pop(0)
        order.append(tid)
        for dependant in dependants[tid]:
            local[dependant] -= 1
            if local[dependant] == 0:
                ready.append(dependant)
    if len(order) != len(tasks):
        raise ValueError("cyclic dependencies")
    longest = {}
    for tid in order:
        task = by_id[tid]
        longest[tid] = task.duration + max((longest[dep] for dep in task.deps), default=0)
    return max(longest.values()) if longest else 0


def brute_force_critical_path(tasks):
    """Independent check: enumerate every root-to-leaf path explicitly."""
    by_id, _, dependants = _prepare(tasks)
    best = 0

    def walk(tid, total):
        nonlocal best
        total += by_id[tid].duration
        best = max(best, total)
        for dependant in dependants[tid]:
            walk(dependant, total)

    roots = [task.tid for task in tasks if not task.deps]
    for root in roots:
        walk(root, 0)
    return best


def is_valid_schedule(tasks, schedule, workers):
    by_id = {task.tid: task for task in tasks}
    if set(schedule) != set(by_id):
        return False
    for task in tasks:
        start, finish, worker = schedule[task.tid]
        if finish - start != task.duration or not (0 <= worker < workers):
            return False
        for dep in task.deps:
            if schedule[dep][1] > start:
                return False
    for wid in range(workers):
        intervals = sorted((s, f) for tid, (s, f, w) in schedule.items() if w == wid)
        for (_, first_end), (second_start, _) in zip(intervals, intervals[1:]):
            if second_start < first_end:
                return False
    return True


def best_list_makespan(tasks, workers, limit=6):
    """Minimum makespan over all list orders given by topological permutations (>= OPT)."""
    ids = sorted(task.tid for task in tasks)
    if len(ids) > limit:
        raise ValueError("enumeration limited to %d tasks" % limit)
    best = None
    for candidate in permutations(ids):
        rank = {tid: index for index, tid in enumerate(candidate)}
        try:
            _, makespan = simulate_event(tasks, workers, lambda task: (rank[task.tid],))
        except ValueError:
            continue
        if best is None or makespan < best:
            best = makespan
    return best


def threaded_run(tasks, workers, recorded=None):
    """Real threads pulling ready tasks under a lock/condition. Asserts only stable invariants."""
    by_id, indegree, dependants = _prepare(tasks)
    lock = threading.Lock()
    condition = threading.Condition(lock)
    ready = sorted(tid for tid in indegree if indegree[tid] == 0)
    executed = []
    per_worker = {}
    remaining = [len(tasks)]

    def worker(worker_id):
        while True:
            with condition:
                while not ready and remaining[0] > 0:
                    condition.wait()
                if remaining[0] == 0:
                    return
                tid = ready.pop(0)
                remaining[0] -= 1
                executed.append(tid)                   # logged under the lock: a linearisation
                per_worker.setdefault(worker_id, []).append(tid)
                task = by_id[tid]
                deps = list(task.deps)
            for dep in deps:                           # dependencies were logged before this
                assert dep in executed
            with condition:
                for dependant in dependants[tid]:
                    indegree[dependant] -= 1
                    if indegree[dependant] == 0:
                        ready.append(dependant)
                condition.notify_all()

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert all(not thread.is_alive() for thread in threads), "worker threads must finish"
    assert sorted(executed) == sorted(by_id), "every task runs exactly once"
    assert len(set(executed)) == len(executed)
    assert all(0 <= wid < workers for wid in per_worker)
    if recorded is not None:
        recorded.append((list(executed), {k: list(v) for k, v in per_worker.items()}))
    return executed, per_worker


def lcg(seed):
    return (seed * 75 + 74) % 65537


def random_dag(n, seed, edge_chance=35):
    tasks = []
    for i in range(n):
        deps = []
        for j in range(i):
            seed = lcg(seed)
            if seed % 100 < edge_chance:
                deps.append("t%d" % j)
        seed = lcg(seed)
        tasks.append(Task("t%d" % i, 1 + seed % 9, deps))
    return tasks


def main():
    # --- Hand-computed example. --------------------------------------------------------------
    hand = [Task("A", 3), Task("B", 2), Task("C", 4), Task("D", 1, ["A", "B"]),
            Task("E", 1, ["C"])]
    schedule, makespan = simulate_event(hand, 2)
    assert makespan == 7, makespan
    assert schedule == {"A": (0, 3, 0), "B": (0, 2, 1), "C": (2, 6, 1),
                        "D": (3, 4, 0), "E": (6, 7, 0)}, schedule
    assert is_valid_schedule(hand, schedule, 2)
    tick_schedule, tick_makespan = simulate_tick(hand, 2)
    assert tick_schedule == schedule and tick_makespan == makespan
    assert critical_path(hand) == 5 and brute_force_critical_path(hand) == 5
    assert makespan >= critical_path(hand)
    assert makespan <= (2 - 1 / 2) * 5 + 1e-9, "Graham bound with OPT >= critical path"
    # Three workers shorten it to the critical path.
    three, three_makespan = simulate_event(hand, 3)
    assert three_makespan == 5 and is_valid_schedule(hand, three, 3)
    # One worker is the total work, however many times the rule changes.
    for rule in (by_id_rule, longest_first_rule):
        single, single_makespan = simulate_event(hand, 1, rule)
        assert single_makespan == sum(task.duration for task in hand) == 11
        assert is_valid_schedule(hand, single, 1)

    # --- Event and tick formulations agree everywhere. ---------------------------------------
    for n in (5, 6, 7):
        for seed in (13, 47, 89):
            tasks = random_dag(n, seed)
            for workers in (1, 2, 3, n):
                for rule in (by_id_rule, longest_first_rule):
                    event, makespan = simulate_event(tasks, workers, rule)
                    tick, tick_makespan = simulate_tick(tasks, workers, rule)
                    assert event == tick, (n, seed, workers)
                    assert makespan == tick_makespan == max(f for _, f, _ in event.values())
                    assert is_valid_schedule(tasks, event, workers)
                    assert makespan >= max(critical_path(tasks),
                                           -(-sum(t.duration for t in tasks) // workers))
                    assert sum(t.duration for t in tasks) <= workers * makespan
                    assert critical_path(tasks) == brute_force_critical_path(tasks)
                    if workers == 1:
                        assert makespan == sum(t.duration for t in tasks)
                    if workers >= n:
                        assert makespan == critical_path(tasks)

    # --- Graham's bound against the best enumerated list order. ------------------------------
    for seed in (13, 47, 89):
        tasks = random_dag(6, seed)
        for workers in (2, 3):
            best = best_list_makespan(tasks, workers)
            for rule in (by_id_rule, longest_first_rule):
                _, makespan = simulate_event(tasks, workers, rule)
                assert makespan >= max(critical_path(tasks),
                                       -(-sum(t.duration for t in tasks) // workers))
                assert makespan <= (2 - 1 / workers) * best + 1e-9, (seed, workers, makespan, best)

    # The scheduling rule matters and neither rule dominates: on seed 13 with two workers
    # longest-first wins (19 against 21), on seed 89 with three workers id order wins (15 against
    # 16). Both remain valid schedules inside Graham's bound.
    tasks13 = random_dag(6, 13)
    tasks89 = random_dag(6, 89)
    id13, lpt13 = simulate_event(tasks13, 2, by_id_rule)[1], \
        simulate_event(tasks13, 2, longest_first_rule)[1]
    id89, lpt89 = simulate_event(tasks89, 3, by_id_rule)[1], \
        simulate_event(tasks89, 3, longest_first_rule)[1]
    assert lpt13 < id13 and id89 < lpt89, (id13, lpt13, id89, lpt89)
    mixed = [Task("a", 3), Task("b", 3), Task("c", 2), Task("d", 2), Task("e", 2)]
    # A list order can reach the true optimum here: the order 3, 2, 2, 3, 2 packs one worker with
    # 3 + 3 and the other with 2 + 2 + 2 for a makespan of 6, while both simple rules need 7. The
    # enumerated minimum is still only an upper bound on OPT in general (a list schedule never
    # leaves a worker idle), which is exactly why it is safe to use as the OPT proxy above.
    assert critical_path(mixed) == 3 and best_list_makespan(mixed, 2) == 6
    assert simulate_event(mixed, 2, by_id_rule)[1] == 7
    assert simulate_event(mixed, 2, longest_first_rule)[1] == 7
    assert simulate_event(mixed, 1, by_id_rule)[1] == 12
    assert 7 <= (2 - 1 / 2) * best_list_makespan(mixed, 2)
    assert best_list_makespan(tasks13, 2) == 19

    # --- Real threads: the dependency handshake holds for every interleaving. ----------------
    threaded_dags = [hand, mixed, random_dag(6, 13), random_dag(7, 47), random_dag(5, 89)]
    for tasks in threaded_dags:
        for workers in (1, 2, 3, len(tasks)):
            executed, per_worker = threaded_run(tasks, workers)
            assert sorted(executed) == sorted(task.tid for task in tasks)
            position = {tid: index for index, tid in enumerate(executed)}
            by_id = {task.tid: task for task in tasks}
            for task in tasks:
                for dep in task.deps:
                    assert position[dep] < position[task.tid], "dependency violated"
            assert sum(len(v) for v in per_worker.values()) == len(tasks)
    # Repeated real-thread runs stay inside the same invariants.
    recorded = []
    for _ in range(5):
        threaded_run(random_dag(6, 13), 3, recorded=recorded)
    assert all(sorted(entry[0]) == sorted(recorded[0][0]) for entry in recorded)
    assert len(recorded) == 5

    # --- A cycle is rejected, and the empty DAG is a no-op. ---------------------------------
    cyclic = [Task("x", 1, ["y"]), Task("y", 1, ["x"])]
    for simulation in (simulate_event, simulate_tick):
        try:
            simulation(cyclic, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("a cyclic DAG must be rejected")
    assert simulate_event([], 3) == ({}, 0)
    assert critical_path([Task("z", 4)]) == 4
    lonely = [Task("z", 4), Task("w", 0)]
    schedule, makespan = simulate_event(lonely, 1)
    assert makespan == 4 and schedule["w"][1] == 0 and schedule["z"] == (0, 4, 0)
    assert simulate_tick(lonely, 1)[0] == schedule
    assert brute_force_critical_path(lonely) == 4


if __name__ == "__main__":
    main()
    print("PASS")
