"""Task 675 - Implement instruction scheduling heuristics for pipelined CPU model.  (Tier: Expert, Beyond — Compilers, Languages & Solvers)

Static list scheduling for a simple in-order pipelined machine: instructions are DAG nodes
with an operation class that fixes their latency (a load takes three cycles before its
result is available, a multiply four, an ALU op one), the machine issues at most `width`
instructions per cycle, and an instruction may only start once every predecessor has
completed.

Three schedulers are compared:
  * `in_order_schedule` - the baseline: keep program order, place each instruction at the
    earliest cycle that respects both dependencies and the issue width;
  * `list_schedule` - the classic heuristic: at every cycle, among the ready instructions,
    issue the `width` highest ones by critical-path height (the longest latency path to a
    sink), ties broken by program order;
  * `optimal_makespan` - brute force over all dependency-respecting orders with the same
    earliest-placement rule, which is exact because some optimal schedule is always
    reproduced by list scheduling in the order of its start times.

Verified on every schedule: dependencies are honoured with the full latency gap, no cycle
issues more than `width` instructions, and every instruction is scheduled. Then the
bounds: critical path <= list <= in-order baseline, list >= optimal, the sweep must contain
instances where list scheduling really beats the in-order baseline, and the makespan is
monotone in the latencies (raising a latency can never shorten the schedule).

Simplified: a single issue width for all instruction classes (no separate load/store or
multiply units), no register-file or bus conflicts, no branch prediction, and the DAG is
given rather than extracted from a basic block.
"""

from itertools import permutations


def check(ok, message):
    if not ok:
        raise AssertionError(message)


LATENCY = {"ALU": 1, "LOAD": 3, "STORE": 1, "MUL": 4, "BR": 1}


class Program:
    """A scheduling region: named instructions with a class and predecessor names."""

    def __init__(self, specs):
        self.kinds, self.deps, self.names = [], [], []
        index = {}
        for name, kind, deps in specs:
            if kind not in LATENCY:
                raise ValueError("unknown operation class %r" % (kind,))
            for dep in deps:
                if dep not in index:
                    raise ValueError("unknown dependency %r of %s" % (dep, name))
            index[name] = len(self.names)
            self.names.append(name)
            self.kinds.append(kind)
            self.deps.append([index[dep] for dep in deps])
        self.succ = [[] for _ in self.names]
        for i, deps in enumerate(self.deps):
            for dep in deps:
                self.succ[dep].append(i)
        if self._has_cycle():
            raise ValueError("the dependency graph must be acyclic")
        self.heights = self._heights()

    def _has_cycle(self):
        state = [0] * len(self.names)

        def visit(node):
            if state[node] == 1:
                return True
            if state[node] == 2:
                return False
            state[node] = 1
            for nxt in self.succ[node]:
                if visit(nxt):
                    return True
            state[node] = 2
            return False

        return any(visit(i) for i in range(len(self.names)))

    def _heights(self):
        """Critical-path height: the longest latency path from the node to a sink."""
        heights = [0] * len(self.names)

        def height(node):
            if heights[node]:
                return heights[node]
            heights[node] = LATENCY[self.kinds[node]] + max(
                (height(s) for s in self.succ[node]), default=0)
            return heights[node]

        for i in range(len(self.names)):
            height(i)
        return heights

    def critical_path(self):
        return max(self.heights)


def earliest_placement(program, order, width):
    """Place instructions in the given order at the earliest legal cycle."""
    if width < 1:
        raise ValueError("the issue width must be at least one")
    start = [None] * len(program.names)
    occupied = {}
    for node in order:
        cycle = 0
        for dep in program.deps[node]:
            if start[dep] is None:
                raise ValueError("order violates the dependency from %s" % program.names[dep])
            cycle = max(cycle, start[dep] + LATENCY[program.kinds[dep]])
        while occupied.get(cycle, 0) >= width:
            cycle += 1
        start[node] = cycle
        occupied[cycle] = occupied.get(cycle, 0) + 1
    return start


def in_order_schedule(program, width):
    return earliest_placement(program, list(range(len(program.names))), width)


def list_schedule(program, width):
    """Critical-path priority list scheduling, cycle by cycle."""
    if width < 1:
        raise ValueError("the issue width must be at least one")
    n = len(program.names)
    start = [None] * n
    issue_count = {}
    scheduled = 0
    cycle = 0
    while scheduled < n:
        ready = [i for i in range(n) if start[i] is None and all(
            start[dep] is not None
            and start[dep] + LATENCY[program.kinds[dep]] <= cycle for dep in program.deps[i])]
        ready.sort(key=lambda i: (-program.heights[i], i))
        for node in ready[:width - issue_count.get(cycle, 0)]:
            start[node] = cycle
            issue_count[cycle] = issue_count.get(cycle, 0) + 1
            scheduled += 1
        cycle += 1
        if cycle > 8 * n + 16:
            raise AssertionError("list scheduler failed to converge")
    return start


def makespan(program, start):
    return max(start[i] + LATENCY[program.kinds[i]] for i in range(len(program.names)))


def schedule_problems(program, start, width):
    """Every way a schedule could be illegal (empty list means valid)."""
    problems = []
    n = len(program.names)
    if any(s is None for s in start):
        problems.append("unscheduled instruction")
    counts = {}
    for i, s in enumerate(start):
        if s is None:
            continue
        if s < 0:
            problems.append("%s starts before cycle 0" % program.names[i])
        counts[s] = counts.get(s, 0) + 1
        for dep in program.deps[i]:
            needed = start[dep] + LATENCY[program.kinds[dep]]
            if s < needed:
                problems.append("%s starts %d cycle(s) before its operand is ready"
                                % (program.names[i], needed - s))
    for cycle, count in counts.items():
        if count > width:
            problems.append("cycle %d issues %d instructions (width %d)" % (cycle, count, width))
    return problems


def optimal_makespan(program, width):
    """Brute force: the best earliest-placement schedule over all legal orders."""
    n = len(program.names)
    best = None
    for order in permutations(range(n)):
        position = {node: i for i, node in enumerate(order)}
        if any(position[dep] > position[node]
               for node in range(n) for dep in program.deps[node]):
            continue
        span = makespan(program, earliest_placement(program, order, width))
        if best is None or span < best:
            best = span
    return best


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_program(seed, size=7):
    """Random DAG: every instruction may depend on earlier instructions only."""
    state = seed
    kinds = ("ALU", "LOAD", "MUL", "STORE", "BR")
    specs = []
    for i in range(size):
        state, raw = splitmix64(state)
        kind = kinds[raw % len(kinds)]
        deps = []
        for j in range(i):
            state, raw = splitmix64(state)
            if raw % 3 == 0:
                deps.append("i%d" % j)
        specs.append(("i%d" % i, kind, deps))
    return Program(specs)


def main():
    # ---- a load-use stall in program order -------------------------------------------
    hand = Program([("a", "LOAD", []), ("b", "ALU", ["a"]), ("c", "ALU", ["b"])])
    starts = in_order_schedule(hand, 2)
    check(starts == [0, 3, 4], "the ALU op must wait exactly for the load: %r" % (starts,))
    check(schedule_problems(hand, starts, 2) == [], "the baseline schedule must be legal")
    check(makespan(hand, starts) == 5, "makespan = load latency + two ALU ops")
    check(hand.critical_path() == 5, "the critical path must match the makespan here")
    check(list_schedule(hand, 2) == starts, "list scheduling agrees on this chain")
    check(optimal_makespan(hand, 2) == 5, "brute force must find the same optimum")

    # ---- the heuristic reorders independent work -------------------------------------
    reorder = Program([("a", "LOAD", []), ("b", "ALU", []), ("c", "ALU", ["a"]),
                       ("d", "MUL", ["b"]), ("e", "ALU", ["b"])])
    baseline = in_order_schedule(reorder, 2)
    listed = list_schedule(reorder, 2)
    check(schedule_problems(reorder, listed, 2) == [], "the list schedule must be legal")
    check(makespan(reorder, listed) <= makespan(reorder, baseline),
          "list scheduling must never be worse than program order")
    check(makespan(reorder, listed) >= reorder.critical_path(),
          "no schedule can beat the critical path")
    check(makespan(reorder, listed) >= optimal_makespan(reorder, 2), "the optimum is a bound")

    # ---- randomized sweep -------------------------------------------------------------
    improvements = exact_hits = 0
    for i in range(60):
        program = random_program(0x51EE + i * 7919, 5 + i % 3)
        width = 1 + i % 3
        listed = list_schedule(program, width)
        baseline = in_order_schedule(program, width)
        problems = schedule_problems(program, listed, width)
        check(problems == [], "illegal list schedule (%d, width %d): %r"
              % (i, width, problems))
        check(schedule_problems(program, baseline, width) == [],
              "illegal baseline schedule (%d)" % i)
        check(makespan(program, listed) <= makespan(program, baseline),
              "list scheduling lost to program order (%d, width %d)" % (i, width))
        check(makespan(program, listed) >= program.critical_path(),
              "critical path violated (%d)" % i)
        if makespan(program, listed) < makespan(program, baseline):
            improvements += 1
        if len(program.names) <= 7:
            best = optimal_makespan(program, width)
            check(optimal_makespan(program, width) == best, "the brute force is deterministic")
            check(makespan(program, listed) >= best, "optimum beaten (%d, width %d)" % (i, width))
            if makespan(program, listed) == best:
                exact_hits += 1
        # monotonicity: with the same order, a slower operation cannot shorten the schedule
        stretched = Program([(name, "MUL" if i == 0 and program.kinds[0] != "MUL" else kind,
                              [program.names[j] for j in program.deps[i]])
                             for i, (name, kind) in enumerate(zip(program.names, program.kinds))])
        order = sorted(range(len(program.names)), key=lambda node: listed[node])
        if LATENCY[stretched.kinds[0]] >= LATENCY[program.kinds[0]]:
            check(makespan(stretched, earliest_placement(stretched, order, width)) >=
                  makespan(program, earliest_placement(program, order, width)),
                  "raising a latency may not shorten a fixed-order schedule (%d)" % i)
    check(improvements > 5,
          "the sweep must show the heuristic winning somewhere (%d)" % improvements)
    check(exact_hits > 0, "the heuristic must be optimal on some instances (%d)" % exact_hits)

    # ---- width one serializes issue ---------------------------------------------------
    serial = random_program(0x2222, 6)
    single = list_schedule(serial, 1)
    check(len(set(single)) == len(single),
          "with one issue slot per cycle every instruction gets its own cycle: %r" % (single,))
    check(makespan(serial, single) >= len(serial.names),
          "a single-issue machine needs at least one cycle per instruction")
    wide = list_schedule(serial, 4)
    check(sum(wide) <= sum(single), "a wider machine cannot issue later (total start times)")

    # ---- errors and determinism -------------------------------------------------------
    rejected = 0
    for bad in (lambda: Program([("a", "ALU", ["ghost"])]),
                lambda: Program([("a", "ALU", []), ("b", "FROB", ["a"])]),
                lambda: Program([("a", "ALU", ["a"])]),
                lambda: earliest_placement(hand, [1, 0, 2], 2),
                lambda: list_schedule(hand, 0)):
        try:
            bad()
        except (ValueError, IndexError):
            rejected += 1
    check(rejected == 5, "invalid programs and orders must be rejected (%d of 5)" % rejected)
    check(list_schedule(hand, 2) == list_schedule(hand, 2), "scheduling must be deterministic")

    pinned = Program([("l0", "LOAD", []), ("l1", "LOAD", []), ("x", "MUL", ["l0"]),
                      ("y", "ALU", ["l1"]), ("z", "ALU", ["x", "y"])])
    pinned_start = list_schedule(pinned, 2)
    check(schedule_problems(pinned, pinned_start, 2) == [], "pinned schedule must be legal")
    gap = min(pinned_start[j] - pinned_start[i] - LATENCY[pinned.kinds[i]]
              for i in range(len(pinned.names)) for j in pinned.succ[i])
    check(gap >= 0, "every dependency keeps the full latency gap")

    print("hand chain: %r, makespan %d; reorder: baseline %d vs list %d (critical path %d)"
          % (starts, makespan(hand, starts), makespan(reorder, baseline),
             makespan(reorder, listed), reorder.critical_path()))
    print("sweep: %d improvements over program order, %d provably optimal of 60"
          % (improvements, exact_hits))
    print("PASS")


if __name__ == "__main__":
    main()
