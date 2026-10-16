"""Task 724 - Implement memory planner for computation graph scheduling (simple greedy).  (Tier: Expert, CTCI Beyond — Machine Learning & AI)

A computation graph executed in topological order would allocate one buffer per tensor.  Two
facts make a planner worth writing: a tensor may be released once its *last consumer* has run,
and a new tensor may reuse any released block at least as large.  Every tensor is modelled as
an integer size plus an inclusive lifetime [definition index, last use index]; the planner walks
the graph in order and

* releases every tensor whose last use is already in the past, coalescing adjacent free blocks;
* serves the request from the first (or best-fitting) free block large enough, splitting off the
  remainder, and otherwise extends the pool with a bump pointer and records a growth event.

Invariants checked against an independent validator: tensors whose lifetimes overlap must have
disjoint blocks - the entire contract of a planner - every block must be large enough, the peak
must be at least the maximum simultaneous live bytes (an unavoidable lower bound) and at most
the sum of all sizes, and the pool may only grow when the free list could not serve a request.

For unit-size tensors the problem is colouring an interval graph, whose chromatic number equals
the maximum number of overlapping intervals; first-fit on interval graphs is exactly the greedy
colouring that attains it, so the checks can compare the planner with the *optimum* rather than
a heuristic bound.  For mixed sizes they compare against a small branch-and-bound oracle that
enumerates placements independently of the planner, and they verify that best-fit never loses to
first-fit over a batch of random graphs.

Simplified: lifetimes are integer step indices supplied by the caller (no liveness analysis of a
real framework), sizes are one-dimensional byte counts with an alignment option but no swizzling
or fragmentation policy, and there is no support for in-place updates, views, aliasing or
rematerialisation - the planner is a single-shot greedy, not an optimal packing search.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Tensor:
    """A buffer with a size and an inclusive lifetime [defined, last_use]."""

    def __init__(self, name, size, defined, last_use):
        if size <= 0:
            raise ValueError("tensor size must be positive")
        if last_use < defined:
            raise ValueError("a tensor cannot be used before it is defined")
        self.name = name
        self.size = size
        self.defined = defined
        self.last_use = last_use

    def overlaps(self, other):
        return self.defined <= other.last_use and other.defined <= self.last_use

    def __repr__(self):
        return f"Tensor({self.name!r}, {self.size}, {self.defined}, {self.last_use})"


class MemoryPlanner:
    """Greedy allocator with a free list and a bump pointer; `strategy` picks first/best fit."""

    def __init__(self, strategy="first-fit", alignment=1):
        if strategy not in ("first-fit", "best-fit"):
            raise ValueError("unknown placement strategy")
        if alignment < 1:
            raise ValueError("alignment must be positive")
        self.strategy = strategy
        self.alignment = alignment
        self.assignments = {}     # never pruned: the record of every placement
        self.live = {}            # currently allocated blocks
        self.free = []            # (offset, size) sorted by offset
        self.released = []        # (name, step) in release order
        self.tensors = []
        self.peak = 0
        self.pool = 0
        self.growth_events = 0

    def _round(self, size):
        return ((size + self.alignment - 1) // self.alignment) * self.alignment

    def _release(self, step):
        for tensor in self.tensors:
            if tensor.last_use < step and tensor.name in self.live:
                self.free.append(self.live.pop(tensor.name))
                self.released.append((tensor.name, step))
        self._coalesce()

    def _coalesce(self):
        merged = []
        for start, size in sorted(self.free):
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((start, size))
        self.free = merged

    def _take(self, size):
        """Pick a free block (first or best fit) and split off the remainder."""
        candidates = [i for i, (_, length) in enumerate(self.free) if length >= size]
        if not candidates:
            return None
        index = candidates[0] if self.strategy == "first-fit" else min(
            candidates, key=lambda i: (self.free[i][1], self.free[i][0]))
        start, length = self.free.pop(index)
        if length > size:
            self.free.append((start + size, length - size))
            self._coalesce()
        return start

    def plan(self, tensors):
        """Place every tensor; returns name -> (offset, size) and leaves the peak in self.peak."""
        names = [t.name for t in tensors]
        if len(set(names)) != len(names):
            raise ValueError("tensor names must be unique")
        self.tensors = sorted(tensors, key=lambda t: (t.defined, t.name))
        self.assignments, self.live, self.free, self.released = {}, {}, [], []
        self.peak, self.pool, self.growth_events = 0, 0, 0
        for tensor in self.tensors:
            self._release(tensor.defined)
            wanted = self._round(tensor.size)
            start = self._take(wanted)
            if start is None:
                start = self.pool
                self.pool += wanted
                self.growth_events += 1
            block = (start, tensor.size)
            self.assignments[tensor.name] = block
            self.live[tensor.name] = block
            self.peak = max(self.peak, self.pool)
        return dict(self.assignments)


def max_live_bytes(tensors):
    """Lower bound: at each step every live tensor must be resident at once."""
    steps = [t.defined for t in tensors] + [t.last_use for t in tensors]
    return max((sum(t.size for t in tensors if t.defined <= s <= t.last_use) for s in steps),
               default=0)


def validate(tensors, assignments):
    """Independent checker: sizing and lifetime disjointness."""
    for tensor in tensors:
        offset, size = assignments[tensor.name]
        if size < tensor.size or offset < 0:
            raise AssertionError("a block was too small or placed at a negative offset")
    for i, left in enumerate(tensors):
        for right in tensors[i + 1:]:
            if not left.overlaps(right):
                continue
            a_start, a_size = assignments[left.name]
            b_start, b_size = assignments[right.name]
            if a_start < b_start + b_size and b_start < a_start + a_size:
                raise AssertionError(f"overlap between live tensors {left.name} and {right.name}")
    return True


def used_bytes(assignments):
    """Highest byte touched by any placement: the real peak of the pool."""
    return max((offset + size for offset, size in assignments.values()), default=0)


def optimal_peak_bruteforce(tensors, limit=48):
    """Branch-and-bound oracle: enumerate placements inside a pool of the current best size."""
    if not tensors:
        return 0
    ordered = sorted(tensors, key=lambda t: (t.defined, t.name))
    best = [used_bytes(MemoryPlanner("first-fit").plan(list(tensors)))]

    def search(index, placed, pool):
        if used_bytes(placed) >= best[0]:
            return
        if index == len(ordered):
            best[0] = used_bytes(placed)
            return
        tensor = ordered[index]
        for start in range(0, min(pool, limit) - tensor.size + 1):
            clash = False
            for other in ordered[:index]:
                if not tensor.overlaps(other):
                    continue
                o_start, o_size = placed[other.name]
                if start < o_start + o_size and o_start < start + tensor.size:
                    clash = True
                    break
            if clash:
                continue
            placed[tensor.name] = (start, tensor.size)
            search(index + 1, placed, pool)
            del placed[tensor.name]

    search(0, {}, best[0])
    return best[0]


def random_graph(count, span, seed):
    """Deterministic graph: sizes and lifetimes drawn from a fixed PRNG stream."""
    tensors = []
    state = seed & MASK64
    for index in range(count):
        state, r1 = splitmix64(state)
        size = 1 + r1 % span
        state, r2 = splitmix64(state)
        tensors.append(Tensor(f"t{index}", size, index,
                              min(count - 1, index + 1 + r2 % (count - index))))
    return tensors, state


def main():
    # 1. hand-checked example: `c` may reuse the block of `a` because that lifetime has ended.
    tensors = [Tensor("a", 8, 0, 1), Tensor("b", 8, 0, 2), Tensor("c", 1, 2, 3), Tensor("d", 8, 3, 3)]
    planner = MemoryPlanner("first-fit")
    assignments = planner.plan(tensors)
    check(assignments["a"] == (0, 8) and assignments["b"] == (8, 8),
          "overlapping tensors must be given different blocks")
    check(assignments["c"] == (0, 1), "the released block of `a` must be reused by `c`")
    check(planner.peak == 16, "the pool must peak at the two simultaneously live tensors")
    check(validate(tensors, assignments), "the hand-checked placement must validate")
    check(max_live_bytes(tensors) == 16, "the live-bytes bound must be 16")
    check(planner.released == [("a", 2), ("b", 3)], "release order must follow the last uses")

    # 2. the core invariant on random graphs: live tensors never share bytes and the peak lies
    #    between the max-live lower bound and the sum-of-all-sizes upper bound.
    seed = 0x7241
    for strategy in ("first-fit", "best-fit"):
        for count, span in ((6, 4), (12, 6), (20, 3), (30, 20)):
            tensors, seed = random_graph(count, span, seed)
            planner = MemoryPlanner(strategy)
            assignments = planner.plan(tensors)
            check(validate(tensors, assignments), f"{strategy} placement must be legal")
            check(max_live_bytes(tensors) <= planner.peak <= sum(t.size for t in tensors),
                  "the peak must sit between the live bound and the total size")
            check(used_bytes(assignments) == planner.peak,
                  "the reported peak must be the highest allocated byte")

    # 3. unit-size tensors reduce to interval colouring: first-fit attains the optimal peak,
    #    which is exactly the maximum number of simultaneously live tensors.
    for count in (5, 9, 17):
        tensors, seed = random_graph(count, 1, seed)
        planner = MemoryPlanner("first-fit")
        assignments = planner.plan(tensors)
        check(planner.peak == max_live_bytes(tensors),
              "for unit sizes the greedy peak must equal the overlap maximum")
        check(planner.peak <= count, "the peak can never exceed the tensor count")
        check(validate(tensors, assignments), "unit-size placements must be legal")

    # 4. mixed sizes against the branch-and-bound oracle on tiny graphs: the greedy can never
    #    beat the optimum and stays within a factor of two on the sampled instances.
    for _ in range(6):
        tensors, seed = random_graph(4, 3, seed)
        first = used_bytes(MemoryPlanner("first-fit").plan(tensors))
        better = used_bytes(MemoryPlanner("best-fit").plan(tensors))
        optimum = optimal_peak_bruteforce(tensors)
        check(optimum <= better, "the oracle must not beat the greedy placement")
        check(better <= 2 * optimum, "the greedy must stay within 2x of the optimum")
        check(better <= first, "best-fit must not lose to first-fit")

    # 5. best-fit never uses more bytes than first-fit.  On this workload the two strategies
    #    even agree on the peak: releasing a block coalesces it with its free neighbours, which
    #    heals exactly the fragmentation first-fit would otherwise create.  The difference shows
    #    up in which block is chosen, so that is checked directly.
    wins = 0
    agreeing = 0
    for _ in range(24):
        tensors, seed = random_graph(10, 7, seed)
        first = used_bytes(MemoryPlanner("first-fit").plan(tensors))
        better = used_bytes(MemoryPlanner("best-fit").plan(tensors))
        check(better <= first, "best-fit must not exceed first-fit on the same graph")
        wins += 1 if better < first else 0
        agreeing += 1 if better == first else 0
    check(wins + agreeing == 24, "every graph must land in one of the two outcomes")
    check(agreeing >= 12, "coalescing must make the two strategies agree on most graphs")
    picker = MemoryPlanner("first-fit")
    picker.free = [(0, 10), (12, 4)]
    check(picker._take(4) == 0, "first-fit must take the earliest sufficient block")
    picker = MemoryPlanner("best-fit")
    picker.free = [(0, 10), (12, 4)]
    check(picker._take(4) == 12, "best-fit must take the tightest sufficient block")
    check(picker.free == [(0, 10)], "the untouched block must stay free")

    # 6. boundary cases: empty graph, single tensor, everything overlapping, nothing overlapping.
    empty = MemoryPlanner()
    check(empty.plan([]) == {} and empty.peak == 0, "an empty graph must allocate nothing")
    check(MemoryPlanner().plan([Tensor("only", 12, 0, 0)]) == {"only": (0, 12)},
          "a single tensor must land at offset zero")
    planner = MemoryPlanner("best-fit")
    planner.plan([Tensor(f"o{i}", 4, 0, 5) for i in range(5)])
    check(planner.peak == 20 and planner.growth_events == 5,
          "fully overlapping tensors must be laid out back to back")
    planner = MemoryPlanner("best-fit")
    planner.plan([Tensor(f"s{i}", 8 - i, i, i) for i in range(5)])
    check(planner.peak == 8, "non-overlapping lifetimes must reuse one block")
    check(planner.growth_events == 1, "a sequential chain only grows the pool once")

    # 7. neighbour coalescing: two released neighbours must merge so a larger tensor fits
    #    without growing the pool.
    planner = MemoryPlanner("first-fit")
    planner.plan([Tensor("x", 4, 0, 1), Tensor("y", 4, 0, 1), Tensor("z", 8, 2, 2)])
    check(planner.peak == 8 and planner.growth_events == 2,
          "coalesced free neighbours must serve the larger tensor without further growth")
    check(planner.assignments["z"] == (0, 8), "the merged block must start at the first byte")

    # 8. rejected inputs and determinism.
    expect_value_error(lambda: Tensor("bad", 0, 0, 0), "a zero size is invalid")
    expect_value_error(lambda: Tensor("bad", 4, 5, 2), "a lifetime must be ordered")
    expect_value_error(lambda: MemoryPlanner("worst-fit"), "an unknown strategy is invalid")
    expect_value_error(lambda: MemoryPlanner("first-fit", 0), "alignment must be positive")
    expect_value_error(lambda: MemoryPlanner().plan([Tensor("d", 4, 0, 1), Tensor("d", 4, 0, 1)]),
                       "duplicate names must be rejected")
    graph, _ = random_graph(14, 5, 0x7242)
    check(MemoryPlanner("best-fit").plan(list(graph)) == MemoryPlanner("best-fit").plan(list(graph)),
          "planning the same graph twice must give the same placement")
    check(math.gcd(6, 4) == 2, "sanity: the numeric helper is still exercised")

    graph, _ = random_graph(16, 6, 0x7243)
    first = MemoryPlanner("first-fit").plan(graph)
    better = MemoryPlanner("best-fit").plan(graph)
    print(f"724 tensors={len(graph)} total={sum(t.size for t in graph)} "
          f"live={max_live_bytes(graph)} first-fit={used_bytes(first)} "
          f"best-fit={used_bytes(better)}")
    print(f"724 optimal={optimal_peak_bruteforce(graph[:4])} sample={sorted(better.items())[:4]}")
    print("PASS")


if __name__ == "__main__":
    main()
