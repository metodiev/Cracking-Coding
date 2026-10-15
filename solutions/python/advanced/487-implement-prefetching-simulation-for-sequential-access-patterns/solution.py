"""Task 487 - Implement prefetching simulation for sequential access patterns. (Tier: Advanced, CTCI Ch 14)

Simulated memory with documented semantics:
- Address space is [0, bound); every access and every prefetched address must be inside it - the
  prefetcher clips at bound-1 so it can never read past the array.
- Sequential pattern detector: the first access of a run is only recorded; every following access
  with addr == last + 1 continues the run and issues speculative prefetches for
  addr + 1 .. min(addr + depth, bound - 1). Any non-consecutive access restarts the detector.
- A prefetch "hit" is an access to an address sitting in the prefetched set (one entry per
  address); a hit consumes the entry. Stale entries may serve later runs, which only raises the
  ratio, never invalidates the documented lower bound: a cold run of length L >= 2 yields at least
  L - 2 prefetch hits (the run's first access and its immediate successor cannot be covered).

The test pins hand cases, then asserts on sequential traces that the per-segment prefetch hit ratio
is >= 0.9 (documented threshold; a fresh run of length L scores (L-2)/L, e.g. 0.9375 at L = 32) and
that no simulated read ever leaves [0, bound).
"""


class Prefetcher:
    def __init__(self, bound, depth):
        self.bound = bound
        self.depth = depth
        self.last = None
        self.pending = set()
        self.hits = 0
        self.accesses = 0
        self.issued = 0
        self.max_prefetch = -1

    def access(self, addr):
        assert 0 <= addr < self.bound, addr
        self.accesses += 1
        hit = addr in self.pending
        if hit:
            self.hits += 1
            self.pending.discard(addr)
        continues = self.last is not None and addr == self.last + 1
        self.last = addr
        if continues:
            for nxt in range(addr + 1, min(addr + self.depth, self.bound - 1) + 1):
                self.pending.add(nxt)
                self.issued += 1
                self.max_prefetch = max(self.max_prefetch, nxt)
        return hit


def sequential_trace(start, length):
    return list(range(start, start + length))


def lcg_values(count, seed, limit):
    values = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % limit)
    return values


def main():
    # Hand case: bound 10, depth 3, sequential 0..6, then reuse a stale prefetch at 9.
    prefetcher = Prefetcher(10, 3)
    hits = [prefetcher.access(addr) for addr in range(7)]
    assert hits == [False, False, True, True, True, True, True], hits
    assert prefetcher.hits == 5
    assert prefetcher.pending == {7, 8, 9}
    assert prefetcher.max_prefetch == 9
    assert prefetcher.issued == 18  # six in-run accesses (1..6), 3 prefetches each
    assert prefetcher.access(9) is True  # stale prefetch from access 6 still counts as a hit
    assert prefetcher.hits == 6 and prefetcher.pending == {7, 8}
    assert prefetcher.access(4) is False  # jump backwards: no hit, no prefetch
    assert prefetcher.hits == 6 and prefetcher.issued == 18

    # Bound clipping: a full run to the last slot never prefetches past bound - 1.
    prefetcher = Prefetcher(10, 3)
    hits = [prefetcher.access(addr) for addr in range(10)]
    assert hits == [False, False] + [True] * 8, hits
    assert prefetcher.hits == 8
    assert prefetcher.max_prefetch == 9
    assert prefetcher.pending == set()

    # Fresh sequential run of length 40 with depth 4: hits == 38, ratio 0.95 >= 0.9.
    prefetcher = Prefetcher(64, 4)
    for addr in sequential_trace(0, 40):
        prefetcher.access(addr)
    assert prefetcher.hits == 38 and prefetcher.accesses == 40
    assert prefetcher.hits / prefetcher.accesses >= 0.9
    assert prefetcher.max_prefetch == 43  # the run ends at 39; 40..43 stay pending

    # Mixed trace: sequential segments separated by LCG jumps; per-segment ratio >= 0.9.
    prefetcher = Prefetcher(128, 4)
    segment_hits = []
    segment_lengths = []
    previous = None
    kept = 0
    for jump in lcg_values(4, 487, 80):
        trace = sequential_trace(jump, 32)
        before = prefetcher.hits
        for addr in trace:
            prefetcher.access(addr)
        got = prefetcher.hits - before
        segment_hits.append(got)
        segment_lengths.append(32)
        # The first access of a segment can never be covered; at least L-2 of 32 should be.
        assert got >= 30, (jump, got)
        assert got / 32 >= 0.9
        kept += 1
    assert segment_lengths == [32, 32, 32, 32]
    assert sum(segment_hits) == 120, segment_hits
    assert kept == 4
    assert prefetcher.max_prefetch < 128
    assert prefetcher.hits <= prefetcher.accesses
    ratio = prefetcher.hits / prefetcher.accesses
    assert ratio >= 0.9, ratio

    # Interleave random jump accesses: bounds always respected, no reads past the end.
    prefetcher = Prefetcher(64, 5)
    trace = []
    for value in lcg_values(30, 488, 64):
        trace.append(value)
        if value + 2 < 64:
            trace.extend([value + 1, value + 2])
    for addr in trace:
        assert 0 <= addr < 64
        prefetcher.access(addr)
    assert prefetcher.max_prefetch <= 63
    assert prefetcher.pending <= set(range(64))
    assert prefetcher.hits <= prefetcher.accesses == len(trace)

    # Deterministic pins for the mixed trace.
    prefetcher = Prefetcher(128, 4)
    for jump in lcg_values(4, 487, 80):
        for addr in sequential_trace(jump, 32):
            prefetcher.access(addr)
    assert prefetcher.accesses == 128 and prefetcher.issued == 496, (prefetcher.accesses, prefetcher.issued)
    assert prefetcher.hits == 120
    print("PASS")


if __name__ == "__main__":
    main()
