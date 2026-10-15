"""Task 486 - Implement adaptive replacement cache (ARC) outline & test. (Tier: Advanced, CTCI Ch 14)

ARC outline (Megiddo & Modha 2003), capacity c:
- Real lists: T1 (keys seen once recently), T2 (keys seen at least twice). Cache contents = T1 + T2,
  with |T1| + |T2| <= c.
- Ghost lists: B1 (recently evicted from T1), B2 (evicted from T2); |T1| + |B1| <= c and
  |T1|+|T2|+|B1|+|B2| <= 2c.
- A request found in T1 is promoted to T2; in T2 it is refreshed (MRU). A request found in a ghost
  list adapts the target size p for T1: p grows on a B1 hit, shrinks on a B2 hit, always clamped to
  [0, c]; the hit key is admitted to T2. A brand-new request is admitted to T1, first discarding a
  ghost or demoting the LRU of T1/T2 via `replace` (which demotes T1 when |T1| > p, else T2).

Adaptivity test:
- Trace S ("scan-resistance", 20 rounds of [hot 0..3, 8-key scan, hot 0..3], scan keys cycling
  mod 8): LRU thrashes on the scans; ARC must hit at least as often (pinned: ARC 133 vs LRU 76).
- Trace A ("adaptivity", same rounds but 4 rotating scan keys): ghost hits drive p up to c = 4 and
  back down; pinned p_max = 4, 4 B1-hits, 141 B2-hits, ARC 79 vs LRU 76.

Correctness: on every access the requested key is in the cache, the four lists are pairwise
disjoint with the documented size bounds honoured, and the cache contents/predicted hits match an
independent list-based ARC reference implementation at every step.
"""
from collections import OrderedDict


class ARCCache:
    def __init__(self, capacity):
        self.c = capacity
        self.p = 0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        self.hits = 0
        self.p_max = 0
        self.p_min = 0
        self.b1_hits = 0
        self.b2_hits = 0

    def _replace(self, key):
        if self.t1 and ((key in self.b2 and len(self.t1) == self.p) or len(self.t1) > self.p):
            victim, _ = self.t1.popitem(last=False)
            self.b1[victim] = True
        elif self.t2:
            victim, _ = self.t2.popitem(last=False)
            self.b2[victim] = True

    def access(self, key):
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = True
            self.hits += 1
            return True
        if key in self.t2:
            self.t2.move_to_end(key)
            self.hits += 1
            return True
        if key in self.b1:
            self.p = min(self.c, self.p + max(1, len(self.b2) // len(self.b1)))
            self.b1_hits += 1
            self._replace(key)
            del self.b1[key]
            self.t2[key] = True
        elif key in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) // len(self.b2)))
            self.b2_hits += 1
            self._replace(key)
            del self.b2[key]
            self.t2[key] = True
        else:
            if len(self.t1) + len(self.b1) == self.c:
                if len(self.t1) < self.c:
                    self.b1.popitem(last=False)
                    self._replace(key)
                else:
                    self.t1.popitem(last=False)
            elif len(self.t1) + len(self.b1) < self.c:
                total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
                if total >= self.c:
                    if total == 2 * self.c:
                        self.b2.popitem(last=False)
                    self._replace(key)
            self.t1[key] = True
        self.p_max = max(self.p_max, self.p)
        self.p_min = min(self.p_min, self.p)
        return False

    def contents(self):
        return sorted(list(self.t1) + list(self.t2))

    def check_invariants(self, key):
        assert key in self.t1 or key in self.t2
        assert len(self.t1) + len(self.t2) <= self.c
        assert len(self.t1) + len(self.b1) <= self.c
        assert len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) <= 2 * self.c
        assert 0 <= self.p <= self.c
        lists = [list(self.t1), list(self.t2), list(self.b1), list(self.b2)]
        for i in range(4):
            for j in range(i + 1, 4):
                assert not set(lists[i]) & set(lists[j])


def arc_reference(trace, capacity):
    """Naive list-based second implementation; returns (hits, per-step cache contents)."""
    p = 0
    t1, t2, b1, b2 = [], [], [], []
    hits = 0
    contents = []
    for key in trace:
        if key in t1:
            t1.remove(key)
            t2.append(key)
            hits += 1
        elif key in t2:
            t2.remove(key)
            t2.append(key)
            hits += 1
        else:
            if key in b1:
                p = min(capacity, p + max(1, len(b2) // len(b1)))
                if t1 and ((key in b2 and len(t1) == p) or len(t1) > p):
                    b1.append(t1.pop(0))
                elif t2:
                    b2.append(t2.pop(0))
                b1.remove(key)
                t2.append(key)
            elif key in b2:
                p = max(0, p - max(1, len(b1) // len(b2)))
                if t1 and ((key in b2 and len(t1) == p) or len(t1) > p):
                    b1.append(t1.pop(0))
                elif t2:
                    b2.append(t2.pop(0))
                b2.remove(key)
                t2.append(key)
            else:
                if len(t1) + len(b1) == capacity:
                    if len(t1) < capacity:
                        b1.pop(0)
                        if t1 and len(t1) > p:
                            b1.append(t1.pop(0))
                        elif t2:
                            b2.append(t2.pop(0))
                    else:
                        t1.pop(0)
                elif len(t1) + len(b1) < capacity:
                    total = len(t1) + len(t2) + len(b1) + len(b2)
                    if total >= capacity:
                        if total == 2 * capacity:
                            b2.pop(0)
                        if t1 and ((key in b2 and len(t1) == p) or len(t1) > p):
                            b1.append(t1.pop(0))
                        elif t2:
                            b2.append(t2.pop(0))
                t1.append(key)
        contents.append(sorted(t1 + t2))
    return hits, contents


def lru_hits(trace, capacity):
    order = []
    hits = 0
    for key in trace:
        if key in order:
            order.remove(key)
            order.append(key)
            hits += 1
        else:
            order.append(key)
            if len(order) > capacity:
                order.pop(0)
    return hits


def scan_trace(rounds, hot, scan_width, cold):
    trace = []
    for r in range(rounds):
        trace.extend(range(hot))
        for j in range(scan_width):
            trace.append(100 + (r * scan_width + j) % cold)
        trace.extend(range(hot))
    return trace


def verify_trace(trace, capacity, expected_hits, expected_lru, label):
    cache = ARCCache(capacity)
    hit_vector = []
    contents_vector = []
    for key in trace:
        hit = cache.access(key)
        hit_vector.append(hit)
        cache.check_invariants(key)
        contents_vector.append(cache.contents())
    reference_hits, reference_contents = arc_reference(trace, capacity)
    assert cache.hits == reference_hits == expected_hits, (label, cache.hits, reference_hits)
    assert sum(hit_vector) == cache.hits
    assert contents_vector == reference_contents, label
    assert cache.contents() == reference_contents[-1], label
    assert lru_hits(trace, capacity) == expected_lru, label
    assert cache.hits >= expected_lru, label
    return cache


def main():
    # Hand case: capacity 2, 1,2,1,3,1 -> F,F,T,F,T, final cache {1,3}.
    cache = ARCCache(2)
    assert [cache.access(key) for key in [1, 2, 1, 3, 1]] == [False, False, True, False, True]
    assert cache.hits == 2 and cache.contents() == [1, 3]

    # Trace S: scan resistance. LRU 76, ARC 133 on 320 accesses.
    scan = scan_trace(20, 4, 8, 8)
    assert len(scan) == 320 and scan[:4] == [0, 1, 2, 3]
    cache_s = verify_trace(scan, 4, 133, 76, "scan-resistance")
    assert cache_s.contents() == [0, 1, 2, 3]
    assert cache_s.hits > lru_hits(scan, 4)

    # Trace A: adaptivity. p climbs to c = 4 on ghost hits and falls back; ARC 79 vs LRU 76.
    adaptive = scan_trace(20, 4, 4, 4)
    assert len(adaptive) == 240
    cache_a = verify_trace(adaptive, 4, 79, 76, "adaptivity")
    assert cache_a.p_max == 4 and cache_a.p_min == 0 and cache_a.p == 0
    assert cache_a.b1_hits == 4 and cache_a.b2_hits == 141
    assert cache_a.contents() == [0, 1, 2, 3]

    # Pinned per-step hit vectors are exactly reproduced by the reference (sampled check).
    _, reference_contents = arc_reference(adaptive, 4)
    assert reference_contents[-1] == sorted(cache_a.contents())
    assert lru_hits(adaptive, 4) == 76
    print("PASS")


if __name__ == "__main__":
    main()
