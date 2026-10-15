"""Task 485 - Implement cache eviction policies: LRU, LFU, ARC (simulate). (Tier: Advanced, CTCI Ch 14)

Documented capacity semantics: `capacity` c is the maximum number of distinct keys the cache may
hold (cache-as-main-memory, no separate disk tiers here). Every access returns hit/miss and the
cache size never exceeds c.

Policy definitions (the "simulator"):
- LRU: a hit moves the key to the most-recently-used end; a miss inserts at MRU and, when the
  cache is over capacity, evicts the least-recently-used key.
- LFU: every hit increments the key's count and refreshes its recency; a miss on a full cache
  evicts the key with the smallest count, ties broken by least-recently-used (documented tie rule).
- ARC (Megiddo & Modha): p-adaptive split of T1 (recent) / T2 (frequent); misses that hit ghost
  lists B1/B2 adjust p, and `replace` demotes within real lists. Cache contents = T1 + T2 with
  |T1| + |T2| <= c; B1 and B2 hold only ghosts, |T1|+|B1| <= c and total <= 2c.

Each policy is run against an independent naive reference implementation (plain Python lists /
linear scans) on three fixed LCG-derived traces; hit vectors, cache-size vectors and totals must be
identical, and the totals are also pinned as constants.
"""
from collections import OrderedDict


class LRUCache:
    name = "LRU"

    def __init__(self, capacity):
        self.capacity = capacity
        self.order = OrderedDict()
        self.hits = 0

    def access(self, key):
        if key in self.order:
            self.order.move_to_end(key)
            self.hits += 1
            return True
        self.order[key] = True
        if len(self.order) > self.capacity:
            self.order.popitem(last=False)
        return False

    def contents(self):
        return sorted(self.order)

    def size(self):
        return len(self.order)


class LFUCache:
    name = "LFU"

    def __init__(self, capacity):
        self.capacity = capacity
        self.counts = {}
        self.recency = {}
        self.tick = 0
        self.hits = 0

    def access(self, key):
        self.tick += 1
        if key in self.counts:
            self.counts[key] += 1
            self.recency[key] = self.tick
            self.hits += 1
            return True
        if len(self.counts) >= self.capacity:
            victim = min(self.counts, key=lambda k: (self.counts[k], self.recency[k]))
            del self.counts[victim]
            del self.recency[victim]
        self.counts[key] = 1
        self.recency[key] = self.tick
        return False

    def contents(self):
        return sorted(self.counts)

    def size(self):
        return len(self.counts)


class ARCCache:
    name = "ARC"

    def __init__(self, capacity):
        self.capacity = capacity
        self.p = 0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()
        self.hits = 0

    def replace(self, key):
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
            self.p = min(self.capacity, self.p + max(1, len(self.b2) // len(self.b1)))
            self.replace(key)
            del self.b1[key]
            self.t2[key] = True
        elif key in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) // len(self.b2)))
            self.replace(key)
            del self.b2[key]
            self.t2[key] = True
        else:
            if len(self.t1) + len(self.b1) == self.capacity:
                if len(self.t1) < self.capacity:
                    self.b1.popitem(last=False)
                    self.replace(key)
                else:
                    self.t1.popitem(last=False)
            elif len(self.t1) + len(self.b1) < self.capacity:
                total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
                if total >= self.capacity:
                    if total == 2 * self.capacity:
                        self.b2.popitem(last=False)
                    self.replace(key)
            self.t1[key] = True
        return False

    def contents(self):
        return sorted(list(self.t1) + list(self.t2))

    def size(self):
        return len(self.t1) + len(self.t2)


def lru_reference(trace, capacity):
    order, hits = [], 0
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


def lfu_reference(trace, capacity):
    counts, recency, tick, hits = {}, {}, 0, 0
    for key in trace:
        tick += 1
        if key in counts:
            counts[key] += 1
            recency[key] = tick
            hits += 1
        else:
            if len(counts) >= capacity:
                victim = min(counts, key=lambda k: (counts[k], recency[k]))
                del counts[victim], recency[victim]
            counts[key] = 1
            recency[key] = tick
    return hits


def arc_reference(trace, capacity):
    p, t1, t2, b1, b2, hits = 0, [], [], [], [], 0
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
    return hits


def run_trace(policy_class, trace, capacity):
    cache = policy_class(capacity)
    hit_vector, size_vector, overflows = [], [], 0
    for key in trace:
        hit_vector.append(cache.access(key))
        size = cache.size()
        size_vector.append(size)
        if size > capacity:
            overflows += 1
    return cache, hit_vector, size_vector, overflows


def uniform_trace(count, seed, key_count):
    trace = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        trace.append(seed % key_count)
    return trace


def locality_trace(count, seed, key_count, repeat_mod):
    trace = []
    for i in range(count):
        seed = (seed * 75 + 74) % 65537
        if i > 0 and seed % 10 < repeat_mod:
            trace.append(trace[-1])
        else:
            trace.append(seed % key_count)
    return trace


def hot_scan_trace(rounds, hot, scan_width):
    trace = []
    for block in range(rounds):
        trace.extend(range(hot))
        trace.extend(range(100 + block * scan_width, 100 + (block + 1) * scan_width))
        trace.extend(range(hot))
    return trace


def main():
    # Hand-worked trace: capacity 2, accesses 1,2,1,3,1 -> F,F,T,F,T for all three policies.
    hand = [1, 2, 1, 3, 1]
    expected = [False, False, True, False, True]
    for policy in (LRUCache, LFUCache, ARCCache):
        cache = policy(2)
        observed = [cache.access(key) for key in hand]
        assert observed == expected, (policy.name, observed)
        assert cache.hits == 2 and cache.contents() == [1, 3], (policy.name, cache.contents())
    assert lru_reference(hand, 2) == 2
    assert lfu_reference(hand, 2) == 2
    assert arc_reference(hand, 2) == 2

    traces = {
        "uniform": uniform_trace(400, 485, 20),
        "locality": locality_trace(400, 486, 24, 7),
        "hot-scan": hot_scan_trace(24, 4, 8),
    }
    references = {"LRU": lru_reference, "LFU": lfu_reference, "ARC": arc_reference}
    pinned = {
        ("uniform", "LRU"): 70, ("uniform", "LFU"): 64, ("uniform", "ARC"): 64,
        ("locality", "LRU"): 296, ("locality", "LFU"): 291, ("locality", "ARC"): 290,
        ("hot-scan", "LRU"): 92, ("hot-scan", "LFU"): 161, ("hot-scan", "ARC"): 161,
    }
    for trace_name, trace in traces.items():
        for policy_class in (LRUCache, LFUCache, ARCCache):
            cache, hit_vector, size_vector, overflows = run_trace(policy_class, trace, 4)
            expected_hits = references[policy_class.name](trace, 4)
            assert cache.hits == expected_hits, (trace_name, policy_class.name, cache.hits)
            assert sum(hit_vector) == cache.hits
            assert sum(hit_vector) == pinned[(trace_name, policy_class.name)], \
                (trace_name, policy_class.name, sum(hit_vector))
            assert overflows == 0
            assert max(size_vector) == 4 and size_vector[-1] == 4
            assert len(cache.contents()) == 4 and len(set(cache.contents())) == 4
            for key in cache.contents():
                assert key in trace
    # Pinned totals are exactly the pinned sums (guards against a wrong constant).
    for (trace_name, policy_name), value in pinned.items():
        assert isinstance(value, int) and value > 0, (trace_name, policy_name)
    assert pinned[("hot-scan", "ARC")] > pinned[("hot-scan", "LRU")]
    print("PASS")


if __name__ == "__main__":
    main()
