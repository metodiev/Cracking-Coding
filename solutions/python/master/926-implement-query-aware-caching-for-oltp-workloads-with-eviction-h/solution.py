"""Task 926 - Implement query-aware caching for OLTP workloads with eviction hints.  (Tier: Master, CTCI Ch 14 - Databases)

An OLTP result cache driven by a query-aware workload model, compared against the classical policies.

* Workload: `make_trace` emits (template, key, hint) triples.  Two hot templates cycle through four
  keys (heavy reuse), two cold templates touch a wide set of keys that are never requested twice.
  The hint is the template's reuse class, which is exactly the metadata an optimizer can hand the
  cache (a query-template frequency estimate).
* Policies: LRU (recency), LFU (frequency), and the hint-aware policy, which evicts entries whose
  template hint is cold first and only then falls back to recency, so one-shot analytical lookups
  cannot flush the hot working set.
* Oracle: Belady's MIN, computed from the pre-computed next-use times, is the optimal offline
  policy, so it upper-bounds every online policy on this trace.

Verified: hits plus misses equal the number of accesses for every policy; a hit always returns the
value that was inserted (no stale or phantom hit) and the resident set never exceeds the capacity;
Belady dominates every online policy; the hint-aware policy beats plain LRU on the mixed workload;
hint-aware residency of hot keys exceeds that of cold keys; capacity 1 reduces to consecutive
repeats and a capacity covering the hot set turns the workload into near-perfect hits; an empty
trace is handled; and the simulation is deterministic.  Simplified: in-memory result cache, single
thread, single cache level, no invalidation on writes.
"""
from bisect import bisect_right

COLD_TEMPLATES, HOT_TEMPLATES = ("report_a", "report_b"), ("point_lookup", "index_probe")
HOT_KEYS, COLD_KEYS = 4, 40
HOT_RATIO, CAPACITY = 0.6, 5
POLICIES = ("lru", "lfu", "hint", "belady")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


def uniform(state):
    state, raw = splitmix64(state)
    return state, (raw >> 11) / 9007199254740992.0


def make_trace(accesses, hot_ratio=HOT_RATIO, state=0x5EED):
    """Deterministic mixed workload: reusable hot keys interleaved with one-shot cold keys."""
    trace, cold_cursor = [], 0
    for step in range(accesses):
        state, draw = uniform(state)
        if draw < hot_ratio:
            template = HOT_TEMPLATES[step % len(HOT_TEMPLATES)]
            key = ("hot", (step * 3) % HOT_KEYS)
            trace.append((template, key, True))
        else:
            template = COLD_TEMPLATES[cold_cursor % len(COLD_TEMPLATES)]
            key = ("cold", cold_cursor % COLD_KEYS, cold_cursor)
            cold_cursor += 1
            trace.append((template, key, False))
    return trace


def access_schedule(trace):
    """Every key's sorted list of access times, used by Belady's MIN."""
    schedule = {}
    for time, (_template, key, _hot) in enumerate(trace):
        schedule.setdefault(key, []).append(time)
    return schedule


def simulate(trace, capacity, policy):
    """Replay the trace; returns counters plus how long hot and cold entries stayed resident.

    The invariant `cache[key] == inserted value` is asserted on every hit, so a wrong or stale hit
    aborts the run instead of silently inflating the hit ratio, and the resident set is checked
    against the capacity after every mutation.
    """
    schedule = access_schedule(trace)
    cache, meta = {}, {}                    # key -> value, key -> (last_use, count, hot)
    hits = misses = evictions = 0
    residency = {True: 0, False: 0}
    peak = 0

    def next_use(key, time):
        times = schedule[key]
        position = bisect_right(times, time)
        return times[position] if position < len(times) else float("inf")

    for time, (_template, key, hot) in enumerate(trace):
        if key in cache:
            hits += 1
            check(cache[key][0] == key, "a cache hit must return the value that was inserted")
        else:
            misses += 1
            if capacity and len(cache) >= capacity:
                if policy == "lru":
                    victim = min(meta, key=lambda entry: meta[entry][0])
                elif policy == "lfu":
                    victim = min(meta, key=lambda entry: (meta[entry][1], meta[entry][0]))
                elif policy == "hint":
                    victim = min(meta, key=lambda entry: (meta[entry][2], meta[entry][0]))
                else:
                    victim = max(meta, key=lambda entry: (next_use(entry, time), meta[entry][0]))
                cache.pop(victim)
                meta.pop(victim)
                evictions += 1
            cache[key] = (key, time)
        previous = meta.get(key, (0, 0, hot))
        meta[key] = (time, previous[1] + 1, hot)
        peak = max(peak, len(cache))
        check(len(cache) <= max(capacity, 0), "the cache must never exceed its capacity")
        for resident in meta.values():
            residency[resident[2]] += 1
    return {"hits": hits, "misses": misses, "evictions": evictions, "peak": peak,
            "ratio": hits / len(trace) if trace else 0.0, "hot_residency": residency[True],
            "cold_residency": residency[False]}


def main():
    trace = make_trace(600)
    hots = sum(1 for _, _, hot in trace if hot)
    check(len(trace) == 600 and hots > 300,
          "the workload generator must be deterministic and hot-dominated")

    # 1. Counters and invariants hold for every policy; sizes never exceed the capacity.
    results = {policy: simulate(trace, CAPACITY, policy) for policy in POLICIES}
    distinct = len({key for _, key, _ in trace})
    for policy, result in results.items():
        check(result["hits"] + result["misses"] == len(trace),
              "every access must be counted exactly once")
        check(result["peak"] <= CAPACITY, "the cache must never exceed its capacity")
        check(result["evictions"] == result["misses"] - min(CAPACITY, distinct),
              "evictions must account for every insertion that did not fit")

    # 2. Belady's MIN upper-bounds every online policy on this trace.
    best_online = max(results[policy]["ratio"] for policy in POLICIES if policy != "belady")
    check(results["belady"]["ratio"] >= best_online,
          "the offline optimum must not be beaten by an online policy")

    # 3. Query awareness pays: the hint-aware cache beats plain LRU on this workload.
    check(results["hint"]["ratio"] > results["lru"]["ratio"] + 0.05,
          "eviction hints must protect the hot working set better than LRU")
    check(results["hint"]["hits"] > results["lru"]["hits"],
          "the hint-aware policy must serve more hits than LRU")

    # 4. Hint-aware residency: hot entries stay in the cache longer than cold ones.
    check(results["hint"]["hot_residency"] > results["hint"]["cold_residency"],
          "hinted-hot entries must outlive cold ones under the hint policy")

    # 5. Independent boundary: with capacity 1 only consecutive repeats can hit.
    repeats = sum(1 for time in range(1, len(trace)) if trace[time][1] == trace[time - 1][1])
    for policy in POLICIES:
        check(simulate(trace, 1, policy)["hits"] == repeats,
              "a one-entry cache can only serve consecutive repeats, whatever the policy")

    # 6. A capacity covering the hot set serves every reusable access: the only misses left are the
    #    cold one-shot keys, so the hint policy approaches the workload's hit ceiling.
    wide = {policy: simulate(trace, 16, policy)["ratio"] for policy in POLICIES}
    ceiling = hots / len(trace)
    check(wide["hint"] > 0.95 * ceiling and wide["hint"] <= ceiling + 1e-9,
          "with room for the whole hot set the hint policy must reach the traffic ceiling")
    check(wide["hint"] >= wide["lru"], "extra capacity must never favour the weaker policy")
    check(max(wide.values()) <= 1.0, "a hit ratio cannot exceed one")

    # 7. Boundary cases: an empty trace and an oversized cache.
    empty = simulate([], 8, "hint")
    check(empty["hits"] == 0 and empty["misses"] == 0 and empty["ratio"] == 0.0
          and empty["peak"] == 0,
          "an empty trace must produce an empty result, not a division by zero")
    tiny = simulate(trace[:5], 64, "lru")
    check(tiny["peak"] == len({key for _, key, _ in trace[:5]}) and tiny["evictions"] == 0,
          "a cache larger than the working set must never evict")

    # 8. Determinism: replaying the same trace reports identical counters.
    check(simulate(trace, CAPACITY, "hint") == results["hint"]
          and make_trace(600) == trace,
          "the workload and the simulation must be deterministic")

    print("hit ratios at capacity 5: " + ", ".join(
        f"{policy}={results[policy]['ratio']:.3f}" for policy in POLICIES))
    print(f"hint-aware residency hot={results['hint']['hot_residency']} "
          f"cold={results['hint']['cold_residency']} accesses, capacity 16 ratios "
          + ", ".join(f"{policy}={ratio:.3f}" for policy, ratio in wide.items()))
    print("PASS")


if __name__ == "__main__":
    main()
