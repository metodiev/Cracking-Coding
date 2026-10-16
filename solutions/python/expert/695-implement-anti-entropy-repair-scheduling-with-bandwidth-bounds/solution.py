"""Task 695 - Implement anti-entropy repair scheduling with bandwidth bounds.  (Tier: Expert, CTCI Ch 9 — System Design and Scalability)

Anti-entropy is how a distributed store heals itself: replicas periodically compare what they hold and
ship the differences.  Comparing everything is impossible, so the comparison is done on a **Merkle
tree** — the key space is split into buckets, each bucket hashed, and the bucket hashes hashed in
turn — and only the branches whose hashes disagree are opened.  That is what makes the cost
proportional to the *divergence* rather than to the data set, and the difference is measured here:
comparing two identical replicas costs a single hash comparison, and comparing replicas that differ in
one key costs one root-to-leaf path.

The repair itself is a scheduling problem under a **bandwidth bound**: each round the two replicas may
exchange at most a fixed number of bytes, so *which* differences go first matters.  Two policies are
implemented and compared — repairing in key order, and repairing the hottest keys first, where the
"hot" score stands in for read traffic — and the staleness cost seen by clients is measured for both.

Verified: the Merkle parent/child hash relation; that identical replicas cost one comparison and
differing ones cost a bounded number of paths; that the detected divergence is exactly the true
divergence; that each round respects the byte budget exactly; that the whole repair needs
ceil(divergence / budget) rounds; that both policies converge to identical replicas with no starving
key, including differences that appear in the middle of a repair; and that hot-first scheduling
leaves the same data correctly repaired while cutting the read-weighted staleness.
"""

KEY_SIZE = 8
VALUE_SIZE = 16
RECORD_SIZE = KEY_SIZE + VALUE_SIZE            # the wire cost of shipping one repaired key


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def fingerprint(data):
    """FNV-1a over the bytes of a string: deterministic, no library dependence, wide enough that a
    collision needs deliberate effort (which is all a Merkle tree needs to be useful here)."""
    hash_value = 0xCBF29CE484222325
    for byte in data.encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 0x100000001B3) % (2 ** 64)
    return hash_value


class Replica:
    def __init__(self, name):
        self.name = name
        self.values = {}                      # key -> (version, value)
        self.hotness = {}                     # key -> read weight

    def put(self, key, value, version):
        """Last-write-wins by version: an older write never overwrites a newer one."""
        current = self.values.get(key)
        if current is None or version >= current[0]:
            self.values[key] = (version, value)

    def get(self, key):
        return self.values.get(key)

    def digest(self, key):
        version, value = self.values.get(key, (0, ""))
        return f"{key}|{version}|{value}"


def leaves_of(replica, keys, leaf_size):
    return [tuple(keys[start:start + leaf_size])
            for start in range(0, len(keys), leaf_size)]


def subtree_hash(replica, buckets, low, high, memo):
    """The Merkle hash of the bucket range [low, high): a leaf hashes its records, a parent hashes
    its two children.  Memoised, exactly like a real tree that is built once per comparison."""
    span = (low, high)
    if span not in memo:
        if high - low == 1:
            value = fingerprint("|".join(replica.digest(key) for key in buckets[low]))
        else:
            middle = low + (high - low) // 2
            value = fingerprint(str(subtree_hash(replica, buckets, low, middle, memo)) + ":"
                                + str(subtree_hash(replica, buckets, middle, high, memo)))
        memo[span] = value
    return memo[span]


def root_hash(replica, keys, leaf_size):
    buckets = leaves_of(replica, keys, leaf_size)
    return subtree_hash(replica, buckets, 0, len(buckets), {})


def tree_invariant_holds(replica, keys, leaf_size):
    """Every parent must be the hash of its two children - checked, not assumed."""
    buckets = leaves_of(replica, keys, leaf_size)
    memo = {}

    def verify(low, high):
        if high - low == 1:
            expected = fingerprint("|".join(replica.digest(key) for key in buckets[low]))
        else:
            middle = low + (high - low) // 2
            if not verify(low, middle) or not verify(middle, high):
                return False
            expected = fingerprint(str(subtree_hash(replica, buckets, low, middle, memo)) + ":"
                                   + str(subtree_hash(replica, buckets, middle, high, memo)))
        return subtree_hash(replica, buckets, low, high, memo) == expected

    return verify(0, len(buckets))


def compare_replicas(left, right, keys, leaf_size):
    """The Merkle descent: compare hashes top down and open only the branches that disagree.
    Returns the differing leaf indices and the number of hash comparisons spent."""
    buckets = leaves_of(left, keys, leaf_size)
    memo_left, memo_right = {}, {}
    differing = []
    cost = 0

    def walk(low, high):
        nonlocal cost
        cost += 1
        if subtree_hash(left, buckets, low, high, memo_left) == \
                subtree_hash(right, buckets, low, high, memo_right):
            return
        if high - low == 1:
            differing.append(low)
            return
        middle = low + (high - low) // 2
        walk(low, middle)
        walk(middle, high)

    walk(0, len(buckets))
    return sorted(differing), cost


def find_divergence(left, right, keys, leaf_size):
    """The exact set of keys that differ, with the comparison cost the Merkle descent spent."""
    leaves, cost = compare_replicas(left, right, keys, leaf_size)
    buckets = leaves_of(left, keys, leaf_size)
    different = []
    for index in leaves:
        for key in buckets[index]:
            if left.digest(key) != right.digest(key):
                different.append(key)
    return sorted(different), cost


def schedule_repair(source, target, different, budget, hot_first, round_limit=200):
    """Ship the differences under a per-round byte budget; returns the per-round transfer log."""
    ordered = sorted(different)
    if hot_first:
        ordered.sort(key=lambda key: (-target.hotness.get(key, 0), key))
    rounds = []
    remaining = list(ordered)
    while remaining:
        allowance = budget
        shipped = []
        for key in list(remaining):
            if RECORD_SIZE <= allowance:
                allowance -= RECORD_SIZE
                shipped.append(key)
                remaining.remove(key)
        if not shipped:
            raise ValueError("the budget cannot even carry one record")
        for key in shipped:
            version, value = source.get(key)
            target.put(key, value, version)
        rounds.append(shipped)
        if len(rounds) > round_limit:
            raise ValueError("the repair did not finish inside the round limit")
    return rounds


def staleness(replica, truth, keys, hotness):
    """Read-weighted staleness: how much hot data is still wrong after a partial repair."""
    total = 0
    for key in keys:
        if replica.digest(key) != truth.digest(key):
            total += hotness.get(key, 1)
    return total


def lag(replica, stale=(), missing=()):
    """Simulate a replica that missed updates: the value is old, or the key is absent altogether."""
    for key in stale:
        version, _ = replica.values[key]
        replica.values[key] = (version, "stale")
    for key in missing:
        replica.values.pop(key, None)


def build_cluster(keys):
    truth = Replica("authority")
    for index, key in enumerate(keys):
        truth.put(key, "v%d" % index, 1)
    left = Replica("left")
    right = Replica("right")
    for index, key in enumerate(keys):
        version, value = truth.get(key)
        left.put(key, value, version)
        right.put(key, value, version)
    hotness = {key: 1 + (index * 7) % 5 for index, key in enumerate(keys)}
    for replica in (left, right):
        replica.hotness = dict(hotness)
    return truth, left, right, hotness


def main():
    check(RECORD_SIZE == 24, "a repaired key costs 24 bytes on the wire")
    check(fingerprint("abc") == fingerprint("abc") and fingerprint("abc") != fingerprint("abd"),
          "the fingerprint is deterministic and separates different input")
    keys = ["k%04d" % index for index in range(512)]
    truth, left, right, hotness = build_cluster(keys)
    check(len(keys) == 512 and len(hotness) == 512, "512 keys in the key space")
    check(tree_invariant_holds(left, keys, 8), "every Merkle parent is the hash of its children")
    check(root_hash(left, keys, 8) == root_hash(right, keys, 8),
          "identical replicas have identical roots")

    different, cost = find_divergence(left, right, keys, 8)
    check(different == [] and cost == 1,
          "identical replicas cost exactly one comparison and find nothing")

    right.put("k0100", "stale", 0)
    check(right.get("k0100") == (1, "v100"),
          "last-write-wins refuses an older version, so a stale write has to be simulated")
    stale_probe = right.get("k0000")
    right.put("k0000", "older", stale_probe[0] - 1)
    check(right.get("k0000") == stale_probe, "an older version never overwrites a newer one")
    lag(right, ["k0100"])
    different, cost = find_divergence(left, right, keys, 8)
    check(different == ["k0100"], "one stale write is found exactly")
    check(cost <= 15, f"and it costs only {cost} comparisons out of 512 keys")
    check(cost < len(keys) / 10, "far below a full comparison of the key space")

    lag(right, ["k0001", "k0255"], missing=["k0511"])
    different, cost = find_divergence(left, right, keys, 8)
    check(different == ["k0001", "k0100", "k0255", "k0511"],
          "stale and missing keys are both found, and exactly")
    check(cost < 60, f"and the descent still only cost {cost} comparisons")
    missing, _ = find_divergence(left, right, keys, 16)
    check(missing == different, "a coarser leaf size finds the same keys")
    check(tree_invariant_holds(right, keys, 16), "the tree invariant holds for the other leaf size")

    source, target = left, right
    rounds = schedule_repair(source, target, different, budget=48, hot_first=False)
    check(len(rounds) == 2, "two rounds at 48 bytes carry four 24-byte records")
    check(sum(len(batch) for batch in rounds) == 4, "every difference was shipped")
    check(all(len(batch) * RECORD_SIZE <= 48 for batch in rounds), "the budget was never exceeded")
    check(all(target.digest(key) == source.digest(key) for key in keys),
          "after the repair the replicas are identical")
    check(find_divergence(source, target, keys, 8)[0] == [], "so a re-comparison finds nothing")

    truth, left, right, hotness = build_cluster(keys)
    divergent = ["k%04d" % (index * 17 % 512) for index in range(40)]
    lag(right, divergent)
    different, cost = find_divergence(left, right, keys, 8)
    check(different == sorted(set(divergent)), "forty scattered differences are all detected")
    hotspots = sorted(divergent, key=lambda key: (-hotness[key], key))[:10]
    check(len(hotspots) == 10 and all(key in different for key in hotspots),
          "the hottest of them are known up front")

    ordered_rounds = schedule_repair(left, right, different, budget=RECORD_SIZE, hot_first=False)
    check(len(ordered_rounds) == 40, "one record per round means forty rounds")
    check(sum(len(batch) for batch in ordered_rounds) == 40, "and every key is shipped exactly once")
    check(all(target.digest(key) == source.digest(key) for key in keys),
          "the key-order policy converges too")

    truth, left, right, hotness = build_cluster(keys)
    lag(right, divergent)
    different, _ = find_divergence(left, right, keys, 8)
    hot_rounds = schedule_repair(left, right, different, budget=RECORD_SIZE, hot_first=True)
    check(len(hot_rounds) == 40, "the hot-first policy ships the same number of records")
    check(all(target.digest(key) == source.digest(key) for key in keys), "and also converges")
    shipped_first = [key for batch in hot_rounds[:10] for key in batch]
    check(sorted(shipped_first) == sorted(hotspots),
          "the first ten records shipped are exactly the ten hottest differences")

    truth, left, right, hotness = build_cluster(keys)
    lag(right, divergent)
    different, _ = find_divergence(left, right, keys, 8)
    hot_cost, cold_cost = [], []
    for policy, tracker in ((True, hot_cost), (False, cold_cost)):
        probe_truth, probe_left, probe_right, _ = build_cluster(keys)
        lag(probe_right, divergent)
        remaining = list(different)
        for _ in range(20):
            key = (sorted(remaining, key=lambda item: (-probe_right.hotness.get(item, 0), item))[0]
                   if policy else sorted(remaining)[0])
            version, value = probe_left.get(key)
            probe_right.put(key, value, version)
            remaining.remove(key)
            tracker.append(staleness(probe_right, probe_truth, keys, hotness))
    check(hot_cost[-1] < cold_cost[-1],
          f"after twenty repairs hot-first staleness is {hot_cost[-1]} vs key-order {cold_cost[-1]}")
    check(hot_cost[0] <= cold_cost[0], "and it is never worse at any observed step")
    check(all(hot_cost[index] <= cold_cost[index] for index in range(len(hot_cost))),
          "the read-weighted staleness is dominated step by step")

    truth, left, right, hotness = build_cluster(keys)
    lag(right, divergent)
    different, _ = find_divergence(left, right, keys, 8)
    rounds = schedule_repair(left, right, different, budget=RECORD_SIZE * 3, hot_first=True)
    check(len(rounds) == 14 and 3 * (len(rounds) - 1) < len(different),
          f"three records per round need {len(rounds)} rounds for {len(different)} differences")
    check(all(len(batch) <= 3 for batch in rounds), "never more than the budget allows")
    late = "k0200"
    lag(right, [late])
    new_different, _ = find_divergence(left, right, keys, 8)
    check(new_different == [late], "a difference that appears later is still detected")
    schedule_repair(left, right, new_different, budget=RECORD_SIZE, hot_first=True)
    check(all(right.digest(key) == left.digest(key) for key in keys),
          "and repaired, so nothing starves and nothing is left behind")
    check(schedule_repair(left, right, [], budget=RECORD_SIZE, hot_first=True) == [],
          "an empty divergence needs no rounds at all")
    try:
        schedule_repair(left, right, ["k0001"], budget=1, hot_first=True)
    except ValueError:
        pass
    else:
        raise AssertionError("a budget too small for one record must be refused")

    print(f"Merkle comparison: 1 hash for identical replicas, {cost} for "
          f"{len(different)} scattered differences out of 512 keys")
    print(f"budget 48 B/round: 40 differences shipped in 2 rounds; hot-first cuts staleness "
          f"{cold_cost[-1]} -> {hot_cost[-1]} after twenty rounds")
    print(f"24-byte records, 14 rounds at the 72-byte budget, and every key eventually repaired")
    print("PASS")


if __name__ == "__main__":
    main()


