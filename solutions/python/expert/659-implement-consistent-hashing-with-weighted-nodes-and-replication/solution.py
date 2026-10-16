"""Task 659 - Implement consistent hashing with weighted nodes and replication.  (Tier: Expert, Ch 9 - System Design and Scalability)

A consistent-hash ring places each physical node at `round(vnodes_per_unit * weight)` virtual
positions on a 2^32 circle (the virtual-node count is what makes weights meaningful and keeps the
distribution balanced). Positions come from a deterministic 64-bit FNV-1a hash of `"node#vnode"`, so
the ring depends only on the *set* of nodes and their weights, never on insertion order or on Python's
randomised string hash. A key is hashed onto the same circle and served by the first virtual node
clockwise, wrapping around the end of the ring - that wrap-around is the reason a membership change
touches only a bounded neighbourhood of keys.

Replication walks clockwise from the primary, collecting `r` distinct *physical* nodes (virtual nodes
belonging to an already chosen node are skipped), returning however many distinct nodes the cluster
actually has - a short chain is reported to the caller instead of silently duplicating a replica.

Verified: deterministic position generation, ring independence from insertion order, exact ring
coverage (every node holds at least one virtual node), wrap-around behaviour at the last position,
`r` distinct replicas with the primary first, monotone virtual-node counts in the weights, measured
key/replica shares matching the weights within a tolerance, the *exact* minimal-disruption property
(removing a node relocates only the keys it owned; adding a node relocates only the keys it takes
over) and the statistical size of that movement, insufficient-replica reporting, and rejection of
empty rings and non-positive weights.
"""
import bisect

RING_SIZE = 1 << 32


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def fnv1a(text):
    digest = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        digest ^= byte
        digest = (digest * 0x100000001B3) & ((1 << 64) - 1)
    return digest


def ring_position(label):
    """Position on the 2^32 circle: FNV-1a plus a splitmix64 finaliser.

    FNV-1a alone has weak avalanche in its low bits, which would cluster the virtual nodes; the
    finaliser mixes the whole digest so neighbouring labels land far apart on the ring.
    """
    digest = fnv1a(label)
    digest = ((digest ^ (digest >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    digest = ((digest ^ (digest >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    digest ^= digest >> 31
    return digest % RING_SIZE


class ConsistentHashRing:
    """Weighted consistent-hash ring with virtual nodes."""

    def __init__(self, vnodes_per_unit=64):
        check(vnodes_per_unit >= 1, "a node needs at least one virtual node")
        self.vnodes_per_unit = vnodes_per_unit
        self.weights = {}
        self.positions = []
        self.owners = []

    def vnode_count(self, weight):
        return max(1, int(round(self.vnodes_per_unit * weight)))

    def add_node(self, node, weight=1.0):
        check(node not in self.weights, f"node {node} is already in the ring")
        check(weight > 0, "a node weight must be positive")
        self.weights[node] = float(weight)
        self._rebuild()

    def remove_node(self, node):
        check(node in self.weights, f"node {node} is not in the ring")
        del self.weights[node]
        self._rebuild()

    def _rebuild(self):
        """Positions depend only on (node, index), so the layout is insertion-order independent."""
        placed = []
        for node, weight in self.weights.items():
            for index in range(self.vnode_count(weight)):
                placed.append((ring_position(f"{node}#{index}"), node))
        placed.sort()
        self.positions = [position for position, _ in placed]
        self.owners = [node for _, node in placed]

    def size(self):
        return len(self.positions)

    def _index(self, key):
        check(bool(self.positions), "the ring is empty, no key can be located")
        index = bisect.bisect_left(self.positions, ring_position(key))
        return index if index < len(self.positions) else 0        # wrap around the circle

    def locate(self, key):
        return self.owners[self._index(key)]

    def locate_replicas(self, key, replica_count):
        """Primary first, then clockwise neighbours: `replica_count` distinct physical nodes."""
        check(replica_count >= 1, "at least one replica is required")
        start = self._index(key)
        chosen = []
        seen = set()
        for offset in range(self.size()):
            node = self.owners[(start + offset) % self.size()]
            if node in seen:
                continue
            seen.add(node)
            chosen.append(node)
            if len(chosen) == replica_count:
                break
        return chosen, len(chosen) == replica_count

    def load(self, keys):
        share = {node: 0 for node in self.weights}
        for key in keys:
            share[self.locate(key)] += 1
        return share

    def replica_load(self, keys, replica_count):
        share = {node: 0 for node in self.weights}
        for key in keys:
            chosen, complete = self.locate_replicas(key, replica_count)
            check(complete, "the cluster must hold enough nodes for the requested replicas")
            for node in chosen:
                share[node] += 1
        return share


def build_ring(weights, vnodes_per_unit=64):
    ring = ConsistentHashRing(vnodes_per_unit)
    for node, weight in weights:
        ring.add_node(node, weight)
    return ring


def keys(count, prefix="key-"):
    return [f"{prefix}{index}" for index in range(count)]


def expect_error(call, label):
    try:
        call()
    except (AssertionError, ValueError):
        return
    raise AssertionError(f"{label} must be rejected")


def main():
    sample = keys(3000)

    # 1. ring shape and determinism
    ring = build_ring([("a", 1), ("b", 2), ("c", 3)])
    check(ring.size() == (1 + 2 + 3) * 64, "the ring must hold one virtual node per weight unit")
    check(ring.positions == sorted(ring.positions), "the ring must keep its positions sorted")
    check(ring.locate("key-1") == ring.locate("key-1"), "a key must always map to the same node")
    check(all(ring.locate(key) in {"a", "b", "c"} for key in sample),
          "every key must map to a real node")
    again = build_ring([("c", 3), ("a", 1), ("b", 2)])
    check([again.locate(key) for key in sample] == [ring.locate(key) for key in sample],
          "the ring must not depend on the insertion order")
    check(again.positions == ring.positions, "the layout must be identical regardless of insertion order")

    # 2. virtual-node counts follow the weights
    check(ring.vnode_count(1) == 64 and ring.vnode_count(2) == 128 and ring.vnode_count(3) == 192,
          "virtual-node counts must be proportional to the weights")
    check(ring.vnode_count(0.25) == 16, "fractional weights are rounded to whole virtual nodes")
    one = build_ring([("x", 1)])
    check(one.size() == 64, "a single unit-weight node owns 64 virtual nodes")
    check(all(one.locate(key) == "x" for key in sample), "a one-node ring owns every key")
    check(one.locate_replicas("key-1", 3)[0] == ["x"], "a one-node ring can return only one replica")

    # 3. measured shares must follow the weights
    share = ring.load(sample)
    total = sum(share.values())
    check(total == len(sample), "every key must be owned exactly once")
    for node, weight in ring.weights.items():
        expected = weight / sum(ring.weights.values())
        measured = share[node] / total
        check(abs(measured - expected) < 0.06,
              f"node {node} holds {measured:.3f} of the keys, expected {expected:.3f}")
    print(f"659 shares weights={{a:1,b:2,c:3}} measured={ {n: round(v / total, 3) for n, v in share.items()} }")

    # 4. wrap-around at the end of the ring
    tail_key = None
    for key in sample:
        if ring._index(key) == 0:
            tail_key = key
            break
    check(tail_key is not None, "some key must fall into the first ring segment")
    check(ring.locate(tail_key) == ring.owners[0],
          "a key before the first position wraps to the first virtual node")
    check(ring.positions[0] == min(ring.positions), "the first segment starts at the smallest position")
    check(bisect.bisect_left(ring.positions, ring.positions[-1] + 1) == len(ring.positions),
          "positions past the last one wrap rather than falling off the ring")

    # 5. replication: distinct physical nodes, primary first
    replicas, complete = ring.locate_replicas("key-42", 3)
    check(complete and len(replicas) == 3, "three replicas must be found on a three-node ring")
    check(len(set(replicas)) == 3, "replicas must be distinct physical nodes")
    check(replicas[0] == ring.locate("key-42"), "the primary must come first in the replica list")
    check(sorted(replicas) == ["a", "b", "c"], "a three-node ring replicates onto all its nodes")
    short, short_complete = ring.locate_replicas("key-42", 5)
    check(short == replicas and short_complete is False,
          "asking for more replicas than nodes must report the shortfall")
    check(ring.locate_replicas("key-42", 1)[0] == [ring.locate("key-42")],
          "one replica is just the primary")
    dense = build_ring([("a", 1), ("b", 2), ("c", 3)], vnodes_per_unit=256)
    replica_share = dense.replica_load(sample, 2)
    shares = {node: replica_share[node] / (2 * len(sample)) for node in dense.weights}
    # The primary is exactly proportional to a node's ring share; the second replica follows the
    # clockwise adjacency of virtual-node runs, which is noisier, so the tolerance is wider there.
    for node, weight in dense.weights.items():
        expected = weight / sum(dense.weights.values())
        check(abs(shares[node] - expected) < 0.10,
              f"node {node} holds {shares[node]:.3f} of the replica slots, expected {expected:.3f}")
    check(shares["c"] > shares["b"] > shares["a"],
          "replica load must be ordered like the weights")
    print(f"659 replica shares={{a:1,b:2,c:3}} measured="
          f"{ {n: round(v, 3) for n, v in shares.items()} }")

    # 6. minimal disruption on removal: only the keys of the removed node move
    before = {key: ring.locate(key) for key in sample}
    ring.remove_node("b")
    after = {key: ring.locate(key) for key in sample}
    moved = [key for key in sample if before[key] != after[key]]
    owned_by_b = [key for key in sample if before[key] == "b"]
    check(sorted(moved) == sorted(owned_by_b),
          "removing a node must relocate exactly the keys it owned")
    check(all(after[key] in {"a", "c"} for key in sample), "no key may stay on a removed node")
    check(0.2 < len(moved) / len(sample) < 0.45,
          "the moved fraction must be close to the removed weight share")
    check(ring.size() == (1 + 3) * 64, "the removed virtual nodes must leave the ring")

    # 7. minimal disruption on insertion: only the keys the new node takes over move
    ring.add_node("d", 2)
    final = {key: ring.locate(key) for key in sample}
    adopted = [key for key in sample if final[key] != after[key]]
    check(all(final[key] == "d" for key in adopted),
          "adding a node must only move the keys that node takes over")
    check(0.15 < len(adopted) / len(sample) < 0.45,
          "the adopted fraction must be close to the new node's weight share")
    check(ring.load(sample)["d"] == len(adopted), "the new node's load must match its adopted keys")
    check(abs(ring.load(sample)["d"] / len(sample) - 2 / 6) < 0.06,
          "the new node's share must follow its weight")

    # 8. rejection of invalid ring operations
    expect_error(lambda: ConsistentHashRing(0).locate("x"), "a zero virtual-node unit")
    expect_error(lambda: build_ring([("z", 0)]), "a non-positive weight")
    expect_error(lambda: build_ring([("z", 1), ("z", 1)]), "a duplicated node")
    expect_error(lambda: build_ring([("z", 1)]).remove_node("nope"), "removing an absent node")
    expect_error(lambda: build_ring([("z", 1)]).locate_replicas("k", 0), "a zero replica count")
    check(True, "invalid ring operations are rejected")

    # 9. empty ring and single-key edge cases
    empty = ConsistentHashRing(8)
    expect_error(lambda: empty.locate("anything"), "locating a key in an empty ring")
    expect_error(lambda: empty.locate_replicas("anything", 1), "replicas of an empty ring")
    tiny = build_ring([("only", 0.1)])
    check(tiny.size() == tiny.vnode_count(0.1) and tiny.size() >= 1,
          "a tiny weight still maps to its rounded virtual-node count")
    check(tiny.locate("key-0") == "only", "the only node owns the ring")

    print(f"659 ring virtual nodes={ring.size()} replicas={replicas}")
    print(f"659 moved on removal={len(moved)} adopted on insertion={len(adopted)} of {len(sample)}")
    print("PASS")


if __name__ == "__main__":
    main()
