"""Task 470 - Implement consistent hashing ring (virtual nodes). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Consistent-hashing ring with VNODES = 64 virtual nodes per physical node. Ring
positions come from the FIXED LCG hash (documented, deterministic): fold over bytes
with h = (h * 75 + 74 + ord(ch)) mod 65537, then two LCG avalanche steps
h = (h * 75 + 74) mod 65537 (the plain fold alone clusters badly - measured shares
0.02..0.48; the avalanched hash balances). Nodes are sorted by (position, label);
a key k hashes with the same function and is owned by the first vnode clockwise at
or after its position (wrap-around).

Checks on 4 physical nodes x 64 vnodes and 4096 keys:
  * determinism - rebuilding the ring and the assignment yields identical results;
  * the bisect lookup equals a brute-force clockwise scan on a key sample;
  * balance - each node's key share stays inside the documented [0.18, 0.32] band
    (measured 0.220..0.270; ideal is 0.25);
  * adding "node-e" moves at most the documented 0.30 of all keys (measured 0.175;
    the ideal is 1/5 = 0.2), and every moved key moves exactly to "node-e";
  * removing "node-b" moves exactly node-b's keys (measured 0.252 <= the documented
    0.30) and no other key changes owner.
"""

VNODES = 64
MODULUS = 65537


def lcg_hash(text):
    h = 0
    for ch in text:
        h = (h * 75 + 74 + ord(ch)) % MODULUS
    h = (h * 75 + 74) % MODULUS
    h = (h * 75 + 74) % MODULUS
    return h


class ConsistentHashRing:
    def __init__(self, nodes, vnodes=VNODES):
        self.points = []  # sorted (position, node, vnode index)
        for node in nodes:
            for i in range(vnodes):
                self.points.append((lcg_hash(node + "#" + str(i)), node, i))
        self.points.sort()

    def owner(self, key):
        pos = lcg_hash(key)
        lo, hi = 0, len(self.points)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.points[mid][0] < pos:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(self.points):
            lo = 0  # wrap around
        return self.points[lo][1]

    def owner_brute(self, key):
        pos = lcg_hash(key)
        best = None
        for point in self.points:
            if point[0] >= pos and (best is None or point < best):
                best = point
        if best is None:
            best = self.points[0]
        return best[1]


def assign(ring, keys):
    return {key: ring.owner(key) for key in keys}


def main():
    nodes = ["node-a", "node-b", "node-c", "node-d"]
    keys = ["k%d" % i for i in range(4096)]

    # Determinism.
    ring = ConsistentHashRing(nodes)
    first = assign(ring, keys)
    second = assign(ConsistentHashRing(nodes), keys)
    assert first == second
    assert len({point[0] for point in ring.points}) > len(nodes) * VNODES // 2

    # Bisect lookup equals the brute-force clockwise scan on a fixed sample.
    for i in range(0, 4096, 79):
        assert ring.owner(keys[i]) == ring.owner_brute(keys[i])

    # Balance within the documented band.
    shares = {node: 0 for node in nodes}
    for owner in first.values():
        shares[owner] += 1
    for node, count in shares.items():
        share = count / len(keys)
        assert 0.18 <= share <= 0.32, (node, share)

    # Adding a node: bounded movement, and every moved key goes to the new node.
    ring_plus = ConsistentHashRing(nodes + ["node-e"])
    after_add = assign(ring_plus, keys)
    moved = [key for key in keys if after_add[key] != first[key]]
    assert len(moved) / len(keys) <= 0.30, len(moved) / len(keys)
    assert all(after_add[key] == "node-e" for key in moved)

    # Removing a node: only that node's keys move.
    ring_minus = ConsistentHashRing([node for node in nodes if node != "node-b"])
    after_remove = assign(ring_minus, keys)
    moved_remove = [key for key in keys if after_remove[key] != first[key]]
    assert len(moved_remove) / len(keys) <= 0.30, len(moved_remove) / len(keys)
    assert len(moved_remove) == shares["node-b"]  # only node-b's keys move
    for key in keys:
        if first[key] != "node-b":
            assert after_remove[key] == first[key], key  # untouched owner
    assert all(first[key] == "node-b" for key in moved_remove)

    # Measured values documented in the module docstring.
    assert len(moved) == 716 and len(moved_remove) == 1034


if __name__ == "__main__":
    main()
    print("PASS")
