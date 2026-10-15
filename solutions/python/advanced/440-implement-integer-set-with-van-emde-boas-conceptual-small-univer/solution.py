"""Task 440 - Implement integer set with van Emde Boas (conceptual small universe). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Documented simplification of a van Emde Boas set on the small universe U = 256 = 16 x 16: the top
level keeps min/max, a 16-entry summary of cluster occupancy and 16 lazily created clusters, while
the summary and every cluster is the base case - a 16-bit member bitmask (BitSet16) whose bit scans
replace the recursion below u = 16. The decisive vEB trick is kept: the minimum is stored outside
the clusters, so delete(min) promotes the successor from the cluster/summary structure. Operations
MEMBER / SUCCESSOR / PREDECESSOR / INSERT / DELETE therefore match a full recursive vEB
(O(1) bit work at this two-level scale). Assertions cover hand sequences that cross cluster
boundaries, empty the structure and exercise successor/predecessor at both universe ends, plus LCG
operation scripts cross-checked against a sorted-set oracle with full 0..255 sweeps.
"""


class BitSet16:
    def __init__(self):
        self.mask = 0

    def insert(self, value):
        self.mask |= 1 << value

    def remove(self, value):
        self.mask &= ~(1 << value)

    def member(self, value):
        return (self.mask >> value) & 1 == 1

    def empty(self):
        return self.mask == 0

    @staticmethod
    def _lowest_bit(mask):
        index = 0
        while mask & 1 == 0:
            mask >>= 1
            index += 1
        return index

    @staticmethod
    def _highest_bit(mask):
        index = -1
        while mask:
            mask >>= 1
            index += 1
        return index

    def minimum(self):
        return None if self.mask == 0 else self._lowest_bit(self.mask)

    def maximum(self):
        return None if self.mask == 0 else self._highest_bit(self.mask)

    def successor(self, value):
        higher = self.mask >> (value + 1)
        if higher == 0:
            return None
        return value + 1 + self._lowest_bit(higher)

    def predecessor(self, value):
        lower = self.mask & ((1 << value) - 1)
        return None if lower == 0 else self._highest_bit(lower)


class VanEmdeBoas:
    UNIVERSE = 256
    CLUSTERS = 16

    def __init__(self):
        self.min = None
        self.max = None
        self.count = 0
        self.summary = BitSet16()
        self.clusters = [None] * self.CLUSTERS

    def size(self):
        return self.count

    def member(self, value):
        if value == self.min or value == self.max:
            return True
        cluster = self.clusters[value >> 4]
        return cluster is not None and cluster.member(value & 15)

    def insert(self, value):
        if self.member(value):
            return
        if self.min is None:
            self.min = value
            self.max = value
            self.count = 1
            return
        if value < self.min:
            value, self.min = self.min, value
        if value > self.max:
            self.max = value
        high, low = value >> 4, value & 15
        if self.clusters[high] is None:
            self.clusters[high] = BitSet16()
            self.summary.insert(high)
        self.clusters[high].insert(low)
        self.count += 1

    def delete(self, value):
        if not self.member(value):
            return False
        if self.count == 1:
            self.min = None
            self.max = None
            self.summary = BitSet16()
            self.clusters = [None] * self.CLUSTERS
            self.count = 0
            return True
        if value == self.min:
            successor = self._successor_stored(value)
            self.min = successor
            self._delete_stored(successor)
            return True
        self._delete_stored(value)
        return True

    def _successor_stored(self, value):
        high, low = value >> 4, value & 15
        cluster = self.clusters[high]
        if cluster is not None and low < 15:
            within = cluster.successor(low)
            if within is not None:
                return (high << 4) | within
        next_high = self.summary.successor(high)
        return (next_high << 4) | self.clusters[next_high].minimum()

    def _delete_stored(self, value):
        high, low = value >> 4, value & 15
        self.clusters[high].remove(low)
        if self.clusters[high].empty():
            self.summary.remove(high)
            self.clusters[high] = None
        self.count -= 1
        if value == self.max:
            top = self.summary.maximum()
            self.max = self.min if top is None else (top << 4) | self.clusters[top].maximum()

    def successor(self, value):
        if self.min is None or value >= self.max:
            return None
        if value < self.min:
            return self.min
        high, low = value >> 4, value & 15
        cluster = self.clusters[high]
        if cluster is not None and low < 15:
            within = cluster.successor(low)
            if within is not None:
                return (high << 4) | within
        next_high = self.summary.successor(high)
        if next_high is None:
            return None
        return (next_high << 4) | self.clusters[next_high].minimum()

    def predecessor(self, value):
        if self.min is None or value <= self.min:
            return None
        if value > self.max:
            return self.max
        high, low = value >> 4, value & 15
        cluster = self.clusters[high]
        if cluster is not None and low > 0:
            within = cluster.predecessor(low)
            if within is not None:
                return (high << 4) | within
        previous = self.summary.predecessor(high)
        if previous is None:
            return self.min
        return (previous << 4) | self.clusters[previous].maximum()


def main():
    veb = VanEmdeBoas()
    assert veb.size() == 0
    assert veb.member(0) is False
    assert veb.successor(0) is None
    assert veb.predecessor(255) is None
    assert veb.delete(0) is False

    veb.insert(200)
    assert veb.size() == 1
    assert veb.member(200) and not veb.member(199)
    assert veb.successor(200) is None
    assert veb.successor(5) == 200
    assert veb.predecessor(200) is None
    assert veb.predecessor(255) == 200
    assert veb.delete(200) is True
    assert veb.delete(200) is False
    assert veb.size() == 0

    veb.insert(5)
    veb.insert(100)
    veb.insert(200)
    assert veb.size() == 3
    assert veb.successor(5) == 100
    assert veb.successor(4) == 5
    assert veb.predecessor(100) == 5
    assert veb.predecessor(101) == 100
    assert veb.successor(199) == 200
    assert veb.delete(100) is True
    assert not veb.member(100)
    assert veb.successor(5) == 200
    assert veb.predecessor(200) == 5
    assert veb.delete(5) is True
    assert veb.successor(199) == 200
    assert veb.predecessor(199) is None
    assert veb.delete(200) is True
    assert veb.successor(0) is None

    veb.insert(0)
    veb.insert(15)
    veb.insert(16)
    veb.insert(17)
    veb.insert(255)
    assert veb.successor(0) == 15
    assert veb.successor(15) == 16
    assert veb.successor(16) == 17
    assert veb.successor(17) == 255
    assert veb.predecessor(255) == 17
    assert veb.predecessor(17) == 16
    assert veb.predecessor(16) == 15
    assert veb.predecessor(0) is None
    assert veb.delete(16) is True
    assert veb.successor(15) == 17
    assert veb.delete(0) is True
    assert veb.min == 15
    assert veb.delete(15) is True
    assert veb.min == 17
    assert veb.delete(17) is True
    assert veb.min == 255 and veb.max == 255
    assert veb.delete(255) is True
    assert veb.size() == 0 and veb.min is None

    veb = VanEmdeBoas()
    oracle = []
    seed = 4400
    for step in range(600):
        seed = (seed * 75 + 74) % 65537
        op = seed % 5
        seed = (seed * 75 + 74) % 65537
        value = seed % 256
        if op <= 1:
            veb.insert(value)
            if value not in oracle:
                oracle.append(value)
                oracle.sort()
        elif op == 2:
            expected = value in oracle
            if expected:
                oracle.remove(value)
            assert veb.delete(value) == expected, (step, value)
        elif op == 3:
            assert veb.member(value) == (value in oracle), (step, value)
        else:
            expected_successor = None
            expected_predecessor = None
            for candidate in oracle:
                if candidate > value and expected_successor is None:
                    expected_successor = candidate
                if candidate < value:
                    expected_predecessor = candidate
            assert veb.successor(value) == expected_successor, (step, value)
            assert veb.predecessor(value) == expected_predecessor, (step, value)
        assert veb.size() == len(oracle), step
        if step % 50 == 0:
            for probe in range(256):
                assert veb.member(probe) == (probe in oracle), (step, probe)
    assert len(oracle) > 0

    for probe in range(256):
        assert veb.member(probe) == (probe in oracle), probe
        expected_successor = None
        expected_predecessor = None
        for candidate in oracle:
            if candidate > probe and expected_successor is None:
                expected_successor = candidate
            if candidate < probe:
                expected_predecessor = candidate
        assert veb.successor(probe) == expected_successor, probe
        assert veb.predecessor(probe) == expected_predecessor, probe


if __name__ == "__main__":
    main()
    print("PASS")
