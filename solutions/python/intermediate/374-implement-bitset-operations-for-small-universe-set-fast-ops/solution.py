"""Task 374 - Bitset (bit array) with fast set operations for a small universe. (Tier: Intermediate, CTCI Ch 5)

A fixed universe of n_bits (here <= 256) is packed into 32-bit words.  Every operation
touches only whole words, so add/remove/contains are O(1), count() is O(words) using
a popcount, and union/intersection are O(words) instead of O(universe).  Iteration
extracts the lowest set bit of each word in turn, which yields indices in strictly
ascending order without scanning empty stretches.

Everything is cross-checked against the language's plain set oracle on deterministic
LCG universes: membership of every index, count() against the oracle size, ascending
iteration against the sorted oracle, and union/intersection against the oracle set
operations.  Boundary indices 0, 31, 32 and n_bits-1 are asserted explicitly.

The implementation only uses 32-bit words, so the int/sign quirks of each language
are handled identically: words are treated as unsigned bit patterns (Python ints,
Java/JS masks via >>> and Integer.bitCount / Math.clz32, C# uint).
"""

UNIVERSE = 256


class Bitset:
    """Bitset over indices 0 .. bits-1, backed by 32-bit words."""

    def __init__(self, bits):
        if bits <= 0:
            raise ValueError("universe must be positive")
        self.bits = bits
        self.words = [0] * ((bits + 31) // 32)

    def add(self, index):
        self._validate(index)
        self.words[index >> 5] |= 1 << (index & 31)

    def remove(self, index):
        self._validate(index)
        self.words[index >> 5] &= 0xFFFFFFFF ^ (1 << (index & 31))

    def contains(self, index):
        self._validate(index)
        return (self.words[index >> 5] >> (index & 31)) & 1 == 1

    def count(self):
        total = 0
        for word in self.words:
            total += bin(word).count("1")
        return total

    def union(self, other):
        self._check_same_universe(other)
        result = Bitset(self.bits)
        for i, (a, b) in enumerate(zip(self.words, other.words)):
            result.words[i] = a | b
        return result

    def intersection(self, other):
        self._check_same_universe(other)
        result = Bitset(self.bits)
        for i, (a, b) in enumerate(zip(self.words, other.words)):
            result.words[i] = a & b
        return result

    def iterate(self):
        """All members as indices in ascending order."""
        members = []
        for wi, word in enumerate(self.words):
            while word:
                low = word & -word
                members.append((wi << 5) + low.bit_length() - 1)
                word ^= low
        return members

    def _validate(self, index):
        if not 0 <= index < self.bits:
            raise ValueError("index out of universe: " + str(index))

    def _check_same_universe(self, other):
        if self.bits != other.bits:
            raise ValueError("universe size mismatch")


def lcg_index(seed):
    seed = (seed * 75 + 74) % 65537
    return seed % UNIVERSE, seed


def main():
    # --- boundary behaviour ----------------------------------------------
    empty = Bitset(UNIVERSE)
    assert empty.count() == 0
    assert empty.iterate() == []

    boundary = Bitset(UNIVERSE)
    for index in (0, 31, 32, 33, 63, 64, 255):
        boundary.add(index)
    assert boundary.iterate() == [0, 31, 32, 33, 63, 64, 255]
    assert boundary.count() == 7
    assert boundary.contains(0) and boundary.contains(255)
    assert not boundary.contains(1) and not boundary.contains(254)
    boundary.add(0)                                   # idempotent
    assert boundary.count() == 7
    boundary.remove(1)                                # removing an absent bit is a no-op
    assert boundary.count() == 7
    boundary.remove(0)
    assert not boundary.contains(0) and boundary.count() == 6

    # --- explicit small example -------------------------------------------
    first = Bitset(64)
    second = Bitset(64)
    for index in (1, 5, 40, 63):
        first.add(index)
    for index in (5, 7, 40, 41):
        second.add(index)
    assert first.iterate() == [1, 5, 40, 63]
    assert second.iterate() == [5, 7, 40, 41]
    assert first.union(second).iterate() == [1, 5, 7, 40, 41, 63]
    assert first.intersection(second).iterate() == [5, 40]
    assert first.union(second).count() == 6
    assert first.intersection(second).count() == 2
    assert Bitset(64).union(first).iterate() == first.iterate()
    assert Bitset(64).intersection(first).count() == 0

    full = Bitset(UNIVERSE)
    for index in range(UNIVERSE):
        full.add(index)
    assert full.count() == UNIVERSE
    assert len(full.iterate()) == UNIVERSE
    spark = Bitset(UNIVERSE)
    for index in (1, 5, 40, 63, 255):
        spark.add(index)
    assert full.union(spark).count() == UNIVERSE
    assert full.union(spark).iterate() == list(range(UNIVERSE))
    assert full.intersection(spark).iterate() == spark.iterate()
    assert full.intersection(Bitset(UNIVERSE)).count() == 0

    # --- LCG cross-checks against the plain set oracle --------------------
    seed = 37413
    for trial in range(40):
        oracle_a = set()
        oracle_b = set()
        bits_a = Bitset(UNIVERSE)
        bits_b = Bitset(UNIVERSE)

        for _ in range(60):
            index, seed = lcg_index(seed)
            action, seed = lcg_index(seed)
            if action % 3 == 0:
                bits_a.remove(index)
                oracle_a.discard(index)
            else:
                bits_a.add(index)
                oracle_a.add(index)
            index, seed = lcg_index(seed)
            action, seed = lcg_index(seed)
            if action % 3 == 0:
                bits_b.remove(index)
                oracle_b.discard(index)
            else:
                bits_b.add(index)
                oracle_b.add(index)

        for index in range(UNIVERSE):
            assert bits_a.contains(index) == (index in oracle_a), (trial, index)
            assert bits_b.contains(index) == (index in oracle_b), (trial, index)
        assert bits_a.count() == len(oracle_a), trial
        assert bits_b.count() == len(oracle_b), trial
        assert bits_a.iterate() == sorted(oracle_a), trial
        assert bits_b.iterate() == sorted(oracle_b), trial

        union = bits_a.union(bits_b)
        intersection = bits_a.intersection(bits_b)
        assert union.count() == len(oracle_a | oracle_b), trial
        assert intersection.count() == len(oracle_a & oracle_b), trial
        assert union.iterate() == sorted(oracle_a | oracle_b), trial
        assert intersection.iterate() == sorted(oracle_a & oracle_b), trial
        assert union.iterate() == sorted(set(union.iterate()))            # ascending, deduped
        for index in range(0, UNIVERSE, 7):
            assert union.contains(index) == (index in (oracle_a | oracle_b)), trial
            assert intersection.contains(index) == (index in (oracle_a & oracle_b)), trial


if __name__ == "__main__":
    main()
    print("PASS")
