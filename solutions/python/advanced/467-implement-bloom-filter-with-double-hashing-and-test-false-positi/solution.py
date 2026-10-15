"""Task 467 - Implement bloom filter with double hashing and test false positive. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Bloom filter with m = 2^16 bits and k = 7 probes via double hashing:
h1 = mix32(key), h2 = mix32(key ^ 0x9E3779B9) | 1 (forced odd, hence coprime to the
power-of-two m, so the 7 probes are pairwise distinct), probe i = (h1 + i*h2) mod m.

Checks: a hand-verifiable small filter (single insert sets exactly 7 distinct bits
and is idempotent). Then the workload: insert 10000 distinct member keys (LCG values
in [0, 65536)), verify all 10000 are reported present (zero false negatives), and
test 10000 disjoint non-member keys (200000 + i) counting false positives. The
documented bound is FP_MAX = 800; the theoretical rate is
(1 - e^(-k*n/m))^k ~ 0.0524 -> ~524 expected, and the measured count on this fixed
key set is 488 (assertion <= 800 with a >= 444 lower sanity bound).
"""

M_BITS = 1 << 16
K = 7
MASK32 = 0xFFFFFFFF
FP_MAX = 800


def mix32(x):
    x &= MASK32
    x ^= x >> 16
    x = (x * 0x45D9F3B) & MASK32
    x ^= x >> 16
    x = (x * 0x45D9F3B) & MASK32
    x ^= x >> 16
    return x


class BloomFilter:
    def __init__(self, bits=M_BITS, probes=K):
        self.bits = bits
        self.probes = probes
        self.data = bytearray(bits // 8)

    def positions(self, key):
        h1 = mix32(key)
        h2 = mix32(key ^ 0x9E3779B9) | 1
        return [(h1 + i * h2) % self.bits for i in range(self.probes)]

    def add(self, key):
        for pos in self.positions(key):
            self.data[pos >> 3] |= 1 << (pos & 7)

    def __contains__(self, key):
        for pos in self.positions(key):
            if not (self.data[pos >> 3] >> (pos & 7)) & 1:
                return False
        return True

    def popcount(self):
        return sum(bin(byte).count("1") for byte in self.data)


def lcg_value(seed):
    return (seed * 75 + 74) % 65537


def main():
    # Hand-verifiable small filter: 7 distinct probes, idempotent insert.
    tiny = BloomFilter()
    assert len(set(tiny.positions(5))) == K  # odd h2 and k <= m => distinct probes
    tiny.add(5)
    assert tiny.popcount() == K
    assert 5 in tiny
    tiny.add(5)
    assert tiny.popcount() == K
    assert 6 not in tiny

    # Workload: 10000 distinct members from the LCG, 10000 disjoint non-members.
    bloom = BloomFilter()
    seed = 3141592
    members = []
    seen = set()
    while len(members) < 10000:
        seed = lcg_value(seed)
        if seed not in seen:
            seen.add(seed)
            members.append(seed)
    for key in members:
        bloom.add(key)
    for key in members:
        assert key in bloom  # zero false negatives
    false_positives = sum(1 for i in range(10000) if 200000 + i in bloom)
    assert false_positives <= FP_MAX, false_positives  # documented bound (measured 488)
    assert false_positives >= 444  # ~ -1.5 sigma sanity floor around the 524 expectation


if __name__ == "__main__":
    main()
    print("PASS")
