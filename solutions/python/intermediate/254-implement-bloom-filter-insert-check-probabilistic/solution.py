"""Task 254 - Implement bloom filter insert/check (probabilistic). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

An m-bit filter with k probes. The hash is stable across runs and languages: h1 = fmix32 of
the FNV-1a 32-bit hash of the item, h2 = fmix32(fnv1a(item + "#")) with the low bit forced
to 1; probe i lands on (h1 + i * h2) mod m (double hashing; h2 is odd and m a power of two,
so the probes are distinct). fmix32 (the MurmurHash3 avalanche: xor-shift, multiply by
0x85ebca6b, xor-shift, multiply by 0xc2b2ae35, xor-shift) is needed because raw FNV-1a low
bits are poorly mixed and measured ~10x the theoretical false-positive rate. No randomness
anywhere: every run is identical.
"""

M_BITS = 1024
K_HASHES = 7

# Deterministic values for the exact test set below (see main); they double as a
# cross-language check that all four implementations hash bit for bit the same way.
EXPECTED_BITS = 503
EXPECTED_FP = 74


def fnv1a(text):
    h = 2166136261
    for ch in text:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def fmix32(x):
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return x


class BloomFilter:
    def __init__(self, bits=M_BITS, hashes=K_HASHES):
        self.m = bits
        self.k = hashes
        self.words = bytearray(bits)

    def positions(self, item):
        h1 = fmix32(fnv1a(item))
        h2 = fmix32(fnv1a(item + "#")) | 1
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item):
        for p in self.positions(item):
            self.words[p] = 1

    def maybe_contains(self, item):
        return all(self.words[p] for p in self.positions(item))

    def popcount(self):
        return sum(self.words)


def main():
    f = BloomFilter()
    assert f.popcount() == 0
    for probe in ("item-0", "x", "hello"):
        assert not f.maybe_contains(probe)                # empty filter: everything misses

    members = ["item-%d" % i for i in range(100)]
    for item in members:
        f.add(item)
    assert all(f.maybe_contains(item) for item in members)   # no false negatives
    filled = f.popcount()
    f.add("item-0")
    assert f.popcount() == filled                        # inserting again is a no-op

    non_members = ["other-%d" % i for i in range(10000)]
    false_positives = sum(1 for item in non_members if f.maybe_contains(item))

    assert f.popcount() == EXPECTED_BITS
    assert false_positives == EXPECTED_FP
    # Documented bound: theory gives (1 - e^(-k*n/m))^k ~ 0.7% false positives for
    # n=100, m=1024, k=7; we require the measured rate to stay under 2% (200/10000).
    assert false_positives <= 200

    small = BloomFilter(bits=256, hashes=4)
    for item in ("a", "b", "c"):
        small.add(item)
    assert all(small.maybe_contains(item) for item in ("a", "b", "c"))


if __name__ == "__main__":
    main()
    print("PASS")
