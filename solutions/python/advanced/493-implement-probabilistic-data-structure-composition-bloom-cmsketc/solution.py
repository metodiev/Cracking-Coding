"""Task 493 - Implement probabilistic data structure composition (bloom + cmsketch). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A SketchPipeline composes two sketches over the same item stream: a Bloom filter (relaxed
membership) and a Count-Min sketch (frequency estimation with d rows and w counters).
membership(item) = bloom.maybe_contains(item) AND cms.estimate(item) >= 1 - so a positive
answer always implies an estimated frequency of at least 1, and a Bloom false positive whose
every CMS counter is still empty is rejected (the pipeline tightens membership without ever
introducing a false negative). Hashing is the stable task-254 scheme: h1 = fmix32(fnv1a(item)),
h2 = fmix32(fnv1a(item + '#')) | 1 for the Bloom probes, and per-row fmix32(fnv1a(item + '|r'))
for the CMS; all arithmetic is 32-bit integer work, identical in all four languages. Documented
guarantees: CMS estimates never underestimate the true count and overestimate by at most
epsilon*N with epsilon = e/w and delta = e^(-d) (w = 256, d = 4, N = 754 updates); on 5000
absent probes the Bloom rate (m = 1024, k = 7, n = 250 items, theory ~25.6%) stays under the
documented 28% tolerance, and the pipeline rejects most of those Bloom hits because their CMS
counters are still empty - with the same Bloom filter and a width-1024 CMS almost all are
rejected. The exact popcounts and false-positive counts are asserted as cross-language
constants.
"""


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
    def __init__(self, bits, hashes):
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


class CountMinSketch:
    def __init__(self, width, rows):
        self.w = width
        self.d = rows
        self.counters = [[0] * width for _ in range(rows)]

    def column(self, row, item):
        return fmix32(fnv1a(item + "|" + str(row))) % self.w

    def add(self, item, delta=1):
        for row in range(self.d):
            self.counters[row][self.column(row, item)] += delta

    def estimate(self, item):
        return min(self.counters[row][self.column(row, item)] for row in range(self.d))

    def total(self):
        return sum(sum(row) for row in self.counters)


class SketchPipeline:
    def __init__(self, bloom_bits=1024, bloom_hashes=7, cms_width=256, cms_rows=4):
        self.bloom = BloomFilter(bloom_bits, bloom_hashes)
        self.cms = CountMinSketch(cms_width, cms_rows)

    def add(self, item, count=1):
        self.bloom.add(item)
        self.cms.add(item, count)

    def estimate(self, item):
        return self.cms.estimate(item)

    def membership(self, item):
        """True only when the Bloom filter accepts AND the CMS sees a nonzero count."""
        return self.bloom.maybe_contains(item) and self.cms.estimate(item) >= 1


def lcg_counts(item_count, seed):
    """Deterministic insertion stream: item key-<j> gets 1 + seed % 5 copies."""
    counts = {}
    for j in range(item_count):
        seed = (seed * 75 + 74) % 65537
        counts["key-%d" % j] = 1 + seed % 5
    return counts


def main():
    counts = lcg_counts(250, 123)
    total_updates = sum(counts.values())
    pipeline = SketchPipeline()
    for item, count in counts.items():
        pipeline.add(item, count)

    # Guarantee 1 - no false negatives: every inserted item is a member and its CMS estimate
    # never underestimates the true count (the Count-Min guarantee), within the additive
    # epsilon*N bound, epsilon = e / width.
    slack = (2.718281828459045 / pipeline.cms.w) * total_updates
    for item, true_count in counts.items():
        assert pipeline.membership(item) is True, "no false negatives"
        estimate = pipeline.estimate(item)
        assert estimate >= true_count, "CMS must never underestimate"
        assert estimate <= true_count + slack, "CMS additive error bound"
        assert pipeline.bloom.maybe_contains(item) is True

    # Guarantee 2 - membership implies estimate >= 1, for members and non-members alike
    # (membership is defined as bloom.maybe_contains AND estimate >= 1).
    for item in counts:
        if pipeline.membership(item):
            assert pipeline.estimate(item) >= 1

    # Absent item with no colliding counters: estimate is exactly 0, membership is False.
    assert pipeline.estimate("never-seen") == 0
    assert pipeline.membership("never-seen") is False

    # False-positive behaviour on 5000 absent probes. The shared Bloom config (m=1024, k=7,
    # n=250) gives k*n/m = 1.71 and theory (1 - e^(-kn/m))^k ~ 25.6%, so the documented
    # tolerance is 1400 (28%); the pipeline rate is far lower because a probe must ALSO have
    # all four CMS counters non-zero, and the width-256 sketch is only ~62% saturated.
    probes = ["probe-%d" % j for j in range(5000)]
    bloom_fp = sum(1 for item in probes if pipeline.bloom.maybe_contains(item))
    pipeline_fp = sum(1 for item in probes if pipeline.membership(item))
    assert bloom_fp <= 1400, "documented Bloom tolerance"
    assert pipeline_fp <= bloom_fp - 1000, "the pipeline must reject most Bloom false hits"
    assert pipeline_fp <= 200, "documented pipeline tolerance"

    # Tightness demonstration with the same Bloom filter but a wider CMS (width 1024): a
    # Bloom false positive whose four counters are all still zero is rejected by the
    # estimate >= 1 condition, so the pipeline rate stays strictly below the Bloom rate.
    tight = SketchPipeline(bloom_bits=1024, bloom_hashes=7, cms_width=1024, cms_rows=4)
    for item, count in counts.items():
        tight.add(item, count)
    tight_bloom_fp = sum(1 for item in probes if tight.bloom.maybe_contains(item))
    tight_pipeline_fp = sum(1 for item in probes if tight.membership(item))
    assert tight_pipeline_fp < tight_bloom_fp, "cold counters must reject some Bloom hits"
    assert tight_pipeline_fp <= tight_bloom_fp - 1000
    for item, true_count in counts.items():
        assert tight.membership(item) is True
        assert tight.estimate(item) >= true_count
        assert tight.estimate(item) <= true_count + 1, "wider sketch: error <= e*N/w"

    # Exact cross-language bookkeeping: deterministic 32-bit hashing means popcounts and the
    # measured false positives are the same in every implementation.
    assert total_updates == 754
    assert pipeline.bloom.popcount() == 842
    assert tight.bloom.popcount() == 842
    assert bloom_fp == 1296
    assert pipeline_fp == 194
    assert tight_bloom_fp == 1296
    assert tight_pipeline_fp == 5

    # A single-item pipeline (sanity): membership/estimate agree, nothing else is a member.
    solo = SketchPipeline()
    solo.add("solo", 3)
    assert solo.membership("solo") is True and solo.estimate("solo") == 3
    for probe in ("solo#", "solo1", "sol", "proxy-%d" % 7):
        if solo.membership(probe):
            assert solo.estimate(probe) >= 1
    assert solo.cms.total() == 12, "4 rows x 3 copies"


if __name__ == "__main__":
    main()
    print("PASS")
