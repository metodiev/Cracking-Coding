"""Task 255 - Implement rolling bloom filter (sliding window). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

A sliding-window bloom filter made of S slices (S = window length in steps). Every insert
writes its H probes into ALL slices; every step() clears exactly one slice, rotating through
them. check() ORs the slices, so an item inserted at step t cannot be missed while fewer
than S clears have happened (its bits are still intact in some slice) and is guaranteed
gone once S clears have passed. False positives are possible - that direction is expected,
never a false negative inside the window. Same stable hash as task 254: FNV-1a 32-bit plus
a MurmurHash3 fmix32 avalanche, then double hashing with an odd step.
"""

SLICES = 4
BITS_PER_SLICE = 64
HASHES = 3

# Deterministic expectation for the "loaded" scenario in main (cross-language check).
EXPECTED_HITS = 4


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


def positions(item, bits, hashes):
    h1 = fmix32(fnv1a(item))
    h2 = fmix32(fnv1a(item + "#")) | 1
    return [(h1 + i * h2) % bits for i in range(hashes)]


class RollingBloom:
    def __init__(self, slices=SLICES, bits=BITS_PER_SLICE, hashes=HASHES):
        self.slices = [bytearray(bits) for _ in range(slices)]
        self.bits = bits
        self.hashes = hashes
        self.current = 0

    def insert(self, item):
        probes = positions(item, self.bits, self.hashes)
        for word in self.slices:
            for p in probes:
                word[p] = 1

    def step(self):
        self.current = (self.current + 1) % len(self.slices)
        self.slices[self.current] = bytearray(self.bits)

    def check(self, item):
        probes = positions(item, self.bits, self.hashes)
        return any(all(word[p] for p in probes) for word in self.slices)


def main():
    rb = RollingBloom()
    assert not rb.check("alpha")                         # empty window: guaranteed miss

    rb.insert("alpha")
    assert rb.check("alpha")
    assert all(sum(word) == HASHES for word in rb.slices)    # 3 distinct probes per slice

    rb.step()
    assert rb.check("alpha")                             # age 1 < window: still found
    assert sum(1 for word in rb.slices if sum(word) == 0) == 1   # exactly one rotation

    ghost = RollingBloom()
    ghost.insert("solo")
    for _ in range(SLICES):
        ghost.step()
    assert not ghost.check("solo")                       # older than the window: expired
    assert all(sum(word) == 0 for word in ghost.slices)

    keep = RollingBloom()
    keep.insert("keep")
    for _ in range(SLICES - 1):
        keep.step()
    assert keep.check("keep")                            # age 3: still inside the window
    keep.step()
    assert not keep.check("keep")                        # age 4: every slice was cleared

    timeline = RollingBloom()
    history = []
    step_no = 0

    def add_now(item):
        timeline.insert(item)
        history.append((step_no, item))

    def assert_no_false_negative():
        for t, item in history:
            if step_no - t <= SLICES - 1:
                assert timeline.check(item), item        # inside the window: must be found
            # older items may or may not be found (false positives allowed)

    add_now("alpha")
    for i in range(1, 9):
        timeline.step()
        step_no += 1
        add_now("w%d" % i)
        assert_no_false_negative()

    assert not timeline.check("alpha")                   # 8 steps old: fully evicted

    loaded = RollingBloom()
    members = ["member-%d" % i for i in range(8)]
    for item in members:
        loaded.insert(item)
    assert all(loaded.check(item) for item in members)   # no false negatives when loaded
    hits = sum(1 for i in range(100) if loaded.check("q%d" % i))
    assert hits == EXPECTED_HITS
    assert hits >= 1                                     # false positives do happen: expected


if __name__ == "__main__":
    main()
    print("PASS")
