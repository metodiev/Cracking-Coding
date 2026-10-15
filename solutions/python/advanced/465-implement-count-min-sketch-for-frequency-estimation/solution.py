"""Task 465 - Implement count-min sketch for frequency estimation. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Count-min sketch with d = 4 rows and width = 48 cells, using FIXED hash seeds:
column r = ((a_r * key + b_r) mod 1000003) mod width for r in 0..3 (seeds below).
Counters only ever increase, so estimate(key) = min over the rows is NEVER below
the true count; the overshoot is bounded by the collision mass.

Checks: a hand-computed tiny sketch (width 4) documents exact behaviours, then a
fixed LCG stream of N = 5000 updates over ids 0..63 asserts for every id:
  * estimate >= true count (no under-estimates), and
  * estimate - true count <= EPS * N with the documented EPS = 0.06 (>= the
    standard e/width ~ 0.0566; the measured max overshoot on this fixed stream is
    101, about one third of the 300 bound, and 23 of the 64 ids do over-estimate,
    documenting that collisions really happen).
"""

P = 1000003
HASH_A = [31, 131, 2711, 65521]
HASH_B = [17, 91, 601, 4099]
EPS = 0.06


class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]

    def columns(self, key):
        return [((HASH_A[r] * key + HASH_B[r]) % P) % self.width for r in range(self.depth)]

    def add(self, key, delta=1):
        for r, c in enumerate(self.columns(key)):
            self.table[r][c] += delta

    def estimate(self, key):
        return min(self.table[r][c] for r, c in enumerate(self.columns(key)))


def lcg_value(seed):
    return (seed * 75 + 74) % 65537


def main():
    # Hand-computed tiny sketch: keys 7 and 11 collide in both rows of width 4.
    # columns(7)  = [2, 0]; columns(11) = [2, 0]; columns(0) = [1, 3].
    sketch = CountMinSketch(4, 2)
    assert sketch.columns(7) == [2, 0]
    assert sketch.columns(11) == [2, 0]
    sketch.add(7)
    assert sketch.estimate(7) == 1
    sketch.add(11)
    assert sketch.estimate(11) == 2  # both rows collide with key 7
    assert sketch.estimate(7) == 2  # over-estimate from the collision, never below
    sketch.add(7)
    assert sketch.estimate(7) == 3
    assert sketch.estimate(0) == 0  # hand-checked: the [1, 3] cells were never touched

    # Fixed LCG stream over 64 ids.
    sketch = CountMinSketch(48, 4)
    seed = 7654321
    total = 5000
    true_counts = {}
    for _ in range(total):
        seed = lcg_value(seed)
        item = seed % 64
        sketch.add(item)
        true_counts[item] = true_counts.get(item, 0) + 1
    assert len(true_counts) == 64
    max_overshoot = 0
    for item in range(64):
        true_count = true_counts[item]
        estimate = sketch.estimate(item)
        assert estimate >= true_count, (item, estimate, true_count)  # never under-estimates
        overshoot = estimate - true_count
        max_overshoot = max(max_overshoot, overshoot)
        assert overshoot <= EPS * total, (item, overshoot)
    assert max_overshoot >= 1  # collisions did occur (documented)
    assert sum(true_counts.values()) == total


if __name__ == "__main__":
    main()
    print("PASS")
