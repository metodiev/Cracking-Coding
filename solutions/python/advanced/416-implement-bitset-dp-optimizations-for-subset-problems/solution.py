"""Task 416 - Implement bitset DP optimizations for subset problems. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Subset-sum reachability with a bitset: bit s of `mask` is set iff some subset of the
processed values sums to s. Adding value v shifts the whole reachable set by v:
    mask |= mask << v        (masked to the [0, limit] window after every step)
Python's arbitrary-precision int IS the bitset (one word per 30 bits internally), so a
single OR/SHIFT pair replaces the whole inner loop of the O(n * limit) boolean DP.
The classic boolean DP is kept as the oracle and the reachable sets must match exactly.
Limits beyond 63 bits (e.g. 4000 or 20000) are exercised on purpose: fixed-width
integer bitsets would silently overflow there, Python ints cannot.
"""


def reachable_mask(values, limit):
    mask = 1                                  # the empty subset sums to 0
    window = (1 << (limit + 1)) - 1
    for v in values:
        if v <= limit:
            mask |= (mask << v) & window
    return mask & window


def can_make(values, target, limit=None):
    if limit is None:
        limit = target
    if target < 0 or target > limit:
        return False
    return (reachable_mask(values, limit) >> target) & 1 == 1


def can_partition(values):
    total = sum(values)
    if total % 2 != 0:
        return False
    return can_make(values, total // 2)


def classic_dp(values, limit):
    """Oracle: plain boolean DP over sums 0..limit."""
    reach = [False] * (limit + 1)
    reach[0] = True
    for v in values:
        if v > limit:
            continue
        for s in range(limit, v - 1, -1):
            if reach[s - v]:
                reach[s] = True
    return reach


def lcg_values(count, seed, lo, hi):
    values = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        values.append(lo + seed % (hi - lo + 1))
    return values, seed


def main():
    # classic subset-sum hand cases
    base = [3, 34, 4, 12, 5, 2]
    assert can_make(base, 9) is True            # 3 + 4 + 2 (or 5 + 4)
    assert can_make(base, 30) is False          # sums jump from 26 to 34+
    assert can_make(base, 34) is True           # the single element 34
    assert can_make(base, 0) is True            # the empty subset
    assert can_make(base, 60) is True           # everything
    assert can_make(base, 59) is False
    assert can_make(base, 50) is True           # 34 + 12 + 4

    # hand-derived reachable set of {2, 3, 4, 5, 12, 34}: {2,3,4,5} reaches
    # {0, 2..12, 14} (1 and 13 stay empty), adding 12 fills {0, 2..12, 14..24, 26}
    # (13 and 25 stay empty), and adding 34 mirrors that set 34 higher:
    # {0, 2..12, 14..24, 26, 34, 36..46, 48..58, 60} — 48 values.
    mask = reachable_mask(base, 60)
    assert bin(mask).count("1") == 48
    for s in (0, 2, 12, 14, 24, 26, 34, 36, 46, 48, 58, 60):
        assert ((mask >> s) & 1) == 1, s
    for s in (1, 13, 25, 35, 47, 59):
        assert ((mask >> s) & 1) == 0, s
    assert ((mask >> 25) & 1) == 0
    assert can_make(base, 25) is False

    # equal-partition hand cases
    assert can_partition([1, 5, 11, 5]) is True   # 11 vs 1 + 5 + 5
    assert can_partition([1, 5, 3]) is False      # odd total
    assert can_partition([2, 2, 2, 2]) is True

    # oracle agreement on hand and LCG sets
    for values, limit in [(base, 60), ([1, 5, 11, 5], 22), ([2, 2, 2, 2], 8), ([7], 7), ([], 5)]:
        mask = reachable_mask(values, limit)
        oracle = classic_dp(values, limit)
        for s in range(limit + 1):
            assert (((mask >> s) & 1) == 1) == oracle[s], (values, s)

    seed = 416
    for round_index in range(8):
        n = 10 + seed % 31
        values, seed = lcg_values(n, seed, 1, 60)
        limit = min(sum(values), 1500)
        mask = reachable_mask(values, limit)
        oracle = classic_dp(values, limit)
        for s in range(limit + 1):
            assert (((mask >> s) & 1) == 1) == oracle[s], (round_index, s)
        assert mask.bit_length() <= limit + 1
        # every prefix subset sum is reachable, and its double is not necessarily
        running = 0
        for v in values:
            running += v
            if running <= limit:
                assert ((mask >> running) & 1) == 1
        # with the window fixed at `limit`, larger targets are truncated by design
        assert not can_make(values, limit + 1, limit)

    # limits beyond 63 bits: a single 4000-value jump needs bit 4000
    mask = reachable_mask([4000], 5000)
    assert ((mask >> 4000) & 1) == 1 and ((mask >> 4001) & 1) == 0
    assert mask.bit_length() == 4001
    assert can_make([4000], 4000) is True and can_make([4000], 3999) is False

    # n = 40 stress: no 32/64-bit integer could hold these masks
    seed = 65537 // 2
    for round_index in range(3):
        values, seed = lcg_values(40, seed, 1, 1000)
        limit = 20000
        mask = reachable_mask(values, limit)
        oracle = classic_dp(values, limit)
        reached = 0
        for s in range(limit + 1):
            bit = ((mask >> s) & 1) == 1
            assert bit == oracle[s], (round_index, s)
            reached += 1 if bit else 0
        # one concrete subset must be reachable at any size
        subset_sum = sum(values[:20])
        if subset_sum <= limit:
            assert ((mask >> subset_sum) & 1) == 1
        assert mask.bit_length() <= limit + 1
        # the mask grows monotonically with each value: prefix property
        running = 0
        for v in values[:5]:
            running += v
            assert ((mask >> running) & 1) == 1, (round_index, running)
        assert reached >= 1

    # partition on n = 40 inputs (both parities exercised)
    values = list(range(1, 41))               # total 820, even
    assert can_partition(values) is True      # with 1..n every sum 0..n(n+1)/2 is reachable
    values = list(range(1, 42))               # total 861, odd
    assert can_partition(values) is False


if __name__ == "__main__":
    main()
    print("PASS")
