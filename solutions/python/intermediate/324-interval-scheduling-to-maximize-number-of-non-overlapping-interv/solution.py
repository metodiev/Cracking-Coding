"""Task 324 - Interval scheduling to maximize number of non-overlapping intervals (greedy). (Tier: Intermediate, CTCI Ch 1)

Earliest-finish-time greedy: sort by finish, always take the next interval that
starts strictly after the last finish.  Touching endpoints conflict, so an interval
is compatible only when previous.end < next.start (that is the convention that makes
the classic example [[1,2],[2,3],[3,4],[1,3]] answer 2, not 3).  The greedy count is
cross-checked against an independent DP over the same definition.
"""


def max_non_overlapping(intervals):
    """Greedy earliest finish time; touching endpoints count as overlapping."""
    ordered = sorted(intervals, key=lambda interval: (interval[1], interval[0]))
    count = 0
    last_end = None
    for start, end in ordered:
        if last_end is None or start > last_end:
            count += 1
            last_end = end
    return count


def max_non_overlapping_dp(intervals):
    """Independent DP: best[i] = max(best[i-1], 1 + best[p(i)]) with p by binary search."""
    if not intervals:
        return 0
    ordered = sorted(intervals, key=lambda interval: (interval[1], interval[0]))
    finishes = [interval[1] for interval in ordered]
    best = [0] * (len(ordered) + 1)
    for i in range(1, len(ordered) + 1):
        start = ordered[i - 1][0]
        lo, hi = 0, i - 1                        # last j < i with finishes[j-1] < start
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if finishes[mid - 1] < start:
                lo = mid
            else:
                hi = mid - 1
        j = lo if lo > 0 and finishes[lo - 1] < start else 0
        best[i] = max(best[i - 1], 1 + best[j])
    return best[len(ordered)]


def lcg_intervals(count, seed):
    intervals = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        start = seed % 20
        seed = (seed * 75 + 74) % 65537
        width = 1 + seed % 6
        intervals.append([start, start + width])
    return intervals, seed


def main():
    # classic example: [1,2] and [2,3] touch, so they conflict; best is 2
    assert max_non_overlapping([[1, 2], [2, 3], [3, 4], [1, 3]]) == 2
    assert max_non_overlapping_dp([[1, 2], [2, 3], [3, 4], [1, 3]]) == 2

    # chain with gaps: all four fit; the wide decoy is not worth taking
    chain = [[1, 2], [3, 4], [5, 6], [7, 8], [1, 8]]
    assert max_non_overlapping(chain) == 4
    assert max_non_overlapping_dp(chain) == 4
    assert max_non_overlapping([[1, 2], [3, 4], [5, 6], [7, 8]]) == 4

    assert max_non_overlapping([]) == 0
    assert max_non_overlapping([[1, 5]]) == 1
    assert max_non_overlapping([[1, 3], [2, 4], [3, 5], [4, 6]]) == 2
    assert max_non_overlapping([[1, 2], [1, 2], [1, 2]]) == 1
    assert max_non_overlapping([[1, 4], [4, 4]]) == 1           # touching conflicts

    seed = 909
    for trial in range(400):
        count = 1 + seed % 9
        intervals, seed = lcg_intervals(count, seed)
        greedy = max_non_overlapping(intervals)
        assert greedy == max_non_overlapping_dp(intervals), intervals


if __name__ == "__main__":
    main()
    print("PASS")
