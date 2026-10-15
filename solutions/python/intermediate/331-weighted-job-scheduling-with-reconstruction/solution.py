"""Task 331 - Weighted job scheduling with reconstruction. (Tier: Intermediate, CTCI Ch 8)

Jobs are (start, end, weight) half-open intervals, so end == next start is not an overlap.
Sort by end time, let p[i] be the last job finishing at or before job i starts, then
dp[i] = max(dp[i - 1], weight[i] + dp[p[i]]). Descending through the same dp/pred pair
reconstructs one optimal set. Cross-checked against exhaustive subset search on
deterministic LCG instances of up to 12 jobs.
"""


def weighted_schedule(jobs):
    """Returns (best total weight, sorted indices of one optimal job set)."""
    n = len(jobs)
    if n == 0:
        return 0, []
    order = sorted(range(n), key=lambda i: (jobs[i][1], jobs[i][0]))
    ends = [jobs[i][1] for i in order]

    pred = [-1] * n
    for i in range(n):
        lo, hi = 0, i
        while lo < hi:                       # first index whose end exceeds this start
            mid = (lo + hi) // 2
            if ends[mid] <= jobs[order[i]][0]:
                lo = mid + 1
            else:
                hi = mid
        pred[i] = lo - 1

    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        dp[i] = max(dp[i - 1], jobs[order[i - 1]][2] + dp[pred[i - 1] + 1])

    chosen = []
    i = n
    while i > 0:
        if dp[i] > dp[i - 1]:                # job order[i - 1] belongs to this optimum
            chosen.append(order[i - 1])
            i = pred[i - 1] + 1
        else:
            i -= 1
    chosen.reverse()
    return dp[n], chosen


def brute_schedule(jobs):
    """Exhaustive best weight over all subsets (n <= 12)."""
    n = len(jobs)
    best = 0
    for mask in range(1 << n):
        weight = 0
        picked = []
        for i in range(n):
            if (mask >> i) & 1:
                if any(overlaps(jobs[i], jobs[j]) for j in picked):
                    break
                picked.append(i)
                weight += jobs[i][2]
        else:
            best = max(best, weight)
    return best


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def lcg_jobs(count, seed):
    jobs = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        start = seed % 30
        seed = (seed * 75 + 74) % 65537
        length = 1 + seed % 8
        seed = (seed * 75 + 74) % 65537
        jobs.append((start, start + length, 1 + seed % 20))
    return jobs, seed


def main():
    # Hand-checked: two chains, best picks (1,3,5w) + (4,6,5w) + (6,7,4w) = 14.
    hand = [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4)]
    weight, chosen = weighted_schedule(hand)
    assert weight == 14 and chosen == [0, 2, 3], (weight, chosen)
    assert sum(hand[i][2] for i in chosen) == weight
    assert all(not overlaps(hand[i], hand[j]) for i in chosen for j in chosen if i < j)
    assert weight == brute_schedule(hand)

    assert weighted_schedule([]) == (0, [])
    assert weighted_schedule([(5, 9, 7)]) == (7, [0])

    # Touching intervals (end == next start) do not overlap.
    touch = [(0, 5, 3), (5, 10, 4)]
    assert weighted_schedule(touch) == (7, [0, 1])

    # A single heavy job beats two light ones.
    heavy = [(0, 10, 100), (0, 4, 3), (6, 10, 3)]
    weight, chosen = weighted_schedule(heavy)
    assert weight == 100 and chosen == [0], (weight, chosen)

    seed = 9091
    for trial in range(200):
        seed = (seed * 75 + 74) % 65537
        n = 1 + seed % 12
        jobs, seed = lcg_jobs(n, seed)
        weight, chosen = weighted_schedule(jobs)
        assert weight == brute_schedule(jobs), (trial, jobs)
        assert len(set(chosen)) == len(chosen)
        assert all(not overlaps(jobs[i], jobs[j]) for i in chosen for j in chosen if i < j)
        assert sum(jobs[i][2] for i in chosen) == weight, (trial, jobs, chosen)


if __name__ == "__main__":
    main()
    print("PASS")
