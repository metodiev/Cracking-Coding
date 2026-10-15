"""Task 398 - Implement a divide-and-conquer DP optimization example. (Tier: Intermediate, CTCI Ch 8)

Problem: split an array into exactly k contiguous non-empty groups, minimizing the sum of the
squared group sums. f[t][i] = min over j of f[t-1][j] + (prefix[i]-prefix[j])^2. The squared
cost obeys the quadrangle inequality, so for each layer opt(i) is non-decreasing: the
divide-and-conquer optimization solves a whole layer in O(n log n) evaluations instead of
O(n^2) by scanning only [optL, optR] for the midpoint and recursing with narrowed windows.
The suite asserts opt monotonicity per layer and cross-checks against the plain O(k n^2) DP.
"""

INF = float("inf")


def partition_naive(values, k):
    n = len(values)
    if k > n:
        return INF
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + values[i - 1]
    previous = [0] + [INF] * n
    for t in range(1, k + 1):
        current = [INF] * (n + 1)
        for i in range(t, n + 1):
            for j in range(t - 1, i):
                if previous[j] == INF:
                    continue
                group = prefix[i] - prefix[j]
                cost = previous[j] + group * group
                if cost < current[i]:
                    current[i] = cost
        previous = current
    return previous[n]


def partition_dc(values, k):
    """Returns (answer, opt_choices per layer) so monotonicity can be asserted."""
    n = len(values)
    if k > n:
        return INF, []
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + values[i - 1]
    previous = [0] + [INF] * n
    all_opts = []
    for t in range(1, k + 1):
        current = [INF] * (n + 1)
        opt = [0] * (n + 1)
        # i == t: the only split is j == t-1; the last singleton group costs values[t-1]^2
        current[t] = previous[t - 1] + values[t - 1] * values[t - 1]
        opt[t] = t - 1
        if t + 1 <= n:
            solve(previous, current, opt, prefix, t + 1, n, t - 1, n - 1)
        all_opts.append(opt)
        previous = current
    return previous[n], all_opts


def solve(previous, current, opt, prefix, l, r, optl, optr):
    if l > r:
        return
    mid = (l + r) // 2
    best, best_j = INF, optl
    upper = min(optr, mid - 1)
    for j in range(optl, upper + 1):
        if previous[j] == INF:
            continue
        group = prefix[mid] - prefix[j]
        cost = previous[j] + group * group
        if cost < best:
            best, best_j = cost, j
    current[mid] = best
    opt[mid] = best_j
    solve(previous, current, opt, prefix, l, mid - 1, optl, best_j)
    solve(previous, current, opt, prefix, mid + 1, r, best_j, optr)


def opt_is_monotone(opt, t, n):
    last = -1
    for i in range(t, n + 1):
        if opt[i] < last:
            return False
        last = opt[i]
    return True


def lcg_arrays(count, seed, max_len=60):
    arrays = []
    for _ in range(count):
        n = 2 + seed % (max_len - 1)
        values = []
        for _ in range(n):
            seed = (seed * 75 + 74) % 65537
            values.append(1 + seed % 9)
        arrays.append(values)
    return arrays


def main():
    assert partition_naive([1, 2, 3, 4], 2) == 52
    assert partition_dc([1, 2, 3, 4], 2)[0] == 52
    assert partition_naive([1, 2, 3, 4], 1) == 100
    assert partition_naive([1, 2, 3, 4], 4) == 30
    assert partition_naive([5], 1) == 25
    assert partition_naive([1, 2], 3) == INF
    assert partition_dc([1, 2, 3, 4], 4)[0] == 30

    for values in lcg_arrays(40, seed=804):
        n = len(values)
        for k in range(1, min(n, 8) + 1):
            naive = partition_naive(values, k)
            dc, all_opts = partition_dc(values, k)
            assert dc == naive, (values, k, naive, dc)
            for t, opt in enumerate(all_opts, start=1):
                assert opt_is_monotone(opt, t, n), (values, k, t)


if __name__ == "__main__":
    main()
    print("PASS")
