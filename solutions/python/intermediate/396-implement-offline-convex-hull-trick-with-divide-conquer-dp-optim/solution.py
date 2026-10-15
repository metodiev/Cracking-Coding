"""Task 396 - Implement the offline convex hull trick with divide-and-conquer DP optimization. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

Problem: cover n sorted points with exactly k contiguous segments, minimizing the sum of
squared segment spans (a cost with the quadrangle inequality). f[t][i] = min over j of
f[t-1][j] + (x[i-1]-x[j])^2, solved twice:
  * offline convex hull trick - expand the square and read each layer as lines
    (slope -2x[j], intercept f[t-1][j]+x[j]^2) queried at increasing points; slopes are
    non-increasing and queries non-decreasing, so the hull is monotone and queries use a
    binary search over intersection points. All queries are known up front: offline.
  * divide-and-conquer optimization - for the same layer opt(i) is non-decreasing, so
    solve(mid) scans only [optl, optr] and recurses on both halves.
Both are cross-checked against the plain O(k n^2) DP.
"""

INF = float("inf")


class MinHull:
    """Offline hull for minimum: insert lines with NON-INCREASING slopes, query increasing x."""

    def __init__(self):
        self.ms = []
        self.bs = []

    def add_line(self, m, b):
        if self.ms and m == self.ms[-1]:
            if b >= self.bs[-1]:
                return                      # same slope, worse intercept: dominated
            self.ms.pop()
            self.bs.pop()
        while len(self.ms) >= 2:
            m1, b1 = self.ms[-2], self.bs[-2]
            m2, b2 = self.ms[-1], self.bs[-1]
            # l2 is unnecessary when the intersection of l1 and the new line is not after l1 cap l2
            if (b - b1) * (m1 - m2) <= (b2 - b1) * (m1 - m):
                self.ms.pop()
                self.bs.pop()
            else:
                break
        self.ms.append(m)
        self.bs.append(b)

    def query(self, x):
        if not self.ms:
            raise ValueError("query on an empty hull")
        lo, hi = 0, len(self.ms) - 1
        while lo < hi:                      # values along the hull are convex in the index
            mid = (lo + hi) // 2
            if self.ms[mid] * x + self.bs[mid] <= self.ms[mid + 1] * x + self.bs[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        return self.ms[lo] * x + self.bs[lo]


def partition_bruteforce(xs, k):
    n = len(xs)
    if k > n:
        return INF
    previous = [0] + [INF] * n
    for t in range(1, k + 1):
        current = [INF] * (n + 1)
        for i in range(t, n + 1):
            for j in range(t - 1, i):
                if previous[j] == INF:
                    continue
                cost = previous[j] + (xs[i - 1] - xs[j]) ** 2
                if cost < current[i]:
                    current[i] = cost
        previous = current
    return previous[n]


def partition_offline_cht(xs, k):
    n = len(xs)
    if k > n:
        return INF
    previous = [0] + [INF] * n
    for t in range(1, k + 1):
        current = [INF] * (n + 1)
        hull = MinHull()
        for i in range(t, n + 1):
            j = i - 1                       # insert the layer-(t-1) state that becomes usable
            if previous[j] != INF:
                hull.add_line(-2 * xs[j], previous[j] + xs[j] ** 2)
            best = hull.query(xs[i - 1])
            current[i] = xs[i - 1] ** 2 + best
        previous = current
    return previous[n]


def partition_divide_conquer(xs, k):
    n = len(xs)
    if k > n:
        return INF
    previous = [0] + [INF] * n
    for t in range(1, k + 1):
        current = [INF] * (n + 1)
        current[t] = previous[t - 1]          # i == t: the single valid split j == t-1
        if t + 1 <= n:
            solve_dc(previous, current, xs, t + 1, n, t - 1, n - 1)
        previous = current
    return previous[n]


def solve_dc(previous, current, xs, l, r, optl, optr):
    """Divide-and-conquer optimization over i in [l, r] with opt(i) in [optl, optr]."""
    if l > r:
        return
    mid = (l + r) // 2
    best, best_j = INF, -1
    upper = min(optr, mid - 1)
    for j in range(optl, upper + 1):
        if previous[j] == INF:
            continue
        cost = previous[j] + (xs[mid - 1] - xs[j]) ** 2
        if cost < best:
            best, best_j = cost, j
    current[mid] = best
    if best_j == -1:
        best_j = optl
    solve_dc(previous, current, xs, l, mid - 1, optl, best_j)
    solve_dc(previous, current, xs, mid + 1, r, best_j, optr)


def lcg_arrays(count, seed, length=50):
    arrays = []
    for _ in range(count):
        xs = [0]
        for _ in range(length - 1):
            seed = (seed * 75 + 74) % 65537
            xs.append(xs[-1] + 1 + seed % 20)
        arrays.append(xs)
    return arrays


def main():
    assert partition_bruteforce([1, 3, 5], 2) == 4
    assert partition_bruteforce([1, 3, 5], 1) == 16
    assert partition_bruteforce([1, 3, 5], 3) == 0
    assert partition_bruteforce([1, 3], 2) == 0
    assert partition_bruteforce([1, 3], 3) == INF
    assert partition_bruteforce([1], 1) == 0

    assert partition_offline_cht([1, 3, 5], 2) == 4
    assert partition_divide_conquer([1, 3, 5], 2) == 4

    # standalone hull: minimum over lines with non-increasing slopes at increasing queries
    hull = MinHull()
    for m, b in [(1, 0), (0, 0), (-1, 0)]:      # slopes must be inserted non-increasing
        hull.add_line(m, b)
    expected = {x: min(x, -x, 0) for x in range(-6, 7)}
    for x in range(-6, 7):
        assert hull.query(x) == expected[x], x

    arrays = lcg_arrays(8, seed=99, length=40)
    for xs in arrays:
        for k in range(1, 7):
            brute = partition_bruteforce(xs, k)
            assert partition_offline_cht(xs, k) == brute, (k, "cht")
            assert partition_divide_conquer(xs, k) == brute, (k, "dc")

    # the hull refuses to answer before any line exists
    try:
        MinHull().query(3)
        raise AssertionError("expected a ValueError for an empty hull")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
