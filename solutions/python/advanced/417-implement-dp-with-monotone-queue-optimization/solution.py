"""Task 417 - Implement DP with monotone queue optimization. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Sliding-window minimum DP: jump from j to i at cost a[i], landing windows of size k:
    dp[0] = a[0],   dp[i] = a[i] + min(dp[j] for j in [max(0, i - k), i - 1]).
The naive transition scans k predecessors (O(n*k) overall). A monotone deque keeps the
candidate indices with STRICTLY increasing dp values: expired indices (j < i - k) leave
from the front, and while adding i every index with dp >= dp[i] is popped from the back,
because it is both older and worse. Every step is amortized O(1).
Tie-break: newer index wins (the pop condition keeps equal-valued indices newest-first),
which the tests only use through value equality.
"""

from collections import deque


def sliding_min_dp(a, k):
    """Returns (dp, pred); pred[i] is the j achieving the window minimum."""
    n = len(a)
    if n == 0:
        return [], []
    dp = [0] * n
    pred = [-1] * n
    dp[0] = a[0]
    dq = deque([0])
    for i in range(1, n):
        while dq[0] < i - k:
            dq.popleft()
        j = dq[0]
        dp[i] = a[i] + dp[j]
        pred[i] = j
        while dq and dp[dq[-1]] >= dp[i]:
            dq.pop()
        dq.append(i)
    return dp, pred


def sliding_min_dp_naive(a, k):
    """Oracle: O(n*k) direct evaluation of the recurrence."""
    n = len(a)
    if n == 0:
        return []
    dp = [a[0]]
    for i in range(1, n):
        best = min(dp[max(0, i - k):i])
        dp.append(a[i] + best)
    return dp


def lcg_array(length, seed, lo, hi):
    out = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        out.append(lo + seed % (hi - lo + 1))
    return out, seed


def main():
    # hand-derived values, a = [10, 1, 5, 7, 3, 2]
    a = [10, 1, 5, 7, 3, 2]
    dp2, pred2 = sliding_min_dp(a, 2)
    assert dp2 == [10, 11, 15, 18, 18, 20], dp2
    # at i = 4 the equal-valued index 3 is dropped (newer ties win), so pred[5] = 4
    assert pred2 == [-1, 0, 0, 1, 2, 4], pred2
    assert dp2[4] == a[4] + min(dp2[2:4]) and dp2[5] == a[5] + min(dp2[3:5])
    dp3, pred3 = sliding_min_dp(a, 3)
    assert dp3 == [10, 11, 15, 17, 14, 16], dp3
    dp1, pred1 = sliding_min_dp(a, 1)
    assert dp1 == [10, 11, 16, 23, 26, 28], dp1
    assert pred1 == [-1, 0, 1, 2, 3, 4], pred1
    # k = 1 makes dp a running prefix sum, k >= n opens the whole history
    assert dp1 == [sum(a[: i + 1]) for i in range(len(a))]
    dp_big, _ = sliding_min_dp(a, 100)
    assert dp_big[0] == 10
    assert dp_big[1] == 1 + 10
    assert dp_big[2] == 5 + 10
    assert dp_big[3] == 7 + 10
    assert dp_big[4] == 3 + 10
    assert dp_big[5] == 2 + 10

    # every hand value also matches the naive oracle and the predecessor optimality
    for k in range(1, 8):
        dq_dp, dq_pred = sliding_min_dp(a, k)
        assert dq_dp == sliding_min_dp_naive(a, k), k
        for i in range(1, len(a)):
            assert max(0, i - k) <= dq_pred[i] < i
            assert dq_dp[dq_pred[i]] == min(dq_dp[max(0, i - k):i]), (k, i)

    # degenerate sizes
    assert sliding_min_dp([], 5) == ([], [])
    assert sliding_min_dp([7], 3) == ([7], [-1])
    assert sliding_min_dp_naive([], 5) == []
    assert sliding_min_dp([-4, -8], 2)[0] == [-4, -12]

    seed = 417
    for round_index in range(8):
        n = 120 + seed % 180
        values, seed = lcg_array(n, seed, -50, 50)
        k = 1 + seed % 15
        dp, pred = sliding_min_dp(values, k)
        want = sliding_min_dp_naive(values, k)
        assert dp == want, (round_index, k)
        for i in range(1, n):
            j = pred[i]
            assert max(0, i - k) <= j < i
            assert dp[i] == values[i] + dp[j]
            assert dp[j] == min(dp[max(0, i - k):i]), (round_index, i)
        # monotone deque invariant: candidate values strictly increase from front to back
        dq = deque([0])
        for i in range(1, n):
            while dq[0] < i - k:
                dq.popleft()
            assert dp[dq[0]] == min(dp[max(0, i - k):i]), (round_index, i)
            while dq and dp[dq[-1]] >= dp[i]:
                dq.pop()
            dq.append(i)

    # window bound: widening k never increases any dp value (only relaxes the minimum)
    values, _ = lcg_array(150, 99, 0, 30)
    previous = sliding_min_dp_naive(values, 1)
    for k in range(2, 11):
        current = sliding_min_dp_naive(values, k)
        for i in range(len(values)):
            assert current[i] <= previous[i], (k, i)
        previous = current


if __name__ == "__main__":
    main()
    print("PASS")
