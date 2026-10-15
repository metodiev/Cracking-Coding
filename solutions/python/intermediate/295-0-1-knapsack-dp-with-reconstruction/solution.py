"""Task 295 - 0/1 knapsack (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8)

O(n * capacity) DP over (item, capacity) prefixes; dp[i][c] is the best value using
items[:i]. Reconstruction walks back from (n, capacity) and takes item i-1 whenever the
value strictly improves over dp[i-1][c], which happens exactly when that item belongs to
an optimal solution of the subproblem. Cross-checked against exhaustive subsets on
deterministic LCG instances.
"""


def knapsack(weights, values, capacity):
    """Returns (best value, indices of one optimal item set) for the 0-1 knapsack."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        weight = weights[i - 1]
        value = values[i - 1]
        for c in range(capacity + 1):
            best = dp[i - 1][c]
            if c >= weight and dp[i - 1][c - weight] + value > best:
                best = dp[i - 1][c - weight] + value
            dp[i][c] = best
    chosen = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    chosen.reverse()
    return dp[n][capacity], chosen


def brute_knapsack(weights, values, capacity):
    n = len(weights)
    best = 0
    for mask in range(1 << n):
        weight = 0
        value = 0
        for i in range(n):
            if (mask >> i) & 1:
                weight += weights[i]
                value += values[i]
        if weight <= capacity and value > best:
            best = value
    return best


def lcg_instance(n, seed):
    weights = []
    values = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        weights.append(1 + seed % 12)
        seed = (seed * 75 + 74) % 65537
        values.append(1 + seed % 12)
    return weights, values, seed


def main():
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    value, chosen = knapsack(weights, values, 7)
    assert value == 9 and sorted(weights[i] for i in chosen) == [3, 4], (value, chosen)
    assert len(set(chosen)) == len(chosen)
    assert sum(weights[i] for i in chosen) <= 7
    assert sum(values[i] for i in chosen) == 9

    assert knapsack(weights, values, 0) == (0, [])
    assert knapsack([2, 9], [3, 100], 5) == (3, [0])
    assert knapsack([4], [9], 4) == (9, [0])

    value, chosen = knapsack(weights, values, 100)
    assert value == 17 and sorted(chosen) == [0, 1, 2, 3]  # everything fits

    seed = 555
    for _ in range(120):
        seed = (seed * 75 + 74) % 65537
        n = 1 + seed % 14
        weights, values, seed = lcg_instance(n, seed)
        seed = (seed * 75 + 74) % 65537
        capacity = seed % 40
        value, chosen = knapsack(weights, values, capacity)
        assert value == brute_knapsack(weights, values, capacity), (weights, values, capacity)
        assert len(set(chosen)) == len(chosen)
        assert sum(weights[i] for i in chosen) <= capacity
        assert sum(values[i] for i in chosen) == value


if __name__ == "__main__":
    main()
    print("PASS")
