
"""Task 220 - Implement bitmask DP for TSP small N. (Tier: Intermediate, CTCI Ch 8)

Held-Karp over subsets: dp[mask][i] = cheapest path starting at 0, visiting exactly
the cities in mask, ending at i. Answer = min over i of dp[full][i] + cost(i, 0).
O(2^n * n^2) time, O(2^n * n) space — fine for n <= ~12.
"""


def tsp(dist):
    n = len(dist)
    if n <= 1:
        return 0
    INF = float("inf")
    full = (1 << n) - 1
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0                                # start at city 0, mask = {0}
    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == INF:
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                candidate = dp[mask][last] + dist[last][nxt]
                if candidate < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = candidate
    best = INF
    for last in range(n):
        if dp[full][last] != INF:
            best = min(best, dp[full][last] + dist[last][0])
    return best


def main():
    square = [
        [0, 1, 10, 10],
        [1, 0, 1, 10],
        [10, 1, 0, 1],
        [10, 10, 1, 0],
    ]
    assert tsp(square) == 4                     # 0-1-2-3-0

    triangle = [
        [0, 1, 3],
        [1, 0, 2],
        [3, 2, 0],
    ]
    assert tsp(triangle) == 6                   # any cycle costs 1+2+3

    assert tsp([[0]]) == 0
    assert tsp([[0, 7], [7, 0]]) == 14
    # brute force cross-check on a random 7-city instance
    import itertools
    import random
    rng = random.Random(6)
    n = 7
    d = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = rng.randrange(1, 20)
    best = min(
        sum(d[p[i]][p[i + 1]] for i in range(n - 1)) + d[p[-1]][p[0]]
        for p in itertools.permutations(range(1, n))
    )
    assert tsp(d) == best, (tsp(d), best)


if __name__ == "__main__":
    main()
    print("PASS")
