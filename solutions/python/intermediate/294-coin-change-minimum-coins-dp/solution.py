"""Task 294 - Coin change: minimum coins (DP). (Tier: Intermediate, CTCI Ch 8)

Fewest coins (unlimited supply) summing to `amount`; -1 is the documented sentinel for
"impossible" -- a count can never be negative, so it cannot collide with a real answer.
Greedy fails in general: [1, 3, 4] amount 6 takes 4 + 1 + 1 (3 coins) while 3 + 3 needs
only 2, so this uses the O(amount * len(coins)) bottom-up DP, cross-checked against a
BFS-over-amounts oracle on deterministic LCG inputs.
"""

from collections import deque


def min_coins(coins, amount):
    """Fewest coins summing to amount, or -1 when impossible (documented sentinel)."""
    infinity = float("inf")
    dp = [infinity] * (amount + 1)
    dp[0] = 0
    for s in range(1, amount + 1):
        for coin in coins:
            if coin <= s and dp[s - coin] + 1 < dp[s]:
                dp[s] = dp[s - coin] + 1
    return -1 if dp[amount] == infinity else dp[amount]


def bfs_min_coins(coins, amount):
    """Independent oracle: BFS over reachable amounts gives fewest-coin distances."""
    dist = [-1] * (amount + 1)
    dist[0] = 0
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for coin in coins:
            v = u + coin
            if v <= amount and dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist[amount]


def lcg_coins(n, seed):
    values = []
    while len(values) < n:
        seed = (seed * 75 + 74) % 65537
        value = 1 + seed % 7
        if value not in values:
            values.append(value)
    return sorted(values), seed


def main():
    assert min_coins([1, 2, 5], 11) == 3  # 5 + 5 + 1
    assert min_coins([2], 3) == -1  # sentinel documented above
    assert min_coins([1, 2, 5], 0) == 0
    assert min_coins([1, 3, 4], 6) == 2  # 3 + 3 beats greedy 4 + 1 + 1
    assert min_coins([1, 3, 4], 9) == 3  # 4 + 4 + 1 (or 3 + 3 + 3)
    assert min_coins([5, 7], 11) == -1
    assert min_coins([], 0) == 0

    seed = 13579
    for _ in range(150):
        seed = (seed * 75 + 74) % 65537
        count = 1 + seed % 4
        coins, seed = lcg_coins(count, seed)
        seed = (seed * 75 + 74) % 65537
        amount = seed % 80
        assert min_coins(coins, amount) == bfs_min_coins(coins, amount), (coins, amount)


if __name__ == "__main__":
    main()
    print("PASS")
