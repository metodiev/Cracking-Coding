"""Task 293 - Coin change: number of ways (DP). (Tier: Intermediate, CTCI Ch 8)

Counts UNORDERED combinations of coins summing to the amount (order does not matter).
That is exactly what the coin-by-coin outer loop produces: each coin extends the
reachable sums using only coins processed so far, so every multiset is counted once.
Cross-checked against an exhaustive combination counter on deterministic LCG inputs.
"""


def count_ways(coins, amount):
    """Number of unordered combinations of coins (unlimited supply) summing to amount."""
    ways = [0] * (amount + 1)
    ways[0] = 1
    for coin in coins:
        for s in range(coin, amount + 1):
            ways[s] += ways[s - coin]
    return ways[amount]


def brute_ways(coins, amount):
    """Independent oracle: recursive count over coin index with descending multiplicities."""
    def count(index, remaining):
        if remaining == 0:
            return 1
        if index == len(coins):
            return 0
        total = 0
        take = 0
        while take * coins[index] <= remaining:
            total += count(index + 1, remaining - take * coins[index])
            take += 1
        return total

    return count(0, amount)


def lcg_coins(n, seed):
    values = []
    while len(values) < n:
        seed = (seed * 75 + 74) % 65537
        value = 1 + seed % 8
        if value not in values:
            values.append(value)
    return sorted(values), seed


def main():
    assert count_ways([1, 2, 5], 5) == 4  # {5}, {2,2,1}, {2,1,1,1}, {1,1,1,1,1}
    assert count_ways([2], 3) == 0
    assert count_ways([1, 2, 5], 0) == 1
    assert count_ways([1], 10) == 1
    assert count_ways([], 0) == 1
    assert count_ways([2, 3, 5], 9) == 3  # {5,2,2}, {3,3,3}, {3,2,2,2}
    # order must NOT matter: 3 = 1+1+1 and 1+2 only (ordered counting would answer 3)
    assert count_ways([1, 2], 3) == 2

    seed = 24680
    for _ in range(150):
        seed = (seed * 75 + 74) % 65537
        count = 1 + seed % 5
        coins, seed = lcg_coins(count, seed)
        seed = (seed * 75 + 74) % 65537
        amount = seed % 60
        assert count_ways(coins, amount) == brute_ways(coins, amount), (coins, amount)


if __name__ == "__main__":
    main()
    print("PASS")
