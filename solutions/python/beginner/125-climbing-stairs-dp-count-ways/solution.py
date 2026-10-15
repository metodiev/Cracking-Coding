
"""Task 125 - Climbing stairs DP (count ways). (Tier: Beginner, CTCI Ch 8)

ways(n) = ways(n-1) + ways(n-2) with ways(0) = 1 (empty climb), ways(1) = 1.
O(n) time, O(1) space with two rolling variables (Fibonacci in disguise).
"""


def climb_ways(n):
    if n < 0:
        raise ValueError("n must be >= 0")
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def climb_ways_general(n, steps):
    """Variant: allowed step sizes from `steps` (e.g. [1, 2, 3])."""
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        dp[i] = sum(dp[i - s] for s in steps if s <= i)
    return dp[n]


def main():
    assert [climb_ways(n) for n in range(6)] == [1, 1, 2, 3, 5, 8]
    assert climb_ways(35) == 14930352
    assert climb_ways_general(4, [1, 2, 3]) == 7          # 1111,112,121,13,211,22,31
    assert climb_ways_general(0, [1, 2]) == 1


if __name__ == "__main__":
    main()
    print("PASS")
