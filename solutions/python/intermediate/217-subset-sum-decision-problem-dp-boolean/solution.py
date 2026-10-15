
"""Task 217 - Subset sum decision problem (DP boolean). (Tier: Intermediate, CTCI Ch 8)

dp[s] = True if some subset of the items processed so far sums to s.
Items are used at most once (iterate sums descending). O(n * target) time.
"""


def subset_sum(nums, target):
    if target < 0:
        return False
    dp = [False] * (target + 1)
    dp[0] = True
    for x in nums:
        for s in range(target, x - 1, -1):
            if dp[s - x]:
                dp[s] = True
    return dp[target]


def subset_sum_recursive(nums, target):
    """Exponential memoized variant, cross-checked against the DP on small inputs."""
    n = len(nums)
    memo = {}

    def go(i, remaining):
        if remaining == 0:
            return True
        if i == n or remaining < 0:
            return False
        key = (i, remaining)
        if key in memo:
            return memo[key]
        memo[key] = go(i + 1, remaining) or go(i + 1, remaining - nums[i])
        return memo[key]

    return go(0, target)


def main():
    assert subset_sum([3, 34, 4, 12, 5, 2], 9)           # 4 + 5 or 4 + 3 + 2
    assert not subset_sum([3, 34, 4, 12, 5, 2], 30)
    assert subset_sum([], 0)
    assert not subset_sum([], 5)
    assert subset_sum([5], 5)
    assert not subset_sum([5], 3)

    import random
    rng = random.Random(2)
    for _ in range(200):
        nums = [rng.randrange(1, 15) for _ in range(rng.randrange(0, 9))]
        target = rng.randrange(0, 40)
        assert subset_sum(nums, target) == subset_sum_recursive(nums, target), (nums, target)


if __name__ == "__main__":
    main()
    print("PASS")
