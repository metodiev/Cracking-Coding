"""Task 292 - Minimum subset sum difference (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8)

Split the multiset into two parts minimizing |sum(A) - sum(B)|. Reachability of sums up
to floor(total/2) is the classic subset-sum DP; the best reachable sum s gives the
difference total - 2s, and the same table rebuilds the subset achieving it (taking an
item whenever the remainder stays reachable without it). Cross-checked against
exhaustive subset enumeration on deterministic LCG inputs.
"""


def min_difference(nums):
    """Returns (minimum possible |sum(A) - sum(B)|, one side achieving it)."""
    total = sum(nums)
    half = total // 2
    n = len(nums)
    reachable = [[False] * (half + 1) for _ in range(n + 1)]
    reachable[0][0] = True
    for i in range(1, n + 1):
        value = nums[i - 1]
        for s in range(half + 1):
            reachable[i][s] = reachable[i - 1][s] or (s >= value and reachable[i - 1][s - value])
    best = half
    while best > 0 and not reachable[n][best]:
        best -= 1
    side = []
    s = best
    for i in range(n, 0, -1):
        value = nums[i - 1]
        if s >= value and reachable[i - 1][s - value]:
            side.append(value)
            s -= value
    side.reverse()
    return total - 2 * best, side


def brute_min_difference(nums):
    total = sum(nums)
    n = len(nums)
    best = total
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if (mask >> i) & 1:
                s += nums[i]
        best = min(best, abs(total - 2 * s))
    return best


def lcg_numbers(n, seed):
    nums = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        nums.append(1 + seed % 25)
    return nums, seed


def main():
    diff, side = min_difference([1, 6, 11, 5])
    assert diff == 1 and sum(side) == 11 and sorted(side) == [5, 6], (diff, side)

    diff, side = min_difference([3, 3, 3])
    assert diff == 3 and side == [3], (diff, side)

    diff, side = min_difference([1, 2, 3, 4])
    assert diff == 0 and sorted(side) == [1, 4], (diff, side)

    diff, side = min_difference([7])
    assert diff == 7 and side == [], (diff, side)

    diff, side = min_difference([])
    assert diff == 0 and side == [], (diff, side)

    seed = 777
    for _ in range(200):
        seed = (seed * 75 + 74) % 65537
        count = 1 + seed % 14
        nums, seed = lcg_numbers(count, seed)
        diff, side = min_difference(nums)
        total = sum(nums)
        assert diff == brute_min_difference(nums), (nums, diff)
        assert total - 2 * sum(side) == diff, (nums, diff, side)


if __name__ == "__main__":
    main()
    print("PASS")
