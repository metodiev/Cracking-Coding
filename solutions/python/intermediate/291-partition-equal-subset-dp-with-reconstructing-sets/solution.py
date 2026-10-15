"""Task 291 - Partition equal subset (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8)

O(n * sum) subset-sum DP that also rebuilds a witness: reachable[i][s] says whether sum s
is achievable with nums[:i]. Walking back from (n, target) takes an item whenever the
remainder is still reachable without it, so [1, 5, 11, 5] yields the [1, 5, 5] half.
Cross-checked against exhaustive subset enumeration on deterministic LCG inputs.
"""


def partition(nums):
    """Returns (True, one side whose sum is total/2) or (False, None) if no split exists."""
    total = sum(nums)
    if total % 2 != 0:
        return False, None
    target = total // 2
    n = len(nums)
    reachable = [[False] * (target + 1) for _ in range(n + 1)]
    reachable[0][0] = True
    for i in range(1, n + 1):
        value = nums[i - 1]
        for s in range(target + 1):
            reachable[i][s] = reachable[i - 1][s] or (s >= value and reachable[i - 1][s - value])
    if not reachable[n][target]:
        return False, None
    side = []
    s = target
    for i in range(n, 0, -1):
        value = nums[i - 1]
        if s >= value and reachable[i - 1][s - value]:
            side.append(value)
            s -= value
    side.reverse()
    return True, side


def validate(nums, side):
    """True when side is a sub-multiset of nums whose complement has the same sum."""
    remaining = list(nums)
    for value in side:
        if value not in remaining:
            return False
        remaining.remove(value)
    return sum(side) == sum(remaining)


def brute_partition_exists(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    n = len(nums)
    for mask in range(1 << n):
        if sum(nums[i] for i in range(n) if (mask >> i) & 1) == target:
            return True
    return False


def lcg_numbers(n, seed):
    nums = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        nums.append(1 + seed % 15)
    return nums, seed


def main():
    ok, side = partition([1, 5, 11, 5])
    assert ok and side == [1, 5, 5], side
    assert validate([1, 5, 11, 5], side)

    assert partition([1, 2, 3, 5]) == (False, None)  # total 11 is odd
    assert partition([1, 2]) == (False, None)  # odd total
    assert partition([]) == (True, [])
    assert partition([2, 2, 2]) == (False, None)  # total 6 is even, but no subset sums to 3

    ok, side = partition([1, 2, 3, 4])
    assert ok and sum(side) == 5 and validate([1, 2, 3, 4], side)

    seed = 1234
    for _ in range(200):
        seed = (seed * 75 + 74) % 65537
        count = 2 + seed % 8
        nums, seed = lcg_numbers(count, seed)
        ok, side = partition(nums)
        assert ok == brute_partition_exists(nums), (nums, ok)
        if ok:
            assert validate(nums, side), (nums, side)


if __name__ == "__main__":
    main()
    print("PASS")
