
"""Task 219 - Implement meet-in-the-middle for subset problems (n ~ 40). (Tier: Intermediate, CTCI Ch 8)

Split the array in half, enumerate all 2^(n/2) subset sums of each half (a few
million at n = 40 in the worst case, but each half is 2^20 = 1M values which is
fine in compiled languages and acceptable here), then answer each query with a
binary search over the sorted second half. O(2^(n/2) log 2^(n/2)) per query.
"""

import bisect


def subset_sums(values):
    sums = [0]
    for x in values:
        sums += [s + x for s in sums]
    return sums


class MeetInTheMiddle:
    def __init__(self, nums):
        mid = len(nums) // 2
        self.left = subset_sums(nums[:mid])
        self.right = sorted(subset_sums(nums[mid:]))

    def exists(self, target):
        """True iff some subset (any size) sums exactly to target."""
        for s in self.left:
            i = bisect.bisect_left(self.right, target - s)
            if i < len(self.right) and self.right[i] == target - s:
                return True
        return False

    def closest(self, target):
        """The achievable subset sum closest to target."""
        best = None
        for s in self.left:
            need = target - s
            i = bisect.bisect_left(self.right, need)
            for j in (i - 1, i):
                if 0 <= j < len(self.right):
                    candidate = s + self.right[j]
                    if best is None or abs(candidate - target) < abs(best - target):
                        best = candidate
        return best


def brute_exists(nums, target):
    sums = subset_sums(nums)
    return target in sums


def subset_exists_bitset(nums, target):
    """Independent oracle: subset-sum via a python big-int bitset."""
    if target < 0:
        return False
    bits = 1
    for x in nums:
        bits |= bits << x
    return (bits >> target) & 1 == 1


def main():
    import random
    rng = random.Random(4)

    small = [rng.randrange(1, 100) for _ in range(16)]
    mitm_small = MeetInTheMiddle(small)
    for _ in range(200):
        target = rng.randrange(0, 600)
        assert mitm_small.exists(target) == brute_exists(small, target), target

    # n = 40: brute force would be 2^40 subsets, so verify against a bitset oracle
    big = [rng.randrange(1, 10) for _ in range(40)]
    total = sum(big)
    mitm_big = MeetInTheMiddle(big)
    assert mitm_big.exists(total) == subset_exists_bitset(big, total)
    assert mitm_big.exists(0) == subset_exists_bitset(big, 0)
    assert mitm_big.exists(total + 1) is False
    for _ in range(3):
        target = rng.randrange(0, total + 1)
        assert mitm_big.exists(target) == subset_exists_bitset(big, target), target

    # closest() gets cross-checked against every subset sum on a mid-size array
    small2 = [rng.randrange(1, 30) for _ in range(18)]
    mitm2 = MeetInTheMiddle(small2)
    all_sums = subset_sums(small2)
    for _ in range(50):
        target = rng.randrange(0, 300)
        expected = min(all_sums, key=lambda s: abs(s - target))
        assert abs(mitm2.closest(target) - target) == abs(expected - target), target


if __name__ == "__main__":
    main()
    print("PASS")
