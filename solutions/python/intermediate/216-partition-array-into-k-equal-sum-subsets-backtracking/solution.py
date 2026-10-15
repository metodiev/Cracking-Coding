
"""Task 216 - Partition array into k equal-sum subsets (backtracking). (Tier: Intermediate, CTCI Ch 8)

Total must divide by k. Sort descending, place each value into one of k buckets,
skipping empty-duplicate buckets and overflowing ones. Classic pruning makes it fast
for the sizes used in interviews.
"""


def can_partition_k(nums, k):
    if k <= 0:
        raise ValueError("k must be positive")
    total = sum(nums)
    if total % k != 0:
        return False
    target = total // k
    nums = sorted(nums, reverse=True)
    if nums and nums[0] > target:
        return False
    buckets = [0] * k

    def place(i):
        if i == len(nums):
            return True
        seen = set()
        for b in range(k):
            if buckets[b] in seen:               # identical bucket states are redundant
                continue
            if buckets[b] + nums[i] <= target:
                seen.add(buckets[b])
                buckets[b] += nums[i]
                if place(i + 1):
                    return True
                buckets[b] -= nums[i]
            if buckets[b] == 0:                  # never start multiple empty buckets
                break
        return False

    return place(0)


def main():
    assert can_partition_k([4, 3, 2, 3, 5, 2, 1], 4)      # buckets of 5
    assert can_partition_k([1, 2, 3, 4], 2)               # 5 and 5
    assert can_partition_k([1, 1, 1, 1], 2)
    assert not can_partition_k([2, 2, 2, 2, 3], 3)        # 11 not divisible by 3
    assert not can_partition_k([2, 2, 2, 2, 2, 2], 4)     # 12/4 = 3, but values are 2s
    assert can_partition_k([10, 10, 10, 10, 10, 10], 3)


if __name__ == "__main__":
    main()
    print("PASS")
