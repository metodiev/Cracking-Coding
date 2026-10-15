
"""Task 203 - 4-sum problem (k-sum generalization). (Tier: Intermediate, CTCI Ch 16)

Recursive k-sum over a sorted array with two-pointer base case; duplicate
skipping keeps the result unique. O(n^(k-1)) worst case.
"""


def k_sum(nums, target, k):
    nums = sorted(nums)
    results = []

    def search(start, remaining, k_left, current):
        if k_left == 2:
            lo, hi = start, len(nums) - 1
            while lo < hi:
                total = nums[lo] + nums[hi]
                if total == remaining:
                    results.append(current + [nums[lo], nums[hi]])
                    while lo < hi and nums[lo] == nums[lo + 1]:
                        lo += 1
                    while lo < hi and nums[hi] == nums[hi - 1]:
                        hi -= 1
                    lo += 1
                    hi -= 1
                elif total < remaining:
                    lo += 1
                else:
                    hi -= 1
            return
        for i in range(start, len(nums) - k_left + 1):
            if i > start and nums[i] == nums[i - 1]:
                continue
            if nums[i] * k_left > remaining:
                break
            search(i + 1, remaining - nums[i], k_left - 1, current + [nums[i]])

    search(0, target, k, [])
    return results


def four_sum(nums, target):
    return k_sum(nums, target, 4)


def main():
    assert four_sum([1, 0, -1, 0, -2, 2], 0) == [[-2, -1, 1, 2], [-2, 0, 0, 2], [-1, 0, 0, 1]]
    assert four_sum([2, 2, 2, 2, 2], 8) == [[2, 2, 2, 2]]
    assert four_sum([1, 2, 3, 4], 100) == []
    assert four_sum([], 0) == []
    # generalization still works for k = 2 and k = 3
    assert k_sum([1, 2, 3, 4], 5, 2) == [[1, 4], [2, 3]]
    assert k_sum([-1, 0, 1, 2, -1, -4], 0, 3) == [[-1, -1, 2], [-1, 0, 1]]


if __name__ == "__main__":
    main()
    print("PASS")
