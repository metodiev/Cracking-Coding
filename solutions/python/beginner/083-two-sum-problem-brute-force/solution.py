"""Task 083 - Two-sum problem (brute force). (Tier: Beginner, CTCI Ch 1)

O(n^2) double loop. Returns the indices of the first pair found (smallest i,
then smallest j > i) whose values add to target, or [] when no pair exists.
"""


def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def main():
    assert two_sum_brute([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum_brute([3, 3], 6) == [0, 1]
    assert two_sum_brute([3, 2, 4], 6) == [1, 2]
    assert two_sum_brute([-3, 4, 3, 90], 0) == [0, 2]
    assert two_sum_brute([1, 5, 3, 4], 9) == [1, 3]
    assert two_sum_brute([2, 2, 2, 2], 4) == [0, 1]
    assert two_sum_brute([1, 2, 3], 7) == []
    assert two_sum_brute([], 0) == []
    assert two_sum_brute([5], 5) == []


if __name__ == "__main__":
    main()
    print("PASS")
