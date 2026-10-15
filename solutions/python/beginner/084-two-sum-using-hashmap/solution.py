"""Task 084 - Two-sum using hashmap. (Tier: Beginner, CTCI Ch 1)

One pass, O(n): keep a map value -> earliest index seen so far and, for each new
index j, look up target - nums[j]. A single pass suffices because when index j is
reached, every earlier index has already been stored, so any pair (i < j) is
detected exactly when its later element arrives; each element is visited once.
Returns [i, j] with i < j, or [] when no pair exists.
"""


def two_sum_hash(nums, target):
    seen = {}
    for j, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], j]
        if value not in seen:
            seen[value] = j
    return []


def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def main():
    assert two_sum_hash([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum_hash([3, 2, 4], 6) == [1, 2]
    assert two_sum_hash([3, 3], 6) == [0, 1]
    assert two_sum_hash([-3, 4, 3, 90], 0) == [0, 2]
    assert two_sum_hash([1, 2, 3], 7) == []
    assert two_sum_hash([], 0) == []
    assert two_sum_hash([5], 5) == []

    # agreement with the brute force answer on a battery of cases
    cases = [
        ([2, 7, 11, 15], 9), ([3, 2, 4], 6), ([3, 3], 6), ([0, 0, 1], 0),
        ([-3, 4, 3, 90], 0), ([1, 2, 3], 7), ([], 0), ([5], 5), ([1, 5, 3, 4], 9),
    ]
    for nums, target in cases:
        got = two_sum_hash(nums, target)
        assert (got != []) == (two_sum_brute(nums, target) != []), (nums, target)
        if got:
            i, j = got
            assert 0 <= i < j < len(nums), (nums, target)
            assert nums[i] + nums[j] == target, (nums, target)


if __name__ == "__main__":
    main()
    print("PASS")
