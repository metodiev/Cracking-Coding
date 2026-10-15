"""Task 086 - Count inversion pairs using naive O(n^2). (Tier: Beginner, CTCI Ch 10)

An inversion is a pair i < j with a[i] > a[j]; equal values are not inversions.
"""


def count_inversions_naive(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] > nums[j]:
                count += 1
    return count


def main():
    assert count_inversions_naive([]) == 0
    assert count_inversions_naive([1]) == 0
    assert count_inversions_naive([1, 2, 3]) == 0
    assert count_inversions_naive([1, 1, 1]) == 0
    assert count_inversions_naive([2, 2]) == 0
    assert count_inversions_naive([1, 3, 2]) == 1
    assert count_inversions_naive([3, 2, 1]) == 3
    assert count_inversions_naive([2, 4, 1, 3, 5]) == 3
    assert count_inversions_naive([-1, -2, -3]) == 3
    assert count_inversions_naive([5, 4, 3, 2, 1]) == 10


if __name__ == "__main__":
    main()
    print("PASS")
