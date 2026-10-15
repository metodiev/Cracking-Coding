
"""Task 214 - Subarray with given product (positive integers). (Tier: Intermediate, CTCI Ch 1)

Sliding window over strictly positive values: multiply on the right, divide from the
left while the product exceeds the target. O(n) time, O(1) space.
Returns (start, end) inclusive of the first match found, or None.
"""


def subarray_with_product(nums, target):
    if any(x <= 0 for x in nums):
        raise ValueError("requires positive values")
    if target < 1:
        return None
    left = 0
    product = 1
    for right, x in enumerate(nums):
        product *= x
        while left <= right and product > target:
            product //= nums[left]
            left += 1
        if left <= right and product == target:
            return (left, right)
    return None


def main():
    assert subarray_with_product([2, 3, 1, 2, 4], 6) == (0, 1)      # [2,3]
    assert subarray_with_product([1, 2, 3, 4], 6) == (0, 2)         # [1,2,3] is found first
    assert subarray_with_product([10], 10) == (0, 0)
    assert subarray_with_product([3, 1, 1], 2) is None
    assert subarray_with_product([2, 4], 6) is None
    assert subarray_with_product([1, 1, 1], 1) == (0, 0)
    assert subarray_with_product([], 5) is None


if __name__ == "__main__":
    main()
    print("PASS")
