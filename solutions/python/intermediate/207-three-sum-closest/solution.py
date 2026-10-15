
"""Task 207 - Three-sum closest. (Tier: Intermediate, CTCI Ch 16)

Sort + fix one anchor + two pointers; track the sum closest to target. O(n^2).
"""


def three_sum_closest(nums, target):
    nums = sorted(nums)
    if len(nums) < 3:
        raise ValueError("need at least three values")
    best = nums[0] + nums[1] + nums[2]
    for i in range(len(nums) - 2):
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            if abs(total - target) < abs(best - target):
                best = total
            if total < target:
                lo += 1
            elif total > target:
                hi -= 1
            else:
                return total               # exact match cannot be beaten
    return best


def main():
    assert three_sum_closest([-1, 2, 1, -4], 1) == 2
    assert three_sum_closest([0, 0, 0], 1) == 0
    assert three_sum_closest([1, 1, 1, 1], 5) == 3
    assert three_sum_closest([-5, -2, -1], -3) == -8       # -5 + -2 + -1
    try:
        three_sum_closest([1, 2], 3)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
