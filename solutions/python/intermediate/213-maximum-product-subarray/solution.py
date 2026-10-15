
"""Task 213 - Maximum product subarray. (Tier: Intermediate, CTCI Ch 8)

Track the running max AND min product (a negative minimum flips to a maximum when
multiplied by a negative value). O(n) time, O(1) space.
"""


def max_product(nums):
    if not nums:
        raise ValueError("empty array has no subarray")
    best = cur_max = cur_min = nums[0]
    for x in nums[1:]:
        candidates = (x, cur_max * x, cur_min * x)
        cur_max = max(candidates)
        cur_min = min(candidates)
        best = max(best, cur_max)
    return best


def main():
    assert max_product([2, 3, -2, 4]) == 6            # [2,3]
    assert max_product([-2, 0, -1]) == 0
    assert max_product([-2, 3, -4]) == 24             # whole array
    assert max_product([0, 2]) == 2
    assert max_product([-2]) == -2
    assert max_product([2, -5, -2, -4, 3]) == 24      # [-5,-2,-4,3]? verify: -5*-2=10, *-4=-40, *3=-120... actually best is 24


if __name__ == "__main__":
    main()
    print("PASS")
