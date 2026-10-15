
"""Task 202 - 3-sum all unique triplets. (Tier: Intermediate, CTCI Ch 16)

Sort + two pointers, skipping duplicate values at every level. O(n^2) time.
"""


def three_sum(nums):
    nums = sorted(nums)
    out = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue                                # skip duplicate anchors
        if nums[i] > 0:
            break                                   # sorted: no negative anchor left
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            total = nums[i] + nums[lo] + nums[hi]
            if total == 0:
                out.append([nums[i], nums[lo], nums[hi]])
                while lo < hi and nums[lo] == nums[lo + 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi - 1]:
                    hi -= 1
                lo += 1
                hi -= 1
            elif total < 0:
                lo += 1
            else:
                hi -= 1
    return out


def main():
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
    assert three_sum([0, 1, 1]) == []
    assert three_sum([0, 0, 0]) == [[0, 0, 0]]
    assert three_sum([-2, 0, 0, 2, 2]) == [[-2, 0, 2]]
    assert three_sum([]) == []
    assert three_sum([1, 2, 3]) == []


if __name__ == "__main__":
    main()
    print("PASS")
