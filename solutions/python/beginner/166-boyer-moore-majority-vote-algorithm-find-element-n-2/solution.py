
"""Task 166 - Boyer-Moore majority vote algorithm (element > n/2). (Tier: Beginner, CTCI Ch 6)

Phase 1 cancels pairs of different values; the survivor is the only possible
majority. Phase 2 verifies it (the algorithm alone only finds a *candidate*).
"""


def majority_element(nums):
    """Returns the element occurring more than n/2 times, or None."""
    candidate, count = None, 0
    for x in nums:
        if count == 0:
            candidate, count = x, 1
        elif x == candidate:
            count += 1
        else:
            count -= 1
    if candidate is None:
        return None
    return candidate if nums.count(candidate) > len(nums) // 2 else None


def main():
    assert majority_element([3, 2, 3]) == 3
    assert majority_element([2, 2, 1, 1, 1, 2, 2]) == 2
    assert majority_element([1]) == 1
    assert majority_element([]) is None
    assert majority_element([1, 2, 3]) is None            # no majority (candidate 3 fails verification)
    assert majority_element([1, 1, 2, 2]) is None         # exactly half is NOT > n/2
    assert majority_element([5, 5, 5, 5, 1, 2]) == 5


if __name__ == "__main__":
    main()
    print("PASS")
