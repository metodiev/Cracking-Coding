
"""Task 167 - Find all elements appearing > n/3 (generalized). (Tier: Beginner, CTCI Ch 6)

Generalized Boyer-Moore with TWO candidates: at most two values can exceed n/3.
Phase 1 finds candidates, phase 2 verifies with exact counts.
"""


def majority_elements_n3(nums):
    """Returns a sorted list of values occurring more than n/3 times."""
    c1 = c2 = None
    n1 = n2 = 0
    for x in nums:
        if c1 is not None and x == c1:
            n1 += 1
        elif c2 is not None and x == c2:
            n2 += 1
        elif n1 == 0:
            c1, n1 = x, 1
        elif n2 == 0:
            c2, n2 = x, 1
        else:
            n1 -= 1
            n2 -= 1
    threshold = len(nums) / 3
    return sorted(x for x in {c1, c2} if x is not None and nums.count(x) > threshold)


def main():
    assert majority_elements_n3([3, 2, 3]) == [3]                    # 2/3 > 1/3
    assert majority_elements_n3([1, 1, 1, 3, 3, 2, 2, 2]) == [1, 2]
    assert majority_elements_n3([1, 2, 3]) == []
    assert majority_elements_n3([]) == []
    assert majority_elements_n3([4]) == [4]
    assert majority_elements_n3([1, 1, 2, 2, 3]) == [1, 2]           # 2/5 = 0.4 > 1/3
    assert majority_elements_n3([1, 1, 2, 2]) == [1, 2]              # 2/4 = 0.5 > 1/3


if __name__ == "__main__":
    main()
    print("PASS")
