"""Task 025 - Find duplicate element when exactly one number repeats (1..n). (Tier: Beginner, CTCI Ch 1)"""

def find_duplicate(nums):
    """Floyd cycle detection on the values-as-pointers graph. O(n), O(1) space."""
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow


def main():
    assert find_duplicate([1, 3, 4, 2, 2]) == 2
    assert find_duplicate([3, 1, 3, 4, 2]) == 3
    assert find_duplicate([1, 1]) == 1


if __name__ == "__main__":
    main()
    print("PASS")
