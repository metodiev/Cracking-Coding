"""Task 047 - Find first non-repeating character in a string. (Tier: Beginner, CTCI Ch 1)"""

from collections import Counter


def first_non_repeating(s):
    counts = Counter(s)
    for ch in s:
        if counts[ch] == 1:
            return ch
    return None


def main():
    assert first_non_repeating("aabbcde") == "c"
    assert first_non_repeating("aabb") is None
    assert first_non_repeating("x") == "x"


if __name__ == "__main__":
    main()
    print("PASS")
