"""Task 016 - Count occurrences of each character in a string.  (Tier: Beginner, CTCI Ch 1)"""

from collections import Counter


def char_counts(s):
    return dict(Counter(s))


def main():
    assert char_counts("aabbbc") == {"a": 2, "b": 3, "c": 1}
    assert char_counts("") == {}
    assert char_counts("zzz") == {"z": 3}
    print("PASS")


if __name__ == "__main__":
    main()
