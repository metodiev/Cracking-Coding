"""Task 015 - Check if two strings are anagrams.  (Tier: Beginner, CTCI Ch 1)"""

from collections import Counter


def are_anagrams(a, b):
    return Counter(a) == Counter(b)


def main():
    assert are_anagrams("listen", "silent")
    assert are_anagrams("", "")
    assert not are_anagrams("hello", "world")
    assert not are_anagrams("abc", "ab")
    print("PASS")


if __name__ == "__main__":
    main()
