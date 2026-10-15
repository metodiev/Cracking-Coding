"""Task 046 - Implement substring search (naive). (Tier: Beginner, CTCI Ch 10)"""

def substring_search(text, pattern):
    """Return the first index of pattern in text, or -1. O(n*m) worst case."""
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            return i
    return -1


def main():
    assert substring_search("hello world", "world") == 6
    assert substring_search("aaaa", "aa") == 0
    assert substring_search("abc", "xyz") == -1
    assert substring_search("abc", "") == 0


if __name__ == "__main__":
    main()
    print("PASS")
