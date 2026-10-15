
"""Task 169 - Sliding window for unique characters (longest unique substring). (Tier: Beginner, CTCI Ch 1)

Two pointers with a last-seen index map: O(n) time, O(min(n, alphabet)) space.
"""


def longest_unique_substring(s):
    """Returns (length, substring)."""
    last = {}
    best_len = best_start = 0
    lo = 0
    for hi, ch in enumerate(s):
        if ch in last and last[ch] >= lo:
            lo = last[ch] + 1
        last[ch] = hi
        if hi - lo + 1 > best_len:
            best_len = hi - lo + 1
            best_start = lo
    return best_len, s[best_start:best_start + best_len]


def main():
    assert longest_unique_substring("abcabcbb") == (3, "abc")
    assert longest_unique_substring("bbbbb") == (1, "b")
    assert longest_unique_substring("pwwkew") == (3, "wke")
    assert longest_unique_substring("") == (0, "")
    assert longest_unique_substring("x") == (1, "x")
    assert longest_unique_substring("dvdf") == (3, "vdf")
    assert longest_unique_substring("abba") == (2, "ab")


if __name__ == "__main__":
    main()
    print("PASS")
