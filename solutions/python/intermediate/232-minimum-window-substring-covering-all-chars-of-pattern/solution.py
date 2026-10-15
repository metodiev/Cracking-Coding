"""Task 232 - Minimum window substring covering all chars of pattern. (Tier: Intermediate, CTCI Ch 1)

Sliding window with a need/have counter: `need[ch]` counts how many copies of ch the
window still has to supply and `missing` is their total. Grow the window with `right`;
once missing == 0, shrink from the left while the window stays valid, recording the
smallest window seen. O(|s| + |t|) time, O(k) space for an alphabet of size k.

Conventions: pattern characters must appear in the window at least as often as in t
(with multiplicity). Empty pattern, pattern longer than s, or no valid window -> "".
Tie behaviour: among equal-length minimal windows the LEFTMOST one is returned.
"""

from collections import Counter


def min_window(s, t):
    if not s or not t or len(t) > len(s):
        return ""
    need = Counter(t)
    missing = len(t)
    left = 0
    best_start = 0
    best_len = float("inf")
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        while missing == 0:
            if right - left + 1 < best_len:       # strict: ties keep the leftmost window
                best_start, best_len = left, right - left + 1
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
    return "" if best_len == float("inf") else s[best_start:best_start + best_len]


def brute_min_window(s, t):
    """O(n^2 * k) reference used only for cross-checks; same leftmost tie-break."""
    if not s or not t or len(t) > len(s):
        return ""
    want = Counter(t)
    best = ""
    for left in range(len(s)):
        for right in range(left + len(t), len(s) + 1):
            have = Counter(s[left:right])
            if all(have[ch] >= n for ch, n in want.items()):
                window = s[left:right]
                if not best or len(window) < len(best):
                    best = window
                break
    return best


def main():
    assert min_window("ADOBECODEBANC", "ABC") == "BANC"
    assert min_window("cabwefgewcwaefgcf", "cae") == "cwae"
    assert min_window("a", "a") == "a"
    assert min_window("a", "aa") == ""            # pattern longer than s
    assert min_window("abc", "abcd") == ""
    assert min_window("ab", "cd") == ""           # no window
    assert min_window("aa", "aa") == "aa"         # duplicates in t
    assert min_window("abaab", "aa") == "aa"      # duplicates in t
    assert min_window("bba", "ab") == "ba"
    assert min_window("aabc", "aabc") == "aabc"
    assert min_window("abc", "") == ""            # empty pattern
    assert min_window("", "a") == ""

    import random

    rng = random.Random(232)
    for _ in range(200):                          # cross-check vs brute force
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 14)))
        t = "".join(rng.choice("ab") for _ in range(rng.randint(0, 4)))
        assert min_window(s, t) == brute_min_window(s, t)


if __name__ == "__main__":
    main()
    print("PASS")
