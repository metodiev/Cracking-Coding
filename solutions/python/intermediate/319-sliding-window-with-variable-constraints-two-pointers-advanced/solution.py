"""Task 319 - Sliding window with variable constraints (two pointers, advanced). (Tier: Intermediate, CTCI Ch 1)

Two constraint variants share the same two-pointer skeleton:
  * longest_at_most_k_distinct: shrink while the window holds more than k distinct characters;
  * character_replacement: shrink while (window length - highest frequency in the window) > k,
    i.e. the window needs more than k replacements to become one repeated character.
Both are O(n) amortised because each index enters and leaves the window once. Brute-force O(n^2)
checkers validate both variants on deterministic LCG strings.
"""


def longest_at_most_k_distinct(s, k):
    """Length of the longest substring with at most k distinct characters (0 if k <= 0)."""
    if k <= 0:
        return 0
    counts = {}
    left = 0
    best = 0
    for right, char in enumerate(s):
        counts[char] = counts.get(char, 0) + 1
        while len(counts) > k:
            counts[s[left]] -= 1
            if counts[s[left]] == 0:
                del counts[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best


def character_replacement(s, k):
    """Length of the longest substring obtainable by replacing at most k characters."""
    counts = {}
    left = 0
    best = 0
    max_count = 0
    for right, char in enumerate(s):
        counts[char] = counts.get(char, 0) + 1
        max_count = max(max_count, counts[char])
        while (right - left + 1) - max_count > k:
            counts[s[left]] -= 1
            left += 1
            max_count = max(counts.values())        # recomputed after every shrink
        best = max(best, right - left + 1)
    return best


def brute_at_most_k_distinct(s, k):
    """Try every window; the independent oracle."""
    best = 0
    for i in range(len(s)):
        seen = set()
        for j in range(i, len(s)):
            seen.add(s[j])
            if len(seen) > k:
                break
            best = max(best, j - i + 1)
    return best


def brute_character_replacement(s, k):
    """Try every window and count its maximal frequency; the independent oracle."""
    best = 0
    for i in range(len(s)):
        counts = {}
        for j in range(i, len(s)):
            counts[s[j]] = counts.get(s[j], 0) + 1
            if (j - i + 1) - max(counts.values()) <= k:
                best = max(best, j - i + 1)
    return best


def lcg_strings(length, count, alphabet="abc", seed=3190):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    assert longest_at_most_k_distinct("eceba", 2) == 3                 # "ece"
    assert longest_at_most_k_distinct("aa", 1) == 2
    assert longest_at_most_k_distinct("abcadcacacaca", 3) == 11        # "cadcacacaca"
    assert longest_at_most_k_distinct("abcadcacacaca", 2) == 8         # "cacacaca"
    assert longest_at_most_k_distinct("", 2) == 0
    assert longest_at_most_k_distinct("abc", 0) == 0
    assert longest_at_most_k_distinct("abc", 5) == 3
    assert longest_at_most_k_distinct("abaccc", 1) == 3                 # "ccc"

    assert character_replacement("ABAB", 2) == 4
    assert character_replacement("AABABBA", 1) == 4
    assert character_replacement("AAAA", 2) == 4
    assert character_replacement("ABCDE", 0) == 1
    assert character_replacement("", 0) == 0
    assert character_replacement("A", 0) == 1
    assert character_replacement("ABAB", 0) == 1

    for s in ["eceba", "aa", "abcadcacacaca"] + lcg_strings(20, 30) + lcg_strings(14, 15, "ab"):
        for k in range(0, 4):
            assert longest_at_most_k_distinct(s, k) == brute_at_most_k_distinct(s, k), (s, k)
            assert character_replacement(s, k) == brute_character_replacement(s, k), (s, k)
        # both objectives are monotone in k and bounded by the string length
        assert longest_at_most_k_distinct(s, 4) == len(s)
        for k in range(0, 3):
            assert longest_at_most_k_distinct(s, k) <= longest_at_most_k_distinct(s, k + 1)
            assert character_replacement(s, k) <= character_replacement(s, k + 1)


if __name__ == "__main__":
    main()
    print("PASS")
