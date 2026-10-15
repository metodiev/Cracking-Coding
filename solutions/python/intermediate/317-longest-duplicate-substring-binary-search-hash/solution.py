"""Task 317 - Longest duplicate substring via binary search + rolling hash. (Tier: Intermediate, CTCI Ch 1)

If a duplicated substring of length L exists then so does one of length L-1 (drop the last
character), so "has a duplicate of length L" is monotone and binary search finds the maximum L.
Each probe drops the O(n) substring hashes of that length into a set (double rolling hash, mod
1_000_000_007 and 998_244_353, base 131), giving O(n log n) hashing work overall. The best length
is cross-checked against a naive O(n^2) all-pairs LCP scan, and "" is the documented sentinel
meaning "no duplicate substring exists" (the function always returns the duplicate found at the
smallest starting index for the maximal length, so the result is deterministic).
"""

MOD1 = 1_000_000_007
MOD2 = 998_244_353
BASE = 131


class DoubleRollingHash:
    """Prefix hashes for two moduli; substring hash is O(1)."""

    def __init__(self, s):
        n = len(s)
        self.h1 = [0] * (n + 1)
        self.h2 = [0] * (n + 1)
        self.p1 = [1] * (n + 1)
        self.p2 = [1] * (n + 1)
        for i, char in enumerate(s):
            code = ord(char)
            self.h1[i + 1] = (self.h1[i] * BASE + code) % MOD1
            self.h2[i + 1] = (self.h2[i] * BASE + code) % MOD2
            self.p1[i + 1] = self.p1[i] * BASE % MOD1
            self.p2[i + 1] = self.p2[i] * BASE % MOD2

    def hash(self, lo, hi):
        a = (self.h1[hi] - self.h1[lo] * self.p1[hi - lo]) % MOD1
        b = (self.h2[hi] - self.h2[lo] * self.p2[hi - lo]) % MOD2
        return (a, b)


def longest_duplicate_substring(s):
    n = len(s)
    if n <= 1:
        return ""
    rolling = DoubleRollingHash(s)

    def has_duplicate(length):
        seen = set()
        for i in range(n - length + 1):
            key = rolling.hash(i, i + length)
            if key in seen:
                return True
            seen.add(key)
        return False

    lo, hi, best = 1, n - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if has_duplicate(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best == 0:
        return ""
    for i in range(n - best + 1):
        key = rolling.hash(i, i + best)
        for j in range(i + 1, n - best + 1):
            if rolling.hash(j, j + best) == key:
                return s[i:i + best]
    raise AssertionError("binary search and extraction disagree")


def longest_duplicate_naive_length(s):
    """All-pairs LCP scan; the independent oracle for the best length."""
    best = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            h = 0
            while j + h < len(s) and s[i + h] == s[j + h]:
                h += 1
            if h > best:
                best = h
    return best


def overlapping_count(s, word):
    """Occurrences counted with overlaps (str.count only counts non-overlapping ones)."""
    return sum(1 for i in range(len(s) - len(word) + 1) if s[i:i + len(word)] == word)


def lcg_strings(length, count, alphabet="ab", seed=3170):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    assert longest_duplicate_substring("banana") == "ana"        # "ana" at 1 and 3
    assert longest_duplicate_substring("aaaa") == "aaa"          # "aaa" at 0 and 1
    assert longest_duplicate_substring("abcd") == ""             # no repeated character at all
    assert longest_duplicate_substring("") == ""
    assert longest_duplicate_substring("a") == ""
    assert longest_duplicate_substring("aa") == "a"
    assert longest_duplicate_substring("abab") == "ab"
    assert longest_duplicate_substring("abcabcabc") == "abcabc"  # at 0 and 3
    assert longest_duplicate_substring("mississippi") == "issi"  # at 1 and 4

    for s in ["banana", "aaaa", "abcd", "abab", "abcabcabc", "mississippi", "aabcaabxaaaz"] \
            + lcg_strings(24, 30) + lcg_strings(18, 15, "abc"):
        result = longest_duplicate_substring(s)
        naive = longest_duplicate_naive_length(s)
        assert len(result) == naive, (s, result, naive)
        if naive == 0:
            assert result == "", s
        else:
            assert overlapping_count(s, result) > 1, (s, result)    # a real duplicate ...
            assert result == longest_duplicate_substring(s)         # ... and deterministic


if __name__ == "__main__":
    main()
    print("PASS")
