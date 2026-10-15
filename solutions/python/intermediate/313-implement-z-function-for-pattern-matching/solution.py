"""Task 313 - The Z-function for pattern matching. (Tier: Intermediate, CTCI Ch 1)

z[i] is the length of the longest common prefix of s and s[i:]; z[0] = 0 by convention (the
whole string is trivially a prefix of itself). The linear Z-algorithm reuses the rightmost
matching segment [l, r) to seed each value. Pattern matching concatenates
pattern + sentinel + text ("#" must not occur in either) and reports every position where
z >= |pattern|. All results are cross-checked against a naive search on deterministic LCG strings.
"""

SENTINEL = "#"


def z_function(s):
    n = len(s)
    z = [0] * n                      # z[0] = 0 by convention
    left = right = 0
    for i in range(1, n):
        if i < right:
            z[i] = min(right - i, z[i - left])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > right:
            left, right = i, i + z[i]
    return z


def z_function_naive(s):
    """Direct O(n^2) definition; the independent oracle."""
    n = len(s)
    z = [0] * n
    for i in range(1, n):
        h = 0
        while i + h < n and s[h] == s[i + h]:
            h += 1
        z[i] = h
    return z


def z_search(text, pattern):
    """All start indices of pattern in text via the Z-function on pattern + "#" + text."""
    if pattern == "":
        return list(range(len(text) + 1))
    combined = pattern + SENTINEL + text
    z = z_function(combined)
    m = len(pattern)
    return [i - m - 1 for i in range(m + 1, len(combined)) if z[i] >= m]


def naive_search(text, pattern):
    """Independent oracle: try every position."""
    if pattern == "":
        return list(range(len(text) + 1))
    result = []
    for i in range(len(text) - len(pattern) + 1):
        if text[i:i + len(pattern)] == pattern:
            result.append(i)
    return result


def lcg_strings(length, count, alphabet="ab", seed=3130):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    # Worked example, values checked by hand
    assert z_function("aabcaabxaaaz") == [0, 1, 0, 0, 3, 1, 0, 0, 2, 2, 1, 0]
    assert z_function("aaaa") == [0, 3, 2, 1]
    assert z_function("abab") == [0, 0, 2, 0]
    assert z_function("a") == [0]
    assert z_function("") == []

    text = "aabcaabxaaaz"
    assert z_search(text, "aab") == [0, 4]
    assert z_search(text, "aa") == [0, 4, 8, 9]
    assert z_search(text, "az") == [10]
    assert z_search(text, "x") == [7]
    assert z_search(text, "zz") == []
    assert z_search(text, "bcaab") == [2]
    assert z_search(text, text) == [0]
    assert z_search("abc", "") == [0, 1, 2, 3]
    assert z_search("", "a") == []

    for s in ["a", "aa", "aaaa", "abab", "aabcaabxaaaz", "mississippi"] \
            + lcg_strings(24, 30) + lcg_strings(18, 12, "abc"):
        z = z_function(s)
        assert z == z_function_naive(s), s
        # z[0] is 0 by convention and every other entry is bounded by the suffix length
        assert all(0 <= z[i] <= len(s) - i for i in range(len(s))), s
        patterns = []
        for i in range(len(s)):
            for j in (i + 1, i + 2, i + 3):
                if j <= len(s):
                    patterns.append(s[i:j])
        patterns += ["zz", "ab" * 3, ""]
        for pattern in patterns:
            assert z_search(s, pattern) == naive_search(s, pattern), (s, pattern)


if __name__ == "__main__":
    main()
    print("PASS")
