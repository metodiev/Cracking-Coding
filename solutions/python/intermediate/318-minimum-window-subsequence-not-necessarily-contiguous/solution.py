"""Task 318 - Minimum window subsequence (not necessarily contiguous). (Tier: Intermediate, CTCI Ch 1)

LeetCode 727: the shortest contiguous window of s that contains t as a subsequence. DP over the
prefixes of t: after scanning position i, best_start[j] is the largest start index p such that
s[p..i] contains t[0..j]; when s[i] == t[j] the state is copied from t[0..j-1] while j descends,
so the previous row is still in place. Every completion of the whole pattern yields a candidate
window and the shortest one wins (the leftmost one on ties), which reproduces the classic answers.
An empty/invalid t yields "" (documented). Lengths are cross-checked against a brute force that
tries every window.
"""


def min_window_subsequence(s, t):
    """Shortest window of s containing t as a subsequence, "" if none exists."""
    n, m = len(s), len(t)
    if m == 0 or m > n:
        return ""
    best_start = [-1] * m
    best_length = n + 1
    begin = end = -1
    for i, char in enumerate(s):
        for j in range(m - 1, -1, -1):          # descending keeps row i-1 intact
            if char == t[j]:
                best_start[j] = i if j == 0 else best_start[j - 1]
        if best_start[m - 1] != -1:
            start = best_start[m - 1]
            if i - start + 1 < best_length:     # strictly better keeps the leftmost window
                best_length = i - start + 1
                begin, end = start, i
    if begin == -1:
        return ""
    return s[begin:end + 1]


def is_subsequence(needle, haystack):
    k = 0
    for char in haystack:
        if k < len(needle) and char == needle[k]:
            k += 1
    return k == len(needle)


def brute_min_window(s, t):
    """Tries every window; the independent oracle. Returns (length, window) or (None, "")."""
    if t == "" or len(t) > len(s):
        return (None, "")
    best_length = None
    best_window = ""
    for left in range(len(s)):
        for right in range(left, len(s)):
            window = s[left:right + 1]
            if len(t) <= len(window) and is_subsequence(t, window):
                if best_length is None or len(window) < best_length:
                    best_length = len(window)
                    best_window = window
    if best_length is None:
        return (None, "")
    return (best_length, best_window)


def lcg_strings(length, count, alphabet="ab", seed=3180):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    assert min_window_subsequence("abcdebdde", "bde") == "bcde"     # [1..4], beats [5..8] "bdde"
    assert min_window_subsequence("abc", "abc") == "abc"
    assert min_window_subsequence("abc", "d") == ""
    assert min_window_subsequence("bab", "ab") == "ab"
    assert min_window_subsequence("a", "a") == "a"
    assert min_window_subsequence("a", "aa") == ""
    assert min_window_subsequence("", "a") == ""
    assert min_window_subsequence("", "") == ""
    assert min_window_subsequence("ab", "") == ""

    # Hand-verified absent case: "ezu" needs e < z < u, but every u (indices 41, 42) precedes
    # every z (indices 43, 44), and no u follows a z.
    long_text = "jmeqksfrsdcmsiwvaovxtkwmamwhvlbajfmbfmecwuuzzaqq"
    assert min_window_subsequence(long_text, "ezu") == ""
    assert brute_min_window(long_text, "ezu") == (None, "")

    patterns = lcg_strings(1, 20, "abc", seed=777) + lcg_strings(2, 20, "abc", seed=778) \
        + lcg_strings(3, 20, "abc", seed=779) + ["", "ab", "ba", "z", "cc", "bca"]

    for s in ["abcdebdde", "bab", "abc", "aab", "zzz"] + lcg_strings(30, 20) + lcg_strings(24, 10, "abc"):
        for t in patterns:
            window = min_window_subsequence(s, t)
            expected_length, expected_window = brute_min_window(s, t)
            if expected_length is None:
                assert window == "", (s, t)
            else:
                assert window in s, (s, t, window)                  # a real contiguous window
                assert is_subsequence(t, window), (s, t, window)    # containing t as subsequence
                assert len(window) == expected_length, (s, t, window, expected_window)


if __name__ == "__main__":
    main()
    print("PASS")
