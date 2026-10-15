"""Task 312 - Substring occurrences with a suffix array + LCP (Kasai). (Tier: Intermediate, CTCI Ch 1)

Build the suffix array (naive sort, as in task 310), then the Kasai LCP array where
lcp[i] = LCP(sa[i-1], sa[i]) and lcp[0] = 0. Occurrences of a pattern form a contiguous run of
the suffix array, so binary search finds the first suffix >= pattern and the run is extended with
the LCP array (any neighbour sharing lcp >= |pattern| starts with the pattern too). Every count
is cross-checked against a naive per-position scan on deterministic LCG strings.
"""


def suffix_array(s):
    return sorted(range(len(s)), key=lambda i: s[i:])


def compare_suffixes(s, i, j):
    """Manual lexicographic comparator; the independent oracle."""
    n = len(s)
    while i < n and j < n:
        if s[i] != s[j]:
            return -1 if s[i] < s[j] else 1
        i += 1
        j += 1
    if i == j:
        return 0
    return -1 if i == n else 1          # the shorter suffix is smaller


def is_valid_suffix_array(s, order):
    if sorted(order) != list(range(len(s))):
        return False
    return all(compare_suffixes(s, order[k], order[k + 1]) < 0 for k in range(len(order) - 1))


def kasai_lcp(s, sa):
    """lcp[i] = LCP(sa[i-1], sa[i]) with lcp[0] = 0; O(n) via the Kasai trick."""
    n = len(s)
    rank = [0] * n
    for i, start in enumerate(sa):
        rank[start] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h:
                h -= 1
        else:
            h = 0
    return lcp


def lcp_of_suffixes(s, i, j):
    """Direct character-by-character LCP of two suffixes; the independent oracle."""
    h = 0
    while i + h < len(s) and j + h < len(s) and s[i + h] == s[j + h]:
        h += 1
    return h


def occurrences(s, pattern, sa=None, lcp=None):
    """All start indices of the pattern in s, sorted ascending. The empty pattern matches at
    every valid position (n + 1 of them)."""
    n = len(s)
    m = len(pattern)
    if m == 0:
        return list(range(n + 1))
    if sa is None:
        sa = suffix_array(s)
    if lcp is None:
        lcp = kasai_lcp(s, sa)
    # first index k with suffix sa[k] >= pattern; matching suffixes are exactly the run at k
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if s[sa[mid]:] < pattern:
            lo = mid + 1
        else:
            hi = mid
    k = lo
    if k == n or not s.startswith(pattern, sa[k]):
        return []
    result = [sa[k]]
    t = k + 1
    while t < n and lcp[t] >= m:
        result.append(sa[t])
        t += 1
    result.sort()
    return result


def naive_occurrences(s, pattern):
    """Scan every position; the independent oracle (overlapping matches included)."""
    if pattern == "":
        return list(range(len(s) + 1))
    result = []
    for i in range(len(s) - len(pattern) + 1):
        if s[i:i + len(pattern)] == pattern:
            result.append(i)
    return result


def lcg_strings(length, count, alphabet="ab", seed=3120):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    s = "banana"
    sa = suffix_array(s)
    assert sa == [5, 3, 1, 0, 4, 2]
    assert is_valid_suffix_array(s, sa)
    lcp = kasai_lcp(s, sa)
    assert lcp == [0, 1, 3, 0, 0, 2]                       # hand-checked: a|ana, ana|anana, na|nana
    for i in range(1, len(sa)):
        assert lcp[i] == lcp_of_suffixes(s, sa[i - 1], sa[i]), i

    assert occurrences(s, "ana", sa, lcp) == [1, 3]
    assert occurrences(s, "ba", sa, lcp) == [0]
    assert occurrences(s, "an", sa, lcp) == [1, 3]
    assert occurrences(s, "a", sa, lcp) == [1, 3, 5]
    assert occurrences(s, "n", sa, lcp) == [2, 4]
    assert occurrences(s, "banana", sa, lcp) == [0]
    assert occurrences(s, "ab", sa, lcp) == []
    assert occurrences(s, "xy", sa, lcp) == []
    assert occurrences(s, "bananas", sa, lcp) == []
    assert occurrences(s, "", sa, lcp) == [0, 1, 2, 3, 4, 5, 6]
    assert occurrences("", "a") == []

    for s in ["a", "aa", "aaaa", "abab", "mississippi", "banana", ""] \
            + lcg_strings(20, 30) + lcg_strings(16, 10, "abc"):
        sa = suffix_array(s)
        lcp = kasai_lcp(s, sa)
        assert is_valid_suffix_array(s, sa), s
        for i in range(1, len(sa)):
            assert lcp[i] == lcp_of_suffixes(s, sa[i - 1], sa[i]), (s, i)
        patterns = []
        for i in range(len(s)):
            for j in (i + 1, i + 2, i + 3):
                if j <= len(s):
                    patterns.append(s[i:j])
        patterns += ["zz", "z", "ababab", "", "bbbb", "ccc"]
        for pattern in patterns:
            assert occurrences(s, pattern, sa, lcp) == naive_occurrences(s, pattern), (s, pattern)


if __name__ == "__main__":
    main()
    print("PASS")
