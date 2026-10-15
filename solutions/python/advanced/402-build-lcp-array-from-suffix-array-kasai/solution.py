"""Task 402 - Build the LCP array from the suffix array (Kasai). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Kasai's algorithm walks the suffixes in *text* order, reusing the previous longest common
prefix: if suffix i shares h characters with its suffix-array predecessor, then suffix i+1
shares at least h-1 with its own predecessor, so each position is advanced amortized once
and the whole scan is O(n). `lcp[i]` is defined as the LCP of suffixes sa[i-1] and sa[i],
with `lcp[0] = 0` (the first suffix has no predecessor).

Hand-checked anchors: banana -> [0, 1, 3, 0, 0, 2] (a/ana = 1, ana/anana = 3, ...) and
mississippi -> [0, 1, 1, 4, 0, 0, 1, 0, 2, 1, 3]. A pairwise char-by-char oracle and the
identity "distinct substrings = n(n+1)/2 - sum(lcp)" cross-check it on LCG strings.
"""

def kasai_lcp(s, sa):
    n = len(s)
    lcp = [0] * n
    rank = [0] * n
    for i, start in enumerate(sa):
        rank[start] = i
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            h = 0
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and s[i + h] == s[j + h]:
            h += 1
        lcp[r] = h
        if h:
            h -= 1
    return lcp


def pair_lcp(s, i, j):
    """Direct char-by-char LCP of two suffixes; the independent oracle."""
    n = len(s)
    h = 0
    while i + h < n and j + h < n and s[i + h] == s[j + h]:
        h += 1
    return h


def naive_lcp_array(s, sa):
    if not sa:
        return []
    return [0] + [pair_lcp(s, sa[i - 1], sa[i]) for i in range(1, len(sa))]


def suffix_array_naive(s):
    return sorted(range(len(s)), key=lambda i: s[i:])


def distinct_substrings_brute(s):
    seen = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            seen.add(s[i:j])
    return len(seen)


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def main():
    banana = kasai_lcp("banana", [5, 3, 1, 0, 4, 2])
    assert banana == [0, 1, 3, 0, 0, 2], banana
    assert banana == naive_lcp_array("banana", [5, 3, 1, 0, 4, 2])

    mississippi = kasai_lcp("mississippi", [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2])
    assert mississippi == [0, 1, 1, 4, 0, 0, 1, 0, 2, 1, 3], mississippi
    assert mississippi == naive_lcp_array("mississippi", [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2])

    assert kasai_lcp("", []) == []
    assert kasai_lcp("a", [0]) == [0]
    assert kasai_lcp("aaaa", [3, 2, 1, 0]) == [0, 1, 2, 3]
    assert kasai_lcp("abab", [2, 0, 3, 1]) == [0, 2, 0, 1]

    small = ["", "a", "aaa", "abab", "banana", "mississippi", "abracadabra", "aaaaabaaaab"]
    for s in small:
        sa = suffix_array_naive(s)
        lcp = kasai_lcp(s, sa)
        assert lcp == naive_lcp_array(s, sa), s
        assert len(lcp) == len(s), s
        if lcp:
            assert lcp[0] == 0, s

    # distinct substrings = n(n+1)/2 - sum(lcp): a second, counting-based cross-check.
    assert 15 == distinct_substrings_brute("banana")
    assert 6 * 7 // 2 - sum(banana) == distinct_substrings_brute("banana")

    for alphabet in ("ab", "abc"):
        for trial in range(40):
            s, seed = lcg_string(24, alphabet, 1000 + trial * 7919)
            sa = suffix_array_naive(s)
            assert kasai_lcp(s, sa) == naive_lcp_array(s, sa), s
            assert len(s) * (len(s) + 1) // 2 - sum(kasai_lcp(s, sa)) == distinct_substrings_brute(s), s

    medium, _ = lcg_string(400, "ab", 555)
    sa = suffix_array_naive(medium)
    assert kasai_lcp(medium, sa) == naive_lcp_array(medium, sa), "medium mismatch"
    assert kasai_lcp(medium, sa)[0] == 0


if __name__ == "__main__":
    main()
    print("PASS")
