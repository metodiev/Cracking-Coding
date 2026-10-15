"""Task 403 - Use the suffix array + LCP array for the longest repeated substring. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A substring that occurs at least twice appears as a common prefix of two different suffixes.
Sorting the suffixes makes those two suffixes adjacent *somewhere* in the suffix array, and
the deepest such adjacency has exactly the maximum LCP: so the answer length is max(lcp) and
one witness is the prefix of length max(lcp) of sa[i] at the first i achieving the maximum.

Sentinel convention: when no substring occurs twice (length < 2, or every substring unique,
e.g. "abc"), the function returns the empty string "" - documented here so callers can tell
"no repeat" apart from a genuine one-character repeat only via the length.

Checked against a brute-force substring-counting oracle (same max length on LCG strings; the
witness is re-verified to occur at least twice), plus hand anchors: banana -> "ana",
mississippi -> "issi", aaaa -> "aaa", abab -> "ab".
"""

def suffix_array_doubling(s):
    n = len(s)
    if n == 0:
        return []
    sa = list(range(n))
    rank = [ord(c) for c in s]
    k = 1
    while True:
        def key(i):
            return (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)
        new_rank = [0] * n
        for idx in range(1, n):
            prev, cur = sa[idx - 1], sa[idx]
            new_rank[cur] = new_rank[prev] + (0 if key(prev) == key(cur) else 1)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            return sa
        k <<= 1


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


def longest_repeated_substring(s):
    """One longest substring occurring at least twice, or "" when nothing repeats."""
    if len(s) < 2:
        return ""
    sa = suffix_array_doubling(s)
    lcp = kasai_lcp(s, sa)
    best_len = 0
    best_pos = -1
    for i in range(1, len(s)):
        if lcp[i] > best_len:
            best_len = lcp[i]
            best_pos = sa[i]
    if best_len == 0:
        return ""
    return s[best_pos:best_pos + best_len]


def longest_repeated_length_brute(s):
    """Oracle: longest length L such that some length-L substring occurs twice."""
    counts = {}
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            t = s[i:j]
            counts[t] = counts.get(t, 0) + 1
    return max([len(t) for t, c in counts.items() if c >= 2], default=0)


def count_occurrences(s, pattern):
    if not pattern:
        return 0
    return sum(1 for i in range(len(s) - len(pattern) + 1) if s[i:i + len(pattern)] == pattern)


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def main():
    assert longest_repeated_substring("banana") == "ana"
    assert count_occurrences("banana", "ana") == 2

    assert longest_repeated_substring("mississippi") == "issi"
    assert count_occurrences("mississippi", "issi") == 2

    assert longest_repeated_substring("aaaa") == "aaa"
    assert longest_repeated_substring("abab") == "ab"
    assert longest_repeated_substring("abcabc") == "abc"
    assert longest_repeated_substring("aabcaa") == "aa"
    assert longest_repeated_substring("abracadabra") == "abra"
    assert count_occurrences("abracadabra", "abra") == 2

    # sentinel: no repeated substring at all
    assert longest_repeated_substring("") == ""
    assert longest_repeated_substring("a") == ""
    assert longest_repeated_substring("abc") == ""
    assert longest_repeated_substring("abcd") == ""
    assert longest_repeated_length_brute("abc") == 0

    for alphabet in ("ab", "abc"):
        for trial in range(40):
            s, seed = lcg_string(30, alphabet, 2000 + trial * 104729)
            result = longest_repeated_substring(s)
            best = longest_repeated_length_brute(s)
            assert len(result) == best, s
            if best:
                assert count_occurrences(s, result) >= 2, s
            else:
                assert result == "", s

    medium, _ = lcg_string(400, "ab", 31337)
    result = longest_repeated_substring(medium)
    assert result and count_occurrences(medium, result) >= 2, "medium witness must repeat"
    assert len(result) == longest_repeated_length_brute(medium), "medium length mismatch"


if __name__ == "__main__":
    main()
    print("PASS")
