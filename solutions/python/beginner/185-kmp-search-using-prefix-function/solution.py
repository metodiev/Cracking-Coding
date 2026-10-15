"""Task 185 - KMP search using prefix function. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms)

Reuses the prefix function of task 184: on a mismatch at pattern[j] we fall back
to pi[j - 1] instead of restarting the scan, so the whole search is O(n + m).
Every occurrence is reported, overlapping ones included; an empty pattern has no
match and returns [] (the same convention in all four languages).
"""


def prefix_function(s):
    """Prefix function (LPS array) of `s`, as in task 184."""
    pi = [0] * len(s)
    for i in range(1, len(s)):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi


def kmp_search(text, pattern):
    """All start indices of `pattern` in `text`, in increasing order.

    Matches may overlap. An empty pattern returns [] by convention: "no match"
    avoids an unbounded list of positions.
    """
    if pattern == "":
        return []
    pi = prefix_function(pattern)
    matches = []
    j = 0
    for i, ch in enumerate(text):
        while j > 0 and ch != pattern[j]:
            j = pi[j - 1]
        if ch == pattern[j]:
            j += 1
        if j == len(pattern):
            matches.append(i - len(pattern) + 1)
            j = pi[j - 1]                 # keep going, matches may overlap
    return matches


def kmp_search_naive(text, pattern):
    """Reference scan used by the self-check."""
    if pattern == "":
        return []
    return [i for i in range(len(text) - len(pattern) + 1) if text.startswith(pattern, i)]


def main():
    assert kmp_search("aaaaa", "aaaa") == [0, 1]      # overlapping matches
    assert kmp_search("abababa", "aba") == [0, 2, 4]
    assert kmp_search("abcabcabc", "abc") == [0, 3, 6]
    assert kmp_search("abracadabra", "cad") == [4]
    assert kmp_search("mississippi", "issi") == [1, 4]
    assert kmp_search("aaaa", "b") == []              # no match
    assert kmp_search("ab", "abc") == []              # pattern longer than text
    assert kmp_search("", "a") == []
    assert kmp_search("", "") == []                   # empty pattern: no matches
    assert kmp_search("anything", "") == []
    assert kmp_search("a", "a") == [0]
    assert kmp_search("aaa", "aa") == [0, 1]
    assert kmp_search("aaa", "aaa") == [0]

    # agreement with the naive scan on pseudo-random strings
    import random
    rng = random.Random(13)
    for _ in range(300):
        text = "".join(rng.choice("ab") for _ in range(rng.randint(0, 30)))
        pattern = "".join(rng.choice("ab") for _ in range(rng.randint(1, 5)))
        assert kmp_search(text, pattern) == kmp_search_naive(text, pattern), (text, pattern)


if __name__ == "__main__":
    main()
    print("PASS")
