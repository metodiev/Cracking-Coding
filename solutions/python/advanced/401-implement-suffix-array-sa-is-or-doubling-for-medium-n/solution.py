"""Task 401 - Implement suffix array (SA-IS or doubling) for medium N. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

`SuffixArrayDoubling` sorts the suffixes by their first 2^k characters, doubling k until
every rank is unique: O(n log^2 n) with comparison sorting (O(n log n) with counting-sort
radix passes). No sentinel is appended - a position past the end of the string ranks below
every character, which resolves "one suffix is a prefix of the other" correctly because the
shorter suffix must win.

Correctness is pinned by two independent oracles: a naive suffix sort and the hand-written
character-by-character comparator `is_valid_suffix_array`. The property tests include
medium N: a few hundred characters, so the doubling loop really runs several rounds.
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


def suffix_array_naive(s):
    """Oracle: sort the start indices by comparing the materialized suffixes."""
    return sorted(range(len(s)), key=lambda i: s[i:])


def compare_suffixes(s, i, j):
    """Manual lexicographic comparator; the independent validity oracle."""
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


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def lcg_strings(length, count, alphabet="ab", seed=2718):
    strings = []
    for _ in range(count):
        s, seed = lcg_string(length, alphabet, seed)
        strings.append(s)
    return strings


def main():
    assert suffix_array_doubling("banana") == [5, 3, 1, 0, 4, 2]
    assert suffix_array_doubling("mississippi") == [10, 7, 4, 1, 0, 9, 8, 6, 3, 5, 2]

    assert suffix_array_doubling("") == []
    assert suffix_array_doubling("a") == [0]
    assert suffix_array_doubling("aaaa") == [3, 2, 1, 0]
    assert suffix_array_doubling("abab") == [2, 0, 3, 1]
    assert suffix_array_doubling("abracadabra") == sorted(range(11), key=lambda i: "abracadabra"[i:])

    small = ["", "a", "b", "aa", "ab", "ba", "aaaa", "abab", "banana", "mississippi", "abracadabra"]
    for s in small:
        order = suffix_array_doubling(s)
        assert order == suffix_array_naive(s), s
        assert is_valid_suffix_array(s, order), s

    for alphabet in ("ab", "abc", "abcdefghijklmnopqrstuvwxyz"):
        for s in lcg_strings(28, 40, alphabet):
            order = suffix_array_doubling(s)
            assert order == suffix_array_naive(s), s
            assert is_valid_suffix_array(s, order), s

    medium = [
        lcg_strings(500, 1, "ab", 991)[0],
        lcg_strings(500, 1, "abcdefghijklmnopqrstuvwxyz", 4242)[0],
        "a" * 400,
        "abcab" * 100,
    ]
    for s in medium:
        order = suffix_array_doubling(s)
        assert order == suffix_array_naive(s), "medium naive mismatch"
        assert is_valid_suffix_array(s, order), "medium comparator mismatch"


if __name__ == "__main__":
    main()
    print("PASS")
