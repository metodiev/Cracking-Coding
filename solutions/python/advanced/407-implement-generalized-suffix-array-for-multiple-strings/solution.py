"""Task 407 - Implement a generalized suffix array for multiple strings. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

The m strings are concatenated with m *distinct* sentinels: string k gets chr(1 + k). All
sentinels are below every real (lowercase) character, and they are distinct and ordered by
string index, so a suffix from string i always beats the same text extended in string j when
i < j. This is the cheapest correct sentinel scheme for the generalized suffix array.

Every suffix also carries an owner (which string) and an offset (position inside it), so
occurrences of a pattern are found with two binary searches over the suffix array: the
pattern-prefixed suffixes form one contiguous range, and each entry of that range maps back to
a (string, position) occurrence.

Anchors: ["banana", "ana"] -> occurrences of "ana" = (0,1), (0,3), (1,0); "a" = (0,1), (0,3),
(0,5), (1,0), (1,2). Checks: the combined SA validates against a manual comparator, the two
smallest suffixes are the sentinels in index order, suffixes of each string keep their own
plain relative order, and LCG-built pattern searches agree with a naive scan.
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


def compare_suffixes(s, i, j):
    n = len(s)
    while i < n and j < n:
        if s[i] != s[j]:
            return -1 if s[i] < s[j] else 1
        i += 1
        j += 1
    if i == j:
        return 0
    return -1 if i == n else 1


def is_valid_suffix_array(s, order):
    if sorted(order) != list(range(len(s))):
        return False
    return all(compare_suffixes(s, order[k], order[k + 1]) < 0 for k in range(len(order) - 1))


def build_generalized(strings):
    """Concat with sentinels chr(1+k); return (text, sa, owner, offset, sentinel_positions)."""
    chars = []
    owner = []
    offset = []
    sentinel_positions = []
    for k, s in enumerate(strings):
        for pos, ch in enumerate(s):
            chars.append(ch)
            owner.append(k)
            offset.append(pos)
        sentinel_positions.append(len(chars))
        chars.append(chr(1 + k))
        owner.append(k)
        offset.append(-1)
    text = "".join(chars)
    sa = suffix_array_doubling(text)
    return text, sa, owner, offset, sentinel_positions


def cmp_prefix(text, start, pattern):
    """-1 / 0 / 1: suffix smaller than pattern / starts with it / larger than it."""
    i = 0
    while i < len(pattern):
        if start + i >= len(text):
            return -1
        ch = text[start + i]
        if ch != pattern[i]:
            return -1 if ch < pattern[i] else 1
        i += 1
    return 0


def pattern_range(text, sa, pattern):
    lo, hi = 0, len(sa)
    while lo < hi:                                  # first suffix that is >= pattern
        mid = (lo + hi) // 2
        if cmp_prefix(text, sa[mid], pattern) < 0:
            lo = mid + 1
        else:
            hi = mid
    lower = lo
    hi = len(sa)
    while lo < hi:                                  # first suffix that is > pattern
        mid = (lo + hi) // 2
        if cmp_prefix(text, sa[mid], pattern) <= 0:
            lo = mid + 1
        else:
            hi = mid
    return lower, lo


def find_occurrences(strings, pattern):
    text, sa, owner, offset, _ = build_generalized(strings)
    lower, upper = pattern_range(text, sa, pattern)
    found = sorted((owner[sa[i]], offset[sa[i]]) for i in range(lower, upper) if offset[sa[i]] >= 0)
    return found


def naive_occurrences(strings, pattern):
    found = []
    for k, s in enumerate(strings):
        for i in range(len(s) - len(pattern) + 1):
            if s[i:i + len(pattern)] == pattern:
                found.append((k, i))
    return sorted(found)


def suffix_array_naive(s):
    return sorted(range(len(s)), key=lambda i: s[i:])


def all_short_substrings(strings, max_len):
    out = set()
    for s in strings:
        for i in range(len(s)):
            for j in range(i + 1, min(len(s), i + max_len) + 1):
                out.add(s[i:j])
    return sorted(out)


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def main():
    strings = ["banana", "ana"]
    text, sa, owner, offset, sentinels = build_generalized(strings)
    assert len(text) == 6 + 1 + 3 + 1
    assert is_valid_suffix_array(text, sa)
    assert [sa[0], sa[1]] == sentinels               # sentinels sort first, by string index
    for k, s in enumerate(strings):
        kept = [offset[i] for i in sa if owner[i] == k and offset[i] >= 0]
        assert kept == suffix_array_naive(s), k

    assert find_occurrences(strings, "ana") == [(0, 1), (0, 3), (1, 0)]
    assert find_occurrences(strings, "a") == [(0, 1), (0, 3), (0, 5), (1, 0), (1, 2)]
    assert find_occurrences(strings, "na") == [(0, 2), (0, 4), (1, 1)]   # also inside "ana"
    assert find_occurrences(strings, "ban") == [(0, 0)]
    assert find_occurrences(strings, "banana") == [(0, 0)]
    assert find_occurrences(strings, "b") == [(0, 0)]
    assert find_occurrences(strings, "xyz") == []
    assert find_occurrences(strings, "anan") == [(0, 1)]   # "anana" inside banana
    assert find_occurrences(strings, "nanan") == []
    assert find_occurrences(strings, "bananas") == []

    # edge shapes: single string, empty string inside the list, four strings
    assert find_occurrences(["abab"], "ab") == [(0, 0), (0, 2)]
    assert find_occurrences(["", "ab"], "ab") == [(1, 0)]
    assert find_occurrences(["", ""], "a") == []
    assert find_occurrences([], "a") == []
    quad = ["mississippi", "issi", "i", "ppi"]
    for pattern in all_short_substrings(quad, 3) + ["issi", "ississi", "ssippi", "zz", "q", "ippiz"]:
        assert find_occurrences(quad, pattern) == naive_occurrences(quad, pattern), pattern
    assert find_occurrences(quad, "issi") == naive_occurrences(quad, "issi")
    assert len(find_occurrences(quad, "i")) == 4 + 2 + 1 + 1
    assert find_occurrences(quad, "i") == naive_occurrences(quad, "i")

    for trial in range(30):
        strings, seed = [], 9000 + trial * 65537
        for _ in range(3):
            s, seed = lcg_string(12 + seed % 7, "abc", seed)
            strings.append(s)
        text, sa, owner, offset, sentinels = build_generalized(strings)
        assert is_valid_suffix_array(text, sa), strings
        assert [sa[k] for k in range(len(strings))] == sentinels, strings
        for k, s in enumerate(strings):
            kept = [offset[i] for i in sa if owner[i] == k and offset[i] >= 0]
            assert kept == suffix_array_naive(s), (strings, k)
        for pattern in all_short_substrings(strings, 3) + ["zzz", "q", "abca" * 2]:
            assert find_occurrences(strings, pattern) == naive_occurrences(strings, pattern), pattern


if __name__ == "__main__":
    main()
    print("PASS")
