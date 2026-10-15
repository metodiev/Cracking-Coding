"""Task 408 - Pattern matching with a suffix tree (outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Backing store: the suffix array plus the LCP array, which represent the same object as a
suffix tree - a suffix-tree node corresponds to an LCP interval, and the locus of a pattern is
exactly the maximal interval [lo, hi) of suffixes carrying the pattern as prefix. Descent from
the root consumes the pattern one character at a time: two binary searches over the suffix
array locate the interval (compare the pattern against the suffix at the midpoint), and every
suffix inside the interval is a match, because in a suffix array all pattern-prefixed suffixes
are contiguous. The LCP array documents/validates the locus: all suffixes in [lo, hi) share
at least len(pattern) prefix characters, so min(lcp[lo+1 .. hi-1]) >= len(pattern).

Checks: "banana" anchors, then LCG texts where `locate` (and its interval width) agree with a
naive scan for substrings, mutated and absent patterns. Empty pattern is documented to match
every position.
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


class SuffixTreeIndex:
    """Pattern locator: SA + LCP as the array form of a suffix tree."""

    def __init__(self, text):
        self.text = text
        self.sa = suffix_array_doubling(text)
        self.lcp = kasai_lcp(text, self.sa)

    def _cmp(self, start, pattern):
        """-1 / 0 / 1: suffix at `start` smaller than / starts with / larger than pattern."""
        i = 0
        while i < len(pattern):
            if start + i >= len(self.text):
                return -1
            ch = self.text[start + i]
            if ch != pattern[i]:
                return -1 if ch < pattern[i] else 1
            i += 1
        return 0

    def interval(self, pattern):
        """Locus of the pattern in the suffix tree; empty pattern = the whole array."""
        if not pattern:
            return 0, len(self.sa)
        lo, hi = 0, len(self.sa)
        while lo < hi:                              # first suffix >= pattern
            mid = (lo + hi) // 2
            if self._cmp(self.sa[mid], pattern) < 0:
                lo = mid + 1
            else:
                hi = mid
        lower = lo
        hi = len(self.sa)
        while lo < hi:                              # first suffix > pattern
            mid = (lo + hi) // 2
            if self._cmp(self.sa[mid], pattern) <= 0:
                lo = mid + 1
            else:
                hi = mid
        return lower, lo

    def locate(self, pattern):
        lower, upper = self.interval(pattern)
        return sorted(self.sa[lower:upper])

    def count(self, pattern):
        lower, upper = self.interval(pattern)
        return upper - lower

    def locus_shared_prefix(self, pattern):
        """min lcp over the interval (0 when the interval has one element): tree-locus check."""
        lower, upper = self.interval(pattern)
        if upper - lower <= 1:
            return 0
        return min(self.lcp[lower + 1:upper])


def naive_locate(text, pattern):
    if not pattern:
        return list(range(len(text)))
    return [i for i in range(len(text) - len(pattern) + 1) if text[i:i + len(pattern)] == pattern]


def suffix_array_naive(s):
    return sorted(range(len(s)), key=lambda i: s[i:])


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def check_index(text, extra_patterns):
    index = SuffixTreeIndex(text)
    assert index.sa == suffix_array_naive(text), text
    patterns = extra_patterns[:]
    for i in range(len(text)):
        for j in range(i + 1, min(len(text), i + 5) + 1):
            patterns.append(text[i:j])
    patterns.extend(["zz", "qq", "!"])
    for pattern in patterns:
        positions = index.locate(pattern)
        assert positions == naive_locate(text, pattern), (text, pattern)
        assert index.count(pattern) == len(positions), (text, pattern)
        lower, upper = index.interval(pattern)
        for k in range(lower, upper):
            if pattern:
                assert text[index.sa[k]:index.sa[k] + len(pattern)] == pattern, (text, pattern)
        if pattern and lower > 0:
            prev = index.sa[lower - 1]
            assert text[prev:prev + len(pattern)] != pattern, (text, pattern)
        if pattern and upper < len(index.sa):
            nxt = index.sa[upper]
            assert text[nxt:nxt + len(pattern)] != pattern, (text, pattern)
        if pattern and upper - lower >= 2:
            assert index.locus_shared_prefix(pattern) >= len(pattern), (text, pattern)
    return index


def main():
    banana = check_index("banana", ["ana", "na", "a", "banana", "anana", "bananas", "nan"])
    assert banana.locate("ana") == [1, 3]
    assert banana.locate("na") == [2, 4]
    assert banana.locate("a") == [1, 3, 5]
    assert banana.locate("banana") == [0]
    assert banana.locate("anana") == [1]
    assert banana.locate("") == [0, 1, 2, 3, 4, 5]
    assert banana.locate("nana") == [2]
    assert banana.locate("bananas") == []
    assert banana.interval("ana") == (1, 3)   # suffixes "ana" and "anana"
    assert banana.locus_shared_prefix("ana") >= 3

    check_index("mississippi", ["issi", "iss", "ssi", "mississippi", "ppi", "sipp"])
    check_index("", ["a"])
    check_index("a", ["a", "aa"])
    check_index("aaaa", ["aaa", "aa", "a", "aaaaa"])

    for trial in range(30):
        text, seed = lcg_string(30, "abc", 11000 + trial * 65537)
        mutations = [text[:k] + "d" + text[k + 1:] for k in range(0, min(len(text), 8))]
        check_index(text, mutations[:6])
    for trial in range(10):
        text, seed = lcg_string(60, "ab", 22000 + trial * 7919)
        check_index(text, [text, text[:-1], "z" + text])


if __name__ == "__main__":
    main()
    print("PASS")
