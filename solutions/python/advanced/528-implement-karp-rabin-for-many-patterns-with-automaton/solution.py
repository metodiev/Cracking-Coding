"""Task 528 - Implement Karp-Rabin for many patterns with automaton. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Multi-pattern search that combines the two textbook tools: an Aho-Corasick automaton over the
pattern set supplies the exact reference answer (trie of the patterns, BFS-computed failure
links, an output list per state carrying the patterns that end there, one linear scan of the
text), while Karp-Rabin supplies the hashed filter - a polynomial rolling hash h(window) =
sum c_i * B^(L-1-i) mod M with precomputed powers, so every window of every distinct pattern
length is hashed in O(1) amortised time and compared against the precomputed pattern hashes.
Because a small modulus can collide, each hash hit is then confirmed by a direct character
comparison, which keeps the answer exact; the self-test demonstrates a *real* collision with
B = 10, M = 97 on digit strings ("97" and "00" both hash to 0), where the unverified filter
reports a false positive that the verified matcher rejects. Documented checks: the reported
match sets of the verified rolling-hash matcher, the unverified filter (a superset), the
Aho-Corasick automaton and an O(n * m) brute-force scan are identical for hand-checked examples
("abracadabra" with abra/ca/dabra/a) and for LCG texts of length 60-200 over alphabets of size
2-4 with all substrings of length 1..4 as patterns (duplicated patterns collapse to a set); the
documented counters satisfy candidates == matches for the large prime modulus and
candidates > matches in the collision demo; the number of verification comparisons equals the
number of candidates; the number of hashed windows equals the sum of (n - L + 1) over the
distinct pattern lengths; patterns longer than the text and patterns absent from it produce no
matches; the empty-pattern list and the empty text are handled; and repeated runs are identical.
"""

PRIME_MODULUS = 1000000007
PRIME_BASE = 131
DEMO_MODULUS = 97
DEMO_BASE = 10


def to_codes(text):
    """Characters become their ordinals, which keeps the hash exact for digits and letters."""
    return [ord(ch) for ch in text]


class RollingHash:
    def __init__(self, base, modulus):
        self.base = base
        self.modulus = modulus
        self.powers = [1]

    def _ensure(self, length):
        while len(self.powers) <= length:
            self.powers.append(self.powers[-1] * self.base % self.modulus)

    def hash(self, codes):
        self._ensure(len(codes))
        value = 0
        for code in codes:
            value = (value * self.base + code) % self.modulus
        return value

    def windows(self, codes, length):
        """Rolling hashes of all windows of the given length, yielding (start, hash)."""
        n = len(codes)
        if length <= 0 or length > n:
            return
        self._ensure(length)
        highest = self.powers[length - 1]          # weight of the character leaving the window
        current = 0
        for position, code in enumerate(codes):
            if position >= length:
                current = (current - codes[position - length] * highest) % self.modulus
            current = (current * self.base + code) % self.modulus
            if position >= length - 1:
                yield position - length + 1, current


class AhoCorasick:
    """Exact multi-pattern automaton: goto trie + failure links + per-state outputs."""

    def __init__(self, patterns):
        self.patterns = sorted(set(patterns))
        if any(not pattern for pattern in self.patterns):
            raise ValueError("empty patterns are not supported")
        self.goto = [{}]
        self.fail = [0]
        self.out = [[]]
        for pattern in self.patterns:
            state = 0
            for character in pattern:
                if character not in self.goto[state]:
                    self.goto.append({})
                    self.fail.append(0)
                    self.out.append([])
                    self.goto[state][character] = len(self.goto) - 1
                state = self.goto[state][character]
            self.out[state].append(pattern)
        self._build_failures()

    def _build_failures(self):
        queue = []
        for character, child in sorted(self.goto[0].items()):
            self.fail[child] = 0
            queue.append(child)
        head = 0
        while head < len(queue):
            state = queue[head]
            head += 1
            for character, child in sorted(self.goto[state].items()):
                fallback = self.fail[state]
                while fallback and character not in self.goto[fallback]:
                    fallback = self.fail[fallback]
                self.fail[child] = self.goto[fallback].get(character, 0)
                self.out[child] = self.out[child] + self.out[self.fail[child]]
                queue.append(child)

    def search(self, text):
        """All (start, pattern) matches, found in one pass over the text."""
        matches = []
        state = 0
        for position, character in enumerate(text):
            while state and character not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(character, 0)
            for pattern in self.out[state]:
                matches.append((position - len(pattern) + 1, pattern))
        return sorted(matches)


def brute_force_search(text, patterns):
    matches = []
    for pattern in sorted(set(patterns)):
        for start in range(len(text) - len(pattern) + 1):
            if text[start:start + len(pattern)] == pattern:
                matches.append((start, pattern))
    return sorted(matches)


def karp_rabin_search(text, patterns, base=PRIME_BASE, modulus=PRIME_MODULUS, verify=True):
    """Hashed filter over the distinct pattern lengths; verify=True confirms every hit.

    Returns (matches, stats) with stats counting hashed windows, hash candidates and
    verification comparisons.
    """
    unique = sorted(set(patterns))
    if any(not pattern for pattern in unique):
        raise ValueError("empty patterns are not supported")
    hasher = RollingHash(base, modulus)
    window_codes = to_codes(text)
    by_length = {}
    for pattern in unique:
        by_length.setdefault(len(pattern), []).append(pattern)
    pattern_hashes = {}
    for length, group in by_length.items():
        pattern_hashes[length] = {}
        for pattern in group:
            pattern_hashes[length].setdefault(hasher.hash(to_codes(pattern)), []) \
                .append(pattern)
    matches = []
    hashed_windows = 0
    candidates = 0
    verifications = 0
    for length in sorted(by_length):
        for start, value in hasher.windows(window_codes, length):
            hashed_windows += 1
            for pattern in pattern_hashes[length].get(value, ()):
                candidates += 1
                if verify:
                    verifications += 1
                    if text[start:start + length] != pattern:
                        continue                       # hash collision, rejected
                matches.append((start, pattern))
    stats = {"hashed_windows": hashed_windows, "candidates": candidates,
             "verifications": verifications}
    return sorted(matches), stats


def expected_hashed_windows(text, patterns):
    lengths = sorted({len(pattern) for pattern in patterns})
    return sum(len(text) - length + 1 for length in lengths if length <= len(text))


def lcg_text(length, seed, alphabet):
    return "".join(str((seed := (seed * 75 + 74) % 65537) % alphabet) for _ in range(length))


def main():
    # Hand-checked example: "abracadabra" with the patterns abra, ca, dabra, a, missing.
    text = "abracadabra"
    patterns = ["abra", "ca", "dabra", "a", "missing", "abracadabra"]
    expected = [(0, "a"), (0, "abra"), (0, "abracadabra"), (3, "a"), (4, "ca"), (5, "a"),
                (6, "dabra"), (7, "a"), (7, "abra"), (10, "a")]
    assert brute_force_search(text, patterns) == expected
    assert AhoCorasick(patterns).search(text) == expected
    found, stats = karp_rabin_search(text, patterns)
    assert found == expected, found
    assert stats["hashed_windows"] == expected_hashed_windows(text, patterns)
    assert stats["verifications"] == stats["candidates"] == len(expected), stats

    # Collision demo: with base 10 and modulus 97 the windows "97" and "00" hash to the same
    # value (9 * 10 + 7 = 97 = 0 mod 97), so the unverified filter reports a false positive.
    demo_text = "9700"
    demo_patterns = ["00"]
    verified, verified_stats = karp_rabin_search(demo_text, demo_patterns, DEMO_BASE,
                                                 DEMO_MODULUS)
    unverified, unverified_stats = karp_rabin_search(demo_text, demo_patterns, DEMO_BASE,
                                                     DEMO_MODULUS, verify=False)
    assert verified == [(2, "00")], verified
    assert unverified == [(0, "00"), (2, "00")], unverified
    assert verified_stats["candidates"] == unverified_stats["candidates"] == 2
    assert verified_stats["verifications"] == 2 and unverified_stats["verifications"] == 0
    assert len(unverified) > len(verified), "the raw hash filter admits the collision"
    assert AhoCorasick(demo_patterns).search(demo_text) == verified
    assert brute_force_search(demo_text, demo_patterns) == verified

    # LCG texts: verified hashing, raw hashing, the automaton and the naive scan all agree.
    for length, alphabet, seed in ((60, 2, 11), (80, 3, 23), (120, 2, 37), (200, 4, 53),
                                   (150, 3, 71)):
        text = lcg_text(length, seed, alphabet)
        patterns = []
        for start in range(0, min(len(text) - 3, 40)):
            for size in (1, 2, 3, 4):
                patterns.append(text[start:start + size])
        patterns += ["9", "99", text[:5], text[-3:], "0000"]
        expected = brute_force_search(text, patterns)
        assert AhoCorasick(patterns).search(text) == expected, seed
        found, stats = karp_rabin_search(text, patterns)
        assert found == expected, (seed, len(found), len(expected))
        assert stats["hashed_windows"] == expected_hashed_windows(text, patterns)
        assert stats["verifications"] == stats["candidates"] == len(expected), stats
        raw, raw_stats = karp_rabin_search(text, patterns, verify=False)
        assert raw == expected, "with a 10^9 + 7 modulus no collision appears in these texts"
        assert raw_stats["candidates"] == len(expected)
        # Duplicate patterns must collapse: the answer is a set of (start, pattern) pairs.
        assert karp_rabin_search(text, patterns + patterns)[0] == expected

    # Edge cases: no patterns, an empty text, and patterns longer than the whole text.
    assert brute_force_search("abc", []) == []
    empty_text, empty_stats = karp_rabin_search("abc", [])
    assert empty_text == [] and empty_stats == {"hashed_windows": 0, "candidates": 0,
                                                "verifications": 0}
    none_of_them, none_stats = karp_rabin_search("abc", ["abcd", "zz"])
    assert none_of_them == [] and none_stats["candidates"] == 0
    assert none_stats["hashed_windows"] == 2, "only 'zz' length 2 is hashable in 'abc'"
    text, stats = karp_rabin_search("", ["a"])
    assert text == [] and stats["hashed_windows"] == 0
    assert AhoCorasick(["a"]).search("") == []
    exactly, _ = karp_rabin_search("abc", ["abc"])
    assert exactly == [(0, "abc")]
    # A pattern equal to the text with a one-character alphabet, and unary overlap counting.
    unary, unary_stats = karp_rabin_search("aaaa", ["a", "aa", "aaa", "aaaa"])
    assert unary == [(0, "a"), (0, "aa"), (0, "aaa"), (0, "aaaa"), (1, "a"), (1, "aa"),
                     (1, "aaa"), (2, "a"), (2, "aa"), (3, "a")], unary
    assert unary_stats["hashed_windows"] == 4 + 3 + 2 + 1
    assert unary_stats["verifications"] == len(unary) == 10

    # Errors and determinism.
    for bad in (lambda: AhoCorasick([""]), lambda: karp_rabin_search("abc", ["a", ""])):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("empty patterns must raise ValueError")
    text = lcg_text(100, 97, 3)
    patterns = [text[7:10], text[20:22], text[50:54]]
    first = karp_rabin_search(text, patterns)
    second = karp_rabin_search(text, patterns)
    assert first == second
    assert AhoCorasick(patterns).search(text) == first[0]
    assert brute_force_search(text, patterns) == first[0]


if __name__ == "__main__":
    main()
    print("PASS")
