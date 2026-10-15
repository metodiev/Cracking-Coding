"""Task 410 - FM-index: basic concept and small demo. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Structure for text s: build t = s + "$" with a unique smallest sentinel, store
  * the Burrows-Wheeler last column L of t,
  * the C table: C[c] = number of characters of t strictly smaller than c,
  * rank samples: for every character, the cumulative count in L at each block boundary
    (block size configurable); occ(c, i) = sample + direct scan of the residual (< block chars).

Backward search walks the pattern from the right: starting from the full row range [0, n+1),
for each character c set lo = C[c] + occ(c, lo) and hi = C[c] + occ(c, hi). The interval is
the contiguous range of rows whose rotations start with the pattern suffix processed so far,
so the final width counts the matches; an empty interval means no match.

LF-mapping (the inverse of the sorted first column F): LF[i] = C[L[i]] + rank of L[i] among
equal characters before i. Following LF moves from a row to the rotation starting one position
earlier, so walking LF from a match row until row 0 (the "$..." rotation) gives the position:
after `steps` LF applications the match starts at (n + steps) mod (n + 1) in t.

Checks: hand values for banana ("ana" -> 2, "na" -> 2, "a" -> 3, locate positions [1,3] etc.),
occ() against direct counts for every boundary, LF is one cycle, LCG texts/pattern sets agree
with a naive scan for both count and locate, and several block sizes give identical answers.
"""

SENTINEL = "$"


def bwt_cyclic(s):
    n = len(s)
    if n == 0:
        return "", 0
    order = sorted(range(n), key=lambda i: s[i:] + s[:i])
    last = "".join(s[(i - 1) % n] for i in order)
    return last, order.index(0)


def lf_mapping(last):
    starts = {}
    total = 0
    for ch in sorted(set(last)):
        starts[ch] = total
        total += last.count(ch)
    seen = {}
    lf = [0] * len(last)
    for i, ch in enumerate(last):
        lf[i] = starts[ch] + seen.get(ch, 0)
        seen[ch] = seen.get(ch, 0) + 1
    return lf


class FMIndex:
    def __init__(self, text, block=4):
        self.text = text
        self.block = block
        self.t = text + SENTINEL
        self.last, self.primary = bwt_cyclic(self.t)
        self.c_table = {ch: sum(1 for x in self.t if x < ch) for ch in sorted(set(self.t))}
        self.samples = {}
        for ch in sorted(set(self.t)):
            row = []
            count = 0
            for pos in range(len(self.t) + 1):
                if pos % block == 0:
                    row.append(count)
                if pos < len(self.t) and self.last[pos] == ch:
                    count += 1
            self.samples[ch] = row
        self.lf = lf_mapping(self.last)

    def occ(self, ch, i):
        """Occurrences of ch in L[0:i), from the rank sample plus a short residual scan."""
        start = (i // self.block) * self.block
        count = self.samples[ch][i // self.block]
        for pos in range(start, i):
            if self.last[pos] == ch:
                count += 1
        return count

    def backward_search(self, pattern):
        lo, hi = 0, len(self.t)
        for ch in reversed(pattern):
            if ch not in self.c_table:
                return lo, lo
            lo = self.c_table[ch] + self.occ(ch, lo)
            hi = self.c_table[ch] + self.occ(ch, hi)
            if lo >= hi:
                return lo, hi
        return lo, hi

    def count(self, pattern):
        if not pattern:                      # documented: the empty pattern is not searched
            return 0
        lo, hi = self.backward_search(pattern)
        return hi - lo

    def locate(self, pattern):
        if not pattern:
            return []
        lo, hi = self.backward_search(pattern)
        positions = []
        for row in range(lo, hi):
            steps = 0
            while row != 0:                  # row 0 is the "$..." rotation
                row = self.lf[row]
                steps += 1
            positions.append((len(self.text) + steps) % len(self.t))
        return sorted(positions)


def naive_count(text, pattern):
    return sum(1 for i in range(len(text) - len(pattern) + 1) if text[i:i + len(pattern)] == pattern)


def naive_locate(text, pattern):
    return [i for i in range(len(text) - len(pattern) + 1) if text[i:i + len(pattern)] == pattern]


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def check_against_naive(text, patterns):
    index = FMIndex(text)
    for pattern in patterns:
        assert index.count(pattern) == naive_count(text, pattern), (text, pattern)
        assert index.locate(pattern) == naive_locate(text, pattern), (text, pattern)
    return index


def main():
    index = FMIndex("banana")
    assert index.t == "banana$"
    assert index.last == "annb$aa"
    assert index.c_table == {"$": 0, "a": 1, "b": 4, "n": 5}
    for ch in sorted(set(index.t)):
        for i in range(len(index.t) + 1):
            assert index.occ(ch, i) == index.last[:i].count(ch), (ch, i)

    assert index.count("ana") == 2 and index.count("na") == 2 and index.count("a") == 3
    assert index.count("banana") == 1 and index.count("nana") == 1 and index.count("anana") == 1
    assert index.count("bananas") == 0 and index.count("x") == 0 and index.count("") == 0
    assert index.locate("ana") == [1, 3]
    assert index.locate("na") == [2, 4]
    assert index.locate("a") == [1, 3, 5]
    assert index.locate("banana") == [0]
    assert index.locate("nana") == [2]
    assert index.locate("anana") == [1]
    assert index.locate("bananas") == [] and index.locate("x") == [] and index.locate("") == []
    assert sorted(index.lf) == list(range(len(index.lf)))
    row = 0
    for _ in range(len(index.t)):            # LF is a single cycle for a unique sentinel
        row = index.lf[row]
    assert row == 0

    # rank samples only affect speed: every block size must answer identically
    reference = FMIndex("banana", block=1)
    for block in (1, 2, 3, 5, 8, 16):
        variant = FMIndex("banana", block=block)
        for pattern in ["a", "an", "ana", "anan", "n", "na", "banana$".replace("$", ""), "zz"]:
            assert variant.count(pattern) == reference.count(pattern), block
            assert variant.locate(pattern) == reference.locate(pattern), block
        for ch in sorted(set(variant.t)):
            for i in range(len(variant.t) + 1):
                assert variant.occ(ch, i) == reference.occ(ch, i), (block, ch, i)

    check_against_naive("mississippi", ["i", "is", "issi", "ssi", "ppi", "sipp", "mississippix", "q"])
    check_against_naive("", ["a", "x"])
    check_against_naive("a", ["a", "aa", "b"])
    check_against_naive("aaaa", ["a", "aa", "aaa", "aaaa", "aaaaa", "b"])

    for trial in range(30):
        text, seed = lcg_string(30, "abc", 31000 + trial * 65537)
        patterns = []
        for i in range(len(text)):
            for j in range(i + 1, min(len(text), i + 4) + 1):
                patterns.append(text[i:j])
        patterns.extend([text, text[:-1], "d" + text[:5], "zz", "q"])
        check_against_naive(text, patterns)

    medium, _ = lcg_string(300, "abc", 424242)
    medium_index = FMIndex(medium, block=7)
    for pattern in ["abc", "cba", "cab", "abcabc", "zzz", medium[150:160], medium]:
        assert medium_index.count(pattern) == naive_count(medium, pattern)
        assert medium_index.locate(pattern) == naive_locate(medium, pattern)


if __name__ == "__main__":
    main()
    print("PASS")
