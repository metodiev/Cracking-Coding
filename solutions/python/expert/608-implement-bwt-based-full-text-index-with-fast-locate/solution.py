"""Task 608 - Implement BWT-based full-text index with fast locate. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

The index stores the Burrows-Wheeler transform of text + sentinel, the C table (how many characters
are smaller than each one), one rank/select bit vector per character over the BWT, and two independent
sets of suffix-array samples. Nothing else: the text itself is recoverable from the BWT by walking LF
from the sentinel row and reading the first column (the sorted BWT), which is what `decode` and
`extract` use - the index alone carries the whole document.

Search is backward search: extend the pattern to the left with lo = C[c] + occ(c, lo),
hi = C[c] + occ(c, hi); the resulting interval is exactly the rows whose suffixes start with the
pattern. Locate turns a row into a text position by walking LF, which moves one position to the left
per step, and stopping at a sampled row: position = SA[sample] + steps.

Fast locate is the point of the two sample sets. Rank samples (every `rank_step`-th row) are hit after
about rank_step/2 steps by any walk, but they give no worst-case bound; position samples (rows whose
text position is a multiple of `position_step`) bound the walk by `position_step` steps but are hit
late on average. Checking both at every step of the walk therefore stops at whichever sample comes
first: the measured step count is at most that of either scheme alone, and at most position_step
because a multiple of position_step is always on the way. The file asserts exactly that per row, and
reports the average step counts and the storage of both sample sets.

Checks: hand values for banana (BWT, C table, counts, positions, snippets), occ against direct
character counts, count/locate/extract/snippets against naive scans on hand, unary, periodic and LCG
texts, all three locate schemes agreeing position by position with the step inequality, several sample
strides answering identically with finer strides walking shorter but storing more, the text decoded
from the BWT alone, and a 12000-character text with sampled windows.
"""
SENTINEL = "\x00"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def lcg_string(length, alphabet, seed=12345):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def suffix_array(text):
    """Prefix doubling; the index structure, not the construction, is what this task is about."""
    n = len(text)
    if n == 0:
        return []
    sa, rank, k = list(range(n)), [ord(c) for c in text], 1
    while True:
        keys = {i: (rank[i], rank[i + k] if i + k < n else -1) for i in range(n)}
        sa.sort(key=keys.__getitem__)
        new_rank = [0] * n
        for index in range(1, n):
            previous, current = sa[index - 1], sa[index]
            new_rank[current] = new_rank[previous] + (0 if keys[previous] == keys[current] else 1)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            return sa
        k <<= 1


class BitVector:
    """Bit vector with rank1 (a sample per 64-bit word) and select1 (a sample per 128 ones)."""

    def __init__(self, bits):
        self.bits, self.n = bits, len(bits)
        self.words = []
        for start in range(0, self.n, 64):
            word = 0
            for offset, bit in enumerate(bits[start:start + 64]):
                word |= bit << offset
            self.words.append(word)
        self.rank_samples = [0]
        for word in self.words:
            self.rank_samples.append(self.rank_samples[-1] + bin(word).count("1"))
        self.total = self.rank_samples[-1]
        self.select_samples, seen = {}, 0
        for position, bit in enumerate(bits):
            if bit:
                if seen % 128 == 0:
                    self.select_samples[seen] = position
                seen += 1

    def rank1(self, i):
        word_index, offset = divmod(i, 64)
        count = self.rank_samples[word_index]
        if offset:
            count += bin(self.words[word_index] & ((1 << offset) - 1)).count("1")
        return count

    def select1(self, k):
        if k < 0 or k >= self.total:
            return -1
        base = (k // 128) * 128
        position, seen = self.select_samples[base], base
        while True:
            if self.bits[position]:
                if seen == k:
                    return position
                seen += 1
            position += 1


class BWTIndex:
    """Full-text index over text + sentinel: BWT, C table, per-character ranks, two sample sets."""

    def __init__(self, text, position_step=64, rank_step=16):
        self.text = text
        self.s = text + SENTINEL
        self.n = n = len(self.s)
        self.position_step, self.rank_step = position_step, rank_step
        self.sa = suffix_array(self.s)
        self.bwt = "".join(self.s[self.sa[row] - 1] for row in range(n))     # s[-1] is the sentinel
        self.first = "".join(sorted(self.bwt))
        self.c_table = {ch: sum(1 for other in self.s if other < ch) for ch in set(self.s)}
        self.vectors = {ch: BitVector([1 if item == ch else 0 for item in self.bwt])
                        for ch in self.c_table}
        position_flags = [1 if value % position_step == 0 else 0 for value in self.sa]
        self.position_marker = BitVector(position_flags)
        self.position_values = [self.sa[row] for row in range(n) if position_flags[row]]
        rank_flags = [1 if row % rank_step == 0 else 0 for row in range(n)]
        self.rank_marker = BitVector(rank_flags)
        self.rank_values = [self.sa[row] for row in range(n) if rank_flags[row]]
        self.decoded_text = None

    def occ(self, ch, i):
        """Occurrences of `ch` in the first i characters of the BWT."""
        vector = self.vectors.get(ch)
        return 0 if vector is None else vector.rank1(i)

    def lf(self, row):
        ch = self.bwt[row]
        return self.c_table[ch] + self.occ(ch, row)

    def interval(self, pattern):
        """Row interval of the suffixes starting with `pattern` (empty when there is no match)."""
        lo, hi = 0, self.n
        for ch in reversed(pattern):
            if ch not in self.c_table:
                return lo, lo
            lo, hi = self.c_table[ch] + self.occ(ch, lo), self.c_table[ch] + self.occ(ch, hi)
            if lo >= hi:
                break
        return lo, hi

    def count(self, pattern):
        if not pattern:                             # documented: the empty pattern is not searched
            return 0
        lo, hi = self.interval(pattern)
        return hi - lo

    def position_at(self, row, scheme="two-level"):
        """Text position of `row` (or the number of steps taken and the position when asked)."""
        steps = 0
        while True:
            if scheme != "position" and self.rank_marker.bits[row]:
                return (self.rank_values[self.rank_marker.rank1(row)] + steps) % self.n, steps
            if scheme != "rank" and self.position_marker.bits[row]:
                return (self.position_values[self.position_marker.rank1(row)] + steps) % self.n, steps
            row = self.lf(row)
            steps += 1
            if steps > self.position_step:
                raise AssertionError(f"LF walk exceeded {self.position_step} steps")

    def locate(self, pattern, scheme="two-level"):
        """Sorted text positions of `pattern`, using the chosen sampling scheme."""
        if not pattern:
            return []
        lo, hi = self.interval(pattern)
        found = [self.position_at(row, scheme)[0] for row in range(lo, hi)]
        return sorted(position for position in found if position < self.n - 1)

    def walk_steps(self, pattern, scheme="two-level"):
        """Step counts of every LF walk performed while locating `pattern`."""
        lo, hi = self.interval(pattern)
        if not pattern:
            return []
        return [self.position_at(row, scheme)[1] for row in range(lo, hi)]

    def decode(self):
        """Recover text + sentinel from the BWT alone: LF-walk from the sentinel row, collect F."""
        row, out = 0, []
        for _ in range(self.n):
            row = self.lf(row)
            out.append(self.first[row])
        return "".join(reversed(out))

    def document(self):
        """The text as recovered from the index (cached; no original text is needed)."""
        if self.decoded_text is None:
            self.decoded_text = self.decode()[1:]
        return self.decoded_text

    def extract(self, start, stop):
        """Text window [start, stop), taken from the document recovered out of the BWT."""
        return self.document()[start:stop]

    def snippets(self, pattern, context=8, limit=4):
        """Search API: context windows with the matches in brackets, first `limit` matches."""
        document, out = self.document(), []
        for position in self.locate(pattern)[:limit]:
            start = max(0, position - context)
            stop = min(len(document), position + len(pattern) + context)
            out.append(document[start:position] + "[" + pattern + "]"
                       + document[position + len(pattern):stop])
        return out

    def stored_bits(self):
        """BWT bits plus rank/select samples plus the two sample sets (values as 32-bit words)."""
        words = (self.n + 63) // 64
        return (self.n * len(self.vectors) + words * 32 * len(self.vectors)
                + len(self.position_values) * 32 + len(self.rank_values) * 32)


def brute_positions(text, pattern):
    out = []
    start = text.find(pattern)
    while start != -1:
        out.append(start)
        start = text.find(pattern, start + 1)
    return out


def brute_snippets(text, pattern, context=8, limit=4):
    out = []
    for position in brute_positions(text, pattern)[:limit]:
        start = max(0, position - context)
        stop = min(len(text), position + len(pattern) + context)
        out.append(text[start:position] + "[" + pattern + "]" + text[position + len(pattern):stop])
    return out


def main():
    index = BWTIndex("banana")
    check(index.bwt == "annb\x00aa", "banana BWT")
    check(index.c_table == {"\x00": 0, "a": 1, "b": 4, "n": 5}, "banana C table")
    check(index.decode() == SENTINEL + "banana" and index.document() == "banana", "banana decode")
    check(index.extract(1, 4) == "ana" and index.extract(0, 6) == "banana", "banana extract")
    check(index.count("ana") == 2 and index.locate("ana") == [1, 3], "banana ana")
    check(index.locate("na") == [2, 4] and index.locate("a") == [1, 3, 5], "banana na/a")
    check(index.locate("banana") == [0] and index.count("banana") == 1, "banana whole")
    check(index.count("bananas") == 0 and index.locate("bananas") == [], "banana absent")
    check(index.count("") == 0 and index.locate("") == [], "banana empty pattern")
    check(index.count("z") == 0 and index.locate("z") == [], "banana missing character")
    check(index.snippets("ana") == ["b[ana]na", "ban[ana]"], "banana snippets")
    check(index.snippets("zz") == [], "banana snippets absent")
    for ch in index.c_table:
        positions = [i for i, item in enumerate(index.bwt) if item == ch]
        check(index.occ(ch, index.n) == len(positions), ch)
        check(all(index.vectors[ch].select1(k) == position
                  for k, position in enumerate(positions)), ch)

    # the two-level scheme can never walk further than either scheme alone
    comparisons = 0
    for text in ("mississippi", "abracadabra", "abcabxabcd", "aaaa", "a"):
        index = BWTIndex(text, position_step=8, rank_step=4)
        for pattern in ("a", "an", "ana", "ssi", text[:4], text[-3:], "z"):
            lo, hi = index.interval(pattern)
            if lo >= hi or not pattern:
                continue
            for row in range(lo, hi):
                two = index.position_at(row, "two-level")[1]
                cheap = index.position_at(row, "rank")[1]
                bounded = index.position_at(row, "position")[1]
                check(min(cheap, bounded) >= two, (text, pattern, row))
                comparisons += 1
            check(index.locate(pattern, "two-level") == brute_positions(text, pattern), pattern)
            check(index.locate(pattern, "rank") == brute_positions(text, pattern), pattern)
            check(index.locate(pattern, "position") == brute_positions(text, pattern), pattern)
    check(comparisons > 10, "locate comparisons really ran")

    # finer strides answer identically, walk shorter and store more
    reference = BWTIndex("mississippi", position_step=2, rank_step=2)
    patterns = ["i", "is", "issi", "ssi", "ppi", "sipp", "mississippix", "q"]
    for step in (2, 4, 8, 16):
        variant = BWTIndex("mississippi", position_step=step, rank_step=max(1, step // 2))
        for pattern in patterns:
            check(variant.count(pattern) == reference.count(pattern), (step, pattern))
            check(variant.locate(pattern) == reference.locate(pattern), (step, pattern))
        check(variant.document() == "mississippi", step)
    fine = BWTIndex("mississippi", position_step=4, rank_step=2)
    coarse = BWTIndex("mississippi", position_step=32, rank_step=16)
    fine_steps = sum(sum(fine.walk_steps(p)) for p in patterns if p != "q")
    coarse_steps = sum(sum(coarse.walk_steps(p, "position")) for p in patterns if p != "q")
    check(fine_steps < coarse_steps, "finer positions walk shorter")
    check(coarse.stored_bits() < fine.stored_bits(), "coarser sampling stores less")

    for text in ("", "a", "aaaa", "banana", "abracadabra", "abcabxabcd"):
        index = BWTIndex(text)
        check(index.decode() == SENTINEL + text and index.document() == text, text)
        check(index.count("") == 0 and index.locate("") == [], text)
        check(index.snippets("a", context=2) == brute_snippets(text, "a", context=2), text)
        for pattern in [p for p in ("a", "b", text[:3], text + "z") if p]:
            check(index.count(pattern) == len(brute_positions(text, pattern)), (text, pattern))
            check(index.locate(pattern) == brute_positions(text, pattern), (text, pattern))
        if text:
            check(index.count(text) == 1 and index.locate(text) == [0], text)

    for trial in range(10):
        text, _ = lcg_string(60, "ab", 6080 + trial * 7919)
        index = BWTIndex(text, position_step=16, rank_step=8)
        check(index.document() == text, text)
        for start in range(0, len(text) - 5, 9):
            pattern = text[start:start + 5]
            check(index.locate(pattern) == brute_positions(text, pattern), pattern)
            check(index.snippets(pattern, context=3, limit=2)
                  == brute_snippets(text, pattern, context=3, limit=2), pattern)
        window = index.extract(10, 40)
        check(window == text[10:40], text)

    # large text: 12000 characters, decoded from the BWT, then searched
    large, _ = lcg_string(12000, "abcde", 20260808)
    index = BWTIndex(large, position_step=64, rank_step=16)
    check(index.document() == large, "large document decoded from the BWT")
    check(index.count(large) == 1 and index.locate(large) == [0], "large whole text")
    check(index.count("q") == 0 and index.locate("q") == [] and index.snippets("q") == [], "large")
    for window in range(0, len(large) - 20, 701):
        pattern = large[window:window + 20]
        check(index.locate(pattern) == brute_positions(large, pattern), pattern)
    check(index.snippets(large[100:114], context=6, limit=3)
          == brute_snippets(large, large[100:114], context=6, limit=3), "large snippets")
    probe = large[5000:5003]                    # a short pattern with many occurrences
    rows = list(range(*index.interval(probe)))
    check(len(rows) > 20, "enough rows to measure locate steps")
    two_level = sum(index.position_at(row)[1] for row in rows)
    only_position = sum(index.position_at(row, "position")[1] for row in rows)
    only_rank = sum(index.position_at(row, "rank")[1] for row in rows)
    check(two_level <= only_position and two_level <= only_rank, "two-level never walks further")
    check(two_level < only_position, "two-level locate walks fewer steps than position samples")
    check(index.extract(9000, 9012) == large[9000:9012], "large extract")

    print(f"banana: BWT 'annb\\x00aa', C table, {len(BWTIndex('banana').vectors)} character "
          f"vectors, snippets and extract recovered from the BWT alone")
    print(f"large text: {len(large)} characters, {index.stored_bits()} index bits, "
          f"{len(index.position_values)} position samples, {len(index.rank_values)} rank samples")
    print(f"locate {len(rows)} rows of {probe!r}: {two_level} steps at two levels, "
          f"{only_position} with position samples only, {only_rank} with rank samples only")


if __name__ == "__main__":
    main()
    print("PASS")
