"""Task 605 - Implement FM-index full search with rank/select structures. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

The FM-index is the BWT plus the two tables that make it searchable without the suffix array:
  * C[c] = the number of characters of text + sentinel that are smaller than c;
  * for every character c one bit vector over the BWT with one bit per row, so that occ(c, i) is
    rank1 of that vector at i. The vectors carry the rank/select structures themselves: rank1 adds a
    sample per 64-bit word (plus a popcount of the partial word), and select1 keeps one sampled
    position per 128 ones, so a select costs a short scan instead of a full scan.
Backward search starts from the full row range [0, n) and extends the pattern to the left with
lo = C[c] + occ(c, lo), hi = C[c] + occ(c, hi); the interval is the contiguous range of rows whose
suffixes start with the pattern, so its width is the occurrence count and an empty interval means "no
match". Locate walks LF (LF(i) = C[L[i]] + rank of L[i] among the earlier equal characters), which
moves one position to the left in the text, until a row marked by the sampled suffix array is hit;
sampling by text position (`step` divides the position) bounds the walk by `step` steps.

"Full search" here means the three search modes built on that primitive, all cross-checked against
naive scans: exact count/locate, count of occurrences with at most one substitution (exact matches
plus, for each position and each other character, the exact matches of the substituted pattern - each
such occurrence has its single mismatch in exactly that position, so nothing is counted twice), and
the longest prefix of a query that occurs in the text. text is recovered from the BWT alone by
walking LF from the sentinel row and reading the first column F (= the sorted BWT), which proves the
transform carries the whole text. Checks: hand values for banana, occ/rank/select against direct bit
counting, LF being one permutation cycle, all three search modes against brute force on hand texts,
LCG texts and a 12000-character text, boundary cases (empty pattern, longer pattern, absent
character), and several rank sample widths giving identical answers.
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
    """Prefix doubling; the rank/select structures are what this task is about, not construction."""
    n = len(text)
    if n == 0:
        return []
    sa, rank, k = list(range(n)), [ord(c) for c in text], 1
    while True:
        keys = {i: (rank[i], rank[i + k] if i + k < n else -1) for i in range(n)}
        sa.sort(key=keys.__getitem__)
        new_rank = [0] * n
        for idx in range(1, n):
            prev, cur = sa[idx - 1], sa[idx]
            new_rank[cur] = new_rank[prev] + (0 if keys[prev] == keys[cur] else 1)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            return sa
        k <<= 1


class BitVector:
    """Bit vector with rank1 (sampled every `sample_words` words) and select1 (every 128 ones)."""

    def __init__(self, bits, sample_words=1):
        self.bits, self.n = bits, len(bits)
        self.words = []
        for start in range(0, self.n, 64):
            word = 0
            for offset, bit in enumerate(bits[start:start + 64]):
                word |= bit << offset
            self.words.append(word)
        self.sample_words = sample_words
        self.rank_samples = [0]
        for index, word in enumerate(self.words):
            if index % sample_words == sample_words - 1:
                self.rank_samples.append(self.rank_samples[-1] + bin(word).count("1"))
        self.total = sum(bin(word).count("1") for word in self.words)
        self.select_samples, seen = {}, 0
        for position, bit in enumerate(bits):
            if bit:
                if seen % 128 == 0:
                    self.select_samples[seen] = position
                seen += 1

    def rank1(self, i):
        """Number of ones in [0, i): one sample plus at most 64 * sample_words popcounts."""
        block = i // (64 * self.sample_words)
        count = self.rank_samples[block]
        for index in range(block * 64 * self.sample_words, min(i, self.n)):
            count += self.bits[index]
        return count

    def select1(self, k):
        """Position of the k-th one (0-based), or -1 when there is none."""
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


class FMIndex:
    """FM-index over text + sentinel: BWT, C table, per-character rank/select and sampled SA rows."""

    def __init__(self, text, step=8, rank_sample_words=1):
        self.text = text
        self.s = text + SENTINEL
        self.n = n = len(self.s)
        self.sa = suffix_array(self.s)
        self.bwt = "".join(self.s[self.sa[row] - 1] for row in range(n))     # s[-1] is the sentinel
        self.first = "".join(sorted(self.bwt))               # the first column F, no SA needed
        self.alphabet = sorted(set(self.s))
        # the sentinel is not a substitutable character: a variant containing it would match a
        # window running past the end of the text, which is not an occurrence in the text
        self.substitutions = [ch for ch in self.alphabet if ch != SENTINEL]
        self.c_table = {ch: sum(1 for other in self.s if other < ch) for ch in self.alphabet}
        self.vectors = {ch: BitVector([1 if item == ch else 0 for item in self.bwt], rank_sample_words)
                        for ch in self.alphabet}
        self.step = step
        flags = [1 if value % step == 0 else 0 for value in self.sa]
        self.sampled_rows = BitVector(flags)
        self.sampled_values = [self.sa[row] for row in range(n) if flags[row]]
        self.walk_steps = 0

    def occ(self, ch, i):
        """Occurrences of `ch` in the first i characters of the BWT."""
        vector = self.vectors.get(ch)
        return 0 if vector is None else vector.rank1(i)

    def select(self, row, k):
        """Position of the (k+1)-th occurrence of the BWT character at `row` (0-based)."""
        return self.vectors[self.bwt[row]].select1(k)

    def lf(self, row):
        """Row of the suffix starting one position to the left (the classic LF mapping)."""
        ch = self.bwt[row]
        return self.c_table[ch] + self.occ(ch, row)

    def suffix_at(self, row):
        """Text position of the suffix in `row`, by walking LF to a sampled row (bounded)."""
        steps = 0
        while not self.sampled_rows.bits[row]:
            row = self.lf(row)
            steps += 1
            if steps > self.step:
                raise AssertionError(f"LF walk from row {row} exceeded {self.step} steps")
        self.walk_steps += steps
        value = self.sampled_values[self.sampled_rows.rank1(row)]
        return (value + steps) % self.n         # LF moves one position to the left per step

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

    def locate(self, pattern):
        if not pattern:
            return []
        lo, hi = self.interval(pattern)
        positions = [self.suffix_at(row) for row in range(lo, hi)]
        return sorted(position for position in positions if position < self.n - 1)

    def count_approximate(self, pattern, alphabet=None):
        """Occurrences with at most one substitution, built from exact counts of the variants."""
        if not pattern:
            return 0
        total = self.count(pattern)
        for index in range(len(pattern)):
            for ch in (alphabet or self.substitutions):
                if ch == pattern[index]:
                    continue
                total += self.count(pattern[:index] + ch + pattern[index + 1:])
        return total

    def longest_prefix_match(self, pattern):
        """Length of the longest prefix of `pattern` that occurs in the text."""
        best = 0
        for length in range(1, len(pattern) + 1):
            if self.count(pattern[:length]) == 0:
                break
            best = length
        return best

    def decode(self):
        """Recover the text from the BWT alone: LF-walk from the sentinel row, collect F, reverse."""
        row, out = 0, []
        for _ in range(self.n):
            row = self.lf(row)
            out.append(self.first[row])
        return "".join(reversed(out))


def brute_positions(text, pattern):
    out = []
    start = text.find(pattern)
    while start != -1:
        out.append(start)
        start = text.find(pattern, start + 1)
    return out


def brute_approximate(text, pattern):
    count = 0
    for start in range(len(text) - len(pattern) + 1):
        if sum(1 for k in range(len(pattern)) if text[start + k] != pattern[k]) <= 1:
            count += 1
    return count


def brute_prefix_match(text, pattern):
    best = 0
    while best < len(pattern) and pattern[:best + 1] in text:
        best += 1
    return best


def main():
    index = FMIndex("banana")
    check(index.bwt == "annb\x00aa", "banana BWT")
    check(index.c_table == {"\x00": 0, "a": 1, "b": 4, "n": 5}, "banana C table")
    check(index.decode() == SENTINEL + "banana", "banana decode from LF alone")
    for ch in index.alphabet:
        check(all(index.occ(ch, i) == index.bwt[:i].count(ch) for i in range(index.n + 1)), ch)
        positions = [i for i, item in enumerate(index.bwt) if item == ch]
        check(all(index.vectors[ch].select1(k) == position
                  for k, position in enumerate(positions)), ch)
        check(index.vectors[ch].select1(len(positions)) == -1, ch)
    check(index.count("ana") == 2 and index.locate("ana") == [1, 3], "banana ana")
    check(index.count("na") == 2 and index.locate("na") == [2, 4], "banana na")
    check(index.count("a") == 3 and index.locate("a") == [1, 3, 5], "banana a")
    check(index.count("banana") == 1 and index.locate("banana") == [0], "banana whole")
    check(index.count("bananas") == 0 and index.locate("bananas") == [], "banana absent")
    check(index.count("") == 0 and index.locate("") == [], "banana empty pattern")
    check(index.count("z") == 0 and index.locate("z") == [], "banana missing character")
    check(index.count_approximate("anana") == 1, "banana one substitution")
    check(index.count_approximate("banan") == brute_approximate("banana", "banan"), "banana sub 2")
    check(index.longest_prefix_match("bananas") == 6, "banana longest prefix")
    check(index.longest_prefix_match("bananax") == 6, "banana longest prefix 2")
    check(index.longest_prefix_match("x") == 0, "banana longest prefix 0")
    rows = [index.lf(row) for row in range(index.n)]
    check(sorted(rows) == list(range(index.n)), "LF is a permutation")
    walked, row = 0, 0
    for _ in range(index.n):                        # LF is a single cycle for a unique sentinel
        row = index.lf(row)
        walked += 1
    check(row == 0 and walked == index.n, "LF cycle length")
    check(all(index.vectors[ch].rank1(index.n) == index.bwt.count(ch)
              for ch in index.alphabet), "rank1 totals")

    for text in ("", "a", "aaaa", "mississippi", "abracadabra", "abcabxabcd"):
        index = FMIndex(text, step=2)
        check(index.decode() == SENTINEL + text, text)
        check(index.count("") == 0 and index.locate("") == [], text)
        patterns = [p for p in ("a", "b", "zz", text[:3], text + "z") if p]
        for pattern in patterns:
            check(index.count(pattern) == len(brute_positions(text, pattern)), (text, pattern))
            check(index.locate(pattern) == brute_positions(text, pattern), (text, pattern))
        if text:
            check(index.count(text) == 1 and index.locate(text) == [0], text)

    # rank sample widths must not change any answer
    reference = FMIndex("mississippi", rank_sample_words=1)
    for width in (1, 2, 4, 8):
        variant = FMIndex("mississippi", rank_sample_words=width)
        for pattern in ("i", "issi", "ssi", "ppi", "mississippix", "q"):
            check(variant.count(pattern) == reference.count(pattern), (width, pattern))
            check(variant.locate(pattern) == reference.locate(pattern), (width, pattern))
        for ch in variant.alphabet:
            check(all(variant.occ(ch, i) == reference.occ(ch, i)
                      for i in range(variant.n + 1)), (width, ch))

    rejected = 0
    for trial in range(12):
        text, _ = lcg_string(40, "ab", 6050 + trial * 7919)
        index = FMIndex(text, step=3)
        check(index.decode() == SENTINEL + text, text)
        check(index.locate(text) == brute_positions(text, text), text)
        for start in range(0, len(text) - 6, 7):
            pattern = text[start:start + 6]
            check(index.count(pattern) == len(brute_positions(text, pattern)), pattern)
            check(index.locate(pattern) == brute_positions(text, pattern), pattern)
            mutated = pattern[:3] + ("a" if pattern[3] == "b" else "b") + pattern[4:]
            check(index.count_approximate(mutated) == brute_approximate(text, mutated), mutated)
            check(index.longest_prefix_match(mutated) == brute_prefix_match(text, mutated), mutated)
            rejected += 1
    check(rejected > 0, "mismatch searches really ran")
    for probe in ("babab", "ababa", "b" * 6 + "a", "z"):
        index = FMIndex("ababababab", step=2)
        check(index.count_approximate(probe) == brute_approximate("ababababab", probe), probe)
        check(index.longest_prefix_match(probe) == brute_prefix_match("ababababab", probe), probe)

    # large text: 12000 characters; sampled windows of every search mode
    large, _ = lcg_string(12000, "abcde", 20260505)
    index = FMIndex(large, step=13)
    check(index.n == len(large) + 1 and index.decode() == SENTINEL + large, "large decode")
    check(index.count(large) == 1 and index.locate(large) == [0], "large whole text")
    for window in range(0, len(large) - 16, 907):
        pattern = large[window:window + 16]
        check(index.count(pattern) == len(brute_positions(large, pattern)), pattern)
        check(index.locate(pattern) == brute_positions(large, pattern), pattern)
    for probe in (large[100:120], "z" * 10, large[:6] + "z" + large[7:26]):
        check(index.count_approximate(probe) == brute_approximate(large, probe), probe[:8])
        check(index.longest_prefix_match(probe) == brute_prefix_match(large, probe), probe[:8])
    check(index.count("q") == 0 and index.locate("q") == [], "large missing character")

    print(f"large text: {len(large)} characters, {index.n} BWT rows, {len(index.vectors)} character "
          f"bit vectors, locate walks bounded by {index.step} steps")
    print(f"banana: LF is a single cycle over {FMIndex('banana').n} rows and every search mode "
          f"agrees with the naive scan")


if __name__ == "__main__":
    main()
    print("PASS")
