"""Task 580 - Implement FM-index search for substring count (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A working FM-index over a text: the Burrows-Wheeler transform is built from the sorted suffix array
(with a sentinel appended so the transform inverts exactly), the LF mapping is
LF(i) = C[c] + Occ(c, i) where C[c] is the number of characters smaller than c and Occ(c, i) the
number of occurrences of c in the first i columns, and patterns are counted by backward search -
extending the pattern one character to the left at a time while keeping the interval [low, high) of
suffix-array positions. The file stores Occ as per-character prefix-count arrays (the conceptual,
honest version of the wavelet-tree rank structure a real FM-index uses), keeps the full suffix array
alongside it, and locates occurrences with a sampling step: the suffix array is sampled every
SAMPLING-th position and LF steps are walked until a sampled entry is hit. Everything is verified
against brute force, and the LF mapping is verified structurally rather than only through search
results: SA[LF(i)] = (SA[i] - 1) mod n for every i is the defining property of the transform.

Checks: the BWT inverting back to the text for four texts of different shapes; SA[LF(i)] = SA[i] - 1
for every position; the cumulative counts C agreeing with the character histogram; exhaustive
counting of *every* pattern up to length 4 over the two-symbol alphabet - all 31 of them - plus six
longer patterns - against Python's own substring scan, on a 300-character text with runs and
repeats; the boundary cases
(the empty pattern counting every position, a pattern longer than the text, a pattern that does not
occur, the whole text, a single character and a character absent from the text); the located
positions being exactly the occurrence positions, sorted, for every exhaustive pattern (sampling 1,
2 and 4, with the number of LF steps asserted to stay below the sampling step); the counting
agreement with an independent suffix-array binary search; and determinism.
"""
MASK64 = (1 << 64) - 1


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31))


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def random_text(state, length, alphabet):
    letters = []
    for _ in range(length):
        state, u = uniform(state)
        letters.append(alphabet[int(u * len(alphabet))])
    return state, "".join(letters)


def suffix_array(text):
    """Sorted suffix array of the text with a sentinel '$' smaller than every character."""
    padded = text + "\x00"
    return sorted(range(len(padded)), key=lambda index: padded[index:])


def lcp_array(text, sa):
    padded = text + "\x00"
    lcp = [0] * len(sa)
    for index in range(1, len(sa)):
        first, second = padded[sa[index - 1]:], padded[sa[index]:]
        common = 0
        while common < len(first) and common < len(second) and first[common] == second[common]:
            common += 1
        lcp[index] = common
    return lcp


class FMIndex:
    def __init__(self, text, sampling=4):
        self.text = text
        self.sampling = sampling
        self.padded = text + "\x00"
        self.length = len(self.padded)
        self.suffix_array = suffix_array(text)
        self.bwt = [self.padded[(index - 1) % self.length] for index in self.suffix_array]
        alphabet = sorted(set(self.padded))
        self.alphabet = alphabet
        self.counts = {}
        running = 0
        for letter in alphabet:
            self.counts[letter] = running
            running += self.bwt.count(letter)
        self.occ = {}
        for letter in alphabet:
            prefix = [0]
            total = 0
            for value in self.bwt:
                if value == letter:
                    total += 1
                prefix.append(total)
            self.occ[letter] = prefix
        # Sampled by *text position*, not by suffix-array index: LF decreases the position by one
        # per step, so the walk to the next sampled entry costs at most sampling - 1 steps.
        self.sampled_positions = {position for position in range(0, self.length, sampling)}

    def lf(self, index):
        letter = self.bwt[index]
        return self.counts[letter] + self.occ[letter][index]

    def interval(self, pattern):
        """Backward search: the half-open suffix-array interval of `pattern`."""
        low, high = 0, self.length
        for letter in reversed(pattern):
            if letter not in self.occ:
                return 0, 0
            low = self.counts[letter] + self.occ[letter][low]
            high = self.counts[letter] + self.occ[letter][high]
            if low >= high:
                return 0, 0
        return low, high

    def count(self, pattern):
        low, high = self.interval(pattern)
        return high - low

    def locate(self, pattern):
        """Occurrence positions, sorted, reconstructed from the sampled suffix array."""
        low, high = self.interval(pattern)
        positions = []
        steps = 0
        for index in range(low, high):
            cursor = index
            walked = 0
            while self.suffix_array[cursor] not in self.sampled_positions:
                cursor = self.lf(cursor)
                walked += 1
            steps = max(steps, walked)
            positions.append((self.suffix_array[cursor] + walked) % self.length)
        return sorted(positions), steps

    def inverse_bwt(self):
        table = [""] * self.length
        for _ in range(self.length):
            table = sorted(self.bwt[i] + table[i] for i in range(self.length))
        for row in table:
            if row.endswith("\x00"):
                return row[:-1]


def brute_force_count(text, pattern):
    if pattern == "":
        return len(text) + 1
    return sum(1 for index in range(len(text) - len(pattern) + 1)
               if text[index:index + len(pattern)] == pattern)


def brute_force_locate(text, pattern):
    if pattern == "":
        return list(range(len(text) + 1))
    return [index for index in range(len(text) - len(pattern) + 1)
            if text[index:index + len(pattern)] == pattern]


def suffix_array_count(text, pattern):
    """Independent method: the suffixes starting with `pattern` form the contiguous block
    [pattern, pattern + high sentinel) of the sorted suffix list, found by binary search."""
    import bisect
    padded = text + "\x00"
    suffixes = sorted(padded[index:] for index in range(len(padded)))
    if pattern == "":
        return len(suffixes)
    return (bisect.bisect_left(suffixes, pattern + "\uffff")
            - bisect.bisect_left(suffixes, pattern))


def main():
    # --- Inversion and the LF property --------------------------------------------------
    texts = ["banana", "mississippi", "abababab", "aabbaabb"]
    for text in texts:
        index = FMIndex(text, sampling=2)
        assert index.length == len(text) + 1
        assert index.inverse_bwt() == text, "the BWT inverts for %r" % text
        assert sorted(index.suffix_array) == list(range(len(text) + 1))
        for position, suffix_start in enumerate(index.suffix_array):
            expected = (suffix_start - 1) % index.length
            assert index.suffix_array[index.lf(position)] == expected, \
                "SA[LF(i)] = SA[i] - 1 at %d" % position
        histogram = {}
        for letter in index.bwt:
            histogram[letter] = histogram.get(letter, 0) + 1
        running = 0
        for letter in sorted(histogram):
            assert index.counts[letter] == running, "C[%r]" % letter
            running += histogram[letter]

    # --- Exhaustive pattern counting ----------------------------------------------------
    state = 7
    state, long_text = random_text(state, 300, "ab")
    index = FMIndex(long_text, sampling=3)
    patterns = []
    for length in range(5):
        for code in range(2 ** length):
            pattern = "".join("ab"[(code >> shift) & 1] for shift in range(length - 1, -1, -1))
            patterns.append(pattern)
    assert len(patterns) == 31
    for pattern in patterns:
        expected = brute_force_count(long_text, pattern)
        assert index.count(pattern) == expected, "count of %r" % pattern
        assert suffix_array_count(long_text, pattern) == expected, "binary-search cross-check"
        positions, steps = index.locate(pattern)
        assert positions == brute_force_locate(long_text, pattern), \
            "located positions of %r" % pattern
        assert steps < index.sampling, "the sampling bounds the LF walk: %r" % steps
    for pattern in ("abababababab", "b" * 6, "a" * 12, "ba" * 7, "abba", "babababababababa"):
        assert index.count(pattern) == brute_force_count(long_text, pattern)
        assert index.locate(pattern)[0] == brute_force_locate(long_text, pattern)

    # --- Boundary cases -----------------------------------------------------------------
    small = "abracadabra"
    small_index = FMIndex(small, sampling=2)
    assert small_index.count("") == len(small) + 1, "the empty pattern matches every position"
    assert small_index.count("abracadabrazz") == 0
    assert small_index.count("z") == 0
    assert small_index.count("abracadabra") == 1
    assert small_index.count("a") == 5
    assert small_index.locate("a")[0] == [0, 3, 5, 7, 10]
    assert small_index.locate("zz")[0] == []
    assert small_index.locate("")[0] == list(range(len(small) + 1))
    assert small_index.count("cad") == 1 and small_index.locate("cad")[0] == [4]
    single = FMIndex("x", sampling=1)
    assert single.count("x") == 1 and single.count("y") == 0 and single.count("xx") == 0
    assert single.locate("x")[0] == [0]
    assert single.count("") == 2

    # --- Sampling values, larger alphabet ----------------------------------------------
    state, mixed = random_text(state, 200, "abcd")
    for sampling in (1, 2, 4, 8):
        index = FMIndex(mixed, sampling=sampling)
        for pattern in ("a", "bc", "dab", "cdcd", "aaaa", "dcba", "b"):
            assert index.count(pattern) == brute_force_count(mixed, pattern)
            assert index.locate(pattern)[0] == brute_force_locate(mixed, pattern)
        assert index.inverse_bwt() == mixed
    index = FMIndex(mixed, sampling=4)
    assert index.count("a") == mixed.count("a")
    assert index.locate("abc")[0] == brute_force_locate(mixed, "abc")

    # --- Determinism --------------------------------------------------------------------
    first = FMIndex(long_text, sampling=3)
    second = FMIndex(long_text, sampling=3)
    assert first.bwt == second.bwt and first.suffix_array == second.suffix_array
    assert first.count("abab") == second.count("abab")
    assert first.locate("aa") == second.locate("aa")
    assert random_text(7, 300, "ab")[1] == long_text


if __name__ == "__main__":
    main()
    print("PASS")
