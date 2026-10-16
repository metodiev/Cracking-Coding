"""Task 581 - Implement compressed suffix array prototype (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A working compressed suffix array, not an outline. The suffix array is built by prefix doubling (sort
the suffixes by their first 2^k characters, k = 0, 1, 2, ...) and the LCP array by Kasai's linear
scan, both verified against brute force. The compression proper follows Grossi-Vitter: instead of the
full suffix array the file stores the psi array, psi[i] = the rank of the suffix that starts one
position to the right of the suffix at rank i, so that SA[psi[i]] = SA[i] + 1 (mod n), plus a sparse
sample of SA values. Any missing SA entry is recovered by walking psi until a sampled rank is reached:
SA[i] = SA[sampled] - steps (mod n), which costs at most the sampling step. That is the whole point -
n integers of psi plus about n/k sampled values replace the n entries of SA. The sampling is by text
position, which matters: a walk that tests only ranks would never terminate in bounded time, while
position-multiples of the sampling step are reached within sampling - 1 psi steps. The file measures
the stored-integer count and asserts both properties of psi (SA[psi[i]] = SA[i] + 1 and the inverse
relation SA[psi_inverse[i]] = SA[i] - 1) rather than trusting the search results.

Checks: the doubling suffix array and the Kasai LCP against brute-force sorted suffixes and pairwise
LCP scans on six texts (including a unary string, a two-symbol string and a random one); both psi
relations for every rank; reconstruction of the *complete* suffix array from psi plus samples for
sampling steps 1, 2, 4, 8 and 17, with the assertion that the recovered array equals the true one
element by element; the documented storage count (n psi entries + ceil(n/k) samples, which stays
below 2n while SA alone is n) and the walk length bounded by the sampling step; pattern counting and
locating through the reconstructed array (binary search on the suffix array) compared with brute
force for 40 patterns, plus the boundary patterns (empty, single character, absent, the whole text);
and determinism.
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


def suffix_array_doubling(text):
    """Prefix doubling: sort suffixes by their first 2^k characters until all ranks are distinct."""
    length = len(text)
    order = sorted(range(length), key=lambda index: text[index])
    rank = [0] * length
    for position in range(1, length):
        rank[order[position]] = rank[order[position - 1]]
        if text[order[position]] != text[order[position - 1]]:
            rank[order[position]] += 1
    step = 1
    while step < length:
        key = lambda index: (rank[index], rank[index + step] if index + step < length else -1)
        order = sorted(range(length), key=key)
        new_rank = [0] * length
        for position in range(1, length):
            new_rank[order[position]] = new_rank[order[position - 1]]
            if key(order[position]) != key(order[position - 1]):
                new_rank[order[position]] += 1
        rank = new_rank
        if rank[order[-1]] == length - 1:
            break
        step *= 2
    return order


def lcp_kasai(text, suffix_array):
    length = len(text)
    inverse = [0] * length
    for rank, index in enumerate(suffix_array):
        inverse[index] = rank
    lcp = [0] * length
    common = 0
    for index in range(length):
        if inverse[index] == 0:
            common = 0
            continue
        previous = suffix_array[inverse[index] - 1]
        while (index + common < length and previous + common < length
               and text[index + common] == text[previous + common]):
            common += 1
        lcp[inverse[index]] = common
        if common:
            common -= 1
    return lcp


def brute_force_suffix_array(text):
    return sorted(range(len(text)), key=lambda index: text[index:])


def brute_force_lcp(text, suffix_array):
    lcp = [0] * len(text)
    for rank in range(1, len(text)):
        first, second = text[suffix_array[rank - 1]:], text[suffix_array[rank]:]
        common = 0
        while common < len(first) and common < len(second) and first[common] == second[common]:
            common += 1
        lcp[rank] = common
    return lcp


def psi_array(suffix_array):
    """psi[i] = the rank of the suffix starting one position right of the suffix at rank i."""
    length = len(suffix_array)
    inverse = [0] * length
    for rank, start in enumerate(suffix_array):
        inverse[start] = rank
    return [inverse[(start + 1) % length] for start in suffix_array]


class CompressedSuffixArray:
    def __init__(self, text, sampling=4):
        self.text = text
        self.length = len(text)
        self.sampling = sampling
        self.suffix_array = suffix_array_doubling(text)
        self.lcp = lcp_kasai(text, self.suffix_array)
        self.psi = psi_array(self.suffix_array)
        # Sample every suffix whose *text position* is a multiple of the sampling step: the marker
        # is a rank, so a walk can test it, and because psi advances the position by one, a marked
        # rank is reached within sampling - 1 steps.
        self.sampled = {rank: value for rank, value in enumerate(self.suffix_array)
                        if value % sampling == 0}

    def stored_integers(self):
        """psi always costs n integers; the sample costs about n / sampling of them."""
        return self.length + len(self.sampled)

    def recover(self, rank):
        """SA[rank], walking psi until a sampled rank is reached (at most sampling - 1 steps)."""
        cursor = rank
        steps = 0
        while cursor not in self.sampled:
            cursor = self.psi[cursor]
            steps += 1
        return (self.sampled[cursor] - steps) % self.length, steps

    def rebuilt_suffix_array(self):
        worst = 0
        array = []
        for rank in range(self.length):
            value, steps = self.recover(rank)
            worst = max(worst, steps)
            array.append(value)
        return array, worst

    def count(self, pattern):
        """Count by binary search over the reconstructed suffix array."""
        array, _ = self.rebuilt_suffix_array()
        low, high = 0, self.length
        while low < high:
            middle = (low + high) // 2
            if self.text[array[middle]:] < pattern:
                low = middle + 1
            else:
                high = middle
        first = low
        low, high = first, self.length
        marker = pattern + "\uffff"
        while low < high:
            middle = (low + high) // 2
            if self.text[array[middle]:] < marker:
                low = middle + 1
            else:
                high = middle
        return low - first

    def locate(self, pattern):
        array, _ = self.rebuilt_suffix_array()
        return sorted(start for start in array if self.text.startswith(pattern, start))


def brute_force_count(text, pattern):
    if pattern == "":
        return len(text)
    return sum(1 for index in range(len(text) - len(pattern) + 1)
               if text[index:index + len(pattern)] == pattern)


def main():
    # --- The arrays themselves ----------------------------------------------------------
    texts = ["banana", "mississippi", "aaaaaaaa", "abababab", "abracadabra"]
    state = 11
    state, extra = random_text(state, 120, "abc")
    texts.append(extra)
    for text in texts:
        array = suffix_array_doubling(text)
        assert array == brute_force_suffix_array(text), "the doubling sort matches brute force"
        assert sorted(array) == list(range(len(text)))
        assert lcp_kasai(text, array) == brute_force_lcp(text, array), "Kasai matches brute force"
        psi = psi_array(array)
        for rank in range(len(text)):
            assert array[psi[rank]] == (array[rank] + 1) % len(text), "SA[psi[i]] = SA[i] + 1"
        inverse_psi = [0] * len(text)
        for rank, value in enumerate(psi):
            inverse_psi[value] = rank
        for rank in range(len(text)):
            assert array[inverse_psi[rank]] == (array[rank] - 1) % len(text), \
                "SA[psi^-1[i]] = SA[i] - 1"
        assert lcp_kasai("", []) == []

    # --- Reconstruction from psi plus samples -------------------------------------------
    state, text = random_text(state, 200, "abc")
    truth = brute_force_suffix_array(text)
    for sampling in (1, 2, 4, 8, 17):
        compressed = CompressedSuffixArray(text, sampling=sampling)
        assert compressed.suffix_array == truth
        rebuilt, worst = compressed.rebuilt_suffix_array()
        assert rebuilt == truth, "the complete suffix array is recovered at sampling %d" % sampling
        assert worst <= max(sampling - 1, 0), "the walk is bounded by the sampling step: %r" % worst
        assert compressed.stored_integers() == 200 + len(compressed.sampled)
        assert compressed.stored_integers() <= 2 * 200
        assert compressed.stored_integers() < len(text) + len(text) // sampling + 2
    compressed = CompressedSuffixArray(text, sampling=4)
    assert len(compressed.sampled) == 50
    assert compressed.stored_integers() == 250 < 400, "250 integers replace the 200-entry array plus"
    assert compressed.lcp == brute_force_lcp(text, compressed.suffix_array)
    # The sampling step trades storage for walk length: measure both extremes.
    tight = CompressedSuffixArray(text, sampling=1)
    loose = CompressedSuffixArray(text, sampling=32)
    assert tight.stored_integers() == 400
    assert loose.stored_integers() == 200 + 7
    assert loose.rebuilt_suffix_array()[1] > tight.rebuilt_suffix_array()[1]

    # --- Searching through the reconstructed structure ----------------------------------
    for pattern in ("", "a", "ab", "abc", "bcabc", "cba", "cab", "aaaa", text):
        assert compressed.count(pattern) == brute_force_count(text, pattern), "count %r" % pattern
        assert compressed.locate(pattern) == [index for index in range(len(text))
                                             if text.startswith(pattern, index)]
    checked = 0
    for _ in range(40):
        state, u = uniform(state)
        start = int(u * len(text))
        state, v = uniform(state)
        length = 1 + int(v * 6)
        pattern = text[start:start + length]
        assert compressed.count(pattern) == brute_force_count(text, pattern)
        assert compressed.locate(pattern) == [index for index in range(len(text))
                                             if text.startswith(pattern, index)]
        checked += 1
    assert checked == 40
    assert compressed.count("zzz") == 0 and compressed.locate("zzz") == []

    unary = CompressedSuffixArray("aaaaaa", sampling=2)
    assert unary.suffix_array == [5, 4, 3, 2, 1, 0]
    assert unary.lcp == [0, 1, 2, 3, 4, 5]
    assert unary.count("aaa") == 4 and unary.locate("aaaa") == [0, 1, 2]
    single = CompressedSuffixArray("q", sampling=1)
    assert single.suffix_array == [0] and single.psi == [0] and single.lcp == [0]
    assert single.count("q") == 1 and single.locate("q") == [0]
    assert single.rebuilt_suffix_array()[0] == [0]

    # --- Determinism --------------------------------------------------------------------
    again = CompressedSuffixArray(text, sampling=4)
    assert again.suffix_array == compressed.suffix_array
    assert again.psi == compressed.psi and again.lcp == compressed.lcp
    assert again.rebuilt_suffix_array() == compressed.rebuilt_suffix_array()
    assert again.count("abc") == compressed.count("abc")
    assert random_text(11, 120, "abc")[1] == extra


if __name__ == "__main__":
    main()
    print("PASS")
