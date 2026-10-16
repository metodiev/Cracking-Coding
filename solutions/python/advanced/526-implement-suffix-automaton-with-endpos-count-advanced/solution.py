"""Task 526 - Implement suffix automaton with endpos count (advanced). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Suffix automaton built by the linear-time incremental algorithm: each state v carries
length[v] (the longest string of its endpos-equivalence class), link[v] (the suffix link, whose
class contains exactly the shorter suffixes of the same class), a transition map, cnt[v] (the
endpos size = the number of occurrences) and rep_end[v] (the end position of the class's longest
representative). Extending with a character creates cur with length = length[last] + 1 and
cnt = 1 (one end position) and either redirects transitions or *clones* an existing state whose
length is too large - the clone copies the transition map and link but starts with cnt = 0.
Occurrence counts are then pushed up the suffix-link tree in decreasing length order
(cnt[link[v]] += cnt[v]), which is exactly the endpos aggregation. Documented checks: distinct
substrings as sum over states of length[v] - length[link[v]] against the brute-force set of all
substrings; the 2n-1 state bound; per-state endpos sets rebuilt by merging sets up the link tree
and compared with the brute-force end positions of the state's longest representative substring,
including |endpos(v)| == cnt[v]; the suffix-link chain from `last` covering all n suffix lengths
exactly once; occurrence counts of every prefix compared against brute force (only the whole string is
guaranteed to occur once); the weighted identity
sum_v cnt[v] * (length[v] - length[link[v]]) == the number of (substring, occurrence) pairs
counted by brute force; pattern search through the transitions returning -1 for absent patterns;
hand-checked counts on "ababa" (aba = 2, ba = 2, bab = 1) and "aaaa" (a = 4, aaa = 2); and the
degenerate empty / single-character / unary strings.
"""


class SuffixAutomaton:
    def __init__(self):
        self.transitions = [{}]          # state -> {character: state}
        self.link = [-1]
        self.length = [0]
        self.cnt = [0]
        self.rep_end = [-1]              # end index (inclusive) of the longest string of the class
        self.order = [0]                 # states sorted by length, filled by _count_occurrences
        self.last = 0

    def _new_state(self, length, end):
        self.transitions.append({})
        self.link.append(-1)
        self.length.append(length)
        self.cnt.append(0)
        self.rep_end.append(end)
        return len(self.length) - 1

    def extend(self, character, position):
        cur = self._new_state(self.length[self.last] + 1, position)
        self.cnt[cur] = 1                                    # end position `position`
        p = self.last
        while p != -1 and character not in self.transitions[p]:
            self.transitions[p][character] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.transitions[p][character]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = self._new_state(self.length[p] + 1, position)
                self.transitions[clone] = dict(self.transitions[q])
                self.link[clone] = self.link[q]
                while p != -1 and self.transitions[p].get(character) == q:
                    self.transitions[p][character] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone
        self.last = cur

    def build(self, text):
        for position, character in enumerate(text):
            self.extend(character, position)
        self._count_occurrences()
        return self

    def _count_occurrences(self):
        """Push counts up the suffix-link tree, longest states first (counting sort by length)."""
        states = len(self.length)
        bucket = [0] * (len(self.length) + 1)
        for v in range(states):
            bucket[self.length[v]] += 1
        for i in range(1, len(bucket)):
            bucket[i] += bucket[i - 1]
        order = [0] * states
        for v in range(states):
            bucket[self.length[v]] -= 1
            order[bucket[self.length[v]]] = v
        for index in range(states - 1, 0, -1):
            v = order[index]
            if self.link[v] > 0:
                self.cnt[self.link[v]] += self.cnt[v]
        self.order = order

    def representative(self, text, state):
        """Longest string of the class, read out of the reconstructed end position."""
        length = self.length[state]
        return text[self.rep_end[state] - length + 1:self.rep_end[state] + 1]

    def locate(self, pattern):
        """State reached by the pattern, or -1 when the pattern does not occur."""
        state = 0
        for character in pattern:
            if character not in self.transitions[state]:
                return -1
            state = self.transitions[state][character]
        return state

    def contains(self, pattern):
        return bool(pattern) and self.locate(pattern) >= 0

    def occurrences(self, pattern):
        """Endpos size of the pattern; 0 for the empty pattern and for absent patterns."""
        if not pattern:
            return 0
        state = self.locate(pattern)
        return 0 if state < 0 else self.cnt[state]

    def distinct_substrings(self):
        return sum(self.length[v] - self.length[self.link[v]] for v in range(1, len(self.length)))

    def weighted_occurrences(self):
        """sum over distinct substrings of their occurrence count."""
        return sum(self.cnt[v] * (self.length[v] - self.length[self.link[v]])
                   for v in range(1, len(self.length)))

    def endpos_sets(self):
        """endpos(v) as a real set: state starts with its representative end, children merge in."""
        sets = [set() for _ in self.length]
        for v in range(len(self.length)):
            if self.rep_end[v] >= 0:
                sets[v].add(self.rep_end[v])
        for index in range(len(self.order) - 1, 0, -1):    # longest states first
            v = self.order[index]
            if self.link[v] > 0:
                sets[self.link[v]] |= sets[v]
        return sets


def all_substrings(text):
    return {text[i:j] for i in range(len(text)) for j in range(i + 1, len(text) + 1)}


def brute_force_occurrences(text, pattern):
    if not pattern:
        return 0
    return sum(1 for i in range(len(text) - len(pattern) + 1)
               if text[i:i + len(pattern)] == pattern)


def lcg_text(length, seed, alphabet):
    out = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        out.append(chr(ord('a') + seed % alphabet))
    return "".join(out)


def main():
    # Hand-checked occurrences on "ababa": suffixes are a, ba, aba, baba, ababa.
    automaton = SuffixAutomaton().build("ababa")
    assert automaton.occurrences("aba") == 2
    assert automaton.occurrences("ba") == 2
    assert automaton.occurrences("ab") == 2
    assert automaton.occurrences("bab") == 1
    assert automaton.occurrences("ababa") == 1
    assert automaton.occurrences("a") == 3
    assert automaton.occurrences("baba") == 1
    assert automaton.occurrences("bb") == 0
    assert automaton.occurrences("ababab") == 0
    assert automaton.occurrences("c") == 0
    assert automaton.occurrences("") == 0
    assert automaton.contains("aba") and not automaton.contains("bb")
    assert not automaton.contains("") and not automaton.contains("ababab")
    # Distinct substrings of "ababa": a, b, ab, ba, aba, bab, abab, baba, ababa = 9.
    assert automaton.distinct_substrings() == 9
    assert automaton.distinct_substrings() == len(all_substrings("ababa"))
    # sum over distinct substrings of occurrences: a(3)+b(2)+ab(2)+ba(2)+aba(2)+bab(1)+abab(1)
    # +baba(1)+ababa(1) = 15.
    assert automaton.weighted_occurrences() == 15

    # Hand-checked unary string "aaaa": a -> 4, aa -> 3, aaa -> 2, aaaa -> 1; 4 distinct substrings.
    unary = SuffixAutomaton().build("aaaa")
    assert unary.occurrences("a") == 4
    assert unary.occurrences("aa") == 3
    assert unary.occurrences("aaa") == 2
    assert unary.occurrences("aaaa") == 1
    assert unary.distinct_substrings() == 4
    assert unary.occurrences("aaaaa") == 0 and not unary.contains("aaaaa")
    # A unary string of length n has n+1 states and n transitions (the minimum possible automaton).
    assert len(unary.length) == 5

    # Degenerate inputs.
    empty = SuffixAutomaton().build("")
    assert len(empty.length) == 1 and empty.distinct_substrings() == 0
    assert empty.occurrences("a") == 0 and empty.weighted_occurrences() == 0
    single = SuffixAutomaton().build("z")
    assert single.distinct_substrings() == 1 and single.occurrences("z") == 1
    assert single.occurrences("zz") == 0

    # LCG strings: every documented structural property against brute force.
    for length, alphabet, seed in ((1, 2, 3), (7, 2, 11), (25, 2, 23), (30, 3, 37),
                                   (40, 2, 53), (17, 5, 71)):
        text = lcg_text(length, seed, alphabet)
        automaton = SuffixAutomaton().build(text)
        states = len(automaton.length)
        transitions = sum(len(m) for m in automaton.transitions)
        # The classical bounds are |states| <= 2n-1 and |transitions| <= 3n-4 for n >= 2; the
        # n = 1 case needs 2 states and 1 transition, so 2n / 3n-3 is used there.
        assert states <= (2 * length - 1 if length >= 2 else 2), (seed, states, length)
        assert transitions <= (3 * length - 4 if length >= 2 else 1), (seed, transitions, length)
        substrings = all_substrings(text)
        assert automaton.distinct_substrings() == len(substrings), (seed, text)
        # Per-state endpos structure: length range, occurrence count and the real endpos set.
        sets = automaton.endpos_sets()
        for v in range(1, states):
            assert automaton.length[automaton.link[v]] < automaton.length[v], (seed, v)
            representative = automaton.representative(text, v)
            assert len(representative) == automaton.length[v]
            positions = {i + len(representative) - 1 for i in range(length)
                         if text[i:i + len(representative)] == representative}
            assert positions == sets[v], (seed, v, representative, positions, sets[v])
            assert len(sets[v]) == automaton.cnt[v], (seed, v)
            expected = brute_force_occurrences(text, representative)
            assert expected == automaton.cnt[v], (seed, v, representative)
            # Dropping the last character cannot lose an occurrence: endpos(rep) is a subset of
            # endpos(rep[:-1]), so the shortened string occurs at least as often.
            shorter = representative[:-1]
            if shorter:
                assert automaton.occurrences(shorter) == brute_force_occurrences(text, shorter)
                assert automaton.occurrences(shorter) >= automaton.cnt[v]
        # Every substring and every absent pattern checked through the transition walk.
        for pattern in list(substrings)[:60]:
            state = automaton.locate(pattern)
            assert state >= 0
            assert automaton.occurrences(pattern) == brute_force_occurrences(text, pattern)
            assert automaton.representative(text, state) in substrings
        for pattern in ("", "zz", "qqq", text + "a", text[:-1] + "b"):
            assert automaton.occurrences(pattern) == brute_force_occurrences(text, pattern)
            assert automaton.contains(pattern) == bool(brute_force_occurrences(text, pattern))
        # Weighted identity: occurrence mass over distinct substrings.
        assert automaton.weighted_occurrences() == \
            sum(brute_force_occurrences(text, pattern) for pattern in substrings), seed
        # Prefix counts are not fixed (a prefix may reappear), but the *whole* string occurs
        # exactly once and the suffix-link chain from `last` partitions the n suffixes by length.
        for prefix_end in range(1, length + 1):
            prefix = text[:prefix_end]
            assert automaton.occurrences(prefix) == brute_force_occurrences(text, prefix)
        chain_total = 0
        state = automaton.last
        while state > 0:
            chain_total += automaton.length[state] - automaton.length[automaton.link[state]]
            state = automaton.link[state]
        assert chain_total == length, (seed, chain_total)
        assert automaton.cnt[automaton.last] == 1, "the whole string occurs once"
        assert automaton.occurrences(text) == 1

    # A longer string: 200 characters over a binary alphabet, distinct substrings by brute force.
    text = lcg_text(200, 97, 2)
    automaton = SuffixAutomaton().build(text)
    assert automaton.distinct_substrings() == len(all_substrings(text))
    assert len(automaton.length) <= 2 * 200 - 1
    for pattern in ("a", "aa", "ab", "abb", text[:5], text[100:105], text + "x"):
        assert automaton.occurrences(pattern) == brute_force_occurrences(text, pattern)

    # Determinism and independence between instances.
    first = SuffixAutomaton().build("banana")
    second = SuffixAutomaton().build("banana")
    assert first.distinct_substrings() == second.distinct_substrings() == 15
    assert first.cnt == second.cnt and first.length == second.length
    assert first.occurrences("ana") == 2 and first.occurrences("an") == 2
    assert first.occurrences("banana") == 1 and first.occurrences("nan") == 1
    assert first.occurrences("nana") == 1
    assert first.occurrences("bad") == 0


if __name__ == "__main__":
    main()
    print("PASS")
