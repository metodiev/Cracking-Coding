"""Task 311 - Suffix automaton (basic conceptual, small cases). (Tier: Intermediate, CTCI Ch 1)

A suffix automaton of s is the minimal DFA accepting exactly the substrings of s. Each state v
keeps len[v] (longest string of its endpos class), link[v] (suffix link) and a transition map.
Each state contributes len[v] - len[link[v]] new distinct substrings, so the total number of
distinct substrings is sum(len[v] - len[link[v]]) over v != init. Construction is the textbook
online extension (amortised O(n) states and transitions); the tests keep strings small so the
bookkeeping can be cross-checked against a brute-force substring set.
"""


class SuffixAutomaton:
    def __init__(self):
        self.length = [0]              # len[v]
        self.link = [-1]               # suffix link; -1 marks the initial state
        self.transitions = [{}]        # transitions[v]: char -> state
        self.last = 0

    def extend(self, char):
        cur = len(self.length)
        self.length.append(self.length[self.last] + 1)
        self.link.append(0)
        self.transitions.append({})
        p = self.last
        while p != -1 and char not in self.transitions[p]:
            self.transitions[p][char] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.transitions[p][char]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.length)
                self.length.append(self.length[p] + 1)
                self.link.append(self.link[q])
                self.transitions.append(dict(self.transitions[q]))
                while p != -1 and self.transitions[p].get(char) == q:
                    self.transitions[p][char] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone
        self.last = cur

    def accepts(self, word):
        """DFA walk from the initial state; accepts "" vacuously (the empty substring)."""
        state = 0
        for char in word:
            if char not in self.transitions[state]:
                return False
            state = self.transitions[state][char]
        return True

    def distinct_substrings(self):
        return sum(self.length[v] - self.length[self.link[v]] for v in range(1, len(self.length)))


def build_suffix_automaton(s):
    automaton = SuffixAutomaton()
    for char in s:
        automaton.extend(char)
    return automaton


def brute_substrings(s):
    """The independent oracle: every substring of s, collected in a set."""
    return {s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)}


def all_words(alphabet, max_length):
    words = [""]
    frontier = [""]
    for _ in range(max_length):
        frontier = [w + ch for w in frontier for ch in alphabet]
        words.extend(frontier)
    return words


def lcg_strings(length, count, alphabet="ab", seed=3110):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    # "abab" has exactly the 7 distinct substrings a, b, ab, ba, aba, bab, abab
    abab = build_suffix_automaton("abab")
    assert brute_substrings("abab") == {"a", "b", "ab", "ba", "aba", "bab", "abab"}
    assert abab.distinct_substrings() == 7

    # "aaaa" has 4: a, aa, aaa, aaaa
    aaaa = build_suffix_automaton("aaaa")
    assert aaaa.distinct_substrings() == 4 == len(brute_substrings("aaaa"))

    banana = build_suffix_automaton("banana")
    assert banana.distinct_substrings() == 15 == len(brute_substrings("banana"))

    empty = build_suffix_automaton("")
    assert empty.distinct_substrings() == 0
    assert empty.accepts("")

    # Explicit rejections on the small example
    assert not abab.accepts("bb")
    assert not abab.accepts("baa")
    assert not abab.accepts("baab")
    assert abab.accepts("bab")

    cases = [("a", "ab"), ("ab", "ab"), ("abab", "ab"), ("banana", "ban"),
             ("mississippi", "mips")] + [(s, "ab") for s in lcg_strings(10, 12, "ab")] \
        + [(s, "abc") for s in lcg_strings(9, 8, "abc")]

    for s, alphabet in cases:
        automaton = build_suffix_automaton(s)
        substrings = brute_substrings(s)
        # 1) the len/link formula agrees with the brute-force substring set
        assert automaton.distinct_substrings() == len(substrings), s
        # 2) every substring of s is accepted by walking the automaton
        for word in substrings:
            assert automaton.accepts(word), (s, word)
        # 3) and every short word over the alphabet is accepted iff it really is a substring
        #    (the empty word is accepted vacuously; the brute-force set only holds non-empty ones)
        for word in all_words(alphabet, 4):
            if word == "":
                assert automaton.accepts(word), s
            else:
                assert automaton.accepts(word) == (word in substrings), (s, word)
        # 4) the state count stays within the textbook 2n bound
        assert len(automaton.length) <= 2 * len(s) + 1, s


if __name__ == "__main__":
    main()
    print("PASS")
