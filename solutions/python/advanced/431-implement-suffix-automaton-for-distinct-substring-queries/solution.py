"""Task 431 - Implement suffix automaton for distinct substring queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Query implemented: the distinct-substring count of the automaton's own string, i.e. the sum
of len[state] - len[link[state]] over all non-initial states ("abab" -> 7, "aaaa" -> 4).
Two further suffix-automaton queries back the accept/reject checks: is_substring(t) walks the
transition function from the initial state, and occurrences(t) reads the end-position count of
the reached state (terminal counts are propagated along suffix links in decreasing-length
order). Patterns passed to occurrences are non-empty; every answer is cross-checked against a
brute-force set of all substrings on LCG-generated strings.
"""


class SuffixAutomaton:
    def __init__(self, text):
        self.length = [0]
        self.link = [-1]
        self.next = [{}]
        self.ends = [0]
        last = 0
        for ch in text:
            cur = len(self.length)
            self.length.append(self.length[last] + 1)
            self.link.append(-1)
            self.next.append({})
            self.ends.append(1)
            p = last
            while p != -1 and ch not in self.next[p]:
                self.next[p][ch] = cur
                p = self.link[p]
            if p == -1:
                self.link[cur] = 0
            else:
                q = self.next[p][ch]
                if self.length[p] + 1 == self.length[q]:
                    self.link[cur] = q
                else:
                    clone = len(self.length)
                    self.length.append(self.length[p] + 1)
                    self.link.append(self.link[q])
                    self.next.append(dict(self.next[q]))
                    self.ends.append(0)
                    while p != -1 and self.next[p].get(ch) == q:
                        self.next[p][ch] = clone
                        p = self.link[p]
                    self.link[q] = clone
                    self.link[cur] = clone
            last = cur
        order = sorted(range(1, len(self.length)), key=lambda state: self.length[state], reverse=True)
        for state in order:
            self.ends[self.link[state]] += self.ends[state]

    def distinct_substrings(self):
        return sum(self.length[v] - self.length[self.link[v]] for v in range(1, len(self.length)))

    def walk(self, pattern):
        state = 0
        for ch in pattern:
            state = self.next[state].get(ch, -1)
            if state == -1:
                return -1
        return state

    def is_substring(self, pattern):
        return self.walk(pattern) != -1

    def occurrences(self, pattern):
        state = self.walk(pattern)
        return 0 if state == -1 else self.ends[state]


def brute_distinct_substrings(text):
    return len({text[i:j] for i in range(len(text)) for j in range(i + 1, len(text) + 1)})


def brute_occurrences(text, pattern):
    count = 0
    for i in range(len(text) - len(pattern) + 1):
        if text[i:i + len(pattern)] == pattern:
            count += 1
    return count


def lcg_strings(count, seed, max_len=40, alphabet="abc"):
    strings = []
    for _ in range(count):
        length = 1 + seed % max_len
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    assert SuffixAutomaton("").distinct_substrings() == 0
    assert SuffixAutomaton("a").distinct_substrings() == 1
    assert SuffixAutomaton("abab").distinct_substrings() == 7
    assert SuffixAutomaton("aaaa").distinct_substrings() == 4
    assert SuffixAutomaton("abc").distinct_substrings() == 6
    assert SuffixAutomaton("ababa").distinct_substrings() == 9

    banana = SuffixAutomaton("banana")
    assert banana.distinct_substrings() == 15
    assert banana.is_substring("ban")
    assert banana.is_substring("ana")
    assert banana.is_substring("nan")
    assert not banana.is_substring("bn")
    assert not banana.is_substring("bananas")
    assert banana.occurrences("an") == 2
    assert banana.occurrences("ana") == 2
    assert banana.occurrences("na") == 2
    assert banana.occurrences("x") == 0

    abab = SuffixAutomaton("abab")
    assert abab.occurrences("ab") == 2
    assert abab.occurrences("ba") == 1
    assert abab.occurrences("abab") == 1

    aaaa = SuffixAutomaton("aaaa")
    assert aaaa.occurrences("a") == 4
    assert aaaa.occurrences("aa") == 3
    assert aaaa.occurrences("aaa") == 2
    assert aaaa.occurrences("aaaa") == 1

    rejected = 0
    accepted = 0
    for text in lcg_strings(40, seed=431):
        auto = SuffixAutomaton(text)
        assert auto.distinct_substrings() == brute_distinct_substrings(text), text
        assert auto.is_substring(text)
        assert auto.occurrences(text) == 1
        for start in range(len(text)):
            for end in range(start + 1, min(start + 6, len(text)) + 1):
                pattern = text[start:end]
                assert auto.is_substring(pattern), (text, pattern)
                assert auto.occurrences(pattern) == brute_occurrences(text, pattern), (text, pattern)
                accepted += 1
        for mutation in ("d", "bd", "abz", text + "d"):
            if auto.is_substring(mutation):
                assert auto.occurrences(mutation) == brute_occurrences(text, mutation)
            else:
                assert brute_occurrences(text, mutation) == 0
                rejected += 1
    assert rejected > 0 and accepted > 0


if __name__ == "__main__":
    main()
    print("PASS")
