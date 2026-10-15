"""Task 404 - Implement a suffix automaton and count different substrings. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A suffix automaton is the minimal DFA recognizing all substrings of s. Each state stands for
an equivalence class of substrings sharing the same end-position set, with `length` the
longest member of the class and `link` the state for the longest proper suffix outside it.
Every substring in the class is a suffix of the state's longest string with length in
(length(link), length], so the class contributes exactly length - length(link) distinct
substrings and the total is sum over states except the initial one.

Anchors: "abab" -> 7, "aaaa" -> 4, "abc" -> 6. Cross-checks: the count matches a brute-force
substring set, every substring is accepted by walking transitions from the initial state, and
generated non-substrings are rejected. `occurrence_count` additionally propagates end-position
counts (original states start at 1, clones at 0) along suffix links in decreasing-length
order, and is checked against a naive scan.
"""

class SuffixAutomaton:
    def __init__(self):
        self.next = [{}]          # state -> char -> state
        self.link = [-1]          # suffix link; -1 only for the initial state
        self.length = [0]         # longest string of the class
        self.is_clone = [False]
        self.last = 0
        self.text = ""

    def extend(self, ch):
        cur = len(self.length)
        self.next.append({})
        self.link.append(0)
        self.length.append(self.length[self.last] + 1)
        self.is_clone.append(False)
        p = self.last
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
                self.next.append(dict(self.next[q]))
                self.link.append(self.link[q])
                self.length.append(self.length[p] + 1)
                self.is_clone.append(True)
                while p != -1 and self.next[p].get(ch) == q:
                    self.next[p][ch] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone
        self.last = cur
        self.text += ch

    def state_for(self, t):
        """Walk t from the initial state; return the state or -1 when rejected."""
        state = 0
        for ch in t:
            state = self.next[state].get(ch, -1)
            if state == -1:
                return -1
        return state

    def contains(self, t):
        return self.state_for(t) != -1

    def distinct_substrings(self):
        """sum(length - length(link)) over all states but the initial one."""
        return sum(self.length[v] - self.length[self.link[v]] for v in range(1, len(self.length)))

    def endpos_sizes(self):
        order = sorted(range(1, len(self.length)), key=lambda v: self.length[v], reverse=True)
        count = [0 if self.is_clone[v] else 1 for v in range(len(self.length))]
        count[0] = 0
        for v in order:
            count[self.link[v]] += count[v]
        return count

    def occurrence_count(self, t):
        state = self.state_for(t)
        if state == -1:
            return 0
        return self.endpos_sizes()[state]


def build_suffix_automaton(s):
    automaton = SuffixAutomaton()
    for ch in s:
        automaton.extend(ch)
    return automaton


def all_substrings(s):
    return {s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)}


def count_occurrences(s, t):
    if not t:
        return 0
    return sum(1 for i in range(len(s) - len(t) + 1) if s[i:i + len(t)] == t)


def suffix_array_naive(s):
    return sorted(range(len(s)), key=lambda i: s[i:])


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


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def main():
    abab = build_suffix_automaton("abab")
    assert abab.distinct_substrings() == 7
    assert all_substrings("abab") == {"a", "b", "ab", "ba", "aba", "bab", "abab"}

    aaaa = build_suffix_automaton("aaaa")
    assert aaaa.distinct_substrings() == 4
    assert all_substrings("aaaa") == {"a", "aa", "aaa", "aaaa"}

    assert build_suffix_automaton("").distinct_substrings() == 0
    assert build_suffix_automaton("a").distinct_substrings() == 1
    assert build_suffix_automaton("abc").distinct_substrings() == 6
    assert build_suffix_automaton("banana").distinct_substrings() == 15
    assert build_suffix_automaton("abab").contains("")
    assert build_suffix_automaton("abab").contains("baba") is False
    assert abab.state_for("abab") != -1 and abab.state_for("ababa") == -1

    # Counting cross-check: the sum(length - length(link)) identity against a brute set,
    # and the SA+LCP identity n(n+1)/2 - sum(lcp) as a second, unrelated oracle.
    for s in ["", "a", "aa", "ab", "banana", "mississippi", "abababab", "aabbaabb"]:
        automaton = build_suffix_automaton(s)
        assert automaton.distinct_substrings() == len(all_substrings(s)), s
        lcp = kasai_lcp(s, suffix_array_naive(s))
        assert automaton.distinct_substrings() == len(s) * (len(s) + 1) // 2 - sum(lcp), s

    # Every substring accepted, every generated non-substring rejected.
    abab_substrings = all_substrings("abab")
    for t in abab_substrings:
        assert abab.contains(t), t
    for t in ["c", "aba b", "bb", "abb", "baa", "ababa", "babab", "xyz"]:
        assert not abab.contains(t), t
    for a in abab_substrings:
        for b in abab_substrings:
            joined = a + b
            assert abab.contains(joined) == (joined in abab_substrings), joined

    # Occurrence counts via end-position propagation vs the naive scan.
    for s in ["banana", "abab", "aaaa", "mississippi", "abababab"]:
        automaton = build_suffix_automaton(s)
        for t in all_substrings(s):
            assert automaton.occurrence_count(t) == count_occurrences(s, t), (s, t)
        assert automaton.occurrence_count("xx") == 0, s

    assert build_suffix_automaton("banana").occurrence_count("ana") == 2
    assert build_suffix_automaton("banana").occurrence_count("a") == 3
    assert build_suffix_automaton("mississippi").occurrence_count("issi") == 2

    for alphabet in ("ab", "abc"):
        for trial in range(30):
            s, seed = lcg_string(20, alphabet, 3000 + trial * 65537)
            automaton = build_suffix_automaton(s)
            substrings = all_substrings(s)
            assert automaton.distinct_substrings() == len(substrings), s
            for t in substrings:
                assert automaton.contains(t), (s, t)
                assert automaton.occurrence_count(t) == count_occurrences(s, t), (s, t)
            for t in [alphabet[0] + alphabet[-1] + alphabet[0], "zzz", "q", s + alphabet[0]]:
                if t not in substrings:
                    assert not automaton.contains(t), (s, t)


if __name__ == "__main__":
    main()
    print("PASS")
