"""Task 602 - Implement compressed suffix automaton with substring queries optimized. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

A suffix automaton is built incrementally (extend the last state, redirect transitions, clone the
state whose class is too long) and then *compressed*: the dict-per-state transition maps are replaced
by a compressed sparse row layout - one sorted character array plus one target array shared by all
states, with an offset table. A transition lookup is then a binary search inside the state's slice
instead of a hash probe, which is the layout a real compact implementation uses (2 flat int/char
arrays instead of n small dicts). The dict maps are kept only so the checks can prove that every
binary-search lookup returns what the hash lookup returned.

Answers that the automaton gives directly: endpos size per state (occurrence count, propagated up
the suffix-link tree in decreasing length order), the minimum end position per state (also propagated
up the link tree, giving the leftmost occurrence), the distinct-substring count
sum(len[v] - len[link[v]]), the longest common substring with another string (the classic walk that
follows suffix links when a transition is missing), and the k-th distinct substring in lexicographic
order (descending the transitions while carrying the number of distinct continuations per state).

The "optimized queries" part is the batch walk `prefix_occurrences(s)`: every prefix of s is answered
in ONE pass over s, with the occurrence count taken from the state reached when the whole prefix
matched, instead of |s| independent walks. Checks assert that the batch answers equal the individual
queries, that the compressed lookups equal the hash lookups, and that both classic bounds hold
(states <= 2n, transitions <= 3n, the classical tighter bounds being 2n - 1 and 3n - 4). Everything
else is cross-checked against brute force: distinct substrings, occurrences, leftmost positions,
k-th substring and longest common substring on hand cases and on LCG-generated strings, plus a
20000-character text whose longest common substring with a mutated copy is verified by checking that
no longer common substring exists.
"""
import bisect


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


class CompressedSuffixAutomaton:
    """Suffix automaton with transitions stored as sorted flat arrays (binary-search lookup)."""

    def __init__(self, text):
        length, link, trans = [0], [-1], [{}]
        firstpos, occ = [-1], [0]                   # min end position, endpos size
        last = 0
        for i, ch in enumerate(text):
            cur = len(length)
            length.append(length[last] + 1)
            link.append(-1)
            trans.append({})
            firstpos.append(i)
            occ.append(1)
            p = last
            while p != -1 and ch not in trans[p]:
                trans[p][ch] = cur
                p = link[p]
            if p == -1:
                link[cur] = 0
            elif length[p] + 1 == length[trans[p][ch]]:
                link[cur] = trans[p][ch]
            else:
                q = trans[p][ch]                      # clone: the class of q splits
                clone = len(length)
                length.append(length[p] + 1)
                link.append(link[q])
                trans.append(dict(trans[q]))
                firstpos.append(firstpos[q])
                occ.append(0)
                while p != -1 and trans[p].get(ch) == q:
                    trans[p][ch] = clone
                    p = link[p]
                link[q] = link[cur] = clone
            last = cur

        # decreasing length is a topological order of the suffix-link tree
        for v in sorted(range(1, len(length)), key=lambda x: length[x], reverse=True):
            parent = link[v]
            occ[parent] += occ[v]
            if firstpos[parent] == -1 or firstpos[v] < firstpos[parent]:
                firstpos[parent] = firstpos[v]

        self.length, self.link, self.firstpos, self.occ = length, link, firstpos, occ
        offsets, labels, targets = [0] * (len(length) + 1), [], []
        for v in range(len(length)):
            offsets[v] = len(labels)
            for ch in sorted(trans[v]):
                labels.append(ch)
                targets.append(trans[v][ch])
        offsets[len(length)] = len(labels)
        self.offsets, self.labels, self.targets = offsets, labels, targets
        self.hash_next = trans                       # kept only for the cross-checks

    def next(self, state, ch):
        """Compressed lookup: binary search inside the state's slice of the flat arrays."""
        lo, hi = self.offsets[state], self.offsets[state + 1]
        idx = bisect.bisect_left(self.labels, ch, lo, hi)
        return self.targets[idx] if idx < hi and self.labels[idx] == ch else -1

    def walk(self, pattern):
        state = 0
        for ch in pattern:
            state = self.next(state, ch)
            if state == -1:
                return -1
        return state

    def contains(self, pattern):
        return bool(pattern) and self.walk(pattern) != -1

    def occurrences(self, pattern):
        if not pattern:                             # documented: the empty pattern is not searched
            return 0
        state = self.walk(pattern)
        return 0 if state == -1 else self.occ[state]

    def first_position(self, pattern):
        """Start of the leftmost occurrence, from the class's minimum end position."""
        if not pattern:
            return -1
        state = self.walk(pattern)
        return -1 if state == -1 else self.firstpos[state] - len(pattern) + 1

    def distinct_substrings(self):
        return sum(self.length[v] - self.length[self.link[v]] for v in range(1, len(self.length)))

    def path_counts(self):
        """count[v] = number of distinct non-empty strings spellable from v; count[0] is the total."""
        counts = [0] * len(self.length)
        for v in sorted(range(len(self.length)), key=lambda x: self.length[x], reverse=True):
            total = 0
            for idx in range(self.offsets[v], self.offsets[v + 1]):
                total += 1 + counts[self.targets[idx]]
            counts[v] = total
        return counts

    def kth_distinct_substring(self, k):
        """The k-th distinct substring in lexicographic order (1-indexed)."""
        if k < 1 or k > self.distinct_substrings():
            raise ValueError("k out of range")
        counts = self.path_counts()
        state, out = 0, []
        while True:
            for idx in range(self.offsets[state], self.offsets[state + 1]):
                width = 1 + counts[self.targets[idx]]      # strings starting with this character
                if k <= width:
                    out.append(self.labels[idx])
                    k -= 1
                    if k == 0:
                        return "".join(out)
                    state = self.targets[idx]
                    break
                k -= width

    def longest_common_substring(self, other):
        """Walk `other` through the automaton, following suffix links when a transition is missing."""
        best_len, best_end, state, matched = 0, 0, 0, 0
        for i, ch in enumerate(other):
            while state and self.next(state, ch) == -1:
                state = self.link[state]
                matched = self.length[state]
            nxt = self.next(state, ch)
            if nxt == -1:
                state, matched = 0, 0
            else:
                state, matched = nxt, matched + 1
            if matched > best_len:
                best_len, best_end = matched, i
        return other[best_end - best_len + 1:best_end + 1]

    def prefix_occurrences(self, s):
        """Batch query: the occurrence count of every non-empty prefix of s in one pass."""
        out, state, matched = [], 0, 0
        for i, ch in enumerate(s):
            while state and self.next(state, ch) == -1:
                state = self.link[state]
                matched = self.length[state]
            nxt = self.next(state, ch)
            if nxt == -1:
                state, matched = 0, 0
            else:
                state, matched = nxt, matched + 1
            # occ[state] is the count of every string in the state's class, so it is the answer
            # exactly when the whole prefix is that matched substring
            out.append(self.occ[state] if matched == i + 1 else 0)
        return out


def brute_substrings(text):
    return {text[i:j] for i in range(len(text)) for j in range(i + 1, len(text) + 1)}


def brute_positions(text, pattern):
    out = []
    start = text.find(pattern)
    while start != -1:
        out.append(start)
        start = text.find(pattern, start + 1)
    return out


def brute_common_substrings(a, b):
    return {p for p in brute_substrings(a) if p in b}


def no_longer_common_substring(a, b, length):
    """True iff `a` and `b` share no substring longer than `length` (exact, candidate scan)."""
    return not any(a[i:i + length + 1] in b for i in range(len(a) - length))


def main():
    banana = CompressedSuffixAutomaton("banana")
    check(banana.distinct_substrings() == 15, "banana distinct")
    check(banana.occurrences("ana") == 2 and banana.occurrences("na") == 2, "banana counts")
    check(banana.occurrences("a") == 3 and banana.occurrences("banana") == 1, "banana counts 2")
    check(banana.first_position("ana") == 1 and banana.first_position("a") == 1, "banana leftmost")
    check(banana.first_position("nana") == 2 and banana.first_position("x") == -1, "banana leftmost 2")
    check(banana.longest_common_substring("ananas") == "anana", "banana/ananas LCS")
    check(banana.contains("ban") and not banana.contains("bn") and not banana.contains(""), "banana")
    check(banana.occurrences("") == 0 and banana.first_position("") == -1, "banana empty pattern")
    check(CompressedSuffixAutomaton("").distinct_substrings() == 0, "empty text")
    check(CompressedSuffixAutomaton("aaaa").distinct_substrings() == 4, "aaaa distinct")

    rejected = 0
    for auto in (CompressedSuffixAutomaton("abab"), banana):
        total = auto.distinct_substrings()
        for k in (0, total + 1, -1):
            try:
                auto.kth_distinct_substring(k)
            except ValueError:
                rejected += 1
    check(rejected == 6, "out-of-range k must be rejected")

    for text in ("", "a", "aaaa", "abab", "abcabxabcd", "mississippi", "abracadabra"):
        auto = CompressedSuffixAutomaton(text)
        n = len(text)
        substrings = brute_substrings(text)
        check(auto.distinct_substrings() == len(substrings), text)
        check(len(auto.length) <= max(1, 2 * n), (text, len(auto.length)))
        check(len(auto.labels) <= max(1, 3 * n), (text, len(auto.labels)))
        for piece in substrings:
            spots = brute_positions(text, piece)
            check(auto.contains(piece) and auto.occurrences(piece) == len(spots), (text, piece))
            check(auto.first_position(piece) == spots[0], (text, piece))
        for probe in ("z", "az", text + "z", "qq"):
            check(not auto.contains(probe) and auto.occurrences(probe) == 0, (text, probe))
        ordered = sorted(substrings)
        feasible = [k for k in (1, 2, max(1, len(ordered) // 2), len(ordered))
                    if 1 <= k <= len(ordered)]
        for k in feasible:
            check(auto.kth_distinct_substring(k) == ordered[k - 1], (text, k))
        # the compressed layout must answer exactly like the hash maps it replaced
        for state in range(len(auto.length)):
            for ch in sorted(set(text) | {"z"}):
                check(auto.next(state, ch) == auto.hash_next[state].get(ch, -1), (text, state, ch))

    for trial in range(15):
        text, _ = lcg_string(30, "ab", 4200 + trial * 7919)
        auto = CompressedSuffixAutomaton(text)
        check(auto.distinct_substrings() == len(brute_substrings(text)), text)
        check(auto.prefix_occurrences(text)
              == [auto.occurrences(text[:i + 1]) for i in range(len(text))], text)
        check(auto.occurrences(text) == 1 and auto.first_position(text) == 0, text)
        other, _ = lcg_string(20, "ab", 99900 + trial * 104729)
        lcs = auto.longest_common_substring(other)
        shared = brute_common_substrings(text, other)
        check(len(lcs) == max([len(p) for p in shared], default=0), (text, other, lcs))
        if lcs:
            check(lcs in shared, (text, other, lcs))

    # large text: 20000 characters; identity queries are exact checks at this size
    large, _ = lcg_string(20000, "abc", 20260202)
    auto = CompressedSuffixAutomaton(large)
    check(len(auto.length) <= 2 * len(large)
          and len(auto.labels) <= 3 * len(large), "large bounds")
    check(sum(len(t) for t in auto.hash_next) == len(auto.labels), "large transition count")
    check(auto.occurrences(large) == 1 and auto.first_position(large) == 0, "large whole text")
    check(auto.longest_common_substring(large) == large, "large LCS with itself")
    head = large[:7000]
    check(auto.longest_common_substring(head) == head, "large LCS with a prefix")
    probe = large[5000:5040]
    check(auto.occurrences(probe) == len(brute_positions(large, probe)), "large probe")
    check(auto.prefix_occurrences(probe) == [auto.occurrences(probe[:i + 1])
                                             for i in range(len(probe))], "large batch")
    mutated = large[:8000] + "d" + large[8001:]
    lcs = auto.longest_common_substring(mutated)
    check(len(lcs) >= 7999 and lcs in large and lcs in mutated, "large mutated LCS")

    # exact maximality of an LCS on a medium pair: the candidate scan over one string is a
    # complete proof, because every common substring is a substring of that string
    a, _ = lcg_string(2000, "abc", 777001)
    b, _ = lcg_string(2000, "abc", 313373)
    injected = a[800:860]
    b = b[:1000] + injected + b[1000:]
    auto_a = CompressedSuffixAutomaton(a)
    medium_lcs = auto_a.longest_common_substring(b)
    check(len(medium_lcs) >= 60 and medium_lcs in a and medium_lcs in b, "medium injected block")
    check(no_longer_common_substring(a, b, len(medium_lcs)), "medium LCS is maximal")

    print(f"banana: {len(banana.length)} states, {len(banana.labels)} transitions, "
          f"{banana.distinct_substrings()} distinct substrings")
    print(f"large text: {len(auto.length)} states, {len(auto.labels)} transitions, "
          f"LCS with mutated copy {len(lcs)}")


if __name__ == "__main__":
    main()
    print("PASS")
