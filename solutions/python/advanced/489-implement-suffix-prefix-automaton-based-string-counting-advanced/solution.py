"""Task 489 - Implement suffix-prefix automaton based string counting (advanced). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Documented clean task: given a finite set of strings S, count the distinct strings x that are both
a non-empty prefix of some s in S and a non-empty suffix of some s in S, i.e. |P ∩ Q| where
P = {s[:i] : s in S, 1 <= i <= |s|} and Q = {s[j:] : s in S, 0 <= j < |s|}.

Automaton approach: build a trie over all prefixes of S (every trie node corresponds to a member of
P, and the trie accepts exactly P - it is the "prefix automaton" of S). For each suffix x = s[j:]
of every s in S, feed x to the automaton from the root in one walk: x is accepted iff x in P, so
the walk decides membership x in P ∩ Q. A set of the accepted x counts each string once. The walk
costs O(|x|) per suffix, O(sum |s|^2) overall - documented as fine for the small sets used here.

The result is asserted against a brute-force set intersection on small LCG-generated sets
(30 sets of 5 strings over {a, b}, lengths 1..12) and on hand cases; the random-set count total is
pinned. Alphabet for the large case is {a, b, c}.
"""


class Trie:
    def __init__(self):
        self.children = {}
        self.size = 0

    def insert(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = Trie()
                self.size += 1
            node = node.children[ch]

    def accepts(self, word):
        node = self
        for ch in word:
            node = node.children.get(ch)
            if node is None:
                return False
        return True


def build_prefix_automaton(strings):
    trie = Trie()
    for s in strings:
        for i in range(1, len(s) + 1):
            trie.insert(s[:i])
    return trie


def count_prefix_suffix_intersection(strings):
    trie = build_prefix_automaton(strings)
    matched = set()
    walked = 0
    for s in strings:
        for j in range(len(s)):
            x = s[j:]
            walked += 1
            if trie.accepts(x):
                matched.add(x)
    return len(matched), walked, trie.size


def brute_prefix_suffix_intersection(strings):
    prefixes = {s[:i] for s in strings for i in range(1, len(s) + 1)}
    suffixes = {s[j:] for s in strings for j in range(len(s))}
    return prefixes & suffixes


def lcg_strings(count, seed, length, alphabet):
    strings = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        length_i = 1 + seed % length
        chars = []
        for _ in range(length_i):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    # Hand cases.
    assert count_prefix_suffix_intersection(["ababa", "baba"])[0] == 5
    assert brute_prefix_suffix_intersection(["ababa", "baba"]) == {"a", "ba", "aba", "baba", "ababa"}
    assert count_prefix_suffix_intersection(["abc"])[0] == 1
    assert brute_prefix_suffix_intersection(["abc"]) == {"abc"}
    assert count_prefix_suffix_intersection(["a"])[0] == 1
    assert count_prefix_suffix_intersection(["ab", "ba"])[0] == 4
    assert brute_prefix_suffix_intersection(["ab", "ba"]) == {"a", "b", "ab", "ba"}
    assert count_prefix_suffix_intersection(["aaaa"])[0] == 4  # a, aa, aaa, aaaa

    # The automaton accepts exactly the prefix set P.
    for strings in (["ababa", "baba"], ["abc", "cba", "bac"]):
        trie = build_prefix_automaton(strings)
        prefixes = {s[:i] for s in strings for i in range(1, len(s) + 1)}
        assert trie.size == len(prefixes), (trie.size, len(prefixes))
        for x in prefixes:
            assert trie.accepts(x)
        assert not trie.accepts("z" * 3)

    # Random sets: automaton == brute force, and the total count is pinned.
    total = 0
    for k in range(30):
        strings = lcg_strings(5, 489 + k, 12, "ab")
        answer, walked, nodes = count_prefix_suffix_intersection(strings)
        expected = brute_prefix_suffix_intersection(strings)
        assert answer == len(expected), (k, strings, answer, sorted(expected))
        total_suffixes = sum(len(s) for s in strings)
        assert walked == total_suffixes
        assert nodes == len({s[:i] for s in strings for i in range(1, len(s) + 1)})
        total += answer
    assert total == 330, total

    # Larger case with a third letter.
    strings = lcg_strings(6, 777, 14, "abc")
    answer, walked, nodes = count_prefix_suffix_intersection(strings)
    expected = brute_prefix_suffix_intersection(strings)
    assert answer == len(expected) == 11, (answer, len(expected))
    assert walked == sum(len(s) for s in strings)
    assert nodes == len({s[:i] for s in strings for i in range(1, len(s) + 1)})
    assert answer == len(expected)

    # Duplicate strings collapse: the count only depends on the set.
    assert count_prefix_suffix_intersection(["abc", "abc", "abc"])[0] == \
        count_prefix_suffix_intersection(["abc"])[0]
    print("PASS")


if __name__ == "__main__":
    main()
