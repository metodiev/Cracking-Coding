"""Task 314 - Aho-Corasick multi-pattern matching (outline). (Tier: Intermediate, CTCI Ch 1)

A trie of the patterns (goto function) plus BFS-computed failure links turns multi-pattern search
into a single left-to-right pass: at every text position follow failure links until a transition
exists, then report every pattern in the output set of the reached state (each state inherits the
outputs of its failure target). Total work is O(|text| + matches). Overlapping matches are
reported, and every result is cross-checked with a naive per-pattern scan on LCG strings.
"""

from collections import deque


class AhoCorasick:
    """Patterns are assumed non-empty."""

    def __init__(self, patterns):
        self.patterns = list(patterns)
        self.goto = [{}]
        self.fail = [0]
        self.output = [set()]
        for pattern in self.patterns:
            node = 0
            for char in pattern:
                nxt = self.goto[node].get(char)
                if nxt is None:
                    nxt = len(self.goto)
                    self.goto[node][char] = nxt
                    self.goto.append({})
                    self.fail.append(0)
                    self.output.append(set())
                node = nxt
            self.output[node].add(pattern)

        queue = deque(self.goto[0].values())    # depth-1 states already fail to the root
        while queue:
            node = queue.popleft()
            self.output[node] |= self.output[self.fail[node]]
            for char, child in self.goto[node].items():
                f = self.fail[node]
                while f != 0 and char not in self.goto[f]:
                    f = self.fail[f]
                self.fail[child] = self.goto[f].get(char, 0)
                queue.append(child)

    def search(self, text):
        """pattern -> ascending list of start indices (overlapping matches included)."""
        found = {pattern: [] for pattern in self.patterns}
        node = 0
        for i, char in enumerate(text):
            while node != 0 and char not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(char, 0)
            for pattern in self.output[node]:
                found[pattern].append(i - len(pattern) + 1)
        return found


def naive_search_all(text, patterns):
    """Independent oracle: scan each pattern on its own."""
    found = {}
    for pattern in patterns:
        hits = []
        for i in range(len(text) - len(pattern) + 1):
            if text[i:i + len(pattern)] == pattern:
                hits.append(i)
        found[pattern] = hits
    return found


def all_words(alphabet, max_length):
    words = []
    frontier = [""]
    for _ in range(max_length):
        frontier = [w + ch for w in frontier for ch in alphabet]
        words.extend(frontier)
    return words


def lcg_strings(length, count, alphabet="ab", seed=3140):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    # Classic example: hand-checked against "ushers" (u0 s1 h2 e3 r4 s5)
    patterns = ["he", "she", "his", "hers"]
    automaton = AhoCorasick(patterns)
    result = automaton.search("ushers")
    assert result == {"he": [2], "she": [1], "hers": [2], "his": []}
    assert result == naive_search_all("ushers", patterns)

    # Overlapping matches are all reported
    overlapping = AhoCorasick(["aa", "aaa"])
    assert overlapping.search("aaaa") == {"aa": [0, 1, 2], "aaa": [0, 1]}
    single = AhoCorasick(["a", "ab"])
    assert single.search("ab") == {"a": [0], "ab": [0]}
    assert single.search("") == {"a": [], "ab": []}

    for alphabet in ("ab", "abc"):
        pattern_sets = [
            all_words(alphabet, 3),
            [alphabet[0] * k for k in range(1, 4)],
            [w for w in all_words(alphabet, 2) if len(w) == 2],
        ]
        texts = ["", alphabet, alphabet * 4, alphabet[0] * 6] + lcg_strings(18, 20, alphabet)
        for patterns in pattern_sets:
            automaton = AhoCorasick(patterns)
            for text in texts:
                result = automaton.search(text)
                oracle = naive_search_all(text, patterns)
                assert result == oracle, (patterns, text)
                for pattern, hits in result.items():
                    for start in hits:                        # every report is a real match
                        assert text[start:start + len(pattern)] == pattern, (text, pattern, start)


if __name__ == "__main__":
    main()
    print("PASS")
