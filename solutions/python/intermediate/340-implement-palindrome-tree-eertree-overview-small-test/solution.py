"""Task 340 - Implement palindrome tree (Eertree) overview & small test. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

Eertree over a string: two roots (length -1 imaginary, length 0 empty), one node per distinct
palindromic substring, each with a suffix link to the longest proper palindromic suffix.
Every position walks the suffix links to find (or create) its longest palindromic suffix and
counts one occurrence; propagating counts along suffix links from the longest node down turns
them into full occurrence counts. Cross-checked against brute-force palindromic substring
sets and counts on deterministic LCG strings.
"""


class Eertree:
    """Palindromic tree: distinct palindromes with their occurrence counts."""

    def __init__(self, text):
        self.text = text
        # Node 0 = imaginary root (length -1), node 1 = empty root (length 0).
        self.length = [-1, 0]
        self.link = [0, 0]
        self.next = [{}, {}]
        self.occurrences = [0, 0]
        self.palindrome = [None, ""]
        self.last = 1
        for position, char in enumerate(text):
            self._extend(position, char)
        # A node's stored count tallies how often its palindrome was the longest palindromic
        # suffix; adding each node into its suffix link (longest node first) lifts that into
        # the true occurrence count of every palindrome.
        for node in sorted(range(2, len(self.length)), key=lambda v: -self.length[v]):
            self.occurrences[self.link[node]] += self.occurrences[node]

    def _extend(self, position, char):
        current = self.last
        while True:
            index = position - 1 - self.length[current]
            if index >= 0 and self.text[index] == char:
                break
            current = self.link[current]

        if char in self.next[current]:
            self.last = self.next[current][char]
            self.occurrences[self.last] += 1
            return

        node = len(self.length)
        self.length.append(self.length[current] + 2)
        self.next.append({})
        self.palindrome.append(
            char if self.length[node] == 1 else char + self.palindrome[current] + char)
        if self.length[node] == 1:
            self.link.append(1)
        else:
            candidate = self.link[current]
            while True:
                index = position - 1 - self.length[candidate]
                if index >= 0 and self.text[index] == char:
                    break
                candidate = self.link[candidate]
            self.link.append(self.next[candidate][char])
        self.next[current][char] = node
        self.occurrences.append(0)
        self.last = node
        self.occurrences[node] += 1

    def distinct_count(self):
        return len(self.length) - 2

    def counts(self):
        """Map palindrome -> occurrence count, for every distinct palindromic substring."""
        return {self.palindrome[node]: self.occurrences[node] for node in range(2, len(self.length))}


def brute_distinct(text):
    found = set()
    for start in range(len(text)):
        for end in range(start + 1, len(text) + 1):
            piece = text[start:end]
            if piece == piece[::-1]:
                found.add(piece)
    return found


def brute_counts(text):
    distinct = brute_distinct(text)
    counts = {}
    for piece in distinct:
        size = len(piece)
        counts[piece] = sum(1 for start in range(len(text) - size + 1)
                            if text[start:start + size] == piece)
    return counts


def lcg_string(count, seed, alphabet="ab"):
    chars = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def main():
    # Hand-checked: "abba" has distinct palindromes {a, b, bb, abba}.
    tree = Eertree("abba")
    assert tree.distinct_count() == 4
    assert tree.counts() == {"a": 2, "b": 2, "bb": 1, "abba": 1}
    assert set(tree.counts()) == {"a", "b", "bb", "abba"}
    assert set(tree.counts()) == brute_distinct("abba")
    assert tree.counts() == brute_counts("abba")
    assert len(tree.length) - 2 == len(brute_distinct("abba"))

    # "aaaa": four distinct palindromes with counts 4, 3, 2, 1.
    tree = Eertree("aaaa")
    assert tree.distinct_count() == 4
    assert tree.counts() == {"a": 4, "aa": 3, "aaa": 2, "aaaa": 1}
    assert tree.counts() == brute_counts("aaaa")

    # Edge cases: empty string, single character, no palindromes beyond the characters
    # themselves, and a string that is entirely one palindrome family.
    assert Eertree("").distinct_count() == 0
    assert Eertree("").counts() == {}
    single = Eertree("x")
    assert single.distinct_count() == 1 and single.counts() == {"x": 1}
    assert Eertree("abcd").distinct_count() == 4
    assert Eertree("abcd").counts() == {"a": 1, "b": 1, "c": 1, "d": 1}
    assert Eertree("aba").counts() == {"a": 2, "b": 1, "aba": 1}
    assert Eertree("abccba").counts() == {"a": 2, "b": 2, "c": 2, "cc": 1, "bccb": 1, "abccba": 1}

    # LCG strings: distinct palindromes and every occurrence count must match brute force.
    seed = 1717
    for trial in range(60):
        seed = (seed * 75 + 74) % 65537
        size = seed % 40
        seed = (seed * 75 + 74) % 65537
        alphabet = "ab" if seed % 2 == 0 else "abc"
        text, seed = lcg_string(size, seed, alphabet)
        tree = Eertree(text)
        expected = brute_counts(text)
        assert tree.counts() == expected, (trial, text, tree.counts(), expected)
        assert tree.distinct_count() == len(expected), (trial, text)
        assert all(len(piece) <= len(text) for piece in tree.counts())
        for piece, count in tree.counts().items():
            assert piece == piece[::-1] and count >= 1


if __name__ == "__main__":
    main()
    print("PASS")
