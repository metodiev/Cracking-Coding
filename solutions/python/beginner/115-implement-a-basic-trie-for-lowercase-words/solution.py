"""Task 115 - Implement a basic trie for lowercase words. (Tier: Beginner, CTCI Ch 4)"""


class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self):
        self.children = {}          # char -> TrieNode (dict keeps it sparse)
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        if not node.is_word:
            node.is_word = True
            self.word_count += 1

    def contains(self, word):
        node = self._walk(word)
        return node is not None and node.is_word

    def starts_with(self, prefix):
        return self._walk(prefix) is not None

    def _walk(self, s):
        node = self.root
        for ch in s:
            node = node.children.get(ch)
            if node is None:
                return None
        return node


def main():
    t = Trie()
    for word in ("cat", "car", "card", "dog"):
        t.insert(word)
    assert t.word_count == 4
    t.insert("cat")                       # duplicate: no double count
    assert t.word_count == 4

    assert t.contains("cat") and t.contains("card")
    assert not t.contains("ca")           # prefix but not a word
    assert not t.contains("care")
    assert t.starts_with("ca") and t.starts_with("")
    assert not t.starts_with("z")


if __name__ == "__main__":
    main()
    print("PASS")
