"""Task 116 - Insert and search in trie. (Tier: Beginner, CTCI Ch 4)

Array-of-26 implementation (dense, fast) plus word collection per subtree.
"""


class Node:
    __slots__ = ("kids", "end")

    def __init__(self):
        self.kids = [None] * 26
        self.end = False


class WordTrie:
    def __init__(self):
        self.root = Node()

    def insert(self, word):
        node = self.root
        for ch in word:
            i = ord(ch) - ord("a")
            if node.kids[i] is None:
                node.kids[i] = Node()
            node = node.kids[i]
        node.end = True

    def search(self, word):
        node = self._node_for(word)
        return node is not None and node.end

    def count_with_prefix(self, prefix):
        node = self._node_for(prefix)
        return 0 if node is None else self._count(node)

    def words_with_prefix(self, prefix):
        node = self._node_for(prefix)
        out = []
        if node is not None:
            self._collect(node, list(prefix), out)
        return out

    def _node_for(self, s):
        node = self.root
        for ch in s:
            node = node.kids[ord(ch) - ord("a")]
            if node is None:
                return None
        return node

    def _count(self, node):
        total = 1 if node.end else 0
        for kid in node.kids:
            if kid is not None:
                total += self._count(kid)
        return total

    def _collect(self, node, path, out):
        if node.end:
            out.append("".join(path))
        for i, kid in enumerate(node.kids):
            if kid is not None:
                path.append(chr(ord("a") + i))
                self._collect(kid, path, out)
                path.pop()


def main():
    t = WordTrie()
    for word in ("app", "apple", "apply", "apt", "banana"):
        t.insert(word)
    assert t.search("app") and t.search("apply")
    assert not t.search("ap") and not t.search("apples")
    assert t.count_with_prefix("app") == 3        # app, apple, apply
    assert t.count_with_prefix("b") == 1
    assert t.count_with_prefix("z") == 0
    assert t.words_with_prefix("app") == ["app", "apple", "apply"]


if __name__ == "__main__":
    main()
    print("PASS")
