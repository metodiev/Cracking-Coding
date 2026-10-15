"""Task 252 - Design a phonebook using trie with counts and delete. (Tier: Intermediate, CTCI Ch 4)

A lowercase-letter trie where every node stores how many times the name ending at that node
was added (count 0 means "not a name, only a prefix"). delete() decrements the count and
prunes childless zero-count nodes bottom-up, so deleting every name returns the structure to
its empty state (just the root). Prefix listings emit names in lexicographic order.
"""


class TrieNode:
    __slots__ = ("children", "count")

    def __init__(self):
        self.children = {}
        self.count = 0


class Phonebook:
    def __init__(self):
        self.root = TrieNode()

    def _walk(self, name):
        node = self.root
        for ch in name:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def insert(self, name):
        """Adds a non-empty lowercase name; returns its new count."""
        node = self.root
        for ch in name:
            child = node.children.get(ch)
            if child is None:
                child = TrieNode()
                node.children[ch] = child
            node = child
        node.count += 1
        return node.count

    def count(self, name):
        """Exact count for a complete name (0 for missing names and for prefixes)."""
        node = self._walk(name)
        return 0 if node is None else node.count

    def delete(self, name):
        """Decrements the name's count and prunes dead branches; True iff it existed."""
        if not name:
            return False

        def rec(node, i):
            if i == len(name):
                if node.count == 0:
                    return False, False
                node.count -= 1
                return True, node.count == 0 and not node.children
            child = node.children.get(name[i])
            if child is None:
                return False, False
            found, prune = rec(child, i + 1)
            if not found:
                return False, False
            if prune:
                del node.children[name[i]]
            return True, node.count == 0 and not node.children

        return rec(self.root, 0)[0]

    def list_prefix(self, prefix):
        """All stored (name, count) pairs starting with prefix, lexicographically sorted."""
        node = self._walk(prefix)
        if node is None:
            return []
        out = []

        def dfs(current, path):
            if current.count > 0:
                out.append((prefix + path, current.count))
            for ch in sorted(current.children):
                dfs(current.children[ch], path + ch)

        dfs(node, "")
        return out

    def words(self):
        return self.list_prefix("")

    def node_count(self):
        """Number of nodes including the root; an empty phonebook is 1."""
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            total += 1
            stack.extend(node.children.values())
        return total


ALL = [("alan", 1), ("albert", 1), ("alice", 2), ("b", 1), ("barb", 1), ("bob", 1)]


def load(book):
    assert book.insert("alice") == 1
    assert book.insert("alice") == 2                      # duplicate inserts increment
    assert book.insert("alan") == 1
    assert book.insert("albert") == 1
    assert book.insert("bob") == 1
    assert book.insert("barb") == 1
    assert book.insert("b") == 1


def main():
    book = Phonebook()
    assert book.node_count() == 1

    load(book)
    # 1 root + 17 shared-prefix nodes: a,al,ala,alan,alb,albe,alber,albert,ali,alic,alice,
    # b,ba,bar,barb,bo,bob
    assert book.node_count() == 18

    assert book.count("alice") == 2
    assert book.count("alan") == 1
    assert book.count("alicia") == 0
    assert book.count("al") == 0                          # a prefix is not a name
    assert book.count("alic") == 0
    assert book.count("") == 0

    assert book.list_prefix("al") == [("alan", 1), ("albert", 1), ("alice", 2)]
    assert book.list_prefix("ali") == [("alice", 2)]
    assert book.list_prefix("b") == [("b", 1), ("barb", 1), ("bob", 1)]
    assert book.list_prefix("z") == []
    assert book.list_prefix("") == ALL
    assert book.words() == ALL

    assert book.delete("alice") is True                   # one reference left
    assert book.count("alice") == 1 and book.node_count() == 18
    assert book.list_prefix("al") == [("alan", 1), ("albert", 1), ("alice", 1)]
    assert book.delete("alice") is True                   # gone: prune alice/alic/ali
    assert book.count("alice") == 0 and book.node_count() == 15
    assert book.list_prefix("al") == [("alan", 1), ("albert", 1)]
    assert book.list_prefix("ali") == []
    assert book.delete("alice") is False
    assert book.delete("zzz") is False
    assert book.delete("") is False

    assert book.delete("b") is True                       # count hits 0 but "bob" survives
    assert book.count("b") == 0 and book.node_count() == 15
    assert book.list_prefix("b") == [("barb", 1), ("bob", 1)]

    assert book.delete("barb") is True                    # prunes barb/bar/ba
    assert book.node_count() == 12
    assert book.words() == [("alan", 1), ("albert", 1), ("bob", 1)]

    assert book.delete("alan") is True                    # prunes alan/ala
    assert book.node_count() == 10
    assert book.delete("albert") is True                  # prunes albert..a entirely
    assert book.node_count() == 4                         # root + b + bo + bob
    assert book.words() == [("bob", 1)]

    assert book.delete("bob") is True
    assert book.node_count() == 1                         # back to the empty state
    assert book.words() == [] and book.list_prefix("") == []
    assert book.delete("bob") is False


if __name__ == "__main__":
    main()
    print("PASS")
