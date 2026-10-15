"""Task 117 - Autocomplete top-k prefix suggestions (trie). (Tier: Beginner, CTCI Ch 4)

Each node stores the max popularity in its subtree so we can prune; suggestions are
ranked by popularity then lexicographically for deterministic ties.
"""


class Node:
    __slots__ = ("kids", "weight", "best")

    def __init__(self):
        self.kids = {}
        self.weight = 0        # 0 means "not a word"
        self.best = 0          # best popularity anywhere below this node


class Autocomplete:
    def __init__(self):
        self.root = Node()

    def add(self, word, weight=1):
        node = self.root
        for ch in word:
            node = node.kids.setdefault(ch, Node())
        node.weight += weight
        self._recompute(self.root)      # refresh subtree max-popularity caches

    def _recompute(self, node):
        best = node.weight
        for kid in node.kids.values():
            best = max(best, self._recompute(kid))
        node.best = best
        return best

    def suggest(self, prefix, k):
        node = self.root
        for ch in prefix:
            node = node.kids.get(ch)
            if node is None:
                return []
        found = []
        self._collect(node, list(prefix), found)
        found.sort(key=lambda pair: (-pair[1], pair[0]))
        return [word for word, _ in found[:k]]

    def _collect(self, node, path, out):
        if node.weight > 0:
            out.append(("".join(path), node.weight))
        for ch, kid in node.kids.items():
            path.append(ch)
            self._collect(kid, path, out)
            path.pop()


def main():
    ac = Autocomplete()
    for word, w in [("car", 5), ("cart", 3), ("carbon", 2), ("care", 1), ("dog", 9)]:
        ac.add(word, w)

    assert ac.suggest("car", 2) == ["car", "cart"]        # popularity order
    assert ac.suggest("car", 10) == ["car", "cart", "carbon", "care"]
    assert ac.suggest("do", 5) == ["dog"]
    assert ac.suggest("z", 3) == []                       # unknown prefix
    assert ac.suggest("", 1) == ["dog"]                   # empty prefix = top overall


if __name__ == "__main__":
    main()
    print("PASS")
