"""Task 251 - Implement skip list insert/search/delete. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

A skip list is a layered sorted linked list: level 0 holds every key, and each higher level
is a sparse "express lane". A new node's height comes from repeated coin flips; here the
"coin" is a deterministic 31-bit LCG (state = (state * 1103515245 + 12345) mod 2^31, bit 16
is the flip), so every language builds the exact same structure. Invariant checked below:
level i's chain is exactly the subsequence of the level-0 chain whose node has more than i
forward pointers, therefore each level is sorted and a subsequence of level 0.
"""

MAX_LEVEL = 6


class Node:
    __slots__ = ("key", "level", "nxt")

    def __init__(self, key, level):
        self.key = key
        self.level = level
        self.nxt = [None] * level


class SkipList:
    def __init__(self, seed=20251):
        self.head = Node(-1, MAX_LEVEL)
        self.size = 0
        self._state = seed

    def _coin(self):
        self._state = (self._state * 1103515245 + 12345) & 0x7FFFFFFF
        return (self._state >> 16) & 1

    def _random_level(self):
        level = 1
        while level < MAX_LEVEL and self._coin() == 1:
            level += 1
        return level

    def _find(self, key):
        """Last node with key < target at every level (search/insert/delete helper)."""
        update = [self.head] * MAX_LEVEL
        node = self.head
        for i in range(MAX_LEVEL - 1, -1, -1):
            while node.nxt[i] is not None and node.nxt[i].key < key:
                node = node.nxt[i]
            update[i] = node
        return update

    def insert(self, key):
        """True iff the key was not already present."""
        update = self._find(key)
        after = update[0].nxt[0]
        if after is not None and after.key == key:
            return False
        node = Node(key, self._random_level())
        for i in range(node.level):
            node.nxt[i] = update[i].nxt[i]
            update[i].nxt[i] = node
        self.size += 1
        return True

    def search(self, key):
        node = self.head
        for i in range(MAX_LEVEL - 1, -1, -1):
            while node.nxt[i] is not None and node.nxt[i].key < key:
                node = node.nxt[i]
            if node.nxt[i] is not None and node.nxt[i].key == key:
                return True
        return False

    def delete(self, key):
        """True iff the key was present (forward pointers are spliced out at every level)."""
        update = self._find(key)
        target = update[0].nxt[0]
        if target is None or target.key != key:
            return False
        for i in range(target.level):
            update[i].nxt[i] = target.nxt[i]
        self.size -= 1
        return True

    def walk(self, level=0):
        out = []
        node = self.head.nxt[level]
        while node is not None:
            out.append(node.key)
            node = node.nxt[level]
        return out

    def heights(self):
        return {node.key: node.level for node in self._iterate()}

    def _iterate(self):
        node = self.head.nxt[0]
        while node is not None:
            yield node
            node = node.nxt[0]


# Deterministic promotion bits for seed 20251 over the insertion order below.
HEIGHTS = {3: 1, 6: 1, 7: 1, 9: 1, 12: 1, 19: 1, 17: 4, 26: 4, 21: 6, 25: 1}
KEYS = [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]


def main():
    sl = SkipList()
    assert all(sl.insert(k) for k in KEYS)
    assert sl.size == 10
    assert sl.heights() == HEIGHTS                       # same structure in every language
    assert max(HEIGHTS.values()) == MAX_LEVEL            # the top lane is actually used
    assert sl.walk(0) == sorted(KEYS)                    # full level-0 walk is sorted

    for k in KEYS:
        assert sl.search(k)
    for k in (-1, 0, 5, 10, 20, 22, 30):
        assert not sl.search(k)

    heights = sl.heights()
    walk0 = sl.walk(0)
    for i in range(MAX_LEVEL):                           # index consistency
        chain = sl.walk(i)
        assert chain == [k for k in walk0 if heights[k] > i]
        assert chain == sorted(chain)

    assert sl.insert(3) is False and sl.size == 10       # duplicates rejected

    assert sl.delete(19) and sl.size == 9
    assert not sl.search(19)
    assert sl.delete(19) is False
    assert sl.delete(4) is False and sl.delete(100) is False
    heights = sl.heights()
    for i in range(MAX_LEVEL):
        assert sl.walk(i) == [k for k in sl.walk(0) if heights[k] > i]

    assert sl.delete(3) and sl.delete(25)                 # head-adjacent and near-tail keys
    assert sl.delete(21)                                  # the top-lane node
    for k in (6, 7, 9, 12, 17, 26):
        assert sl.delete(k)
    assert sl.size == 0
    assert all(sl.walk(i) == [] for i in range(MAX_LEVEL))
    assert not any(sl.search(k) for k in KEYS)

    assert sl.insert(42) and sl.search(42) and sl.size == 1
    assert sl.delete(42) and sl.size == 0


if __name__ == "__main__":
    main()
    print("PASS")
