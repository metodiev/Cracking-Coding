"""Task 433 - Implement order statistics tree using balanced BST (rank/select). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

The balanced BST is a treap: a BST ordered by key plus a min-heap ordered by priority, kept
balanced by rotations on insert and by merging the two children on delete. Each node caches the
size of its subtree and a multiplicity count, so it acts as a multiset: rank(x) returns the
number of stored elements strictly below x and select(k) returns the k-th smallest (1-based,
counting multiplicities). Priorities are drawn from the shared LCG stream, which makes the tree
shape - and therefore every assertion - fully deterministic. Assertions compare insert/delete/
rank/select/count against a sorted-list oracle (bisect) on LCG operation scripts and re-check the
heap/BST/size invariants with a structural validator.
"""

from bisect import bisect_left


class Node:
    __slots__ = ("key", "priority", "count", "size", "left", "right")

    def __init__(self, key, priority):
        self.key = key
        self.priority = priority
        self.count = 1
        self.size = 1
        self.left = None
        self.right = None


class OrderStatisticsTree:
    def __init__(self):
        self.root = None
        self.total = 0

    def size(self):
        return self.total

    @staticmethod
    def _size_of(node):
        return node.size if node is not None else 0

    def _update(self, node):
        node.size = self._size_of(node.left) + node.count + self._size_of(node.right)

    def _rotate_right(self, node):
        child = node.left
        node.left = child.right
        child.right = node
        self._update(node)
        self._update(child)
        return child

    def _rotate_left(self, node):
        child = node.right
        node.right = child.left
        child.left = node
        self._update(node)
        self._update(child)
        return child

    def insert(self, key, priority):
        self.root = self._insert(self.root, key, priority)
        self.total += 1

    def _insert(self, node, key, priority):
        if node is None:
            return Node(key, priority)
        if key == node.key:
            node.count += 1
            node.size += 1
            return node
        if key < node.key:
            node.left = self._insert(node.left, key, priority)
            if node.left.priority < node.priority:
                return self._rotate_right(node)
        else:
            node.right = self._insert(node.right, key, priority)
            if node.right.priority < node.priority:
                return self._rotate_left(node)
        self._update(node)
        return node

    def delete(self, key):
        self.root, removed = self._delete(self.root, key)
        if removed:
            self.total -= 1
        return removed

    def _delete(self, node, key):
        if node is None:
            return None, False
        if key == node.key:
            if node.count > 1:
                node.count -= 1
                node.size -= 1
                return node, True
            return self._merge(node.left, node.right), True
        if key < node.key:
            node.left, removed = self._delete(node.left, key)
        else:
            node.right, removed = self._delete(node.right, key)
        self._update(node)
        return node, removed

    def _merge(self, left, right):
        if left is None:
            return right
        if right is None:
            return left
        if left.priority <= right.priority:
            left.right = self._merge(left.right, right)
            self._update(left)
            return left
        right.left = self._merge(left, right.left)
        self._update(right)
        return right

    def rank(self, key):
        return self._rank(self.root, key)

    def _rank(self, node, key):
        if node is None:
            return 0
        if key <= node.key:
            return self._rank(node.left, key)
        return self._size_of(node.left) + node.count + self._rank(node.right, key)

    def select(self, k):
        if not 1 <= k <= self.total:
            raise IndexError("select index out of range")
        return self._select(self.root, k)

    def _select(self, node, k):
        left = self._size_of(node.left)
        if k <= left:
            return self._select(node.left, k)
        if k <= left + node.count:
            return node.key
        return self._select(node.right, k - left - node.count)

    def count_of(self, key):
        node = self.root
        while node is not None:
            if key == node.key:
                return node.count
            node = node.left if key < node.key else node.right
        return 0

    def inorder(self):
        out = []

        def walk(node):
            if node is None:
                return
            walk(node.left)
            out.extend([node.key] * node.count)
            walk(node.right)

        walk(self.root)
        return out

    def validate(self):
        def walk(node, low, high):
            if node is None:
                return 0, True
            if not low < node.key < high:
                return 0, False
            left_size, left_ok = walk(node.left, low, node.key)
            right_size, right_ok = walk(node.right, node.key, high)
            if not (left_ok and right_ok and node.size == left_size + node.count + right_size):
                return 0, False
            if node.left is not None and node.left.priority < node.priority:
                return 0, False
            if node.right is not None and node.right.priority < node.priority:
                return 0, False
            return node.size, True

        size, ok = walk(self.root, float("-inf"), float("inf"))
        return ok and size == self.total


def main():
    tree = OrderStatisticsTree()
    assert tree.size() == 0
    for key, priority in ((5, 100), (1, 90), (9, 80), (3, 70), (3, 110)):
        tree.insert(key, priority)
    assert tree.inorder() == [1, 3, 3, 5, 9]
    assert tree.size() == 5
    assert tree.validate()
    assert tree.select(1) == 1
    assert tree.select(2) == 3
    assert tree.select(3) == 3
    assert tree.select(4) == 5
    assert tree.select(5) == 9
    assert tree.rank(1) == 0
    assert tree.rank(3) == 1
    assert tree.rank(4) == 3
    assert tree.rank(9) == 4
    assert tree.count_of(3) == 2
    assert tree.count_of(4) == 0
    assert tree.delete(3) is True
    assert tree.inorder() == [1, 3, 5, 9]
    assert tree.select(3) == 5
    assert tree.delete(100) is False
    assert tree.validate()

    tree = OrderStatisticsTree()
    oracle = []
    seed = 4331
    priorities = []
    for step in range(400):
        seed = (seed * 75 + 74) % 65537
        op = seed % 5
        if op <= 1:
            seed = (seed * 75 + 74) % 65537
            key = seed % 60
            seed = (seed * 75 + 74) % 65537
            tree.insert(key, seed)
            priorities.append(key)
            oracle.insert(bisect_left(oracle, key), key)
        elif op == 2:
            seed = (seed * 75 + 74) % 65537
            key = seed % 60
            expected = key in oracle
            if expected:
                oracle.remove(key)
            assert tree.delete(key) == expected, (step, key)
        elif op == 3:
            seed = (seed * 75 + 74) % 65537
            key = seed % 60
            assert tree.rank(key) == bisect_left(oracle, key), (step, key)
            assert tree.count_of(key) == oracle.count(key), (step, key)
        else:
            if oracle:
                seed = (seed * 75 + 74) % 65537
                k = 1 + seed % len(oracle)
                assert tree.select(k) == oracle[k - 1], (step, k)
        assert tree.size() == len(oracle)
        if step % 25 == 0:
            assert tree.inorder() == oracle, step
            assert tree.validate(), step
    assert len(oracle) > 0
    assert tree.inorder() == oracle
    assert tree.validate()


if __name__ == "__main__":
    main()
    print("PASS")
