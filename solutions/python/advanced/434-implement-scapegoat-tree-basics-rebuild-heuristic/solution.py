"""Task 434 - Implement scapegoat tree basics (rebuild heuristic). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A scapegoat tree is an unbalanced BST that stays near-balanced by whole-subtree rebuilds instead
of rotations. Documented constants: balance factor alpha = 0.75 (a node may not have a child
subtree larger than alpha * its own size) and the delete trigger size < alpha * max_size, where
max_size is the peak size since the last full rebuild. Insert walks down, then walks back up
updating cached sizes and picks the deepest ancestor whose child exceeds alpha * its size - the
scapegoat - and rebuilds its subtree into a perfectly balanced half-split. Delete is physical:
the key is replaced by its in-order successor and sizes are repaired on the way up; falling below
alpha * max_size rebuilds the whole tree. Keys are unique (duplicate inserts are no-ops), matching
the sorted-list oracle used on LCG operation scripts. Rebuilds are counted: the adversarial
ascending insert 1..200 triggers at least one rebuild (a final assert documents the count), and
every state is checked for BST order, cached sizes and O(log n) height.
"""

import math


class Node:
    __slots__ = ("key", "left", "right", "size")

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.size = 1


class ScapegoatTree:
    def __init__(self, alpha=0.75):
        self.alpha = alpha
        self.root = None
        self.total = 0
        self.max_size = 0
        self.rebuilds = 0

    def size(self):
        return self.total

    def contains(self, key):
        node = self.root
        while node is not None:
            if key == node.key:
                return True
            node = node.left if key < node.key else node.right
        return False

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
            self.total = 1
            self.max_size = 1
            return True
        path = []
        node = self.root
        while True:
            path.append(node)
            if key == node.key:
                return False
            if key < node.key:
                if node.left is None:
                    node.left = Node(key)
                    break
                node = node.left
            else:
                if node.right is None:
                    node.right = Node(key)
                    break
                node = node.right
        scapegoat_index = -1
        for i in range(len(path) - 1, -1, -1):
            current = path[i]
            current.size += 1
            if scapegoat_index == -1:
                left = current.left.size if current.left else 0
                right = current.right.size if current.right else 0
                if left > self.alpha * current.size or right > self.alpha * current.size:
                    scapegoat_index = i
        self.total += 1
        self.max_size = max(self.max_size, self.total)
        if scapegoat_index >= 0:
            self._rebuild_at(path, scapegoat_index)
        return True

    def delete(self, key):
        if not self.contains(key):
            return False
        self.root = self._delete(self.root, key)
        self.total -= 1
        if self.total < self.alpha * self.max_size:
            self.root = self._build(self._collect(self.root), 0, self.total - 1)
            self.max_size = self.total
            self.rebuilds += 1
        return True

    def _delete(self, node, key):
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            return self._delete_node(node)
        self._update(node)
        return node

    def _delete_node(self, node):
        if node.left is None:
            return node.right
        if node.right is None:
            return node.left
        successor = node.right
        while successor.left is not None:
            successor = successor.left
        node.key = successor.key
        node.right = self._delete(node.right, successor.key)
        self._update(node)
        return node

    def inorder(self):
        return self._collect(self.root)

    def _collect(self, node):
        out = []
        stack = []
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            out.append(node.key)
            node = node.right
        return out

    def height(self):
        def depth(node):
            if node is None:
                return 0
            return 1 + max(depth(node.left), depth(node.right))

        return depth(self.root)

    def validate(self):
        def walk(node, low, high):
            if node is None:
                return 0, True
            if not low < node.key < high:
                return 0, False
            left_size, left_ok = walk(node.left, low, node.key)
            right_size, right_ok = walk(node.right, node.key, high)
            if not (left_ok and right_ok):
                return 0, False
            if node.size != 1 + left_size + right_size:
                return 0, False
            return node.size, True

        size, ok = walk(self.root, float("-inf"), float("inf"))
        return ok and size == self.total

    def _update(self, node):
        node.size = 1 + (node.left.size if node.left else 0) + (node.right.size if node.right else 0)

    def _rebuild_at(self, path, index):
        node = path[index]
        rebuilt = self._build(self._collect(node), 0, node.size - 1)
        if index == 0:
            self.root = rebuilt
        else:
            parent = path[index - 1]
            if parent.left is node:
                parent.left = rebuilt
            else:
                parent.right = rebuilt
        self.rebuilds += 1

    def _build(self, keys, lo, hi):
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        node = Node(keys[mid])
        node.left = self._build(keys, lo, mid - 1)
        node.right = self._build(keys, mid + 1, hi)
        self._update(node)
        return node


def main():
    tree = ScapegoatTree()
    assert tree.size() == 0
    assert tree.inorder() == []
    assert tree.validate()
    assert tree.delete(0) is False

    for key in (10, 5, 15, 3):
        assert tree.insert(key) is True
    assert tree.insert(10) is False
    assert tree.inorder() == [3, 5, 10, 15]
    assert tree.contains(15) and not tree.contains(4)
    assert tree.validate()
    assert tree.delete(5) is True
    assert tree.delete(5) is False
    assert tree.inorder() == [3, 10, 15]
    assert tree.validate()

    tree = ScapegoatTree()
    for key in range(1, 201):
        tree.insert(key)
    assert tree.inorder() == list(range(1, 201))
    assert tree.rebuilds >= 1, tree.rebuilds
    ascending_rebuilds = tree.rebuilds
    assert tree.height() <= 3 * math.ceil(math.log2(tree.size() + 1)) + 3, tree.height()
    for key in range(2, 201, 2):
        assert tree.delete(key) is True
    assert tree.inorder() == list(range(1, 201, 2))
    assert tree.validate()
    for key in range(1, 201, 2):
        assert tree.delete(key) is True
    assert tree.size() == 0 and tree.inorder() == []
    assert tree.validate()

    tree = ScapegoatTree()
    oracle = []
    seed = 4340
    for step in range(400):
        seed = (seed * 75 + 74) % 65537
        op = seed % 3
        seed = (seed * 75 + 74) % 65537
        key = seed % 100
        if op == 0:
            got = tree.insert(key)
            expected = key not in oracle
            if expected:
                oracle.append(key)
                oracle.sort()
            assert got == expected, (step, key)
        else:
            got = tree.delete(key)
            expected = key in oracle
            if expected:
                oracle.remove(key)
            assert got == expected, (step, key)
        assert tree.size() == len(oracle)
        if step % 25 == 0:
            assert tree.inorder() == oracle, step
            assert tree.validate(), step
    assert len(oracle) > 0
    assert tree.inorder() == oracle
    assert tree.validate()
    assert tree.height() <= 3 * math.ceil(math.log2(tree.size() + 1)) + 3, tree.height()


if __name__ == "__main__":
    main()
    print("PASS")
