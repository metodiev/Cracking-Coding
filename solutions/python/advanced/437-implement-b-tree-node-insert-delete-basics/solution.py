"""Task 437 - Implement B-tree node insert/delete basics. (Tier: Advanced, CTCI Ch 14)

A B-tree of order m = 4 (minimum degree t = 2): every node holds at most 2t-1 = 3 keys and, when
internal, 2..4 children; non-root nodes keep at least t-1 = 1 key, and all leaves sit at the same
depth. Multiset semantics are obtained by storing collision-free composite keys
key = (value << 20) | serial: each insertion gets a fresh serial, so the textbook unique-key
insert/delete apply unchanged while the sorted traversal still yields a sorted multiset of values.
Insert uses the preemptive split rule - a full node (3 keys) is split around its median before
descending, the median rises into the parent and the upper half becomes a new right sibling.
Delete follows CLRS: a key found in an internal node is replaced by its in-order predecessor or
successor; before descending into a child that holds only t-1 keys the child is repaired by
borrowing a key from a sibling (rotation through the parent) or by merging with a sibling - the
documented merge/borrow counter is asserted to be positive after the LCG workload. The sorted
traversal is compared against a sorted multiset oracle on LCG operation scripts, and structure
invariants (key order, occupancy, uniform leaf depth) are re-validated throughout.
"""

from bisect import bisect_left

T = 2
SHIFT = 20


class BTreeNode:
    __slots__ = ("keys", "children")

    def __init__(self):
        self.keys = []
        self.children = []


class BTree:
    def __init__(self):
        self.root = BTreeNode()
        self.size = 0
        self.next_serial = 0
        self.splits = 0
        self.merges = 0
        self.borrows = 0

    def insert(self, value):
        key = (value << SHIFT) | self.next_serial
        self.next_serial += 1
        if len(self.root.keys) == 2 * T - 1:
            new_root = BTreeNode()
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_nonfull(self.root, key)
        self.size += 1

    def _split_child(self, parent, index):
        node = parent.children[index]
        right = BTreeNode()
        median = node.keys[T - 1]
        right.keys = node.keys[T:]
        node.keys = node.keys[:T - 1]
        if node.children:
            right.children = node.children[T:]
            node.children = node.children[:T]
        parent.keys.insert(index, median)
        parent.children.insert(index + 1, right)
        self.splits += 1

    def _insert_nonfull(self, node, key):
        index = bisect_left(node.keys, key)
        if node.children:
            if len(node.children[index].keys) == 2 * T - 1:
                self._split_child(node, index)
                if key > node.keys[index]:
                    index += 1
            self._insert_nonfull(node.children[index], key)
        else:
            node.keys.insert(index, key)

    def _lower_bound(self, node, base):
        index = bisect_left(node.keys, base)
        if node.children:
            found = self._lower_bound(node.children[index], base)
            if found is not None:
                return found
            if index < len(node.keys):
                return node.keys[index]
            return None
        return node.keys[index] if index < len(node.keys) else None

    def contains(self, value):
        key = self._lower_bound(self.root, value << SHIFT)
        return key is not None and (key >> SHIFT) == value

    def delete(self, value):
        key = self._lower_bound(self.root, value << SHIFT)
        if key is None or (key >> SHIFT) != value:
            return False
        self._delete(self.root, key)
        if not self.root.keys and self.root.children:
            self.root = self.root.children[0]
        self.size -= 1
        return True

    def _delete(self, node, key):
        index = bisect_left(node.keys, key)
        if index < len(node.keys) and node.keys[index] == key:
            if not node.children:
                node.keys.pop(index)
            else:
                self._delete_internal(node, index)
            return
        if not node.children:
            return
        flag = index == len(node.keys)
        if len(node.children[index].keys) == T - 1:
            self._fill(node, index)
            if flag and index > len(node.keys):
                index -= 1
        self._delete(node.children[index], key)

    def _delete_internal(self, node, index):
        key = node.keys[index]
        left, right = node.children[index], node.children[index + 1]
        if len(left.keys) > T - 1:
            predecessor = self._max_key(left)
            node.keys[index] = predecessor
            self._delete(left, predecessor)
        elif len(right.keys) > T - 1:
            successor = self._min_key(right)
            node.keys[index] = successor
            self._delete(right, successor)
        else:
            self._merge(node, index)
            self._delete(node.children[index], key)

    def _fill(self, node, index):
        if index > 0 and len(node.children[index - 1].keys) > T - 1:
            self._borrow_from_left(node, index)
        elif index < len(node.keys) and len(node.children[index + 1].keys) > T - 1:
            self._borrow_from_right(node, index)
        elif index < len(node.keys):
            self._merge(node, index)
        else:
            self._merge(node, index - 1)

    def _borrow_from_left(self, node, index):
        child = node.children[index]
        left = node.children[index - 1]
        child.keys.insert(0, node.keys[index - 1])
        node.keys[index - 1] = left.keys.pop()
        if left.children:
            child.children.insert(0, left.children.pop())
        self.borrows += 1

    def _borrow_from_right(self, node, index):
        child = node.children[index]
        right = node.children[index + 1]
        child.keys.append(node.keys[index])
        node.keys[index] = right.keys.pop(0)
        if right.children:
            child.children.append(right.children.pop(0))
        self.borrows += 1

    def _merge(self, node, index):
        left = node.children[index]
        right = node.children[index + 1]
        left.keys.append(node.keys.pop(index))
        left.keys.extend(right.keys)
        if left.children:
            left.children.extend(right.children)
        node.children.pop(index + 1)
        self.merges += 1

    def _min_key(self, node):
        while node.children:
            node = node.children[0]
        return node.keys[0]

    def _max_key(self, node):
        while node.children:
            node = node.children[-1]
        return node.keys[-1]

    def inorder_values(self):
        out = []

        def walk(node):
            for i, key in enumerate(node.keys):
                if node.children:
                    walk(node.children[i])
                out.append(key >> SHIFT)
            if node.children:
                walk(node.children[-1])

        walk(self.root)
        return out

    def validate(self):
        def walk(node, is_root, depth, leaf_depth):
            if len(node.keys) > 2 * T - 1:
                return -1
            for i in range(1, len(node.keys)):
                if node.keys[i - 1] >= node.keys[i]:
                    return -1
            if node.children:
                if len(node.children) != len(node.keys) + 1:
                    return -1
                if len(node.children) < (2 if is_root else T):
                    return -1
                for child in node.children:
                    leaf_depth = walk(child, False, depth + 1, leaf_depth)
                    if leaf_depth < 0:
                        return -1
                return leaf_depth
            if not is_root and len(node.keys) < T - 1:
                return -1
            if leaf_depth != -1 and leaf_depth != depth:
                return -1
            return depth

        return walk(self.root, True, 0, -1) >= 0


def main():
    tree = BTree()
    assert tree.validate()
    assert tree.inorder_values() == []
    assert tree.delete(5) is False

    for value in (10, 20, 30, 40):
        tree.insert(value)
    assert tree.splits >= 1, tree.splits
    assert tree.inorder_values() == [10, 20, 30, 40]
    assert tree.validate()
    assert tree.contains(30) and not tree.contains(35)
    assert tree.size == 4

    tree.insert(10)
    tree.insert(10)
    assert tree.inorder_values() == [10, 10, 10, 20, 30, 40]
    assert tree.delete(10) is True
    assert tree.inorder_values() == [10, 10, 20, 30, 40]
    assert tree.delete(10) is True
    assert tree.delete(10) is True
    assert tree.contains(10) is False
    assert tree.delete(10) is False
    assert tree.inorder_values() == [20, 30, 40]

    for value in (35, 25, 15, 5, 45, 50, 55, 60):
        tree.insert(value)
    assert tree.inorder_values() == [5, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    assert tree.validate()
    for value in (30, 5, 60, 20, 45, 15, 55, 25, 35, 50, 40):
        assert tree.delete(value) is True
    assert tree.inorder_values() == []
    assert tree.size == 0
    assert tree.validate()

    tree = BTree()
    oracle = []
    seed = 4370
    for step in range(300):
        seed = (seed * 75 + 74) % 65537
        op = seed % 4
        seed = (seed * 75 + 74) % 65537
        if op <= 1:
            value = seed % 30
            tree.insert(value)
            oracle.insert(bisect_left(oracle, value), value)
        elif op == 2 and oracle:
            value = oracle.pop(seed % len(oracle))
            assert tree.delete(value) == True, (step, value)
            still_present = value in oracle
            assert tree.delete(value) == still_present, (step, value)
            if still_present:
                oracle.remove(value)
        else:
            value = 100 + seed % 5
            assert tree.delete(value) is False, step
        assert tree.size == len(oracle), step
        if step % 20 == 0 or step == 299:
            assert tree.inorder_values() == oracle, step
            assert tree.validate(), step
    assert len(oracle) > 0
    assert tree.inorder_values() == oracle
    assert tree.validate()
    assert tree.splits > 0 and tree.merges > 0, (tree.splits, tree.merges, tree.borrows)


if __name__ == "__main__":
    main()
    print("PASS")
