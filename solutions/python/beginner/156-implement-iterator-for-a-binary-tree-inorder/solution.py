"""Task 156 - Implement iterator for a binary tree (inorder). (Tier: Beginner, CTCI Ch 4)

InOrderIterator walks the tree in-order with an explicit stack (no recursion):
push the left spine of the current subtree, next() pops one node and then
descends that node's right subtree's left spine. O(1) amortised per step,
O(h) extra memory, and hasNext() never consumes anything.
"""


class TreeNode:
    __slots__ = ("val", "left", "right")

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    """Build a binary tree from level-order values (None marks a missing child)."""
    from collections import deque

    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    q = deque([root])
    i = 1
    while q and i < len(values):
        node = q.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            q.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            q.append(node.right)
        i += 1
    return root


class InOrderIterator:
    def __init__(self, root):
        self._stack = []
        self._push_left_spine(root)

    def _push_left_spine(self, node):
        while node is not None:
            self._stack.append(node)
            node = node.left

    def has_next(self):
        return len(self._stack) > 0

    def next(self):
        """Next in-order value; raises StopIteration once exhausted."""
        if not self._stack:
            raise StopIteration("iterator exhausted")
        node = self._stack.pop()
        self._push_left_spine(node.right)
        return node.val


def main():
    # valid BST: in-order must be ascending
    bst = build_tree([8, 3, 10, 1, 6, None, 14, None, None, 4, 7, 13])
    it = InOrderIterator(bst)
    values = []
    while it.has_next():
        values.append(it.next())
    assert values == [1, 3, 4, 6, 7, 8, 10, 13, 14]
    assert values == sorted(values)                  # ascending for a BST

    # exhaustion: hasNext() is false and next() raises
    assert it.has_next() is False
    try:
        it.next()
        raise AssertionError("expected StopIteration")
    except StopIteration:
        pass

    # empty tree is immediately exhausted
    empty = InOrderIterator(None)
    assert empty.has_next() is False
    try:
        empty.next()
        raise AssertionError("expected StopIteration")
    except StopIteration:
        pass

    # hasNext is idempotent (peeks without consuming) and works one step at a time
    it2 = InOrderIterator(build_tree([1, 2, 3]))
    assert it2.has_next() and it2.has_next()
    assert it2.next() == 2
    assert it2.next() == 1
    assert it2.has_next()
    assert it2.next() == 3
    assert it2.has_next() is False

    # single node
    it3 = InOrderIterator(build_tree([5]))
    assert it3.has_next() and it3.next() == 5 and not it3.has_next()


if __name__ == "__main__":
    main()
    print("PASS")
