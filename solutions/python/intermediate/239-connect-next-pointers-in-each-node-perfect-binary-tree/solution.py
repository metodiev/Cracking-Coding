"""Task 239 - Connect next pointers in each node (perfect binary tree). (Tier: Intermediate, CTCI Ch 4)

Every node gets a `next` pointer to its right neighbour on the same level, or None at
the end of a level. The classic solution uses O(1) extra space: descend level by level
through the already-set next pointers of the parent level; for a parent p,
p.left.next = p.right and, when p.next exists, p.right.next = p.next.left.

Inputs are perfect binary trees (every internal node has two children).
"""

from collections import deque


class Node:
    __slots__ = ("val", "left", "right", "next")

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = None


def build_tree(values):
    """Build a binary tree from level-order values (None marks a missing child)."""
    if not values or values[0] is None:
        return None
    root = Node(values[0])
    q = deque([root])
    i = 1
    while q and i < len(values):
        node = q.popleft()
        if i < len(values) and values[i] is not None:
            node.left = Node(values[i])
            q.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = Node(values[i])
            q.append(node.right)
        i += 1
    return root


def connect(root):
    """Set next pointers level by level using O(1) extra space; returns root."""
    if root is None:
        return None
    leftmost = root
    while leftmost.left is not None:
        node = leftmost
        while node is not None:
            node.left.next = node.right
            if node.next is not None:
                node.right.next = node.next.left
            node = node.next
        leftmost = leftmost.left
    return root


def levels_via_next(root):
    """Read the levels through the next pointers (fails fast if a chain loops)."""
    levels = []
    leftmost = root
    while leftmost is not None:
        level = []
        node = leftmost
        while node is not None and len(level) <= 1000:
            level.append(node.val)
            node = node.next
        if node is not None:
            raise AssertionError("next chain longer than expected")
        levels.append(level)
        leftmost = leftmost.left
    return levels


def main():
    root = connect(build_tree([1, 2, 3, 4, 5, 6, 7]))
    assert levels_via_next(root) == [[1], [2, 3], [4, 5, 6, 7]]
    assert root.next is None
    assert root.left.next is root.right
    assert root.right.next is None
    assert root.left.left.next is root.left.right
    assert root.left.right.next is root.right.left
    assert root.right.left.next is root.right.right
    assert root.right.right.next is None

    single = connect(build_tree([1]))
    assert levels_via_next(single) == [[1]]
    assert single.next is None

    assert connect(None) is None

    big = connect(build_tree(list(range(1, 16))))
    assert levels_via_next(big) == [
        [1],
        [2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ]

    again = connect(root)                         # connecting twice changes nothing
    assert levels_via_next(again) == [[1], [2, 3], [4, 5, 6, 7]]


if __name__ == "__main__":
    main()
    print("PASS")
