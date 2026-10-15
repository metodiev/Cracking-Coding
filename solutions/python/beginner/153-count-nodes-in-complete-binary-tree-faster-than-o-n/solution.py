"""Task 153 - Count nodes in complete binary tree faster than O(n). (Tier: Beginner, CTCI Ch 4)

Uses the left-depth vs right-depth trick: walk the left spine and the right
spine of a subtree. If both heights are equal the subtree is a perfect binary
tree of height h, so it holds exactly 2^h - 1 nodes and is counted in O(h)
without visiting anything else. Otherwise recurse into both children.

Each height probe costs O(log n) and there are O(log n) recursive levels, so
the total cost is O(log^2 n) - far better than the naive O(n) walk. The trick
is only valid for COMPLETE trees, where a missing level is always filled
left-to-right; the tests cross-check the fast count against a naive O(n) count.
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


def _left_height(node):
    h = 0
    while node is not None:
        h += 1
        node = node.left
    return h


def _right_height(node):
    h = 0
    while node is not None:
        h += 1
        node = node.right
    return h


def count_nodes(root):
    """O(log^2 n) node count for a complete binary tree (O(1) for a perfect one)."""
    if root is None:
        return 0
    lh = _left_height(root)
    rh = _right_height(root)
    if lh == rh:                       # perfect subtree: 2^h - 1, no need to descend
        return (1 << lh) - 1
    return 1 + count_nodes(root.left) + count_nodes(root.right)


def naive_count(root):
    """Plain O(n) count used only to cross-check count_nodes."""
    if root is None:
        return 0
    return 1 + naive_count(root.left) + naive_count(root.right)


def main():
    assert count_nodes(None) == 0
    assert count_nodes(build_tree([1])) == 1

    # Level-order arrays without gaps build complete (and sometimes perfect) trees.
    for n in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 31):
        root = build_tree(list(range(n)))
        assert count_nodes(root) == n, n
        assert count_nodes(root) == naive_count(root), n

    # A perfect tree of height h is answered directly by 2^h - 1.
    full = build_tree(list(range(15)))
    assert count_nodes(full) == 15 == naive_count(full)

    # Larger complete tree: exact powers-of-two boundaries and one past them.
    for n in (63, 64, 65, 100, 1000):
        root = build_tree(list(range(n)))
        assert count_nodes(root) == n == naive_count(root), n

    # The last level partially filled (left-to-right) still counts correctly.
    partial = build_tree([1, 2, 3, 4, 5, 6, None])
    assert count_nodes(partial) == 6 == naive_count(partial)


if __name__ == "__main__":
    main()
    print("PASS")
