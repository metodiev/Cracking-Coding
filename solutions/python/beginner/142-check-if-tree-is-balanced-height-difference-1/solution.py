"""Task 142 - Check if tree is balanced (height difference <= 1). (Tier: Beginner, CTCI Ch 4)

Balanced = at EVERY node the two subtree heights differ by at most 1.
The bottom-up pass returns a height, or the -1 sentinel as soon as a node fails,
so each node is visited once: O(n) time, O(h) recursion depth.
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


def height(node):
    """Number of nodes on the longest downward path (empty tree = 0)."""
    if node is None:
        return 0
    return 1 + max(height(node.left), height(node.right))


def height_or_failure(node):
    """Subtree height, or -1 when some node inside the subtree is unbalanced."""
    if node is None:
        return 0
    left = height_or_failure(node.left)
    if left == -1:
        return -1
    right = height_or_failure(node.right)
    if right == -1:
        return -1
    if abs(left - right) > 1:
        return -1                             # this node already violates the rule
    return 1 + max(left, right)


def is_balanced(root):
    """O(n) bottom-up: the -1 sentinel short-circuits everything above it."""
    return height_or_failure(root) != -1


def is_balanced_naive(root):
    """O(n^2) contrast: recompute both subtree heights at every node."""
    if root is None:
        return True
    if abs(height(root.left) - height(root.right)) > 1:
        return False
    return is_balanced_naive(root.left) and is_balanced_naive(root.right)


def main():
    assert is_balanced(None) and is_balanced_naive(None)
    assert is_balanced(build_tree([1])) and is_balanced(build_tree([1, 2]))
    assert is_balanced(build_tree([3, 9, 20, None, None, 15, 7]))
    assert is_balanced(build_tree([1, 2, 3, 4, 5]))
    assert not is_balanced(build_tree([1, 2, 2, 3, 3, None, None, 4, 4]))
    # deeply unbalanced chains
    assert not is_balanced(build_tree([1, 2, None, 3, None, 4]))
    assert not is_balanced(build_tree([1, None, 2, None, 3]))
    assert height(build_tree([1, 2, 3, 4, 5, None, 6])) == 3
    assert height(None) == 0
    for data in ([3, 9, 20, None, None, 15, 7], [1, 2, 2, 3, 3, None, None, 4, 4],
                 [1, 2, None, 3, None, 4], [1, 2, 3, 4, 5], [1, 2, 3]):
        tree = build_tree(data)
        assert is_balanced(tree) == is_balanced_naive(tree), data


if __name__ == "__main__":
    main()
    print("PASS")
