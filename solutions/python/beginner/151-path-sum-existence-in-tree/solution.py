"""Task 151 - Path sum existence in tree. (Tier: Beginner, CTCI Ch 4)

hasPathSum(root, target) is True iff some root-to-leaf path sums to target.
An empty tree has no paths (False), and a node with only one child is NOT a
path end: every path must finish at a true leaf (both children missing).
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


def has_path_sum(root, target):
    """True iff a root-to-leaf path sums to target (leaf = both children None)."""
    if root is None:
        return False
    if root.left is None and root.right is None:
        return root.val == target
    return has_path_sum(root.left, target - root.val) or has_path_sum(root.right, target - root.val)


def main():
    root = build_tree([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1])
    assert has_path_sum(root, 22) is True                  # 5 + 4 + 11 + 2
    assert has_path_sum(root, 18) is True                  # 5 + 8 + 4 + 1
    assert has_path_sum(root, 27) is True                  # 5 + 4 + 11 + 7
    assert has_path_sum(root, 5) is False                  # 5 is not a leaf
    assert has_path_sum(None, 0) is False                  # empty tree has no paths
    assert has_path_sum(build_tree([1]), 1) is True
    assert has_path_sum(build_tree([1]), 2) is False
    assert has_path_sum(build_tree([1, 2]), 1) is False    # node 1 has one missing child
    assert has_path_sum(build_tree([1, 2]), 3) is True     # 1 -> 2
    assert has_path_sum(build_tree([-2, None, -3]), -5) is True   # negative values
    assert has_path_sum(build_tree([0, 1, 1]), 1) is True
    assert has_path_sum(build_tree([0, 1, 1]), 0) is False


if __name__ == "__main__":
    main()
    print("PASS")
