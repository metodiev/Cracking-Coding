"""Task 140 - Max depth of binary tree. (Tier: Beginner, CTCI Ch 4)"""

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


def max_depth(node):
    """1 + max(depth(left), depth(right)); an empty tree has depth 0."""
    if node is None:
        return 0
    return 1 + max(max_depth(node.left), max_depth(node.right))


def max_depth_iterative(root):
    """BFS variant: count levels (useful to contrast recursion vs iteration)."""
    from collections import deque

    if root is None:
        return 0
    depth = 0
    q = deque([root])
    while q:
        depth += 1
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return depth


def main():
    assert max_depth(build_tree([1, 2, 3, 4, 5, None, 6])) == 3
    assert max_depth(None) == 0
    assert max_depth(build_tree([7])) == 1
    # skewed tree: depth = node count
    skewed = build_tree([1, 2, None, 3, None, 4])
    assert max_depth(skewed) == 4 and max_depth_iterative(skewed) == 4
    for data in ([1, 2, 3, 4, 5, None, 6], [1, None, 2], [1, 2, 3]):
        assert max_depth(build_tree(data)) == max_depth_iterative(build_tree(data))


if __name__ == "__main__":
    main()
    print("PASS")
