"""Task 141 - Min depth of binary tree. (Tier: Beginner, CTCI Ch 4)

Minimum depth = nodes on the shortest root-to-leaf path; a node with only one
child is not a leaf, so [1, 2, None, 3] has min depth 3 (not 2).
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


def min_depth(root):
    """BFS level count: the first level holding a leaf is the shallowest one (O(n) worst case)."""
    from collections import deque

    if root is None:
        return 0
    depth = 0
    q = deque([root])
    while q:
        depth += 1
        for _ in range(len(q)):
            node = q.popleft()
            if node.left is None and node.right is None:
                return depth
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
    return depth


def min_depth_recursive(node):
    """Follow the only child instead of treating a missing child as an empty leaf."""
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    if node.left is None:
        return 1 + min_depth_recursive(node.right)
    if node.right is None:
        return 1 + min_depth_recursive(node.left)
    return 1 + min(min_depth_recursive(node.left), min_depth_recursive(node.right))


def main():
    assert min_depth(None) == 0 and min_depth_recursive(None) == 0
    assert min_depth(build_tree([7])) == 1 and min_depth_recursive(build_tree([7])) == 1
    # trap: node 2 has no left child, so 2 is not a leaf -> the 3 sits at depth 3
    trap = build_tree([1, 2, None, 3])
    assert min_depth(trap) == 3 and min_depth_recursive(trap) == 3
    sample = build_tree([3, 9, 20, None, None, 15, 7])
    assert min_depth(sample) == 2 and min_depth_recursive(sample) == 2
    chain = build_tree([1, 2, None, 3, None, 4])
    assert min_depth(chain) == 4 and min_depth_recursive(chain) == 4
    for data in ([1, 2, 3, 4, 5, None, 6], [1, 2, None, 3], [1, 2, 3], [1, None, 2]):
        tree = build_tree(data)
        assert min_depth(tree) == min_depth_recursive(tree), data


if __name__ == "__main__":
    main()
    print("PASS")
