"""Task 152 - All root-to-leaf paths with given sum. (Tier: Beginner, CTCI Ch 8)

pathSumPaths(root, target) returns every root-to-leaf path (a list of values)
whose values sum to target. Paths are emitted left before right, a node with a
missing child is NOT a path end, and the empty tree yields [].
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


def path_sum_paths(root, target):
    """All root-to-leaf paths summing to target, left subtree before right."""
    result = []
    path = []

    def dfs(node, remaining):
        if node is None:
            return
        path.append(node.val)
        if node.left is None and node.right is None:
            if remaining == node.val:
                result.append(list(path))     # snapshot: dfs mutates `path` afterwards
        else:
            dfs(node.left, remaining - node.val)
            dfs(node.right, remaining - node.val)
        path.pop()

    dfs(root, target)
    return result


def main():
    root = build_tree([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1])
    assert path_sum_paths(root, 22) == [[5, 4, 11, 2], [5, 8, 4, 5]]
    assert path_sum_paths(root, 26) == [[5, 8, 13]]
    assert path_sum_paths(root, 27) == [[5, 4, 11, 7]]
    assert path_sum_paths(root, 100) == []             # no matching path
    assert path_sum_paths(None, 0) == []               # empty tree
    assert path_sum_paths(build_tree([7]), 7) == [[7]]         # single node match
    assert path_sum_paths(build_tree([7]), 1) == []            # single node miss
    assert path_sum_paths(build_tree([1, 2]), 1) == []         # one missing child is not an end
    assert path_sum_paths(build_tree([1, 2]), 3) == [[1, 2]]
    assert path_sum_paths(build_tree([-1, -2, -3]), -3) == [[-1, -2]]
    # two matches that share a prefix must both be reported
    assert path_sum_paths(build_tree([1, 2, 2]), 3) == [[1, 2], [1, 2]]


if __name__ == "__main__":
    main()
    print("PASS")
