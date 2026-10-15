"""Task 139 - Level order traversal (BFS) of tree. (Tier: Beginner, CTCI Ch 4)"""

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


def level_order(root):
    """List of levels, left to right."""
    from collections import deque

    if root is None:
        return []
    levels = []
    q = deque([root])
    while q:
        level = []
        for _ in range(len(q)):               # snapshot of the current level size
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        levels.append(level)
    return levels


def main():
    assert level_order(build_tree([1, 2, 3, 4, 5, None, 6])) == [[1], [2, 3], [4, 5, 6]]
    assert level_order(None) == []
    assert level_order(build_tree([7])) == [[7]]
    assert level_order(build_tree([1, 2, 3])) == [[1], [2, 3]]


if __name__ == "__main__":
    main()
    print("PASS")
