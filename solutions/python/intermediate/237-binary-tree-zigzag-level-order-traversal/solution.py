"""Task 237 - Binary tree zigzag level order traversal. (Tier: Intermediate, CTCI Ch 4)

Level-order BFS with a per-level snapshot; odd levels (0-based) are reversed, so the
result alternates left->right / right->left. An empty tree yields []. O(n) time.
"""

from collections import deque


class TreeNode:
    __slots__ = ("val", "left", "right")

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    """Build a binary tree from level-order values (None marks a missing child)."""
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


def zigzag_level_order(root):
    if root is None:
        return []
    levels = []
    q = deque([root])
    left_to_right = True
    while q:
        level = []
        for _ in range(len(q)):                   # snapshot of the current level size
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        levels.append(level if left_to_right else level[::-1])
        left_to_right = not left_to_right
    return levels


def main():
    assert zigzag_level_order(build_tree([3, 9, 20, None, None, 15, 7])) == [[3], [20, 9], [15, 7]]
    assert zigzag_level_order(build_tree([1, 2, 3, 4, 5, 6, 7])) == [[1], [3, 2], [4, 5, 6, 7]]
    assert zigzag_level_order(build_tree([1, 2, 3, 4, None, None, 5])) == [[1], [3, 2], [4, 5]]
    assert zigzag_level_order(build_tree([1])) == [[1]]
    assert zigzag_level_order(build_tree([1, 2])) == [[1], [2]]
    assert zigzag_level_order(None) == []


if __name__ == "__main__":
    main()
    print("PASS")
