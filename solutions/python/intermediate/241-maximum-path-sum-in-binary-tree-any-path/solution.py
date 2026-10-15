"""Task 241 - Maximum path sum in binary tree (any path). (Tier: Intermediate, CTCI Ch 4)

At every node the best path through it is val + max(0, left_gain) + max(0, right_gain).
The value returned upward is val + max(0, best single-side gain): a path cannot split.
"""

class TreeNode:
    __slots__ = ("val", "left", "right")

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    """Level-order builder: None marks a missing child."""
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


def preorder(node, out=None):
    out = [] if out is None else out
    if node:
        out.append(node.val)
        preorder(node.left, out)
        preorder(node.right, out)
    return out


def inorder(node, out=None):
    out = [] if out is None else out
    if node:
        inorder(node.left, out)
        out.append(node.val)
        inorder(node.right, out)
    return out


def max_path_sum(root):
    best = [float("-inf")]

    def gain(node):
        if node is None:
            return 0
        left = max(0, gain(node.left))
        right = max(0, gain(node.right))
        best[0] = max(best[0], node.val + left + right)
        return node.val + max(left, right)

    if root is None:
        raise ValueError("empty tree has no path")
    gain(root)
    return best[0]


def main():
    assert max_path_sum(build_tree([1, 2, 3])) == 6                 # 2 -> 1 -> 3
    assert max_path_sum(build_tree([-10, 9, 20, None, None, 15, 7])) == 42
    assert max_path_sum(build_tree([-3])) == -3                     # negatives must be taken
    assert max_path_sum(build_tree([2, -1])) == 2                   # skipping the -1
    assert max_path_sum(build_tree([1, -2, -3])) == 1


if __name__ == "__main__":
    main()
    print("PASS")
