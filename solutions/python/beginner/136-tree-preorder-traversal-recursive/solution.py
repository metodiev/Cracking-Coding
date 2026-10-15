"""Task 136 - Tree preorder traversal recursive. (Tier: Beginner, CTCI Ch 4)"""

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

def preorder(node, out=None):
    """Node, left, right."""
    out = [] if out is None else out
    if node is None:
        return out
    out.append(node.val)
    preorder(node.left, out)
    preorder(node.right, out)
    return out




def main():
    t = build_tree([1, 2, 3, 4, 5, None, 6])
    assert preorder(t) == [1, 2, 4, 5, 3, 6]
    assert preorder(None) == []
    assert preorder(build_tree([7])) == [7]


if __name__ == "__main__":
    main()
    print("PASS")
