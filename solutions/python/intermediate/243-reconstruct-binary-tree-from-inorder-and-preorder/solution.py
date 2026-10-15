"""Task 243 - Reconstruct binary tree from inorder and preorder. (Tier: Intermediate, CTCI Ch 4)

The first preorder value is the root; its index in the inorder array splits both arrays.
O(n) with a value -> index map.
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


def build_from_pre_in(preorder, inorder):
    index = {value: i for i, value in enumerate(inorder)}
    pos = [0]

    def build(lo, hi):
        if lo > hi:
            return None
        root_val = preorder[pos[0]]
        pos[0] += 1
        node = TreeNode(root_val)
        mid = index[root_val]
        node.left = build(lo, mid - 1)
        node.right = build(mid + 1, hi)
        return node

    if not preorder or len(preorder) != len(inorder):
        raise ValueError("preorder and inorder must be non-empty and equal-length")
    return build(0, len(inorder) - 1)


def main():
    pre = [3, 9, 20, 15, 7]
    ino = [9, 3, 15, 20, 7]
    root = build_from_pre_in(pre, ino)
    assert preorder(root) == pre
    assert inorder(root) == ino
    # single node and a left-spine
    assert inorder(build_from_pre_in([1], [1])) == [1]
    spine = build_from_pre_in([1, 2, 3], [3, 2, 1])
    assert preorder(spine) == [1, 2, 3] and inorder(spine) == [3, 2, 1]


if __name__ == "__main__":
    main()
    print("PASS")
