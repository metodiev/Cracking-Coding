"""Task 244 - Reconstruct binary tree from inorder and postorder. (Tier: Intermediate, CTCI Ch 4)

The last postorder value is the root; walk postorder backwards (right subtree first).
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


def build_from_in_post(inorder, postorder):
    index = {value: i for i, value in enumerate(inorder)}
    pos = [len(postorder) - 1]

    def build(lo, hi):
        if lo > hi:
            return None
        root_val = postorder[pos[0]]
        pos[0] -= 1
        node = TreeNode(root_val)
        mid = index[root_val]
        node.right = build(mid + 1, hi)      # right first when walking backwards
        node.left = build(lo, mid - 1)
        return node

    if not postorder or len(postorder) != len(inorder):
        raise ValueError("inorder and postorder must be non-empty and equal-length")
    return build(0, len(inorder) - 1)


def postorder_traverse(node, out=None):
    out = [] if out is None else out
    if node:
        postorder_traverse(node.left, out)
        postorder_traverse(node.right, out)
        out.append(node.val)
    return out


def main():
    ino = [9, 3, 15, 20, 7]
    post = [9, 15, 7, 20, 3]
    root = build_from_in_post(ino, post)
    assert inorder(root) == ino
    assert postorder_traverse(root) == post
    assert inorder(build_from_in_post([1], [1])) == [1]
    spine = build_from_in_post([3, 2, 1], [3, 2, 1])
    assert postorder_traverse(spine) == [3, 2, 1] and inorder(spine) == [3, 2, 1]


if __name__ == "__main__":
    main()
    print("PASS")
