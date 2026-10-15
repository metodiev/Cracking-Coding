"""Task 245 - Reconstruct binary tree from preorder with null markers. (Tier: Intermediate, CTCI Ch 4)

Tokens like "1,2,#,#,3,#,#" where "#" marks a missing child; parse with a pointer.
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


def build_from_preorder_tokens(tokens):
    """tokens: iterable of strings/ints; '#' (or None) marks a missing child."""
    seq = list(tokens)

    def build(pos):
        if pos >= len(seq) or seq[pos] in ("#", None):
            return None, pos + 1
        node = TreeNode(int(seq[pos]))
        left, pos = build(pos + 1)
        right, pos = build(pos)
        node.left, node.right = left, right
        return node, pos

    root, _ = build(0)
    return root


def main():
    root = build_from_preorder_tokens("1,2,#,#,3,#,#".split(","))
    assert preorder(root) == [1, 2, 3]
    assert inorder(root) == [2, 1, 3]

    root2 = build_from_preorder_tokens("1,2,4,#,#,5,#,#,3,#,6,#,#".split(","))
    assert preorder(root2) == [1, 2, 4, 5, 3, 6]
    assert inorder(root2) == [4, 2, 5, 1, 3, 6]

    assert build_from_preorder_tokens(["#"]) is None
    single = build_from_preorder_tokens(["9", "#", "#"])
    assert single.val == 9 and single.left is None and single.right is None


if __name__ == "__main__":
    main()
    print("PASS")
