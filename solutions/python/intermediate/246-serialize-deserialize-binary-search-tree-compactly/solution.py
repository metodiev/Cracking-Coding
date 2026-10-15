"""Task 246 - Serialize/deserialize binary search tree compactly. (Tier: Intermediate, CTCI Ch 4)

A BST needs no null markers: its preorder alone is enough, because the valid child range is
bounded by the ancestors' values. Rebuild by walking the preorder with (lo, hi) bounds ->
7 values instead of 15 tokens for the 7-node tree below.
"""

class TreeNode:
    __slots__ = ("val", "left", "right")

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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

def serialize(root):
    return ",".join(str(v) for v in preorder(root))


def serialize_with_nulls(root):
    """The naive form kept for comparison: explicit '#' markers for missing children."""
    tokens = []

    def walk(node):
        if node is None:
            tokens.append("#")
            return
        tokens.append(str(node.val))
        walk(node.left)
        walk(node.right)

    walk(root)
    return ",".join(tokens)


def deserialize(data):
    values = [int(tok) for tok in data.split(",")] if data else []
    pos = [0]

    def build(lo, hi):
        if pos[0] >= len(values):
            return None
        value = values[pos[0]]
        if not (lo <= value <= hi):      # outside this subtree's allowed range
            return None
        pos[0] += 1
        node = TreeNode(value)
        node.left = build(lo, value - 1)
        node.right = build(value + 1, hi)
        return node

    return build(float("-inf"), float("inf"))


def build_bst(values):
    root = None
    for value in values:
        if root is None:
            root = TreeNode(value)
            continue
        node = root
        while True:
            if value < node.val:
                if node.left is None:
                    node.left = TreeNode(value)
                    break
                node = node.left
            else:
                if node.right is None:
                    node.right = TreeNode(value)
                    break
                node = node.right
    return root


def main():
    values = [5, 3, 8, 2, 4, 7, 9]
    root = build_bst(values)
    data = serialize(root)
    assert data == "5,3,2,4,8,7,9"
    assert serialize_with_nulls(root) == "5,3,2,#,#,4,#,#,8,7,#,#,9,#,#"

    assert len(data.split(",")) < len(serialize_with_nulls(root).split(","))   # compact
    clone = deserialize(data)
    assert preorder(clone) == preorder(root) and inorder(clone) == sorted(values)

    assert serialize(deserialize("42")) == "42"
    assert deserialize("") is None


if __name__ == "__main__":
    main()
    print("PASS")
