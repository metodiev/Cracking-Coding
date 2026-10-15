"""Task 148 - Serialize and deserialize binary tree (preorder). (Tier: Beginner, CTCI Ch 4)

Format (identical in all four languages): preorder values separated by commas,
with "#" marking every missing child, e.g. "1,2,#,#,3,#,#". An empty tree is "#"
and a single node is "7,#,#", so every tree with n nodes uses exactly 2n + 1
tokens and the stream can be rebuilt by a single left-to-right pass.
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


def serialize(root):
    """Preorder walk that also emits a sentinel for each missing child."""
    parts = []

    def walk(node):
        if node is None:
            parts.append("#")
            return
        parts.append(str(node.val))
        walk(node.left)
        walk(node.right)

    walk(root)
    return ",".join(parts)


def deserialize(data):
    """Rebuild the tree from the preorder stream produced by serialize."""
    if data == "":
        return None
    tokens = data.split(",")
    pos = [0]                                 # mutable cursor shared with build

    def build():
        token = tokens[pos[0]]
        pos[0] += 1
        if token == "#":
            return None
        node = TreeNode(int(token))
        node.left = build()                   # the left subtree comes next
        node.right = build()
        return node

    return build()


def preorder_values(root):
    """Preorder values only (the sentinels are dropped)."""
    if root is None:
        return []
    return [root.val] + preorder_values(root.left) + preorder_values(root.right)


def are_identical(a, b):
    """Round-trip check: same shape and same values everywhere."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.val == b.val and are_identical(a.left, b.left) and are_identical(a.right, b.right)


def main():
    assert serialize(None) == "#"
    assert deserialize("#") is None and deserialize("") is None
    assert serialize(build_tree([7])) == "7,#,#"
    assert serialize(build_tree([1, 2, None, 3])) == "1,2,3,#,#,#,#"
    assert serialize(build_tree([1, 2, 5, 3, 4, None, 6])) == "1,2,3,#,#,4,#,#,5,#,6,#,#"
    assert len(serialize(build_tree([1, 2, 3])).split(",")) == 7      # 2n + 1 tokens
    for data in ([], [7], [1, 2, None, 3], [1, 2, 5, 3, 4, None, 6], [1, 2, 3, 4, 5, None, 6]):
        tree = build_tree(data)
        text = serialize(tree)
        assert [str(v) for v in preorder_values(tree)] == [t for t in text.split(",") if t != "#"], data
        rebuilt = deserialize(text)
        assert are_identical(tree, rebuilt), data
        assert serialize(rebuilt) == text, data                        # string round trip
        if tree is not None:
            assert rebuilt is not tree, data                           # a fresh structure


if __name__ == "__main__":
    main()
    print("PASS")
