"""Task 143 - Symmetric tree check (mirror). (Tier: Beginner, CTCI Ch 4)

A tree is symmetric when its left and right subtrees are mirror images.
Traversal order alone is not enough: the pairing matters, so compare
a.left with b.right and a.right with b.left.
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


def is_mirror(a, b):
    """Two subtrees mirror each other when their mirrored children match."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.val == b.val and is_mirror(a.left, b.right) and is_mirror(a.right, b.left)


def is_symmetric(root):
    """Recursive: root is symmetric when its two subtrees mirror each other."""
    if root is None:
        return True
    return is_mirror(root.left, root.right)


def is_symmetric_iterative(root):
    """Stack of mirrored (a, b) pairs; an empty tree is symmetric by definition."""
    if root is None:
        return True
    stack = [(root.left, root.right)]
    while stack:
        a, b = stack.pop()
        if a is None and b is None:
            continue
        if a is None or b is None or a.val != b.val:
            return False
        stack.append((a.left, b.right))
        stack.append((a.right, b.left))
    return True


def mirror(root):
    """Independent oracle used below: build the mirror image of a tree."""
    if root is None:
        return None
    return TreeNode(root.val, mirror(root.right), mirror(root.left))


def identical(a, b):
    """Structural equality: same shape and same values everywhere."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.val == b.val and identical(a.left, b.left) and identical(a.right, b.right)


def main():
    assert is_symmetric(None) and is_symmetric(build_tree([]))
    assert is_symmetric(build_tree([1]))
    assert is_symmetric(build_tree([1, 2, 2]))
    assert is_symmetric(build_tree([1, 2, 2, 3, 4, 4, 3]))
    # both 3s hang as RIGHT children, so the mirrored pairing fails -> not symmetric
    assert not is_symmetric(build_tree([1, 2, 2, None, 3, None, 3]))
    # mirrored placement: right child under the left 2, left child under the right 2
    assert is_symmetric(build_tree([1, 2, 2, None, 3, 3, None]))
    assert not is_symmetric(build_tree([1, 2, 3]))
    # cross-check both implementations against "tree equals its own mirror image"
    for data in ([], [1], [1, 2, 2], [1, 2, 2, 3, 4, 4, 3], [1, 2, 2, None, 3, None, 3],
                 [1, 2, 2, None, 3, 3, None], [1, 2, 3], [1, 2, 2, 3, None, None, 3]):
        tree = build_tree(data)
        expected = identical(tree, mirror(tree))
        assert is_symmetric(tree) == expected, data
        assert is_symmetric_iterative(tree) == expected, data


if __name__ == "__main__":
    main()
    print("PASS")
