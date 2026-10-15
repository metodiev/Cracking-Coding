"""Task 135 - Check if two binary trees are identical. (Tier: Beginner, CTCI Ch 4)"""

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


def are_identical(a, b):
    """Same shape AND same values, recursively."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return (a.val == b.val
            and are_identical(a.left, b.left)
            and are_identical(a.right, b.right))


def main():
    t1 = build_tree([1, 2, 3, None, 4])
    t2 = build_tree([1, 2, 3, None, 4])
    t3 = build_tree([1, 2, 3, 4])
    assert are_identical(t1, t2)
    assert not are_identical(t1, t3)          # different shape
    assert not are_identical(build_tree([1, 2]), build_tree([1, 3]))   # different value
    assert are_identical(None, None)
    assert not are_identical(build_tree([1]), None)
    assert are_identical(build_tree([7]), build_tree([7]))


if __name__ == "__main__":
    main()
    print("PASS")
