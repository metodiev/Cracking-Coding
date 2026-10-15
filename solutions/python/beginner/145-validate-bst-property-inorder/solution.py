"""Task 145 - Validate BST property (inorder). (Tier: Beginner, CTCI Ch 4)

Two equivalent O(n) routes:
  1. min/max range recursion - every node must stay inside the open interval
     inherited from its ancestors ("long"/infinity bounds, not int bounds, so
     INT_MIN/INT_MAX values are still legal);
  2. inorder must be strictly increasing (see is_valid_bst_inorder below).
Comparing only a node with its own children is NOT enough: [5,1,4,null,null,3,6]
passes that weaker test although 3 < 5 lives in the right subtree of 5.
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


def is_valid_bst(root):
    """Range recursion: each node carries the (lo, hi) window it must fit in."""
    return valid_range(root, float("-inf"), float("inf"))


def valid_range(node, lo, hi):
    if node is None:
        return True
    if not lo < node.val < hi:
        return False
    return valid_range(node.left, lo, node.val) and valid_range(node.right, node.val, hi)


def is_valid_bst_inorder(root):
    """Alternative route: walk the tree in order with an explicit stack."""
    previous = float("-inf")
    stack = []
    node = root
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        if node.val <= previous:
            return False
        previous = node.val
        node = node.right
    return True


def main():
    assert is_valid_bst(None) and is_valid_bst_inorder(None)
    assert is_valid_bst(build_tree([2, 1, 3]))
    assert is_valid_bst(build_tree([7]))
    assert is_valid_bst(build_tree([3, 2, None, 1]))
    # classic pitfall: 4 sits in the right subtree of 5 but is smaller than 5
    assert not is_valid_bst(build_tree([5, 1, 4, None, None, 3, 6]))
    # 6 < 10 while living in the right subtree of 10
    assert not is_valid_bst(build_tree([10, 5, 15, None, None, 6, 20]))
    assert not is_valid_bst(build_tree([5, 4, 6, None, None, 3, 7]))
    # duplicates violate the strict ordering
    assert not is_valid_bst(build_tree([1, 1]))
    # range bounds must be wider than int: INT_MIN / INT_MAX are valid node values
    assert is_valid_bst(build_tree([2 ** 31 - 1, -(2 ** 31), None]))
    for data in ([2, 1, 3], [5, 1, 4, None, None, 3, 6], [10, 5, 15, None, None, 6, 20],
                 [1, 1], [5, 4, 6, None, None, 3, 7], [3, 2, None, 1], [2 ** 31 - 1, -(2 ** 31), None]):
        tree = build_tree(data)
        assert is_valid_bst(tree) == is_valid_bst_inorder(tree), data


if __name__ == "__main__":
    main()
    print("PASS")
