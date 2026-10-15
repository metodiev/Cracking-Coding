"""Task 144 - Convert sorted array to balanced BST. (Tier: Beginner, CTCI Ch 4)

The middle element becomes the root, so both halves stay equally deep and the
resulting tree has the minimal possible height for those values.
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


def sorted_array_to_bst(nums):
    """Height-balanced BST from ascending nums; None for an empty array."""
    def build(lo, hi):
        if lo > hi:
            return None
        mid = (lo + hi) // 2                      # left-biased middle keeps it stable
        node = TreeNode(nums[mid])
        node.left = build(lo, mid - 1)
        node.right = build(mid + 1, hi)
        return node

    return build(0, len(nums) - 1)


def inorder(root):
    """Left - node - right values; sorted output proves the BST ordering."""
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


def height(node):
    if node is None:
        return 0
    return 1 + max(height(node.left), height(node.right))


def is_height_balanced(root):
    """Check the requirement directly: every node's subtree heights differ by <= 1."""
    if root is None:
        return True
    return (abs(height(root.left) - height(root.right)) <= 1
            and is_height_balanced(root.left) and is_height_balanced(root.right))


def main():
    assert sorted_array_to_bst([]) is None
    assert sorted_array_to_bst([5]).val == 5
    small = sorted_array_to_bst([1, 2, 3])
    assert (small.val, small.left.val, small.right.val) == (2, 1, 3)
    for nums in ([], [1], [1, 2, 3], [1, 2, 3, 4, 5, 6], [-10, -3, 0, 5, 9],
                 list(range(1, 8)), [1, 1, 1], [0, 1000000]):
        tree = sorted_array_to_bst(nums)
        assert inorder(tree) == nums, nums              # BST property: sorted inorder
        assert is_height_balanced(tree), nums
        if nums:
            assert tree.val == nums[(len(nums) - 1) // 2], nums   # middle element is the root
    assert height(sorted_array_to_bst(list(range(1, 8)))) == 3
    assert height(sorted_array_to_bst([-10, -3, 0, 5, 9])) == 3
    assert height(sorted_array_to_bst([1, 2, 3, 4, 5, 6])) == 3


if __name__ == "__main__":
    main()
    print("PASS")
