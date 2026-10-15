"""Task 147 - Lowest common ancestor in binary tree (general). (Tier: Beginner, CTCI Ch 4)

No BST ordering is available, so a postorder search is used: a subtree reports
the first key it finds from below; the node where both subtrees report a key is
the split point and therefore the LCA. When only one key is found (or either
key is not part of the tree at all) the answer is None.
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


def find(root, val):
    """Plain depth-first search: a general tree has no ordering to exploit."""
    if root is None or root.val == val:
        return root
    return find(root.left, val) or find(root.right, val)


def contains(root, target):
    """Identity search: is that very node object part of this tree?"""
    if root is None:
        return False
    return root is target or contains(root.left, target) or contains(root.right, target)


def lowest_common_ancestor(root, p, q):
    """Postorder LCA; None when either node is missing from the tree."""
    if root is None or p is None or q is None:
        return None
    if not contains(root, p) or not contains(root, q):
        return None
    return search(root, p, q)


def search(node, p, q):
    """Classic postorder: report a target from below, else report the split node."""
    if node is None or node is p or node is q:
        return node
    left = search(node.left, p, q)
    right = search(node.right, p, q)
    if left is not None and right is not None:
        return node                       # p and q were found in different subtrees
    return left if left is not None else right


def main():
    tree = build_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    assert lowest_common_ancestor(tree, find(tree, 5), find(tree, 1)).val == 3     # root splits them
    assert lowest_common_ancestor(tree, find(tree, 5), find(tree, 4)).val == 5     # one key is an ancestor
    assert lowest_common_ancestor(tree, find(tree, 6), find(tree, 7)).val == 5     # both in the left branch
    assert lowest_common_ancestor(tree, find(tree, 6), find(tree, 4)).val == 5     # deeper split
    assert lowest_common_ancestor(tree, find(tree, 7), find(tree, 4)).val == 2     # siblings under 2
    assert lowest_common_ancestor(tree, find(tree, 0), find(tree, 4)).val == 3     # different subtrees
    assert lowest_common_ancestor(tree, find(tree, 3), find(tree, 5)).val == 3     # one key is the root
    assert lowest_common_ancestor(tree, find(tree, 7), find(tree, 7)).val == 7     # node with itself
    assert lowest_common_ancestor(tree, find(tree, 0), find(tree, 99)) is None     # missing key
    assert lowest_common_ancestor(tree, None, find(tree, 5)) is None
    assert lowest_common_ancestor(None, find(tree, 5), find(tree, 1)) is None
    stray = TreeNode(5)                                                            # same value, other object
    assert lowest_common_ancestor(tree, find(tree, 3), stray) is None
    chain = build_tree([1, None, 2, None, 3])
    assert lowest_common_ancestor(chain, find(chain, 1), find(chain, 3)).val == 1


if __name__ == "__main__":
    main()
    print("PASS")
