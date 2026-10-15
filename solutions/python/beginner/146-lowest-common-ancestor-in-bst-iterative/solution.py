"""Task 146 - Lowest common ancestor in BST (iterative). (Tier: Beginner, CTCI Ch 4)

The BST ordering turns the search into a walk: while both keys sit on the same
side of the current node, step that way. The first node where the keys split
(or where one key IS the node) is the lowest common ancestor. No extra memory:
O(h) steps, O(1) space.
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
    """Locate a node by value using the BST ordering; None when it is absent."""
    while root is not None and root.val != val:
        root = root.left if val < root.val else root.right
    return root


def lowest_common_ancestor(root, p, q):
    """Iterative split-point walk; None when a key is missing from this BST."""
    if root is None or p is None or q is None:
        return None
    node = root
    while node is not None:
        if p.val < node.val and q.val < node.val:
            node = node.left                  # both keys live in the left subtree
        elif p.val > node.val and q.val > node.val:
            node = node.right                 # both keys live in the right subtree
        else:
            return node                       # split point, or one key is node itself
    return None


def main():
    tree = build_tree([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
    assert lowest_common_ancestor(tree, find(tree, 2), find(tree, 8)).val == 6
    assert lowest_common_ancestor(tree, find(tree, 2), find(tree, 4)).val == 2   # ancestor == key
    assert lowest_common_ancestor(tree, find(tree, 0), find(tree, 5)).val == 2
    assert lowest_common_ancestor(tree, find(tree, 4), find(tree, 5)).val == 4
    assert lowest_common_ancestor(tree, find(tree, 7), find(tree, 9)).val == 8
    assert lowest_common_ancestor(tree, find(tree, 2), find(tree, 2)).val == 2   # node with itself
    assert lowest_common_ancestor(tree, find(tree, 0), find(tree, 0)).val == 0
    assert lowest_common_ancestor(tree, find(tree, 0), find(tree, 3)).val == 2
    assert lowest_common_ancestor(tree, find(tree, 0), find(tree, 99)) is None   # missing key
    assert lowest_common_ancestor(tree, None, find(tree, 8)) is None
    assert lowest_common_ancestor(None, find(tree, 2), find(tree, 8)) is None
    # skewed BST: 2 and 4 sit in opposite subtrees of the root 3; 5 is 4's ancestor
    chain = build_tree([3, 1, 5, None, 2, 4])
    assert lowest_common_ancestor(chain, find(chain, 2), find(chain, 4)).val == 3
    assert lowest_common_ancestor(chain, find(chain, 1), find(chain, 2)).val == 1
    assert lowest_common_ancestor(chain, find(chain, 4), find(chain, 5)).val == 5


if __name__ == "__main__":
    main()
    print("PASS")
