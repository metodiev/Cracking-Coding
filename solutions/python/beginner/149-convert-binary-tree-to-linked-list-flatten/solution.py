"""Task 149 - Convert binary tree to linked list (flatten). (Tier: Beginner, CTCI Ch 4)

LeetCode 114: rewrite the tree IN PLACE into its preorder order, using the right
pointers as the "next" links and leaving every left pointer empty.
[1,2,5,3,4,None,6] becomes the chain 1 -> 2 -> 3 -> 4 -> 5 -> 6.
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


def flatten(root):
    """Recursive: flatten both subtrees, then splice the left chain in front of the right one."""
    if root is None:
        return None
    left, right = root.left, root.right
    flatten(left)
    flatten(right)
    root.left = None
    root.right = left
    node = root
    while node.right is not None:            # walk to the tail of the flattened left part
        node = node.right
    node.right = right
    return root


def flatten_iterative(root):
    """Preorder with an explicit stack: link each node to the previously visited one."""
    if root is None:
        return None
    stack = [root]
    prev = None
    while stack:
        node = stack.pop()
        if prev is not None:
            prev.left = None
            prev.right = node
        if node.right:
            stack.append(node.right)         # right first so the left side is visited first
        if node.left:
            stack.append(node.left)
        prev = node
    return root


def preorder_values(root):
    if root is None:
        return []
    return [root.val] + preorder_values(root.left) + preorder_values(root.right)


def right_chain(root):
    """Values following the right pointers (the flattened preorder order)."""
    values = []
    while root is not None:
        values.append(root.val)
        root = root.right
    return values


def all_left_null(node):
    """After flattening nothing may remain on a left pointer."""
    while node is not None:
        if node.left is not None:
            return False
        node = node.right
    return True


def main():
    assert flatten(None) is None and flatten_iterative(None) is None
    assert right_chain(flatten(build_tree([7]))) == [7]
    sample = build_tree([1, 2, 5, 3, 4, None, 6])
    assert right_chain(flatten(sample)) == [1, 2, 3, 4, 5, 6]
    for data in ([], [7], [1, 2, 5, 3, 4, None, 6], [1, 2, None, 3], [1, None, 2],
                 [1, 2, None, 3, None, 4], [1, 2, 3, 4, 5, None, 6]):
        for flat in (flatten, flatten_iterative):
            tree = build_tree(data)
            expected = preorder_values(tree)     # preorder is the target order
            result = flat(tree)
            assert right_chain(result) == expected, (data, flat.__name__)
            assert all_left_null(result), (data, flat.__name__)


if __name__ == "__main__":
    main()
    print("PASS")
