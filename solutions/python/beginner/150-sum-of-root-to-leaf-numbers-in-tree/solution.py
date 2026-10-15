"""Task 150 - Sum of root-to-leaf numbers in tree. (Tier: Beginner, CTCI Ch 4)

Every root-to-leaf path spells a decimal number (1 -> 2 -> 3 means 123). The DFS
carries the prefix built so far: acc * 10 + node.val, and adds the accumulated
value whenever a leaf closes a path.
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


def sum_numbers(root):
    """Recursive DFS with the running prefix; an empty tree contributes 0."""
    def walk(node, acc):
        if node is None:
            return 0
        acc = acc * 10 + node.val
        if node.left is None and node.right is None:
            return acc                        # a leaf closes exactly one root-to-leaf number
        return walk(node.left, acc) + walk(node.right, acc)

    return walk(root, 0)


def sum_numbers_iterative(root):
    """Stack of (node, prefix) pairs; a leaf adds its prefix to the total."""
    if root is None:
        return 0
    total = 0
    stack = [(root, root.val)]
    while stack:
        node, acc = stack.pop()
        if node.left is None and node.right is None:
            total += acc
            continue
        if node.left is not None:
            stack.append((node.left, acc * 10 + node.left.val))
        if node.right is not None:
            stack.append((node.right, acc * 10 + node.right.val))
    return total


def main():
    assert sum_numbers(None) == 0 and sum_numbers_iterative(None) == 0
    assert sum_numbers(build_tree([7])) == 7
    assert sum_numbers(build_tree([1, 2, 3])) == 25                  # 12 + 13
    assert sum_numbers(build_tree([4, 9, 0, 5, 1])) == 1026          # 495 + 491 + 40
    assert sum_numbers(build_tree([1, 0, 0])) == 20                  # 10 + 10
    assert sum_numbers(build_tree([1, 2, None, 3])) == 123           # one path only
    assert sum_numbers(build_tree([1, 2, 3, 4, 5])) == 262           # 124 + 125 + 13
    for data in ([], [7], [1, 2, 3], [4, 9, 0, 5, 1], [1, 0, 0], [1, 2, None, 3], [1, 2, 3, 4, 5],
                 [1, 9, 9, None, 1]):
        tree = build_tree(data)
        assert sum_numbers(tree) == sum_numbers_iterative(tree), data


if __name__ == "__main__":
    main()
    print("PASS")
