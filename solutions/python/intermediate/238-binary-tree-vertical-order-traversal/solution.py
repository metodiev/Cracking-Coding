"""Task 238 - Binary tree vertical order traversal. (Tier: Intermediate, CTCI Ch 4)

Columns are numbered from the root (root = column 0, left child = column - 1, right
child = column + 1). A BFS records every node with its column: level order makes each
column read top-to-bottom, and BFS visits same-row nodes left-to-right, which settles
the ties. Columns are emitted in increasing column order. O(n log n) time (sorting
the columns) and O(n) space.

The companion vertical_order_sorted helper collects (column, row, preorder index) and
sorts, giving an independent check that the BFS ordering is deterministic.
"""

from collections import deque


class TreeNode:
    __slots__ = ("val", "left", "right")

    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    """Build a binary tree from level-order values (None marks a missing child)."""
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


def vertical_order(root):
    if root is None:
        return []
    columns = {}
    q = deque([(root, 0)])
    while q:
        node, col = q.popleft()
        columns.setdefault(col, []).append(node.val)
        if node.left:
            q.append((node.left, col - 1))
        if node.right:
            q.append((node.right, col + 1))
    return [columns[col] for col in sorted(columns)]


def vertical_order_sorted(root):
    """Independent helper: sort (column, row, preorder index) then group by column."""
    items = []

    def dfs(node, col, row):
        if node is None:
            return
        items.append((col, row, len(items), node.val))
        dfs(node.left, col - 1, row + 1)
        dfs(node.right, col + 1, row + 1)

    dfs(root, 0, 0)
    items.sort()
    result = []
    for col, _row, _seq, val in items:
        if not result or result[-1][0] != col:
            result.append((col, [val]))
        else:
            result[-1][1].append(val)
    return [values for _col, values in result]


def main():
    classic = build_tree([3, 9, 20, None, None, 15, 7])
    assert vertical_order(classic) == [[9], [3, 15], [20], [7]]
    assert vertical_order_sorted(classic) == [[9], [3, 15], [20], [7]]

    perfect = build_tree([1, 2, 3, 4, 5, 6, 7])
    assert vertical_order(perfect) == [[4], [2], [1, 5, 6], [3], [7]]
    assert vertical_order_sorted(perfect) == [[4], [2], [1, 5, 6], [3], [7]]

    deeper = build_tree([3, 9, 20, None, None, 15, 7, 8, 11])
    assert vertical_order(deeper) == [[9, 8], [3, 15], [20, 11], [7]]
    assert vertical_order_sorted(deeper) == [[9, 8], [3, 15], [20, 11], [7]]

    chain = build_tree([1, 2, None, 3, None, 4])
    assert vertical_order(chain) == [[4], [3], [2], [1]]

    assert vertical_order(build_tree([5])) == [[5]]
    assert vertical_order(None) == []

    import random

    rng = random.Random(238)
    for _ in range(200):                          # BFS order == sorted-DFS order
        n = rng.randint(0, 14)
        nodes = [TreeNode(v) for v in range(1, n + 1)]
        for i in range(n):
            if 2 * i + 1 < n:
                nodes[i].left = nodes[2 * i + 1]
            if 2 * i + 2 < n:
                nodes[i].right = nodes[2 * i + 2]

        def prune(node):
            if node is None or rng.random() < 0.25:
                return None
            node.left = prune(node.left)
            node.right = prune(node.right)
            return node

        root = prune(nodes[0]) if n else None
        assert vertical_order(root) == vertical_order_sorted(root)


if __name__ == "__main__":
    main()
    print("PASS")
