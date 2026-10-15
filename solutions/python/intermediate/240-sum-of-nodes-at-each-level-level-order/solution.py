"""Task 240 - Sum of nodes at each level (level order). (Tier: Intermediate, CTCI Ch 4)

Level-order BFS with a per-level snapshot sums each level independently: one pass
over the tree, O(n) time. The empty tree yields [].

The recursive level_sums_dfs helper accumulates sums by depth and is used as an
independent cross-check of the BFS result.
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


def level_sums(root):
    if root is None:
        return []
    sums = []
    q = deque([root])
    while q:
        total = 0
        for _ in range(len(q)):                   # snapshot of the current level size
            node = q.popleft()
            total += node.val
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        sums.append(total)
    return sums


def level_sums_dfs(root):
    """Independent helper: accumulate sums by depth with a DFS."""
    sums = []

    def visit(node, depth):
        if node is None:
            return
        if depth == len(sums):
            sums.append(0)
        sums[depth] += node.val
        visit(node.left, depth + 1)
        visit(node.right, depth + 1)

    visit(root, 0)
    return sums


def main():
    assert level_sums(build_tree([1, 2, 3, 4, 5, None, 6])) == [1, 5, 15]  # {1}, {2,3}, {4,5,6}
    assert level_sums(build_tree([1, 2, 3])) == [1, 5]
    assert level_sums(build_tree([1])) == [1]
    assert level_sums(None) == []
    assert level_sums(build_tree([1, 2, None, 3, None, 4])) == [1, 2, 3, 4]
    assert level_sums(build_tree([-1, -2, -3])) == [-1, -5]
    assert level_sums(build_tree([0, 0, 0, 0])) == [0, 0, 0]

    import random

    rng = random.Random(240)
    for _ in range(200):                          # BFS result == DFS depth accumulation
        n = rng.randint(0, 14)
        nodes = [TreeNode(rng.randint(-20, 20)) for _ in range(n)]
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
        assert level_sums(root) == level_sums_dfs(root)


if __name__ == "__main__":
    main()
    print("PASS")
