"""Task 236 - Count nodes at distance K from target in binary tree. (Tier: Intermediate, CTCI Ch 4)

Distance counts edges, so k = 0 is the target itself. A BFS over the tree records
every node's parent, then a second BFS radiates outward from the target (to children
and to the parent) in k steps and counts the nodes reached at exactly k. O(n) time.

Conventions: the first node with val == target in level order is used when values
repeat; a missing target, an empty tree, or k beyond the tree's diameter return 0.
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


def count_nodes_at_distance(root, target_val, k):
    if root is None or k < 0:
        return 0
    parent = {root: None}
    target = None
    q = deque([root])
    while q:                                      # level order also picks the target
        node = q.popleft()
        if target is None and node.val == target_val:
            target = node
        for child in (node.left, node.right):
            if child is not None:
                parent[child] = node
                q.append(child)
    if target is None:
        return 0
    seen = {target}
    frontier = [target]                           # nodes exactly `step` edges away
    for _ in range(k):
        next_frontier = []
        for node in frontier:
            for nxt in (node.left, node.right, parent[node]):
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    next_frontier.append(nxt)
        frontier = next_frontier
        if not frontier:
            return 0
    return len(frontier)


def find_path_to(root, target_val):
    """BFS path root -> first node with target_val (same order as the main function)."""
    q = deque([(root, [root])])
    while q:
        node, path = q.popleft()
        if node.val == target_val:
            return path
        for child in (node.left, node.right):
            if child is not None:
                q.append((child, path + [child]))
    return None


def brute_count(root, target_val, k):
    """O(n * h) reference: distance via common prefix of root paths, one node at a time."""
    if root is None or k < 0:
        return 0
    target_path = find_path_to(root, target_val)
    if target_path is None:
        return 0
    count = 0
    stack = [(root, [root])]
    while stack:
        node, path = stack.pop()
        common = 0
        while (common < len(path) and common < len(target_path)
               and path[common] is target_path[common]):
            common += 1
        if (len(path) - common) + (len(target_path) - common) == k:
            count += 1
        if node.left is not None:
            stack.append((node.left, path + [node.left]))
        if node.right is not None:
            stack.append((node.right, path + [node.right]))
    return count


def main():
    classic = build_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    assert count_nodes_at_distance(classic, 5, 2) == 3        # {1, 7, 4}
    assert count_nodes_at_distance(classic, 5, 1) == 3        # {3, 6, 2}
    assert count_nodes_at_distance(classic, 5, 3) == 2        # {0, 8}
    assert count_nodes_at_distance(classic, 3, 2) == 4        # {6, 2, 0, 8}
    assert count_nodes_at_distance(classic, 5, 0) == 1        # target itself
    assert count_nodes_at_distance(classic, 3, 0) == 1

    # target = root, k = depth of the tree in edges
    assert count_nodes_at_distance(classic, 3, 3) == 2        # {7, 4}, the deepest nodes
    skewed = build_tree([1, 2, None, 3, None, 4])             # chain 1-2-3-4
    assert count_nodes_at_distance(skewed, 1, 3) == 1         # deepest node, root target

    # nonexistent target and k beyond the diameter
    assert count_nodes_at_distance(classic, 99, 1) == 0
    assert count_nodes_at_distance(skewed, 1, 4) == 0
    assert count_nodes_at_distance(classic, 5, 10) == 0

    # single node / empty tree
    assert count_nodes_at_distance(build_tree([7]), 7, 0) == 1
    assert count_nodes_at_distance(build_tree([7]), 7, 1) == 0
    assert count_nodes_at_distance(None, 1, 0) == 0
    assert count_nodes_at_distance(build_tree([1, 2, 3]), 99, 0) == 0

    # repeated values: the first node in level order is the target
    assert count_nodes_at_distance(build_tree([1, 2, 1]), 1, 1) == 2
    assert count_nodes_at_distance(build_tree([1, 2, 1]), 1, 2) == 0

    import random

    rng = random.Random(236)
    for _ in range(200):                          # cross-check vs path-based brute force
        n = rng.randint(0, 12)
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
        reachable = []
        stack = [root] if root is not None else []
        while stack:
            node = stack.pop()
            reachable.append(node)
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        target_val = rng.choice(reachable).val if reachable else 1
        for k in range(0, n + 3):
            assert count_nodes_at_distance(root, target_val, k) == \
                brute_count(root, target_val, k), (target_val, k)


if __name__ == "__main__":
    main()
    print("PASS")
