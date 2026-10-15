"""Task 477 - Implement B+ tree leaf-range query and insert basics. (Tier: Advanced, CTCI Ch 14 — Databases)

B+ tree with a configurable leaf capacity: insert splits leaves on overflow, promotes a
separator (the first key of the new right leaf) and splits internal nodes recursively (middle
child promoted), keeping all leaves at equal depth and linked left-to-right. range(lo, hi)
descends to the leaf that can contain lo and then walks the leaf chain collecting keys in
[lo, hi]. Every insert/scan is checked against a sorted-list brute-force oracle, including the
ascending boundary-split cascade 1..12 (which is hand-traced to yield root key [5], six leaves
and height 3 for capacity 3) and a randomized workload.
"""


class Node:
    def __init__(self, is_leaf):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []      # internal nodes only
        self.next = None        # leaf nodes only: right sibling


class BPlusTree:
    def __init__(self, max_keys=3):
        self.max_keys = max_keys
        self.root = Node(True)
        self.size = 0

    # ---------- lookups ----------
    def _leaf_for(self, key):
        node = self.root
        while not node.is_leaf:
            idx = 0
            while idx < len(node.keys) and node.keys[idx] <= key:
                idx += 1
            node = node.children[idx]
        return node

    def contains(self, key):
        leaf = self._leaf_for(key)
        return key in leaf.keys

    def all_keys(self):
        leaf = self._leaf_for(-10 ** 9)
        out = []
        while leaf is not None:
            out.extend(leaf.keys)
            leaf = leaf.next
        return out

    def range(self, lo, hi):
        if lo > hi:
            return []
        leaf = self._leaf_for(lo)
        out = []
        while leaf is not None:
            for key in leaf.keys:
                if key > hi:
                    return out
                if key >= lo:
                    out.append(key)
            leaf = leaf.next
        return out

    # ---------- insertion ----------
    def insert(self, key):
        if self.contains(key):
            return False
        res = self._insert(self.root, key)
        if res is not None:
            sep, right = res
            new_root = Node(False)
            new_root.keys = [sep]
            new_root.children = [self.root, right]
            self.root = new_root
        self.size += 1
        return True

    def _insert(self, node, key):
        if node.is_leaf:
            pos = 0
            while pos < len(node.keys) and node.keys[pos] < key:
                pos += 1
            node.keys.insert(pos, key)
            if len(node.keys) <= self.max_keys:
                return None
            mid = len(node.keys) // 2
            right = Node(True)
            right.keys = node.keys[mid:]
            node.keys = node.keys[:mid]
            right.next = node.next
            node.next = right
            return (right.keys[0], right)
        idx = 0
        while idx < len(node.keys) and node.keys[idx] <= key:
            idx += 1
        res = self._insert(node.children[idx], key)
        if res is None:
            return None
        sep, right = res
        node.keys.insert(idx, sep)
        node.children.insert(idx + 1, right)
        if len(node.keys) <= self.max_keys:
            return None
        mid = len(node.children) // 2
        promoted = node.keys[mid - 1]
        right_node = Node(False)
        right_node.children = node.children[mid:]
        right_node.keys = node.keys[mid:]
        node.children = node.children[:mid]
        node.keys = node.keys[:mid - 1]
        return (promoted, right_node)


def lcg(seed):
    return (seed * 75 + 74) % 65537


def min_key(node):
    while not node.is_leaf:
        node = node.children[0]
    return node.keys[0]


def collect_leaves(node):
    if node.is_leaf:
        return [node]
    out = []
    for child in node.children:
        out.extend(collect_leaves(child))
    return out


def chain_leaves(tree):
    leaf = tree._leaf_for(-10 ** 9)
    out = []
    while leaf is not None:
        out.append(leaf)
        leaf = leaf.next
    return out


def check_invariants(tree):
    depths = []

    def walk(node, depth):
        assert node.keys == sorted(node.keys), "keys sorted"
        assert len(node.keys) <= tree.max_keys, "node capacity"
        if node.is_leaf:
            depths.append(depth)
            return
        assert len(node.children) == len(node.keys) + 1, "children = keys + 1"
        for i, child in enumerate(node.children[1:]):
            assert min_key(child) == node.keys[i], "separator equals right subtree minimum"
        for child in node.children:
            walk(child, depth + 1)

    walk(tree.root, 0)
    assert len(set(depths)) == 1, "all leaves at equal depth"
    by_chain = chain_leaves(tree)
    by_dfs = collect_leaves(tree.root)
    assert len(by_chain) == len(by_dfs), "leaf chain covers every leaf"
    for a, b in zip(by_chain, by_dfs):
        assert a.keys == b.keys, "leaf chain order matches tree order"


def leaf_count(tree):
    return len(collect_leaves(tree.root))


def height(tree):
    depth = 1
    node = tree.root
    while not node.is_leaf:
        node = node.children[0]
        depth += 1
    return depth


def main():
    # Ascending inserts force a deterministic split cascade (hand-traced).
    tree = BPlusTree(3)
    brute = []
    for key in range(1, 13):
        assert tree.insert(key)
        brute.append(key)
        check_invariants(tree)
        for lo, hi in [(1, 12), (4, 9), (5, 5), (3, 3), (10, 20), (0, 0), (12, 12), (2, 11)]:
            assert tree.range(lo, hi) == [k for k in brute if lo <= k <= hi], (lo, hi)
    assert tree.root.keys == [5]
    assert leaf_count(tree) == 6
    assert height(tree) == 3
    assert tree.all_keys() == brute
    assert not tree.insert(7), "duplicates are rejected"
    assert tree.all_keys() == brute and tree.size == 12

    # Descending inserts exercise the left-splitting mirror image.
    tree2 = BPlusTree(3)
    brute2 = []
    for key in range(12, 0, -1):
        assert tree2.insert(key)
        brute2.append(key)
        check_invariants(tree2)
        assert tree2.range(1, 12) == sorted(brute2)
    assert tree2.all_keys() == list(range(1, 13))

    # Randomized inserts and range scans against the sorted-list oracle.
    tree3 = BPlusTree(4)
    inserted = set()
    seed = 4242
    for _ in range(40):
        seed = lcg(seed)
        key = seed % 64
        if key in inserted:
            assert not tree3.insert(key)
        else:
            assert tree3.insert(key)
            inserted.add(key)
        check_invariants(tree3)
    keys_sorted = sorted(inserted)
    assert len(keys_sorted) > 25
    assert tree3.all_keys() == keys_sorted
    for _ in range(80):
        seed = lcg(seed)
        lo = seed % 70
        seed = lcg(seed)
        hi = lo + seed % (70 - lo)
        expected = [k for k in keys_sorted if lo <= k <= hi]
        assert tree3.range(lo, hi) == expected, (lo, hi)
    assert tree3.range(65, 63) == []
    assert tree3.range(200, 300) == []
    assert tree3.range(0, 0) == ([0] if 0 in inserted else [])


if __name__ == "__main__":
    main()
    print("PASS")
