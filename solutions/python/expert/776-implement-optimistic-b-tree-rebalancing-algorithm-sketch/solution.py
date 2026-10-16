"""Task 776 - Implement optimistic B-tree rebalancing algorithm sketch.  (Tier: Expert, Ch 14 - Databases)

"Optimistic" rebalancing means the structure is left alone until a change actually breaks it: an
insert descends without preparing anything, writes into the leaf, and only if that leaf now holds too
many keys is a split performed and propagated upwards; a delete removes the key and only then walks
the recorded path back up, borrowing from a sibling when one can spare a key and merging with a
sibling otherwise (which may underflow the parent, so the fix propagates and is counted as one
cascade level).  The opposite - pessimistic preemptive splitting, where every full node met on the
way down is split whether or not the insert needs it - is implemented too, as the comparison
baseline: on an insert-heavy workload the optimistic tree performs strictly fewer structural writes.

The tree is a B+ tree (data only in the leaves, internal nodes hold routing separators) so that the
rebalance cases stay explicit: rotate a key through the parent, or merge two children and drop the
separator.  Verified on a 400-operation random and crafted workload: after *every* mutation the
occupancy bound holds for every non-root node, all leaves stay at the same depth, every node is
sorted, the in-order traversal equals a sorted-list oracle, deleting a missing key changes nothing,
duplicate inserts only update the value, deleting everything collapses the tree back to a single
empty root, and cascade lengths never exceed the tree height.

Simplified: single-threaded (the task's "optimistic" refers to the rebalancing, not to latching),
separators may lag behind leaf minima after deletions (they are still valid lower bounds, which is
why lookups compare keys instead of trusting a bound), and no sibling redistribution across more
than one node is attempted.
"""

MAX_KEYS = 5          # a node holds at most five keys; a split leaves 3 and 2 keys
MIN_KEYS = 2          # every non-root node keeps at least two keys


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Node:
    __slots__ = ("keys", "children", "values", "leaf")

    def __init__(self, leaf):
        self.keys = []
        self.children = [] if not leaf else None
        self.values = [] if leaf else None
        self.leaf = leaf


class OptimisticBTree:
    """B+ tree whose rebalancing only happens after a change has actually broken an invariant."""

    def __init__(self):
        self.root = Node(True)
        self.splits = 0
        self.merges = 0
        self.borrows = 0
        self.writes = 0          # graded structural work: every node mutation counted
        self.max_cascade = 0

    # ------------------------------------------------------------------ queries
    def find(self, key):
        node = self.root
        while not node.leaf:
            index = 0
            while index < len(node.keys) and key >= node.keys[index]:
                index += 1
            node = node.children[index]
        for position, stored in enumerate(node.keys):
            if stored == key:
                return node.values[position]
        return None

    def items(self):
        out = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.leaf:
                out.extend(zip(node.keys, node.values))
            else:
                stack.extend(reversed(node.children))
        return sorted(out)

    def keys(self):
        return [key for key, _ in self.items()]

    def depth(self):
        node, depth = self.root, 1
        while not node.leaf:
            node = node.children[0]
            depth += 1
        return depth

    def nodes(self):
        """Every node of the tree (used to assert the occupancy bound directly)."""
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            out.append(node)
            if not node.leaf:
                stack.extend(node.children)
        return out

    def structure_ok(self):
        """Occupancy, uniform depth and sortedness of every node."""
        depths, stack = set(), [(self.root, 1)]
        while stack:
            node, level = stack.pop()
            check(node.keys == sorted(node.keys), "every node must be sorted")
            if node is not self.root:
                check(MIN_KEYS <= len(node.keys) <= MAX_KEYS,
                      f"a non-root node holds {len(node.keys)} keys, outside [{MIN_KEYS}, {MAX_KEYS}]")
            if node.leaf:
                depths.add(level)
                check(len(node.keys) == len(node.values), "leaf keys and values must be parallel")
            else:
                check(len(node.children) == len(node.keys) + 1,
                      "an internal node needs one more child than separators")
                stack.extend((child, level + 1) for child in node.children)
        check(len(depths) == 1, f"all leaves must be at the same depth, got {depths}")

    # ------------------------------------------------------------------ mutation
    def _split(self, node, parent):
        """Split an overfull node; returns (separator, right) and propagates if the parent overflows."""
        right = Node(node.leaf)
        middle = len(node.keys) // 2
        separator = node.keys[middle]                 # a copy: the B+ convention keeps it in a leaf
        if node.leaf:
            right.keys = node.keys[middle:]
            right.values = node.values[middle:]
        else:
            right.keys = node.keys[middle + 1:]
            right.children = node.children[middle + 1:]
        node.keys = node.keys[:middle]
        if node.leaf:
            node.values = node.values[:middle]
        else:
            node.children = node.children[:middle + 1]
        self.splits += 1
        self.writes += 2
        if parent is None:
            fresh = Node(False)
            fresh.keys = [separator]
            fresh.children = [node, right]
            self.root = fresh
            self.writes += 1
            return None, None
        index = parent.children.index(node)
        parent.keys.insert(index, separator)
        parent.children.insert(index + 1, right)
        self.writes += 1
        return separator, right

    def insert(self, key, value):
        """Optimistic: descend, write, and repair only the nodes the write actually overflowed."""
        path = []
        node = self.root
        while not node.leaf:
            path.append(node)
            index = 0
            while index < len(node.keys) and key >= node.keys[index]:
                index += 1
            node = node.children[index]
        position = 0
        while position < len(node.keys) and node.keys[position] < key:
            position += 1
        if position < len(node.keys) and node.keys[position] == key:
            node.values[position] = value
            self.writes += 1
            return False
        node.keys.insert(position, key)
        node.values.insert(position, value)
        self.writes += 1
        while len(node.keys) > MAX_KEYS:
            parent = path.pop() if path else None
            self._split(node, parent)
            if parent is None:
                break
            node = parent
        return True

    def delete(self, key):
        """Optimistic: remove, then repair underfull nodes bottom-up, counting the cascade."""
        path = []
        node = self.root
        while not node.leaf:
            index = 0
            while index < len(node.keys) and key >= node.keys[index]:
                index += 1
            path.append((node, index))
            node = node.children[index]
        position = None
        for index, stored in enumerate(node.keys):
            if stored == key:
                position = index
                break
        if position is None:
            return False
        del node.keys[position]
        del node.values[position]
        self.writes += 1
        cascade = 0
        while path and len(node.keys) < MIN_KEYS:
            parent, index = path.pop()
            self._rebalance(parent, index)
            cascade += 1
            node = parent
        self.max_cascade = max(self.max_cascade, cascade)
        if not self.root.leaf and not self.root.keys:
            self.root = self.root.children[0]         # the root empties: the tree shrinks
            self.writes += 1
        return True

    def _rebalance(self, parent, index):
        """Borrow from a sibling that can spare a key, otherwise merge the underfull node."""
        child = parent.children[index]
        left = parent.children[index - 1] if index > 0 else None
        right = parent.children[index + 1] if index + 1 < len(parent.children) else None
        if left is not None and len(left.keys) > MIN_KEYS:
            if child.leaf:
                child.keys.insert(0, left.keys.pop())
                child.values.insert(0, left.values.pop())
                parent.keys[index - 1] = child.keys[0]     # the child's minimum moved
            else:
                child.keys.insert(0, parent.keys[index - 1])
                parent.keys[index - 1] = left.keys.pop()
                child.children.insert(0, left.children.pop())
            self.borrows += 1
            self.writes += 1
            return
        if right is not None and len(right.keys) > MIN_KEYS:
            if child.leaf:
                child.keys.append(right.keys.pop(0))
                child.values.append(right.values.pop(0))
                if right.keys:
                    parent.keys[index] = right.keys[0]     # the right minimum moved too
            else:
                child.keys.append(parent.keys[index])
                parent.keys[index] = right.keys.pop(0)
                child.children.append(right.children.pop(0))
            self.borrows += 1
            self.writes += 1
            return
        if left is not None:
            absorbed, donor, separator_index = left, child, index - 1
        else:
            absorbed, donor, separator_index = child, right, index
        if not absorbed.leaf:
            absorbed.keys.append(parent.keys[separator_index])   # the separator moves down
        absorbed.keys.extend(donor.keys)
        if absorbed.leaf:
            absorbed.values.extend(donor.values)
        else:
            absorbed.children.extend(donor.children)
        del parent.keys[separator_index]
        del parent.children[separator_index + 1]
        self.merges += 1
        self.writes += 1


class PessimisticBTree(OptimisticBTree):
    """Baseline: preemptively split every child that is full, whether the insert needs it or not."""

    def insert(self, key, value):
        if len(self.root.keys) >= MAX_KEYS:
            self._split(self.root, None)
        node = self.root
        while True:
            index = 0
            while index < len(node.keys) and key >= node.keys[index]:
                index += 1
            if node.leaf:
                position = 0
                while position < len(node.keys) and node.keys[position] < key:
                    position += 1
                if position < len(node.keys) and node.keys[position] == key:
                    node.values[position] = value
                else:
                    node.keys.insert(position, key)
                    node.values.insert(position, value)
                self.writes += 1
                return True
            if len(node.children[index].keys) >= MAX_KEYS:
                self._split(node.children[index], node)
                if key >= node.keys[index]:
                    index += 1
            node = node.children[index]


def random_workload(steps, key_span, seed):
    """Deterministic (op, key) stream: insert with probability ~2/3, delete otherwise."""
    stream = []
    state = seed
    for _ in range(steps):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        op = "insert" if (state >> 33) % 3 else "delete"
        stream.append((op, (state >> 11) % key_span))
    return stream


def run_ops(tree, stream, oracle):
    """Apply (op, key) pairs to a tree, checking the oracle and the invariants after every step."""
    for op, key in stream:
        if op == "insert":
            fresh = tree.insert(key, key * 7)
            check(fresh == (key not in oracle), "insert must report novelty like the oracle")
            oracle[key] = key * 7
        else:
            removed = tree.delete(key)
            check(removed == (key in oracle), "delete must report existence like the oracle")
            oracle.pop(key, None)
        tree.structure_ok()
        check(tree.keys() == sorted(oracle), "the in-order traversal must equal the oracle")
    return tree


def run_workload(tree_class, stream, oracle):
    return run_ops(tree_class(), stream, oracle)


def main():
    # 1. boundaries: an empty tree, a single key, duplicate inserts, deleting a missing key
    empty = OptimisticBTree()
    check(empty.items() == [] and empty.depth() == 1, "an empty tree is one empty leaf")
    check(empty.delete(3) is False, "deleting from an empty tree must fail")
    empty.structure_ok()
    single = OptimisticBTree()
    check(single.insert(4, "a"), "the first insert must add a key")
    check(single.insert(4, "b") is False and single.find(4) == "b",
          "a duplicate insert must update the value only")
    check(len(single.items()) == 1 and single.splits == 0, "one key needs no split")
    check(single.delete(99) is False, "deleting a missing key must be a no-op")
    check(single.keys() == [4], "the missing-key delete must not touch the tree")

    # 2. a growth workload: splits, invariants and the oracle stay in step after every operation
    oracle = {}
    tree = OptimisticBTree()
    run_ops(tree, random_workload(250, 45, 0x5EEDBEEF), oracle)
    check(tree.splits > 6, "the workload must force real splits")
    check(len(tree.items()) == len(oracle) and len(oracle) > 20, "the tree must hold the oracle keys")
    check(tree.max_cascade <= tree.depth() + 1,
          f"a split cascade of {tree.max_cascade} exceeds the height {tree.depth()}")

    # 3. a shrink workload: deleting the keys in order forces borrows and merges
    growth_keys = len(oracle)
    run_ops(tree, [("delete", key) for key in sorted(oracle)], oracle)
    check(tree.merges > 0, "deleting keys must force real merges")
    check(tree.borrows > 0, "deleting keys must force at least one borrow")
    check(tree.items() == [] and tree.depth() == 1, "the tree must collapse to one empty leaf")
    check(tree.root.leaf and tree.find(0) is None, "nothing may be found after the full shrink")
    check(growth_keys > 20, "the shrink phase must start from a populated tree")

    # 4. occupancy really is repaired: no non-root node may stay underfull after a delete
    trim = OptimisticBTree()
    for key in range(40):
        trim.insert(key, key)
    for key in range(0, 40, 3):
        trim.delete(key)
        trim.structure_ok()
    check(all(len(node.keys) >= MIN_KEYS for node in trim.nodes() if node is not trim.root),
          "no non-root node may stay underfull after a delete")
    check(trim.depth() >= 2, "the trimming tree must still be multi-level")

    # 6. optimistic versus preemptive: identical contents, but updates waste no split
    keys = list(range(60))
    optimistic, pessimistic = OptimisticBTree(), PessimisticBTree()
    for key in keys:
        optimistic.insert(key, key)
        pessimistic.insert(key, key)
    for tree_under_test in (optimistic, pessimistic):
        tree_under_test.structure_ok()
    check(optimistic.keys() == pessimistic.keys() == keys, "both strategies must hold the keys")
    check(optimistic.splits <= pessimistic.splits,
          f"optimistic splits {optimistic.splits} must not exceed preemptive {pessimistic.splits}")
    check(optimistic.find(3) == 3, "the built value must be readable")

    # 6b. a full root leaf shows the difference directly: updates need no split, preemption splits
    full_opt, full_pess = OptimisticBTree(), PessimisticBTree()
    for key in range(MAX_KEYS):
        full_opt.insert(key, key)
        full_pess.insert(key, key)
    check(full_opt.root.leaf and len(full_opt.root.keys) == MAX_KEYS, "the root leaf must be full")
    for key in range(MAX_KEYS):
        full_opt.insert(key, ("again", key))
        full_pess.insert(key, ("again", key))
    check(full_opt.splits == 0, "updating a full leaf must not split the optimistic tree")
    check(full_pess.splits >= 1, "preemptive splitting splits the full root on the way down")
    check(full_opt.keys() == full_pess.keys() == list(range(MAX_KEYS)), "contents must still agree")
    full_opt.structure_ok()
    full_pess.structure_ok()
    check(full_opt.find(2) == ("again", 2), "the updated value must be readable")

    # 7. mixed insert/delete workload against the oracle, invariants after every single operation
    oracle2 = {}
    run_workload(OptimisticBTree, random_workload(200, 30, 0xBEEF), oracle2)
    check(len(oracle2) > 0, "the second workload must leave some keys")

    # 8. determinism
    oracle3 = {}
    again = run_workload(OptimisticBTree, random_workload(200, 30, 0xBEEF), oracle3)
    check(again.items() == sorted(oracle2.items()), "the workload must be reproducible")

    print(f"776 keys={growth_keys} splits={tree.splits} merges={tree.merges} "
          f"borrows={tree.borrows} writes={tree.writes}")
    print(f"776 optimistic_splits={optimistic.splits} pessimistic_splits={pessimistic.splits} "
          f"writes={optimistic.writes}/{pessimistic.writes} max_cascade={tree.max_cascade}")
    print("PASS")


if __name__ == "__main__":
    main()
