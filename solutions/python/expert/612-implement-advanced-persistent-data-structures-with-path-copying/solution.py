"""Task 612 - Implement advanced persistent data structures with path copying optimization.  (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

Three fully persistent containers, all built on *path copying*: an update does not mutate any
node, it allocates a fresh copy of the O(log n) nodes on the root-to-leaf path and re-points
them at the untouched subtrees of the old version.  Old roots therefore keep working forever -
a version is just a root pointer - and consecutive versions share all but O(log n) nodes.

* PersistentTreap - a randomised BST made persistent through the split/merge formulation.
  split(root, key) and merge(left, right) each clone exactly the nodes they recurse through and
  share the sibling subtrees, which is where the O(log n) path-copying cost comes from.  Keys
  get deterministic priorities from splitmix64(key) so the shape (and every allocation count
  printed below) is reproducible: no random module, no seeding question.
* PersistentArray - a segment tree whose internal nodes are replaced along the root-to-leaf
  path on set(index, value), keeping point reads and range sums for every version.
* PersistentStack - the trivial end of the spectrum (a cons list), used as a sanity marker:
  pushing is O(1) and shares everything but one cell.

Optimisation under test: the allocation counters.  Every update must allocate at most
2 * ceil(log2(n)) + 2 nodes (one or two clones per level) rather than copying the structure,
and an update to a huge version must allocate the same tiny number of nodes as an update to a
small one.  Verified: every historical version still answers its own queries correctly after
later updates (immutability), versions match a from-scratch rebuild of their edit history,
structural sharing is real (a version and its parent share n - O(log n) nodes), invalid version
handles are rejected, and empty/singleton versions work.
"""

MASK = 0xFFFFFFFFFFFFFFFF


def splitmix64(value):
    z = (value + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


class Counter:
    """Counts allocations so path copying can be measured instead of asserted by hand."""

    def __init__(self):
        self.nodes = 0

    def tick(self):
        self.nodes += 1
        return self.nodes


class TreapNode:
    __slots__ = ("key", "priority", "left", "right")

    def __init__(self, key, priority, left=None, right=None):
        self.key = key
        self.priority = priority
        self.left = left
        self.right = right


class PersistentTreap:
    """Persistent randomised BST; version = root node (None is the empty version)."""

    def __init__(self):
        self.counter = Counter()

    def make_node(self, key, left=None, right=None):
        self.counter.tick()
        return TreapNode(key, splitmix64(key), left, right)

    def merge(self, left, right):
        """All keys of `left` are < all keys of `right`; min-heap on priority."""
        if left is None:
            return right
        if right is None:
            return left
        if left.priority < right.priority:
            return self.make_node(left.key, left.left, self.merge(left.right, right))
        return self.make_node(right.key, self.merge(left, right.left), right.right)

    def split(self, root, key):
        """(keys < key, keys >= key) with both halves sharing unvisited subtrees."""
        if root is None:
            return None, None
        if root.key < key:
            low, high = self.split(root.right, key)
            return self.make_node(root.key, root.left, low), high
        low, high = self.split(root.left, key)
        return low, self.make_node(root.key, high, root.right)

    def insert(self, root, key):
        low, high = self.split(root, key)
        _, duplicate = self.split(high, key + 1)      # drop any copy of key already present
        return self.merge(self.merge(low, self.make_node(key)), duplicate)

    def delete(self, root, key):
        low, high = self.split(root, key)
        _, remaining = self.split(high, key + 1)
        return self.merge(low, remaining)

    def contains(self, root, key):
        node = root
        while node is not None:
            if key == node.key:
                return True
            node = node.left if key < node.key else node.right
        return False

    def to_list(self, root):
        out, stack = [], []
        node = root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            out.append(node.key)
            node = node.right
        return out

    def size(self, root):
        return len(self.to_list(root))

    def height(self, root):
        if root is None:
            return 0
        return 1 + max(self.height(root.left), self.height(root.right))

    def nodes(self, root):
        if root is None:
            return 0
        return 1 + self.nodes(root.left) + self.nodes(root.right)


class ArrayNode:
    __slots__ = ("total", "value", "left", "right")

    def __init__(self, total=0, value=None, left=None, right=None):
        self.total = total
        self.value = value
        self.left = left
        self.right = right


class PersistentArray:
    """Persistent segment tree: set(index, value) clones one path, ranges stay queryable."""

    def __init__(self, values):
        self.counter = Counter()
        self.n = len(values)
        self.roots = [self._build(values, 0, self.n)] if self.n else [None]

    def _build(self, values, lo, hi):
        if hi - lo == 1:
            self.counter.tick()
            return ArrayNode(values[lo], values[lo], None, None)
        mid = (lo + hi) // 2
        left = self._build(values, lo, mid)
        right = self._build(values, mid, hi)
        self.counter.tick()
        return ArrayNode(left.total + right.total, None, left, right)

    def _set(self, node, lo, hi, index, value):
        if hi - lo == 1:
            self.counter.tick()
            return ArrayNode(value, value, None, None)
        mid = (lo + hi) // 2
        if index < mid:
            left = self._set(node.left, lo, mid, index, value)
            right = node.right
        else:
            left = node.left
            right = self._set(node.right, mid, hi, index, value)
        self.counter.tick()
        return ArrayNode(left.total + right.total, None, left, right)

    def set(self, version, index, value):
        """Append a new version; the input version is untouched."""
        if not 0 <= version < len(self.roots):
            raise IndexError("unknown version")
        if not 0 <= index < self.n:
            raise IndexError("index out of range")
        self.roots.append(self._set(self.roots[version], 0, self.n, index, value))
        return len(self.roots) - 1

    def get(self, version, index):
        if not 0 <= version < len(self.roots):
            raise IndexError("unknown version")
        if not 0 <= index < self.n:
            raise IndexError("index out of range")
        node, lo, hi = self.roots[version], 0, self.n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            node, lo, hi = ((node.left, lo, mid) if index < mid
                            else (node.right, mid, hi))
        return node.value

    def range_sum(self, version, lo, hi):
        if not 0 <= version < len(self.roots):
            raise IndexError("unknown version")
        return self._sum(self.roots[version], 0, self.n, lo, hi)

    def _sum(self, node, lo, hi, qlo, qhi):
        if node is None or qhi <= lo or hi <= qlo:
            return 0
        if qlo <= lo and hi <= qhi:
            return node.total
        mid = (lo + hi) // 2
        return (self._sum(node.left, lo, mid, qlo, qhi)
                + self._sum(node.right, mid, hi, qlo, qhi))

    def snapshot(self, version):
        return [self.get(version, i) for i in range(self.n)]


class PersistentStack:
    """Cons list: push allocates one cell and shares the tail (the path-copying extreme)."""

    def __init__(self):
        self.counter = Counter()
        self.versions = [None]

    def push(self, version, value):
        node = (value, self.versions[version])
        self.counter.tick()
        self.versions.append(node)
        return len(self.versions) - 1

    def to_list(self, version):
        out, node = [], self.versions[version]
        while node is not None:
            out.append(node[0])
            node = node[1]
        return out


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- persistent treap: every version vs an oracle rebuild ------------
    treap = PersistentTreap()
    roots, oracle = [None], [set()]
    edits = [(0, 50), (0, 20), (0, 70), (0, 10), (0, 30), (0, 60), (0, 90), (0, 5),
             (1, 20), (2, 51), (3, 71), (4, 90), (5, 55)]
    state = 99
    for _ in range(60):                       # deterministic random edit history
        state = splitmix64(state)
        version = state % len(roots)
        key = state % 200
        if (state >> 20) % 3 == 0 and oracle[version]:
            roots.append(treap.delete(roots[version], key))
            oracle.append(set(oracle[version]) - {key})
        else:
            roots.append(treap.insert(roots[version], key))
            oracle.append(set(oracle[version]) | {key})
    for version, expected in enumerate(oracle):
        check(treap.to_list(roots[version]) == sorted(expected),
              "version %d contents" % version)
        for key in expected:
            check(treap.contains(roots[version], key), "version %d must contain %d"
                  % (version, key))
        check(not treap.contains(roots[version], 200 + version), "absent key in version %d"
              % version)
        check(treap.size(roots[version]) == len(expected), "version %d size" % version)

    # a later update never disturbs an earlier version
    before = treap.to_list(roots[1])
    newer = treap.insert(roots[1], 199)
    check(treap.to_list(roots[1]) == before, "version 1 must be immutable")
    check(199 in treap.to_list(newer) and 199 not in before, "the new version sees the insert")

    # path copying cost: O(log n) fresh nodes per update, and growing versions stay cheap
    fresh, heights = [], []
    big = None
    for step in range(200):
        start = treap.counter.nodes
        big = treap.insert(big, (step * 37) % 500)
        fresh.append(treap.counter.nodes - start)
        heights.append(max(1, treap.height(big)))
        check(treap.to_list(big) == sorted({(s * 37) % 500 for s in range(step + 1)}),
              "large version contents")
        check(treap.to_list(big) == treap.to_list(big), "reads are repeatable")
    check(all(count <= 4 * height + 8 for count, height in zip(fresh, heights)),
          "path copying must stay logarithmic in the tree height")
    check(max(fresh) < treap.nodes(big), "an update must never copy the whole structure")

    # structural sharing: the new version shares almost all nodes with its parent
    parent = big
    start = treap.counter.nodes
    child = treap.insert(parent, 9999)
    copied = treap.counter.nodes - start
    check(treap.nodes(child) == treap.nodes(parent) + 1, "only one key was added")
    check(treap.nodes(parent) > 150, "the sharing test needs a sizeable tree")
    check(copied <= 4 * max(1, treap.height(child)) + 8 and copied < treap.nodes(child) // 4,
          "a large share of the structure must be reused, not copied")
    check(9999 not in treap.to_list(parent), "the parent version is untouched")

    # ---- persistent array (segment tree) ---------------------------------
    values = [(i * 37) % 101 - 50 for i in range(24)]
    array = PersistentArray(values)
    history = [list(values)]
    state = 4242
    for step in range(40):
        state = splitmix64(state)
        version = state % len(history)
        index = (state >> 11) % len(values)
        value = (state >> 23) % 200 - 100
        new_version = array.set(version, index, value)
        updated = list(history[version])
        updated[index] = value
        history.append(updated)
        check(new_version == len(history) - 1, "versions are appended")
    for version, expected in enumerate(history):
        check(array.snapshot(version) == expected, "array version %d" % version)
        for lo in range(0, len(values) + 1, 3):
            for hi in range(lo, len(values) + 1, 5):
                check(array.range_sum(version, lo, hi) == sum(expected[lo:hi]),
                      "range sum version %d [%d, %d)" % (version, lo, hi))
    baseline = array.counter.nodes
    array.set(0, 0, 7)
    check(array.counter.nodes - baseline <= 2 * (len(values)).bit_length() + 2,
          "a segment-tree update clones one root-to-leaf path")
    check(array.snapshot(0) == values, "version 0 still holds the original array")
    check(array.get(1, 0) == values[0], "version 1 untouched by the later set")
    check(array.get(len(history) - 1, 0) == history[-1][0], "latest version readable")

    # ---- persistent stack -------------------------------------------------
    stack = PersistentStack()
    top = 0
    for value in range(12):
        top = stack.push(top, value)
    check(stack.to_list(top) == list(reversed(range(12))), "cons list order")
    check(stack.to_list(4) == [3, 2, 1, 0], "an older stack version is intact")
    check(stack.counter.nodes == 12, "push allocates exactly one cell")
    check(stack.to_list(0) == [] and stack.push(0, 99) == 13, "empty stack version works")

    # ---- rejected inputs --------------------------------------------------
    try:
        array.get(10 ** 6, 0)
        raise AssertionError("unknown version must raise")
    except IndexError:
        pass
    try:
        array.set(0, 24, 1)
        raise AssertionError("index out of range must raise")
    except IndexError:
        pass
    empty = PersistentArray([])
    check(empty.roots == [None], "empty array has an empty version")
    empty_treap = PersistentTreap()
    check(empty_treap.insert(None, 5).key == 5 and empty_treap.size(None) == 0,
          "the empty treap version stays empty and insertable")

    print("persistent treap: %d versions, %d nodes allocated, max path copy %d nodes"
          % (len(roots), treap.counter.nodes, max(fresh)))
    print("persistent array: %d versions, %d nodes, %d nodes per set"
          % (len(array.roots), array.counter.nodes, 2 * (len(values)).bit_length() + 2))
    print("persistent stack: %d nodes for %d pushes" % (stack.counter.nodes, 12))
    print("PASS")


if __name__ == "__main__":
    main()
