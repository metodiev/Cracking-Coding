"""Task 613 - Implement fully persistent union-find (complex).  (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

A *fully* persistent union-find answers find/connected for any historical version, and new
versions may branch off old ones - both children of one version stay independent.  That rules
out path compression: compression rewrites the parent pointers of nodes that older versions
also point at, so it would silently change history.  The classic fully persistent answer is
therefore union by rank without compression, with the parent/rank arrays living in a
path-copying persistent array:

* PersistentArray - a segment tree where set(root, i, v) allocates the O(log n) nodes on the
  root-to-leaf path and shares the rest, so the two array writes of a union cost O(log n)
  nodes, never a copy of the array.
* PersistentUnionFind - versions are (parentRoot, rankRoot) handles in an append-only list;
  a version can be the parent of many later versions (a version *tree*, not just a list).
  find(version, x) walks parent pointers inside that version, union(version, a, b) links the
  smaller rank under the larger and appends a new version; union inside one set is a no-op
  and returns the very same version, so version handles stay canonical.
  Cost: find = O(log n) pointer steps x O(log n) array access = O(log^2 n), height O(log n).

Verified: for every version, the equivalence classes exactly match an oracle that rebuilds a
fresh union-find along that version's ancestry; branching versions are independent (a union in
one branch is invisible in its sibling and in the base); the rank bound really bounds the
pointer-chain length; old versions are byte-identical before and after later operations; and a
deliberate counter-example shows that an in-place path-compressing structure *does* mutate
state shared with a previous version - which is exactly why the persistent one does not
compress.
"""

MASK = 0xFFFFFFFFFFFFFFFF


class SegmentNode:
    __slots__ = ("total", "value", "left", "right")

    def __init__(self, total, value, left, right):
        self.total, self.value, self.left, self.right = total, value, left, right


class PersistentArray:
    """Persistent segment tree over an index range; every root is an immutable array."""

    def __init__(self, values):
        self.values = list(values)
        self.n = len(self.values)
        self.allocated = 0
        self.root = self._build(0, self.n)

    def _new(self, total, value, left, right):
        self.allocated += 1
        return SegmentNode(total, value, left, right)

    def _build(self, lo, hi):
        if hi - lo == 1:
            return self._new(self.values[lo], self.values[lo], None, None)
        mid = (lo + hi) // 2
        left, right = self._build(lo, mid), self._build(mid, hi)
        return self._new(left.total + right.total, None, left, right)

    def get(self, root, index):
        if not 0 <= index < self.n:
            raise IndexError("array index out of range")
        lo, hi = 0, self.n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if index < mid:
                root, hi = root.left, mid
            else:
                root, lo = root.right, mid
        return root.value

    def set(self, root, index, value):
        """Return the root of a new array; `root` itself is never written to."""
        if not 0 <= index < self.n:
            raise IndexError("array index out of range")
        return self._set(root, 0, self.n, index, value)

    def _set(self, node, lo, hi, index, value):
        if hi - lo == 1:
            return self._new(node.total, value, None, None)
        mid = (lo + hi) // 2
        if index < mid:
            left = self._set(node.left, lo, mid, index, value)
            right = node.right
        else:
            left = node.left
            right = self._set(node.right, mid, hi, index, value)
        return self._new(left.total + right.total, None, left, right)

    def snapshot(self, root):
        return [self.get(root, i) for i in range(self.n)]

    def nodes(self, root):
        if root is None:
            return 0
        return 1 + self.nodes(root.left) + self.nodes(root.right)


class PersistentUnionFind:
    """Union by rank over a persistent array; versions form a tree, not a list."""

    def __init__(self, size):
        self.size = size
        self.array = PersistentArray(list(range(size)))
        self.ranks = PersistentArray([0] * size)
        self.versions = [(self.array.root, self.ranks.root, -1)]   # (parentRoot, rankRoot, parentVersion)
        self.find_steps = 0

    def branch(self, version):
        """An explicit copy of a version, so the two can diverge independently."""
        if not 0 <= version < len(self.versions):
            raise IndexError("unknown version")
        parent_root, rank_root, _ = self.versions[version]
        self.versions.append((parent_root, rank_root, version))
        return len(self.versions) - 1

    def find(self, version, x):
        """Root of x in that version, plus the number of pointer steps taken."""
        if not 0 <= version < len(self.versions):
            raise IndexError("unknown version")
        if not 0 <= x < self.size:
            raise IndexError("element out of range")
        parent_root = self.versions[version][0]
        steps = 0
        node = self.array.get(parent_root, x)
        while node != x:
            x = node
            node = self.array.get(parent_root, x)
            steps += 1
        self.find_steps += steps
        return x, steps

    def connected(self, version, a, b):
        return self.find(version, a)[0] == self.find(version, b)[0]

    def union(self, version, a, b):
        """Append a new version joining a and b; returns its handle (old handle if already
        joined - a no-op must stay canonical so callers can compare versions for equality)."""
        root_a, _ = self.find(version, a)
        root_b, _ = self.find(version, b)
        if root_a == root_b:
            return version
        parent_root, rank_root, _ = self.versions[version]
        rank_a = self.ranks.get(rank_root, root_a)
        rank_b = self.ranks.get(rank_root, root_b)
        if rank_a < rank_b:
            root_a, root_b = root_b, root_a
            rank_a, rank_b = rank_b, rank_a
        parent_root = self.array.set(parent_root, root_b, root_a)
        if rank_a == rank_b:
            rank_root = self.ranks.set(rank_root, root_a, rank_a + 1)
        self.versions.append((parent_root, rank_root, version))
        return len(self.versions) - 1

    # -- diagnostics ------------------------------------------------------
    def classes(self, version):
        """Canonical partition of the elements at that version, as sorted class lists."""
        groups = {}
        for x in range(self.size):
            groups.setdefault(self.find(version, x)[0], []).append(x)
        return sorted(sorted(group) for group in groups.values())

    def parent_fingerprint(self, version):
        return tuple(self.array.snapshot(self.versions[version][0]))

    def deep_copy_size(self, version):
        """If nothing were shared, a version would need this many array cells."""
        return len(self.versions) * self.size * 2


class MutableUnionFind:
    """Deliberate counter-example: path compressing in place, as a *shared* mutable array."""

    def __init__(self, parent):
        self.parent = parent          # not a copy: this is the mistake being demonstrated

    def find(self, x):
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        for node in path:             # compression rewrites cells older versions still see
            self.parent[node] = x
        return x

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def oracle_classes(history, version, size):
    """Independent oracle: rebuild a plain union-find along the version's ancestry chain."""
    chain, v = [], version
    while v != 0:
        chain.append(v)
        v = history[v][3]
    parent, rank = list(range(size)), [0] * size

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for v in reversed(chain):
        kind = history[v][0]
        if kind == "union":
            a, b = history[v][1], history[v][2]
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                if rank[root_a] < rank[root_b]:
                    root_a, root_b = root_b, root_a
                parent[root_b] = root_a
                if rank[root_a] == rank[root_b]:
                    rank[root_a] += 1
    groups = {}
    for x in range(size):
        groups.setdefault(find(x), []).append(x)
    return sorted(sorted(group) for group in groups.values())


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    SIZE = 12
    uf = PersistentUnionFind(SIZE)
    history = [None]                     # history[v] = (kind, a, b) that *created* version v
    versions = [0]

    check(uf.classes(0) == [[x] for x in range(SIZE)], "version 0 is all singletons")
    check(not uf.connected(0, 0, 1), "distinct singletons are not connected")
    check(uf.union(0, 3, 3) == 0, "union of an element with itself is a no-op version")

    # deterministic edit history over a version *tree* (branching, not a straight line)
    state = 20250
    union_noops = 0
    for step in range(60):
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK
        base = versions[state % len(versions)]
        a, b = (state >> 8) % SIZE, (state >> 24) % SIZE
        if (state >> 40) % 5 == 0:
            versions.append(uf.branch(base))
            history.append(("branch", base, None, base))
        else:
            new_version = uf.union(base, a, b)
            if new_version == base:                     # already in one class: canonical no-op
                union_noops += 1
            else:
                versions.append(new_version)
                history.append(("union", a, b, base))
        check(len(uf.versions) == len(history), "version handles and history stay aligned")
        for version in sorted(set(versions)):
            check(uf.classes(version) == oracle_classes(history, version, SIZE),
                  "version %d disagrees with the oracle" % version)

    # branching independence: two children of one version must not see each other's unions
    base = versions[0]
    left = uf.union(base, 0, 1)
    right = uf.union(base, 2, 3)
    check(uf.connected(left, 0, 1) and not uf.connected(left, 2, 3), "left branch")
    check(uf.connected(right, 2, 3) and not uf.connected(right, 0, 1), "right branch")
    check(not uf.connected(base, 0, 1) and not uf.connected(base, 2, 3), "base untouched")

    # immutability: an old version's arrays are identical before and after later edits
    old_version = versions[len(versions) // 2]
    before = uf.parent_fingerprint(old_version)
    for step in range(20):
        versions.append(uf.union(versions[-1], step % SIZE, (step * 5 + 1) % SIZE))
    check(uf.parent_fingerprint(old_version) == before,
          "a historical version must not change when new versions are built")

    # union by rank: the pointer chain stays within the rank bound and memory is shared
    tall = 0
    joined = 0
    for x in range(1, SIZE):
        tall = uf.union(tall, 0, x)
        joined += 1
    _, steps = uf.find(tall, SIZE - 1)
    bound = 1 if SIZE <= 1 else SIZE.bit_length() - 1
    check(steps <= bound, "rank bound: chain length %d within %d" % (steps, bound))
    check(uf.classes(tall) == [list(range(SIZE))], "everything is one class now")
    check(uf.connected(tall, 0, SIZE - 1) and uf.find(tall, 5)[0] == uf.find(tall, 9)[0],
          "connectivity after joining everything")
    shared = uf.array.allocated
    check(shared < uf.deep_copy_size(len(versions)) // 4,
          "persistent arrays must share far more than they copy")

    # older versions still answer with the pre-join partition
    check(uf.classes(base) == [[x] for x in range(SIZE)], "the base version is still disjoint")

    # ---- the counter-example: in-place compression mutates a shared old version --
    parent = list(range(6))
    parent[3], parent[4] = 2, 3                  # chain 4 -> 3 -> 2, as earlier unions left it
    naive = MutableUnionFind(parent)             # both "versions" point at the same array
    snapshot = list(parent)
    check(naive.find(4) == 2, "compression still answers with the true root")
    check(parent != snapshot, "in-place compression does mutate the shared array")
    check(parent[4] == 2 and parent[3] == 2, "the find path is rewritten onto the root")
    check(snapshot[4] == 3 and snapshot[3] == 2, "the snapshot keeps the pre-compression shape")
    check(uf.parent_fingerprint(old_version) == before,
          "the fully persistent union-find performs no such in-place write")

    # rejected inputs
    for thunk in (lambda: uf.find(len(versions), 0), lambda: uf.find(0, SIZE),
                  lambda: uf.branch(len(versions)), lambda: uf.array.get(uf.array.root, SIZE)):
        try:
            thunk()
            raise AssertionError("invalid handle must raise")
        except IndexError:
            pass

    print("persistent union-find: %d versions over %d elements, %d array nodes allocated"
          % (len(versions), SIZE, uf.array.allocated + uf.ranks.allocated))
    print("no-op unions folded into an existing version: %d" % union_noops)
    print("find cost: %d pointer steps measured, rank bound %d, chain of depth %d"
          % (uf.find_steps, bound, steps))
    print("sharing: %d nodes vs %d cells for full copies"
          % (shared, uf.deep_copy_size(len(versions))))
    print("PASS")


if __name__ == "__main__":
    main()
