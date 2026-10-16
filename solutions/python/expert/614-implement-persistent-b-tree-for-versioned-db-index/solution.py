"""Task 614 - Implement persistent B-tree for versioned DB index.  (Tier: Expert, CTCI Ch 14 — Databases)

A copy-on-write B-tree index: every node is immutable and insert/delete rebuild only the nodes
on the touched root-to-leaf path, sharing every untouched sibling with the previous version.  A
version is therefore just a root pointer, which is exactly what a versioned database index
(MVCC, time-travel tables, snapshot reads) hands to its readers.

Convention: for an internal node, separator[i] is the *maximum key of children[i]*, and a search
descends into children[i] as long as key <= separator[i].  With that convention the invariants
are simple to state and to check recursively:

  * a node has at most ORDER children (a leaf at most ORDER - 1 entries); separators are in
    increasing order and each one equals the maximum of the child to its left;
  * every non-root internal node keeps at least MIN_CHILDREN children and every non-root leaf at
    least one entry; all leaves sit at the same depth;
  * insert splits an overflowing node and pushes its median separator into the parent, which is
    rebuilt rather than mutated - the tree grows only by splitting the root;
  * delete merges an underflowing child with a sibling (re-splitting if the merge overflows) and
    the root collapses when it is left with a single child.

Documented simplification: a production B-tree first *borrows* an entry from a richer sibling and
merges only as a last resort; here an underflowing node is merged immediately and the merge is
undone by a split when needed.  Both keep the invariants that matter (order, occupancy, uniform
depth, ordered separators); the merge variant simply performs more node rewrites, so the checks
bound node allocation instead of pinning exact counts.  The headline measurement is write
amplification: an n-entry index needs n nodes, yet one update allocates only O(log_b n) of them.
Deterministic: keys come from a hand-written splitmix64 PRNG; no threads, no clock, no I/O.
"""

ORDER = 4                      # at most ORDER children per internal node (so 3 keys)
MIN_CHILDREN = (ORDER + 1) // 2


class Leaf:
    __slots__ = ("entries",)

    def __init__(self, entries):
        self.entries = tuple(entries)                    # sorted (key, value) pairs


class Internal:
    __slots__ = ("keys", "children")

    def __init__(self, keys, children):
        self.keys, self.children = tuple(keys), tuple(children)


def splitmix64(value):
    mask = 0xFFFFFFFFFFFFFFFF
    z = (value + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return z ^ (z >> 31)


class PersistentBTree:
    def __init__(self):
        self.allocated = 0
        self.root = self._leaf([])

    def _leaf(self, entries):
        self.allocated += 1
        return Leaf(sorted(entries))

    def _internal(self, keys, children):
        self.allocated += 1
        return Internal(keys, children)

    # -- queries ----------------------------------------------------------
    def descend(self, node, key):
        while isinstance(node, Internal):
            index = 0
            while index < len(node.keys) and key > node.keys[index]:
                index += 1
            node = node.children[index]
        return node

    def get(self, root, key):
        for candidate, value in self.descend(root, key).entries:
            if candidate == key:
                return value
        return None

    def items(self, root):
        """Ordered iteration; leaves are not chained (a shared chain needs mutable links)."""
        out, stack = [], [root]
        while stack:
            node = stack.pop()
            if isinstance(node, Leaf):
                out.extend(node.entries)
            else:
                stack.extend(reversed(node.children))
        return out

    def keys(self, root):
        return [key for key, _ in self.items(root)]

    def range_scan(self, root, low, high):
        return [(key, value) for key, value in self.items(root) if low <= key < high]

    def height(self, root):
        height, node = 1, root
        while isinstance(node, Internal):
            node, height = node.children[0], height + 1
        return height

    def max_key(self, node):
        while isinstance(node, Internal):
            node = node.children[-1]
        return node.entries[-1][0]

    def nodes(self, root):
        if isinstance(root, Leaf):
            return 1
        return 1 + sum(self.nodes(child) for child in root.children)

    # -- updates ----------------------------------------------------------
    def insert(self, root, key, value):
        new_root, split = self._insert(root, key, value)
        if split is None:
            return new_root
        return self._internal([split[0]], [new_root, split[1]])

    def _insert(self, node, key, value):
        """(new node, split) where split is None or (median separator, right sibling)."""
        if isinstance(node, Leaf):
            entries = [entry for entry in node.entries if entry[0] != key]
            entries.append((key, value))
            entries.sort()
            if entries == list(node.entries):
                return node, None                        # identical content: share the leaf
            if len(entries) <= ORDER - 1:
                return self._leaf(entries), None
            cut = len(entries) // 2
            return (self._leaf(entries[:cut]),
                    (entries[cut - 1][0], self._leaf(entries[cut:])))
        index = 0
        while index < len(node.keys) and key > node.keys[index]:
            index += 1
        child, split = self._insert(node.children[index], key, value)
        if split is None and child is node.children[index]:
            return node, None                            # nothing changed: share this subtree
        children = list(node.children)
        children[index] = child
        if split is not None:
            children.insert(index + 1, split[1])
        keys = [self.max_key(c) for c in children[:-1]]
        if len(children) <= ORDER:
            return self._internal(keys, children), None
        middle = len(keys) // 2
        return (self._internal(keys[:middle], children[:middle + 1]),
                (keys[middle], self._internal(keys[middle + 1:], children[middle + 1:])))

    def delete(self, root, key):
        new_root, removed = self._delete(root, key)
        if not removed:
            return root                                  # missing key: same version back
        while isinstance(new_root, Internal) and len(new_root.children) == 1:
            new_root = new_root.children[0]              # root collapse
        return new_root

    def _delete(self, node, key):
        if isinstance(node, Leaf):
            entries = [entry for entry in node.entries if entry[0] != key]
            if len(entries) == len(node.entries):
                return node, False
            return self._leaf(entries), True
        index = 0
        while index < len(node.keys) and key > node.keys[index]:
            index += 1
        child, removed = self._delete(node.children[index], key)
        if not removed:
            return node, False
        children = list(node.children)
        children[index] = child
        minimum = 1 if isinstance(child, Leaf) else MIN_CHILDREN
        size = len(child.entries) if isinstance(child, Leaf) else len(child.children)
        if size < minimum:
            children = self._rebalance(children, index)
        return self._internal([self.max_key(c) for c in children[:-1]], children), True

    def _rebalance(self, children, index):
        """Merge an underflowing child with a sibling; re-split the merge if it overflows."""
        if len(children) == 1:
            return children                              # the root collapse handles this
        other = index + 1 if index + 1 < len(children) else index - 1
        first, second = min(index, other), max(index, other)
        merged = self._merge(children[first], children[second])
        children[first:second + 1] = [merged]
        if isinstance(merged, Internal) and len(merged.children) > ORDER:
            middle = len(merged.keys) // 2
            children[first:first + 1] = [
                self._internal(merged.keys[:middle], merged.children[:middle + 1]),
                self._internal(merged.keys[middle + 1:], merged.children[middle + 1:])]
        return children

    def _merge(self, left, right):
        if isinstance(left, Leaf):
            return self._leaf(list(left.entries) + list(right.entries))
        keys = list(left.keys) + [self.max_key(left)] + list(right.keys)
        return self._internal(keys, list(left.children) + list(right.children))

    # -- integrity --------------------------------------------------------
    def validate(self, root, is_root=True, lower=None, upper=None):
        """True when every invariant holds, otherwise a message naming the first break."""
        if isinstance(root, Leaf):
            keys = [key for key, _ in root.entries]
            if keys != sorted(keys):
                return "leaf keys out of order"
            if not is_root and not keys:
                return "empty non-root leaf"
            if any(lower is not None and key <= lower for key in keys):
                return "leaf key <= its lower bound"
            if any(upper is not None and key > upper for key in keys):
                return "leaf key > its upper bound"
            return True
        if len(root.children) > ORDER:
            return "internal node with %d children" % len(root.children)
        if not is_root and len(root.children) < MIN_CHILDREN:
            return "internal node with only %d children" % len(root.children)
        if len(root.keys) != len(root.children) - 1:
            return "separator count does not match the child count"
        if list(root.keys) != sorted(root.keys) or len(set(root.keys)) != len(root.keys):
            return "separators out of order"
        for index, child in enumerate(root.children[:-1]):
            if root.keys[index] != self.max_key(child):
                return "separator %d is not its left child's maximum" % index
        for index, child in enumerate(root.children):
            status = self.validate(child, False,
                                   lower if index == 0 else root.keys[index - 1],
                                   upper if index == len(root.children) - 1
                                   else root.keys[index])
            if status is not True:
                return "child %d: %s" % (index, status)
            if self.height(child) != self.height(root.children[0]):
                return "leaves sit at different depths"
        return True


class PersistentIndex:
    """Version manager over the copy-on-write tree, as a DB index would expose it."""

    def __init__(self):
        self.tree = PersistentBTree()
        self.versions = [self.tree.root]

    def check_version(self, version):
        if not 0 <= version < len(self.versions):
            raise IndexError("unknown index version")

    def put(self, version, key, value):
        self.check_version(version)
        before = self.tree.allocated
        self.versions.append(self.tree.insert(self.versions[version], key, value))
        self.last_put_nodes = self.tree.allocated - before      # copy-on-write cost of that put
        return len(self.versions) - 1

    def remove(self, version, key):
        self.check_version(version)
        self.versions.append(self.tree.delete(self.versions[version], key))
        return len(self.versions) - 1

    def get(self, version, key):
        self.check_version(version)
        return self.tree.get(self.versions[version], key)

    def keys(self, version):
        self.check_version(version)
        return self.tree.keys(self.versions[version])

    def scan(self, version, low, high):
        self.check_version(version)
        return self.tree.range_scan(self.versions[version], low, high)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    index = PersistentIndex()
    expected = [{}]
    state = 777

    # ---- insert history with an oracle per version ------------------------
    for step in range(120):
        state = splitmix64(state)
        version = state % len(expected)
        key, value = (state >> 13) % 300, "v%d" % (state >> 25)
        new_version = index.put(version, key, value)
        expected.append({**expected[version], key: value})
        check(new_version == len(expected) - 1, "versions are appended")
        check(index.get(new_version, key) == value, "a fresh key is immediately readable")
        if step % 10 == 0:
            root = index.versions[new_version]
            check(index.tree.validate(root) is True, "invariants after insert")
            check(index.keys(new_version) == sorted(expected[new_version]),
                  "ordered iteration after insert")
            check(index.last_put_nodes <= 3 * index.tree.height(root) + 1,
                  "an insert allocates O(log n) nodes")

    check(len(expected) > 100, "the edit history must be sizeable")
    for version in range(0, len(expected), 5):
        check(index.keys(version) == sorted(expected[version]),
              "version %d iteration vs the oracle" % version)
        check(index.tree.validate(index.versions[version]) is True,
              "version %d invariants" % version)
        for probe in range(0, 300, 61):
            check(index.get(version, probe) == expected[version].get(probe),
                  "version %d point lookup" % version)

    for version in (0, 40, len(expected) - 1):
        fresh = PersistentBTree()
        root = fresh.root
        for key, value in sorted(expected[version].items()):
            root = fresh.insert(root, key, value)
        check(fresh.keys(root) == index.keys(version),
              "from-scratch rebuild of version %d" % version)
        check(fresh.validate(root) is True, "the rebuilt tree is valid")
        check(fresh.height(root) == index.tree.height(index.versions[version]),
              "the rebuild has the same height")

    # ---- ranges and immutability -----------------------------------------
    latest = len(expected) - 1
    check(index.scan(latest, 100, 200) ==
          [(k, expected[latest][k]) for k in sorted(expected[latest]) if 100 <= k < 200],
          "range scan")
    check(index.scan(latest, 5, 5) == [] and index.scan(latest, 900, 999) == [],
          "degenerate ranges are empty")
    frozen = index.keys(11)
    index.put(11, 12345, "late")
    expected.append({**expected[11], 12345: "late"})
    check(index.keys(11) == frozen, "an old version must stay immutable")
    check(index.get(11, 12345) is None, "the old version does not see the late write")
    check(index.keys(len(index.versions) - 1) == sorted(expected[-1]),
          "the branch carrying the late write sees it")

    # ---- write amplification ---------------------------------------------
    amplification = []
    for step in range(40):
        before = index.tree.allocated
        index.versions.append(index.tree.insert(index.versions[-1], 1000 + step, "x"))
        expected.append({**expected[-1], 1000 + step: "x"})
        amplification.append(index.tree.allocated - before)
    total = index.tree.nodes(index.versions[-1])
    height = index.tree.height(index.versions[-1])
    check(max(amplification) <= 3 * height, "an update rewrites only a few nodes per level")
    check(max(amplification) < total // 4, "copy-on-write must not copy the whole tree")
    check(index.keys(len(expected) - 1) == sorted(expected[-1]), "post-update iteration")
    check(index.tree.validate(index.versions[-1]) is True, "invariants after bulk inserts")

    # ---- deletions --------------------------------------------------------
    live = dict(expected[-1])
    victims = sorted(live)[:70]
    for key in victims:
        index.versions.append(index.tree.delete(index.versions[-1], key))
        del live[key]
        expected.append(dict(live))
    check(index.keys(len(expected) - 1) == sorted(live), "iteration after deletions")
    check(index.tree.validate(index.versions[-1]) is True, "invariants after deletions")
    check(all(index.get(len(expected) - 1, key) is None for key in victims[:10]),
          "deleted keys are gone")
    unchanged = index.versions[-1]
    check(index.tree.delete(unchanged, 10 ** 6) is unchanged,
          "deleting a missing key returns the very same version")
    before_delete = index.keys(7)
    index.remove(7, sorted(expected[7])[0])
    check(index.keys(7) == before_delete, "a deletion leaves older versions untouched")

    # a from-scratch tree that re-inserts and re-deletes the same keys must match
    fresh = PersistentBTree()
    root = fresh.root
    replay = {**expected[-1], **{key: "back" for key in victims[:15]}}
    for key, value in sorted(replay.items()):
        root = fresh.insert(root, key, value)
    for key in victims[:15]:
        root = fresh.delete(root, key)
    check(fresh.validate(root) is True, "the replayed deletion tree is valid")
    check(fresh.keys(root) == sorted(expected[-1]),
          "delete replay reproduces the live key set")

    # ---- rejected inputs and degenerate trees ----------------------------
    for thunk in (lambda: index.get(len(index.versions) + 5, 1),
                  lambda: index.put(-1, 1, 2),
                  lambda: index.remove(len(index.versions), 1)):
        try:
            thunk()
            raise AssertionError("unknown version must raise")
        except IndexError:
            pass
    empty = PersistentBTree()
    check(empty.get(empty.root, 1) is None and empty.items(empty.root) == [],
          "the empty index answers lookups")
    check(empty.validate(empty.root) is True, "the empty tree is valid")
    single = empty.insert(empty.root, 5, "only")
    check(empty.get(single, 5) == "only" and empty.keys(single) == [5], "single-entry tree")
    check(empty.keys(empty.root) == [], "the empty version stayed empty")
    check(empty.delete(empty.root, 5) is empty.root, "deleting from an empty tree is a no-op")

    print("persistent B-tree: %d versions, %d nodes allocated, height %d, %d live entries"
          % (len(index.versions), index.tree.allocated, height, total))
    print("write amplification: max %d nodes per update on an index of %d nodes"
          % (max(amplification), total))
    print("occupancy: order %d, >= %d children per internal node" % (ORDER, MIN_CHILDREN))
    print("PASS")


if __name__ == "__main__":
    main()
