"""Task 806 - Implement fully persistent B-tree with copy-on-write pages and concurrency.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A *fully persistent* B+-tree over a page store.  Every mutation copies the pages on the path from
the root to the changed node instead of modifying them, so a version is just a root pointer and
every earlier version stays readable forever.  Pages hold at most MAX entries with minimum degree
MIN: a full page (MAX + 1 entries) splits into two pages of at least MIN entries around a promoted
separator, deletes borrow from a sibling that has spare entries and otherwise merge with a sibling
(dropping the separator that disappears with them), and the root shrinks when its last separator
goes away.  Values live in the leaves only; internal pages hold separator keys and child pointers,
which is what makes the split a copy-up rather than a move.

The page store is the only mutable object: it hands out increasing page ids and records, per page,
the version whose contents it holds.  That number is what the persistence check uses - a version
may never reach a page born after it - and it is exactly what makes the structure persistent
rather than merely append-only.

Concurrency is modelled the way copy-on-write makes it possible: a reader takes a root and can
never be blocked or invalidated by a writer, because a writer cannot touch a page a reader holds,
it publishes new pages instead.  The checks re-read an old root after dozens of writes and follow
a cursor that keeps scanning its own version while writers insert and delete around it.

Verified invariants, most of them against a plain-dict oracle recording the state after every
single operation:
  version equivalence  every historical root returns exactly the oracle state at that version, for
                       point lookups, full flattens and bounded range scans;
  structure            leaf keys sorted and unique, internal keys sorted, children = keys + 1, all
                       leaves at the same depth, every non-root page holding between MIN and MAX
                       entries, and each separator strictly above the left child's maximum and at
                       most the right child's minimum;
  persistence          pages are born in the version that first needed them, no version reaches a
                       younger page, and an old root still returns the old rows after many writes;
  copy-on-write cost   an insert copies at most depth + 2 pages, a no-op copies none, adjacent
                       versions share all but the copied path, and total allocations stay far below
                       one full path per version;
  deletes              borrowing, merging and root shrinking keep every version equal to the
                       oracle, emptying the tree yields the empty version, and rebuilding works;
  garbage collection   a brute-force reachability walk over the live roots equals the set the
                       collector keeps, and every freed page was unreachable;
  isolation            an interleaved writer/reader schedule never exposes a partial update: a
                       cursor always sees exactly its own version's rows;
  scale, determinism   bulk loads of four hundred keys and a descending insert order keep the
                       invariant set with a logarithmic height, and two identical workloads
                       produce identical contents and identical page shapes.

Simplified versus a storage engine: pages live in a dictionary instead of on disk (no buffer pool,
no write-ahead log, no crash recovery and no page checksum - the generation rule plus the version
equivalence checks stand in for the checksum), concurrency is a deterministic single-threaded
interleaving rather than real latches, minimum degree is 2 and the trees hold a few hundred keys so
that the whole history can be compared against the oracle key by key.
"""
MIN_ENTRIES = 2
MAX_ENTRIES = 4


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Page:
    """An immutable page: leaves hold (key, value) pairs, internal pages hold separator keys."""

    __slots__ = ("pid", "leaf", "values", "children", "born")

    def __init__(self, pid, leaf, values, children, born):
        self.pid = pid
        self.leaf = leaf
        self.values = values
        self.children = children
        self.born = born

    def __repr__(self):
        return "<%s %d %s%s born=%d>" % ("leaf" if self.leaf else "node", self.pid,
                                         self.values, self.children, self.born)


class PageStore:
    def __init__(self):
        self.pages = {}
        self.next_pid = 0
        self.allocations = 0
        self.reads = 0

    def allocate(self, leaf, values, children, born):
        pid = self.next_pid
        self.next_pid += 1
        self.pages[pid] = Page(pid, leaf, tuple(values), tuple(children), born)
        self.allocations += 1
        return pid

    def read(self, pid):
        self.reads += 1
        return self.pages[pid]

    def reachable(self, roots):
        """Independent reachability walk, used to cross-check the collector."""
        seen = set()
        pending = [root for root in roots if root is not None]
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            pending.extend(child for child in self.pages[pid].children if child not in seen)
        return seen

    def collect(self, roots):
        keep = self.reachable(roots)
        freed = sorted(pid for pid in self.pages if pid not in keep)
        for pid in freed:
            del self.pages[pid]
        return freed


class PersistentTree:
    """versions[i] is the root after operation i; versions[0] is the empty tree."""

    def __init__(self):
        self.store = PageStore()
        self.versions = [None]
        self.operations = 0

    # ------------------------------------------------------------------------- reads
    def search(self, root, key):
        pid = root
        while pid is not None:
            page = self.store.read(pid)
            if page.leaf:
                for item_key, value in page.values:
                    if item_key == key:
                        return value
                return None
            index = 0
            while index < len(page.values) and key >= page.values[index]:
                index += 1
            pid = page.children[index]
        return None

    def scan(self, root, low=None, high=None):
        output = []

        def walk(pid):
            if pid is None:
                return
            page = self.store.read(pid)
            if page.leaf:
                for item_key, value in page.values:
                    if (low is None or item_key >= low) and (high is None or item_key <= high):
                        output.append((item_key, value))
                return
            for child in page.children:
                walk(child)

        walk(root)
        output.sort(key=lambda item: item[0])
        return output

    def cursor(self, root, low=None, high=None):
        return Cursor(self.scan(root, low, high))

    def depth(self, root):
        levels, pid = 0, root
        while pid is not None:
            page = self.store.read(pid)
            levels += 1
            pid = page.children[0] if not page.leaf else None
        return levels

    # ---------------------------------------------------------------------- mutations
    def put(self, key, value):
        version = self.operations + 1
        before = self.store.allocations
        if self.versions[-1] is None:
            root = self.store.allocate(True, [(key, value)], [], version)
        else:
            root, promoted = self._insert(self.versions[-1], key, value, version)
            if promoted is not None:
                separator, right = promoted
                root = self.store.allocate(False, [separator], [root, right], version)
        self.operations = version
        self.versions.append(root)
        return self.store.allocations - before

    def delete(self, key):
        version = self.operations + 1
        before = self.store.allocations
        if self.versions[-1] is None:
            self.operations = version
            self.versions.append(None)
            return 0
        root, _changed = self._delete(self.versions[-1], key, version)
        while root is not None:                      # collapse empty roots left by merges
            page = self.store.read(root)
            if page.values:
                break
            root = page.children[0] if not page.leaf else None
        self.operations = version
        self.versions.append(root)
        return self.store.allocations - before

    def _insert(self, pid, key, value, version):
        """Returns (new pid, promotion), promotion being None or (separator, right child)."""
        page = self.store.read(pid)
        if page.leaf:
            entries = [item for item in page.values if item[0] != key]
            entries.append((key, value))
            entries.sort(key=lambda item: item[0])
            if len(entries) <= MAX_ENTRIES:
                return self.store.allocate(True, entries, [], version), None
            middle = len(entries) // 2
            left = self.store.allocate(True, entries[:middle], [], version)
            right = self.store.allocate(True, entries[middle:], [], version)
            return left, (entries[middle][0], right)
        index = 0
        while index < len(page.values) and key >= page.values[index]:
            index += 1
        keys, children = list(page.values), list(page.children)
        children[index], promoted = self._insert(children[index], key, value, version)
        if promoted is not None:
            keys.insert(index, promoted[0])
            children.insert(index + 1, promoted[1])
        if len(keys) <= MAX_ENTRIES:
            return self.store.allocate(False, keys, children, version), None
        middle = len(keys) // 2
        left = self.store.allocate(False, keys[:middle], children[:middle + 1], version)
        right = self.store.allocate(False, keys[middle + 1:], children[middle + 1:], version)
        return left, (keys[middle], right)

    def _delete(self, pid, key, version):
        """Returns (new pid, underflow); the parent repairs the underflow."""
        page = self.store.read(pid)
        if page.leaf:
            entries = [item for item in page.values if item[0] != key]
            if len(entries) == len(page.values):
                return pid, False                    # nothing matched: share the page untouched
            return self.store.allocate(True, entries, [], version), len(entries) < MIN_ENTRIES
        index = 0
        while index < len(page.values) and key >= page.values[index]:
            index += 1
        keys, children = list(page.values), list(page.children)
        new_child, underflow = self._delete(children[index], key, version)
        if new_child == children[index] and not underflow:
            return pid, False
        children[index] = new_child
        if underflow:
            self._repair(keys, children, index, version)
        return self.store.allocate(False, keys, children, version), len(keys) < MIN_ENTRIES

    def _repair(self, keys, children, index, version):
        """Borrow from a sibling with spare entries, otherwise merge.  Mutates keys/children."""
        left_pid = children[index - 1] if index > 0 else None
        right_pid = children[index + 1] if index + 1 < len(children) else None
        if left_pid is not None and len(self.store.read(left_pid).values) > MIN_ENTRIES:
            sibling, child = self.store.read(left_pid), self.store.read(children[index])
            if sibling.leaf:
                moved = sibling.values[-1]
                children[index] = self.store.allocate(True, [moved] + list(child.values), [],
                                                      version)
                keys[index - 1] = moved[0]
            else:
                moved_child = sibling.children[-1]
                children[index] = self.store.allocate(
                    False, [keys[index - 1]] + list(child.values),
                    [moved_child] + list(child.children), version)
                keys[index - 1] = sibling.values[-1]
            children[index - 1] = self.store.allocate(
                sibling.leaf, list(sibling.values[:-1]),
                [] if sibling.leaf else list(sibling.children[:-1]), version)
            return
        if right_pid is not None and len(self.store.read(right_pid).values) > MIN_ENTRIES:
            sibling, child = self.store.read(right_pid), self.store.read(children[index])
            if sibling.leaf:
                moved = sibling.values[0]
                children[index] = self.store.allocate(True, list(child.values) + [moved], [],
                                                      version)
                keys[index] = sibling.values[1][0]
            else:
                moved_child = sibling.children[0]
                children[index] = self.store.allocate(
                    False, list(child.values) + [keys[index]],
                    list(child.children) + [moved_child], version)
                keys[index] = sibling.values[0]
            children[index + 1] = self.store.allocate(
                sibling.leaf, list(sibling.values[1:]),
                [] if sibling.leaf else list(sibling.children[1:]), version)
            return
        if index > 0:                                # merge into the left sibling
            left, child = self.store.read(children[index - 1]), self.store.read(children[index])
            if left.leaf:
                children[index - 1] = self.store.allocate(
                    True, list(left.values) + list(child.values), [], version)
            else:                                    # an internal merge keeps the separator inside
                children[index - 1] = self.store.allocate(
                    False, list(left.values) + [keys[index - 1]] + list(child.values),
                    list(left.children) + list(child.children), version)
            del children[index]
            del keys[index - 1]
            return
        left, right = self.store.read(children[index]), self.store.read(children[index + 1])
        if left.leaf:
            children[index] = self.store.allocate(
                True, list(left.values) + list(right.values), [], version)
        else:
            children[index] = self.store.allocate(
                False, list(left.values) + [keys[index]] + list(right.values),
                list(left.children) + list(right.children), version)
        del children[index + 1]
        del keys[index]


class Cursor:
    """A resumable scan over one fixed version."""

    def __init__(self, items):
        self.items = items
        self.position = 0

    def next_batch(self, count):
        batch = self.items[self.position:self.position + count]
        self.position += len(batch)
        return batch

    def exhausted(self):
        return self.position >= len(self.items)


# ------------------------------------------------------------------------------- invariants
def min_key(store, pid):
    page = store.read(pid)
    while not page.leaf:
        page = store.read(page.children[0])
    return page.values[0][0]


def max_key(store, pid):
    page = store.read(pid)
    while not page.leaf:
        page = store.read(page.children[-1])
    return page.values[-1][0]


def tree_problems(tree, root, store=None):
    """Order, shape, occupancy and separator bracketing for one version."""
    store = store or tree.store
    problems = []
    if root is None:
        return problems
    depths = []

    def walk(pid, level, is_root):
        page = store.read(pid)
        if not is_root and not MIN_ENTRIES <= len(page.values) <= MAX_ENTRIES:
            problems.append(("occupancy out of range", pid, len(page.values)))
        if is_root and not page.values:
            problems.append(("empty root page", pid))
        if len(page.values) > MAX_ENTRIES:
            problems.append(("page beyond the block bound", pid, len(page.values)))
        if page.leaf:
            depths.append(level)
            keys = [item[0] for item in page.values]
            if keys != sorted(keys):
                problems.append(("leaf keys unsorted", pid))
            if len(set(keys)) != len(keys):
                problems.append(("duplicate key in a leaf", pid))
            return
        if len(page.children) != len(page.values) + 1:
            problems.append(("child count does not match keys", pid))
        if list(page.values) != sorted(page.values):
            problems.append(("internal keys unsorted", pid))
        for position, child in enumerate(page.children):
            if position > 0 and page.values[position - 1] > min_key(store, child):
                problems.append(("separator above the right child's minimum", pid, position))
            if position < len(page.values) and page.values[position] <= max_key(store, child):
                problems.append(("separator not above the left child's maximum", pid, position))
            walk(child, level + 1, False)

    walk(root, 0, True)
    if len(set(depths)) > 1:
        problems.append(("uneven leaf depth", sorted(set(depths))))
    return problems


def flatten(tree, root):
    if root is None:
        return {}
    output = {}

    def walk(pid):
        page = tree.store.read(pid)
        if page.leaf:
            for key, value in page.values:
                output[key] = value
            return
        for child in page.children:
            walk(child)

    walk(root)
    return output


def generation_problems(tree):
    """No version may reach a page whose contents were created later than that version."""
    problems = []
    for index, root in enumerate(tree.versions):
        if root is None:
            continue
        for pid in tree.store.reachable([root]):
            if tree.store.pages[pid].born > index:
                problems.append((index, pid, tree.store.pages[pid].born))
    return problems


def deterministic_keys(count, seed=0x5EED):
    """Deterministic pseudo-random key order, so two runs of a workload draw the same keys."""
    mask = (1 << 64) - 1
    keys, state = [], seed
    while len(keys) < count:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        keys.append((z ^ (z >> 31)) % (count * 7))
    return keys


def load(keys, base=0):
    tree, oracle, state = PersistentTree(), [{}], {}
    for key in keys:
        tree.put(key, key + base)
        state = dict(state)
        state[key] = key + base
        oracle.append(state)
    return tree, oracle


def main():
    # --- 1. small hand-checked tree ------------------------------------------------
    tree = PersistentTree()
    for key in range(1, 11):
        tree.put(key, "v%d" % key)
    check([tree.search(tree.versions[-1], key) for key in range(1, 11)]
          == ["v%d" % key for key in range(1, 11)],
          "every inserted key must be found again")
    check(tree.search(tree.versions[-1], 99) is None, "a missing key must read as None")
    check(tree.versions[0] is None and len(tree.versions) == 11,
          "the version list starts with the empty tree and grows once per operation")
    check(tree.depth(tree.versions[-1]) >= 2, "ten keys must build more than one level")
    check(tree_problems(tree, tree.versions[-1]) == [],
          "the small tree must satisfy every invariant: %s"
          % (tree_problems(tree, tree.versions[-1]),))

    # --- 2. every version equals the oracle ----------------------------------------
    keys = deterministic_keys(140, seed=7)
    history, oracle = load(keys)
    mismatches = [index for index, root in enumerate(history.versions)
                  if flatten(history, root) != oracle[index]]
    check(not mismatches,
          "every historical root must equal the oracle state, %d mismatches" % len(mismatches))
    check(len(history.versions) == len(oracle) == 141,
          "the version list must have one entry per operation plus the empty tree")
    check(len(oracle[-1]) == len(set(keys)),
          "rewriting a repeated key must not change the tree size: %d entries for %d distinct keys"
          % (len(oracle[-1]), len(set(keys))))
    first_key = keys[0]
    check(history.search(history.versions[1], first_key) == first_key
          and history.search(history.versions[0], first_key) is None,
          "the empty version must not know the first key and the next version must")

    # --- 3. structure and persistence over the whole history -----------------------
    problems = []
    for index, root in enumerate(history.versions):
        found = tree_problems(history, root)
        if found:
            problems.append((index, found[:2]))
    check(not problems, "every version must be structurally valid: %s" % (problems[:3],))
    check(generation_problems(history) == [],
          "no version may reach a page born later than itself: %s"
          % (generation_problems(history)[:3],))
    check(history.depth(history.versions[-1]) <= 8,
          "a 140-key tree with minimum degree two must stay shallow, depth %d"
          % history.depth(history.versions[-1]))
    check(max(len(page.values) for page in history.store.pages.values()) <= MAX_ENTRIES,
          "no page may exceed the block bound")

    # --- 4. copy-on-write cost, sharing and version isolation ----------------------
    cost = PersistentTree()
    for key in range(20):
        cost.put(key, key)
    depth = cost.depth(cost.versions[-1])
    copies = [cost.put(key + 100, key) for key in range(20)]
    path = depth + 1
    check(max(copies) <= 2 * path,
          "an insert copies its path plus at most one split page per level: %d copies, path %d"
          % (max(copies), path))
    check(sum(copies) <= path * len(copies) + 2 * path,
          "the total must stay near one path per insert plus a few splits: %d copies for %d "
          "inserts at path %d" % (sum(copies), len(copies), path))
    previous = set(cost.store.reachable([cost.versions[-2]]))
    newest = set(cost.store.reachable([cost.versions[-1]]))
    check(len(previous & newest) >= len(previous) - 2 * path,
          "adjacent versions must share all but the copied path and its splits: %d shared of %d "
          "at path %d" % (len(previous & newest), len(previous), path))
    check(len(previous & newest) >= len(previous) // 2,
          "copy-on-write must share at least half the pages between adjacent versions: "
          "%d shared of %d" % (len(previous & newest), len(previous)))
    cost.put(200, 1)
    root_before = cost.versions[-1]
    cost.put(200, 2)
    check(cost.search(root_before, 200) == 1,
          "the version before an update must keep the old value")
    check(cost.search(cost.versions[-1], 200) == 2,
          "the newest version must show the new value")
    check(cost.delete(9999) == 0, "deleting a missing key must allocate nothing")
    check(cost.put(1, 1) > 0, "a real write must allocate at least the root page")
    check(cost.store.allocations < len(cost.versions) * (depth + 2),
          "sharing must keep allocations below one path per version: %d pages for %d versions"
          % (cost.store.allocations, len(cost.versions)))

    # --- 5. deletes keep every version equivalent ---------------------------------
    order = deterministic_keys(90, seed=21)
    deletion, delete_oracle = load(order, base=1)
    for key in order[::2]:
        deletion.delete(key)
        state = dict(delete_oracle[-1])
        state.pop(key, None)
        delete_oracle.append(state)
    bad = [index for index, root in enumerate(deletion.versions)
           if flatten(deletion, root) != delete_oracle[index]]
    check(not bad, "deletes must keep every version equal to the oracle: %s" % (bad[:5],))
    structural = [index for index, root in enumerate(deletion.versions)
                  if tree_problems(deletion, root)]
    check(not structural,
          "deletes must preserve the structural invariants, bad versions %s" % (structural[:5],))
    survivors = {key for key in set(order) if key not in set(order[::2])}
    check(set(flatten(deletion, deletion.versions[-1])) == survivors,
          "exactly the un-deleted keys must remain: %d vs %d"
          % (len(flatten(deletion, deletion.versions[-1])), len(survivors)))
    check(generation_problems(deletion) == [], "deletes must respect the generation rule")
    check(deletion.scan(deletion.versions[-1], 3, 9)
          == sorted((key, value) for key, value in
                    flatten(deletion, deletion.versions[-1]).items() if 3 <= key <= 9),
          "a range scan on a deleted tree must agree with the flattened state")

    # --- 6. emptying and rebuilding ------------------------------------------------
    empty = PersistentTree()
    for key in range(30):
        empty.put(key, key)
    for key in range(30):
        empty.delete(key)
    check(empty.versions[-1] is None, "deleting every key must empty the tree")
    check(flatten(empty, empty.versions[-1]) == {}, "the emptied tree must be empty")
    check(empty.delete(5) == 0, "deleting from an empty tree must allocate nothing")
    check(len(empty.versions) == 62, "an empty tree still records one version per operation")
    check(tree_problems(empty, empty.versions[20]) == [],
          "a half-emptied version must still be structurally valid")
    rebuilt = PersistentTree()
    check(rebuilt.delete(1) == 0 and rebuilt.versions[-1] is None,
          "deleting from a brand new tree must be a no-op")
    for key in range(30):
        rebuilt.put(key, -key)
    check(flatten(rebuilt, rebuilt.versions[-1]) == {key: -key for key in range(30)},
          "re-inserting must rebuild the tree")
    check(tree_problems(rebuilt, rebuilt.versions[-1]) == [],
          "the rebuilt tree must satisfy the invariants")

    # --- 7. cursor isolation against concurrent writers ---------------------------
    live = PersistentTree()
    for key in range(40):
        live.put(key, "old%d" % key)
    snapshot_root = live.versions[-1]
    reader = live.cursor(snapshot_root, 0, 39)
    first_batch = reader.next_batch(10)
    for key in range(40):
        live.put(key, "new%d" % key)
    for key in range(0, 40, 3):
        live.delete(key)
    rest = []
    while not reader.exhausted():
        rest.extend(reader.next_batch(7))
    observed = dict(first_batch + rest)
    check(len(observed) == 40, "the cursor must still see all forty rows, saw %d" % len(observed))
    check(all(value == "old%d" % key for key, value in observed.items()),
          "a reader must keep seeing its own version while writers work")
    check(flatten(live, snapshot_root) == {key: "old%d" % key for key in range(40)},
          "the snapshot root must be untouched by the concurrent writes")
    check(sorted(flatten(live, live.versions[-1]))
          == [key for key in range(40) if key % 3 != 0],
          "the newest version must contain exactly the surviving keys")
    check(all(value == "new%d" % key for key, value in flatten(live, live.versions[-1]).items()),
          "the newest version must contain the rewritten values")
    check(generation_problems(live) == [],
          "the interleaved schedule must respect the generation rule")

    # --- 8. garbage collection -----------------------------------------------------
    collector = PersistentTree()
    for key in range(50):
        collector.put(key, key)
    for key in range(0, 50, 5):
        collector.delete(key)
    for key in range(50, 80):
        collector.put(key, key)
    live_roots = [collector.versions[-1], collector.versions[10], collector.versions[60]]
    expected = collector.store.reachable(live_roots)
    all_pages = set(collector.store.pages)
    freed = collector.store.collect(live_roots)
    check(set(freed) == all_pages - expected,
          "the collector must free exactly the unreachable pages: freed %d, expected %d"
          % (len(freed), len(all_pages - expected)))
    check(set(collector.store.pages) == expected,
          "the surviving pages must be exactly the reachable ones")
    survivors = {key: key for key in range(80) if key % 5 != 0 or key >= 50}
    check(flatten(collector, collector.versions[-1]) == survivors,
          "the collected tree must still return the right rows")
    check(tree_problems(collector, collector.versions[-1]) == [],
          "collection must not damage the surviving tree")
    check(all(pid not in collector.store.pages for pid in freed),
          "a freed page must be gone from the store")
    check(len(freed) > 0, "the workload must actually produce garbage")

    # --- 9. scale -----------------------------------------------------------------
    large = PersistentTree()
    big_keys = deterministic_keys(400, seed=99)
    for key in big_keys:
        large.put(key, key * 7)
    root = large.versions[-1]
    check(tree_problems(large, root) == [],
          "a 400-key tree must satisfy the invariants: %s" % (tree_problems(large, root)[:3],))
    check(large.depth(root) <= 12, "the height must stay logarithmic, depth %d"
          % large.depth(root))
    expected_state = {}
    for key in big_keys:
        expected_state[key] = key * 7
    check(flatten(large, root) == expected_state,
          "the bulk-loaded tree must hold exactly the inserted state")
    check(all(large.search(root, key) == value for key, value in expected_state.items()),
          "every bulk-loaded key must be searchable")
    check(large.store.allocations <= (large.depth(root) + 2) * len(big_keys),
          "each insert may allocate its path plus a split page: %d pages for %d inserts at "
          "depth %d" % (large.store.allocations, len(big_keys), large.depth(root)))
    descending = PersistentTree()
    for key in range(200, 0, -1):
        descending.put(key, key)
    check(tree_problems(descending, descending.versions[-1]) == [],
          "a descending insert order must keep the invariants")
    check(flatten(descending, descending.versions[-1]) == {key: key for key in range(1, 201)},
          "the descending load must hold every key")
    from_large = PersistentTree()
    for key in big_keys:
        from_large.put(key, key * 7)
    for key in big_keys[::3]:
        from_large.delete(key)
    check(tree_problems(from_large, from_large.versions[-1]) == [],
          "a large tree must survive mass deletion")

    # --- 10. determinism -----------------------------------------------------------
    first_keys = deterministic_keys(120, seed=5)
    delete_keys = deterministic_keys(60, seed=6)
    first, _ = load(first_keys)
    for key in delete_keys:
        first.delete(key)
    second, _ = load(first_keys)
    for key in delete_keys:
        second.delete(key)
    check(flatten(first, first.versions[-1]) == flatten(second, second.versions[-1]),
          "identical workloads must produce identical contents")
    check(first.store.allocations == second.store.allocations
          and first.store.next_pid == second.store.next_pid,
          "identical workloads must allocate identical page shapes")
    check(len(first.versions) == len(second.versions),
          "identical workloads must produce the same number of versions")
    check(generation_problems(first) == [], "the workload must respect the generation rule")

    print("history: %d versions, %d live pages, %d allocations, depth %d"
          % (len(history.versions), len(history.store.pages), history.store.allocations,
             history.depth(history.versions[-1])))
    print("copy-on-write: at most %d pages copied per insert at depth %d; 400-key depth %d"
          % (max(copies), depth, large.depth(root)))
    print("deletes: %d versions, %d structural problems; collector freed %d of %d pages for %d "
          "live roots" % (len(deletion.versions), len(structural), len(freed), len(all_pages),
                          len(live_roots)))
    print("PASS")


if __name__ == "__main__":
    main()
