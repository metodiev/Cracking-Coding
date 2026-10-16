"""Task 632 - Implement anti-entropy Merkle tree reconciliation demo.  (Tier: Expert, Ch 9 — System Design and Scalability)

Two replicas keep the same key space and both hash it the same way: the key space is cut into a fixed
number of ranges by the tree's shape alone (not by how many keys a replica happens to hold), each leaf
hashes either "absent" or "key:version:value", and every internal node hashes its two children. Two
replicas therefore produce the same root whenever they hold the same data, and any difference - a
changed value, an added key, a deleted key - changes exactly the path from that leaf to the root.
Anti-entropy then walks the two trees together: if two subtree hashes are equal the whole subtree is
skipped (nothing to send), otherwise the walk descends, so the exchange costs O(k log n) hashes and k
entry transfers for k differences instead of n entries. Both trees are compared and the winning entry
per key (higher version, ties broken by value) is written into *both* stores, so one pass converges
them; a second pass transfers nothing.

The demo is deliberately small and deterministic: a 2^8 key space, keys "k000".."k255", and a seeded
LCG to pick which entries differ. Hashes are sha256 over a canonical string, so the digests are stable
across runs and processes. What is verified: shape independence from the contents, root equality,
insertion-order independence, sensitivity to every kind of change, exact transfer counts (one entry
per genuinely differing key, none for equal subtrees), the visit bound of O(k log n) against the
naive n-entry comparison, convergence in one pass and idempotence afterwards, deletion propagation,
empty/single element boundaries, corrupted-leaf repair, and run-to-run determinism.

Checks: root properties, diff detection per change kind, exact transfer and visit counts for 1 and
several differences, convergence and idempotence, deletions, empty and single-key stores, and the
efficiency comparison against a full transfer.
"""

import hashlib


def check(ok, message):
    if not ok:
        raise AssertionError(message)


ABSENT = "absent"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def entry_digest(key, entry):
    """A leaf hashes the whole entry: 'absent' (no entry at all), a tombstone, or a version+value,
    so a value change, an add and a delete each change the path to the root."""
    if entry is None:
        return digest(f"{key}:{ABSENT}")
    version, value = entry
    return digest(f"{key}:{version}:{'~' if value is None else value}")


def merge_entries(left, right):
    """Deterministic conflict resolution: higher version wins; at the same version a live value beats
    a tombstone and otherwise the greater value string wins. `None` means 'no entry at all'."""
    if left is None:
        return right
    if right is None:
        return left
    if left[0] != right[0]:
        return left if left[0] > right[0] else right
    if left[1] is None or right[1] is None:
        if left[1] is None and right[1] is None:
            return left
        return right if left[1] is None else left
    return left if left[1] >= right[1] else right


class MerkleNode:
    __slots__ = ("lo", "hi", "digest", "left", "right", "key", "entry")

    def __init__(self, lo, hi, node_digest, left=None, right=None, key=None, entry=None):
        self.lo = lo
        self.hi = hi
        self.digest = node_digest
        self.left = left
        self.right = right
        self.key = key
        self.entry = entry                    # leaves only; body nodes have entry None

    def is_leaf(self):
        return self.left is None


def build_tree(store, bits=8):
    """Hash tree over the fixed key space [0, 2^bits): the shape depends only on `bits`."""
    def build(lo, hi):
        if hi - lo == 1:
            key = f"k{lo:03d}"
            return MerkleNode(lo, hi, entry_digest(key, store.get(key)), key=key,
                              entry=store.get(key))
        mid = (lo + hi) // 2
        left, right = build(lo, mid), build(mid, hi)
        return MerkleNode(lo, hi, digest(f"{left.digest}|{right.digest}"), left, right)
    return build(0, 1 << bits)


class Replica:
    """A replica of the key space: entries are (version, value) pairs."""

    def __init__(self, name, entries=None, bits=8):
        self.name = name
        self.bits = bits
        self.entries = dict(entries or {})
        self.root = build_tree(self.entries, bits)
        self.trees_built = 1

    def rebuild(self):
        self.root = build_tree(self.entries, self.bits)
        self.trees_built += 1
        return self.root

    def set(self, key, version, value):
        self.rebuild()
        self.entries[key] = (version, value)
        return self.rebuild()
    def remove(self, key, version):
        """A delete is a tombstone with a version, so it can never be resurrected by a stale copy."""
        self.rebuild()
        self.entries[key] = (version, None)
        return self.rebuild()

    def snapshot(self):
        return dict(self.entries)

    def visible(self):
        """What a reader sees: tombstones are deletions, not data."""
        return {key: entry for key, entry in self.entries.items() if entry[1] is not None}

    def keys(self):
        return sorted(self.entries)


def reconcile(first, second, resolution=merge_entries):
    """One anti-entropy pass: descend only into differing ranges and repair both stores.

    Returns a plan with the number of visited tree nodes, the transferred keys and the details.
    """
    plan = {"visited": 0, "transfers": [], "skipped_subtrees": 0, "details": []}

    def walk(left, right):
        plan["visited"] += 1
        if left.digest == right.digest:
            plan["skipped_subtrees"] += 1        # identical subtree: nothing to send below here
            return
        if left.is_leaf():
            key = left.key
            winner = resolution(left.entry, right.entry)
            if left.entry != right.entry:
                plan["transfers"].append(key)
                plan["details"].append((key, left.entry, right.entry, winner))
            if winner is None:
                first.entries.pop(key, None)
                second.entries.pop(key, None)
            else:
                first.entries[key] = winner
                second.entries[key] = winner
            return
        walk(left.left, right.left)
        walk(left.right, right.right)

    walk(first.root, second.root)
    first.rebuild()
    second.rebuild()
    return plan


def naive_diff(first, second):
    """The baseline anti-entropy sends every entry it holds; used to price the tree walk."""
    return sorted(set(first.entries) | set(second.entries))


def differing_leaves(first, second, found=None):
    """List the leaves whose digests differ - the 'only the changed path differs' probe."""
    found = [] if found is None else found
    if first.digest == second.digest:
        return found
    if first.is_leaf():
        found.append((first.key, first.entry, second.entry))
        return found
    differing_leaves(first.left, second.left, found)
    differing_leaves(first.right, second.right, found)
    return found


def lcg(seed, modulus=65537):
    state = seed

    def nxt(bound):
        nonlocal state
        state = (state * 75 + 74) % modulus
        return state % bound

    return nxt


def make_population(count, seed=8):
    return {f"k{index:03d}": (1, f"v{seed}-{index}") for index in range(count)}


def main():
    bits = 8
    domain = 1 << bits

    # --- 1. root properties --------------------------------------------------------------------
    population = make_population(64, seed=1)
    left = Replica("A", population, bits)
    right = Replica("B", population, bits)
    check(left.root.digest == right.root.digest, "identical stores hash to the same root")
    check(len(left.root.digest) == 64 and left.root.is_leaf() is False, "the root is a sha256 body node")
    check(Replica("C", dict(reversed(list(population.items()))), bits).root.digest == left.root.digest,
          "the tree shape follows the key space, not the insertion order")

    changed = Replica("D", population, bits)
    changed.set("k010", 2, "tampered")
    check(changed.root.digest != left.root.digest, "a changed value changes the root")

    added = Replica("E", population, bits)
    added.set("k200", 1, "new")
    check(added.root.digest != left.root.digest, "an added key changes the root")

    removed = Replica("F", population, bits)
    removed.remove("k010", 2)
    check(removed.root.digest != left.root.digest, "a deleted key changes the root")
    check("k010" in removed.snapshot() and "k010" not in removed.visible(),
          "the delete is kept as a tombstone, invisible to readers")

    check(differing_leaves(left.root, changed.root) == [("k010", (1, "v1-10"), (2, "tampered"))],
          "only the leaf of the changed key differs")
    check(len(differing_leaves(left.root, removed.root)) == 1
          and differing_leaves(left.root, removed.root)[0][0] == "k010",
          "a deleted key shows up as exactly one differing leaf too")

    # --- 2. one difference costs one transfer and O(log n) visits -------------------------------
    first = Replica("A", make_population(domain, seed=3), bits)
    second = Replica("B", make_population(domain, seed=3), bits)
    first.set("k123", 7, "one-change")
    plan = reconcile(first, second)
    check(plan["transfers"] == ["k123"], f"exactly the one difference is transferred: {plan}")
    check(plan["visited"] <= 2 * bits + 2, f"only the two paths are walked: {plan['visited']} nodes")
    check(plan["skipped_subtrees"] >= bits - 1, "the untouched half of every level is skipped")
    check(first.snapshot() == second.snapshot(), "one pass converges both replicas")
    again = reconcile(first, second)
    check(again["transfers"] == [] and again["visited"] == 1,
          "a second pass visits only the root and transfers nothing")
    check(plan["visited"] < len(naive_diff(first, second)) / 8,
          "the tree walk is far cheaper than sending every entry")

    # --- 3. several differences, deletions and conflicts ----------------------------------------
    first = Replica("A", make_population(domain, seed=4), bits)
    second = Replica("B", make_population(domain, seed=4), bits)
    first.remove("k005", 4)                       # deleted on A (a tombstone with a version)
    second.remove("k006", 4)                      # deleted on B
    first.entries["k007"] = (5, "newer")          # newer version on A
    second.entries["k007"] = (3, "older")
    second.entries["k008"] = (2, "added-on-b")
    first.entries["k009"] = (1, "same-version-a")
    second.entries["k009"] = (1, "same-version-b")  # tie broken by the value string
    first.rebuild()
    second.rebuild()
    plan = reconcile(first, second)
    check(sorted(plan["transfers"]) == [f"k{index:03d}" for index in (5, 6, 7, 8, 9)],
          f"five differing keys were found: {plan['transfers']}")
    check(len(plan["transfers"]) == len(plan["details"]), "each transfer is described once")
    check(first.snapshot() == second.snapshot(), "the replicas converge on one combined state")
    check(first.entries["k007"] == (5, "newer"), "the higher version wins a conflict")
    check(first.entries["k009"] == (1, "same-version-b"), "an equal version is broken by the value")
    check("k005" not in first.visible() and "k005" not in second.visible()
          and "k006" not in second.visible(), "deletions propagate to both replicas")
    check(first.entries["k008"] == (2, "added-on-b"), "a key only one side had is pulled over")
    check(plan["visited"] <= 2 * bits * len(plan["transfers"]) + 2 * bits,
          f"visits stay bounded by O(k log n): {plan['visited']}")
    check(plan["visited"] < domain // 4, "and are far from a full scan")

    # --- 4. efficiency against the naive exchange -----------------------------------------------
    first = Replica("A", make_population(domain, seed=5), bits)
    second = Replica("B", make_population(domain, seed=5), bits)
    picked = [f"k{index:03d}" for index in (0, 17, 33, 64, 99, 128, 200, 255)]
    for index, key in enumerate(picked):
        first.entries[key] = (9, f"chg{index}")
    first.rebuild()
    plan = reconcile(first, second)
    check(sorted(plan["transfers"]) == picked, "every changed key is transferred exactly once")
    cost = plan["visited"]
    naive = len(naive_diff(first, second))
    check(naive == domain and cost <= 2 * bits * len(picked) + 2 * bits,
          f"the walk visited {cost} nodes instead of comparing {naive} entries")
    check(cost * 4 < 2 * naive,
          f"reconciliation stays a fraction of a full two-way exchange: {cost} vs {2 * naive}")
    second_plan = reconcile(first, second)
    check(second_plan["transfers"] == [] and second_plan["visited"] == 1,
          "reconciling an already consistent pair is a single root comparison")

    # --- 5. corrupted leaf repair ---------------------------------------------------------------
    first = Replica("A", make_population(domain, seed=6), bits)
    second = Replica("B", make_population(domain, seed=6), bits)
    second.entries["k077"] = (1, "corrupted")     # silent bit rot on B
    second.rebuild()
    plan = reconcile(first, second)
    check(plan["transfers"] == ["k077"], "the corrupted key is the only one exchanged")
    check(first.snapshot() == second.snapshot() and second.entries["k077"] == (1, "v6-77"),
          "anti-entropy repairs the corruption from the healthy replica")
    corrupted = Replica("C", make_population(domain, seed=6), bits)
    corrupted.entries["k128"] = (99, "newer-version")
    corrupted.rebuild()
    plan = reconcile(first, corrupted)
    check(plan["transfers"] == ["k128"] and first.entries["k128"] == (99, "newer-version"),
          "a higher version flows in the other direction as well")
    check(corrupted.entries == first.entries, "and both sides end up identical again")

    # --- 6. boundaries -------------------------------------------------------------------------
    empty_a, empty_b = Replica("A", {}, bits), Replica("B", {}, bits)
    check(empty_a.root.digest == empty_b.root.digest, "two empty replicas have the same root")
    check(reconcile(empty_a, empty_b)["transfers"] == [] and empty_a.trees_built == 2,
          "nothing to reconcile, and the roots are rebuilt after the pass")

    filled = Replica("A", {"k000": (1, "only")}, bits)
    check(reconcile(empty_b, filled)["transfers"] == ["k000"], "an empty replica pulls the entry")
    check(empty_b.snapshot() == filled.snapshot() == {"k000": (1, "only")},
          "both replicas now hold the single entry")
    check(reconcile(filled, Replica("B", {"k000": (1, "only")}, bits))["visited"] == 1,
          "a single-key space is reconciled with one comparison")

    same = Replica("A", {"k001": (1, "x")}, bits)
    other = Replica("B", {"k001": (1, "x")}, bits)
    check(same.root.digest == other.root.digest and reconcile(same, other)["visited"] == 1,
          "one equal key short-circuits at the root")

    # --- 7. determinism -----------------------------------------------------------------------
    seed_run = Replica("A", make_population(domain, seed=11), bits)
    first_plan = reconcile(seed_run, Replica("B", make_population(domain, seed=12), bits))
    repeat = Replica("A", make_population(domain, seed=11), bits)
    second_plan = reconcile(repeat, Replica("B", make_population(domain, seed=12), bits))
    check(first_plan["transfers"] == second_plan["transfers"]
          and first_plan["visited"] == second_plan["visited"],
          "the same inputs always produce the same walk and the same transfers")
    check(len(first_plan["transfers"]) == domain,
          "replicas with disjoint data transfer every key of the domain")

    plan = reconcile(Replica("A", make_population(domain, seed=5), bits),
                     Replica("B", make_population(domain, seed=5), bits))
    print(f"merkle: domain={domain} bits={bits} equal-roots-visited={plan['visited']} "
          f"cheap-walk-visits={cost} naive-entries={naive}")


if __name__ == "__main__":
    main()
    print("PASS")
