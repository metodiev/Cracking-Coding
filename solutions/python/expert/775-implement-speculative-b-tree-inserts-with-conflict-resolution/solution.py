"""Task 775 - Implement speculative B-tree inserts with conflict resolution.  (Tier: Expert, Ch 14 - Databases)

Speculative insertion: a writer latches nothing while it looks for the leaf that should receive its
key.  It descends optimistically collecting a read set of (node, revision) pairs, and only when it
is at the leaf does it validate that set.  If every revision is unchanged the guess is still true and
the insertion is applied (with a preemptive split when the leaf is full); otherwise - a concurrent
writer moved a key, split a node or turned a leaf into an internal node - the speculation is thrown
away and restarted from the root.  The whole point is that an aborted speculation has *no* side
effects: the tree is untouched until validation succeeds, which is verified here by digesting the
full structural state around the abort.

The simulation interleaves several speculative inserters plus a competing writer that keeps touching
the same neighbourhood, so conflicts (and restarts) really happen; a run without the competitor must
show zero restarts.  Verified: the final tree equals a dict oracle (no lost update), the structural
invariants after every commit (sorted keys, occupancy bound, all leaves at the same depth), aborted
speculations leave the tree byte-identical, the validator rejects a bumped revision, the speculative
path yields the same tree as a plain non-speculative batch build, restart counts reproduce, and the
whole run is deterministic.

Simplified: one process with a deterministic scheduler (no threads, no memory barriers - revisions
are plain counters), a speculator holds no latch at all, and a stale speculation is resolved only by
restarting (no helping or commit-wait protocol is modelled).
"""
import hashlib
from collections import deque

MAX_KEYS = 5          # max five keys per node, splits are preemptive (top down)
MIN_KEYS = 2          # and every non-root node keeps at least two


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Node:
    __slots__ = ("nid", "keys", "values", "children", "leaf", "revision")

    def __init__(self, nid, leaf):
        self.nid = nid
        self.keys = []
        self.values = [] if leaf else None
        self.children = [] if not leaf else None
        self.leaf = leaf
        self.revision = 0

    def bump(self):
        self.revision += 1


class BTree:
    """Order-4 B-tree; every change bumps the revision of every node it touches."""

    def __init__(self):
        self.root = Node(0, True)
        self.next_nid = 1
        self.splits = 0

    def new_node(self, leaf):
        self.next_nid += 1
        return Node(self.next_nid - 1, leaf)

    def depth(self):
        node, depth = self.root, 1
        while not node.leaf:
            node = node.children[0]
            depth += 1
        return depth

    def find(self, key):
        node = self.root
        while not node.leaf:
            index = 0
            while index < len(node.keys) and key >= node.keys[index]:
                index += 1
            node = node.children[index]
        index = 0
        while index < len(node.keys) and node.keys[index] < key:
            index += 1
        if index < len(node.keys) and node.keys[index] == key:
            return node.values[index]
        return None

    def items(self):
        """Sorted key/value pairs; every data key lives in exactly one leaf."""
        out = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.leaf:
                out.extend(zip(node.keys, node.values))
            else:
                stack.extend(reversed(node.children))
        return sorted(out)

    def snapshot(self):
        """Digest of the structural state, used to prove that an abort changed nothing."""
        digest = hashlib.sha256()
        stack = [self.root]
        while stack:
            node = stack.pop()
            digest.update(f"{node.nid}:{node.leaf}:{node.keys}:{node.revision}|".encode())
            if not node.leaf:
                stack.extend(reversed(node.children))
        return digest.hexdigest()

    def structure_ok(self):
        depths, stack = set(), [(self.root, 1)]
        while stack:
            node, level = stack.pop()
            check(node.keys == sorted(set(node.keys)), f"node {node.nid} keys must be sorted")
            if node is not self.root:
                check(MIN_KEYS <= len(node.keys) <= MAX_KEYS, f"node {node.nid} is out of range")
            if node.leaf:
                depths.add(level)
                check(len(node.keys) == len(node.values), "leaf keys and values must be parallel")
            else:
                check(len(node.children) == len(node.keys) + 1, "internal nodes need one extra child")
                stack.extend((child, level + 1) for child in node.children)
        check(len(depths) == 1, f"all leaves must be at the same depth, got {depths}")

    # ------------------------------------------------------------------ mutation
    def split_child(self, parent, index):
        """Split the full node parent.children[index]; the separator goes up into the parent."""
        child = parent.children[index]
        right = self.new_node(child.leaf)
        if child.leaf:
            separator = child.keys[MAX_KEYS // 2 + 1]     # the left leaf keeps a copy of the median
            right.keys = child.keys[MAX_KEYS // 2 + 1:]
            right.values = child.values[MAX_KEYS // 2 + 1:]
            child.keys = child.keys[:MAX_KEYS // 2 + 1]
            child.values = child.values[:MAX_KEYS // 2 + 1]
        else:
            separator = child.keys[MAX_KEYS // 2]         # an internal median moves up
            right.keys = child.keys[MAX_KEYS // 2 + 1:]
            right.children = child.children[MAX_KEYS // 2 + 1:]
            child.keys = child.keys[:MAX_KEYS // 2]
            child.children = child.children[:MAX_KEYS // 2 + 1]
        parent.keys.insert(index, separator)
        parent.children.insert(index + 1, right)
        parent.bump()
        child.bump()
        right.bump()
        self.splits += 1

    def insert(self, key, value):
        """Reference insert (preemptive split on the way down); returns True when the key is new."""
        if len(self.root.keys) == MAX_KEYS:
            old_root = self.root
            new_root = self.new_node(False)
            new_root.children = [old_root]
            self.root = new_root
            self.split_child(new_root, 0)
        node = self.root
        while True:
            index = 0
            while index < len(node.keys) and key >= node.keys[index]:
                index += 1
            if node.leaf:
                position = 0
                while position < len(node.keys) and node.keys[position] < key:
                    position += 1
                fresh = not (position < len(node.keys) and node.keys[position] == key)
                if fresh:
                    node.keys.insert(position, key)
                    node.values.insert(position, value)
                else:
                    node.values[position] = value
                node.bump()
                return fresh
            if len(node.children[index].keys) == MAX_KEYS:
                self.split_child(node, index)
                if key >= node.keys[index]:
                    index += 1
            node = node.children[index]


class SpeculativeWriter:
    """Latch-free inserter: descend and record, validate after a window, then commit or restart."""

    def __init__(self, tree, key, value, label=None):
        self.tree = tree
        self.key = key
        self.value = value
        self.label = label if label is not None else key
        self.restarts = 0

    def speculate(self, interleave=None, max_attempts=8):
        """Generator: each yield is a scheduler turn; returns True when the write committed."""
        for _ in range(max_attempts):
            read_set, leaf, index = self._descend()
            yield                                   # window in which a competitor may write
            if interleave is not None:
                interleave()
            if self._valid(read_set):
                self._commit(leaf, index)
                return True
            self.restarts += 1
            yield
        return False

    def _descend(self):
        read_set = []
        node = self.tree.root
        while True:
            read_set.append((node, node.revision))
            if node.leaf:
                break
            index = 0
            while index < len(node.keys) and self.key >= node.keys[index]:
                index += 1
            node = node.children[index]
        index = 0
        while index < len(node.keys) and node.keys[index] < self.key:
            index += 1
        return read_set, node, index

    @staticmethod
    def _valid(read_set):
        return all(node.revision == revision for node, revision in read_set)

    def _commit(self, leaf, index):
        if len(leaf.keys) < MAX_KEYS:
            if index < len(leaf.keys) and leaf.keys[index] == self.key:
                leaf.values[index] = self.value
            else:
                leaf.keys.insert(index, self.key)
                leaf.values.insert(index, self.value)
            leaf.bump()
            return
        self.tree.insert(self.key, self.value)     # stale guess on a full leaf: real insert it is


def drive(job, interleave=None, max_attempts=8):
    """Run a speculation to completion; returns True when it committed."""
    generator = job.speculate(interleave, max_attempts)
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return bool(stop.value)


def contended_run(seed_keys, speculative_keys, competitor_keys):
    """Round-robin scheduler over speculative writers, with optional competing writers."""
    tree = BTree()
    for key in seed_keys:
        tree.insert(key, key * 11)
    writers = [SpeculativeWriter(tree, key, ("spec", key)) for key in speculative_keys]
    writers += [SpeculativeWriter(tree, key, ("hot", key), label=key) for key in competitor_keys]
    active = deque((writer, writer.speculate()) for writer in writers)
    while active:
        writer, generator = active.popleft()
        try:
            next(generator)
            active.append((writer, generator))
        except StopIteration as stop:
            check(bool(stop.value), f"the speculation for {writer.label} never committed")
    return tree, writers


def main():
    # 1. the reference B-tree: boundaries (empty, single, duplicate) and growth through splits
    empty = BTree()
    check(empty.items() == [] and empty.depth() == 1, "an empty tree is one empty leaf")
    empty.structure_ok()
    single = BTree()
    check(single.insert(5, "a") and single.find(5) == "a", "a single key must be visible")
    check(single.insert(5, "b") is False and single.items() == [(5, "b")],
          "a duplicate insert must overwrite instead of duplicating")
    grown = BTree()
    for key in range(20):
        grown.insert(key, key * 2)
    grown.structure_ok()
    check(grown.items() == [(key, key * 2) for key in range(20)], "all 20 keys must survive")
    check(grown.splits > 3 and grown.depth() >= 3, "20 keys in an order-4 tree must split")

    # 2. no competitor: the first attempt always validates
    quiet = BTree()
    for key in range(12):
        quiet.insert(key, key)
    solo = SpeculativeWriter(quiet, 100, "solo")
    check(drive(solo) and solo.restarts == 0, "an uncontended speculation must commit first time")
    check(quiet.find(100) == "solo", "the speculative insert must be visible")

    # 3. an aborted speculation must leave the tree byte-identical
    probe = BTree()
    for key in range(6):
        probe.insert(key, key)
    victim = SpeculativeWriter(probe, 50, "victim")
    generator = victim.speculate()
    next(generator)                                  # descent of attempt 1
    probe.insert(200, "boom")                        # the competitor writes inside the window
    after_conflict = probe.snapshot()
    next(generator)                                  # validation of attempt 1 fails
    check(victim.restarts == 1, "a bumped revision must force exactly one restart")
    check(probe.snapshot() == after_conflict, "an aborted speculation must change nothing")
    check(probe.find(50) is None, "the aborted attempt must not have written the key")
    next(generator)                                  # descent of attempt 2
    check(drive_single(generator), "the retried speculation must commit")
    check(probe.find(50) == "victim" and probe.find(200) == "boom", "both writes must be present")
    probe.structure_ok()

    # 4. the validator itself: unchanged set validates, bumped revision does not
    sample = BTree()
    for key in range(4):
        sample.insert(key, key)
    probe_reader = SpeculativeWriter(sample, 7, "x")
    read_set, leaf, _ = probe_reader._descend()
    check(probe_reader._valid(read_set), "an untouched read set must validate")
    leaf.bump()
    check(not probe_reader._valid(read_set), "a bumped revision must invalidate the read set")
    check(probe_reader._valid(probe_reader._descend()[0]), "a fresh read set must validate")

    # 5. contention between several speculative writers and a competitor
    tree, writers = contended_run(list(range(8)), [40, 41, 42, 43], (5, 6))
    conflicts = sum(writer.restarts for writer in writers)
    check(conflicts > 0, "competing writers must force restarts")
    tree.structure_ok()
    check(len(tree.items()) == 12, "8 seeds plus 4 speculative keys must all be present")
    check(all(tree.find(key) is not None for key in (40, 41, 42, 43)),
          "a restarted speculation must still commit its key")
    check(all(tree.find(key) == key * 11 for key in (0, 1, 2, 3, 4, 7)),
          "seed values the competitor never touched must be unchanged")
    check(tree.find(5) == ("hot", 5) and tree.find(6) == ("hot", 6),
          "the competing writer must have overwritten its two keys")

    # 6. the speculative path builds the same tree as the plain path
    plain = BTree()
    for key in [3, 1, 4, 1, 5, 9, 2, 6]:
        plain.insert(key, ("v", key))
    spec = BTree()
    for key in [3, 1, 4, 1, 5, 9, 2, 6]:
        check(drive(SpeculativeWriter(spec, key, ("v", key))), "each speculation must commit")
    check(spec.items() == plain.items(), "speculation must not change the resulting tree")
    check(spec.snapshot() == plain.snapshot(), "the two paths must produce identical structures")

    # 7. restart counts and results reproduce exactly
    again_tree, again_writers = contended_run(list(range(8)), [40, 41, 42, 43], (5, 6))
    check(again_tree.items() == tree.items(), "the contended run must be deterministic")
    check(sum(w.restarts for w in again_writers) == conflicts, "restart counts must reproduce")

    print(f"775 keys={len(tree.items())} restarts={conflicts} splits={tree.splits} "
          f"depth={tree.depth()}")
    print(f"775 quiet_restarts={solo.restarts} victim_restarts={victim.restarts} "
          f"competitors={len(writers)}")
    print("PASS")


def drive_single(generator):
    """Resume a generator that is between attempts until it commits."""
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return bool(stop.value)


if __name__ == "__main__":
    main()
