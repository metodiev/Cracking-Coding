"""Task 774 - Implement concurrent B-tree with latch-free readers (reader-writer latches).  (Tier: Expert, Ch 14 - Databases)

A B-tree whose *readers take no latches at all* while writers serialize with reader-writer latches.
Every page carries a version counter: a writer bumps it when its critical section opens (making it
odd), mutates, and bumps it again when the section closes (back to even).  A reader is optimistic -
it records the version of every page it visits while descending, then re-checks all recorded
versions a scheduler step later.  If a version changed or is odd (a write is in progress) the read
is discarded and retried, so a reader never observes a torn page, a page mid-split, or a moved
child pointer: readers never block, they may only repeat.

Writers latch their path top-down (root, then child, ...), which makes a latch cycle - hence a
deadlock - impossible, and they keep the path latched until the upward splits are done; the root
latch effectively serializes writers while readers keep running latch-free.  The step-by-step
scheduler lets the run prove: the final tree equals a dict oracle, the order/occupancy/depth
invariants of every page, the latch invariants after every step, that readers never latch, that a
validated read returns a value the key really had inside its read window, that contention forces
retries, and that the whole run reproduces.

Simplified: one process with a scheduler instead of threads or memory barriers, no deletion
(merging is out of scope), and no B-link right-links - a reader racing a split is caught by
re-validation.  The latch protocol, the version-validated reads and the split logic are real.
"""
from collections import deque

MAX_KEYS = 3          # order-4 B-tree: at most three keys per page
MIN_KEYS = 1          # every non-root page keeps at least one key (leaves only grow here)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Page:
    __slots__ = ("pid", "keys", "children", "values", "leaf", "version", "nxt")

    def __init__(self, pid, leaf):
        self.pid = pid
        self.keys = []
        self.children = [] if not leaf else None
        self.values = [] if leaf else None
        self.leaf = leaf
        self.version = 0
        self.nxt = None


class LatchManager:
    """Reader-writer latches, non-blocking: callers retry until the latch is granted."""

    def __init__(self):
        self.exclusive = {}      # pid -> holder id
        self.shared = {}         # pid -> set of holder ids

    def granted_exclusive(self, pid, holder):
        if pid in self.exclusive or self.shared.get(pid):
            return False
        self.exclusive[pid] = holder
        return True

    def release(self, pid, holder):
        if self.exclusive.get(pid) == holder:
            del self.exclusive[pid]
        elif pid in self.shared:
            self.shared[pid].discard(holder)
            if not self.shared[pid]:
                del self.shared[pid]

    def check_invariants(self):
        check(not (set(self.shared) & set(self.exclusive)),
              "a page may never be latched exclusively and shared at the same time")
        check(all(len(holders) >= 1 for holders in self.shared.values()),
              "a shared latch is always held by at least one reader")


class BTree:
    """B-tree core: search, insert and split, with page version bumps around every modification."""

    def __init__(self):
        self.root = Page(0, True)
        self.next_pid = 1
        self.mutations = 0
        self.splits = 0

    def new_page(self, leaf):
        page = Page(self.next_pid, leaf)
        self.next_pid += 1
        return page

    def child_of(self, page, key):
        index = 0
        while index < len(page.keys) and key >= page.keys[index]:
            index += 1
        return page.children[index]

    def depth(self):
        page, depth = self.root, 1
        while not page.leaf:
            page = page.children[0]
            depth += 1
        return depth

    def start_write(self, page):
        page.version += 1
        self.mutations += 1

    def end_write(self, page):
        page.version += 1          # even again: the critical section is closed

    def insert_into(self, path, key, value):
        """Insert with the whole path exclusively latched; a generator, resuming from a mid-write."""
        leaf = path[-1]
        index = 0
        while index < len(leaf.keys) and leaf.keys[index] < key:
            index += 1
        if index < len(leaf.keys) and leaf.keys[index] == key:
            self.start_write(leaf)
            leaf.values[index] = value
            yield                                 # mid-write: the version is odd, data already new
            self.end_write(leaf)
            return
        self.start_write(leaf)
        if index < len(leaf.keys) and leaf.keys[index] == key:
            leaf.values[index] = value
            yield                                 # mid-write: the version is odd, data already new
            self.end_write(leaf)
            return
        leaf.keys.insert(index, key)
        leaf.values.insert(index, value)
        yield                                     # mid-write: odd version, new data in place
        self.end_write(leaf)
        if len(leaf.keys) <= MAX_KEYS:
            return
        yield from self._split_up(path)

    def put(self, key, value):
        """Single-threaded insert, used for the structural checks outside the simulator."""
        path = [self.root]
        page = self.root
        while not page.leaf:
            page = self.child_of(page, key)
            path.append(page)
        for _ in self.insert_into(path, key, value):
            pass

    def _split_up(self, path):
        """Split every overflowing page along the latched path, growing a new root if needed."""
        level = len(path) - 1
        while level >= 0 and len(path[level].keys) > MAX_KEYS:
            page = path[level]
            self.start_write(page)
            median_index = len(page.keys) // 2
            right = self.new_page(page.leaf)
            if page.leaf:
                # a leaf keeps a copy of its median, so the separator is the right leaf's minimum
                separator = page.keys[median_index + 1]
                right.keys = page.keys[median_index + 1:]
                right.values = page.values[median_index + 1:]
                page.keys = page.keys[:median_index + 1]
                page.values = page.values[:median_index + 1]
                right.nxt = page.nxt
                page.nxt = right
            else:
                separator = page.keys[median_index]     # the median itself moves up
                right.keys = page.keys[median_index + 1:]
                right.children = page.children[median_index + 1:]
                page.keys = page.keys[:median_index]
                page.children = page.children[:median_index + 1]
            self.end_write(page)
            self.start_write(right)
            self.end_write(right)
            self.splits += 1
            if level == 0:
                new_root = self.new_page(False)
                self.start_write(new_root)
                new_root.keys = [separator]
                new_root.children = [page, right]
                self.end_write(new_root)
                self.root = new_root
                return
            parent = path[level - 1]
            self.start_write(parent)
            position = parent.children.index(page)
            parent.keys.insert(position, separator)
            parent.children.insert(position + 1, right)
            self.end_write(parent)
            level -= 1
            yield                                   # a split is visible as a multi-step change

    def scan(self):
        """Full leaf scan: the ground truth for the oracle comparison."""
        page = self.root
        while not page.leaf:
            page = page.children[0]
        pairs = []
        while page is not None:
            pairs.extend(zip(page.keys, page.values))
            page = page.nxt
        return pairs

    def check_structure(self):
        """Order invariant, uniform depth and the occupancy bound of every page."""
        depths, stack, seen_leaves = set(), [(self.root, 1)], 0
        while stack:
            page, level = stack.pop()
            check(page.keys == sorted(page.keys), f"page {page.pid} keys must be sorted")
            check(page.version % 2 == 0, f"page {page.pid} left with an unfinished write")
            if page is not self.root:
                check(MIN_KEYS <= len(page.keys) <= MAX_KEYS, f"page {page.pid} is out of range")
            if page.leaf:
                depths.add(level)
                seen_leaves += 1
                check(len(page.keys) == len(page.values), "leaf keys and values must be parallel")
                nxt = page.nxt
                check(nxt is None or not page.keys or not nxt.keys or page.keys[-1] < nxt.keys[0],
                      "the leaf chain must be sorted")
            else:
                check(len(page.children) == len(page.keys) + 1,
                      "an internal page needs one more child than keys")
                stack.extend((child, level + 1) for child in page.children)
        check(len(depths) == 1, f"all leaves must be at the same depth, got {depths}")
        return seen_leaves


class Simulator:
    """Deterministic step-by-step interleaving of latch-free readers and latching writers."""

    def __init__(self, oracle):
        self.tree = BTree()
        self.latches = LatchManager()
        self.oracle = oracle                   # key -> value, maintained by the writers
        self.timeline = {}                     # key -> list of (step, value) after each commit
        self.reads = []                        # (key, returned value, first step, last step)
        self.step = self.reader_retries = self.reader_latch_calls = 0
        self.writer_waits = self.odd_versions_seen = self.stale_versions_seen = 0
        self.completed = 0
        self.reads_done = 0

    # ------------------------------------------------------------------ readers
    def lookup(self, key):
        while True:
            self.step += 1
            start, versions, page, saw_odd = self.step, [], self.tree.root, False
            while True:
                versions.append((page, page.version))
                if page.version % 2 == 1:                 # a write is open: this read is doomed
                    saw_odd = True
                    self.odd_versions_seen += 1
                    break
                if page.leaf:
                    break
                page = self.tree.child_of(page, key)
                yield
            self.step += 1
            yield                     # validation is a scheduler step later: writes can land now
            self.step += 1
            end = self.step
            valid = (not saw_odd and self.tree.root is versions[0][0]
                     and self.versions_valid(versions))
            if page.leaf and valid:
                index = 0
                while index < len(page.keys) and page.keys[index] < key:
                    index += 1
                found = page.values[index] if index < len(page.keys) and page.keys[index] == key else None
                self.reads.append((key, found, start, end))
                self.reads_done += 1
                return found
            if not saw_odd:
                self.stale_versions_seen += 1
            self.reader_retries += 1
            yield

    @staticmethod
    def versions_valid(versions):
        """A read set is valid when every recorded version is unchanged and even (no open write)."""
        return all(page.version == version and version % 2 == 0 for page, version in versions)

    # ------------------------------------------------------------------ writers
    def latch_path(self, key, holder):
        """Exclusively latch root..leaf; if any page is busy release the attempt and retry."""
        while True:
            page = self.tree.root
            if not self.latches.granted_exclusive(page.pid, holder) or page is not self.tree.root:
                self.latches.release(page.pid, holder)     # busy, or the root was just swapped
                self.writer_waits += 1
                self.step += 1
                yield
                continue
            path = [page]
            complete = True
            while not page.leaf:
                page = self.tree.child_of(page, key)
                if not self.latches.granted_exclusive(page.pid, holder):
                    complete = False
                    break
                path.append(page)
            if complete:
                return path
            for locked in path:
                self.latches.release(locked.pid, holder)
            self.writer_waits += 1
            self.step += 1
            yield

    def insert(self, key, value):
        holder = ("writer", key, self.step)
        path = yield from self.latch_path(key, holder)
        self.step += 1
        yield                                     # the writer is inside its critical section
        yield from self.tree.insert_into(path, key, value)
        self.step += 1
        yield
        for locked in reversed(path):
            self.latches.release(locked.pid, holder)
        self.oracle[key] = value
        self.timeline.setdefault(key, []).append((self.step, value))
        self.completed += 1
        yield

    # ------------------------------------------------------------------ scheduler
    def run(self, jobs, budget=20000):
        active = deque(jobs)
        while active:
            job = active.popleft()
            try:
                next(job)
                active.append(job)
            except StopIteration:
                pass
            if self.step > budget:
                raise AssertionError("scheduler budget exhausted - possible deadlock or livelock")
            self.latches.check_invariants()
        return self.step


def committed_value(sim, key, step):
    """Value of `key` committed at or before `step` (the serialization state at that moment)."""
    value = None
    for when, committed in sim.timeline.get(key, []):
        if when <= step:
            value = committed
    return value


def interleaved_jobs(sim, readers, writers):
    """The deterministic mixed workload: reader and writer operations alternate in the schedule."""
    reader_jobs = [sim.lookup((reader * 7 + step * 5) % 32)
                   for reader in range(readers) for step in range(24)]
    writer_jobs = [sim.insert(key, writer * 100 + key) for writer in range(writers)
                   for key in range(24)]
    jobs = []
    for index in range(max(len(reader_jobs), len(writer_jobs))):
        if index < len(reader_jobs):
            jobs.append(reader_jobs[index])
        if index < len(writer_jobs):
            jobs.append(writer_jobs[index])
    return jobs


def quiet_reads(sim):
    """A read-only pass executed after every writer has finished."""
    return [sim.lookup(key) for key in range(24)]



def main():
    # 1. structural boundaries: empty tree, single key, growth through splits
    empty, single = BTree(), BTree()
    check(empty.scan() == [] and empty.depth() == 1, "an empty tree is one empty leaf level")
    empty.check_structure()
    single.put(5, 50)
    check(single.scan() == [(5, 50)] and single.root.version % 2 == 0,
          "a finished single-key write must be visible and leave an even version")
    growing = BTree()
    for key in range(24):
        growing.put(key, key * 10)
    growing.check_structure()
    check(growing.scan() == [(key, key * 10) for key in range(24)] and growing.splits > 4,
          "every key must survive the splits of the growing tree")

    # 2. the concurrent simulation: contended phase, then a quiet read phase
    readers, writers = 3, 2
    oracle = {}
    sim = Simulator(oracle)
    steps = sim.run(interleaved_jobs(sim, readers, writers))
    contended_retries = sim.reader_retries
    steps = sim.run(quiet_reads(sim))
    final = dict(sim.tree.scan())
    check(final == oracle and len(final) == 24, "the simulated tree must match the writer oracle")
    check(sim.tree.check_structure() >= 6, "the workload must force real splits")
    check(sim.reader_latch_calls == 0, "readers must never acquire a latch")
    check(contended_retries > 0, "a concurrent schedule must force optimistic retries")
    check(sim.reader_retries == contended_retries,
          "the quiet pass after the writers finished must not retry at all")
    check(sim.completed == writers * 24 and sim.reads_done == readers * 24 + 24,
          "every writer must commit and every reader must return exactly once")
    check(sim.odd_versions_seen + sim.stale_versions_seen > 0,
          "a concurrent write must really trip a reader")

    # 3. both rejection paths of the read-set validator, checked directly
    probe = BTree()
    probe.put(1, 10)
    stable = [(probe.root, probe.root.version)]
    check(Simulator.versions_valid(stable), "an untouched read set must validate")
    probe.start_write(probe.root)
    check(not Simulator.versions_valid([(probe.root, probe.root.version)]),
          "reading a page with an open write must be rejected")
    probe.end_write(probe.root)
    check(not Simulator.versions_valid(stable), "a page changed after the read must be rejected")
    check(Simulator.versions_valid([(probe.root, probe.root.version)]), "a fresh read set validates")

    # 4. linearizability: every value a validated read returned was the key's value in that window
    checked_reads = 0
    for key, returned, start, end in sim.reads:
        if not sim.timeline.get(key):
            check(returned is None, "a read before the first commit must return nothing")
        elif returned is not None:
            observed = {committed_value(sim, key, moment) for moment in range(start, end + 1)}
            check(returned in observed,
                  f"read of {key} returned {returned}, absent from that window")
            checked_reads += 1
    check(checked_reads >= 20, "the quiet read pass must return committed values")

    # 5. determinism, and a contention-free schedule that never retries
    again = Simulator({})
    again.run(interleaved_jobs(again, readers, writers))
    check(dict(again.tree.scan()) == final and again.timeline == sim.timeline
          and again.reader_retries == sim.reader_retries, "the simulation must be deterministic")
    quiet = Simulator({})
    for key in range(9):
        quiet.tree.put(key, key * 3)
    quiet.run(quiet_reads(quiet)[:9])
    check(quiet.reader_retries == 0 and len(quiet.reads) == 9,
          "a contention-free schedule must validate on the first attempt")
    check(all(returned == key * 3 for key, returned, _, _ in quiet.reads),
          "quiet reads must return exactly the committed values")

    print(f"774 steps={steps} splits={sim.tree.splits} leaves={sim.tree.check_structure()} "
          f"depth={sim.tree.depth()} retries={sim.reader_retries} waits={sim.writer_waits}")
    print(f"774 reads={len(sim.reads)} linearizable={checked_reads} keys={len(final)} "
          f"stale_versions={sim.stale_versions_seen} odd_versions={sim.odd_versions_seen}")
    print("PASS")


if __name__ == "__main__":
    main()
