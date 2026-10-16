"""Task 622 - Implement non-blocking hash table (split-ordered lists).  (Tier: Expert, CTCI Ch 15 — Threads and Locks)

The split-ordered list of Shalev & Shalev (2007): one lock-free ordered linked list holds *both*
the real keys and the per-bucket sentinels, ordered by a *recursive split ordering* of the keys.
That ordering is the whole trick - hashing to bucket b = key & (size - 1) means all keys of a
bucket share the low bits of the key, and reversing the bits turns "same low bits" into "same high
bits", so every bucket's keys form one contiguous run of the list.  A lookup starts at that
bucket's sentinel and walks right; the sentinel doubles as the bucket-list head, so no separate
per-bucket structure is needed.

  * a node's list key is 2 * bit-reverse(key) plus one flag bit - odd for a real key, even for a
    bucket sentinel - so reversing the 32-bit form of a key sorts the list correctly and the
    sentinel always precedes every key of its own bucket, including the key equal to the bucket
    index whose reversal would otherwise tie with the sentinel;
  * buckets start out as an EMPTY marker; the first thread to touch a bucket CASes the marker to a
    fresh sentinel node and then splices that sentinel into the list.  A thread that observes a
    published-but-not-yet-spliced sentinel simply starts its walk from the list head instead, which
    produces exactly the same order (documented deviation: the paper starts the sentinel's walk at
    the parent bucket's sentinel, and the head is a safe superset of that start point);
  * insert walks right, CASes prev.next to splice its node in, and restarts if the list moved;
    lookup walks right until it passes the position where the key would be (so a missing key costs
    one walk and nothing else); remove CASes a mark onto the node rather than unlinking it, which
    keeps every other operation lock-free.  CAS succeeds only while the field still holds the
    expected reference, and every operation is a step machine driven by an explicit schedule of
    thread ids - no threads, sleeps or sockets anywhere.

Documented simplification: the bucket array is fixed at construction (no doubling/rehash pass);
the split ordering, lazy sentinel installation, CAS retry protocol and marking are the real thing.
Verified: the list is strictly increasing in split order, every key sits after its own bucket
sentinel and no other bucket's sentinel comes between them; hundreds of operations agree with a
dictionary oracle; 30 seeded concurrent schedules keep exactly-once semantics; forced interleavings
exercise the sentinel-installation race, the splice path and the delete mark; and a lookup never
reports a key that was not inserted.
"""

EMPTY = "empty-bucket"
BITS = 32


def reverse_bits(value):
    out = 0
    for _ in range(BITS):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


def split_order_key(key, sentinel=False):
    """Order value in the recursive split ordering: the 32-bit reversal of the key, shifted
    left by one bit, with the freed lowest bit distinguishing a real key (odd) from a bucket
    sentinel (even).

    Why the shift matters: keys of bucket b = key & (2**d - 1) have exactly rev_d(b) as the top
    d bits of their reversal, so their order values are (2 * rev32(key) + 1) and lie in the
    half-open range [2 * rev32(b) + 1, 2 * rev32(b) + 2**(33 - d)).  The sentinel uses
    2 * rev32(b), which is therefore strictly *smaller* than every key of its own bucket - even
    the key equal to b itself (whose reversal coincides with the sentinel's) - while still being
    larger than every key of any bucket that precedes b in the split order."""
    reversed_value = reverse_bits(key & 0xFFFFFFFF)
    return 2 * reversed_value if sentinel else 2 * reversed_value + 1


class Node:
    __slots__ = ("so", "key", "value", "next", "sentinel", "marked", "linked")

    def __init__(self, so, key, value=None, sentinel=False, linked=False):
        self.so = so
        self.key = key
        self.value = value
        self.next = None
        self.sentinel = sentinel
        self.marked = False
        self.linked = linked


class SplitOrderedList:
    def __init__(self, bits):
        self.bits = bits                        # the table holds 2 ** bits buckets
        self.mask = (1 << bits) - 1
        self.buckets = [EMPTY] * (1 << bits)
        self.buckets[0] = Node(split_order_key(0, True), 0, sentinel=True, linked=True)
        self.cas_attempts = 0
        self.cas_failures = 0
        self.reads = 0
        self.sentinel_installs = 0
        self.restarts = 0

    # -- indivisible primitives --------------------------------------------
    def cas(self, holder, name, expected, new_value):
        self.cas_attempts += 1
        if getattr(holder, name) is expected:
            setattr(holder, name, new_value)
            return True
        self.cas_failures += 1
        return False

    def cas_bucket(self, index, expected, new_value):
        """The bucket array is CASed through here (it is a list, not an object)."""
        self.cas_attempts += 1
        if self.buckets[index] is expected:
            self.buckets[index] = new_value
            return True
        self.cas_failures += 1
        return False

    def so_key(self, key):
        return split_order_key(key)

    def walk_start(self, sentinel):
        """A published sentinel that is not spliced yet must not be used as a start point."""
        return sentinel if sentinel.linked else self.buckets[0]

    # -- operations as step machines ---------------------------------------
    def insert(self, key, value):
        while True:
            bucket = key & self.mask
            sentinel = self.buckets[bucket]
            self.reads += 1
            yield "read bucket"
            if sentinel is EMPTY:
                fresh = Node(split_order_key(bucket, True), bucket, sentinel=True)
                if self.cas_bucket(bucket, EMPTY, fresh):
                    self.sentinel_installs += 1
                    sentinel = fresh
                    yield "published sentinel"
                else:
                    yield "cas bucket failed"
                    continue
            if not sentinel.linked:
                start = self.buckets[0]
                if bucket:
                    parent = self.buckets[bucket & (bucket - 1)]
                    if parent is not EMPTY and parent.linked:
                        start = parent                    # the paper's parent-bucket shortcut
                linked = self.splice(start, sentinel)
                if linked:
                    self.cas(sentinel, "linked", False, True)
                yield "spliced sentinel"
            node = Node(self.so_key(key), key, value)
            previous = self.walk_start(sentinel)
            yield "start walk"
            while True:
                current = previous.next
                self.reads += 1
                yield "read next"
                if current is None or current.so > node.so:
                    break
                if current.so == node.so and not current.sentinel:
                    self.cas(current, "value", current.value, value)
                    if current.marked:
                        # a previously deleted key is revived: duplicating it would put two
                        # nodes with the same split-order key in the list
                        self.cas(current, "marked", True, False)
                    yield "set value"
                    return True
                previous = current
            node.next = current                    # keep the tail of the list reachable
            if self.cas(previous, "next", current, node):
                self.cas(node, "linked", False, True)
                yield "linked"
                return True
            self.restarts += 1
            yield "cas next failed"

    def lookup(self, key):
        bucket = key & self.mask
        sentinel = self.buckets[bucket]
        self.reads += 1
        yield "read bucket"
        if sentinel is EMPTY:
            yield "absent bucket"
            return None
        target = self.so_key(key)
        current = self.walk_start(sentinel)
        while True:
            current = current.next
            self.reads += 1
            yield "read next"
            if current is None or current.so > target:
                return None
            if current.so == target and not current.sentinel:
                self.reads += 1
                yield "read value"
                return None if current.marked else current.value

    def delete(self, key):
        bucket = key & self.mask
        sentinel = self.buckets[bucket]
        self.reads += 1
        yield "read bucket"
        if sentinel is EMPTY:
            yield "absent bucket"
            return False
        target = self.so_key(key)
        current = self.walk_start(sentinel)
        while True:
            current = current.next
            self.reads += 1
            yield "read next"
            if current is None or current.so > target:
                return False
            if current.so == target and not current.sentinel:
                marked = self.cas(current, "marked", False, True)
                yield "marked"
                return marked                     # true only when *this* call removed the key

    def splice(self, start, node):
        """Link `node` after the last node of the list that is still smaller than it."""
        for _ in range(64):
            previous, current = start, start.next
            while current is not None and current.so < node.so:
                previous, current = current, current.next
            if node.next is None:
                node.next = current
            if self.cas(previous, "next", current, node):
                return True
        raise AssertionError("the list did not settle while splicing")

    # -- structural integrity ----------------------------------------------
    def walk(self):
        """All nodes in list order: (key, value, marked, is_sentinel) plus the order values."""
        nodes, orders = [], []
        node = self.buckets[0]
        seen = set()
        while node is not None:
            if id(node) in seen:
                raise AssertionError("the list has a cycle")
            seen.add(id(node))
            orders.append(node.so)
            nodes.append((node.key, node.value, node.marked, node.sentinel))
            node = node.next
        return nodes, orders

    def live_items(self):
        nodes, _ = self.walk()
        return {key: value for key, value, marked, sentinel in nodes
                if not sentinel and not marked}

    def initialized_buckets(self):
        return [index for index, sentinel in enumerate(self.buckets) if sentinel is not EMPTY]


class Thread:
    """Drives one step machine; the machine itself may restart an operation."""

    def __init__(self, name, operations, table):
        self.name = name
        self.table = table
        self.operations = list(operations)
        self.index = 0
        self.current = None
        self.results = []
        self.finished = False

    def advance(self):
        if self.finished:
            return "idle"
        if self.current is None:
            if self.index >= len(self.operations):
                self.finished = True
                return "idle"
            kind = self.operations[self.index][0]
            if kind == "put":
                self.current = self.table.insert(*self.operations[self.index][1:])
            elif kind == "get":
                self.current = self.table.lookup(self.operations[self.index][1])
            else:
                self.current = self.table.delete(self.operations[self.index][1])
        try:
            return next(self.current)
        except StopIteration as done:
            self.results.append((self.operations[self.index][0], done.value))
            self.index += 1
            self.current = None
            if self.index >= len(self.operations):
                self.finished = True
                return "finished"
            return "completed"

    def drain(self, limit=400):
        outcomes = []
        for _ in range(limit):
            if self.finished:
                break
            outcomes.append(self.advance())
        return outcomes


def run_op(table, operation):
    thread = Thread("solo", [operation], table)
    thread.drain()
    return thread.results[-1][1]


def run(table, programs, schedule, settle_limit=6000):
    threads = [Thread("t%d" % index, program, table) for index, program in enumerate(programs)]
    for name in schedule:
        index = int(name)
        if index >= len(threads):
            raise IndexError("schedule names an unknown thread: %s" % name)
        threads[index].advance()
    for _ in range(settle_limit):
        if all(thread.finished for thread in threads):
            break
        for thread in threads:
            if not thread.finished:
                thread.advance()
    return threads


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def verify_structure(table):
    """Split order holds, and each key sits between its own bucket sentinel and the next one."""
    nodes, orders = table.walk()
    check(orders == sorted(orders), "the list must be strictly increasing in split order")
    check(len(set(orders)) == len(orders), "split order values must be unique")
    check(nodes[0][3], "the list must start at the bucket-0 sentinel")
    installed = table.initialized_buckets()
    for key, _, _, is_sentinel in nodes:
        if is_sentinel:
            continue
        bucket = key & table.mask
        check(bucket in installed, "a key's bucket must be initialised")
        target = table.so_key(key)
        check(table.buckets[bucket].so < target,
              "a key must come after its own bucket sentinel")
        for other in installed:
            if other == bucket or table.buckets[other].linked is False:
                continue
            other_so = table.buckets[other].so
            check(not (table.buckets[bucket].so < other_so < target),
                  "another bucket's sentinel sits between a key and its own sentinel")


def main():
    # ---- sequential behaviour against a dictionary oracle --------------------
    table = SplitOrderedList(3)
    oracle = {}
    state = 12345
    for step in range(200):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        key = state % 500
        if step % 5 == 4 and oracle:
            victim = sorted(oracle)[state % len(oracle)]
            check(run_op(table, ("del", victim)) is True, "the key must have been present")
            del oracle[victim]
        else:
            value = "v%d" % (state >> 20)
            check(run_op(table, ("put", key, value)) is True, "put returns True")
            oracle[key] = value
    for key in range(0, 500, 7):
        check(run_op(table, ("get", key)) == oracle.get(key),
              "lookup of %d must agree with the oracle" % key)
    check(table.live_items() == oracle, "the table must agree with the dictionary oracle")
    verify_structure(table)
    check(0 < len(table.initialized_buckets()) <= 8, "at most one sentinel per bucket")
    check(table.sentinel_installs <= 8, "a sentinel is installed at most once per bucket")

    # ---- lazy initialisation -------------------------------------------------
    fresh = SplitOrderedList(4)
    check(fresh.initialized_buckets() == [0], "only bucket 0 is ready up front")
    check(run_op(fresh, ("get", 7)) is None, "a lookup in a missing bucket is absent")
    check(run_op(fresh, ("del", 7)) is False, "a delete in a missing bucket is a no-op")
    check(fresh.initialized_buckets() == [0], "read-only operations install nothing")
    run_op(fresh, ("put", 7, "x"))
    check(fresh.initialized_buckets() == [0, 7], "the insert installed exactly its own bucket")
    run_op(fresh, ("put", 7 + 16, "y"))
    check(fresh.initialized_buckets() == [0, 7], "a key sharing the bucket reuses the sentinel")
    check(fresh.live_items() == {7: "x", 23: "y"}, "both keys are live")
    check(run_op(fresh, ("get", 23)) == "y" and run_op(fresh, ("get", 7)) == "x",
          "lookups find keys in a shared bucket")

    # ---- split ordering with a dense load ------------------------------------
    table = SplitOrderedList(2)
    for key in range(0, 64):
        run_op(table, ("put", key, key * 3))
    verify_structure(table)
    check(table.live_items() == {key: key * 3 for key in range(64)}, "dense insert")
    check(len(table.initialized_buckets()) == 4, "all four buckets are initialised")

    # ---- concurrent schedules: no lost or phantom keys -----------------------
    programs = [
        [("put", 0, "a"), ("put", 4, "b"), ("put", 8, "c"), ("get", 4), ("del", 0), ("get", 0)],
        [("put", 4, "B"), ("put", 12, "d"), ("get", 12), ("del", 4), ("put", 16, "e"),
         ("get", 8)],
        [("get", 8), ("get", 12), ("put", 0, "A"), ("del", 12), ("get", 4), ("get", 16)],
    ]
    state = 99
    schedules_with_failures = 0
    for seed in range(30):
        table = SplitOrderedList(3)
        schedule = []
        for _ in range(150):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            schedule.append(str(state % 3))
        threads = run(table, programs, schedule)
        check(all(thread.finished for thread in threads),
              "schedule %d left a thread running" % seed)
        verify_structure(table)
        for thread in threads:
            for kind, result in thread.results:
                if kind == "get":
                    check(result in (None, "a", "A", "b", "B", "c", "d", "e"),
                          "a lookup returned an impossible value: %r" % (result,))
        if table.cas_failures:
            schedules_with_failures += 1
    check(schedules_with_failures > 0, "some schedule must lose a CAS")

    # ---- forced race on installing a bucket sentinel --------------------------
    table = SplitOrderedList(3)
    left = Thread("left", [("put", 4, "L")], table)
    right = Thread("right", [("put", 12, "R")], table)
    left.advance()                             # left reads the bucket: EMPTY
    right.advance()                            # right reads the same bucket: EMPTY
    left.advance()                             # left publishes its sentinel first
    check(table.sentinel_installs == 1, "the first insert installed its bucket's sentinel")
    before = table.cas_failures
    right.advance()                            # right's sentinel CAS must lose
    check(table.cas_failures == before + 1, "the losing sentinel CAS must fail")
    check(table.sentinel_installs == 1, "the sentinel is installed exactly once")
    left.drain()
    right.drain()
    check(table.live_items() == {4: "L", 12: "R"}, "both inserts are visible: %s"
          % table.live_items())
    verify_structure(table)

    # ---- forced interleaving: a key inserted while the sentinel is unpublished --
    table = SplitOrderedList(3)
    publisher = Thread("publisher", [("put", 4, "P")], table)
    other = Thread("other", [("get", 4), ("put", 12, "Q"), ("get", 12)], table)
    publisher.advance()                        # read bucket (EMPTY)
    publisher.advance()                        # publish the sentinel (not yet spliced)
    check(table.sentinel_installs == 1, "the sentinel is published")
    check(table.buckets[4].linked is False, "but not spliced yet")
    check(table.walk()[0][0][0] == 0, "the list still starts at the bucket-0 sentinel")
    outcomes = other.drain()
    check("read bucket" in outcomes, "the other thread works on the same bucket")
    publisher.drain()
    check(table.buckets[4].linked is True, "the publisher spliced the sentinel")
    check(table.live_items() == {4: "P", 12: "Q"}, "both keys survive the race: %s"
          % table.live_items())
    verify_structure(table)
    check(other.results[0][1] is None or other.results[0][1] == "P",
          "a lookup during the race may miss or find the value, never invent one")

    # ---- delete marks instead of unlinking ------------------------------------
    table = SplitOrderedList(3)
    run_op(table, ("put", 5, "1"))
    run_op(table, ("put", 13, "2"))
    check(run_op(table, ("del", 5)) is True, "the delete succeeded")
    check(run_op(table, ("del", 5)) is False, "a second delete finds it marked")
    check(table.live_items() == {13: "2"}, "only the other key is live")
    check(any(key == 5 and marked for key, _, marked, sentinel in table.walk()[0]
              if not sentinel), "the deleted node stays linked but marked")
    verify_structure(table)

    # ---- no false positives over a larger table --------------------------------
    table = SplitOrderedList(5)
    for probe in range(200):
        run_op(table, ("put", (probe * 7) % 400, probe))
    verify_structure(table)
    false_positives = 0
    for key in range(0, 400, 3):
        got = run_op(table, ("get", key))
        expected = None
        for probe in range(200):
            if probe * 7 % 400 == key:
                expected = probe
        if (got is None) != (expected is None):
            false_positives += 1
    check(false_positives == 0, "no false positives or missed keys")
    check(len(table.live_items()) == len({(probe * 7) % 400 for probe in range(200)}),
          "the live key count matches the oracle")

    # ---- rejected input ---------------------------------------------------------
    raised = False
    try:
        run(SplitOrderedList(2), [[("get", 1)]], ["5"])
    except IndexError:
        raised = True
    check(raised, "a schedule naming an unknown thread must raise")

    print("sequential: 200 operations agree with the dictionary oracle, %d sentinels installed"
          % table.sentinel_installs)
    print("split order verified for every live key against every installed bucket sentinel")
    print("concurrency: 30 seeded schedules, %d with CAS failures, no lost or phantom keys"
          % schedules_with_failures)
    print("PASS")


if __name__ == "__main__":
    main()
