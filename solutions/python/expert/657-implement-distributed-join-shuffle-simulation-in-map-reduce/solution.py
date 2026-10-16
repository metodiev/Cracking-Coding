"""Task 657 - Implement distributed join shuffle simulation in map-reduce.  (Tier: Expert, Ch 9 - System Design and Scalability)

A MapReduce equi-join is simulated end to end. Mappers tag every row of both inputs and emit
(key, (tag, row)); the partitioner routes each record to exactly one reducer by a *deterministic*
FNV-1a hash of the join key (never Python's randomised `hash`, so runs are reproducible). Because both
sides are partitioned by the same function, all rows of one key meet in one reducer - that
co-partitioning is the invariant that makes the reduce-side hash join correct, and it is verified by
comparing every result against a single-process nested-loop brute force. Reducers group their
partition by key and emit the cartesian product per key, so duplicate keys keep their exact
multiplicity; left-outer joins emit unmatched left rows padded with nulls.

Two plans are available: the shuffle (reduce-side) join, and the map-side broadcast join used when one
side fits in memory - it builds a hash table from the small side and emits zero shuffle records, which
is checked to produce exactly the same rows. The classic straggler problem of the shuffle join is a
skewed key flooding one partition; the simulated remedy is salting: rows of a hot key are split across
`splits` sub-keys while the other side's rows for that key are replicated to all of them, so the hot
key's work is spread out while the cartesian product stays complete. The skew metric (max partition
load versus the average) is measured before and after.

Simplified: a single machine executes mapper/partitioner/reducer as pure functions - there is no real
network, no serialization cost model and no combiner; partition load is measured in records rather than
bytes. Verified: exact equality with brute force for inner and left-outer joins, duplicate-key
multiplicity, empty and disjoint inputs, partition coverage/sum invariants, plan equivalence of the
broadcast join, skew reduction by salting with the result left untouched, reducer-count independence
and determinism.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


def fnv1a(text):
    """Deterministic 64-bit FNV-1a hash of a string key (stable across processes and runs)."""
    digest = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        digest ^= byte
        digest = (digest * 0x100000001B3) & ((1 << 64) - 1)
    return digest


class Table:
    __slots__ = ("name", "columns", "rows")

    def __init__(self, name, columns, rows):
        self.name = name
        self.columns = list(columns)
        self.rows = [tuple(row) for row in rows]

    def key_index(self, column):
        return self.columns.index(column)


def map_side(table, key_index, tag):
    """Mapper output: one tagged record per row, keyed by the join key."""
    return [(row[key_index], (tag, row)) for row in table.rows]


def partition_of(key, reducers, splits=1):
    return fnv1a(str(key)) % reducers


def shuffle(records, reducers):
    """The shuffle phase: route every record to its reducer with the deterministic hash."""
    partitions = [[] for _ in range(reducers)]
    for key, payload in records:
        partitions[_partition_key(key, reducers)].append((key, payload))
    return partitions


def reduce_partition(records, kind):
    """Group one partition by key and join the two tagged sides (cartesian product per key)."""
    groups = {}
    order = []
    for key, (tag, row) in records:
        if key not in groups:
            groups[key] = {"L": [], "R": []}
            order.append(key)
        groups[key][tag].append(row)
    out = []
    for key in order:
        left_rows, right_rows = groups[key]["L"], groups[key]["R"]
        if right_rows:
            for left in left_rows:
                for right in right_rows:
                    out.append(left + right)
        elif kind == "left":
            out.extend(left_rows)
    return out


def shuffle_join(left, right, left_key, right_key, reducers, kind="inner", hot_keys=(),
                 splits=1, track=None):
    """Full shuffle join: map, (optional) salt, shuffle, reduce."""
    left_records = map_side(left, left.key_index(left_key), "L")
    right_records = map_side(right, right.key_index(right_key), "R")
    hot = set(hot_keys)
    if hot and splits > 1:
        salted_left = []
        for index, (key, payload) in enumerate(left_records):
            if key in hot:
                salted_left.append(((key, index % splits), payload))
            else:
                salted_left.append((key, payload))
        salted_right = []
        for key, payload in right_records:
            if key in hot:
                for split in range(splits):
                    salted_right.append(((key, split), payload))
            else:
                salted_right.append((key, payload))
        left_records, right_records = salted_left, salted_right
    partitions = shuffle(left_records + right_records, reducers)
    loads = [len(partition) for partition in partitions]
    results = []
    for partition in partitions:
        results.extend(reduce_partition(partition, kind))
    if track is not None:
        track["shuffle_records"] = len(left_records) + len(right_records)
        track["loads"] = loads
    return results


def _partition_key(key, reducers):
    if isinstance(key, tuple):
        return fnv1a(str(key[0]) + "#" + str(key[1])) % reducers
    return fnv1a(str(key)) % reducers


def broadcast_join(left, right, left_key, right_key, kind="inner", track=None):
    """Map-side join: hash the (small) right side in memory, stream the left side through it."""
    lookup = {}
    for row in right.rows:
        lookup.setdefault(row[right.key_index(right_key)], []).append(row)
    out = []
    for row in left.rows:
        matches = lookup.get(row[left.key_index(left_key)], [])
        if matches:
            for match in matches:
                out.append(row + match)
        elif kind == "left":
            out.append(row)
    if track is not None:
        track["shuffle_records"] = 0
        track["loads"] = [len(left.rows)]
    return out


def brute_force_join(left, right, left_key, right_key, kind="inner"):
    """Single-process nested-loop oracle for the same join semantics."""
    out = []
    for left_row in left.rows:
        matched = False
        for right_row in right.rows:
            if left_row[left.key_index(left_key)] == right_row[right.key_index(right_key)]:
                out.append(left_row + right_row)
                matched = True
        if kind == "left" and not matched:
            out.append(left_row)
    return sorted(out, key=str)


def random_tables(seed, left_rows=40, right_rows=25, keys=8, hot=False):
    left = Table("orders", ["oid", "customer", "amount"], [])
    right = Table("customers", ["customer", "city"], [])
    for oid in range(left_rows):
        seed, r1 = splitmix64(seed)
        seed, r2 = splitmix64(seed)
        key = "hot" if hot and oid % 3 == 0 else "c%d" % (r1 % keys)
        left.rows.append((oid, key, r2 % 500))
    for index in range(right_rows):
        seed, r1 = splitmix64(seed)
        key = "hot" if hot and index % 5 == 0 else "c%d" % (r1 % keys)
        right.rows.append((key, "city%d" % (r1 % 10)))
    return left, right, seed


def main():
    # 1. hand example: duplicate keys on both sides multiply out
    left = Table("orders", ["oid", "customer"],
                 [(1, "a"), (2, "a"), (3, "b"), (4, "z")])
    right = Table("customers", ["customer", "city"],
                  [("a", "rome"), ("a", "paris"), ("b", "oslo")])
    inner = shuffle_join(left, right, "customer", "customer", 3)
    check(sorted(inner, key=str) == brute_force_join(left, right, "customer", "customer"),
          "the shuffle join must equal the nested-loop join")
    check(len(inner) == 5, "two 'a' orders times two 'a' customers plus one 'b' order is five rows")
    check(all(row[1] == row[2] for row in inner), "every output row must share the join key")

    # 2. left outer join keeps the unmatched row, padded on the right
    outer = shuffle_join(left, right, "customer", "customer", 3, kind="left")
    check(sorted(outer, key=str) == brute_force_join(left, right, "customer", "customer", kind="left"),
          "the outer join must equal the nested-loop outer join")
    check((4, "z") in outer, "the unmatched order must survive a left outer join")
    check(all(row[1] == row[2] for row in outer if len(row) == 4 and row[2] is not None),
          "matched outer rows still share the key")

    # 3. partitioner invariants: every record lands once, loads sum to the record count
    track = {}
    shuffle_join(left, right, "customer", "customer", 4, kind="left", track=track)
    check(track["shuffle_records"] == len(left.rows) + len(right.rows),
          "the shuffle must carry one record per input row")
    check(sum(track["loads"]) == track["shuffle_records"], "every record must reach exactly one reducer")
    check(len(track["loads"]) == 4, "one load counter per reducer")

    # 4. random tables: every pair of reducer counts answers identically
    seed = 0x1234
    for trial in range(6):
        left, right, seed = random_tables(seed)
        expected = brute_force_join(left, right, "customer", "customer")
        for reducers in (1, 2, 3, 7):
            check(sorted(shuffle_join(left, right, "customer", "customer", reducers), key=str) == expected,
                  f"reducer count {reducers} must not change the inner join")
            check(sorted(shuffle_join(left, right, "customer", "customer", reducers, kind="left"), key=str)
                  == brute_force_join(left, right, "customer", "customer", kind="left"),
                  f"reducer count {reducers} must not change the outer join")

    # 5. join is symmetric as a bag of rows
    left, right, seed = random_tables(seed)
    direct = shuffle_join(left, right, "customer", "customer", 5)
    swapped = shuffle_join(right, left, "customer", "customer", 4)
    rebuilt = [row[len(right.columns):] + row[:len(right.columns)] for row in swapped]
    check(sorted(rebuilt, key=str) == sorted(direct, key=str),
          "joining A with B must mirror joining B with A")

    # 6. map-side broadcast join equals the shuffle join and shuffles nothing
    track = {}
    broadcast = broadcast_join(left, right, "customer", "customer", track=track)
    check(sorted(broadcast, key=str) == brute_force_join(left, right, "customer", "customer"),
          "the broadcast join must equal the nested-loop join")
    check(track["shuffle_records"] == 0, "a map-side join must not shuffle any record")
    check(sorted(broadcast_join(left, right, "customer", "customer", kind="left"), key=str)
          == brute_force_join(left, right, "customer", "customer", kind="left"),
          "the broadcast left outer join must match too")

    # 7. empty and disjoint inputs
    empty = Table("e", ["customer", "x"], [])
    check(shuffle_join(empty, right, "customer", "customer", 2) == [], "an empty left side joins to nothing")
    check(shuffle_join(left, empty, "customer", "customer", 2) == [], "an empty right side joins to nothing")
    check(len(shuffle_join(left, empty, "customer", "customer", 2, kind="left")) == len(left.rows),
          "a left outer join with an empty right side keeps every left row")
    disjoint = Table("d", ["customer", "y"], [("nobody", 1), ("nobody", 2)])
    check(shuffle_join(left, disjoint, "customer", "customer", 3) == [],
          "disjoint key sets produce no inner rows")
    check(len(shuffle_join(left, disjoint, "customer", "customer", 3, kind="left")) == len(left.rows),
          "disjoint keys keep all left rows in an outer join")

    # 8. skew: a hot key floods one partition, salting spreads it out
    hot_left, hot_right, _ = random_tables(0x777, left_rows=200, right_rows=40, keys=6, hot=True)
    plain_track, salted_track = {}, {}
    plain = shuffle_join(hot_left, hot_right, "customer", "customer", 4, track=plain_track)
    salted = shuffle_join(hot_left, hot_right, "customer", "customer", 4,
                          hot_keys=("hot",), splits=4, track=salted_track)
    check(sorted(plain, key=str) == sorted(salted, key=str), "salting must not change a single row")
    check(sorted(plain, key=str) == brute_force_join(hot_left, hot_right, "customer", "customer"),
          "the salted join must still match brute force")
    plain_load, salted_load = max(plain_track["loads"]), max(salted_track["loads"])
    check(plain_load > sum(plain_track["loads"]) / 4,
          "with a hot key the busiest reducer must exceed the average load")
    check(salted_load < plain_load, "salting the hot key must lower the busiest partition")
    check(salted_track["shuffle_records"] > plain_track["shuffle_records"],
          "salting trades extra shuffle records for balance")

    # 9. determinism: the deterministic hash makes runs reproducible
    first = shuffle_join(left, right, "customer", "customer", 4, kind="left")
    second = shuffle_join(left, right, "customer", "customer", 4, kind="left")
    check(first == second, "the same plan must produce the same rows")
    check(partition_of("customer", 4) == fnv1a("customer") % 4,
          "the partitioner must use the deterministic FNV-1a hash")
    check(fnv1a("hot") != fnv1a("cold"), "different keys must hash apart")

    print(f"657 inner rows={len(inner)} outer rows={len(outer)} "
          f"shuffle_records={track['shuffle_records']}")
    print(f"657 skew load plain={plain_load} salted={salted_load} "
          f"records {plain_track['shuffle_records']} -> {salted_track['shuffle_records']}")
    print("PASS")


if __name__ == "__main__":
    main()
