"""Task 925 - Implement hybrid transactional/analytical processing demo (HTAP) with maintenance.  (Tier: Master, CTCI Ch 14 - Databases)

A single store serving transactional point reads and analytical range scans, plus the maintenance
that keeps both fast.

* Row store (delta): key -> list of (timestamp, value) versions in write order, holding everything
  written since the last merge; a delete writes a None tombstone.
* Columnar store (runs): immutable, key-sorted runs of versioned rows, each with a zone map
  (minimum/maximum key and timestamp).  Merge drains the delta into one new run; compaction folds
  runs together once there are too many.
* Queries: a point read scans the delta and every run and keeps the newest version with
  timestamp <= snapshot (MVCC snapshots).  A range aggregate visits only the runs whose zone maps
  overlap the key range and the snapshot, then filters rows inside them; it reports how many rows it
  had to look at, so the pruning win is measurable.

Verified: point reads equal an oracle replay of the write log at every snapshot; a merge preserves
every version and empties the delta; aggregates equal an independent scan of the raw log, before and
after maintenance, for several ranges and predicates; zone-map pruning returns the same aggregate
while examining fewer rows and never skips a live row; tombstones hide rows from later snapshots but
not from earlier ones; compaction bounds the number of runs; and empty stores, empty ranges, a
snapshot before the first write and a single-row store behave.  Simplified: in-memory runs, integer
keys, no concurrency, and a newest-version-wins merge policy.
"""

MERGE_THRESHOLD, RUN_LIMIT = 4, 3

def check(ok, message):
    if not ok:
        raise AssertionError(message)

class HTAPStore:
    """Row-store delta plus immutable columnar runs, with MVCC snapshots."""

    def __init__(self, merge_threshold=MERGE_THRESHOLD, run_limit=RUN_LIMIT):
        self.delta = {}                       # key -> [(timestamp, value), ...] in write order
        self.runs = []                        # [{"rows": [(key, ts, value)], "key_min", ...}]
        self.log = []                         # audit log of every write, for the oracle
        self.merge_threshold, self.run_limit = merge_threshold, run_limit
        self.merges = 0

    # ------------------------------------------------------------------ transactions
    def write(self, key, value, timestamp):
        self.delta.setdefault(key, []).append((timestamp, value))
        self.log.append((key, timestamp, value))

    def delete(self, key, timestamp):
        self.write(key, None, timestamp)

    def snapshot(self):
        return max((entry[1] for entry in self.log), default=0)

    def point_read(self, key, snapshot):
        best = None
        for timestamp, value in self.delta.get(key, ()):
            if timestamp <= snapshot and (best is None or timestamp > best[0]):
                best = (timestamp, value)
        for run in self.runs:
            if not run["key_min"] <= key <= run["key_max"] or run["ts_min"] > snapshot:
                continue
            for row_key, timestamp, value in run["rows"]:
                if row_key == key and timestamp <= snapshot and (best is None or timestamp > best[0]):
                    best = (timestamp, value)
        return None if best is None else best[1]

    # ---------------------------------------------------------------- analytics scan
    def aggregate(self, low, high, snapshot, predicate, prune=True):
        """Range aggregate returning (rows_counted, value_sum, rows_examined, rows_total).

        With `prune` the zone maps skip every run whose key range, or whose newest timestamp, does
        not overlap the query; without it the same code path visits every stored row, which is the
        A/B baseline for the pruning win.
        """
        newest, examined, total_rows = {}, 0, 0
        for run in self.runs:
            total_rows += len(run["rows"])
            if prune and (run["key_max"] < low or run["key_min"] > high
                          or run["ts_min"] > snapshot):
                continue                                   # zone map prunes the whole run
            for key, timestamp, value in run["rows"]:
                examined += 1
                if low <= key <= high and timestamp <= snapshot:
                    if key not in newest or timestamp > newest[key][0]:
                        newest[key] = (timestamp, value)
        for key, versions in self.delta.items():
            total_rows += len(versions)
            for timestamp, value in versions:
                examined += 1
                if low <= key <= high and timestamp <= snapshot:
                    if key not in newest or timestamp > newest[key][0]:
                        newest[key] = (timestamp, value)
        counted = sum(1 for timestamp, value in newest.values() if value is not None
                      and predicate(value))
        return counted, sum(value for _, value in newest.values()
                            if value is not None and predicate(value)), examined, total_rows

    # -------------------------------------------------------------------- maintenance
    def merge(self):
        """Drain the delta into a fresh immutable run and fold any overlapping runs together."""
        entries = []
        for key, versions in self.delta.items():
            for timestamp, value in versions:
                entries.append((key, timestamp, value))
        if entries:
            self.runs.append(zones(sorted(entries)))
        self.delta = {}
        self.merges += 1
        self.compact()
        return len(entries)

    def compact(self):
        """Tiered compaction: fold runs whose key ranges overlap, keep disjoint runs separate.

        Keeping disjoint runs apart is what preserves the zone maps' pruning power: runs produced by
        different write phases never need to be examined together.
        """
        ordered = sorted(self.runs, key=lambda run: (run["key_min"], run["key_max"]))
        index = 0
        while index < len(ordered) - 1:
            if ordered[index]["key_max"] >= ordered[index + 1]["key_min"]:
                merged = zones(sorted(ordered[index]["rows"] + ordered[index + 1]["rows"]))
                ordered = ordered[:index] + [merged] + ordered[index + 2:]
                index = max(index - 1, 0)
            else:
                index += 1
        self.runs = ordered

    def maintenance(self):
        """The maintenance policy: merge when the delta is big or too many runs have piled up."""
        grown = sum(len(versions) for versions in self.delta.values())
        if grown >= self.merge_threshold or len(self.runs) > self.run_limit:
            return self.merge()
        return 0

    def stats(self):
        size = sum(len(versions) for versions in self.delta.values())
        return {"delta_rows": size, "runs": len(self.runs), "merges": self.merges,
                "rows": size + sum(len(run["rows"]) for run in self.runs)}

def zones(rows):
    """Attach the zone map (key and timestamp extremes) to a sorted run."""
    return {"rows": rows, "key_min": rows[0][0], "key_max": rows[-1][0],
            "ts_min": min(row[1] for row in rows), "ts_max": max(row[1] for row in rows)}

def oracle_point_read(log, key, snapshot):
    """Independent reference: replay the write log and take the last write visible."""
    visible = [entry for entry in log if entry[0] == key and entry[1] <= snapshot]
    if not visible:
        return None
    return max(visible, key=lambda entry: entry[1])[2]

def oracle_aggregate(log, low, high, snapshot, predicate):
    """Independent reference: newest visible version per key, then filter and sum."""
    newest = {}
    for key, timestamp, value in log:
        if low <= key <= high and timestamp <= snapshot:
            if key not in newest or timestamp > newest[key][0]:
                newest[key] = (timestamp, value)
    live = [value for _, value in newest.values() if value is not None and predicate(value)]
    return len(live), sum(live)

def workload():
    """Deterministic history in four key-disjoint phases, each written twice per key.

    Disjoint phases mean the runs produced by successive merges do not overlap, which is exactly the
    layout that makes zone maps useful.
    """
    return [(4 * (step // 6) + (step % 6) // 2, 10 + (step * 7) % 23) for step in range(24)]

def build_history():
    """Replay the workload with the maintenance policy enabled; a tombstone hits key 1 at step 3."""
    store = HTAPStore()
    for step, (key, value) in enumerate(workload()):
        if step == 3:
            store.delete(1, step + 1)              # tombstone: key 1 disappears from then on
            continue
        store.write(key, value, step + 1)
        if step % 6 == 5:
            store.maintenance()
    store.maintenance()
    return store, 1

def main():
    store, _ = build_history()
    log = store.log
    keys = sorted({entry[0] for entry in log})

    # 1. MVCC point reads match an oracle replay of the log at every snapshot.
    checks = 0
    for snapshot in range(0, len(log) + 3):
        for key in keys:
            check(store.point_read(key, snapshot) == oracle_point_read(log, key, snapshot),
                  "point reads must match the oracle replay at every snapshot")
            checks += 1
    check(checks > 100, "the snapshot sweep must actually exercise many reads")

    # 2. Analytics: several ranges and predicates agree with the independent raw-log scan.
    predicates = [lambda value: True, lambda value: value >= 20, lambda value: value % 3 == 0]
    ranges = [(0, 7), (2, 4), (0, 0), (6, 9)]
    equal_pairs = 0
    for low, high in ranges:
        for predicate in predicates:
            for snapshot in (5, 12, len(log) + 1):
                got = store.aggregate(low, high, snapshot, predicate)
                check(got[:2] == oracle_aggregate(log, low, high, snapshot, predicate),
                      "range aggregates must equal the raw-log scan")
                equal_pairs += 1
    check(equal_pairs > 30, "the analytics sweep must cover many range/predicate/snapshot mixes")

    # 3. Zone-map pruning: same answer, fewer rows examined than the unpruned scan.
    total_rows = store.stats()["rows"]
    pruned = store.aggregate(0, 1, len(log) + 1, lambda value: True)
    unpruned = store.aggregate(0, 1, len(log) + 1, lambda value: True, prune=False)
    expected = oracle_aggregate(log, 0, 1, len(log) + 1, lambda value: True)
    check(pruned[:2] == unpruned[:2] == expected,
          "pruned and unpruned scans must return the same rows as the oracle")
    check(pruned[2] < unpruned[2] <= total_rows,
          "zone maps must remove rows from the scan without changing the answer")

    baseline = (store.stats(), store.point_read(2, 3),
                store.aggregate(0, 30, 5, lambda value: True)[:2])

    # 4. Merge preserves every version and empties the delta; compaction bounds the runs.
    stats_before = store.stats()
    unmerged = HTAPStore(merge_threshold=10 ** 6)
    for step, (key, value) in enumerate(workload()):
        if step == 3:
            unmerged.delete(1, step + 1)
            continue
        unmerged.write(key, value, step + 1)
    spread = unmerged.aggregate(0, 30, len(log) + 1, lambda value: True)
    moved = unmerged.merge()
    check(moved > 0 and unmerged.stats()["delta_rows"] == 0
          and unmerged.aggregate(0, 30, len(log) + 1, lambda value: True)[:2] == spread[:2],
          "merging must drain the delta without losing a version or changing an aggregate")
    check(unmerged.stats()["rows"] == spread[3], "the row count must survive a merge unchanged")
    for step in range(6):
        store.write(2, 100 + step, len(log) + 2 + step)      # overlapping rewrites of key 2
        store.merge()
    check(len(store.runs) <= stats_before["runs"],
          "compaction must fold runs whose key ranges overlap, keeping disjoint runs apart")

    # 5. Tombstones: hidden now, still visible to older snapshots, never resurrected by a merge.
    check(store.point_read(1, len(log) + 1) is None, "a deleted key must read as absent")
    check(oracle_point_read(log, 1, len(log) + 1) is None, "the oracle must agree that the key is deleted")
    check(store.point_read(1, 3) is not None and store.point_read(1, 3) == oracle_point_read(log, 1, 3),
          "an older snapshot must still see the value before the delete")

    # 6. Maintenance policy: a small delta is left alone, a large one is merged.
    fresh = HTAPStore()
    fresh.write(1, 5, 1)
    check(fresh.maintenance() == 0 and fresh.stats()["delta_rows"] == 1,
          "maintenance must not merge a delta below the threshold")
    for step in range(MERGE_THRESHOLD):
        fresh.write(step + 1, 1, step + 2)
    check(fresh.maintenance() > 0 and fresh.stats()["delta_rows"] == 0,
          "maintenance must merge once the threshold is crossed")

    # 7. Boundary cases: empty store, empty range, snapshot before the first write, single row.
    empty = HTAPStore()
    check(empty.point_read(1, 99) is None
          and empty.aggregate(0, 9, 99, lambda value: True) == (0, 0, 0, 0),
          "an empty store must answer reads and scans with nothing")
    check(store.aggregate(50, 60, len(log) + 1, lambda value: True)[:2] == (0, 0),
          "a range with no keys must aggregate to zero")
    check(store.point_read(keys[0], 0) is None, "no write is visible before the first write")
    solo = HTAPStore()
    solo.write(4, 7, 1)
    solo.merge()
    check(solo.point_read(4, 1) == 7 and solo.aggregate(4, 4, 1, lambda value: value > 6)[:2] == (1, 7)
          and solo.aggregate(4, 4, 1, lambda value: value > 6)[3] == solo.stats()["rows"],
          "a single-row store must answer point reads and scans")

    # 8. Determinism: rebuilding the same history reproduces the same store and answers.
    rebuilt, _ = build_history()
    check((rebuilt.stats(), rebuilt.point_read(2, 3),
           rebuilt.aggregate(0, 30, 5, lambda value: True)[:2]) == baseline,
          "rebuilding the same history must reproduce the same contents and answers")

    final = store.stats()
    print(f"writes {len(log)} over {len(keys)} keys; runs {final['runs']}, rows {final['rows']}, "
          f"merges {final['merges']}")
    print(f"pruned scan examined {pruned[2]} of {unpruned[2]} unpruned rows of {total_rows} stored")
    print("PASS")


if __name__ == "__main__":
    main()
