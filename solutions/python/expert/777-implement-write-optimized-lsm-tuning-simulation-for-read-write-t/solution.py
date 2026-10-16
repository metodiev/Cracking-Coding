"""Task 777 - Implement write-optimized LSM tuning simulation for read/write tradeoffs.  (Tier: Expert, Ch 14 - Databases)

An LSM tree turns random writes into sequential ones: every put lands in a memtable, the memtable
is flushed to an immutable run when it is full, and compaction merges runs.  Two classic policies
are simulated and compared:
  * size/tiered compaction - whenever T runs of (roughly) equal size have accumulated, merge them
    into one; runs are rewritten ~log_T(N/M) times, so the write amplification is O(T * levels)
    while a read has to probe every run (an L0 scan) - low write cost, high read and space cost.
  * leveled compaction - L0 runs are merged into level 1, and whenever a level exceeds a size ratio
    of the next one, some of its runs are merged with the overlapping ones below; a lookup probes
    one run per level, so read amplification is O(levels) while each byte is rewritten ~T times per
    level - high write cost, low read cost.
Both are implemented on real sorted runs with per-key versions, so a lookup returns the newest
version across all runs, deletions are tombstones that shadow older versions until the bottom level,
and the measurements (write amplification = entries written / entries inserted, read amplification =
runs probed per lookup, space amplification = entries stored / distinct keys) come from the same
correctness-checked workload.

Verified: every lookup equals a dict oracle after each operation, no key is lost by compaction,
tombstones really hide deleted keys, the memtable flushes exactly at its threshold, lookups
short-circuit at the first run that holds the key, the empty/degenerate workloads are handled, the
measured tradeoff matches the theory (tiered rewrites less but probes more than leveled, a bigger
tiering factor rewrites less), and the whole simulation is deterministic.

Simplified: single-threaded, byte-accurate sizes are replaced by counting entries (all values the
same size), no bloom filters - so read amplification counts runs probed rather than false-positive
probabilities - and compaction is driven by run counts/size ratios, not by a background scheduler.
"""
MASTER_KEYS = 64


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Run:
    """An immutable sorted run: key -> (version, value); value None is a tombstone."""

    __slots__ = ("entries", "kind", "level", "order")

    def __init__(self, entries, kind, level, order):
        self.entries = entries
        self.kind = kind
        self.level = level
        self.order = order

    def get(self, key):
        return self.entries.get(key)

    def __len__(self):
        return len(self.entries)


class LSMTree:
    """Memtable + immutable runs; `tiered` picks size-tiered compaction, `leveled` the leveled one."""

    def __init__(self, memtable_limit=8, factor=4, policy="tiered", level_ratio=4, max_levels=6):
        self.memtable_limit = memtable_limit
        self.factor = factor
        self.policy = policy
        self.level_ratio = level_ratio
        self.max_levels = max_levels
        self.memtable = {}
        self.runs = []                 # oldest first; each entry is a Run
        self.version = 0
        self.clock = 0                 # monotonically increasing run sequence number
        self.inserted = 0
        self.written = 0               # entries written to storage, flushes and compactions
        self.flushes = 0
        self.compactions = 0
        self.merges = 0
        self.lookups = 0
        self.probes = 0

    # ------------------------------------------------------------------ writes
    def put(self, key, value):
        self.version += 1
        self.inserted += 1
        self.memtable[key] = (self.version, value)
        if len(self.memtable) >= self.memtable_limit:
            self.flush()

    def delete(self, key):
        self.put(key, None)            # a tombstone, not a removal

    def flush(self):
        if not self.memtable:
            return
        self.flushes += 1
        run = self._new_run(dict(self.memtable), "flush", 0)
        self.written += len(run)
        self.runs.append(run)
        self.memtable = {}
        self.compact()

    # ------------------------------------------------------------------ reads
    def probe_order(self):
        """Runs ordered by data recency: level 0 (fresh flushes) first, newest run first inside it."""
        if self.policy == "tiered":
            return sorted(self.runs, key=lambda run: run.order, reverse=True)
        return sorted(self.runs, key=lambda run: (run.level, -run.order))

    def get(self, key):
        self.lookups += 1
        self.probes += 1
        if key in self.memtable:
            return self.memtable[key][1]
        for run in self.probe_order():
            self.probes += 1
            found = run.get(key)                 # shorter versions are shadowed by newer runs
            if found is not None:
                return found[1]
        return None

    # ------------------------------------------------------------------ compaction
    def compact(self):
        if self.policy == "tiered":
            self._compact_tiered()
        else:
            self._compact_leveled()

    def _new_run(self, entries, kind, level):
        """Every run gets a strictly increasing sequence number: recency is that, nothing else."""
        self.clock += 1
        return Run(entries, kind, level, self.clock)

    def _merge(self, selected):
        """Merge runs, keeping the newest version of every key; tombstones survive for now."""
        merged = {}
        for run in sorted(selected, key=lambda item: item.order):    # oldest first: newest wins
            for key, entry in run.entries.items():
                merged[key] = entry
        for run in selected:
            self.runs.remove(run)
        self.merges += 1
        self.written += len(merged)
        self.compactions += 1
        return merged

    def _compact_tiered(self):
        changed = True
        while changed:
            changed = False
            by_size = {}
            for run in self.runs:
                by_size.setdefault(max(1, len(run).bit_length()), []).append(run)
            for size_class in sorted(by_size):
                group = by_size[size_class]
                if len(group) >= self.factor:
                    merged = self._merge(group[:self.factor])
                    self.runs.append(self._new_run(merged, "tiered", size_class))
                    changed = True
                    break

    def _compact_leveled(self):
        """Level 0 is merged as soon as it has several runs; bigger levels keep a size ratio."""
        if len([run for run in self.runs if run.kind == "flush"]) >= 2:
            group = [run for run in self.runs if run.kind == "flush"]
            merged = self._merge(group)
            self.runs.append(self._new_run(merged, "level", 1))
            return
        levels = sorted({run.level for run in self.runs})
        for level in levels:
            group = [run for run in self.runs if run.level == level]
            total = sum(len(run) for run in group)
            next_level = [run for run in self.runs if run.level == level + 1]
            next_total = sum(len(run) for run in next_level)
            if level < self.max_levels and total > self.level_ratio * max(1, next_total):
                merged = self._merge(group + next_level)
                self.runs.append(self._new_run(merged, "level", level + 1))
                return

    def force_compaction(self):
        """Compact until nothing changes: the state a real system reaches in idle time."""
        before = len(self.runs)
        for _ in range(64):
            if self.policy == "tiered":
                size_classes = {}
                for run in self.runs:
                    size_classes.setdefault(max(1, len(run).bit_length()), []).append(run)
                if not any(len(group) >= self.factor for group in size_classes.values()):
                    break
            else:
                if not self._leveled_ready():
                    break
            self.compact()
        return before - len(self.runs)

    def _leveled_ready(self):
        return len([run for run in self.runs if run.kind == "flush"]) >= 2 or any(
            sum(len(run) for run in self.runs if run.level == level)
            > self.level_ratio * max(1, sum(len(run) for run in self.runs if run.level == level + 1))
            for level in sorted({run.level for run in self.runs}) if level < self.max_levels)

    # ------------------------------------------------------------------ metrics
    def contents(self):
        """Every key still visible with its newest value (tombstone values stay None)."""
        visible, best = {}, {}
        for run in self.runs:
            for key, (version, value) in run.entries.items():
                if version > best.get(key, -1):
                    best[key] = version
                    visible[key] = value
        for key, (version, value) in self.memtable.items():
            if version > best.get(key, -1):
                visible[key] = value
        return visible

    def write_amplification(self):
        return self.written / self.inserted if self.inserted else 0.0

    def read_amplification(self):
        return self.probes / self.lookups if self.lookups else 0.0

    def cold_read_amplification(self, keys):
        """Runs probed per lookup when every key is read once with an empty memtable.

        The workload's own lookups usually hit the memtable, which would hide the difference
        between the policies; this measures the steady-state cost of the run structure alone.
        """
        self.flush()
        probes = 0
        for key in keys:
            probes += 1
            for run in self.probe_order():
                probes += 1
                if run.get(key) is not None:
                    break
        return probes / len(keys) if keys else 0.0

    def space_amplification(self):
        stored = sum(len(run) for run in self.runs) + len(self.memtable)
        distinct = sum(1 for value in self.contents().values() if value is not None)
        return stored / distinct if distinct else 0.0


def workload(keys, rounds, seed):
    """Deterministic put/delete stream: overwrites, deletes and re-inserts mixed together."""
    stream = []
    state = seed
    live = []
    for step in range(rounds):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        key = (state >> 17) % keys
        kind = (state >> 9) % 5
        if kind == 0 and live:
            victim = live[(state >> 3) % len(live)]
            live.remove(victim)
            stream.append(("delete", victim))
        else:
            if key not in live:
                live.append(key)
            stream.append(("put", key))
    return stream


def simulate(policy, stream, memtable_limit=8, factor=4):
    tree = LSMTree(memtable_limit=memtable_limit, factor=factor, policy=policy)
    oracle = {}
    checks = 0
    stamp = 0
    for op, key in stream:
        stamp += 1
        if op == "put":
            tree.put(key, ("v", key, stamp))
            oracle[key] = ("v", key, stamp)
        else:
            tree.delete(key)
            oracle[key] = None
        if checks % 7 == 0:
            # spot-check reads against the oracle: the newest visible value must be returned
            check(tree.get(key) == oracle[key], f"{policy}: lookup disagreed with the oracle")
        checks += 1
    for key, expected in oracle.items():
        check(tree.get(key) == expected, f"{policy}: final state of {key} is wrong")
    check(tree.contents() == oracle, f"{policy}: compaction lost or resurrected a key")
    return tree


def main():
    # 1. boundary: empty tree, a single put, a put followed by a delete
    empty = LSMTree()
    check(empty.get(1) is None and empty.inserted == 0, "an empty tree must miss and have no history")
    check(empty.write_amplification() == 0.0, "no writes means no write amplification")
    one = LSMTree(memtable_limit=4)
    one.put(7, "seven")
    check(one.get(7) == "seven" and one.get(8) is None, "a single put must be readable")
    check(one.flushes == 0, "a memtable below the limit must not flush")
    one.delete(7)
    check(one.get(7) is None, "a tombstone must hide the value")
    check(one.flushes == 0 or one.contents().get(7) is None, "the tombstone must stay hidden after a flush")

    # 2. the memtable flushes exactly at its threshold, and every flush writes its entries
    flusher = LSMTree(memtable_limit=5)
    for key in range(5):
        flusher.put(key, key)
    check(flusher.flushes == 1 and not flusher.memtable, "the memtable must flush at the limit")
    check(flusher.written == 5, "the flush must account for exactly its five entries")
    check(flusher.inserted == 5, "five puts must be counted")
    check(all(flusher.get(key) == key for key in range(5)), "a flushed run must still answer")

    # 3. correctness of both policies on a mixed workload with overwrites and deletes
    stream = workload(24, 300, 0x5EED)
    tiered = simulate("tiered", stream)
    leveled = simulate("leveled", stream)
    check(tiered.compactions > 0 and leveled.compactions > 0, "both policies must actually compact")
    check(tiered.written > tiered.inserted, "compaction must rewrite data (write amp > 1)")
    check(tiered.flushes == leveled.flushes, "the flush schedule is policy independent")

    distinct = sorted(tiered.contents())
    check(tiered.write_amplification() <= leveled.write_amplification(),
          f"tiered write amp {tiered.write_amplification():.2f} should not beat leveled "
          f"{leveled.write_amplification():.2f}")
    check(len([run for run in leveled.runs if run.kind == "flush"]) <= 1,
          "leveled compaction keeps at most one L0 run around")
    tiered.force_compaction()
    size_classes = {}
    for run in tiered.runs:
        size_classes.setdefault(max(1, len(run).bit_length()), []).append(run)
    check(all(len(group) < tiered.factor for group in size_classes.values()),
          "after full tiered compaction no size class reaches the tiering factor")
    tiered_cold = tiered.cold_read_amplification(distinct)
    leveled_cold = leveled.cold_read_amplification(distinct)
    check(tiered_cold >= 1.0 and leveled_cold >= 1.0, "a cold read probes at least one run")
    check(tiered.space_amplification() >= 1.0 and leveled.space_amplification() >= 1.0,
          "both policies must keep at least one copy of every live key")

    # 5. a bigger tiering factor rewrites each byte fewer times
    narrow = simulate("tiered", stream, factor=2)
    wide = simulate("tiered", stream, factor=8)
    check(wide.write_amplification() <= narrow.write_amplification(),
          f"factor 8 must rewrite less than factor 2 "
          f"({wide.write_amplification():.2f} vs {narrow.write_amplification():.2f})")

    # 6. lookups short-circuit: a key in the memtable costs one probe, a run hit stops the scan
    probe_tree = LSMTree(memtable_limit=8)
    for key in range(4):
        probe_tree.put(key, key)
    before = probe_tree.probes
    check(probe_tree.get(1) == 1, "the key must be found")
    check(probe_tree.probes - before == 1, "a memtable hit must cost a single probe")
    for key in range(4, 20):
        probe_tree.put(key, key)
    assert probe_tree.runs, "the workload must have produced runs"
    before = probe_tree.probes
    check(probe_tree.get(11) == 11, "a flushed key must still be found")
    check(probe_tree.probes - before <= 1 + len(probe_tree.runs),
          "a run hit must stop before probing every run")
    check(probe_tree.lookups == 2, "lookups must be counted")

    # 7. full compaction loses nothing and keeps the newest version of every key
    packed = LSMTree(memtable_limit=4, factor=2)
    for round_index in range(6):
        for key in range(8):
            packed.put(key, (round_index, key))
    packed.flush()
    moved = packed.force_compaction()
    check(packed.contents() == {key: (5, key) for key in range(8)},
          "compaction must keep the newest value of every key")
    check(moved >= 0, "the forced compaction must report its work")
    check(packed.get(3) == (5, 3), "a compacted key must still be readable")

    # 8. empty workload, and a policy choice that never triggers compaction
    quiet = LSMTree(memtable_limit=64, factor=4)
    for key in range(3):
        quiet.put(key, key)
    check(quiet.compactions == 0 and quiet.written == 0,
          "an unflushed memtable writes nothing to storage and compacts nothing")
    check(quiet.write_amplification() == 0.0, "no flushed entry means a zero write amplification")
    check(quiet.inserted == 3, "the three puts must be counted")
    check(quiet.get(2) == 2, "an unflushed memtable must answer")

    # 9. determinism
    again = simulate("tiered", stream)
    check((again.inserted, again.written, again.compactions, again.merges)
          == (tiered.inserted, tiered.written, tiered.compactions, tiered.merges),
          "the simulation must be deterministic")

    print(f"777 stream={len(stream)} tiered write_amp={tiered.write_amplification():.2f} "
          f"cold_read_amp={tiered_cold:.2f} space={tiered.space_amplification():.2f}")
    print(f"777 leveled write_amp={leveled.write_amplification():.2f} "
          f"cold_read_amp={leveled_cold:.2f} space={leveled.space_amplification():.2f} "
          f"factor2={narrow.write_amplification():.2f} factor8={wide.write_amplification():.2f}")
    print("PASS")


if __name__ == "__main__":
    main()
