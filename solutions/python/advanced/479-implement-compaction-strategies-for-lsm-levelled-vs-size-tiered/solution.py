"""Task 479 - Implement compaction strategies for LSM (levelled vs size-tiered). (Tier: Advanced, CTCI Ch 14 — Databases)

Two compaction strategies over one fixed workload (128 writes, 256-key space, flush every 8
ops = 16 sectors of <= 8 entries):
  * size-tiered: whenever four runs sit within a factor of two of each other's size they are
    merged (classic size-tiered compaction);
  * levelled (binary fanout 2): two runs in a level are merged into the next level, a binary
    counter schedule.
Both merge by key, newest value wins, so every engine must equal the last-write oracle after
each flush. Documented outcome for this write-heavy parameterisation: levelled pays more
write amplification (600 vs 357 entry writes, 15 vs 5 compactions) while size-tiered keeps
more runs alive (peak 6 vs 4), i.e. higher read amplification; both keep all 107 live keys and
collapse to a single run of 107. All counts are hand-checked from the deterministic merge logs.
"""

KEYS = 256
OPS = 128
FLUSH_EVERY = 8
MERGE_FANOUT = 4
FACTOR = 2


class SizeTieredEngine:
    def __init__(self):
        self.runs = []                 # chronological; each run is {key: value}
        self.total_written = 0         # entry writes: flush + merge outputs
        self.merge_count = 0
        self.peak_runs = 0
        self.peak_entries = 0

    def flush(self, memtable):
        self.runs.append(dict(memtable))
        self.total_written += len(memtable)
        self._compact()
        self.peak_runs = max(self.peak_runs, len(self.runs))
        self.peak_entries = max(self.peak_entries, sum(len(run) for run in self.runs))

    def _compact(self):
        while True:
            sizes = sorted((len(run), idx) for idx, run in enumerate(self.runs))
            target = None
            for i in range(len(sizes) - 3):
                if sizes[i + 3][0] <= FACTOR * sizes[i][0]:
                    target = sorted(idx for _size, idx in sizes[i:i + 4])
                    break
            if target is None:
                return
            merged = {}
            for i in target:           # chronological order: newer runs overwrite older
                merged.update(self.runs[i])
            self.total_written += len(merged)
            self.merge_count += 1
            new_runs = []
            inserted = False
            for i, run in enumerate(self.runs):
                if i in target:
                    if not inserted:
                        new_runs.append(merged)
                        inserted = True
                else:
                    new_runs.append(run)
            self.runs = new_runs

    def materialize(self):
        out = {}
        for run in self.runs:
            out.update(run)
        return out


class LevelledEngine:
    def __init__(self):
        self.levels = []               # levels[i] is a chronological list of runs
        self.total_written = 0
        self.merge_count = 0
        self.peak_runs = 0
        self.peak_entries = 0

    def _ensure_level(self, i):
        while len(self.levels) <= i:
            self.levels.append([])

    def flush(self, memtable):
        self._ensure_level(0)
        self.levels[0].append(dict(memtable))
        self.total_written += len(memtable)
        self._compact()
        runs = [run for level in self.levels for run in level]
        self.peak_runs = max(self.peak_runs, len(runs))
        self.peak_entries = max(self.peak_entries, sum(len(run) for run in runs))

    def _compact(self):
        while True:
            for i, level in enumerate(self.levels):
                if len(level) >= 2:
                    merged = {}
                    merged.update(level[0])        # older
                    merged.update(level[1])        # newer wins
                    self.total_written += len(merged)
                    self.merge_count += 1
                    del level[0:2]
                    self._ensure_level(i + 1)
                    self.levels[i + 1].append(merged)
                    break
            else:
                return

    def materialize(self):
        out = {}
        for level in reversed(self.levels):        # higher levels hold older data
            for run in level:
                out.update(run)
        return out


def lcg(seed):
    return (seed * 75 + 74) % 65537


def gen_writes(count, seed, keyspace):
    ops = []
    for _ in range(count):
        seed = lcg(seed)
        key = seed % keyspace
        seed = lcg(seed)
        value = seed % 1000
        ops.append((key, value))
    return ops


def main():
    ops = gen_writes(OPS, 24680, KEYS)
    oracle = {}
    st = SizeTieredEngine()
    lv = LevelledEngine()
    for start in range(0, OPS, FLUSH_EVERY):
        memtable = {}
        for key, value in ops[start:start + FLUSH_EVERY]:
            memtable[key] = value
            oracle[key] = value
        st.flush(memtable)
        lv.flush(memtable)
        assert st.materialize() == oracle, ("size-tiered lost a key", start)
        assert lv.materialize() == oracle, ("levelled lost a key", start)

    assert len(oracle) == 107
    assert st.materialize() == oracle and lv.materialize() == oracle

    # Size-tiered: merges the four <=8-entry runs at flushes 4, 8, 12, 16 and then the four
    # ~30-entry runs (the last flush performs two merges).
    assert st.total_written == 357
    assert st.merge_count == 5
    assert st.peak_runs == 6
    assert st.peak_entries == 115
    assert len(st.runs) == 1 and len(st.runs[0]) == 107

    # Levelled: binary-counter schedule, 8 + 4 + 2 + 1 compactions.
    assert lv.total_written == 600
    assert lv.merge_count == 15
    assert lv.peak_runs == 4
    assert lv.peak_entries == 112
    assert len(lv.levels) == 5
    for level in lv.levels[:-1]:
        assert level == []
    assert len(lv.levels[-1]) == 1 and len(lv.levels[-1][0]) == 107

    # Documented qualitative outcomes for this parameterisation.
    assert lv.total_written > st.total_written, "levelled rewrites more entries (write amplification)"
    assert lv.merge_count > st.merge_count, "levelled compacts more often, in smaller steps"
    assert st.peak_runs > lv.peak_runs, "size-tiered keeps more runs alive (read amplification)"
    assert st.peak_entries >= lv.peak_entries, "size-tiered temporarily stores more copies"


if __name__ == "__main__":
    main()
    print("PASS")
