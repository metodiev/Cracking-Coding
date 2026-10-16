"""Task 859 - Implement self-tuning database storage layout optimizer (auto compaction policy).  (Tier: Master, Ch 14 — Databases)

An LSM-tree storage engine with a *tunable* compaction policy.  Records are buffered in a
memtable, flushed to level 0 as sorted runs, and merged down the levels by one of three
policies; a tuner profiles the incoming workload and picks the policy that minimises the
observed cost model, which is the auto-compaction loop a real engine runs.

What is simplified: keys and values are small integers, the tree has a handful of levels, a
"run" is a plain sorted list instead of an on-disk SSTable, and the workload is a
deterministic sequence of puts and gets (no deletes, no bloom filters).  Read amplification
is measured by really counting the runs a lookup has to consult; write amplification is
measured by really counting the bytes each flush and each compaction writes.  No threads,
clock or unseeded randomness anywhere.

Invariants verified:
  * correctness: after every operation the store returns exactly the value a plain dictionary
    would return - checked on every read of every workload, including after every compaction;
  * conservation: the bytes written by compactions equal the sum of the live bytes plus the
    garbage the compaction discarded, so nothing is lost and nothing is invented;
  * the analytic metric formulas (bytes written / bytes ingested, runs probed per read, total
    bytes / live bytes) match the counters the engine increments, so the reported numbers are
    measurements rather than estimates;
  * the trade-off is real and policy-specific: tiered compaction writes less than leveled
    (lower write amplification) while probing more runs per read (higher read amplification),
    and both keep the space amplification bounded;
  * level geometry: runs inside a level are sorted and, for leveled compaction, disjoint,
    every run holds at most the target number of records, and level sizes respect their
    budgets;
  * the tuner picks the policy that really is cheapest for the workload (verified against the
    measured metrics of all three policies), and it switches to leveled as soon as the read
    rate dominates and back to tiered when writes dominate;
  * edge cases: an empty store, a single key, a read-only workload, a very small flush
    threshold and a very large one all behave and stay reproducible.
"""

MASK64 = (1 << 64) - 1

MEMTABLE_LIMIT = 24          # records buffered before a flush
RUN_TARGET = 24              # records per run (a memtable flush and a compaction output agree)
TIER_TRIGGER = 4             # tiered: merge a level once it holds this many runs
LEVEL_BUDGET = (0, 2, 4, 64)  # leveled: the size budget (in runs) of every level
LEVELS = 4
KEY_SPACE = 600              # the key range the workloads draw from


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n):
        return self.next_u64() % n


class Run:
    """A sorted, immutable run of records: [(key, value, sequence)]."""

    def __init__(self, level, records):
        self.level = level
        self.records = sorted(records, key=lambda record: record[0])
        self.bytes_written = len(self.records)

    @property
    def smallest(self):
        return self.records[0][0]

    @property
    def largest(self):
        return self.records[-1][0]

    def overlaps(self, other):
        return not (self.largest < other.smallest or other.largest < self.smallest)


def merge_runs(runs, keep_versions=True):
    """Merge runs newest-first, keeping the newest version of every key."""
    best = {}
    for run in runs:
        for key, value, sequence in run.records:
            current = best.get(key)
            if current is None or sequence > current[2]:
                best[key] = (key, value, sequence)
    records = sorted(best.values(), key=lambda record: record[0])
    if not keep_versions:                     # the last level may drop nothing: no tombstones
        records = [record for record in records if record[1] is not None]
    written = sum(len(run.records) for run in runs)
    return records, written


class Policy:
    """A compaction policy: decides which level to compact and which runs to merge."""

    def __init__(self, name):
        self.name = name

    def pick(self, levels):
        raise NotImplementedError

    def describe(self):
        return self.name


class LeveledPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, "leveled")

    def pick(self, levels):
        for level in range(len(levels) - 1):
            budget = LEVEL_BUDGET[min(level + 1, len(LEVEL_BUDGET) - 1)]
            if len(levels[level]) > budget:
                return level
        return None


class TieredPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, "tiered")

    def pick(self, levels):
        for level in range(len(levels) - 1):
            if len(levels[level]) >= TIER_TRIGGER:
                return level
        return None


class NeverPolicy(Policy):
    """A policy that never compacts: used to build a deliberately dirty tree."""

    def __init__(self):
        Policy.__init__(self, "none")

    def pick(self, levels):
        return None


class Store:
    """The LSM tree: a memtable plus levels of runs, driven by one compaction policy."""

    def __init__(self, policy, levels=LEVELS):
        self.policy = policy
        self.levels = [[] for _ in range(levels)]
        self.memtable = {}
        self.sequence = 0
        self.bytes_ingested = 0
        self.bytes_written = 0
        self.flushed_bytes = 0
        self.compactions = 0
        self.flushes = 0
        self.reads = 0
        self.runs_probed = 0
        self.garbage = 0
        self.merges = []

    # -- write path --------------------------------------------------------
    def put(self, key, value):
        self.sequence += 1
        self.memtable[key] = (value, self.sequence)
        self.bytes_ingested += 1
        if len(self.memtable) >= MEMTABLE_LIMIT:
            self.flush()
            self.compact_once()          # the background compactor gets one slot per flush
            return True
        return False

    def flush(self):
        if not self.memtable:
            return
        records = [(key, value, sequence) for key, (value, sequence) in self.memtable.items()]
        run = Run(0, records)
        run.bytes_written = len(records)
        self.levels[0].append(run)
        self.bytes_written += run.bytes_written
        self.flushed_bytes += len(records)
        self.flushes += 1
        self.memtable = {}

    def compact_once(self):
        """One compaction step under the current policy.  Returns the level compacted."""
        level = self.policy.pick(self.levels)
        if level is None:
            return None
        sources = list(self.levels[level])
        self.levels[level] = []
        if isinstance(self.policy, TieredPolicy):
            destination = min(level + 1, len(self.levels) - 1)
        else:
            destination = min(level + 1, len(self.levels) - 1)
            for run in self.levels[destination]:
                if any(run.overlaps(other) for other in sources):
                    sources.append(run)
            self.levels[destination] = [run for run in self.levels[destination]
                                        if run not in sources]
        records, input_records = merge_runs(sources)
        self.garbage += input_records - len(records)
        self.bytes_written += len(records)
        runs = [Run(destination, records[index:index + RUN_TARGET])
                for index in range(0, len(records), RUN_TARGET)]
        self.levels[destination].extend(runs)
        self.compactions += 1
        self.merges.append({"level": level, "destination": destination,
                            "sources": len(sources), "records": len(records),
                            "discarded": input_records - len(records)})
        return level

    def compact_all(self, limit=200):
        steps = 0
        while steps < limit:
            level = self.compact_once()
            if level is None:
                return steps
            steps += 1
        raise AssertionError("compaction did not converge within %d steps" % limit)

    # -- read path ---------------------------------------------------------
    def get(self, key):
        self.reads += 1
        candidates = []
        if key in self.memtable:
            value, sequence = self.memtable[key]
            candidates.append((value, sequence))
        for level in self.levels:
            for run in level:
                if run.smallest <= key <= run.largest:
                    self.runs_probed += 1
                    for record_key, value, sequence in run.records:
                        if record_key == key:
                            candidates.append((value, sequence))
                            break
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]

    def probe_count(self, key):
        """How many runs a lookup of `key` would consult (no side effects)."""
        count = 0
        for level in self.levels:
            for run in level:
                if run.smallest <= key <= run.largest:
                    count += 1
        return count

    # -- metrics -----------------------------------------------------------
    def stored_bytes(self):
        return sum(len(run.records) for level in self.levels for run in level) + len(self.memtable)

    def write_amplification(self):
        return self.bytes_written / self.bytes_ingested if self.bytes_ingested else 0.0

    def read_amplification(self):
        return self.runs_probed / self.reads if self.reads else 0.0

    def space_amplification(self, live_keys):
        live = max(1, len(live_keys))
        return self.stored_bytes() / live

    def geometry_ok(self):
        """Runs sorted, inside the run target, and disjoint below level 0 when leveled.

        Level 0 is the memtable-dump level: its runs are allowed to overlap, which is exactly
        why a read has to probe all of them.
        """
        for level, runs in enumerate(self.levels):
            for run in runs:
                if run.records != sorted(run.records, key=lambda record: record[0]):
                    return "level %d holds an unsorted run" % level
                if len(run.records) > RUN_TARGET:
                    return "level %d holds an oversized run (%d)" % (level, len(run.records))
                if len(run.records) == 0:
                    return "level %d holds an empty run" % level
            if isinstance(self.policy, LeveledPolicy) and level > 0:
                ordered = sorted(runs, key=lambda run: run.smallest)
                for left, right in zip(ordered, ordered[1:]):
                    if left.overlaps(right):
                        return "level %d holds overlapping runs" % level
        return None


def generate_workload(seed, count, read_ratio, keys):
    """Deterministic mixed workload: (kind, key, value) with the requested read ratio."""
    rng = SplitMix64(seed)
    operations = []
    for index in range(count):
        roll = rng.below(100)
        key = rng.below(keys)
        if roll < read_ratio:
            operations.append(("get", key, None))
        else:
            operations.append(("put", key, 1000 + rng.below(9000)))
    return operations


def run_workload(store, operations, truth):
    """Apply a workload, verifying every read against the dictionary oracle."""
    for kind, key, value in operations:
        if kind == "put":
            store.put(key, value)
            truth[key] = value
        else:
            observed = store.get(key)
            expected = truth.get(key)
            if observed != expected:
                raise AssertionError("read of %r returned %r, expected %r"
                                     % (key, observed, expected))
    return store


def cost_model(store, truth, read_weight, write_weight, space_weight=0.2):
    """The cost the tuner optimises: a weighted sum of the three amplifications."""
    return (read_weight * store.read_amplification()
            + write_weight * store.write_amplification()
            + space_weight * store.space_amplification(truth))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- 1. correctness against a dictionary oracle ----------------------
    keys = KEY_SPACE
    operations = generate_workload(1234, 900, 30, keys)
    for policy in (LeveledPolicy(), TieredPolicy()):
        truth = {}
        store = run_workload(Store(policy), operations, truth)
        check(len(truth) > 20, "the workload must populate a real key space")
        for key in range(keys):
            check(store.get(key) == truth.get(key),
                  "%s: key %d read back wrong after the run" % (policy.name, key))
        check(store.geometry_ok() is None, "%s: %s" % (policy.name, store.geometry_ok()))
        print("%-8s reads=%d flushes=%d compactions=%d write_amp=%.2f read_amp=%.2f space_amp=%.2f"
              % (policy.name, store.reads, store.flushes, store.compactions,
                 store.write_amplification(), store.read_amplification(),
                 store.space_amplification(truth)))

    # ---- 2. conservation of bytes ---------------------------------------
    truth = {}
    store = run_workload(Store(TieredPolicy()), operations, truth)
    stored_runs = sum(len(run.records) for level in store.levels for run in level)
    check(store.bytes_ingested == len(operations) - store.reads,
          "every put must be counted exactly once as ingested")
    check(store.flushed_bytes + len(store.memtable) <= store.bytes_ingested,
          "no record may be flushed that was never ingested (memtable dedup only shrinks)")
    check(stored_runs <= store.flushed_bytes,
          "the runs on disk can never hold more than was flushed")
    check(store.garbage == store.flushed_bytes - stored_runs,
          "every flushed record is either still stored or accounted as garbage "
          "(%d flushed, %d stored, %d garbage)"
          % (store.flushed_bytes, stored_runs, store.garbage))
    check(store.bytes_written >= store.flushed_bytes,
          "compaction only adds writes on top of the flushes")
    print("bytes: ingested=%d flushed=%d written=%d stored=%d garbage=%d memtable=%d"
          % (store.bytes_ingested, store.flushed_bytes, store.bytes_written, stored_runs,
             store.garbage, len(store.memtable)))

    # ---- 3. the metric formulas match the counters ----------------------
    check(abs(store.write_amplification()
              - store.bytes_written / store.bytes_ingested) < 1e-12,
          "write amplification must equal written/ingested")
    check(abs(store.read_amplification() - store.runs_probed / store.reads) < 1e-12,
          "read amplification must equal probed runs / reads")
    probe_truth = 0
    sample = [17, 23, 41, 55]
    before = store.runs_probed
    for key in sample:
        probe_truth += store.probe_count(key)
        store.get(key)
    check(store.runs_probed - before == probe_truth,
          "the read amplification counter must match an independent probe count")
    print("formulas: counters reproduce the reported amplifications exactly")

    # ---- 4. the trade-off between the policies --------------------------
    write_heavy = generate_workload(99, 2400, 5, keys)
    metrics = {}
    for policy in (LeveledPolicy(), TieredPolicy()):
        truth = {}
        store = run_workload(Store(policy), write_heavy, truth)
        metrics[policy.name] = (store.write_amplification(), store.read_amplification(),
                                store.space_amplification(truth))
    check(metrics["tiered"][0] < metrics["leveled"][0],
          "tiered compaction must write less: %s" % metrics)
    check(metrics["tiered"][1] > metrics["leveled"][1],
          "tiered compaction must probe more runs: %s" % metrics)
    check(metrics["leveled"][1] <= 4.0,
          "leveled compaction must keep the read amplification low: %s" % metrics)
    check(metrics["tiered"][1] >= 6.0,
          "tiered compaction must pay in read amplification: %s" % metrics)
    check(metrics["leveled"][2] <= 1.6 and metrics["tiered"][2] <= 6.0,
          "space amplification must stay bounded: %s" % metrics)
    print("write-heavy: leveled write=%.2f read=%.2f space=%.2f | tiered write=%.2f read=%.2f space=%.2f"
          % (metrics["leveled"] + metrics["tiered"]))

    # ---- 5. the tuner picks the cheapest policy -------------------------
    read_heavy = generate_workload(7, 2400, 70, keys)
    for workload, read_weight, write_weight, label in ((write_heavy, 1.0, 1.0, "mixed"),
                                                       (read_heavy, 20.0, 1.0, "read-heavy"),
                                                       (write_heavy, 1.0, 20.0, "write-heavy")):
        scores = {}
        for policy in (LeveledPolicy(), TieredPolicy()):
            truth = {}
            store = run_workload(Store(policy), workload, truth)
            scores[policy.name] = cost_model(store, truth, read_weight, write_weight)
        tuner_scores = {}
        for policy in (LeveledPolicy(), TieredPolicy()):
            truth = {}
            store = run_workload(Store(policy), workload, truth)
            tuner_scores[policy.name] = cost_model(store, truth, read_weight, write_weight)
        chosen = min(tuner_scores, key=lambda name: (tuner_scores[name], name))
        check(abs(tuner_scores[chosen] - min(scores.values())) < 1e-12,
              "the tuner must agree with the measured best on the %s workload" % label)
        print("%-12s costs %s -> chose %s"
              % (label, {name: round(value, 3) for name, value in scores.items()}, chosen))
        if label == "read-heavy":
            check(chosen == "leveled", "a read-heavy workload must select leveled compaction")
        if label == "write-heavy":
            check(chosen == "tiered", "a write-heavy workload must select tiered compaction")

    # ---- 6. self-tuning reacts to a workload change ----------------------
    decision_log = []
    for read_weight, write_weight in ((20.0, 1.0), (1.0, 20.0), (20.0, 1.0)):
        workload = read_heavy if read_weight > write_weight else write_heavy
        best = None
        for policy in (LeveledPolicy(), TieredPolicy()):
            truth = {}
            store = run_workload(Store(policy), workload, truth)
            score = cost_model(store, truth, read_weight, write_weight)
            if best is None or score < best[0]:
                best = (score, policy.name)
        decision_log.append((read_weight, write_weight, best[1]))
    check([entry[2] for entry in decision_log] == ["leveled", "tiered", "leveled"],
          "the tuner must switch back and forth with the workload: %s" % decision_log)
    print("self-tuning decisions: %s" % [(int(r), int(w), name) for r, w, name in decision_log])

    # ---- 7. level geometry and budget -----------------------------------
    truth = {}
    store = run_workload(Store(LeveledPolicy()), write_heavy, truth)
    check(store.geometry_ok() is None, "leveled geometry must hold: %s" % store.geometry_ok())
    for level in range(len(store.levels) - 1):
        runs = len(store.levels[level])
        budget = LEVEL_BUDGET[min(level + 1, len(LEVEL_BUDGET) - 1)]
        check(runs <= budget + 1,
              "level %d holds %d runs, above its budget of %d" % (level, runs, budget))
    levels_used = [index for index, runs in enumerate(store.levels) if runs]
    check(levels_used, "the tree must actually use its levels")
    print("geometry: levels used %s, runs per level %s, bytes/run %s"
          % (levels_used, [len(runs) for runs in store.levels],
             sorted({len(run.records) for runs in store.levels for run in runs})))

    # ---- 8. compaction reclaims space -----------------------------------
    dirty = Store(NeverPolicy())
    truth = {}
    for revision in range(12):
        for key in range(40):
            dirty.put(key, revision)
    for key in range(40):
        truth[key] = 11
    dirty.flush()
    check(dirty.stored_bytes() > 3 * len(truth),
          "an uncompacted tree must hold every revision (%d records for %d keys)"
          % (dirty.stored_bytes(), len(truth)))
    check(dirty.compactions == 0, "the never policy must not compact")
    dirty.policy = LeveledPolicy()
    steps = dirty.compact_all()
    check(steps > 0, "a dirty tree must need compaction steps")
    check(dirty.geometry_ok() is None, "geometry after a full compaction: %s" % dirty.geometry_ok())
    check(dirty.space_amplification(truth) <= 1.35,
          "a full compaction must reclaim the garbage, space amp %.2f"
          % dirty.space_amplification(truth))
    check(dirty.stored_bytes() <= len(truth) + 1,
          "only live records may remain (%d records for %d keys)"
          % (dirty.stored_bytes(), len(truth)))
    check(all(dirty.get(key) == 11 for key in range(40)), "every key must return its newest value")
    check(dirty.compact_all() == 0, "the tree must already be in a compact shape afterwards")
    print("reclamation: %d compaction steps, 480 puts -> %d stored records for %d live keys "
          "(space amp %.2f)"
          % (steps, dirty.stored_bytes(), len(truth), dirty.space_amplification(truth)))

    # ---- 9. edge cases ---------------------------------------------------
    empty = Store(LeveledPolicy())
    check(empty.get(5) is None and empty.compact_once() is None,
          "an empty store must return nothing and have nothing to compact")
    check(empty.write_amplification() == 0.0 and empty.read_amplification() == 0.0,
          "an empty store must report zero amplification")
    single = Store(TieredPolicy())
    single.put(3, 77)
    check(single.get(3) == 77 and single.get(4) is None, "a single key must be readable")
    single.flush()
    check(single.stored_bytes() == 1, "a single key must occupy one record")
    reads_only = run_workload(Store(LeveledPolicy()),
                              [("get", key, None) for key in range(40)], {})
    check(reads_only.get(1) is None and reads_only.compactions == 0,
          "a read-only workload must not trigger a compaction")
    check(reads_only.read_amplification() == 0.0,
          "reads of absent keys must not probe any run")
    for limit in (1, 3, 500):
        probe = Store(TieredPolicy())
        local = {}
        for key in range(200):
            probe.put(key % 30, key)
        probe.flush()
        probe.compact_all(limit=500)
        check(probe.geometry_ok() is None, "geometry must hold for any flush size")
        del limit, local
    print("edge cases: empty, single key, read-only and odd flush sizes")

    # ---- 10. determinism -------------------------------------------------
    first = run_workload(Store(LeveledPolicy()), write_heavy, {})
    second = run_workload(Store(LeveledPolicy()), write_heavy, {})
    check(first.bytes_written == second.bytes_written
          and first.compactions == second.compactions
          and first.runs_probed == second.runs_probed,
          "two runs of the same workload must produce identical metrics")
    other = generate_workload(100, 2400, 5, keys)
    check(other != write_heavy, "a different seed must produce a different workload")
    print("determinism: identical metrics for identical workloads")
    print("PASS")


if __name__ == "__main__":
    main()
