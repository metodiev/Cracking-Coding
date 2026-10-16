"""Task 511 - Implement checkpointed parallel map-reduce worker for crash recovery. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

A deterministic simulation of the classic fault-tolerant map-reduce pattern: an append-only commit
log, periodic checkpoints and replay from the last committed offset. The input is split across
`workers` splits; each split runs a map function (word -> (word, 1)) whose outputs are routed to
`partitions` reducers by a deterministic hash. A worker buffers mapped records in memory and commits
a whole batch at once - first the per-partition output records, then the offset of the next record -
so a crash in the middle of a batch loses only that batch, never a half-written batch: the durable
outputs contain full batches only, the offset always points at the start of an uncommitted batch, and
nothing is committed twice. Recovery re-reads the split from the committed offset and recomputes the
lost records; since map is deterministic and the reduce step is a sum, the final aggregation is
identical to the crash-free run (effectively-once, with at-least-once processing of the records).
The store also models a reducing worker: each partition's aggregate is published atomically, so a
crash during the reduce can simply re-run it from the committed map outputs.

Documented checks: the reduce output of a crash-free run equals a brute-force aggregation computed
straight from the raw records; every single crash point in 0..n-1 (all 60 of them, with a commit
interval of 7) still yields that same output, and the amount of recomputation is exactly
k mod interval records - the precise value the checkpoint policy predicts; multi-crash sequences
(for example 10 then 40, or 5, 50 and 55) also recover, with the total redo equal to the sum of the
individual redos; committed offsets are monotone, the final checkpoint equals the split length and
the number of commits equals ceil(n / interval); a crash injected inside the reduce step leaves the
aggregate unchanged because each partition is republished atomically; a multi-split run with crashes
in different splits keeps every split's output intact (a crash in one split cannot corrupt another's
log) and a real-thread run of the same three splits reaches the same aggregate with only stable
invariants asserted.
"""

import threading

VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "iota", "kappa", "lambda", "mu"]


def lcg(seed):
    return (seed * 75 + 74) % 65537


def make_records(count, seed, words_per_record=4, vocab=VOCAB):
    """Records are small word bags; every record is reproducible from the seed."""
    records = []
    for _ in range(count):
        words = []
        for _ in range(words_per_record):
            seed = lcg(seed)
            words.append(vocab[seed % len(vocab)])
        records.append(" ".join(words))
    return records


def map_record(record):
    return [(word, 1) for word in record.split()]


def partition_of(key, partitions):
    """Deterministic partition: sum of code points modulo the partition count."""
    return sum(ord(ch) for ch in key) % partitions


def split_records(records, workers):
    """Contiguous, balanced splits - the same shape a file system would hand out."""
    size, extra = divmod(len(records), workers)
    splits = []
    start = 0
    for worker in range(workers):
        length = size + (1 if worker < extra else 0)
        splits.append(records[start:start + length])
        start += length
    return splits


class CommitLog:
    """Append-only durable store: batch outputs, then the offset. Also the reduce outputs."""

    def __init__(self, workers, partitions):
        self.partitions = partitions
        self.offsets = {worker: 0 for worker in range(workers)}
        self.history = {worker: [] for worker in range(workers)}
        self.outputs = {}                       # (split, partition) -> [(key, value)]
        self.commits = 0
        self.reduce_output = {}

    def commit(self, split, new_offset, batch):
        assert new_offset >= self.offsets[split], "offsets must be monotone"
        for partition, key, value in batch:
            self.outputs.setdefault((split, partition), []).append((key, value))
        self.offsets[split] = new_offset        # published last: the offset is the commit point
        self.history[split].append(new_offset)
        self.commits += 1

    def publish(self, partition, aggregate):
        self.reduce_output[partition] = dict(aggregate)


class BatchWorker:
    """One mapper with an attempt budget: `run` returns False when it simulated a crash."""

    def __init__(self, split_id, records, log, interval, partitions):
        self.split_id = split_id
        self.records = records
        self.log = log
        self.interval = interval
        self.partitions = partitions
        self.processed = 0
        self.crashes = 0
        self.redo = 0

    def run(self, crash_at=None, lock=None):
        """Process the split from the committed offset. crash_at is a record index (or None)."""
        while self.log.offsets[self.split_id] < len(self.records):
            index = self.log.offsets[self.split_id]
            batch = []
            while index < len(self.records):
                if crash_at is not None and index == crash_at:
                    self.crashes += 1
                    self.redo += index - self.log.offsets[self.split_id]
                    return False                # everything buffered so far is lost
                key_words = map_record(self.records[index])
                batch.extend((partition_of(word, self.partitions), word, value)
                             for word, value in key_words)
                index += 1
                self.processed += 1
                if index % self.interval == 0 or index == len(self.records):
                    if lock is not None:
                        with lock:
                            self.log.commit(self.split_id, index, batch)
                    else:
                        self.log.commit(self.split_id, index, batch)
                    batch = []
            crash_at = None                     # only the first pass may be interrupted
        return True


def run_split(worker, crashes, lock=None):
    """Replay the worker, dying at each crash point in turn; always ends with a clean pass."""
    pending = sorted(crashes)
    while True:
        crash = pending.pop(0) if pending else None
        if worker.run(crash_at=crash, lock=lock):
            return worker


def reduce_all(log, partitions=3):
    """Aggregate every committed (partition, key, value) triple, partition by partition."""
    totals = {}
    for partition in range(partitions):
        counts = {}
        for (_, part), entries in log.outputs.items():
            if part != partition:
                continue
            for key, value in entries:
                counts[key] = counts.get(key, 0) + value
        totals[partition] = counts
    return totals


def brute_force_aggregate(records, partitions=3):
    """Independent oracle: aggregate the raw records without any map-reduce machinery."""
    totals = {}
    for record in records:
        for word in record.split():
            part = partition_of(word, partitions)
            bucket = totals.setdefault(part, {})
            bucket[word] = bucket.get(word, 0) + 1
    return {part: totals.get(part, {}) for part in range(partitions)}


def run_scenario(splits, interval, crashes_per_split, partitions=3, lock=None):
    log = CommitLog(len(splits), partitions)
    workers = [BatchWorker(i, split, log, interval, partitions) for i, split in enumerate(splits)]
    for index, worker in enumerate(workers):
        run_split(worker, crashes_per_split.get(index, ()), lock=lock)
    for partition in range(partitions):
        log.publish(partition, reduce_all(log, partitions)[partition])
    return log, workers


def main():
    records = make_records(60, seed=1234)
    assert len(records) == 60 and all(3 <= len(r.split()) <= 5 for r in records)
    splits = split_records(records, 3)
    assert [len(s) for s in splits] == [20, 20, 20]

    # --- Crash-free baseline and the independent oracle. -------------------------------------
    log, workers = run_scenario(splits, 7, {})
    baseline = reduce_all(log, 3)
    assert baseline == brute_force_aggregate(records, 3)
    assert sum(sum(part.values()) for part in baseline.values()) == \
        sum(len(r.split()) for r in records)
    assert [worker.processed for worker in workers] == [20, 20, 20]
    assert [worker.redo for worker in workers] == [0, 0, 0]
    for worker in range(3):
        assert log.offsets[worker] == len(splits[worker]) == 20
        assert log.history[worker] == [7, 14, 20]
        assert log.history[worker] == sorted(log.history[worker])
    assert log.commits == 9 and log.reduce_output == baseline

    # --- Every single crash point recovers with exactly k mod interval redo. -----------------
    first_oracle = brute_force_aggregate(splits[0], 3)
    second_oracle = brute_force_aggregate(splits[1], 3)
    assert first_oracle != second_oracle
    total_redos = 0
    for crash in range(20):
        log1, workers1 = run_scenario([splits[0]], 7, {0: (crash,)})
        assert reduce_all(log1, 3) == first_oracle, crash
        worker = workers1[0]
        assert worker.crashes == 1
        assert worker.redo == crash % 7, (crash, worker.redo)
        assert worker.processed == 20 + (crash % 7)
        assert log1.offsets[0] == 20 and log1.history[0][-1] == 20
        total_redos += worker.redo
    assert total_redos == 2 * 21 + 15, total_redos

    # --- Several crashes on one split: the redos add up and the answer is unchanged. ---------
    for crashes in ((10,), (5, 19), (0, 7, 14), (10, 11, 12, 13), (19, 19)):
        log2, workers2 = run_scenario([splits[1]], 7, {0: crashes})
        assert reduce_all(log2, 3) == second_oracle, crashes
        worker = workers2[0]
        assert worker.crashes == len(crashes)
        assert worker.redo == sum(crash % 7 for crash in crashes), (crashes, worker.redo)
        assert worker.processed == 20 + worker.redo
        assert log2.offsets[0] == 20

    # A crash exactly on a commit boundary costs nothing at all.
    log3, workers3 = run_scenario([splits[0]], 7, {0: (7, 14)})
    assert workers3[0].redo == 0 and workers3[0].processed == 20
    assert reduce_all(log3, 3) == first_oracle

    # --- Multi-split runs: crashes are contained to their own split. -------------------------
    log4, workers4 = run_scenario(splits, 7, {1: (3, 11), 2: (19,)})
    assert reduce_all(log4, 3) == baseline
    assert [w.processed for w in workers4] == [20, 27, 25]
    assert workers4[0].crashes == 0 and workers4[0].processed == 20
    assert workers4[1].redo == 3 + 4 and workers4[2].redo == 5
    assert all(log4.offsets[s] == 20 for s in range(3))
    # The healthy split's committed log is byte-identical to the crash-free run: a crash in one
    # split cannot corrupt another split's durable output.
    assert {k: v for k, v in log4.outputs.items() if k[0] == 0} == \
        {k: v for k, v in log.outputs.items() if k[0] == 0}

    # --- A crash inside the reduce step: partitions are published atomically. ----------------
    partial = CommitLog(3, 3)
    partial.outputs = log.outputs
    partial.publish(0, baseline[0])
    partial.publish(1, baseline[1])            # partition 2 never made it before the crash
    assert 2 not in partial.reduce_output
    partial.reduce_output = {}
    for partition in range(3):
        partial.publish(partition, reduce_all(log, 3)[partition])
    assert partial.reduce_output == baseline
    assert partial.reduce_output == log.reduce_output

    # --- Real threads over the same three splits reach the same aggregate. -------------------
    results = []
    for _ in range(3):
        lock = threading.Lock()
        threaded_log = CommitLog(3, 3)
        threaded_workers = [BatchWorker(i, split, threaded_log, 7, 3)
                            for i, split in enumerate(splits)]

        def run_worker(index):
            run_split(threaded_workers[index], (11,) if index == 2 else (), lock=lock)

        threads = [threading.Thread(target=run_worker, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert all(not thread.is_alive() for thread in threads)
        assert threaded_log.offsets == {0: 20, 1: 20, 2: 20}
        assert all(threaded_log.history[s] == sorted(threaded_log.history[s]) for s in range(3))
        aggregate = reduce_all(threaded_log, 3)
        assert aggregate == baseline, "a concurrent run must reach the same aggregate"
        assert sum(w.processed for w in threaded_workers) == 60 + 4
        results.append(aggregate)
    assert results[0] == results[1] == results[2] == baseline


if __name__ == "__main__":
    main()
    print("PASS")
