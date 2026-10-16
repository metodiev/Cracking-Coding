"""Task 936 - Implement transparent operator checkpointing and resume in compute graph.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A deterministic, in-memory streaming runtime with barrier-aligned checkpoints.  Two source tasks
emit epochs of keyed records; a single stateful aggregate operator (per-key running sum) receives
both channels and snapshots its keyed state plus the per-source offsets only when a barrier has
arrived on every input; the snapshot goes to the checkpoint store, the operator acks, and the
transactional sink stages the epoch's records and commits them once that checkpoint completes.  A
crash is a raised exception in the middle of an epoch: the runtime rolls back to the latest
committed checkpoint (restoring the operator state, truncating the sink output, discarding the
aborted epoch's staged records) and replays the sources from the stored offsets - that rollback of
both state and staged output is what makes the end-to-end result exactly-once.

Simplified versus Flink/Spark: no network, no watermarks, no async snapshotting and no alignment
timeout - a crashed task is restarted in place.  Verified: an interrupted-and-resumed run equals
the checkpoint-free reference run (independent oracle) for failures before an epoch, in the middle
of an epoch and before the first checkpoint; every committed checkpoint is fully aligned and commits
are monotone; a stale checkpoint is honoured; snapshots are immutable copies; the zero-epoch
boundary and determinism.
"""
MASK64 = (1 << 64) - 1
SOURCES = ("s1", "s2")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed-seed, so the generated workload is identical on every run."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, bound):
        return self.next() % bound


def source_records(source, epoch):
    rng = Rng(len(source) * 100_000 + epoch)
    return [("k%d" % rng.below(3), rng.below(9) + 1) for _ in range(2)]


class Crash(Exception):
    """Models a task crash; the driver recovers from the latest committed checkpoint."""


class Pipeline:
    """Two aligned source channels -> one stateful aggregate -> transactional sink."""

    def __init__(self, sources=SOURCES):
        self.sources = tuple(sources)
        self.state = {}          # operator keyed state
        self.acked = {}          # epoch -> channels whose barrier has arrived
        self.checkpoints = {}    # epoch -> (state copy, per-source offsets)
        self.sink = {}           # epoch -> records committed with that checkpoint
        self.staged = {}         # epoch -> records not yet committed
        self.offsets = {name: 0 for name in self.sources}
        self.commits = []        # commit order

    def snapshot(self):
        return (dict(self.state), dict(self.offsets))

    def deliver(self, epoch, source, records, barrier):
        for key, value in records:
            self.state[key] = self.state.get(key, 0) + value
            self.staged.setdefault(epoch, []).append((key, value))
        if not barrier:
            return
        self.acked.setdefault(epoch, set()).add(source)
        if len(self.acked[epoch]) == len(self.sources):
            self.checkpoints[epoch] = self.snapshot()
            self.sink[epoch] = self.staged.pop(epoch, [])
            self.commits.append(epoch)

    def run(self, start, stop, fail_at=None, midway=False):
        """Process epochs [start, stop); midway crashes after the first channel of an epoch."""
        for epoch in range(start, stop):
            if fail_at == epoch and not midway:
                raise Crash(epoch)
            for index, source in enumerate(self.sources):
                if fail_at == epoch and midway and index == 1:
                    raise Crash(epoch)
                self.deliver(epoch, source, source_records(source, epoch), True)
                self.offsets[source] = epoch + 1

    def rollback(self, epoch):
        """Discard everything after `epoch` and return the epoch to replay from."""
        if epoch is None:
            self.state, self.sink, self.staged = {}, {}, {}
            self.checkpoints, self.acked, self.commits = {}, {}, []
            self.offsets = {name: 0 for name in self.sources}
            return 0
        state, offsets = self.checkpoints[epoch]
        self.state = dict(state)
        self.offsets = dict(offsets)
        self.sink = {key: value for key, value in self.sink.items() if key <= epoch}
        self.checkpoints = {key: value for key, value in self.checkpoints.items() if key <= epoch}
        self.acked = {key: value for key, value in self.acked.items() if key <= epoch}
        self.commits = [key for key in self.commits if key <= epoch]
        self.staged = {}
        return epoch + 1

    def output(self):
        return [record for epoch in sorted(self.sink) for record in self.sink[epoch]]


def recover(epochs, fail_at=None, midway=False):
    """Run, crash, roll back to the latest committed checkpoint, resume; return the pipeline."""
    pipeline = Pipeline()
    try:
        pipeline.run(0, epochs, fail_at, midway)
    except Crash:
        start = pipeline.rollback(max(pipeline.checkpoints) if pipeline.checkpoints else None)
        pipeline.run(start, epochs)
    return pipeline


def oracle(epochs):
    """Independent reference: replay the same inputs with no checkpoints and no failure."""
    state, output = {}, []
    for epoch in range(epochs):
        for source in SOURCES:
            for key, value in source_records(source, epoch):
                state[key] = state.get(key, 0) + value
                output.append((key, value))
    return state, output


def main():
    epochs = 7
    state, output = oracle(epochs)
    clean = Pipeline()
    clean.run(0, epochs)

    # 1) happy path: a clean run reproduces the checkpoint-free reference exactly.
    check(clean.state == state and clean.output() == output,
          f"clean run must match the oracle: {clean.state} vs {state}")
    # 2) only complete epochs are visible: sink, checkpoints and commits agree, nothing staged.
    check(sorted(clean.sink) == sorted(clean.checkpoints) == clean.commits == list(range(epochs))
          and clean.staged == {},
          f"committed={sorted(clean.sink)} checkpoints={sorted(clean.checkpoints)}")
    # 3) barrier alignment: no checkpoint is committed before every channel acked that epoch.
    check(all(clean.acked[epoch] == set(SOURCES) for epoch in clean.sink),
          f"alignment broken: {sorted(clean.acked.items())}")
    # 4) the stored snapshot is an independent oracle-checked copy of the operator state.
    prefix_state, _ = oracle(3)
    snapshot = clean.checkpoints[2][0]
    clean.state["poison"] = 1
    check(snapshot == prefix_state and clean.checkpoints[2][0] == prefix_state
          and "poison" not in clean.checkpoints[2][0],
          f"checkpoint 2 must equal the reference prefix state: {snapshot} vs {prefix_state}")

    # 5) failure between epochs: resume replays from the committed offset, exactly once.
    boundary = recover(epochs, fail_at=3)
    check(boundary.state == state and boundary.output() == output,
          f"resume after a crash before epoch 3 diverged: {boundary.state}")
    # 6) failure in the middle of an epoch: the partially applied epoch is discarded.
    midway = recover(epochs, fail_at=5, midway=True)
    check(midway.state == state and midway.output() == output
          and midway.output().count(("k0", 1)) == output.count(("k0", 1)),
          f"resume after a mid-epoch crash must not lose or duplicate records: {midway.state}")
    # 7) boundary: a crash before the first checkpoint restarts the job from scratch.
    first = Pipeline()
    try:
        first.run(0, epochs, fail_at=0, midway=True)
    except Crash:
        start = first.rollback(None)
    check(start == 0 and first.state == {} and first.staged == {} and first.output() == [],
          f"a crash before the first checkpoint must rewind to the empty state: {first.state}")
    first.run(start, epochs)
    check(first.state == state and first.output() == output,
          "restarting from scratch must still reach the reference result")

    # 8) rewinding to a stale checkpoint and replaying from there also converges.
    stale = recover(epochs, fail_at=6)
    replay_from = stale.rollback(1)
    stale.run(replay_from, epochs)
    check(replay_from == 2 and stale.state == state and stale.output() == output
          and stale.commits == list(range(epochs)),
          f"replay from stale checkpoint 1 diverged: start={replay_from} {stale.state}")
    # 9) degenerate input plus determinism.
    empty = Pipeline()
    empty.run(0, 0)
    check(empty.state == {} and empty.output() == [] and empty.checkpoints == {},
          "zero epochs must produce no state and no checkpoint")
    twin = recover(epochs, fail_at=5, midway=True)
    check(twin.output() == midway.output() and twin.state == midway.state
          and twin.commits == midway.commits,
          "two identical runs must be bit-identical")

    print(f"epochs={epochs} keys={len(state)} records={len(output)} "
          f"checkpoints={len(clean.checkpoints)}")
    print(f"resumed state matches oracle at fail_at=3/{len(boundary.output())} records and "
          f"mid-epoch fail_at=5/{len(midway.output())} records")
    print("PASS")


if __name__ == "__main__":
    main()
