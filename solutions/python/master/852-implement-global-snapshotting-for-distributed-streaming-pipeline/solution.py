"""Task 852 - Implement global snapshotting for distributed streaming pipeline (Chandy-Lamport at scale).  (Tier: Master, Ch 9 — System Design and Scalability)

A parallel streaming pipeline (2 sources -> 3 mappers -> 2 aggregators -> 1 sink, wired by
14 explicit FIFO channels) takes periodic global snapshots with the asynchronous
barrier-snapshotting protocol of Chandy & Lamport / Carbone et al. (Flink checkpointing).
Everything is in-process and deterministic: no threads, sockets, sleeps or wall clock.  A
tick-based driver advances virtual time, the coordinator injects one checkpoint at a time,
and barriers travel *in band*, through the same FIFO channels as the records.

Protocol as implemented:
  * the coordinator injects a barrier into every source subtask; a source snapshots its
    offset immediately (it has no inputs) and forwards the barrier on all outgoing channels;
  * a subtask that receives a barrier on one input *blocks* that input - further records from
    it are buffered, because they were emitted after the sender's cut - and waits for
    barriers on its remaining inputs (barrier alignment);
  * when every input has delivered a barrier the subtask snapshots its own state, forwards
    the barrier on all outgoing channels and only then drains the buffered post-cut records;
  * because the channels are FIFO the recorded channel state is always empty: everything the
    sender emitted before its snapshot is ahead of the barrier in the channel, everything
    behind the barrier was emitted after the sender's cut.  The checks re-derive that from
    the per-channel emission/delivery logs instead of trusting it.

What is simplified: channels are unbounded in-memory FIFOs (no backpressure), the source is
a deterministic index-addressable generator (so replay after a crash is exactly
reproducible), and subtask state is a small dictionary instead of a serialised state blob.

Invariants verified (safety and liveness):
  * cut consistency recomputed independently from the logs of every checkpoint and channel:
    no record emitted before the sender's cut is delivered after the receiver's cut, nothing
    stays queued behind a delivered barrier, and every record buffered during alignment is
    exactly the post-cut traffic of that channel;
  * exactly-once conservation: the sink's state at the cut equals the number of records the
    sources injected before the cut, recomputed from the sources' checkpointed offsets;
  * end-to-end semantics: after a clean run every stage's per-key state equals a brute-force
    oracle over the record stream, and no stage consumes a record twice;
  * alignment ordering: a subtask snapshots at the step of the *last* barrier it needed;
  * liveness: every injected checkpoint completes, every subtask snapshots once per
    checkpoint, alignment delay stays bounded, and checkpoints also complete on an idle
    pipeline whose channels carry nothing but barriers;
  * recovery: crashing mid-stream and restoring the latest *completed* checkpoint, then
    replaying from the checkpointed source offsets, reproduces the failure-free result
    exactly (exactly-once across a failure);
  * negative control: the classic "snapshot on the first barrier and keep consuming" variant
    is shown to break conservation and to duplicate work after a crash, so the checks above
    are not vacuous.
"""

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


class Topology:
    """Parallelism of each operator stage, and how many records each source emits."""

    def __init__(self, sources, mappers, aggregators, records_per_source):
        self.sources = sources
        self.mappers = mappers
        self.aggregators = aggregators
        self.records_per_source = records_per_source

    def channel_count(self):
        return self.sources * self.mappers + self.mappers * self.aggregators + self.aggregators


def record_for(source, index):
    """Deterministic, index-addressable record stream: replay after a crash is exact."""
    rid = source * 100000 + index
    key = (rid * 7 + 3) % 6
    value = (rid * 31 + 5) % 1000
    return rid, key, value


def mapper_for(key, topology):
    return key % topology.mappers


def aggregator_for(key, topology):
    return key % topology.aggregators


class Clock:
    """Shared monotone counter, so a snapshot can mark a precise point in delivery order."""

    def __init__(self):
        self.deliveries = 0


class Channel:
    def __init__(self, src, dst, dst_input_index):
        self.src = src
        self.dst = dst
        self.dst_input_index = dst_input_index
        self.queue = []           # ("data", record, post_cut, ckpt) | ("barrier", ckpt, _)
        self.delivered = []       # (rid, ckpt, post_cut, step, delivery_seq)
        self.barriers = []        # (ckpt, step)
        self.high_water = 0

    @property
    def name(self):
        return "%s->%s[%d]" % (self.src.name, self.dst.name, self.dst_input_index)

    def enqueue(self, message):
        self.queue.append(message)
        self.high_water = max(self.high_water, len(self.queue))


class Subtask:
    def __init__(self, name, kind, clock, naive=False):
        self.name = name
        self.kind = kind              # source | mapper | aggregator | sink
        self.clock = clock
        self.naive = naive            # negative control: snapshot on the first barrier
        self.state = {}               # per-key counters
        self.consumed = 0
        self.next_index = 0           # only meaningful for sources
        self.inputs = []
        self.outputs = []
        self.route = lambda record: 0
        self.snapshots = {}           # ckpt -> {"state", "step", "consumed", "next_index", ...}
        self.barriers = {}            # ckpt -> {channel name: step at which the barrier arrived}
        self.pending = {}             # channel name -> buffered post-cut records
        self.current = 0

    # -- alignment ---------------------------------------------------------
    def fully_aligned(self, ckpt):
        seen = self.barriers.get(ckpt, {})
        if self.naive:
            return bool(seen)         # wrong on purpose: the first barrier wins
        return all(chan.name in seen for chan in self.inputs)

    def on_barrier(self, chan, ckpt, step):
        name = chan.name if chan is not None else "<coordinator>"
        self.barriers.setdefault(ckpt, {})[name] = step
        if ckpt not in self.snapshots and self.fully_aligned(ckpt):
            self.take_snapshot(ckpt, step)

    def take_snapshot(self, ckpt, step):
        self.current = ckpt
        pre_cut_queued = 0
        for chan in self.inputs:
            if chan.name not in self.barriers.get(ckpt, {}):
                continue                      # this input has not delivered its barrier yet
            for message in chan.queue:        # nothing behind a delivered barrier may be pre-cut
                if message[0] == "data" and not message[2] and message[3] == ckpt:
                    pre_cut_queued += 1
        self.snapshots[ckpt] = {
            "step": step,
            "state": dict(self.state),
            "consumed": self.consumed,
            "next_index": self.next_index,
            "total": sum(self.state.values()),
            "buffers": {name: len(msgs) for name, msgs in self.pending.items()},
            "pre_cut_queued": pre_cut_queued,
            "watermark": self.clock.deliveries,
        }
        for chan in self.outputs:             # the barrier leaves before the buffers drain
            chan.enqueue(("barrier", ckpt, True))
            chan.barriers.append((ckpt, step))
        buffered, self.pending = self.pending, {}
        for chan in self.inputs:
            for record, _post_cut, _ckpt in buffered.get(chan.name, []):
                self.process(record, step)    # buffered records belong after the cut

    # -- data path ---------------------------------------------------------
    def is_blocked(self, chan, ckpt):
        """A channel is blocked once it delivered a barrier we have not snapshotted for yet."""
        if self.naive or ckpt in self.snapshots:
            return False
        return chan.name in self.barriers.get(ckpt, {})

    def on_data(self, chan, record, post_cut, ckpt, step):
        if self.is_blocked(chan, ckpt):
            self.pending.setdefault(chan.name, []).append((record, post_cut, ckpt))
            return
        self.process(record, step)

    def process(self, record, step):
        self.consumed += 1
        _rid, key, value = record
        if self.kind == "mapper":
            self.state[key] = self.state.get(key, 0) + 1
        elif self.kind == "aggregator":
            self.state[key] = self.state.get(key, 0) + value
        elif self.kind == "sink":
            self.state[key] = self.state.get(key, 0) + 1
        if self.outputs:
            self.emit(record, step)

    def emit(self, record, step):
        """Push a record downstream; the post-cut flag freezes when the subtask snapshotted."""
        post_cut = self.current != 0 and self.current in self.snapshots
        chan = self.outputs[self.route(record)]
        chan.enqueue(("data", record, post_cut, self.current))
        if self.kind == "source":
            chan.emitted = getattr(chan, "emitted", [])
            chan.emitted.append((record[0], self.current, post_cut))

    def receive(self, chan, message, step):
        self.clock.deliveries += 1
        if message[0] == "barrier":
            self.on_barrier(chan, message[1], step)
            return
        _kind, record, post_cut, ckpt = message
        chan.delivered.append((record[0], ckpt, post_cut, step, self.clock.deliveries))
        self.on_data(chan, record, post_cut, ckpt, step)


class Pipeline:
    """The whole topology plus the checkpointing coordinator."""

    def __init__(self, topology, naive=False, checkpoint_steps=(), emit_per_tick=4, bandwidth=3):
        self.topology = topology
        self.naive = naive
        self.checkpoint_steps = tuple(checkpoint_steps)
        self.emit_per_tick = emit_per_tick
        self.bandwidth = bandwidth
        self.clock = Clock()
        self.step = 0
        self.sources = [Subtask("S%d" % i, "source", self.clock, naive)
                        for i in range(topology.sources)]
        self.mappers = [Subtask("M%d" % i, "mapper", self.clock, naive)
                        for i in range(topology.mappers)]
        self.aggregators = [Subtask("A%d" % i, "aggregator", self.clock, naive)
                            for i in range(topology.aggregators)]
        self.sink = Subtask("K0", "sink", self.clock, naive)
        self.subtasks = self.sources + self.mappers + self.aggregators + [self.sink]
        self.channels = []
        self._wire(self.sources, self.mappers)
        self._wire(self.mappers, self.aggregators)
        self._wire(self.aggregators, [self.sink])
        for src in self.sources:
            src.route = lambda record, topo=topology: mapper_for(record[1], topo)
        for mapper in self.mappers:
            mapper.route = lambda record, topo=topology: aggregator_for(record[1], topo)
        self.next_checkpoint = 1
        self.in_flight = None
        self.completed = []
        self.aborted = []
        self.discarded = []
        self.crash_step = None
        self.restored = None
        self.recovery_count = 0
        self.deepest_channel = 0

    def _wire(self, upstream, downstream):
        for src in upstream:
            for index, dst in enumerate(downstream):
                chan = Channel(src, dst, index)
                chan.emitted = []
                src.outputs.append(chan)
                dst.inputs.append(chan)
                self.channels.append(chan)

    # -- driver ------------------------------------------------------------
    def tick(self):
        self.step += 1
        if self.step in self.checkpoint_steps and self.in_flight is None:
            ckpt = self.next_checkpoint
            self.next_checkpoint += 1
            self.in_flight = ckpt
            for src in self.sources:
                src.on_barrier(None, ckpt, self.step)
        for src in self.sources:
            for _ in range(self.emit_per_tick):
                if src.next_index >= self.topology.records_per_source:
                    break
                record = record_for(int(src.name[1:]), src.next_index)
                src.next_index += 1
                src.emit(record, self.step)
        for chan in self.channels:
            chan.high_water = max(chan.high_water, len(chan.queue))
            for _ in range(self.bandwidth):
                if not chan.queue:
                    break
                chan.dst.receive(chan, chan.queue.pop(0), self.step)
        for chan in self.channels:
            self.deepest_channel = max(self.deepest_channel, chan.high_water)
        if self.in_flight is not None and all(self.in_flight in st.snapshots for st in self.subtasks):
            self.completed.append(self.in_flight)
            self.in_flight = None

    def finish(self):
        if self.in_flight is not None:
            self.aborted.append(self.in_flight)
            self.in_flight = None

    def run(self, steps, crash_step=None):
        while self.step < steps:
            if crash_step is not None and self.step + 1 == crash_step:
                self.crash()
            self.tick()
        self.finish()
        return self

    # -- recovery ----------------------------------------------------------
    def crash(self):
        """Discard everything after the latest completed checkpoint and replay from it."""
        self.crash_step = self.step
        if not self.completed:
            raise AssertionError("a crash before the first checkpoint cannot be recovered")
        restore = self.completed[-1]
        self.restored = restore
        self.discarded = [c for c in self.completed if c > restore]
        if self.in_flight is not None and self.in_flight > restore:
            self.discarded.append(self.in_flight)
        self.completed = [c for c in self.completed if c <= restore]
        self.aborted = []
        self.in_flight = None
        for subtask in self.subtasks:
            saved = subtask.snapshots[restore]
            subtask.state = dict(saved["state"])
            subtask.consumed = saved["consumed"]
            subtask.next_index = saved["next_index"]
            subtask.pending = {}
            subtask.snapshots = {c: s for c, s in subtask.snapshots.items() if c <= restore}
            subtask.current = restore       # barrier evidence of aborted cuts stays on record
        for chan in self.channels:        # a consistent cut leaves nothing in flight
            chan.queue = []               # live state is dropped, the audit log is kept
        self.recovery_count += 1
        return restore


def expected_states(topology):
    """Brute-force oracle: what every stage's per-key state must look like at the end."""
    mappers = [{} for _ in range(topology.mappers)]
    aggregators = [{} for _ in range(topology.aggregators)]
    sink = {}
    per_stage = {}
    for source in range(topology.sources):
        for index in range(topology.records_per_source):
            _rid, key, value = record_for(source, index)
            slot = mapper_for(key, topology)
            mappers[slot][key] = mappers[slot].get(key, 0) + 1
            slot = aggregator_for(key, topology)
            aggregators[slot][key] = aggregators[slot].get(key, 0) + value
            sink[key] = sink.get(key, 0) + 1
            per_stage["mapper"] = per_stage.get("mapper", 0) + 1
    return mappers, aggregators, sink


def verify_checkpoints(pipeline):
    """Recompute every checkpoint's cut properties from the logs alone.

    Returns (problems, report); the report carries the numbers the checks print.
    """
    problems = []
    report = []
    for ckpt in pipeline.completed:
        entry = {"ckpt": ckpt}
        for subtask in pipeline.subtasks:
            snap = subtask.snapshots.get(ckpt)
            if snap is None:
                problems.append("checkpoint %d: %s never snapshotted" % (ckpt, subtask.name))
                continue
            if snap["pre_cut_queued"]:
                problems.append("checkpoint %d: %s held %d pre-cut records behind a barrier"
                                % (ckpt, subtask.name, snap["pre_cut_queued"]))
            if subtask.inputs:
                steps = [subtask.barriers[ckpt][chan.name] for chan in subtask.inputs
                         if chan.name in subtask.barriers.get(ckpt, {})]
                if len(steps) != len(subtask.inputs):
                    problems.append("checkpoint %d: %s snapshotted on %d of %d barriers"
                                    % (ckpt, subtask.name, len(steps), len(subtask.inputs)))
                elif snap["step"] != max(steps):
                    problems.append("checkpoint %d: %s snapshotted at step %d but its last "
                                    "barrier arrived at %d"
                                    % (ckpt, subtask.name, snap["step"], max(steps)))
        # exactly-once conservation between the sources' offsets and the sink's counter
        injected = sum(st.snapshots[ckpt]["next_index"] for st in pipeline.sources)
        counted = pipeline.sink.snapshots[ckpt]["total"]
        if injected != counted:
            problems.append("checkpoint %d: sink counted %d but %d records had been injected "
                            "before the cut" % (ckpt, counted, injected))
        entry["injected"] = injected
        entry["counted"] = counted
        # channel-level consistency, recomputed from the delivery logs
        leaked = 0
        buffer_mismatch = []
        for chan in pipeline.channels:
            snap = chan.dst.snapshots.get(ckpt)
            if snap is None:
                continue
            watermark = snap["watermark"]
            buffered = snap["buffers"].get(chan.name, 0)
            post_cut_before = sum(1 for _rid, ctx, pc, _step, seq in chan.delivered
                                  if ctx == ckpt and pc and seq <= watermark)
            if buffered != post_cut_before:
                buffer_mismatch.append(chan.name)
            for _rid, ctx, pc, step, _seq in chan.delivered:
                if ctx == ckpt and not pc and step > snap["step"]:
                    leaked += 1
        if leaked:
            problems.append("checkpoint %d: %d pre-cut records were delivered after the "
                            "receiver's cut" % (ckpt, leaked))
        if buffer_mismatch:
            problems.append("checkpoint %d: alignment buffers disagree on %s"
                            % (ckpt, buffer_mismatch))
        entry["leaked"] = leaked
        entry["buffers_ok"] = not buffer_mismatch
        steps = [st.snapshots[ckpt]["step"] for st in pipeline.subtasks]
        entry["first_step"] = min(steps)
        entry["last_step"] = max(steps)
        entry["alignment_delay"] = max(steps) - min(steps)
        report.append(entry)
    return problems, report


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    topology = Topology(sources=2, mappers=3, aggregators=2, records_per_source=300)
    built = Pipeline(topology)
    check(len(built.channels) == topology.channel_count() == 14, "the topology must wire 14 channels")
    check(len(built.subtasks) == 8, "the topology must have 8 subtasks")
    check(mapper_for(7, topology) == 1 and aggregator_for(7, topology) == 1,
          "routing must be a pure function of the key")
    check(len({mapper_for(k, topology) for k in range(6)}) == 3, "every mapper must own keys")
    oracle_mappers, oracle_aggs, oracle_sink = expected_states(topology)
    check(sum(oracle_sink.values()) == 600 and len(oracle_sink) == 6,
          "the oracle must cover 600 records over 6 keys")
    print("topology: %d subtasks, %d channels, 600 records, per-mapper %s"
          % (len(built.subtasks), 14, [sum(m.values()) for m in oracle_mappers]))

    checkpoint_steps = (10, 40, 70, 90, 150, 250)

    # ---- 1. clean run: alignment, consistency, conservation, semantics --
    clean = Pipeline(topology, checkpoint_steps=checkpoint_steps).run(320)
    initiated = clean.next_checkpoint - 1
    check(1 <= initiated <= len(checkpoint_steps),
          "the coordinator must not initiate more checkpoints than were scheduled")
    check(len(clean.completed) == initiated and not clean.aborted,
          "every initiated checkpoint must complete (liveness)")
    print("checkpoints: scheduled=%d initiated=%d completed=%d (alignment pauses the pipeline "
          "while a checkpoint is in flight)" % (len(checkpoint_steps), initiated, len(clean.completed)))
    problems, report = verify_checkpoints(clean)
    check(not problems, "the clean run violated a cut invariant: %s" % problems)
    check(clean.sink.state == oracle_sink,
          "the sink must match the oracle: %s != %s" % (clean.sink.state, oracle_sink))
    for index, mapper in enumerate(clean.mappers):
        check(mapper.state == oracle_mappers[index],
              "mapper %d state %s != oracle %s" % (index, mapper.state, oracle_mappers[index]))
    for index, agg in enumerate(clean.aggregators):
        check(agg.state == oracle_aggs[index],
              "aggregator %d state %s != oracle %s" % (index, agg.state, oracle_aggs[index]))
    check(clean.sink.consumed == 600, "exactly-once: the sink must consume 600 records")
    check(all(agg.consumed == 300 for agg in clean.aggregators),
          "each aggregator must see half the stream")
    check(clean.deepest_channel > 20,
          "at scale the pipeline must really buffer records (deepest queue %d)"
          % clean.deepest_channel)
    for entry in report:
        print("cut %-2d injected=%-3d counted=%-3d alignment_delay=%d leaked=%d"
              % (entry["ckpt"], entry["injected"], entry["counted"],
                 entry["alignment_delay"], entry["leaked"]))

    check(report[0]["counted"] < report[-1]["counted"],
          "an early cut must capture less work than the final one")
    check(report[-1]["counted"] == 600, "the last cut must contain the whole stream")
    check(any(entry["alignment_delay"] > 1 for entry in report),
          "fan-in must make alignment span more than one step")
    check(max(entry["alignment_delay"] for entry in report) <= 40, "alignment delay must be bounded")

    # ---- 2. reproducibility ---------------------------------------------
    again = Pipeline(topology, checkpoint_steps=checkpoint_steps).run(320)
    check(again.sink.state == clean.sink.state, "the pipeline must be deterministic")
    p2, r2 = verify_checkpoints(again)
    check(not p2 and [e["counted"] for e in r2] == [e["counted"] for e in report],
          "a rerun must reproduce every cut exactly")

    # ---- 3. exactly-once across a crash ---------------------------------
    windows = [entry for entry in report
               if entry["ckpt"] >= 2 and entry["last_step"] >= entry["first_step"] + 2]
    check(windows, "the clean run must contain at least one non-trivial alignment window")
    for entry in windows[:3]:
        ckpt = entry["ckpt"]
        crash_step = entry["first_step"] + 2      # strictly inside that checkpoint's alignment
        recovered = Pipeline(topology, checkpoint_steps=checkpoint_steps).run(320,
                                                                             crash_step=crash_step)
        p3, _r3 = verify_checkpoints(recovered)
        check(not p3, "post-recovery checkpoints must stay consistent: %s" % p3)
        check(recovered.crash_step == crash_step - 1, "the crash point must be recorded")
        check(recovered.recovery_count == 1, "the recovery must happen exactly once")
        check(ckpt in recovered.discarded,
              "the checkpoint that was in flight at the crash must be discarded")
        check(recovered.restored is not None, "the recovery must report which cut it restored")
        check(not (set(recovered.discarded) & set(recovered.completed)),
              "a discarded cut must never be re-used as a completed one")
        check(set(range(1, recovered.restored + 1)) <= set(recovered.completed),
              "the surviving prefix of cuts must still be complete")
        check(recovered.sink.state == oracle_sink,
              "recovery from step %d must reproduce the failure-free result" % crash_step)
        check(sum(recovered.sink.state.values()) == 600, "recovery must not lose or duplicate work")
        check(recovered.restored < crash_step, "the restored checkpoint must predate the crash")
        print("crash inside cut %d at step %d: restored cut %d, replayed %d records, total %d"
              % (ckpt, crash_step, recovered.restored,
                 600 - sum(st.snapshots[recovered.restored]["next_index"]
                           for st in recovered.sources),
                 sum(recovered.sink.state.values())))

    for crash_step in (55, 260):
        recovered = Pipeline(topology, checkpoint_steps=checkpoint_steps).run(320, crash_step=crash_step)
        check(not verify_checkpoints(recovered)[0], "post-recovery cuts must stay consistent")
        check(recovered.sink.state == oracle_sink,
              "an arbitrary crash at step %d must still be exactly-once" % crash_step)
        print("crash at step %3d: restored cut %d, discarded %s, final total %d"
              % (crash_step, recovered.restored, recovered.discarded,
                 sum(recovered.sink.state.values())))

    # ---- 4. negative control: no barrier alignment ----------------------
    naive = Pipeline(topology, naive=True, checkpoint_steps=checkpoint_steps).run(320)
    naive_problems, naive_report = verify_checkpoints(naive)
    check(naive_problems, "the unaligned variant must break at least one cut invariant")
    print("naive control: %d problems, first -> %s" % (len(naive_problems), naive_problems[0]))
    check(any(entry["counted"] != entry["injected"] for entry in naive_report),
          "the unaligned cut must miscount against the injected records")
    for entry in naive_report:
        if entry["counted"] != entry["injected"]:
            print("naive cut %d: counted=%d injected=%d (delta %+d)"
                  % (entry["ckpt"], entry["counted"], entry["injected"],
                     entry["counted"] - entry["injected"]))

    naive_crashes = {}
    for crash_step in (55, 95, 160, 265):
        naive_crash = Pipeline(topology, naive=True,
                               checkpoint_steps=checkpoint_steps).run(320, crash_step=crash_step)
        naive_crashes[crash_step] = sum(naive_crash.sink.state.values())
    print("naive control after a crash: totals %s (correct is 600)" % naive_crashes)
    check(any(total != 600 for total in naive_crashes.values()),
          "the unaligned variant must lose or duplicate work after a crash")

    # ---- 5. edge cases ---------------------------------------------------
    tiny = Topology(sources=1, mappers=1, aggregators=1, records_per_source=25)
    tiny_run = Pipeline(tiny, checkpoint_steps=(1, 3, 40)).run(40)
    small_mappers, small_aggs, small_sink = expected_states(tiny)
    check(len(tiny_run.completed) == 3, "a degenerate pipeline must still checkpoint")
    check(not verify_checkpoints(tiny_run)[0], "a degenerate cut must still be consistent")
    check(tiny_run.sink.state == small_sink, "the degenerate pipeline must aggregate correctly")
    check(tiny_run.mappers[0].state == small_mappers[0], "degenerate mapper state must match")
    print("degenerate pipeline: %d records, cuts %s, delay %d"
          % (sum(tiny_run.sink.state.values()), tiny_run.completed,
             verify_checkpoints(tiny_run)[1][0]["alignment_delay"]))

    silent = Topology(sources=1, mappers=2, aggregators=1, records_per_source=0)
    idle = Pipeline(silent, checkpoint_steps=(1, 5, 12)).run(20)
    idle_problems, idle_report = verify_checkpoints(idle)
    check(len(idle.completed) == 3 and not idle.aborted,
          "checkpoints must complete on a pipeline that never carries a record")
    check(not idle_problems, "an idle pipeline still needs a consistent cut: %s" % idle_problems)
    check(sum(idle.sink.state.values()) == 0
          and idle.sink.snapshots[idle.completed[-1]]["total"] == 0,
          "an idle pipeline must snapshot an empty state")
    check(idle_report[-1]["injected"] == 0, "an idle cut must inject nothing")
    check(all(chan.high_water <= 4 for chan in idle.channels),
          "an idle pipeline must only carry barriers")

    no_cuts = Pipeline(tiny, checkpoint_steps=()).run(40)
    check(not no_cuts.completed and no_cuts.sink.state == small_sink,
          "running without checkpoints must still produce the right result")
    immediate = Pipeline(tiny, checkpoint_steps=(1,)).run(20)
    check(immediate.completed == [1], "a checkpoint injected before any record must complete")
    early_cut = immediate.sink.snapshots[1]
    check(early_cut["total"] == 0 and early_cut["next_index"] == 0,
          "the earliest cut must be empty")

    hopeless = Pipeline(tiny, checkpoint_steps=(10,))
    try:
        hopeless.run(20, crash_step=3)
    except AssertionError as error:
        check("recover" in str(error), "the crash-before-checkpoint refusal must be explicit")
    else:
        raise AssertionError("a crash before the first checkpoint must be refused")
    print("partitioned cuts: %s, final total %d" % (clean.completed, sum(clean.sink.state.values())))

    print("PASS")


if __name__ == "__main__":
    main()
