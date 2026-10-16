"""Task 661 - Implement chain replication simulation for strong consistency.  (Tier: Expert, Ch 9 - System Design and Scalability)

A chain head...tail is simulated on a virtual clock: a write enters at the head, each replica appends it
before passing it on, the tail - the only replica that acks - returns the acknowledgement along the chain,
and reads are served by the tail alone. That single direction of data flow is what makes the scheme
strongly consistent, and the invariants asserted after every single event are its consequences: each
successor's log is a prefix of its predecessor's, every log holds consecutive sequence numbers (no gaps),
and no acknowledgement exists before the tail appended the record. Failures are instantaneous
reconfigurations that bypass the dead replica (a forward in flight is handed to its successor, an ack to
its predecessor, and every hop is idempotent by sequence number), and a rejoining replica catches up from
the tail before it serves anything.

Verified: ack semantics, the exact healthy ack latency 2 * sum(hop latencies), the prefix and gap
invariants after every event, head/tail/middle failures without losing an acknowledged write,
retransmission idempotence, rejoin ordering and determinism. Simplified: perfect instantaneous failure
detection, one-shot catch-up, fixed per-replica hop latency.
"""
import heapq

MAX_EVENTS = 20000


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class ChainReplication:
    """Chain replication with per-replica hop latency, reconfiguration and catch-up."""

    def __init__(self, chain, hop_latency):
        self.chain = list(chain)
        check(len(self.chain) >= 1, "a chain needs at least one replica")
        self.hop = dict(hop_latency)
        for replica in self.chain:
            check(replica in self.hop, "every replica needs a hop latency")
        self.logs = {replica: [] for replica in chain}
        self.events = []
        self.order = 0
        self.now = 0.0
        self.next_seq = 0
        self.acked = []                     # (seq, value, ack_time)
        self.tail_append = {}               # seq -> time the tail appended it
        self.bypass_next = {}               # removed replica -> node that followed it
        self.bypass_prev = {}               # removed replica -> node that preceded it
        self.failed = set()
        self.catching_up = set()
        self.reads = []                     # (served_by, seq, value, latency)

    # ------------------------------------------------------------------ scheduling
    def _schedule(self, delay, kind, payload):
        check(len(self.events) < MAX_EVENTS, "too many pending events")
        self.order += 1
        heapq.heappush(self.events, (self.now + delay, self.order, kind, payload))

    def _successor_of(self, source):
        if source is None:
            return self.chain[0] if self.chain else None
        if source in self.chain:
            index = self.chain.index(source)
            return self.chain[index + 1] if index + 1 < len(self.chain) else None
        following = self.bypass_next.get(source)
        return self._successor_of(following) if following is not None else None

    def _predecessor_of(self, source):
        if source is None:
            return None
        if source in self.chain:
            index = self.chain.index(source)
            return self.chain[index - 1] if index > 0 else None
        preceding = self.bypass_prev.get(source)
        return self._predecessor_of(preceding) if preceding is not None else None

    # ------------------------------------------------------------------ log handling
    def _append(self, replica, seq, value):
        log = self.logs[replica]
        if log and log[-1][0] == seq:
            return False                     # idempotent: a retransmitted record is a no-op
        check(not log or log[-1][0] < seq, "a log may never be reordered")
        check(not log or log[-1][0] == seq - 1, "a log may never develop a gap")
        log.append((seq, value))
        return True

    # ------------------------------------------------------------------ client API
    def submit(self, value, delay=0.0):
        """Send a write to the head; the ack arrives later and is recorded in `acked`."""
        check(bool(self.chain), "a write needs a live chain")
        self.next_seq += 1
        seq = self.next_seq
        # the client-to-head hop is charged to the head, every later hop to the node it reaches
        self._schedule(delay + self.hop[self.chain[0]], "forward", (seq, value, None))
        return seq

    def read(self):
        """A read is served by the tail only; it returns the tail's newest record."""
        check(bool(self.chain), "a read needs a live chain")
        tail = self.chain[-1]
        check(tail not in self.catching_up, "a catching-up replica may never serve a read")
        log = self.logs[tail]
        latency = self.hop[tail]
        entry = log[-1] if log else (0, None)
        self.reads.append((tail, entry[0], entry[1], latency))
        return entry[1], entry[0], latency, tail

    # ------------------------------------------------------------------ events
    def _deliver_forward(self, seq, value, source):
        target = self._successor_of(source)
        if target is None:
            return                           # the source was the tail: the ack was already sent
        self._append(target, seq, value)
        if target == self.chain[-1]:
            self.tail_append[seq] = self.now
            predecessor = self._predecessor_of(target)
            if predecessor is None:                     # a single-replica chain
                self._schedule(self.hop[target], "client_ack", (seq, value))
            else:
                self._schedule(self.hop[target], "ack", (seq, value, predecessor))
        else:
            following = self._successor_of(target)
            # cost accounting: a forward hop is charged to the node it reaches, an ack hop to the
            # node it leaves, so one full trip costs exactly the sum of the hop latencies
            self._schedule(self.hop[following], "forward", (seq, value, target))

    def _deliver_ack(self, seq, value, holder):
        predecessor = self._predecessor_of(holder)
        if predecessor is None:
            self._schedule(self.hop[holder], "client_ack", (seq, value))
            return
        self._schedule(self.hop[holder], "ack", (seq, value, predecessor))

    def step(self):
        """Process the earliest pending event."""
        check(bool(self.events), "no pending event")
        time, _, kind, payload = heapq.heappop(self.events)
        self.now = max(self.now, time)
        if kind == "forward":
            self._deliver_forward(*payload)
        elif kind == "ack":
            self._deliver_ack(*payload)
        elif kind == "client_ack":
            self.acked.append((payload[0], payload[1], self.now))
        elif kind == "fail":
            self._reconfigure(*payload)
        elif kind == "rejoin":
            self._join_chain(*payload)
        elif kind == "finish_join":
            self._finish_join(*payload)
        else:
            raise AssertionError(f"unknown event {kind}")
        return kind

    def run_all(self):
        while self.events:
            self.step()
        return self

    def run_until(self, time):
        while self.events and self.events[0][0] <= time:
            self.step()
        self.now = max(self.now, time)
        return self

    # ------------------------------------------------------------------ reconfiguration
    def fail(self, replica, delay=0.0):
        self._schedule(delay, "fail", (replica,))

    def _reconfigure(self, replica):
        check(replica in self.chain, f"replica {replica} is not in the chain")
        index = self.chain.index(replica)
        following = self.chain[index + 1] if index + 1 < len(self.chain) else None
        preceding = self.chain[index - 1] if index > 0 else None
        self.bypass_next[replica] = following
        self.bypass_prev[replica] = preceding
        self.chain.pop(index)
        self.failed.add(replica)
        check(self.chain, "a chain may never be reconfigured to nothing")

    def rejoin(self, replica, catchup_delay=5.0, delay=0.0):
        self._schedule(delay, "rejoin", (replica, catchup_delay))

    def _join_chain(self, replica, catchup_delay):
        """Enter the catch-up phase: the replica is not in the chain and serves no reads yet."""
        check(replica not in self.chain, "a replica already in the chain need not rejoin")
        check(replica not in self.catching_up, "a replica cannot catch up twice at once")
        self.failed.discard(replica)
        self.catching_up.add(replica)
        self._schedule(catchup_delay, "finish_join", (replica,))

    def _finish_join(self, replica):
        check(replica in self.catching_up, "only a catching-up replica can finish its catch-up")
        tail = self.chain[-1]
        self.logs[replica] = list(self.logs[tail])          # one-shot state transfer
        self.catching_up.discard(replica)
        self.chain.append(replica)
        self.bypass_next.pop(replica, None)
        self.bypass_prev.pop(replica, None)
        self.check_invariants()

    # ------------------------------------------------------------------ invariants
    def check_invariants(self):
        for replica in self.chain:
            seqs = [seq for seq, _ in self.logs[replica]]
            check(seqs == list(range(1, len(seqs) + 1)),
                  "every live log must hold consecutive sequence numbers from one")
        for index in range(len(self.chain) - 1):
            predecessor = [seq for seq, _ in self.logs[self.chain[index]]]
            successor = [seq for seq, _ in self.logs[self.chain[index + 1]]]
            check(successor == predecessor[:len(successor)],
                  "each successor's log must be a prefix of its predecessor's log")
        tail_seqs = {seq for seq, _ in self.logs[self.chain[-1]]}
        for seq, _, _ in self.acked:
            check(seq in tail_seqs, "an acknowledged write must be present in the tail's log")
        for seq, value, ack_time in self.acked:
            check(ack_time >= self.tail_append[seq],
                  "the ack may never be issued before the tail appended the record")
        head = self.chain[0]
        head_values = [value for _, value in self.logs[head]]
        for index in range(1, len(self.chain)):
            follower = [value for _, value in self.logs[self.chain[index]]]
            check(follower == head_values[:len(follower)],
                  "all replicas must observe the same write order")


def main():
    hops = {"r1": 1.0, "r2": 2.0, "r3": 3.0, "r4": 4.0}
    total_hop = sum(hops.values())

    # 1. a write is acked only after the tail has appended it, with the exact round-trip latency
    chain = ChainReplication(["r1", "r2", "r3"], hops)
    seq = chain.submit("v1")
    chain.run_all()
    check(seq == 1, "the first write gets sequence number one")
    check(len(chain.acked) == 1, "the write must be acknowledged exactly once")
    tail_time = chain.tail_append[1]
    check(abs(tail_time - sum(hops[r] for r in ("r1", "r2", "r3"))) < 1e-9,
          "the tail appends after the sum of the forward hop latencies")
    ack_time = chain.acked[0][2]
    check(abs(ack_time - 2 * sum(hops[r] for r in ("r1", "r2", "r3"))) < 1e-9,
          "the ack latency must be the forward trip plus the backward trip")
    chain.check_invariants()
    check([seq for seq, _ in chain.logs["r1"]] == [1], "the head holds the record")
    check(chain.logs["r3"] == [(1, "v1")], "the tail holds the record")

    # 2. clean-chain behaviour: every replica is a prefix of the previous one, acked order preserved
    chain = ChainReplication(["r1", "r2", "r3", "r4"], hops)
    for index in range(5):
        chain.submit(f"v{index}")
    chain.run_all()
    chain.check_invariants()
    check([entry[1] for entry in chain.acked] == [f"v{i}" for i in range(5)],
          "writes must be acknowledged in submission order")
    check(all(len(chain.logs[replica]) == 5 for replica in chain.chain),
          "a quiescent chain must have every replica fully caught up")
    check(all(entry[2] > 0 for entry in chain.acked), "every write must record an ack time")
    check(chain.acked[0][2] == chain.acked[-1][2],
          "writes submitted together on a quiescent chain are acknowledged together")
    for index in range(1, len(chain.acked)):
        check(chain.acked[index][2] >= chain.acked[index - 1][2] - 1e-9,
              "acknowledgements cannot travel back in time")
    check(abs(chain.acked[0][2] - 2 * total_hop) < 1e-9,
          f"a healthy chain acks after exactly 2 * {total_hop}")

    # 3. reads are served by the tail and see every acknowledged write
    value, version, latency, served_by = chain.read()
    check(served_by == "r4" and latency == hops["r4"], "reads must be served by the tail")
    check(value == "v4" and version == 5, "the tail read returns the newest acknowledged write")
    check(latency < chain.acked[0][2], "a tail read is cheaper than a write round trip")

    # 4. a longer chain costs more per write, a shorter one less
    short = ChainReplication(["r1", "r2"], {"r1": 1.0, "r2": 2.0})
    short.submit("v1")
    short.run_all()
    long = ChainReplication(["r1", "r2", "r3", "r4"], hops)
    long.submit("v1")
    long.run_all()
    check(short.acked[0][2] < long.acked[0][2],
          "adding replicas to the chain must increase the write latency")
    check(abs(short.acked[0][2] - 2 * 3.0) < 1e-9, "the two-replica chain acks after twice 3")

    # 5. progress in the middle of a write: the tail has not been reached yet
    partial = ChainReplication(["r1", "r2", "r3"], hops)
    partial.submit("v1")
    partial.run_until(1.0)
    check(len(partial.acked) == 0, "no ack may exist before the round trip completes")
    check(len(partial.logs["r1"]) == 1, "the head appends after the client hop")
    check(len(partial.logs["r2"]) == 0, "the middle replica has not been reached yet")
    check(len(partial.logs["r3"]) == 0, "the tail has not been reached yet")
    partial.check_invariants()
    partial.run_until(3.0)
    check(len(partial.logs["r2"]) == 1, "the forward hop reaches the middle replica")
    check(len(partial.logs["r3"]) == 0, "the tail is reached last")
    partial.run_all()
    partial.check_invariants()
    check(len(partial.acked) == 1, "the write completes once the tail is reached")

    # 6. tail failure: reconfiguration without losing acknowledged writes
    failing = ChainReplication(["r1", "r2", "r3", "r4"], hops)
    failing.submit("v1")
    failing.run_all()
    failing.fail("r4", delay=1.0)
    failing.run_all()
    check(failing.chain == ["r1", "r2", "r3"], "the failed tail must leave the chain")
    check("r4" in failing.failed, "the failed replica must be marked down")
    failing.check_invariants()
    value, version, _, served_by = failing.read()
    check(served_by == "r3" and value == "v1", "the new tail serves the acknowledged write")
    failing.submit("v2")
    failing.run_all()
    check([entry[1] for entry in failing.acked] == ["v1", "v2"],
          "the shortened chain must still accept writes")
    failing.check_invariants()

    # 7. head failure: the next replica takes over and nothing is lost
    decapitated = ChainReplication(["r1", "r2", "r3"], hops)
    decapitated.submit("v1")
    decapitated.run_all()
    decapitated.fail("r1", delay=0.5)
    decapitated.run_all()
    check(decapitated.chain == ["r2", "r3"], "the failed head must leave the chain")
    decapitated.submit("v2")
    decapitated.run_all()
    check([entry[1] for entry in decapitated.acked] == ["v1", "v2"],
          "the new head must accept writes")
    decapitated.check_invariants()

    # 8. a mid-chain failure while a write is in flight is bypassed without a gap or duplicate
    in_flight = ChainReplication(["r1", "r2", "r3", "r4"], hops)
    seq = in_flight.submit("v1")
    in_flight.run_until(1.0)                  # the head has it, r2 is about to be reached
    check(len(in_flight.logs["r1"]) == 1, "the head appended the record")
    in_flight.fail("r2", delay=0.5)
    in_flight.run_all()
    in_flight.check_invariants()
    check(in_flight.chain == ["r1", "r3", "r4"], "the failed middle replica must be bypassed")
    check([entry[1] for entry in in_flight.acked] == ["v1"],
          "an in-flight write must survive a middle failure")
    check(all(log[-1] == (seq, "v1") for log in in_flight.logs.values() if log),
          "no gap and no duplicate records after the bypass")
    check(len(in_flight.logs["r4"]) == 1, "the tail received exactly one copy")
    in_flight.submit("v2")
    in_flight.run_all()
    in_flight.check_invariants()
    check(len(in_flight.logs["r1"]) == 2 and len(in_flight.logs["r4"]) == 2,
          "the reconfigured chain keeps making progress")

    # 9. retransmission idempotence: replaying a forward adds nothing to the log
    replay = ChainReplication(["r1", "r2"], {"r1": 1.0, "r2": 1.0})
    replay._append("r2", 1, "v1")
    replay._append("r2", 1, "v1")
    check(replay.logs["r2"] == [(1, "v1")], "a duplicate forward must be ignored")
    try:
        replay._append("r2", 3, "v3")
        raise AssertionError("a gap in the log must be rejected")
    except AssertionError as error:
        check(str(error).startswith("a log may never develop a gap"), "the gap guard must fire")
    replay._append("r2", 2, "v2")
    check([seq for seq, _ in replay.logs["r2"]] == [1, 2], "the log must continue in order")
    try:
        replay._append("r2", 1, "v1")
        raise AssertionError("reordering must be rejected")
    except AssertionError as error:
        check(str(error).startswith("a log may never be reordered"), "the order guard must fire")

    # 10. rejoin: the recovered replica catches up before it serves reads or writes
    recovering = ChainReplication(["r1", "r2", "r3"], hops)
    recovering.submit("v1")
    recovering.run_all()
    recovering.fail("r3", delay=0.5)
    recovering.run_all()
    recovering.submit("v2")
    recovering.run_all()
    check(recovering.chain == ["r1", "r2"], "the chain is short while r3 is down")
    recovering.rejoin("r3", catchup_delay=4.0)
    recovering.step()                        # the rejoin event starts the catch-up phase
    check("r3" in recovering.catching_up, "a rejoining replica must catch up before joining")
    check("r3" not in recovering.chain, "a catching-up replica is not part of the chain")
    recovered_value, recovered_version, _, served_by = recovering.read()
    check(served_by != "r3" and recovered_value == "v2" and recovered_version == 2,
          "reads keep working while a replica catches up")
    recovering.run_all()
    check(recovering.chain == ["r1", "r2", "r3"], "the recovered replica rejoins behind the tail")
    check(recovering.logs["r3"] == recovering.logs["r2"],
          "the recovery must transfer the tail's full log")
    check(not recovering.catching_up, "the catch-up phase must be finished")
    recovering.submit("v3")
    recovering.run_all()
    recovering.check_invariants()
    check(len(recovering.logs["r3"]) == 3, "the recovered replica keeps in step afterwards")

    # 11. every event of a mixed schedule keeps the invariants
    mixed = ChainReplication(["r1", "r2", "r3", "r4"], hops)
    for index in range(6):
        mixed.submit(f"v{index}")
    mixed.fail("r2", delay=7.0)
    mixed.rejoin("r2", catchup_delay=3.0, delay=12.0)
    guard = 0
    while mixed.events:
        mixed.step()
        guard += 1
        check(guard < 400, "the mixed schedule must terminate")
        mixed.check_invariants()
    check(len(mixed.acked) == 6, "every submitted write must be acknowledged in the mixed schedule")
    check([entry[1] for entry in mixed.acked] == [f"v{i}" for i in range(6)],
          "acknowledged order must match submission order")

    # 12. determinism
    first, second = ChainReplication(["r1", "r2", "r3"], hops), ChainReplication(["r1", "r2", "r3"], hops)
    for runner in (first, second):
        for index in range(4):
            runner.submit(f"v{index}")
        runner.fail("r2", delay=2.5)
        runner.rejoin("r2", catchup_delay=2.0, delay=6.0)
        runner.run_all()
    check(first.acked == second.acked, "identical schedules must produce identical acks")
    check(first.logs == second.logs, "identical schedules must produce identical logs")
    check([entry[:2] for entry in first.reads] == [entry[:2] for entry in second.reads],
          "reads must be reproducible")

    print(f"661 healthy ack latency={chain.acked[0][2]} read latency={latency} "
          f"chain={chain.chain}")
    print(f"661 mixed schedule: acks={len(mixed.acked)} chain={mixed.chain} "
          f"failed={sorted(mixed.failed)}")
    print("PASS")


if __name__ == "__main__":
    main()
