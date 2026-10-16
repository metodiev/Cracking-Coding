"""Task 760 - Distributed snapshot-based auditing for integrity.  (Tier: Expert, Ch 9 - System Design and Scalability)

Chandy-Lamport gives a *consistent cut* of a distributed computation without stopping it: a process
that takes a snapshot records its own state and sends a marker on every outgoing channel; a process
that receives a marker for the first time records its state, forwards markers, and starts recording
each channel until that channel's marker arrives - the messages recorded on a channel are exactly
the messages that were sent before the sender's snapshot and received after the receiver's.  This
file combines that algorithm with per-process *hash chains*: every event (a local step, a send, a
receive) is appended together with `H(previous_hash || seq || kind || detail)`, so a process's log
carries a tamper-evident state digest, and the snapshot records the chain head at the cut.

The audit then has three independent parts: (1) recompute every chain and localise the first entry
that does not match its recorded hash; (2) verify *cross-process* links - the send and receive events
of one message id must exist in the two logs and agree on the payload, and no receive may refer to a
message id nobody sent; (3) verify the cut - the recorded in-transit set must equal the set computed
independently from the event indices (send inside the cut, receive outside), and no receive may sit
inside the cut while its send is outside.  The third part is the oracle for the Chandy-Lamport
channel recording: the algorithm's bookkeeping is compared with the analytic set difference.

The checks also demonstrate the classic limitation: an attacker who edits an event *and* recomputes
the rest of that process's chain passes part (1) completely, but part (2) fails because the peer's
log still holds the original payload - which is precisely why a chain without a trusted head or a
MAC cannot detect a full rewrite on its own.

Simplified versus a real auditing system: payloads are short demo strings, the hash is a
deterministic splitmix-based toy hash (integrity demonstration, not a cryptographic claim; a real
deployment uses SHA-256 and signed checkpoints), the message log plays the role of independently
recorded evidence, channels are FIFO by construction, delivery is scheduled deterministically and
there is no failure/recovery model.  Verified: exact localisation of every tamper, deterministic
replay, cut consistency, the recorded in-transit set, and the fact that cross-process auditing
catches a chain rewrite.
"""
from collections import deque

MASK64 = (1 << 64) - 1
GENESIS = "0000000000000000"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def chain_hash(previous, kind, detail, seq):
    """Toy hash of one log entry: chained, order-sensitive, payload-sensitive."""
    state = int(previous, 16)
    for byte in ("%d|%s|%s" % (seq, kind, detail)).encode():
        state = (state ^ byte) & MASK64
        state, mix = splitmix64(state)
        state = (state ^ mix) & MASK64
    return "%016x" % state


class Process:
    """One process: an append-only hash-chained event log plus the snapshot bookkeeping."""

    def __init__(self, index):
        self.index = index
        self.events = []                       # (seq, kind, detail, hash)
        self.head = GENESIS
        self.recorded = None                   # {"count", "head", "step"} once it snaps
        self.recording = set()                 # channels still being recorded
        self.channel_in_transit = {}           # sender -> [message ids]

    def record(self, kind, detail):
        seq = len(self.events)
        self.head = chain_hash(self.head, kind, detail, seq)
        self.events.append((seq, kind, detail, self.head))
        return seq

    def record_state(self, step):
        self.recorded = {"count": len(self.events), "head": self.head, "step": step}

    def receive(self, mid, payload, sender, simulation, step):
        index = self.record("recv", "%s:%s" % (mid, payload))
        simulation.messages[mid]["recv_event"] = index
        if self.recorded is not None and sender in self.recording:
            self.channel_in_transit.setdefault(sender, []).append(mid)


class Simulation:
    """Deterministic message-passing computation with one Chandy-Lamport snapshot."""

    def __init__(self, count, seed, snapshot_at, steps):
        self.processes = [Process(index) for index in range(count)]
        self.seed = seed & MASK64
        self.snapshot_at = snapshot_at
        self.steps = steps
        self.pending = []                      # (deliver_step, mid, sender, receiver, payload, marker)
        self.channel_clock = {}
        self.messages = {}
        self.markers = {}
        self.sequence = 0
        self.step_index = 0
        self.history = []                      # (step, pid, kind, detail) - the ground truth

    # ------------------------------------------------------------------ message plumbing
    def schedule(self, sender, receiver, mid, payload, marker):
        delay = 1 + (int(mid[1:]) % 3)
        channel = (sender, receiver)
        deliver_at = max(self.step_index + delay, self.channel_clock.get(channel, 0) + 1)
        self.channel_clock[channel] = deliver_at      # FIFO on every channel, by construction
        self.pending.append((deliver_at, mid, sender, receiver, payload, marker))

    def new_message(self, sender, receiver, payload):
        mid = "m%d" % len(self.messages)
        self.messages[mid] = {"from": sender, "to": receiver, "payload": payload,
                              "send_step": self.step_index, "recv_step": None,
                              "send_event": None, "recv_event": None}
        return mid

    def send(self, sender, receiver, payload):
        mid = self.new_message(sender, receiver, payload)
        index = self.processes[sender].record("send", "%s:%s" % (mid, payload))
        self.messages[mid]["send_event"] = index
        self.history.append((self.step_index, sender, "send", mid))
        self.schedule(sender, receiver, mid, payload, False)
        return mid

    def send_marker(self, sender, receiver):
        mid = "k%d" % len(self.markers)
        self.markers[mid] = {"from": sender, "to": receiver}
        self.schedule(sender, receiver, mid, "MARKER", True)

    def start_snapshot(self):
        initiator = self.processes[0]
        initiator.record_state(self.step_index)
        initiator.recording = {other.index for other in self.processes if other.index != 0}
        for other in self.processes[1:]:
            self.send_marker(0, other.index)
        self.history.append((self.step_index, 0, "snapshot", "start"))

    def receive_marker(self, receiver, sender, mid):
        process = self.processes[receiver]
        if process.recorded is None:
            process.record_state(self.step_index)
            # the channel the marker arrived on is closed at once; every other incoming channel
            # is recorded until its own marker arrives (markers go back along reverse channels too)
            process.recording = {other.index for other in self.processes
                                 if other.index not in (receiver, sender)}
            for other in self.processes:
                if other.index != receiver:
                    self.send_marker(receiver, other.index)
        else:
            process.recording.discard(sender)
        self.history.append((self.step_index, receiver, "marker", mid))

    # ------------------------------------------------------------------------ the schedule
    def run(self):
        while self.step_index < self.steps:
            if self.step_index == self.snapshot_at:
                self.start_snapshot()
            due = sorted([entry for entry in self.pending if entry[0] <= self.step_index],
                         key=lambda entry: (entry[0], entry[1]))
            if due:
                self.pending.remove(due[0])
                deliver_at, mid, sender, receiver, payload, marker = due[0]
                if marker:
                    self.receive_marker(receiver, sender, mid)
                else:
                    self.messages[mid]["recv_step"] = self.step_index
                    self.processes[receiver].receive(mid, payload, sender, self, self.step_index)
                    self.history.append((self.step_index, receiver, "recv", mid))
            else:
                self.seed, r = splitmix64(self.seed)
                pid = r % len(self.processes)
                action = (r >> 8) % 10
                if action < 4 or len(self.processes) == 1:
                    self.processes[pid].record("local", "ticks%d" % (r % 97))
                    self.history.append((self.step_index, pid, "local", "tick"))
                else:
                    receiver = (pid + 1 + (r >> 16) % (len(self.processes) - 1)) % len(self.processes)
                    self.send(pid, receiver, "value%d" % (r % 100))
            self.step_index += 1
        return self

    # ------------------------------------------------------------------------- the audit
    def expected_in_transit(self):
        """Analytic in-transit set: sent inside the cut, received outside it."""
        expected = {}
        for mid, info in self.messages.items():
            sender, receiver = info["from"], info["to"]
            sender_cut = self.processes[sender].recorded
            receiver_cut = self.processes[receiver].recorded
            if sender_cut is None or receiver_cut is None:
                continue
            send_inside = info["send_event"] is not None and info["send_event"] < sender_cut["count"]
            recv_inside = info["recv_event"] is not None and info["recv_event"] < receiver_cut["count"]
            if send_inside and not recv_inside:
                expected.setdefault(sender, []).append(mid)
        return {channel: sorted(mids) for channel, mids in expected.items()}

    def recorded_in_transit(self):
        recorded = {}
        for process in self.processes:
            for channel, mids in process.channel_in_transit.items():
                recorded.setdefault(channel, []).extend(mids)
        return {channel: sorted(mids) for channel, mids in recorded.items()}

    def cut_violations(self):
        """A receive inside the cut whose send is outside would break the snapshot's consistency."""
        bad = []
        for mid, info in self.messages.items():
            sender_cut = self.processes[info["from"]].recorded
            receiver_cut = self.processes[info["to"]].recorded
            if sender_cut is None or receiver_cut is None or info["recv_event"] is None:
                continue
            send_inside = info["send_event"] < sender_cut["count"]
            recv_inside = info["recv_event"] < receiver_cut["count"]
            if recv_inside and not send_inside:
                bad.append(mid)
        return bad


def verify_chain(process):
    """Recompute a process's chain; returns the first index that does not match, or -1.

    The sequence number of every entry must equal its position, so deleting, inserting or
    reordering entries is detected even when the attacker recomputes hashes.
    """
    head = GENESIS
    for index, (seq, kind, detail, stored) in enumerate(process.events):
        if seq != index:
            return index
        head = chain_hash(head, kind, detail, index)
        if head != stored:
            return index
    return -1


def verify_links(processes, messages):
    """Cross-process audit: matching send/receive pairs, and no receive without a send."""
    problems = []
    for mid, info in sorted(messages.items()):
        sender, receiver = info["from"], info["to"]
        expected = "%s:%s" % (mid, info["payload"])
        send_index = info["send_event"]
        if send_index is None or processes[sender].events[send_index][2] != expected:
            problems.append(("send", mid, sender))
        recv_index = info["recv_event"]
        if recv_index is not None and processes[receiver].events[recv_index][2] != expected:
            problems.append(("recv", mid, receiver))
    for process in processes:
        for seq, kind, detail, _hash in process.events:
            if kind == "recv" and detail.split(":")[0] not in messages:
                problems.append(("orphan", detail.split(":")[0], process.index))
    return problems


def audit(simulation):
    """Chain audit + link audit + cut audit, aggregated into one report."""
    chains = {process.index: verify_chain(process) for process in simulation.processes}
    report = {"bad_chains": {index: at for index, at in chains.items() if at >= 0},
              "link_problems": verify_links(simulation.processes, simulation.messages),
              "cut_violations": simulation.cut_violations()}
    report["ok"] = not report["bad_chains"] and not report["link_problems"] \
        and not report["cut_violations"]
    return report


def build(seed=0x5EED, count=3, snapshot_at=12, steps=40):
    return Simulation(count, seed, snapshot_at, steps).run()


def main():
    simulation = build()
    processes = simulation.processes

    # 1. chain basics: order- and payload-sensitive, and reproducible
    first = Process(0)
    first.record("local", "a")
    head_after_one = first.head
    first.record("local", "b")
    check(head_after_one != first.head, "appending an event must advance the head")
    swapped = Process(0)
    swapped.record("local", "b")
    swapped.record("local", "a")
    check(swapped.head != first.head, "the hash chain must be order-sensitive")
    same = Process(0)
    same.record("local", "a")
    check(same.head == head_after_one, "identical logs must give identical heads")
    check(Process(0).head == GENESIS, "a fresh process starts at the genesis hash")

    # 2. the computation really is distributed, with messages in flight
    check(len(simulation.messages) >= 10, "the run must exchange messages: %d"
          % len(simulation.messages))
    check(len(simulation.history) > 30, "the run must perform a reasonable number of steps")
    check(sum(len(p.events) for p in processes) ==
          sum(1 for _step, _pid, kind, _detail in simulation.history if kind in ("local", "send", "recv")),
          "every local/send/recv step must append exactly one event")

    # 3. every chain verifies, and the snapshot recorded a prefix of each chain
    for process in processes:
        check(verify_chain(process) == -1, "the untouched chain of process %d must verify" % process.index)
        check(process.recorded is not None, "every process must have snapped")
        check(process.recorded["count"] <= len(process.events), "a snapshot is a prefix of the log")
        prefix = GENESIS
        for seq, kind, detail, _stored in process.events[:process.recorded["count"]]:
            prefix = chain_hash(prefix, kind, detail, seq)
        check(prefix == process.recorded["head"],
              "the recorded state hash must be the head of the recorded prefix")
    check(processes[0].recorded["step"] <= processes[1].recorded["step"],
          "the initiator records first, the others after their markers arrive")

    # 4. the Chandy-Lamport channel recording equals the analytically computed in-transit set
    recorded = simulation.recorded_in_transit()
    expected = simulation.expected_in_transit()
    check(recorded == expected,
          "the algorithm's in-transit set must match the analytic one:\n%r\n%r" % (recorded, expected))
    check(simulation.cut_violations() == [],
          "no receive may sit inside the cut while its send is outside")
    check(expected, "this run must actually have messages in transit at the cut")
    check(all(mid.startswith("m") for mids in expected.values() for mid in mids),
          "only data messages (never markers) count as in-transit work")

    # 5. the audit of an untouched run is clean
    clean = audit(simulation)
    check(clean["ok"] and clean["bad_chains"] == {} and clean["link_problems"] == [],
          "a healthy run must pass all three audit stages: %r" % (clean,))

    # 6. local tampering is localised by the chain audit, without any cross-process help
    tampered = build()
    victim = tampered.processes[0]
    target = next(index for index, event in enumerate(victim.events) if event[1] == "send")
    seq, kind, detail, _stored = victim.events[target]
    victim.events[target] = (seq, kind, detail.replace(":", ":9", 1), _stored)
    report = audit(tampered)
    check(report["bad_chains"] == {0: target},
          "the chain audit must name the exact edited entry: %r" % (report["bad_chains"],))
    deleted = build()
    del deleted.processes[1].events[3]
    check(verify_chain(deleted.processes[1]) == 3,
          "removing an entry must break the chain exactly at that index")

    # 7. a full chain rewrite passes the local audit but fails the cross-process one
    rewritten = build()
    peer = rewritten.processes[0]
    index = next(i for i, event in enumerate(peer.events) if event[1] == "send")
    mid = peer.events[index][2].split(":")[0]
    original_payload = rewritten.messages[mid]["payload"]
    forged = "value777"
    seq, kind, detail, _stored = peer.events[index]
    peer.events[index] = (seq, kind, "%s:%s" % (mid, forged), _stored)
    head = GENESIS
    rebuilt = []
    for position, (_seq, entry_kind, entry_detail, _old) in enumerate(peer.events):
        head = chain_hash(head, entry_kind, entry_detail, position)
        rebuilt.append((position, entry_kind, entry_detail, head))
    peer.events = rebuilt
    peer.head = head
    check(verify_chain(peer) == -1, "a rewritten chain verifies locally - that is the danger")
    report = audit(rewritten)
    check(report["bad_chains"] == {}, "the local audit cannot see the rewrite")
    check(("send", mid, 0) in report["link_problems"],
          "the audit must expose the forged payload on the sender's side: %r"
          % (report["link_problems"],))
    check(("recv", mid, rewritten.messages[mid]["to"]) not in report["link_problems"],
          "the untouched peer log stays perfectly consistent")
    check(not report["ok"], "the combined audit must fail")
    check(original_payload != forged and rewritten.messages[mid]["payload"] == original_payload,
          "the record of the message is what makes the rewrite visible")

    # 8. a forged receive appended to a log (chain fixed) is caught as an orphan
    orphaned = build()
    culprit = orphaned.processes[2]
    culprit.record("recv", "m9999:value1")
    check(orphaned.messages.get("m9999") is None, "the forged message id was never sent")
    report = audit(orphaned)
    check(("orphan", "m9999", 2) in report["link_problems"],
          "a receive without a matching send must be reported: %r" % (report["link_problems"],))
    check(verify_chain(culprit) == -1, "the appended entry itself is a valid chain entry")

    # 9. determinism of the whole pipeline
    twin = build()
    check([p.events for p in twin.processes] == [p.events for p in processes],
          "two runs with the same seed must produce identical logs")
    check(twin.recorded_in_transit() == recorded and audit(twin)["ok"], "and identical snapshots")
    check([(mid, info["payload"]) for mid, info in sorted(twin.messages.items())] ==
          [(mid, info["payload"]) for mid, info in sorted(simulation.messages.items())],
          "message payloads must be reproducible")

    # 10. boundary: a single process (no channels at all)
    solo = build(count=1, snapshot_at=3, steps=8)
    check(len(solo.messages) == 0 and solo.processes[0].recorded is not None,
          "a single process still takes a snapshot")
    check(solo.recorded_in_transit() == {} and solo.cut_violations() == [],
          "there is nothing in transit without channels")
    check(audit(solo)["ok"], "the single-process audit must be clean")

    # 11. boundary: a snapshot before any event, and one at the very last step
    early = build(snapshot_at=0, steps=6)
    check(early.processes[0].recorded["count"] == 0
          and early.processes[0].recorded["head"] == GENESIS,
          "the initiator may snap before doing any work")
    check(early.recorded_in_transit() == early.expected_in_transit(),
          "the analytic and recorded in-transit sets must agree for an early snapshot too")
    late = build(snapshot_at=39, steps=40)
    check(late.recorded_in_transit() == late.expected_in_transit(),
          "a snapshot at the last step must still be consistent")
    check(audit(early)["ok"] and audit(late)["ok"], "both boundary snapshots must audit clean")

    # 12. in-transit messages are attributed to the right channel, and the count adds up
    total_in_transit = sum(len(mids) for mids in recorded.values())
    send_inside = sum(1 for info in simulation.messages.values()
                      if info["send_event"] is not None
                      and info["send_event"] < processes[info["from"]].recorded["count"])
    check(total_in_transit <= send_inside, "in-transit messages are a subset of those sent inside")
    check(all(mid in simulation.messages for mids in recorded.values() for mid in mids),
          "every in-transit id must be a real message")

    print("760 processes=%d events=%d messages=%d markers=%d"
          % (len(processes), sum(len(p.events) for p in processes), len(simulation.messages),
             len(simulation.markers)))
    print("760 snapshot at step %d: recorded counts %s, in-transit %r"
          % (simulation.snapshot_at, [p.recorded["count"] for p in processes], recorded))
    print("760 audit on a healthy run: ok=%s; after a chain rewrite: bad_chains=%r links=%r"
          % (clean["ok"], report["bad_chains"], report["link_problems"]))
    print("PASS")


if __name__ == "__main__":
    main()
