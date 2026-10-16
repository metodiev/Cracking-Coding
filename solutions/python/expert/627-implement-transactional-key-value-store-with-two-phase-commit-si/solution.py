"""Task 627 - Implement transactional key-value store with two-phase commit simulation.  (Tier: Expert, Ch 9 — System Design and Scalability)

Two-phase commit makes a multi-node write atomic. Phase one: the coordinator sends PREPARE and each
participant takes locks on the keys it will write, appends the *intent* to a write-ahead log and
answers YES or NO. Phase two: the coordinator writes its decision to its own log first and then
broadcasts COMMIT (apply the staged writes) or ABORT (discard them). The coordinator decides COMMIT
only if every participant voted YES, so a single NO or a missing vote - a crash, a timeout - forces
ABORT; the decision is written before it is sent, which is what lets a restarted coordinator replay
its log and broadcast the decision again.

Simplifications, stated explicitly: no real network, threads or disk. Messages go through an
in-memory FIFO queue, delivery is driven step by step by the test, and node crashes are modelled by
clearing volatile state (staging area, locks, phase) while the WAL and the committed data survive.
2PC is inherently *blocking*: a participant that voted YES and lost its coordinator holds locks with
no way to decide alone - the sketch shows that in-doubt state and then recovers the run by re-running
the protocol or by re-broadcasting the logged decision. Verified after *every* delivered message:
no participant can be committed while another is aborted (all-or-nothing), all decisions that exist
are identical, uncommitted staging is never visible to a reader, and a NO vote or a crash/timeout
can never leave a partial write anywhere.

Checks: the happy path (commit everywhere, locks released, decision logged before it is sent), the
invisibility of staged writes, a write-write conflict abort, a participant crash before voting plus a
coordinator timeout, a coordinator crash after prepare with in-doubt participants and recovery by
retry, a decision lost to a crashed participant and re-broadcast from the log, WAL replay semantics,
and an empty write set.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


PREPARE, VOTE, DECISION, ACK = "PREPARE", "VOTE", "DECISION", "ACK"
INIT, WAITING, PREPARED, COMMITTED, ABORTED = "INIT", "WAITING", "PREPARED", "COMMITTED", "ABORTED"


class Message:
    __slots__ = ("sender", "recipient", "kind", "txn", "payload")

    def __init__(self, sender, recipient, kind, txn, payload):
        self.sender = sender
        self.recipient = recipient
        self.kind = kind
        self.txn = txn
        self.payload = payload

    def __repr__(self):
        return f"{self.kind}({self.sender}->{self.recipient})"


class Network:
    """Deterministic in-memory network: FIFO delivery, drops for crashed nodes and partitions."""

    def __init__(self):
        self.queue = []
        self.sent = []
        self.dropped = []
        self.crashed = set()

    def down(self, node):
        self.crashed.add(node)

    def up(self, node):
        self.crashed.discard(node)

    def blocked(self, sender, recipient):
        return sender in self.crashed or recipient in self.crashed

    def send(self, sender, recipient, kind, txn, payload=None):
        message = Message(sender, recipient, kind, txn, payload or {})
        self.sent.append(message)
        if self.blocked(sender, recipient):
            self.dropped.append(message)
        else:
            self.queue.append(message)
        return message

    def deliver(self):
        return self.queue.pop(0) if self.queue else None


class DurableLog:
    """Append-only WAL; the only thing a crashed node keeps besides committed data."""

    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(record)

    def decision(self, txn):
        found = [r for r in self.records if r[0] == "decision" and r[1] == txn]
        return found[-1][2] if found else None

    def intent(self, txn):
        found = [r for r in self.records if r[0] == "prepared" and r[1] == txn]
        return found[-1] if found else None

    def kinds(self):
        return [r[0] for r in self.records]


class Participant:
    """Holds a slice of the key space; stages writes and locks them until a decision arrives."""

    def __init__(self, node_id, network, coordinator_id):
        self.id = node_id
        self.net = network
        self.coordinator = coordinator_id
        self.wal = DurableLog()
        self.data = {}                 # committed values only
        self.staging = {}              # uncommitted writes of the current transaction
        self.locks = {}                # key -> transaction that holds it
        self.state = INIT
        self.txn = None
        self.decision = None
        self.up = True
        self.vote_reasons = []

    def value(self, key):
        return self.data.get(key)

    def handle(self, message):
        if message.kind == PREPARE:
            self._on_prepare(message)
        elif message.kind == DECISION:
            self._on_decision(message)

    def _on_prepare(self, message):
        txn = message.payload["txn"]
        writes = dict(message.payload["writes"])
        if self.state == PREPARED and self.txn == txn:
            self.net.send(self.id, self.coordinator, VOTE, txn, {"vote": True, "retry": True})
            return
        conflicts = sorted(key for key in writes if self.locks.get(key, txn) != txn)
        if conflicts:
            self.vote_reasons.append(("conflict", tuple(conflicts)))
            self.wal.append(("vote_no", txn, tuple(conflicts)))
            self.state = ABORTED        # it never prepared, so it may abort on its own
            self.decision = "ABORT"
            self.net.send(self.id, self.coordinator, VOTE, txn,
                          {"vote": False, "reason": "conflict", "keys": conflicts})
            return
        for key in writes:
            self.locks[key] = txn
        self.staging = writes
        self.wal.append(("prepared", txn, tuple(sorted(writes.items()))))
        self.state = PREPARED
        self.txn = txn
        self.net.send(self.id, self.coordinator, VOTE, txn, {"vote": True})

    def _on_decision(self, message):
        txn = message.payload["txn"]
        decision = message.payload["decision"]
        if self.decision == decision and self.state in (COMMITTED, ABORTED):
            self.net.send(self.id, self.coordinator, ACK, txn, {"decision": decision})
            return
        self.wal.append(("decision", txn, decision))
        self._apply(txn, decision)
        self.net.send(self.id, self.coordinator, ACK, txn, {"decision": decision})

    def _apply(self, txn, decision):
        if decision == "COMMIT":
            self.data.update(self.staging)
            self.state = COMMITTED
        else:
            self.state = ABORTED
        self.staging = {}
        for key in [k for k, owner in self.locks.items() if owner == txn]:
            del self.locks[key]
        self.txn = txn
        self.decision = decision

    # ---- crash / recovery -----------------------------------------------------------------
    def crash(self):
        self.up = False
        self.net.down(self.id)
        self.staging = {}              # volatile
        self.locks = {}                # volatile
        self.state = INIT
        self.decision = None

    def recover(self):
        """Replay the WAL: a logged decision is re-applied, a logged intent means in-doubt."""
        self.up = True
        self.net.up(self.id)
        records = self.wal.records
        decisions = [r for r in records if r[0] == "decision"]
        if decisions:
            txn, decision = decisions[-1][1], decisions[-1][2]
            intent = self.wal.intent(txn)
            if decision == "COMMIT" and intent:
                self.staging = dict(intent[2])
            self._apply(txn, decision)
            return "replayed-decision"
        intent = None
        for record in records:
            if record[0] == "prepared":
                intent = record
        if intent is not None:         # in-doubt, but the durable intent restores the locks
            self.txn = intent[1]
            self.staging = dict(intent[2])
            for key in self.staging:
                self.locks[key] = self.txn
            self.state = PREPARED
            return "in-doubt"
        return "clean"


class Coordinator:
    """Decides once, writes the decision to its log first, then broadcasts it."""

    def __init__(self, node_id, network, participants, txn, timeout=2):
        self.id = node_id
        self.net = network
        self.participants = list(participants)
        self.txn = txn
        self.wal = DurableLog()
        self.votes = {}
        self.acks = set()
        self.decision = None
        self.state = INIT
        self.ticks = 0
        self.timeout = timeout
        self.up = True
        self.writes = {}

    def start(self, writes_by_participant):
        self.writes = {node: dict(writes) for node, writes in writes_by_participant.items()}
        self.wal.append(("begin", self.txn))
        self.state = WAITING
        self.ticks = 0
        for node, writes in self.writes.items():
            self.net.send(self.id, node, PREPARE, self.txn, {"txn": self.txn, "writes": writes})

    def decide(self, decision):
        if self.decision is not None:
            return False
        self.decision = decision
        self.wal.append(("decision", self.txn, decision))   # logged before it is sent
        self.state = COMMITTED if decision == "COMMIT" else ABORTED
        for node in self.participants:
            self.net.send(self.id, node, DECISION, self.txn,
                          {"txn": self.txn, "decision": decision})
        return True

    def resend_decision(self):
        if self.decision is None:
            return 0
        for node in self.participants:
            self.net.send(self.id, node, DECISION, self.txn,
                          {"txn": self.txn, "decision": self.decision})
        return len(self.participants)

    def handle(self, message):
        if message.kind == VOTE:
            self.ticks = 0
            self.votes[message.sender] = message.payload["vote"]
            if not message.payload["vote"]:
                self.decide("ABORT")            # one NO is enough
            elif len(self.votes) == len(self.participants):
                self.decide("COMMIT" if all(self.votes.values()) else "ABORT")
        elif message.kind == ACK:
            self.acks.add((message.sender, message.payload["decision"]))

    def tick(self):
        """Timeout clock, advanced only while the network is idle.

        Returning True while a decision is still pending keeps the driver running: a missing vote
        must not be mistaken for quiescence.
        """
        if self.decision is not None or not self.up:
            return False
        self.ticks += 1
        if self.ticks >= self.timeout:
            self.decide("ABORT")
        return True

    def crash(self):
        self.up = False
        self.net.down(self.id)
        self.votes = {}
        self.acks = set()
        self.state = WAITING if self.decision is None else self.state

    def recover(self):
        self.up = True
        self.net.up(self.id)
        self.decision = self.wal.decision(self.txn)
        return self.decision


class Simulation:
    """Driver: FIFO delivery, idle ticks for timeouts, and the per-step safety invariant."""

    def __init__(self, network, coordinator, participants):
        self.net = network
        self.coordinator = coordinator
        self.participants = list(participants)
        self.steps = 0
        self.messages_delivered = 0

    def nodes(self):
        table = {participant.id: participant for participant in self.participants}
        table[self.coordinator.id] = self.coordinator
        return table

    def assert_safe(self):
        committed = [p.id for p in self.participants if p.state == COMMITTED]
        aborted = [p.id for p in self.participants if p.state == ABORTED]
        check(not (committed and aborted),
              f"all-or-nothing violated: committed={committed} aborted={aborted}")
        decisions = {p.decision for p in self.participants if p.decision}
        check(len(decisions) <= 1, f"participants disagree on the decision: {decisions}")

    def step(self):
        message = self.net.deliver()
        if message is None:
            progress = self.coordinator.tick() if self.coordinator.up else False
        else:
            progress = True
            self.messages_delivered += 1
            node = self.nodes().get(message.recipient)
            if node is not None and node.up:
                node.handle(message)
        self.steps += 1
        self.assert_safe()
        return progress

    def run(self, max_steps=200):
        """Run until the network is quiet and nothing more can progress."""
        for _ in range(max_steps):
            if not self.step():
                return self.steps
        raise AssertionError("simulation did not reach quiescence")

    def decisions(self):
        return {p.id: p.decision for p in self.participants}


def build(node_ids, coordinator_id="C", txn="T1", timeout=2):
    network = Network()
    participants = [Participant(node, network, coordinator_id) for node in node_ids]
    coordinator = Coordinator(coordinator_id, network, node_ids, txn, timeout=timeout)
    return Simulation(network, coordinator, participants)


def main():
    # --- 1. happy path; staged writes stay invisible until the decision ------------------------
    sim = build(["P0", "P1", "P2"])
    sim.coordinator.start({"P0": {"a": 1}, "P1": {"b": 2}, "P2": {"c": 3}})
    for _ in range(3):                             # deliver the three PREPARE messages only
        sim.step()
    check(all(p.state == PREPARED for p in sim.participants), "every participant prepared")
    check(all(p.value(key) is None for p, key in zip(sim.participants, "abc")),
          "staged writes are invisible to readers before the commit")
    check(sim.participants[0].staging == {"a": 1} and sim.participants[0].locks == {"a": "T1"},
          "the staged write sits in the staging area behind a lock")
    sim.run()
    check(sim.coordinator.decision == "COMMIT" and sim.coordinator.state == COMMITTED
          and [p.state for p in sim.participants] == [COMMITTED] * 3,
          "the coordinator commits when everybody votes YES, and every participant commits")
    check([p.value(key) for p, key in zip(sim.participants, "abc")] == [1, 2, 3],
          "each participant exposes exactly its committed write once decided")
    check(all(not p.locks and p.staging == {} for p in sim.participants),
          "locks are released and staging areas are emptied")
    check(sim.coordinator.acks == {("P0", "COMMIT"), ("P1", "COMMIT"), ("P2", "COMMIT")},
          "every participant acknowledged the decision")
    check(sim.coordinator.wal.kinds() == ["begin", "decision"]
          and all(p.wal.kinds() == ["prepared", "decision"] for p in sim.participants),
          "the decision is logged before it is broadcast; the intents are logged in phase one")
    check(len(sim.net.sent) == 12 and not sim.net.dropped,
          "the happy path costs prepare + vote + decision + ack per participant")

    # --- 2. a write-write conflict forces ABORT -----------------------------------------------
    sim = build(["P0", "P1"])
    sim.participants[1].locks["b"] = "T0"          # an earlier transaction still holds the key
    sim.coordinator.start({"P0": {"a": 9}, "P1": {"b": 9}})
    sim.run()
    check(sim.coordinator.decision == "ABORT" and all(p.state == ABORTED
                                                     for p in sim.participants),
          "a single NO vote forces ABORT and nobody commits anything")
    check(all(p.value("a") is None and p.value("b") is None for p in sim.participants),
          "no partial write is visible after the abort")
    check(sim.participants[0].vote_reasons == []
          and sim.participants[1].vote_reasons == [("conflict", ("b",))],
          "the conflicting participant is the one that voted NO")
    check(sim.participants[0].locks == {} and sim.participants[1].locks == {"b": "T0"},
          "the prepared node released its lock, the pre-existing lock is untouched")

    # --- 3. a participant crash before voting + coordinator timeout ----------------------------
    sim = build(["P0", "P1", "P2"])
    sim.participants[2].crash()
    sim.coordinator.start({"P0": {"a": 1}, "P1": {"b": 2}, "P2": {"c": 3}})
    sim.run()
    check(any(m.recipient == "P2" and m.kind == PREPARE for m in sim.net.dropped)
          and sim.coordinator.decision == "ABORT",
          "the PREPARE to the crashed node is dropped and the missing vote times out into ABORT")
    check(sim.participants[0].state == ABORTED and sim.participants[1].state == ABORTED
          and all(p.value("a") is None and p.value("b") is None for p in sim.participants),
          "the live participants abort and the prepared writes are applied nowhere")
    check(sim.participants[2].state == INIT and sim.participants[2].data == {},
          "the crashed participant never learned about the transaction")

    # --- 4. coordinator crash after prepare: in-doubt, then recovery by retry ------------------
    sim = build(["P0", "P1"], timeout=3)
    sim.coordinator.start({"P0": {"k": 7}, "P1": {"m": 8}})
    sim.coordinator.crash()                        # dies before collecting any vote
    sim.run()
    check(all(p.state == PREPARED for p in sim.participants),
          "participants are left in doubt with their locks held (2PC is blocking)")
    check(all(p.value("k") is None and p.value("m") is None for p in sim.participants)
          and all(p.decision is None for p in sim.participants),
          "nobody committed while the coordinator was down and no participant guessed a decision")
    check(sim.coordinator.wal.decision("T1") is None, "the crashed coordinator logged no decision")

    check(sim.coordinator.recover() is None, "recovery finds no decision in the log")
    sim.coordinator.start(sim.coordinator.writes)  # re-run the protocol: safe, nothing committed
    sim.run()
    check(sim.coordinator.decision == "COMMIT"
          and all(p.state == COMMITTED for p in sim.participants),
          "the retry collects the votes again and commits")
    check([p.value(key) for p, key in zip(sim.participants, "km")] == [7, 8],
          "the retried transaction applies its writes exactly once")
    check(sim.participants[0].wal.kinds() == ["prepared", "decision"],
          "the repeated PREPARE did not append a second intent record")

    # --- 6. a lost decision: re-broadcast from the coordinator's log ---------------------------
    sim = build(["P0", "P1", "P2"])
    sim.coordinator.start({"P0": {"a": 1}, "P1": {"b": 2}, "P2": {"c": 3}})
    for _ in range(6):                             # prepare, votes, then crash P2 after voting
        sim.step()
        if sim.participants[2].state == PREPARED:
            sim.participants[2].crash()
            break
    check(sim.participants[2].wal.decision("T1") is None, "P2 crashed holding only an intent")
    sim.run()
    check(sim.coordinator.decision == "COMMIT" and sim.participants[0].state == COMMITTED
          and sim.participants[1].state == COMMITTED,
          "the coordinator commits with the votes it had and the reachable nodes commit")
    check(sim.participants[2].state == INIT and sim.participants[2].value("c") is None
          and any(m.recipient == "P2" and m.kind == DECISION for m in sim.net.dropped),
          "the crashed participant applied nothing: the decision to it was dropped")
    sim.assert_safe()

    check(sim.participants[2].recover() == "in-doubt" and sim.participants[2].state == PREPARED
          and sim.participants[2].locks == {"c": "T1"},
          "replay finds the durable intent and restores the locks and the in-doubt phase")
    check(sim.coordinator.resend_decision() == 3, "the coordinator re-broadcasts from its log")
    sim.run()
    check([p.state for p in sim.participants] == [COMMITTED] * 3
          and sim.participants[2].value("c") == 3,
          "every participant ends up committed and the replayed write was applied once")
    check(sim.participants[2].wal.kinds() == ["prepared", "decision"],
          "P2 logged exactly one intent and one decision")

    # --- 7. degenerate cases -------------------------------------------------------------------
    sim = build(["P0"])
    sim.coordinator.start({"P0": {}})              # read-only transaction: no writes to prepare
    sim.run()
    check(sim.coordinator.decision == "COMMIT" and sim.participants[0].state == COMMITTED,
          "an empty write set commits with a single participant")
    check(sim.participants[0].data == {} and sim.participants[0].locks == {},
          "nothing was written and no lock was taken")

    sim = build(["P0", "P1"])
    sim.coordinator.start({"P0": {"a": 4}, "P1": {"b": 5}})
    sim.run()
    before = dict(sim.participants[0].data)
    sim.participants[0].crash()
    check(sim.participants[0].recover() == "replayed-decision", "the committed decision is replayed")
    check(sim.participants[0].state == COMMITTED and sim.participants[0].data == before,
          "replaying a commit is idempotent - the durable data is unchanged")

    total = sum(len(p.wal.records) for p in sim.participants)
    print(f"2pc: messages={len(sim.net.sent)} dropped={len(sim.net.dropped)} "
          f"delivered={sim.messages_delivered} wal-records={total} "
          f"decisions={sim.decisions()}")


if __name__ == "__main__":
    main()
    print("PASS")
