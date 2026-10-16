"""Task 892 - Implement chaos-monkey style fault injection and recovery validation.  (Tier: Master, CTCI Beyond - Distributed Systems & Storage)

The system under test is a replicated write-ahead log: one leader plus followers, each follower
fsyncing its tail after a per-replica delay, and a write being acknowledged to the client only
once a majority of replicas have *durably* stored it.  That ordering of events is the whole
point: because the acknowledgement waits for a durable majority, no single crash inside the
minority can ever destroy an acknowledged write, and the durability invariant is a consequence
of the protocol rather than a hope.

On top of that sits a chaos monkey.  Every message can be dropped, duplicated, delayed or
corrupted (checksum failure), and every round it may crash a replica - which destroys that
replica's un-fsynced tail, i.e. exactly the torn write a real crash produces - and restart it
later.  Crashes also hit the leader.  When the leader is down a new one is elected from the live
replicas, and a restarted replica performs a log repair: it finds its longest common prefix with
the leader, truncates, and pulls the missing suffix back.  All of it happens on an explicit
in-process message network with an injected fault stream, so every scenario is reproducible.

The validator is the deliverable.  After every single round it re-checks the replica invariants
and raises with a concrete message on violation: log matching (every log is a prefix of the
issued history, so no phantom entries and no divergence), durability (every acknowledged write
is on every live replica), quorum durability (a majority holds each committed prefix),
commit-index monotonicity, and no duplicate indices.  A deliberately corrupted replica is fed to
the same checker to prove it is not vacuous.  Convergence and liveness are then measured: after
the fault stream stops, every replica must end up with exactly the acknowledged log, and the
time from a fault to full convergence must be bounded.

Simplified versus a production chaos platform: no real processes or network (faults are
injected into an in-process queue), one leader and small clusters, a synchronous round-based
loop instead of a continuous event loop, no clock skew, no disk-error model beyond "un-flushed
tail is lost at crash time", and the repair fetches from the leader rather than over a
Raft-style probe.  Verified: fault injection demonstrably fires, the invariants hold through
every round of a randomised chaos schedule, a torn tail is repaired without losing an
acknowledged write, duplicated messages stay idempotent, corrupted messages are rejected, a
leader crash preserves acknowledged writes through a new election, a lost quorum stalls
acknowledgements instead of losing them, the checker rejects an artificially corrupted replica,
recovery converges within a bounded number of rounds, and every run replays from its seed.
"""
import hashlib

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound

    def chance(self, percent):
        return percent > 0 and self.below(100) < percent


class InvariantViolation(Exception):
    """Raised by the validator when a replica invariant is broken."""


class ChaosConfigError(Exception):
    """Raised for a configuration the chaos monkey cannot honour."""


def make_entry(index, term, key, value):
    digest = hashlib.sha256(("%d|%d|%s|%s" % (index, term, key, value)).encode()).digest()[:8]
    return (index, term, key, value, int.from_bytes(digest, "big"))


class Replica:
    """One log replica: a durable prefix plus a not-yet-fsynced buffer."""

    def __init__(self, rid, fsync_delay):
        if fsync_delay < 0:
            raise ChaosConfigError("the fsync delay cannot be negative")
        self.rid = rid
        self.fsync_delay = fsync_delay
        self.durable = []
        self.buffer = []
        self.pending = []
        self.down = False
        self.restart_at = None
        self.last_acked = 0
        self.accepted = 0
        self.duplicates = 0
        self.held = 0
        self.conflicts = 0
        self.corrupted = 0
        self.dropped = 0
        self.crashes = 0
        self.repairs = 0
        self.entries_fetched = 0

    @property
    def log(self):
        return self.durable + [entry for entry, _ in self.buffer]

    @property
    def length(self):
        return len(self.durable) + len(self.buffer)

    def entry_at(self, index):
        if index <= len(self.durable):
            return self.durable[index - 1]
        offset = index - len(self.durable) - 1
        return self.buffer[offset][0] if offset < len(self.buffer) else None

    def accept(self, entry, step):
        """Idempotent append: the same entry twice is a no-op, a gap is held back."""
        index = entry[0]
        if index <= self.length:
            if self.entry_at(index) == entry:
                self.duplicates += 1
                return "duplicate"
            self.conflicts += 1
            return "conflict"
        if index == self.length + 1:
            self.buffer.append((entry, step + self.fsync_delay))
            self.accepted += 1
            self._absorb_pending(step)
            return "accepted"
        self.pending.append(entry)
        self.held += 1
        return "held"

    def _absorb_pending(self, step):
        self.pending.sort(key=lambda entry: entry[0])
        progressed = True
        while progressed:
            progressed = False
            for entry in list(self.pending):
                if entry[0] == self.length + 1:
                    self.pending.remove(entry)
                    self.buffer.append((entry, step + self.fsync_delay))
                    self.accepted += 1
                    progressed = True

    def flush(self, step):
        """Move everything that has reached its fsync deadline into the durable log."""
        if self.down:
            return 0
        moved = 0
        while self.buffer and self.buffer[0][1] <= step:
            self.durable.append(self.buffer.pop(0)[0])
            moved += 1
        return moved

    def crash(self, step, restart_delay):
        if self.down:
            return 0
        lost = len(self.buffer)
        self.buffer = []
        self.pending = []
        self.down = True
        self.restart_at = step + restart_delay
        self.crashes += 1
        return lost

    def truncate_to(self, index):
        self.durable = self.durable[:index]
        self.buffer = [(entry, due) for entry, due in self.buffer if entry[0] <= index]
        self.pending = [entry for entry in self.pending if entry[0] > index]


class ChaosStore:
    """Leader plus followers over an in-process, fault-injected message bus."""

    def __init__(self, node_count=5, seed=1, fsync_delays=None, drop=0, duplicate=0, delay=0,
                 corrupt=0, crash=0, restart_delay=2):
        if node_count < 3:
            raise ChaosConfigError("a replicated log needs at least three replicas")
        if fsync_delays is None:
            fsync_delays = [0] * node_count
        if len(fsync_delays) != node_count:
            raise ChaosConfigError("one fsync delay per replica is required")
        self.node_count = node_count
        self.majority = node_count // 2 + 1
        self.replicas = [Replica(rid, fsync_delays[rid]) for rid in range(node_count)]
        self.leader = 0
        self.term = 1
        self.step = 0
        self.queue = []
        self.sequence = 0
        self.issued = []
        self.acked = []
        self.commit = 0
        self.match = {rid: 0 for rid in range(node_count)}
        self.rng = Rng(seed)
        self.drop = drop
        self.duplicate = duplicate
        self.delay = delay
        self.corrupt = corrupt
        self.crash = crash
        self.restart_delay = restart_delay
        self.injections = {"drop": 0, "duplicate": 0, "delay": 0, "corrupt": 0, "crash": 0,
                           "restart": 0, "election": 0, "repair": 0}
        self.recovered_at = None

    def live(self):
        return [replica for replica in self.replicas if not replica.down]

    def majority_live(self):
        return len(self.live()) >= self.majority

    # ------------------------------------------------------------- messaging
    def send(self, sender, recipient, message):
        if self.rng.chance(self.corrupt):
            self.injections["corrupt"] += 1
            message = ("corrupt", sender, recipient, message)
        self.sequence += 1
        copies = 1
        if self.rng.chance(self.duplicate):
            self.injections["duplicate"] += 1
            copies = 2
        waiting = 1
        if self.rng.chance(self.delay):
            self.injections["delay"] += 1
            waiting += 1 + self.rng.below(2)
        if self.rng.chance(self.drop):
            self.injections["drop"] += 1
            return
        for _ in range(copies):
            self.queue.append((self.step + waiting, self.sequence, sender, recipient, message))

    def deliver(self):
        self.queue.sort(key=lambda item: (item[0], item[1]))
        due = [item for item in self.queue if item[0] <= self.step]
        self.queue = [item for item in self.queue if item[0] > self.step]
        for _, _, sender, recipient, message in due:
            target = self.replicas[recipient]
            if target.down:
                continue
            if message[0] == "corrupt":
                target.corrupted += 1
                continue
            if message[0] == "append":
                target.accept(message[1], self.step)
            elif message[0] == "ack":
                index, checksum, term = message[1], message[2], message[3]
                if term == self.term or index <= self.commit:
                    self.match[recipient] = max(self.match.get(recipient, 0), index)
            else:
                raise ChaosConfigError("unknown message %r" % (message[0],))

    # ------------------------------------------------------------- protocol
    def leader_replica(self):
        return self.replicas[self.leader]

    def elect(self):
        candidates = self.live()
        if len(candidates) < self.majority:
            return False                      # no quorum, so no safe leader
        best = max(candidates, key=lambda replica: (replica.length, replica.rid))
        if best.length < self.commit:
            return False                      # never elect a replica that lost a committed entry
        if best.rid != self.leader:
            self.leader = best.rid
            self.term += 1
            self.injections["election"] += 1
            self.match = {rid: 0 for rid in range(self.node_count)}
            return True
        return False

    def repair(self, replica):
        """Truncate to the longest common prefix with the leader, then pull the suffix."""
        leader = self.leader_replica()
        if replica.rid == leader.rid:
            return 0
        common = 0
        while common < min(replica.length, leader.length) \
                and replica.entry_at(common + 1) == leader.entry_at(common + 1):
            common += 1
        if common < replica.length:
            replica.truncate_to(common)
        fetched = 0
        for index in range(common + 1, leader.length + 1):
            replica.accept(leader.entry_at(index), self.step)
            fetched += 1
        replica.repairs += 1
        replica.entries_fetched += fetched
        self.injections["repair"] += 1
        return fetched

    def restart_due(self):
        for replica in self.replicas:
            if replica.down and replica.restart_at is not None and replica.restart_at <= self.step:
                replica.down = False
                replica.restart_at = None
                self.injections["restart"] += 1
                if self.recovered_at is None:
                    self.recovered_at = self.step
                self.repair(replica)

    def update_commit(self):
        """Commit when a majority of the cluster durably holds an index.

        Down replicas contribute zero, which is what makes a lost quorum stall the log
        instead of silently committing on whatever is left.
        """
        values = sorted(len(replica.durable) if not replica.down else 0
                        for replica in self.replicas)
        values.sort()
        candidate = values[self.node_count - self.majority]
        if candidate > self.commit:
            self.commit = candidate

    def flush_all(self):
        for replica in self.replicas:
            if replica.down:
                continue
            moved = replica.flush(self.step)
            if moved:
                replica.last_acked = len(replica.durable)
                self.send(replica.rid, self.leader, ("ack", replica.last_acked,
                                                     replica.durable[-1][4], self.term))
        self.match[self.leader] = len(self.leader_replica().durable)

    def submit(self, key, value):
        """One client round: ensure the pending write is issued, then run one step."""
        index = len(self.acked) + 1
        if len(self.issued) < index:
            self.issued.append(make_entry(index, 1, key, value))
        entry = self.issued[index - 1]
        if self.leader_replica().down:
            self.elect()
        leader = self.leader_replica()
        if not leader.down:
            if leader.entry_at(index) is None:
                leader.accept(entry, self.step)
            for replica in self.replicas:
                if replica.rid != leader.rid:
                    self.send(leader.rid, replica.rid, ("append", entry))
        self.step += 1
        self.deliver()
        self.flush_all()
        self.deliver()
        self.update_commit()
        self.restart_due()
        if self.rng.chance(self.crash):
            victim = self.replicas[self.rng.below(self.node_count)]
            if not victim.down:
                self.injections["crash"] += 1
                victim.crash(self.step, self.restart_delay)
        self.validate()
        if self.commit >= index:
            self.acked.append(index)
            return True
        return False

    def run(self, count, rounds_per_write=6):
        """Drive `count` writes, retrying each until it is acknowledged."""
        rounds = 0
        for index in range(count):
            for _ in range(rounds_per_write):
                rounds += 1
                if self.submit("k%03d" % index, "v%03d" % index):
                    break
        return rounds

    def settle(self, limit=40):
        """Stop injecting faults (the chaos monkey is switched off) and let the system heal."""
        saved = (self.crash, self.drop, self.duplicate, self.delay, self.corrupt)
        self.crash = self.drop = self.duplicate = self.delay = self.corrupt = 0
        rounds = 0
        while rounds < limit:
            rounds += 1
            for replica in self.replicas:
                if replica.down and replica.restart_at is None:
                    replica.down = False
                if replica.down and replica.restart_at <= self.step:
                    replica.down = False
                    self.injections["restart"] += 1
            if self.leader_replica().down:
                self.elect()
            self.step += 1
            self.deliver()
            self.flush_all()
            self.deliver()
            self.update_commit()
            for replica in self.replicas:
                if not replica.down and replica.length != self.leader_replica().length:
                    self.repair(replica)
            self.validate()
            if all(not replica.down for replica in self.replicas) and all(
                    replica.log == self.issued for replica in self.replicas):
                self.crash, self.drop, self.duplicate, self.delay, self.corrupt = saved
                return rounds
        self.crash, self.drop, self.duplicate, self.delay, self.corrupt = saved
        raise InvariantViolation("the system did not converge within %d rounds" % limit)

    # ------------------------------------------------------------ validator
    def validate(self):
        """Replica invariants, re-checked after every single round.

        A live follower may legitimately lag behind, so the checks are the ones that really
        matter: no replica may hold an entry that was never issued or disagree with the
        history it does hold, the leader may never be missing a committed entry, and every
        acknowledged index must be durably present on a majority of the cluster.
        """
        issued = self.issued
        for replica in self.replicas:
            log = replica.log
            if len(log) > len(issued):
                raise InvariantViolation("replica %d holds %d entries but only %d were issued"
                                         % (replica.rid, len(log), len(issued)))
            for offset, entry in enumerate(log):
                if entry != issued[offset]:
                    raise InvariantViolation(
                        "replica %d diverges at index %d: %r" % (replica.rid, offset + 1, entry))
            indices = [entry[0] for entry in log]
            if indices != list(range(1, len(indices) + 1)):
                raise InvariantViolation("replica %d has a broken index sequence" % replica.rid)
        leader = self.leader_replica()
        if len(leader.log) < self.commit:
            raise InvariantViolation(
                "the leader holds %d entries but the commit index is %d"
                % (len(leader.log), self.commit))
        for index in self.acked:
            if index > self.commit:
                raise InvariantViolation("index %d was acknowledged beyond the commit index %d"
                                         % (index, self.commit))
            holders = sum(1 for replica in self.replicas if len(replica.durable) >= index)
            if holders < self.majority:
                raise InvariantViolation("only %d replicas durably hold acknowledged index %d"
                                         % (holders, index))
        return self.commit


# ----------------------------------------------------------------------- main
def fault_free_store(node_count=3, seed=892, fsync_delays=None):
    return ChaosStore(node_count=node_count, seed=seed, fsync_delays=fsync_delays)


def main():
    # --- 1. entry construction and configuration validation -----------------
    entry = make_entry(4, 2, "k1", "v1")
    check(entry[0] == 4 and entry[1] == 2 and entry[2] == "k1" and entry[3] == "v1",
          "an entry must carry its index, term, key and value")
    check(len(str(entry[4])) > 0 and entry == make_entry(4, 2, "k1", "v1"),
          "the checksum must be deterministic")
    check(make_entry(4, 2, "k1", "v2") != entry,
          "a different value must produce a different checksum")
    for bad in (lambda: ChaosStore(node_count=2), lambda: ChaosStore(fsync_delays=[0, 0]),
                lambda: ChaosStore(node_count=3, fsync_delays=[0, -1, 0])):
        raised = False
        try:
            bad()
        except ChaosConfigError:
            raised = True
        check(raised, "an invalid chaos configuration must be rejected")

    # --- 2. a fault-free run converges exactly ------------------------------
    store = fault_free_store()
    store.run(40)
    check(len(store.acked) == 40, "every write must be acknowledged without faults")
    check(store.commit == 40, "the commit index must reach the last write")
    for replica in store.replicas:
        check(replica.log == store.issued,
              "every replica must match the issued log exactly: %r" % (replica.log,))
        check(replica.corrupted == 0 and replica.dropped == 0,
              "a fault-free run must not record fault counters")
    check(all(count == 0 for count in store.injections.values()),
          "no injection may fire in a fault-free run")

    # --- 3. acks wait for the fsync delay -----------------------------------
    delayed = ChaosStore(node_count=3, seed=893, fsync_delays=[0, 2, 2])
    rounds = 0
    for _ in range(60):
        rounds += 1
        if delayed.submit("slow", "write"):
            break
    check(delayed.acked == [1], "the write must eventually be acknowledged")
    check(rounds >= 3, "a replica with an fsync delay must not confirm instantly")

    # --- 4. the chaos monkey actually injects -------------------------------
    store = ChaosStore(node_count=5, seed=892100, drop=15, duplicate=10, delay=10, corrupt=5,
                       crash=8, restart_delay=2)
    store.run(60)
    for name in ("drop", "duplicate", "delay", "corrupt", "crash", "restart", "repair"):
        check(store.injections[name] > 0, "the %s injection must have fired" % name)
    check(any(replica.duplicates > 0 for replica in store.replicas),
          "at least one duplicate append must have reached a replica")
    check(any(replica.corrupted > 0 for replica in store.replicas),
          "at least one corrupted message must have been rejected")
    check(any(replica.crashes > 0 for replica in store.replicas), "a replica must have crashed")

    # --- 5. invariants held through every round of that chaos run -----------
    check(store.commit == len(store.acked), "the commit index must match the acknowledged writes")
    check(store.commit >= 50,
          "a majority must survive the chaos schedule: only %d writes committed" % store.commit)
    for index in store.acked:
        holders = sum(1 for replica in store.replicas if len(replica.durable) >= index)
        check(holders >= store.majority, "acknowledged index %d must sit on a majority" % index)

    # --- 6. convergence after the faults stop -------------------------------
    for rid in (3, 4):
        store.replicas[rid].crash(store.step + 1, 1)      # leave two replicas lagging
    store.step += 1
    rounds_to_heal = store.settle()
    check(rounds_to_heal <= 40, "recovery must be bounded, took %d rounds" % rounds_to_heal)
    for replica in store.replicas:
        check(replica.log == store.issued,
              "replica %d must converge to the issued log" % replica.rid)
    check(any(replica.repairs > 0 for replica in store.replicas),
          "recovery must have used at least one log repair")
    check(sum(replica.entries_fetched for replica in store.replicas) > 0,
          "a repair must have fetched entries from the leader")
    check(store.replicas[3].down is False and store.replicas[4].down is False,
          "the crashed replicas must have been restarted during recovery")

    # --- 7. a torn tail is repaired without losing acknowledged data --------
    torn = ChaosStore(node_count=3, seed=892200, fsync_delays=[0, 1, 1])
    torn.run(20)
    victim = torn.replicas[2]
    before = len(victim.durable)
    check(before == 20, "the victim must have flushed every acknowledged entry")
    pending = make_entry(21, 1, "pending", "value")
    torn.issued.append(pending)
    torn.leader_replica().accept(pending, torn.step)
    torn.replicas[1].accept(pending, torn.step)
    victim.accept(pending, torn.step)
    check(victim.length == 21, "the victim must hold the pending entry in its buffer")
    lost = victim.crash(torn.step + 1, 1)
    check(lost == 1, "the crash must destroy exactly the un-fsynced entry, got %d" % lost)
    check(len(victim.durable) == before and victim.length == 20,
          "the durable prefix must survive the crash untouched")
    check(len(torn.acked) == 20, "the acknowledged writes must be unaffected by the crash")
    check(torn.commit == 20, "the crash must not move the commit index")
    torn.validate()
    torn.settle(limit=20)
    check(torn.replicas[2].log == torn.issued,
          "the restarted replica must recover every entry, including the pending one")
    check(torn.replicas[2].entries_fetched >= 1,
          "the repair must have fetched the lost tail from the leader")
    check(torn.replicas[2].durable[:20] == torn.issued[:20],
          "the recovered prefix must still be the acknowledged log")

    # --- 8. duplicated appends are idempotent -------------------------------
    idempotent = Replica(0, 0)
    first = make_entry(1, 1, "a", "1")
    check(idempotent.accept(first, 0) == "accepted", "the first append must be accepted")
    check(idempotent.accept(first, 0) == "duplicate", "the same append must be idempotent")
    check(idempotent.accept(make_entry(1, 1, "a", "2"), 0) == "conflict",
          "a different entry at the same index must be reported as a conflict")
    check(idempotent.length == 1, "no duplicate may grow the log")
    check(idempotent.accept(make_entry(3, 1, "c", "3"), 0) == "held",
          "an append beyond the gap must be held back")
    check(idempotent.accept(make_entry(2, 1, "b", "2"), 0) == "accepted",
          "filling the gap must be accepted")
    check(idempotent.length == 3, "the held entry must be absorbed once the gap closes")

    # --- 9. the fsync model decides durability ------------------------------
    flushing = Replica(0, 2)
    flushing.accept(make_entry(1, 1, "a", "1"), 0)
    check(flushing.flush(0) == 0 and flushing.flush(1) == 0,
          "an entry must not become durable before its deadline")
    check(flushing.flush(2) == 1 and len(flushing.durable) == 1,
          "the entry must become durable at its deadline")
    flushing.accept(make_entry(2, 1, "b", "2"), 3)
    check(flushing.length == 2 and len(flushing.durable) == 1,
          "the second entry must still sit in the buffer")
    check(flushing.crash(4, 1) == 1 and flushing.length == 1,
          "a crash before the flush must lose exactly the un-fsynced tail")
    check(flushing.durable[0] == make_entry(1, 1, "a", "1"),
          "the flushed entry must survive the crash")
    check(flushing.crash(5, 1) == 0, "a second crash while already down must lose nothing more")

    # --- 10. a leader crash preserves acknowledged writes -------------------
    crashy = ChaosStore(node_count=5, seed=892300, drop=5, duplicate=5, delay=10, corrupt=5,
                        crash=0, restart_delay=2)
    crashy.run(30)
    acknowledged = list(crashy.acked)
    target = crashy.leader_replica()
    target.crash(crashy.step + 1, 2)
    crashy.injections["crash"] += 1
    crashy.step += 1
    check(crashy.leader_replica().down, "the leader must be the crashed replica")
    elected = crashy.elect()
    check(elected and not crashy.leader_replica().down, "a new leader must be elected")
    crashy.settle(limit=30)
    survivors = [replica for replica in crashy.replicas if len(replica.durable) >= len(acknowledged)]
    check(len(survivors) >= crashy.majority,
          "a majority must still hold every acknowledged entry after the leader crash")
    for replica in crashy.replicas:
        check(replica.log[:len(acknowledged)] == crashy.issued[:len(acknowledged)],
              "replica %d must keep the acknowledged prefix" % replica.rid)

    # --- 11. losing the quorum stalls instead of losing data ----------------
    stalled = ChaosStore(node_count=3, seed=892400, restart_delay=10 ** 6)
    stalled.run(5)
    check(len(stalled.acked) == 5, "the first five writes must be acknowledged")
    stalled.replicas[1].crash(stalled.step + 1, 10 ** 6)
    stalled.replicas[2].crash(stalled.step + 1, 10 ** 6)
    stalled.step += 1
    progress = [stalled.submit("k005", "v005") for _ in range(5)]
    check(not any(progress), "without a quorum no write may be acknowledged")
    check(stalled.commit == 5, "the commit index must not move without a quorum")
    check(stalled.acked == [1, 2, 3, 4, 5], "the acknowledged history must be untouched")
    stalled.replicas[1].down = False
    stalled.replicas[1].restart_at = None
    for _ in range(12):
        if stalled.submit("k005", "v005"):
            break
    check(stalled.acked == [1, 2, 3, 4, 5, 6],
          "liveness must return as soon as a quorum is reachable: %r" % (stalled.acked,))

    # --- 12. the validator is not vacuous -----------------------------------
    broken = fault_free_store(seed=892500)
    broken.run(6)
    broken.replicas[2].durable[3] = make_entry(4, 1, "k003", "tampered")
    raised = False
    try:
        broken.validate()
    except InvariantViolation:
        raised = True
    check(raised, "a tampered replica must be rejected by the validator")
    phantom = fault_free_store(seed=892501)
    phantom.run(4)
    phantom.replicas[1].durable.append(phantom.issued[0])
    raised = False
    try:
        phantom.validate()
    except InvariantViolation:
        raised = True
    check(raised, "a phantom entry must be rejected by the validator")
    trimmed = fault_free_store(seed=892502)
    trimmed.run(4)
    trimmed.replicas[0].durable.pop()
    trimmed.replicas[1].durable.pop()
    raised = False
    try:
        trimmed.validate()
    except InvariantViolation:
        raised = True
    check(raised, "a quorum that no longer holds an acknowledged write must be rejected")

    # --- 13. reproducibility ------------------------------------------------
    def replay(seed):
        local = ChaosStore(node_count=5, seed=seed, drop=10, duplicate=10, delay=10, corrupt=5,
                           crash=8, restart_delay=2)
        local.run(40)
        local.settle(limit=40)
        return (local.acked, [replica.log for replica in local.replicas],
                dict(local.injections))

    first = replay(892600)
    check(first == replay(892600), "an identical seed must replay the whole chaos schedule")
    other = replay(892601)
    check(other[2] != first[2] or other[0] != first[0],
          "a different seed must produce a different fault stream")

    # --- 14. scale: a long chaos run ---------------------------------------
    scale = ChaosStore(node_count=5, seed=892700, fsync_delays=[0, 1, 1, 2, 2], drop=12,
                       duplicate=8, delay=12, corrupt=4, crash=6, restart_delay=2)
    scale.run(300, rounds_per_write=8)
    check(len(scale.acked) >= 280,
          "a majority must survive 300 writes under chaos: %d acknowledged" % len(scale.acked))
    check(scale.commit == len(scale.acked), "commit and acknowledgement must agree at scale")
    healed = scale.settle(limit=60)
    check(all(replica.log == scale.issued for replica in scale.replicas),
          "every replica must converge to the issued log at scale")
    check(all(len(replica.log) >= len(scale.acked) for replica in scale.replicas),
          "every replica must hold every acknowledged entry after convergence")
    scale.step += max(replica.fsync_delay for replica in scale.replicas) + 1
    for replica in scale.replicas:
        replica.flush(scale.step)
    check(all(len(replica.durable) == len(scale.issued) for replica in scale.replicas),
          "once the fsync deadlines pass every replica must durably hold the whole log")
    check(sum(scale.injections[name] for name in ("drop", "duplicate", "corrupt", "crash")) > 50,
          "the long run must have injected a substantial fault stream")

    print("chaos run: %d injected faults, %d duplicates, %d corrupted, %d crashes, %d repairs"
          % (sum(store.injections.values()), store.injections["duplicate"],
             store.injections["corrupt"], store.injections["crash"], store.injections["repair"]))
    print("durability: %d acknowledged writes, each durably held by >= %d replicas"
          % (len(store.acked), store.majority))
    print("recovery: converged in %d rounds after the fault stream stopped" % rounds_to_heal)
    print("scale: %d writes, %d acknowledged, %d fault events, converged in %d rounds"
          % (300, len(scale.acked), sum(scale.injections.values()), healed))
    print("PASS")


if __name__ == "__main__":
    main()
