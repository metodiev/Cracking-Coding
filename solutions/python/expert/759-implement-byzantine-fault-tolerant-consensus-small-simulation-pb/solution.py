"""Task 759 - Byzantine fault tolerant consensus: a small PBFT simulation.  (Tier: Expert, Ch 9 - System Design and Scalability)

PBFT tolerates f Byzantine replicas with n = 3f+1 replicas because any two quorums of 2f+1 replicas
must overlap in at least f+1 replicas, so at least one of the shared replicas is correct - and a
correct replica never signs (here: never accepts) two different values for the same sequence
number.  The simulation implements the three phases for a single client request:

    1. PRE-PREPARE  the primary (view mod n) assigns sequence number n and broadcasts the digest of
                    the request; a replica accepts it only from the current primary, only inside its
                    watermarks [low, high] and only if the digest equals the one *it* computed from
                    the client's request - a forged or equivocated pre-prepare is therefore ignored;
    2. PREPARE      every replica that accepted the pre-prepare broadcasts its own prepare and counts
                    prepares (including its own); at 2f+1 matching prepares it becomes *prepared* and
                    broadcasts a commit;
    3. COMMIT       at 2f+1 matching commits it executes the request and records the digest.

If a round quiesces without a commit, the replicas vote for the next view (`view-change`), the new
primary collects 2f+1 votes and starts the protocol again - the liveness fallback when the primary
is silent.  Message digests come from a deterministic splitmix-based toy hash, and the checks
recover the classic results instead of assuming them: the quorum-intersection lemma is verified
*exhaustively* over all quorum pairs and all Byzantine sets of size <= f, the safety/liveness bound
n >= 3f+1 is confirmed by exhaustion for f = 1..3, and the protocol is exercised with five fault
models (honest, silent, equivocating primary, digest-forging replica, replica that never got the
request) plus duplicate deliveries, watermark violations and a partition that makes the quorum
unreachable.

Simplified versus the paper: no view-change certificates or prepared-certificate transfer (a new
view simply restarts the protocol for the pending request), no checkpoints/garbage collection, no
signatures (messages are trusted to come from the replica they claim, as the model labels them),
one outstanding client request at a time, and execution is immediate.  Verified invariants: every
committed digest equals the digest of the client request, all correct replicas that commit a given
sequence number commit the same digest, a forged digest never reaches a commit, vote/prepare/commit
sets are idempotent under duplicate delivery, and the run is deterministic.
"""
from collections import deque
from itertools import combinations

MASK64 = (1 << 64) - 1
BAD_MARK = "#forged"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def digest_of(payload):
    """Deterministic toy digest of a request payload (message identity, no crypto claim)."""
    state = 0x243F6A8885A308D3
    for byte in payload:
        state = (state ^ byte) & MASK64
        state, mix = splitmix64(state)
        state = (state + mix) & MASK64
    return "%016x" % state


def quorums(nodes, size):
    return list(combinations(nodes, size))


def intersection_lemma(n, f):
    """The PBFT admission test: quorums of 2f+1 must survive f Byzantine nodes, and n-f >= 2f+1."""
    if n - f < 2 * f + 1:
        return False                        # not enough correct replicas to form a quorum
    for first in quorums(range(n), 2 * f + 1):
        for second in quorums(range(n), 2 * f + 1):
            if len(set(first) & set(second)) < f + 1:
                return False
    return True


def quorum_holds_against_every_byzantine_set(n, f):
    """Stronger form: for every quorum pair and every set B of <= f faulty nodes, one shared
    replica is correct - i.e. the shared part is never covered by the faulty set."""
    for first in quorums(range(n), 2 * f + 1):
        for second in quorums(range(n), 2 * f + 1):
            shared = set(first) & set(second)
            for faulty in quorums(range(n), f):
                if set(faulty) >= shared:
                    return False
    return True


class Replica:
    """One PBFT replica; `behavior` selects the fault model used by the checks."""

    def __init__(self, index, behavior, n, quorum, low, high):
        self.index = index
        self.behavior = behavior
        self.n = n
        self.quorum = quorum
        self.low = low
        self.high = high
        self.view = 0
        self.seq = 1
        self.digest = None
        self.state = {}                       # (view, seq, digest) -> {"prepares", "commits"}
        self.prepared = set()
        self.committed = {}                   # (view, seq) -> digest
        self.replies = 0
        self.sent = 0
        self.votes = set()

    # ------------------------------------------------------------------ plumbing
    def others(self):
        return [other for other in range(self.n) if other != self.index]

    def send(self, queue, message):
        if self.behavior == "silent" or self.behavior == "partitioned":
            return
        for recipient in self.others():
            queue.append((recipient, message))
            self.sent += 1

    def entry(self, view, seq, digest):
        return self.state.setdefault((view, seq, digest), {"prepares": set(), "commits": set()})

    def wire_digest(self, digest):
        """A forging replica puts a wrong digest on the wire (the peers must ignore it)."""
        return digest + BAD_MARK if self.behavior == "forge" else digest

    def on_request(self, queue):
        """The client broadcast the request to this replica; if it is the primary, it proposes."""
        self.propose(queue)

    def propose(self, queue):
        if self.digest is None or self.index != self.view % self.n:
            return
        entry = self.entry(self.view, self.seq, self.digest)
        for recipient in self.others():
            # an equivocating primary sends a different digest to one replica
            payload = self.digest + BAD_MARK if (self.behavior == "equivocate" and recipient == 1) \
                else self.digest
            if self.behavior != "silent":
                queue.append((recipient, ("pre-prepare", self.view, self.seq, payload, self.index)))
                self.sent += 1
        entry["prepares"].add(self.index)            # the primary counts its own prepare as well
        self.send(queue, ("prepare", self.view, self.seq, self.wire_digest(self.digest), self.index))
        self.refresh(self.view, self.seq, self.digest, queue)

    def vote(self, queue, new_view):
        if self.behavior in ("silent", "partitioned"):
            return
        self.votes.add(self.index)                   # a replica counts its own vote too
        self.send(queue, ("view-change", new_view, self.index))

    # ------------------------------------------------------------------ the protocol
    def deliver(self, message, queue):
        if self.behavior in ("silent", "partitioned"):
            return
        kind = message[0]
        if kind == "view-change":
            self.votes.add(message[2])
        elif kind == "new-view":
            self.view = message[1]
        elif kind == "pre-prepare":
            _, view, seq, digest, sender = message
            if view != self.view or sender != view % self.n:
                return
            if not self.low <= seq <= self.high:
                return
            if digest != self.digest:         # an equivocated digest is simply not a pre-prepare
                return
            self.seq = seq
            entry = self.entry(view, seq, digest)
            entry["prepares"].add(self.index)                    # a replica counts its own prepare
            self.send(queue, ("prepare", view, seq, self.wire_digest(digest), self.index))
            self.refresh(view, seq, digest, queue)
        elif kind == "prepare":
            _, view, seq, digest, sender = message
            if view != self.view or digest != self.digest:
                return
            self.entry(view, seq, digest)["prepares"].add(sender)
            self.refresh(view, seq, digest, queue)
        elif kind == "commit":
            _, view, seq, digest, sender = message
            if view != self.view or digest != self.digest:
                return
            self.entry(view, seq, digest)["commits"].add(sender)
            self.refresh(view, seq, digest, queue)

    def refresh(self, view, seq, digest, queue):
        """Move the replica forward whenever a quorum has been reached (idempotent)."""
        entry = self.entry(view, seq, digest)
        if (view, seq, digest) not in self.prepared and len(entry["prepares"]) >= self.quorum:
            self.prepared.add((view, seq, digest))
            entry["commits"].add(self.index)                     # it counts its own commit too
            self.send(queue, ("commit", view, seq, self.wire_digest(digest), self.index))
        if (view, seq) not in self.committed and len(entry["commits"]) >= self.quorum:
            self.committed[(view, seq)] = digest
            self.replies += 1                                    # the reply goes to the client


def drain(queue, replicas, unreachable=(), limit=4000):
    steps = 0
    while queue and steps < limit:
        recipient, message = queue.popleft()
        steps += 1
        if recipient in unreachable or message[-1] in unreachable:
            continue                                  # a partition drops the message entirely
        replicas[recipient].deliver(message, queue)
    return steps


def run(request, behaviors, attempts=2, quorum=3, low=1, high=64, unreachable=()):
    """Run the protocol for one client request; returns the replicas and the message count."""
    n = len(behaviors)
    replicas = [Replica(index, behaviors[index], n, quorum, low, high) for index in range(n)]
    digest = digest_of(request)
    queue = deque()
    for replica in replicas:
        if replica.behavior != "no_request":
            replica.digest = digest
        replica.on_request(queue)
    messages = 0
    for attempt in range(attempts):
        messages += drain(queue, replicas, unreachable)
        active = [r for r in replicas if r.behavior not in ("silent", "partitioned")
                  and r.index not in unreachable]
        if all(r.committed for r in active) or attempt == attempts - 1:
            break
        new_view = replicas[0].view + 1
        for replica in replicas:
            replica.vote(queue, new_view)
        messages += drain(queue, replicas, unreachable)
        primary = replicas[new_view % n]
        if primary.behavior not in ("silent", "partitioned") and len(primary.votes) >= quorum:
            primary.view = new_view
            primary.send(queue, ("new-view", new_view, primary.index))
            messages += drain(queue, replicas, unreachable)
            primary.propose(queue)
    messages += drain(queue, replicas, unreachable)
    return replicas, messages


def digests_committed(replicas):
    return sorted({digest for replica in replicas for digest in replica.committed.values()})


def main():
    nodes = list(range(4))

    # 1. the quorum-intersection lemma, verified exhaustively for n = 4 and n = 7
    check(quorum_holds_against_every_byzantine_set(4, 1), "n=4, f=1: quorums must share a correct node")
    check(quorum_holds_against_every_byzantine_set(7, 2), "n=7, f=2: same for the seven-node case")
    check(not intersection_lemma(3, 1),
          "n=3, f=1 fails the admission test: only n-f=2 correct replicas remain")
    check(quorum_holds_against_every_byzantine_set(3, 1),
          "with a quorum of all three, the shared part is still correct - liveness is what breaks")
    sizes = {f: [n for n in range(3, 3 * f + 3) if intersection_lemma(n, f)] for f in (1, 2, 3)}
    check([min(values) for values in sizes.values()] == [4, 7, 10],
          "the smallest safe replica count must be 3f+1: %r" % (sizes,))
    check(all(all(n >= 3 * f + 1 for n in values) for f, values in sizes.items()),
          "every admissible n must satisfy n >= 3f+1")

    # 2. happy path: a correct primary and three honest replicas
    honest, messages = run(b"transfer 5 to alice", ["honest"] * 4)
    real = digest_of(b"transfer 5 to alice")
    check(all(replica.committed for replica in honest), "every replica must commit")
    check(digests_committed(honest) == [real], "only the real request digest may be committed")
    check(all(replica.replies == 1 for replica in honest), "each replica must reply once")
    check(sum(replica.sent for replica in honest) == 3 * (1 + 4 + 4),
          "3 pre-prepares plus every replica's prepare and commit, each broadcast to 3 peers")
    check(all(len(replica.state[(0, 1, real)]["prepares"]) >= 3 for replica in honest),
          "a quorum of 2f+1 = 3 matching prepares is required")

    # 3. one silent Byzantine replica: liveness survives, safety is unaffected
    silent, _ = run(b"transfer 5 to alice", ["honest", "honest", "honest", "silent"])
    check(sum(1 for r in silent if r.committed) == 3, "the three correct replicas must commit")
    check(not silent[3].committed, "the silent replica commits nothing")
    check(digests_committed(silent) == [real], "the committed digest is still the real one")

    # 4. an equivocating primary cannot split the correct replicas: the client broadcast the request,
    #    so every replica compares the pre-prepare digest with the one it computed itself
    equivocate, _ = run(b"transfer 5 to alice", ["equivocate", "honest", "honest", "honest"])
    check(digests_committed(equivocate) == [real],
          "no replica may commit the equivocated digest: %r" % (digests_committed(equivocate),))
    check(sum(1 for r in equivocate if r.committed) == 4,
          "all four replicas must commit the real value despite the equivocation")
    check(equivocate[1].committed == {(0, 1): real},
          "the replica fed the wrong digest still commits the real one")

    # 5. a digest-forging replica: its prepares/commits are ignored, the run still converges
    forge, _ = run(b"transfer 5 to alice", ["honest", "forge", "honest", "honest"])
    check(digests_committed(forge) == [real], "a forged digest must never be committed")
    check(not any(BAD_MARK in digest for digest in digests_committed(forge)),
          "the forged marker must not appear in any commit")
    check(sum(1 for r in forge if r.committed) == 4, "all four replicas are correct here")

    # 6. a replica that never received the request cannot be tricked into preparing
    starved, _ = run(b"transfer 5 to alice", ["honest", "honest", "honest", "no_request"])
    check(not starved[3].committed, "a replica without the request must not commit")
    check(sum(1 for r in starved if r.committed) == 3, "the other three still reach a quorum")
    check(starved[3].prepared == set(), "it must not even prepare")

    # 7. a silent primary: the view change must restore liveness
    changed, steps = run(b"transfer 5 to alice", ["silent", "honest", "honest", "honest"])
    check(any(replica.view == 1 for replica in changed),
          "the replicas must move to view 1 after the primary stalls")
    check(all(replica.committed for replica in changed if replica.behavior != "silent"),
          "every correct replica must commit after the view change")
    check(digests_committed(changed) == [real], "the value committed after the view change is the real one")
    check(changed[1].view == 1 and len(changed[1].votes) >= 3,
          "the new primary must have collected a quorum of view-change votes")
    check(steps > 0, "the view change must have exchanged messages")

    # 8. duplicate deliveries are idempotent: sets, not counts, decide the quorum
    doubled = [Replica(index, "honest", 4, 3, 1, 64) for index in range(4)]
    queue = deque()
    for replica in doubled:
        replica.digest = real
        replica.on_request(queue)
    first = drain(queue, doubled)
    counts = [len(replica.state[(0, 1, real)]["prepares"]) for replica in doubled]
    commits = [len(replica.state[(0, 1, real)]["commits"]) for replica in doubled]
    replay = deque([(0, ("prepare", 0, 1, real, 1)), (0, ("prepare", 0, 1, real, 1)),
                    (0, ("prepare", 0, 1, real, 1)), (0, ("commit", 0, 1, real, 2)),
                    (0, ("commit", 0, 1, real, 3)), (0, ("pre-prepare", 0, 1, real, 0))])
    second = drain(replay, doubled)
    check(first > 0 and second > 0, "both delivery batches must be processed")
    check(counts == [3, 3, 3, 3] or counts == [4, 4, 4, 4],
          "no more than the four replicas may be counted as prepares: %r" % (counts,))
    check([len(replica.state[(0, 1, real)]["prepares"]) for replica in doubled] == counts,
          "re-delivering the same prepare must not change the quorum count")
    check([len(replica.state[(0, 1, real)]["commits"]) for replica in doubled] == commits,
          "re-delivering the same commit must not change the quorum count")
    check(all(replica.committed for replica in doubled), "duplicates must not break the commit")
    check(len(set(digests_committed(doubled))) == 1, "duplicates must not introduce a second digest")

    # 9. watermarks: a sequence number outside [low, high] is discarded
    for seq in (0, 200):
        bounded = [Replica(index, "honest", 4, 3, 10, 20) for index in range(4)]
        queue = deque()
        for replica in bounded:
            replica.digest = real
            replica.seq = seq
            replica.propose(queue)
        drain(queue, bounded)
        check(not any(replica.committed for replica in bounded),
              "sequence number %d outside the watermarks must be ignored" % seq)
    inside = [Replica(index, "honest", 4, 3, 10, 20) for index in range(4)]
    queue = deque()
    for replica in inside:
        replica.digest = real
        replica.seq = 15
        replica.propose(queue)
    drain(queue, inside)
    check(all(replica.committed for replica in inside), "a sequence number inside the window works")

    # 10. with two replicas unreachable the quorum of 3 cannot be assembled - no commit, no lie
    partitioned, _ = run(b"transfer 5 to alice", ["honest", "honest", "honest", "honest"],
                         attempts=1, unreachable=(2, 3))
    check(sum(1 for r in partitioned if r.committed) == 0,
          "two reachable replicas are not a quorum, so nothing may be committed")

    # 11. every replica that commits agrees on the digest for a given sequence number
    for replicas in (honest, silent, equivocate, forge, changed):
        for seq in {key[1] for replica in replicas for key in replica.committed}:
            values = {replica.committed[key] for replica in replicas for key in replica.committed
                      if key[1] == seq}
            check(len(values) == 1, "replicas must agree on sequence %d: %r" % (seq, values))

    # 12. determinism of the whole simulation
    first_run, first_messages = run(b"transfer 5 to alice", ["honest", "forge", "honest", "silent"])
    second_run, second_messages = run(b"transfer 5 to alice", ["honest", "forge", "honest", "silent"])
    check(first_messages == second_messages, "message counts must be reproducible")
    check([r.committed for r in first_run] == [r.committed for r in second_run],
          "commit logs must be reproducible")

    # 13. boundary: two replicas with a quorum of two reach agreement (n = 2 is below 3f+1, so this
    #     only exercises the mechanism, not Byzantine tolerance)
    pair, _ = run(b"pair", ["honest", "honest"], quorum=2, attempts=1)
    check(all(replica.committed for replica in pair), "a quorum of two must commit")
    check(digests_committed(pair) == [digest_of(b"pair")], "the pair commits the real digest")
    check(all(replica.replies == 1 for replica in pair), "both replicas reply once")

    print("759 quorum: 2f+1=%d of n=%d; intersection lemma holds for (4,1) and (7,2), breaks at n=3f"
          % (3, 4))
    print("759 happy path: %d commits, %d messages, digest %s" % (len(honest), messages, real))
    print("759 faults: silent -> 3 commits, equivocating primary -> %d commits, forged digest -> %s"
          % (sum(1 for r in equivocate if r.committed), "ignored"))
    print("759 view change: view=%d, votes=%d, commits=%d"
          % (changed[1].view, len(changed[1].votes), sum(1 for r in changed if r.committed)))
    print("PASS")


if __name__ == "__main__":
    main()
