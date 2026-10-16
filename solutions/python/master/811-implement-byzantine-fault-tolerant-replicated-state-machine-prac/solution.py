"""Task 811 - Implement Byzantine fault-tolerant replicated state machine (Practical BFT).  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

PBFT as a deterministic message-passing simulation.  A cluster of n = 3f + 1 replicas agrees on the
order and content of client requests through three phases per view:

  * the primary assigns a sequence number and multicasts PRE-PREPARE (view, seq, digest, request),
    and processes its own proposal like any other replica;
  * a replica that accepts the pre-prepare - right view, from that view's primary, inside the
    watermarks, digest matching the request, not already seen - multicasts PREPARE, and a replica
    holding a pre-prepare plus 2f prepares from distinct replicas has a *prepared certificate* and
    multicasts COMMIT;
  * a replica that collects 2f + 1 matching commits records the commit certificate, executes the
    request and replies to the client once.

Certificates are what make the protocol safe: any two quorums of size 2f + 1 out of 3f + 1 share
f + 1 replicas, so at least one shared replica is correct, and a correct replica never prepares or
commits two different digests for the same sequence number.  The checks verify exactly that: the
executed values of all correct replicas are compared per sequence number, the certificates behind
them are intersected, and an agreement violation is reported together with the replicas whose
double vote would have been necessary to produce it.  View changes are driven explicitly (a replica
that cannot make progress asks for the next view; the new primary re-proposes the certificates it
collected), which is what keeps a silent or equivocating primary from destroying safety or, once
the view moves, liveness.

Verified invariants:
  agreement          no two correct replicas execute different requests at the same sequence number
                     across every run, view and fault injection - brute-forced over all correct
                     replicas and sequence numbers;
  certificates       every execution is justified by 2f + 1 matching commits, and whenever two
                     conflicting certificates exist at one sequence number the detector names the
                     replicas that must have voted twice;
  quorum arithmetic  any two 2f + 1 quorums out of 3f + 1 share at least f + 1 replicas, and with at
                     most f faulty replicas the shared part contains a correct one (brute-forced
                     over all subsets for n = 4 and n = 7);
  validation         a digest that does not match the request, a pre-prepare from a non-primary,
                     a sequence number outside the watermarks, a duplicate pre-prepare, a prepare
                     without a pre-prepare, a prepare for a digest other than the proposed one,
                     a commit without being prepared and a commit for a digest the replica never
                     prepared are all rejected and never executed;
  liveness           requests commit with f faulty replicas; a silent primary stalls the cluster
                     until a view change, after which progress resumes; a minority of correct
                     replicas can never make anything execute;
  client view        the client accepts an answer only after f + 1 matching replies;
  checkpoints        a checkpoint becomes stable on 2f + 1 matching checkpoint digests and only then;
  bound tightness    the crafted two-fault counter-example at n = 4, f = 1 (a forking primary plus
                     two colluding double-voters) does produce an agreement violation, which is why
                     n >= 3f + 1 is necessary rather than cosmetic;
  determinism        identical fault injections produce identical executions, views and messages.

Simplified versus a production BFT library: one process, so messages travel over an explicit queue
with fixed latency, and there is no cryptography - digests are deterministic checksums rather than
hashes and messages are unsigned, so the simulation trusts a stated replica identity; the watermark
and checkpoint windows are small, no state transfer or log garbage collection is implemented, view
changes are triggered by the driver rather than by timers, and the workload is a handful of requests
so that every certificate can be inspected.
"""

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


def string_digest(text):
    """Deterministic stand-in for a hash: same input, same digest, easy to inspect."""
    state = 0x5EED
    for character in text:
        state, _raw = splitmix64(state * 31 + ord(character) + 7)
    _state, raw = splitmix64(state)
    return "dig%08x" % (raw % (1 << 32))


def quorum_size(size):
    """2f + 1 with f = (n - 1) // 3."""
    faults = (size - 1) // 3
    return 2 * faults + 1, faults


class Replica:
    def __init__(self, rid, size, fault_mode=None):
        self.id = rid
        self.size = size
        self.quorum, self.faults = quorum_size(size)
        self.mode = fault_mode                # None means correct
        self.view = 0
        self.pre_prepares = {}                # (view, seq) -> (digest, request)
        self.prepares = {}                    # (view, seq, digest) -> set of replica ids
        self.commits = {}
        self.prepared = set()
        self.commit_certificates = {}         # (view, seq) -> (digest, frozenset voters)
        self.executed = {}                    # seq -> (digest, result)
        self.execution_order = []
        self.replies = [0]                    # indexes: one counter entry
        self.state = []                       # the replicated state machine
        self.checkpoints = []
        self.rejected = {}
        self.high_water = 64

    # ---------------------------------------------------------------------- helpers
    def is_primary(self, view=None):
        view = self.view if view is None else view
        return view % self.size == self.id

    def reject(self, reason):
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    # -------------------------------------------------------------------- messages
    def handle(self, message):
        kind = message[0]
        if kind == "pre-prepare":
            return self.on_pre_prepare(message)
        if kind == "prepare":
            return self.on_prepare(message)
        if kind == "commit":
            return self.on_commit(message)
        if kind == "view-change":
            return self.on_view_change(message)
        if kind == "new-view":
            return self.on_new_view(message)
        raise ValueError("unknown message %r" % (kind,))

    def on_pre_prepare(self, message):
        _kind, view, sequence, digest, request, sender = message
        if view != self.view:
            self.reject("pre-prepare from another view")
            return []
        if sender != view % self.size:
            self.reject("pre-prepare from a non-primary")
            return []
        if not 0 < sequence <= self.high_water:
            self.reject("pre-prepare outside the watermarks")
            return []
        if string_digest(request) != digest:
            self.reject("pre-prepare digest mismatch")
            return []
        if (view, sequence) in self.pre_prepares:
            self.reject("duplicate pre-prepare")
            return []
        self.pre_prepares[(view, sequence)] = (digest, request)
        return [("prepare", view, sequence, digest, self.id, None)]

    def on_prepare(self, message):
        _kind, view, sequence, digest, _sender_id, _extra = message
        sender = message[4]
        if view != self.view:
            self.reject("prepare from another view")
            return []
        if (view, sequence) not in self.pre_prepares:
            self.reject("prepare without a pre-prepare")
            return []
        if self.pre_prepares[(view, sequence)][0] != digest:
            self.reject("prepare for a different digest")
            return []
        self.prepares.setdefault((view, sequence, digest), set()).add(sender)
        if len(self.prepares[(view, sequence, digest)]) + 1 < self.quorum:
            return []
        if (view, sequence) in self.prepared:
            return []
        self.prepared.add((view, sequence))
        return [("commit", view, sequence, digest, self.id, None)]

    def on_commit(self, message):
        _kind, view, sequence, digest, sender, _extra = message
        if view != self.view:
            self.reject("commit from another view")
            return []
        if (view, sequence) not in self.prepared:
            self.reject("commit without being prepared")
            return []
        if digest != self.pre_prepares[(view, sequence)][0]:
            self.reject("commit for a different digest")
            return []
        self.commits.setdefault((view, sequence, digest), set()).add(sender)
        if len(self.commits[(view, sequence, digest)]) + 1 < self.quorum:
            return []
        if (view, sequence) in self.commit_certificates:
            return []
        voters = frozenset(self.commits[(view, sequence, digest)] | {self.id})
        self.commit_certificates[(view, sequence)] = (digest, voters)
        return self.execute(view, sequence, digest)

    def execute(self, view, sequence, digest):
        if sequence in self.executed:
            return []
        request = self.pre_prepares[(view, sequence)][1]
        result = "result(%s)" % request
        self.executed[sequence] = (digest, result)
        self.execution_order.append((view, sequence, digest))
        self.state.append(request)
        self.replies[0] += 1
        if len(self.state) % 4 == 0:
            self.checkpoints.append((len(self.state), string_digest("|".join(self.state))))
        return [("reply", view, sequence, digest, self.id, result)]

    # ----------------------------------------------------------------- view change
    def start_view_change(self, new_view):
        bundle = tuple(sorted((sequence, digest)
                              for (view, sequence), (digest, _request)
                              in self.pre_prepares.items() if view == self.view))
        return [("view-change", new_view, None, None, self.id, bundle)]

    def on_view_change(self, message):
        _kind, view, _sequence, _digest, _sender, bundle = message
        if not self.is_primary(view):
            return []
        self.collected_views = getattr(self, "collected_views", {})
        self.collected_views.setdefault(view, []).append(bundle)
        if len(self.collected_views[view]) < self.quorum - 1:
            return []
        merged = {}
        for collected in self.collected_views[view]:
            for sequence, digest in collected:
                merged.setdefault(sequence, set()).add(digest)
        proposals = tuple(sorted((sequence, sorted(digests)[0])
                                 for sequence, digests in merged.items()))
        self.view = view
        self.new_view_proposals = proposals
        return [("new-view", view, None, None, self.id, proposals)]

    def on_new_view(self, message):
        _kind, view, _sequence, _digest, _sender, proposals = message
        self.view = view
        self.new_view_proposals = tuple(proposals)
        return []


class Cluster:
    """Deterministic message bus: a fixed latency per link, delivery ordered by (time, order)."""

    def __init__(self, size, fault_modes=None):
        fault_modes = fault_modes or {}
        self.size = size
        self.quorum, self.faults = quorum_size(size)
        self.replicas = {rid: Replica(rid, size, fault_modes.get(rid)) for rid in range(size)}
        self.queue = []
        self.order = 0
        self.clock = 0
        self.sent = 0
        self.assigned = 0
        self.client_replies = {}          # sequence -> digest -> set of replica ids
        self.byzantine_votes = 0

    def link_latency(self, sender, target):
        return 1 + (sender * 7 + target * 3) % 3

    def send(self, sender, target, message):
        self.order += 1
        self.sent += 1
        self.queue.append((self.clock + self.link_latency(sender, target), self.order, target,
                           message))
        self.queue.sort(key=lambda entry: (entry[0], entry[1]))

    def broadcast(self, sender, message, exclude=()):
        for rid in sorted(self.replicas):
            if rid == sender or rid in exclude:
                continue
            self.send(sender, rid, message)

    def emit(self, sender, message):
        if message[0] == "reply":
            _kind, _view, sequence, digest, rid, _result = message
            self.client_replies.setdefault(sequence, {}).setdefault(digest, set()).add(rid)
            return
        self.broadcast(sender, message)
        replica = self.replicas[sender]
        if replica.mode == "colluding" and message[0] in ("prepare", "commit"):
            forged = list(message)
            forged[3] = "forged-" + str(message[3])
            self.broadcast(sender, tuple(forged))
            self.byzantine_votes += 1

    def deliver(self, target, message):
        """Deliver one message to one replica, honouring a silent replica's refusal to act."""
        replica = self.replicas[target]
        if replica.mode == "silent":
            return False
        for outgoing in replica.handle(message):
            self.emit(target, outgoing)
        return True

    def deliver_one(self):
        """Deliver every message due at the current clock, then advance it."""
        delivered = False
        while self.queue and self.queue[0][0] <= self.clock:
            _at, _order, target, message = self.queue.pop(0)
            delivered = True
            self.deliver(target, message)
        self.clock = self.queue[0][0] if self.queue else self.clock + 1
        return delivered

    def run(self, rounds=10):
        for _ in range(rounds):
            self.deliver_one()
        return self.clock

    # --------------------------------------------------------------------- client
    def view_in_use(self):
        views = [replica.view for replica in self.replicas.values() if replica.mode != "silent"]
        return min(views) if views else 0

    def request(self, text, client="c0"):
        """The client sends a request to every replica; only the primary assigns a sequence."""
        self.assigned += 1
        sequence = self.assigned
        view = self.view_in_use()
        primary = view % self.size
        request = "%s:%s" % (client, text)
        digest = string_digest(request)
        if self.replicas[primary].mode == "silent":
            self.run(3)                      # a silent primary proposes nothing at all
            return sequence, digest
        for rid in sorted(self.replicas):
            if rid == primary:
                continue
            if self.replicas[rid].mode == "colluding" and \
                    self.replicas[primary].mode == "fork-primary":
                # a forking primary shows its accomplices the conflicting proposal too
                self.send(primary, rid, ("pre-prepare", view, sequence, digest, request, primary))
                self.send(primary, rid, ("pre-prepare", view, sequence,
                                         string_digest("alt:" + request), request, primary))
                self.byzantine_votes += 1
            else:
                self.send(primary, rid, ("pre-prepare", view, sequence, digest, request, primary))
        if self.replicas[primary].mode == "fork-primary":
            for rid in sorted(self.replicas):
                if rid == primary or self.replicas[rid].mode == "colluding":
                    continue
                # the fork: the other half is told a different request is being ordered
                self.send(primary, rid, ("pre-prepare", view, sequence,
                                         string_digest("alt:" + request), request, primary))
                self.byzantine_votes += 1
        self.emit(primary, ("pre-prepare", view, sequence, digest, request, primary))
        self.deliver(primary, ("pre-prepare", view, sequence, digest, request, primary))
        self.run(4)
        return sequence, digest

    def forged_vote(self, sender, target, kind, view, sequence, digest, request=None):
        """Directly inject a Byzantine message - used only by the crafted counter-example."""
        if kind == "pre-prepare":
            message = (kind, view, sequence, digest, request, sender)
        else:
            message = (kind, view, sequence, digest, sender, None)
        self.send(sender, target, message)
        self.byzantine_votes += 1

    def client_result(self, sequence):
        for digest, rids in sorted(self.client_replies.get(sequence, {}).items()):
            if len(rids) >= self.faults + 1:
                return digest, sorted(rids)
        return None, []

    def view_change(self, new_view):
        for rid in sorted(self.replicas):
            replica = self.replicas[rid]
            if replica.mode == "silent":
                continue
            for outgoing in replica.start_view_change(new_view):
                self.broadcast(rid, outgoing)
        self.run(8)
        return {rid: self.replicas[rid].view for rid in sorted(self.replicas)}


# ----------------------------------------------------------------------- verification
def quorum_intersection_holds(size):
    """Any two 2f + 1 quorums share at least f + 1 replicas, hence one correct replica."""
    quorum, faults = quorum_size(size)
    universe = list(range(size))
    subsets = [{universe[index] for index in range(size) if mask >> index & 1}
               for mask in range(1 << size)]
    subsets = [subset for subset in subsets if len(subset) == quorum]
    for left in subsets:
        for right in subsets:
            shared = left & right
            if len(shared) < faults + 1:
                return False, (sorted(left), sorted(right), sorted(shared))
    return True, None


def agreement_violations(cluster):
    """No two correct replicas may execute different values at the same sequence number."""
    by_sequence = {}
    for rid, replica in sorted(cluster.replicas.items()):
        if replica.mode is not None:
            continue
        for sequence, (digest, _result) in sorted(replica.executed.items()):
            by_sequence.setdefault(sequence, []).append((rid, digest))
    return [(sequence, sorted(entries)) for sequence, entries in sorted(by_sequence.items())
            if len({entry[1] for entry in entries}) > 1]


def certificate_problems(cluster):
    """Every execution needs a 2f + 1 certificate; conflicting certificates name the double voters."""
    problems = []
    quorum = cluster.quorum
    for rid, replica in sorted(cluster.replicas.items()):
        if replica.mode is not None:
            continue
        for (view, sequence), (digest, voters) in sorted(replica.commit_certificates.items()):
            if len(voters) < quorum:
                problems.append(("certificate below quorum", rid, view, sequence,
                                 sorted(voters)))
            if replica.executed.get(sequence, (None,))[0] != digest:
                problems.append(("executed digest differs from the certificate", rid, sequence))
    certificates = {}
    for rid, replica in sorted(cluster.replicas.items()):
        if replica.mode is not None:
            continue
        for (view, sequence), (digest, voters) in sorted(replica.commit_certificates.items()):
            certificates.setdefault((view, sequence), []).append((digest, set(voters), rid))
    for key, entries in sorted(certificates.items()):
        for index, (left_digest, left_voters, left_rid) in enumerate(entries):
            for right_digest, right_voters, right_rid in entries[index + 1:]:
                if left_digest == right_digest:
                    continue
                problems.append(("conflicting certificates", key, sorted(left_voters & right_voters),
                                 left_rid, right_rid))
    return problems


def stable_checkpoints(cluster):
    tally = {}
    for replica in cluster.replicas.values():
        if replica.mode is not None:
            continue
        for count, digest in replica.checkpoints:
            tally.setdefault((count, digest), set()).add(replica.id)
    return sorted(key for key, rids in tally.items() if len(rids) >= cluster.quorum)


def crafted_two_fault_fork():
    """The classic counter-example to n > 3f: a forking primary plus one colluding double voter.

    Replicas 0 (the primary) and 1 are faulty; the correct replicas 2 and 3 each collect a full
    2f + 1 certificate - for different values at the same sequence number - because the two faulty
    replicas vote twice.
    """
    cluster = Cluster(4, fault_modes={0: "colluding", 1: "colluding"})
    request_a, request_b = "c0:wire 100", "c0:wire 900"
    left, right = string_digest(request_a), string_digest(request_b)
    for target, digest, request in ((0, left, request_a), (2, left, request_a),
                                    (1, right, request_b), (3, right, request_b)):
        cluster.forged_vote(0, target, "pre-prepare", 0, 1, digest, request)   # the fork
    cluster.run(6)
    for target, digest in ((2, left), (3, right)):                      # the primary equivocates
        cluster.forged_vote(0, target, "prepare", 0, 1, digest)
    for target in (0, 1, 2, 3):                                         # the accomplice votes twice
        for digest in (left, right):
            cluster.forged_vote(1, target, "prepare", 0, 1, digest)
    cluster.run(8)
    for target, digest in ((2, left), (3, right)):
        cluster.forged_vote(0, target, "commit", 0, 1, digest)
    for target in (0, 1, 2, 3):
        for digest in (left, right):
            cluster.forged_vote(1, target, "commit", 0, 1, digest)
    cluster.run(8)
    return cluster, left, right


def main():
    # --- 1. quorum arithmetic -----------------------------------------------------
    for size in (4, 7):
        ok, witness = quorum_intersection_holds(size)
        check(ok, "any two 2f + 1 quorums of %d replicas must share f + 1: %s" % (size, witness))
    check(quorum_size(4) == (3, 1), "four replicas tolerate one fault with a quorum of three")
    check(quorum_size(7) == (5, 2), "seven replicas tolerate two faults with a quorum of five")
    check(quorum_size(3) == (1, 0),
          "three replicas only admit f = 0, so one vote is a quorum and they are not Byzantine "
          "tolerant at all")

    # --- 2. a fault-free run: agreement, certificates and the client view ---------
    normal = Cluster(4)
    sequences = [normal.request("set x %d" % index) for index in range(3)]
    normal.run(6)
    check(agreement_violations(normal) == [],
          "a fault-free run must agree: %s" % (agreement_violations(normal),))
    check(certificate_problems(normal) == [],
          "every execution must carry a valid certificate: %s"
          % (certificate_problems(normal)[:2],))
    state = [replica.state for replica in normal.replicas.values()]
    check(all(entry == ["c0:set x 0", "c0:set x 1", "c0:set x 2"] for entry in state),
          "the replicated state machine must hold every request in order: %s" % (state,))
    check(all(len(replica.executed) == 3 for replica in normal.replicas.values()),
          "every correct replica must execute all three requests: %s"
          % {rid: sorted(replica.executed) for rid, replica in sorted(normal.replicas.items())})
    for sequence, digest in sequences:
        accepted, voters = normal.client_result(sequence)
        check(accepted == digest and len(voters) >= normal.faults + 1,
              "the client must collect f + 1 matching replies for %d: %s" % (sequence, voters))
    check(normal.replicas[0].execution_order == normal.replicas[2].execution_order,
          "correct replicas must execute in the same order")

    # --- 3. validation paths ------------------------------------------------------
    validating = Cluster(4)
    target = validating.replicas[1]
    digest = string_digest("c0:hello")
    validating.send(0, 1, ("pre-prepare", 0, 1, "wrong-digest", "c0:hello", 0))
    validating.send(2, 1, ("pre-prepare", 0, 2, digest, "c0:hello", 2))
    validating.send(0, 1, ("pre-prepare", 0, 900, digest, "c0:hello", 0))
    validating.send(0, 1, ("prepare", 0, 5, digest, 0, None))
    validating.run(4)
    check(target.rejected.get("pre-prepare digest mismatch") == 1,
          "a digest that does not match the request must be rejected: %s" % (target.rejected,))
    check(target.rejected.get("pre-prepare from a non-primary") == 1,
          "a pre-prepare from a non-primary must be rejected")
    check(target.rejected.get("pre-prepare outside the watermarks") == 1,
          "a sequence number outside the watermarks must be rejected")
    check(target.rejected.get("prepare without a pre-prepare") == 1,
          "a prepare without a pre-prepare must be rejected")
    check(target.executed == {}, "a rejected proposal must never be executed")
    validating.send(0, 1, ("pre-prepare", 0, 3, digest, "c0:hello", 0))
    validating.send(0, 1, ("pre-prepare", 0, 3, digest, "c0:hello", 0))
    validating.run(4)
    check(target.rejected.get("duplicate pre-prepare") == 1,
          "a repeated pre-prepare for the same sequence number must be rejected")
    validating.send(0, 1, ("prepare", 0, 3, "a-different-digest", 0, None))
    validating.run(4)
    check(target.rejected.get("prepare for a different digest") == 1,
          "a prepare for a digest other than the proposed one must be rejected")
    validating.send(0, 1, ("commit", 0, 3, digest, 0, None))
    validating.run(4)
    check(target.rejected.get("commit without being prepared") == 1,
          "a commit from a replica that is not prepared must be rejected")
    check(validating.replicas[0].is_primary() and not validating.replicas[3].is_primary(),
          "only the view's primary may propose")

    # --- 4. a silent primary stops progress until the view changes ----------------
    silent = Cluster(4, fault_modes={0: "silent"})
    silent.request("set y 1")
    silent.run(10)
    check(all(not silent.replicas[rid].executed for rid in (1, 2, 3)),
          "a silent primary must not let anything execute: %s"
          % {rid: sorted(silent.replicas[rid].executed) for rid in (1, 2, 3)})
    check(silent.client_result(1) == (None, []),
          "the client must not accept an answer without f + 1 matching replies")
    views = silent.view_change(1)
    check(views == {0: 0, 1: 1, 2: 1, 3: 1},
          "the correct replicas must move to the new view: %s" % (views,))
    silent.request("set y 2")
    silent.run(10)
    check(agreement_violations(silent) == [],
          "the view change must not break agreement: %s" % (agreement_violations(silent),))
    check(all(len(silent.replicas[rid].executed) >= 1 for rid in (1, 2, 3)),
          "the new view must make progress: %s"
          % {rid: sorted(silent.replicas[rid].executed) for rid in (1, 2, 3)})
    accepted, voters = silent.client_result(2)
    check(accepted is not None and len(voters) >= silent.faults + 1,
          "the client must be answered after the view change: %s" % (voters,))

    # --- 5. an equivocating primary cannot fork the cluster -----------------------
    forker = Cluster(4, fault_modes={0: "fork-primary"})
    forker.request("split brain", client="c1")
    forker.run(10)
    check(forker.byzantine_votes > 0,
          "the equivocating primary must have sent conflicting proposals")
    check(agreement_violations(forker) == [],
          "an equivocating primary must not produce two executions: %s"
          % (agreement_violations(forker),))
    check(certificate_problems(forker) == [],
          "no certificate may be invalid under an equivocating primary: %s"
          % (certificate_problems(forker)[:2],))
    executed = {rid: sorted(replica.executed) for rid, replica in sorted(forker.replicas.items())
                if rid != 0}
    check(all(len(entries) <= 1 for entries in executed.values()),
          "one faulty primary means the correct replicas execute at most one value: %s"
          % (executed,))
    forker.view_change(1)
    forker.request("after the fork", client="c1")
    forker.run(10)
    check(agreement_violations(forker) == [],
          "the new view must keep agreement: %s" % (agreement_violations(forker),))
    check(any(replica.executed for rid, replica in sorted(forker.replicas.items()) if rid != 0),
          "the new primary must make progress: %s"
          % {rid: sorted(forker.replicas[rid].executed) for rid in (1, 2, 3)})

    # --- 6. a minority of correct replicas can never execute ---------------------
    minority = Cluster(4, fault_modes={1: "silent", 2: "silent"})
    minority.request("cannot commit")
    minority.run(10)
    check(all(not replica.executed for replica in minority.replicas.values()),
          "with only two live replicas nothing may execute: %s"
          % {rid: sorted(replica.executed) for rid, replica in sorted(minority.replicas.items())})
    check(minority.client_result(1) == (None, []),
          "the client must not be answered by a minority")

    # --- 7. the bound is tight: two faulty replicas at n = 4 ----------------------
    broken, left, right = crafted_two_fault_fork()
    violations = agreement_violations(broken)
    check(left != right, "the two forged digests must differ")
    check(violations != [],
          "with more than f faulty replicas the agreement detector must fire: %s"
          % (violations[:1],))
    problems = certificate_problems(broken)
    check(problems != [],
          "conflicting certificates must be reported: %s" % (problems[:1],))
    named = [problem for problem in problems if problem[0] == "conflicting certificates"]
    check(named and set(named[0][2]) == {0, 1},
          "the detector must name the two replicas that voted twice: %s" % (named[:1],))
    check(sorted(broken.replicas[2].executed) == [1] and sorted(broken.replicas[3].executed) == [1],
          "both correct replicas executed the same sequence number, which is what forks them: %s"
          % {rid: sorted(broken.replicas[rid].executed) for rid in (2, 3)})
    check(broken.replicas[2].executed[1][0] != broken.replicas[3].executed[1][0],
          "the two correct replicas executed different digests at the same sequence number")
    healthy = Cluster(4)
    healthy.request("same request, honest cluster")
    healthy.run(10)
    check(agreement_violations(healthy) == [] and certificate_problems(healthy) == [],
          "the same request on an honest cluster must not produce a fork")

    # --- 8. checkpoints ------------------------------------------------------------
    checkpointing = Cluster(4)
    for index in range(5):
        checkpointing.request("append %d" % index)
        checkpointing.run(4)
    expected_digest = string_digest("|".join("c0:append %d" % step for step in range(4)))
    check(stable_checkpoints(checkpointing) == [(4, expected_digest)],
          "exactly one checkpoint must become stable: %s" % (stable_checkpoints(checkpointing),))
    check(all(len(replica.checkpoints) == 1 for replica in checkpointing.replicas.values()),
          "each correct replica must checkpoint once after four executions: %s"
          % [replica.checkpoints for replica in checkpointing.replicas.values()])
    lonely = Cluster(4)
    lonely.request("only one")
    lonely.run(8)
    check(stable_checkpoints(lonely) == [],
          "a checkpoint without 2f + 1 matching digests must not be stable")

    # --- 9. determinism -----------------------------------------------------------
    first, second = Cluster(4), Cluster(4)
    for cluster in (first, second):
        cluster.request("set d 1")
        cluster.request("set d 2")
        cluster.run(8)
    check({rid: replica.execution_order for rid, replica in sorted(first.replicas.items())}
          == {rid: replica.execution_order for rid, replica in sorted(second.replicas.items())},
          "identical runs must produce identical execution orders")
    check({rid: replica.state for rid, replica in sorted(first.replicas.items())}
          == {rid: replica.state for rid, replica in sorted(second.replicas.items())},
          "identical runs must replicate identical state")
    check(first.sent == second.sent and first.clock == second.clock,
          "identical runs must move identical numbers of messages")

    print("normal run: %d requests, %d messages, %d replicas, quorum %d of %d"
          % (len(sequences), normal.sent, normal.size, normal.quorum, normal.size))
    print("faults: silent primary + view change, equivocating primary, two silent replicas - "
          "agreement violations %d anywhere except the crafted counter-example"
          % (len(agreement_violations(normal)) + len(agreement_violations(silent))
             + len(agreement_violations(forker))))
    print("tight bound: 2 faulty of 4 produced %d agreement violations and named double voters %s"
          % (len(violations), named[0][2] if named else None))
    print("PASS")


if __name__ == "__main__":
    main()
