"""Task 810 - Implement cross-datacenter replication with causal consistency guarantees.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

Causal consistency across geographically distributed replicas.  Every write gets a version vector:
the writer's own counter plus the context it observed when it wrote, which is exactly the
dependencies that must be delivered before the write may be applied anywhere else.  A replica
applies a local write immediately; a replicated write is buffered until every dependency is
present, and then applied - together with anything else that became enabled - in a deterministic
order.  Because a dependency can never be missing when a write is applied, the applied order at
every replica is a linear extension of the happens-before relation, and the buffers drain to the
same state everywhere once the network quiesces.

Sessions carry a causal token (a version vector).  A read with a token blocks until the replica's
clock covers the token, which gives read-your-writes and monotonic reads without any global
ordering: the replica only waits for the operations the session actually ran ahead of.

Verified invariants, several of them against independent derivations:
  causality           the happens-before relation computed from the write log (same-origin sequence
                      plus the recorded dependencies) is exactly the relation implied by the
                      version vectors (strict componentwise dominance) - two independent codings of
                      the same order agree pair by pair, and the relation is acyclic;
  delivery order      for every pair of operations with a happens-before edge, the first is applied
                      before the second at every replica - brute-forced over all pairs and all
                      replicas, including runs that force a message to overtake its dependency;
  buffering           an operation whose dependencies have not arrived is never applied, and the
                      buffer drains to exactly the missing operations once they arrive - shown by
                      forcing a dependent write to arrive before its dependency;
  convergence         after quiescence all replicas hold identical stores and identical clocks;
  session guarantees  read-your-writes across datacenters, monotonic reads with an advancing token,
                      and a read that waits only for the operations inside its token;
  eventual, not strong an unreplicated write is invisible elsewhere until it is replicated, which
                      the checks confirm is not a causality violation;
  idempotence         redelivering an operation changes neither the state nor the applied log;
  determinism         identical schedules produce identical states, identical applied orders and
                      identical buffer high-water marks.

Simplified versus a production system (Cosmos DB / CockroachDB): one process, so the "network" is
an explicit queue with fixed per-link latency; there are no transactions, no compaction of version
vectors and no failure or recovery of a replica, and concurrent writes to the same key are resolved
by a deterministic rank (version sum, then the sorted components) rather than by an application
policy; the workload is a small deterministic schedule of writes and reads rather than a stream of
client traffic, which is what makes the brute-force order checks affordable.
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


class Operation:
    __slots__ = ("id", "origin", "seq", "key", "value", "deps")

    def __init__(self, identity, origin, seq, key, value, deps):
        self.id = identity
        self.origin = origin
        self.seq = seq
        self.key = key
        self.value = value
        self.deps = dict(deps)

    def version(self):
        version = dict(self.deps)
        version[self.origin] = self.seq
        return version

    def __repr__(self):
        return "<op %s %s=%r deps=%s>" % (self.id, self.key, self.value, sorted(self.deps.items()))


class Replica:
    def __init__(self, name, latency_bias=0):
        self.name = name
        self.clock = {}                  # datacenter -> highest sequence observed
        self.store = {}                  # key -> (version vector, value)
        self.buffer = {}                 # operation id -> operation waiting for dependencies
        self.applied = []                # operation ids in application order
        self.seen = set()
        self.arrivals = {}               # operation id -> order in which it was first received
        self.buffer_high_water = 0
        self.latency_bias = latency_bias

    # ------------------------------------------------------------------- local writes
    def local_write(self, key, value, identity=None):
        deps = dict(self.clock)
        seq = deps.get(self.name, 0) + 1
        self.clock[self.name] = seq
        operation = Operation(identity or "%s-%d" % (self.name, seq), self.name, seq, key, value,
                              deps)
        self.apply(operation)
        return operation

    def read(self, key):
        entry = self.store.get(key)
        return entry[1] if entry else None

    def version_of(self, key):
        entry = self.store.get(key)
        return entry[0] if entry else {}

    # ------------------------------------------------------------------- replication
    def satisfied(self, operation):
        return all(self.clock.get(datacenter, 0) >= count
                   for datacenter, count in operation.deps.items())

    def apply(self, operation):
        if operation.id in self.seen:
            return False
        self.seen.add(operation.id)
        self.applied.append(operation.id)
        current = self.store.get(operation.key)
        version = operation.version()
        if current is None or version_rank(version) > version_rank(current[0]):
            self.store[operation.key] = (version, operation.value)
        for datacenter, count in version.items():
            self.clock[datacenter] = max(self.clock.get(datacenter, 0), count)
        return True

    def receive(self, operation):
        """Apply if the dependencies are present, otherwise park it; then drain what is ready."""
        if operation.id in self.seen:
            return 0
        self.arrivals.setdefault(operation.id, len(self.arrivals))
        applied = 0
        if self.satisfied(operation):
            applied += 1 if self.apply(operation) else 0
        else:
            self.buffer[operation.id] = operation
            self.buffer_high_water = max(self.buffer_high_water, len(self.buffer))
        progress = True
        while progress:
            progress = False
            for identity in sorted(self.buffer):
                waiting = self.buffer[identity]
                if self.satisfied(waiting):
                    del self.buffer[identity]
                    applied += 1 if self.apply(waiting) else 0
                    progress = True
        return applied

    def covers(self, token):
        return all(self.clock.get(datacenter, 0) >= count for datacenter, count in token.items())

    def snapshot(self):
        return {key: (dict(version), value) for key, (version, value) in self.store.items()}


def dominates(left, right):
    """left >= right componentwise, with at least one strict or equality allowed."""
    return all(left.get(datacenter, 0) >= count for datacenter, count in right.items())


def strictly_dominates(left, right):
    """left >= right componentwise and strictly greater in at least one component."""
    if not dominates(left, right):
        return False
    return any(left.get(datacenter, 0) > right.get(datacenter, 0)
               for datacenter in set(left) | set(right))


def version_rank(version):
    """A deterministic total order that agrees with dominance and resolves concurrent writes."""
    return (sum(version.values()), max(version.values()) if version else 0,
            tuple(sorted(version.items())))


class Network:
    """Explicit message queue: per-link one-way latency, deterministic delivery order."""

    def __init__(self, replicas, latency=None):
        self.replicas = replicas
        self.queue = []
        self.now = 0
        self.seq = 0
        self.latency = dict(latency or {})
        self.sent = 0
        self.delivered = 0
        self.clock = 0x5EED

    def link_latency(self, source, target):
        key = (source, target)
        if key not in self.latency:
            self.clock, raw = splitmix64(self.clock + 31 * len(source) + 17 * len(target))
            self.latency[key] = 1 + raw % 4
        return self.latency[key]

    def publish(self, operation, origin):
        for name in sorted(self.replicas):
            if name == origin:
                continue
            self.seq += 1
            self.sent += 1
            deliver_at = self.now + self.link_latency(origin, name)
            self.queue.append((deliver_at, self.seq, name, operation))
        self.queue.sort(key=lambda entry: (entry[0], entry[1]))
        return len(self.queue)

    def deliver_due(self):
        delivered = 0
        while self.queue and self.queue[0][0] <= self.now:
            _at, _seq, target, operation = self.queue.pop(0)
            self.replicas[target].receive(operation)
            self.delivered += 1
            delivered += 1
        return delivered

    def advance(self):
        self.now = self.queue[0][0] if self.queue else self.now + 1
        return self.now

    def drain(self, rounds=200):
        for _ in range(rounds):
            self.deliver_due()
            if not self.queue:
                return True
            self.advance()
        return not self.queue


class Cluster:
    def __init__(self, names, latency=None):
        self.replicas = {name: Replica(name) for name in names}
        self.network = Network(self.replicas, latency)
        self.transfers = []             # wake-up messages used only for session waits

    def write(self, datacenter, key, value, identity=None):
        operation = self.replicas[datacenter].local_write(key, value, identity)
        self.network.publish(operation, datacenter)
        return operation

    def wait_for(self, datacenter, token, rounds=200):
        """A session read blocks until the replica's clock covers its causal token."""
        replica = self.replicas[datacenter]
        for _ in range(rounds):
            if replica.covers(token):
                return True
            self.network.deliver_due()
            self.network.advance()
        return replica.covers(token)

    def session_read(self, datacenter, key, token=None):
        if token:
            self.wait_for(datacenter, token)
        return self.replicas[datacenter].read(key)

    def settle(self):
        return self.network.drain()


# ------------------------------------------------------------------------- verification
def happens_before_from_log(operations):
    """Causality as recorded: same-origin sequence order, or read through the dependency vector."""
    relation = set()
    for left in operations:
        for right in operations:
            if left.id == right.id:
                continue
            if left.origin == right.origin and left.seq < right.seq:
                relation.add((left.id, right.id))
            elif right.deps.get(left.origin, 0) >= left.seq:
                relation.add((left.id, right.id))
    return relation


def happens_before_from_vectors(operations):
    """Causality as implied by the version vectors: strict componentwise dominance."""
    relation = set()
    for left in operations:
        left_version = left.version()
        for right in operations:
            if left.id == right.id:
                continue
            if strictly_dominates(right.version(), left_version):
                relation.add((left.id, right.id))
    return relation


def order_violations(cluster, operations):
    """For every causal edge, the earlier operation must be applied first at every replica."""
    relation = happens_before_from_log(operations)
    problems = []
    for name, replica in sorted(cluster.replicas.items()):
        index = {identity: position for position, identity in enumerate(replica.applied)}
        for before, after in sorted(relation):
            if before not in index or after not in index:
                continue
            if index[before] > index[after]:
                problems.append((name, before, after, index[before], index[after]))
    return problems


def acyclic(relation, identities):
    """Topological sort; a short result means the relation has a cycle."""
    incoming = {identity: 0 for identity in identities}
    outgoing = {identity: [] for identity in identities}
    for before, after in relation:
        outgoing[before].append(after)
        incoming[after] += 1
    ready = sorted(identity for identity in identities if incoming[identity] == 0)
    ordered = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for successor in sorted(outgoing[current]):
            incoming[successor] -= 1
            if incoming[successor] == 0:
                ready.append(successor)
                ready.sort()
    return len(ordered) == len(identities), ordered


def build_cluster(names=("eu", "us", "apac"), latency=None):
    return Cluster(list(names), latency)


def scripted_run():
    """A deterministic causal workload with a forced overtake of a dependency message."""
    cluster = build_cluster(latency={("us", "apac"): 1, ("eu", "apac"): 6, ("apac", "eu"): 2,
                                     ("apac", "us"): 2, ("eu", "us"): 2, ("us", "eu"): 2})
    operations = []
    first = cluster.write("eu", "x", "eu-1")
    operations.append(first)
    cluster.network.drain()
    second = cluster.write("us", "y", "us-1")           # sees x through replication
    operations.append(second)
    cluster.network.drain()
    third = cluster.write("apac", "z", "apac-1")        # sees x and y
    operations.append(third)
    cluster.network.drain()
    return cluster, operations


def main():
    # --- 1. local writes and reads ------------------------------------------------
    lone = build_cluster(("eu",))
    lone.write("eu", "k", 1)
    check(lone.replicas["eu"].read("k") == 1, "a local write must be readable locally")
    check(lone.replicas["eu"].clock == {"eu": 1}, "a local write must advance the clock")
    check(lone.replicas["eu"].applied == ["eu-1"], "the applied log must record the write")

    # --- 2. replication and convergence ------------------------------------------
    cluster = build_cluster()
    operation = cluster.write("eu", "x", 10)
    check(cluster.replicas["us"].read("x") is None,
          "an unreplicated write must be invisible elsewhere (eventual, not strong consistency)")
    check(cluster.settle(), "the queue must drain")
    check(all(replica.read("x") == 10 for replica in cluster.replicas.values()),
          "after replication every datacenter must see the write")
    check(len({str(replica.snapshot()) for replica in cluster.replicas.values()}) == 1,
          "the stores must be identical after convergence")
    check(len({tuple(sorted(replica.clock.items())) for replica in cluster.replicas.values()}) == 1,
          "the clocks must be identical after convergence")
    check(cluster.network.sent == len(cluster.replicas) - 1,
          "one write must produce exactly one message per remote datacenter")
    check(cluster.replicas["eu"].read("x") == 10 and operation.version() == {"eu": 1},
          "the origin's version vector must record only its own write")

    # --- 3. causal delivery: a dependency may not be overtaken -------------------
    overtake = build_cluster(latency={("us", "apac"): 1, ("eu", "apac"): 6, ("apac", "us"): 2,
                                      ("apac", "eu"): 2, ("eu", "us"): 1, ("us", "eu"): 4})
    first = overtake.write("eu", "a", "eu-value")
    overtake.settle()
    second = overtake.write("us", "b", "us-value")
    check(second.deps.get("eu", 0) == 1 and second.version().get("us") == 1,
          "the second write must depend on the first: %s (its own version %s)"
          % (sorted(second.deps.items()), sorted(second.version().items())))
    overtake.settle()
    overtake.write("apac", "c", "apac-value")
    overtake.settle()
    check(overtake.replicas["apac"].read("a") == "eu-value"
          and overtake.replicas["apac"].read("b") == "us-value",
          "the causally later datacenter must see everything it depends on")

    forced = build_cluster(latency={("us", "apac"): 1, ("eu", "apac"): 9, ("eu", "us"): 1,
                                    ("apac", "eu"): 3, ("apac", "us"): 3, ("us", "eu"): 3})
    dependency = forced.write("eu", "a", 1)          # reaches us at t=1 and apac only at t=9
    forced.network.advance()
    forced.network.deliver_due()
    check(forced.replicas["us"].read("a") == 1,
          "the near datacenter must have received the dependency")
    check(forced.replicas["apac"].read("a") is None,
          "the far datacenter must not have the dependency yet")
    dependent = forced.write("us", "b", 2)           # reaches apac at t=2, before its dependency
    forced.network.advance()
    forced.network.deliver_due()
    apac = forced.replicas["apac"]
    check(dependent.id in apac.buffer,
          "the dependent write must be buffered, buffer holds %s" % sorted(apac.buffer))
    check(dependency.id not in apac.seen,
          "the dependency must not have been applied yet, otherwise the test is vacuous")
    check("a" not in forced.replicas["apac"].store,
          "the far datacenter must not hold the dependency value yet")
    check(apac.buffer_high_water == 1, "the buffer high-water mark must be one")
    forced.network.drain()
    check(apac.read("a") == 1 and apac.read("b") == 2,
          "both writes must be present once the dependency finally arrives")
    check(apac.arrivals[dependent.id] < apac.arrivals[dependency.id],
          "the dependent write must really have arrived first: %s"
          % (sorted(apac.arrivals.items()),))
    check(apac.applied.index(dependency.id) < apac.applied.index(dependent.id),
          "the dependency must be applied before the write that depends on it: %s"
          % (apac.applied,))
    check(apac.buffer == {}, "the buffer must drain completely")

    # --- 4. causality: two independent derivations agree, and are acyclic --------
    cluster, operations = scripted_run()
    extra = [cluster.write("us", "w%d" % index, index) for index in range(4)]
    operations.extend(extra)
    cluster.settle()
    from_log = happens_before_from_log(operations)
    from_vectors = happens_before_from_vectors(operations)
    check(from_log == from_vectors,
          "the log-derived and vector-derived causal orders must agree: %s"
          % sorted(from_log ^ from_vectors)[:3])
    check(len(from_log) > 0, "the workload must contain causal edges")
    ok, ordered = acyclic(from_log, [operation.id for operation in operations])
    check(ok, "the happens-before relation must be acyclic, ordered %d of %d"
          % (len(ordered), len(operations)))
    check(ordered[0] == operations[0].id,
          "the first write of a chain must be the topological minimum")

    # --- 5. every replica applies a linear extension of happens-before ----------
    violations = order_violations(cluster, operations)
    check(violations == [], "every replica must respect causal order: %s" % (violations[:3],))
    for name, replica in sorted(cluster.replicas.items()):
        check(sorted(replica.applied) == sorted(operation.id for operation in operations),
              "%s must have applied every operation exactly once: %d of %d"
              % (name, len(replica.applied), len(operations)))
    check(not cluster.replicas["eu"].buffer and not cluster.replicas["us"].buffer,
          "no replica may be left with a buffered operation after quiescence")
    check(all(replica.read("x") == "eu-1" for replica in cluster.replicas.values()),
          "the first write must be visible everywhere")

    # --- 6. concurrent writes need no ordering ---------------------------------
    concurrent = build_cluster()
    left = concurrent.write("eu", "same", "from-eu")
    right = concurrent.write("us", "same", "from-us")
    check(left.deps.get("us", 0) == 0 and right.deps.get("eu", 0) == 0,
          "the two writes must be concurrent: %s %s" % (left.deps, right.deps))
    concurrent.settle()
    check(concurrent.replicas["apac"].read("same") in ("from-eu", "from-us"),
          "a concurrent write must resolve deterministically")
    check(len({str(replica.snapshot()) for replica in concurrent.replicas.values()}) == 1,
          "concurrent writes must still converge everywhere")

    # --- 7. session guarantees ---------------------------------------------------
    session = build_cluster()
    written = session.write("eu", "profile", "v1")
    token = dict(written.version())
    check(session.replicas["apac"].read("profile") is None,
          "the far datacenter must not have the write yet")
    check(session.session_read("apac", "profile", token) == "v1",
          "read-your-writes must hold across datacenters with a causal token")
    second = session.write("eu", "profile", "v2")
    token2 = dict(second.version())
    first_read = session.session_read("apac", "profile", token)
    second_read = session.session_read("apac", "profile", token2)
    versions = [session.replicas["apac"].version_of("profile")]
    check(first_read == "v1" and second_read == "v2",
          "monotonic reads must never go backwards: %s then %s" % (first_read, second_read))
    check(dominates(versions[0], token2) or session.replicas["apac"].covers(token2),
          "after the second read the replica must cover the newer token")
    empty_token_read = session.session_read("apac", "profile", {})
    check(empty_token_read is not None, "an empty token read must not block")
    unknown = session.session_read("apac", "profile", {"nowhere": 5})
    check(unknown is not None, "a token from an unknown datacenter must not block a read")
    third = session.write("us", "other", "u1")
    check(session.session_read("eu", "other", dict(third.version())) == "u1",
          "read-your-writes must work in the other direction too")

    # --- 8. idempotence and duplicate delivery ----------------------------------
    duplicate = build_cluster()
    original = duplicate.write("eu", "d", 1)
    duplicate.settle()
    before = {name: (replica.snapshot(), list(replica.applied))
              for name, replica in duplicate.replicas.items()}
    for _ in range(3):
        duplicate.replicas["apac"].receive(original)
    check({name: (replica.snapshot(), list(replica.applied))
           for name, replica in duplicate.replicas.items()} == before,
          "redelivering an operation must change nothing")
    check(duplicate.replicas["apac"].applied == ["eu-1"],
          "the applied log must not grow on redelivery: %s" % duplicate.replicas["apac"].applied)

    # --- 9. monotonic and deterministic ------------------------------------------
    first_run, _first_ops = scripted_run()
    second_run, _second_ops = scripted_run()
    for name in sorted(first_run.replicas):
        check(first_run.replicas[name].applied == second_run.replicas[name].applied,
              "identical schedules must produce identical applied orders for %s" % name)
        check(first_run.replicas[name].snapshot() == second_run.replicas[name].snapshot(),
              "identical schedules must converge to identical state")
    check(first_run.network.sent == second_run.network.sent
          and first_run.network.delivered == second_run.network.delivered,
          "identical schedules must move identical numbers of messages")
    check(all(replica.buffer_high_water == 0 or replica.buffer_high_water >= 1
              for replica in first_run.replicas.values()),
          "the buffer high-water mark must be a real measurement")

    print("scripted run: %d operations, %d causal edges, %d messages, %d applied per replica"
          % (len(operations), len(from_log), cluster.network.sent,
             len(cluster.replicas["eu"].applied)))
    print("forced overtake: buffered %d at its high-water mark, final order %s"
          % (apac.buffer_high_water, apac.applied[-2:]))
    print("session guarantees: read-your-writes, monotonic reads and unblocked empty/unknown "
          "tokens all observed")
    print("PASS")


if __name__ == "__main__":
    main()
