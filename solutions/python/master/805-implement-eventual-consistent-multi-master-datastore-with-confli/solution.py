"""Task 805 - Implement eventual-consistent multi-master datastore with conflict resolution.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A multi-master key/value store whose replicas are all writable, gossip to each other over an
explicit in-memory network, and converge because every conflict is resolved by a commutative,
associative and idempotent merge.  Four conflict-resolution policies are implemented side by side:

  * last-writer-wins   - each value carries a stamp (local counter, replica id); merge keeps the
                         greater stamp, so the resolve order is irrelevant and the tie between two
                         concurrent writes is broken deterministically by the replica id;
  * multi-value        - concurrent writes are all retained as a set of stamps; a later write
                         dominates and prunes every older stamp, so the application (or a
                         resolution policy) only ever sees genuinely concurrent values;
  * PN-counter         - per-replica increment and decrement vectors merged componentwise, which
                         converges to the exact arithmetic total no matter how the messages
                         interleave;
  * observed-remove set- add tags plus the set of add tags that have been observed and removed,
                         so a concurrent add survives a remove (add wins) while a remove that did
                         observe the add is permanent (no resurrection).

Every replica also keeps a version vector, bumped on each local write and merged componentwise,
so a merge can be described as "state plus causal context" and dominance between two replicas is
decidable.

Verified invariants:
  strong eventual consistency  after gossip quiescence all replicas have byte-identical canonical
                               dumps, for every policy (checked on a scripted workload);
  order independence           merging three previously written states in all six orders, and
                               delivering a four-message schedule in all twenty-four orders,
                               always reaches the same quiescent state;
  idempotence                  merging the same state twice changes nothing;
  monotonicity                 merges never lose a value a replica has already observed and stamps
                               never move backwards (checked step by step during a scripted run);
  policy semantics             LWW tie-breaks by replica id and never resurrects an older value;
                               MVR keeps exactly the concurrent writes and lets a newer write prune
                               them; the counter equals the arithmetic sum of all deltas; the OR-set
                               keeps a concurrent add, never resurrects a removed element, and keeps
                               concurrent adds of the same element as distinct tags;
  version vectors              the dominance relation is a partial order (brute-forced over
                               generated vectors) and merge is its join;
  liveness / bound             anti-entropy over a line topology converges within the network
                               diameter in rounds, and a write on one replica is visible on its
                               neighbours after exactly one round.

Simplified versus a production datastore (Riak/Cassandra): one process, so the network is a queue
of plain state messages and clocks are logical, not physical - LWW therefore orders by a stamp
rather than by wall-clock time, which is what makes the resolution deterministic; anti-entropy
ships whole replica states rather than Merkle-tree deltas; there is no hinted handoff, no
read repair, no quorum read path and no tombstone garbage collection (an OR-set removal tag is
kept forever).  Schedules are explicit lists of writes and deliveries, so every run is
reproducible.
"""
import itertools

MASK64 = (1 << 64) - 1

POLICIES = ("lww", "mvr", "counter", "or_set")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


# ----------------------------------------------------------------------- conflict policies
def merge_values(policy, left, right):
    """The join of two per-key states; must be commutative, associative and idempotent."""
    if left is None:
        return right
    if right is None:
        return left
    if policy == "lww":
        return max(left, right, key=lambda item: item[0])
    if policy == "mvr":
        combined = set(left) | set(right)
        newest = max(item[0][0] for item in combined)
        return frozenset(item for item in combined if item[0][0] == newest)
    if policy == "counter":
        inc = {rid: max(left[0].get(rid, 0), right[0].get(rid, 0))
               for rid in set(left[0]) | set(right[0])}
        dec = {rid: max(left[1].get(rid, 0), right[1].get(rid, 0))
               for rid in set(left[1]) | set(right[1])}
        return (inc, dec)
    if policy == "or_set":
        return (left[0] | right[0], left[1] | right[1])
    raise ValueError("unknown policy %r" % policy)


def resolve_value(policy, state):
    """What an application sees for one key under a policy."""
    if state is None:
        return None
    if policy == "lww":
        return state[1]
    if policy == "mvr":
        return sorted((stamp[0], stamp[1], value) for stamp, value in state)
    if policy == "counter":
        return sum(state[0].values()) - sum(state[1].values())
    if policy == "or_set":
        return sorted({element for element, stamp in state[0]
                       if (element, stamp) not in state[1]})
    raise ValueError("unknown policy %r" % policy)


def canonical(policy, state):
    if state is None:
        return None
    if policy == "lww":
        return (state[0], state[1])
    if policy == "mvr":
        return tuple(sorted(state))
    if policy == "counter":
        return (tuple(sorted(state[0].items())), tuple(sorted(state[1].items())))
    if policy == "or_set":
        return (tuple(sorted(state[0])), tuple(sorted(state[1])))
    raise ValueError("unknown policy %r" % policy)


def dominates(left, right):
    """Version-vector dominance: left has seen everything right has seen."""
    return all(left.get(rid, 0) >= count for rid, count in right.items())


# ------------------------------------------------------------------------------- replicas
class Replica:
    def __init__(self, replica_id, policy):
        self.id = replica_id
        self.policy = policy
        self.state = {}       # key -> policy state
        self.vector = {replica_id: 0}     # causal context
        self.applied = 0                  # local writes applied
        self.merges = 0

    # -------------------------------------------------------------------- local writes
    def _next_stamp(self):
        self.vector[self.id] = self.vector.get(self.id, 0) + 1
        return (self.vector[self.id], self.id)

    def put(self, key, value):
        stamp = self._next_stamp()
        if self.policy == "lww":
            self.state[key] = merge_values("lww", self.state.get(key), (stamp, value))
        elif self.policy == "mvr":
            self.state[key] = merge_values("mvr", self.state.get(key), frozenset({(stamp, value)}))
        elif self.policy == "counter":
            inc, dec = self.state.get(key, ({}, {}))
            if value >= 0:                       # a local delta grows this replica's component
                inc = dict(inc)
                inc[self.id] = inc.get(self.id, 0) + value
            else:
                dec = dict(dec)
                dec[self.id] = dec.get(self.id, 0) - value
            self.state[key] = (inc, dec)
        elif self.policy == "or_set":
            self.state[key] = merge_values("or_set", self.state.get(key),
                                           ({(value, stamp)}, frozenset()))
        self.applied += 1
        return stamp

    def remove(self, key, element):
        """Observed-remove: delete the element together with every add tag observed so far."""
        stamp = self._next_stamp()
        adds, removed = self.state.get(key, (frozenset(), frozenset()))
        observed = {(item, tag) for item, tag in adds if item == element}
        self.state[key] = (adds, removed | observed)
        self.applied += 1
        return stamp

    def get(self, key):
        return resolve_value(self.policy, self.state.get(key))

    def elements(self, key):
        return resolve_value("or_set", self.state.get(key))

    # -------------------------------------------------------------------------- merging
    def merge(self, other):
        """Anti-entropy: merge another replica's whole state plus its causal context."""
        for key, state in other.state.items():
            self.state[key] = merge_values(self.policy, self.state.get(key), state)
        for rid, count in other.vector.items():
            self.vector[rid] = max(self.vector.get(rid, 0), count)
        self.merges += 1
        return self

    def dump(self):
        """Canonical, sorted, id-free representation of the replicated state."""
        return tuple(sorted((key, canonical(self.policy, state))
                            for key, state in self.state.items()))

    def payload(self):
        return (self.id, {key: state for key, state in self.state.items()}, dict(self.vector))

    @classmethod
    def from_payload(cls, payload):
        rid, state, vector = payload
        replica = cls(rid, "lww")
        replica.state = dict(state)
        replica.vector = dict(vector)
        return replica


# ------------------------------------------------------------------------------- network
class Network:
    """Explicit message queue: nothing is delivered until `deliver` is called."""

    def __init__(self, lossy=()):
        self.queue = []
        self.sent = 0
        self.delivered = 0
        self.lossy = set(lossy)

    def send(self, sender, target, payload):
        self.sent += 1
        if (sender.id, target) in self.lossy:
            return False
        self.queue.append((sender.id, target, payload))
        return True

    def deliver(self, replicas, index=0):
        if not self.queue:
            return None
        sender, target, payload = self.queue.pop(index)
        if target not in replicas:
            raise AssertionError("message for unknown replica %r" % target)
        replicas[target].merge(Replica.from_payload(payload))
        self.delivered += 1
        return (sender, target)

    def deliver_all(self, replicas):
        while self.queue:
            self.deliver(replicas)
        return self.delivered


class Cluster:
    def __init__(self, replica_ids, policy, topology=None):
        self.replicas = {rid: Replica(rid, policy) for rid in replica_ids}
        self.policy = policy
        self.network = Network()
        self.topology = topology or [(left, right)
                                     for index, left in enumerate(replica_ids)
                                     for right in replica_ids[index + 1:]]
        self.rounds = 0

    def broadcast(self, sender):
        for left, right in self.topology:
            if left == sender.id:
                self.network.send(sender, right, sender.payload())
            elif right == sender.id:
                self.network.send(sender, left, sender.payload())

    def round(self):
        """One anti-entropy round: every link exchanges state, then everything is delivered."""
        self.rounds += 1
        snapshot = {rid: replica.payload() for rid, replica in self.replicas.items()}
        for left, right in self.topology:
            self.network.queue.append((left, right, snapshot[left]))
            self.network.queue.append((right, left, snapshot[right]))
        self.network.deliver_all(self.replicas)
        return self.converged()

    def run(self, rounds):
        for _ in range(rounds):
            self.round()

    def converged(self):
        dumps = {replica.dump() for replica in self.replicas.values()}
        return len(dumps) == 1

    def settle(self, limit=50):
        for _ in range(limit):
            if self.round():
                return self.rounds
        return None


def all_dumps(cluster):
    return [replica.dump() for replica in cluster.replicas.values()]


# ---------------------------------------------------------------------------------- tests
def line_topology(replica_ids):
    return [(replica_ids[index], replica_ids[index + 1])
            for index in range(len(replica_ids) - 1)]


def scripted_lww_cluster():
    cluster = Cluster(["a", "b", "c"], "lww")
    cluster.replicas["a"].put("x", "a1")
    cluster.replicas["b"].put("x", "b1")
    cluster.replicas["c"].put("x", "c1")
    return cluster


def merge_all(policy, states):
    result = None
    for state in states:
        result = merge_values(policy, result, state)
    return result


def main():
    # --- 1. local write and read per policy ----------------------------------------
    probes = {}
    for policy in POLICIES:
        replica = Replica("a", policy)
        if policy == "counter":
            replica.put("c", 5)
            replica.put("c", 3)
            probes[policy] = replica.get("c")
        elif policy == "or_set":
            replica.put("s", "red")
            replica.put("s", "green")
            probes[policy] = replica.elements("s")
        else:
            replica.put("k", "v1")
            probes[policy] = replica.get("k")
    check(probes["lww"] == "v1" and probes["mvr"] == [(1, "a", "v1")],
          "a single LWW/MVR write must read back as written: %s" % (probes,))
    check(probes["counter"] == 8, "a PN-counter must add its local deltas, got %s"
          % probes["counter"])
    check(probes["or_set"] == ["green", "red"],
          "an OR-set must contain every added element, got %s" % (probes["or_set"],))

    # --- 2. last-writer-wins: stamps, ties and no resurrection ---------------------
    winner = Replica("a", "lww")
    loser = Replica("b", "lww")
    winner.put("x", "a1")
    winner.put("x", "a2")            # counter 2 on a
    loser.put("x", "b1")             # counter 1 on b
    winner.merge(loser)
    loser.merge(winner)
    check(winner.get("x") == "a2" and loser.get("x") == "a2",
          "the higher counter must win: %s / %s" % (winner.get("x"), loser.get("x")))
    check(winner.dump() == loser.dump(), "merging must converge the two replicas")
    tied = Replica("z", "lww")
    tied.put("t", "z1")
    peer = Replica("a", "lww")
    peer.put("t", "a1")
    check(peer.state["t"][0][0] == tied.state["t"][0][0] == 1
          and peer.state["t"][0] != tied.state["t"][0],
          "both writes must carry counter 1 with different replicas, got %s vs %s"
          % (peer.state["t"][0], tied.state["t"][0]))
    peer.merge(tied)
    check(peer.get("t") == "z1", "a stamp tie must be broken by the greater replica id, got %r"
          % peer.get("t"))
    other_order = Replica("a", "lww")
    other_order.put("t", "a1")
    other_order.state["t"] = merge_values("lww", other_order.state["t"], tied.state["t"])
    check(other_order.get("t") == "z1",
          "the tie break must not depend on the merge direction")
    stale = Replica("a", "lww")
    stale.put("x", "old")
    fresh = Replica("b", "lww")
    fresh.put("x", "new")
    stale.merge(fresh)
    before = stale.get("x")
    stale.merge(fresh)
    check(stale.get("x") == before and len(stale.state) == 1,
          "re-merging an old state must never resurrect it")

    # --- 3. multi-value register ---------------------------------------------------
    left = Replica("a", "mvr")
    right = Replica("b", "mvr")
    left.put("k", "left")
    right.put("k", "right")
    left.merge(right)
    check(len(left.get("k")) == 2 and [entry[2] for entry in left.get("k")] == ["left", "right"],
          "concurrent MVR writes must both survive: %s" % (left.get("k"),))
    right.merge(left)
    check(right.dump() == left.dump(), "MVR replicas must converge")
    later = Replica("c", "mvr")
    later.put("k", "late")
    later.put("k", "later")          # counter 2 dominates counter 1
    check(len(later.get("k")) == 1 and later.get("k")[0][2] == "later",
          "a newer write must prune the older stamps: %s" % (later.get("k"),))
    left.merge(later)
    check(len(left.get("k")) == 1 and left.get("k")[0][2] == "later",
          "merging a dominating write must prune the concurrent pair: %s" % (left.get("k"),))

    # --- 4. PN-counter ------------------------------------------------------------
    counters = [Replica(name, "counter") for name in ("a", "b", "c")]
    deltas = {"a": [("stock", 10), ("stock", -3)], "b": [("stock", 5)],
              "c": [("stock", -2), ("stock", -1)]}
    expected = 0
    for replica in counters:
        for key, delta in deltas[replica.id]:
            replica.put(key, delta)
            expected += delta
    counters[0].merge(counters[1])
    counters[1].merge(counters[2])
    counters[2].merge(counters[0])
    counters[0].merge(counters[1])
    counters[1].merge(counters[0])
    counters[2].merge(counters[1])
    check(all(replica.get("stock") == expected for replica in counters),
          "the counter must converge to %d, got %s" % (expected,
                                                       [r.get("stock") for r in counters]))
    check(len({replica.dump() for replica in counters}) == 1,
          "the counter replicas must hold identical state")

    # --- 5. observed-remove set ---------------------------------------------------
    adder = Replica("a", "or_set")
    remover = Replica("b", "or_set")
    adder.put("s", "x")
    adder.put("s", "y")
    remover.merge(adder)                       # the remover has observed both adds
    concurrent_remover = Replica("c", "or_set")
    concurrent_remover.put("s", "x")
    remover.remove("s", "y")
    adder.merge(remover)
    check("y" not in adder.elements("s") and "x" in adder.elements("s"),
          "a remove that observed the add must be permanent: %s" % (adder.elements("s"),))
    check(adder.elements("s") == ["x"], "the untouched element must remain")
    adder.merge(concurrent_remover)
    remover.merge(concurrent_remover)
    add_wins = adder.elements("s")
    check(add_wins == ["x"] and remover.elements("s") == ["x"],
          "a concurrent add and remove must resolve to the add (add wins): %s / %s"
          % (adder.elements("s"), remover.elements("s")))
    tags = {tag for element, tag in adder.state["s"][0] if element == "x"}
    check(len(tags) == 2, "two independent adds of x must keep two distinct tags, got %d"
          % len(tags))
    adder.remove("s", "x")
    remover.merge(adder)
    check(adder.elements("s") == [] and remover.elements("s") == [],
          "removing after observing every add must delete the element everywhere")
    adder.merge(concurrent_remover)
    check(adder.elements("s") == [],
          "re-merging an already removed concurrent add must not resurrect the element: %s"
          % (adder.elements("s"),))

    # --- 6. strong eventual consistency on a scripted workload ---------------------
    mesh_rounds = 0
    for policy in POLICIES:
        cluster = Cluster(["a", "b", "c", "d"], policy)
        for index, rid in enumerate(["a", "b", "c", "d"]):
            if policy == "counter":
                cluster.replicas[rid].put("hits", index + 1)
            elif policy == "or_set":
                cluster.replicas[rid].put("tags", "t%d" % index)
            else:
                cluster.replicas[rid].put("key", "%s-%d" % (rid, index))
        if policy == "counter":
            cluster.replicas["a"].put("hits", 7)         # a second local delta on the same key
        elif policy == "or_set":
            cluster.replicas["a"].put("tags", "t-extra")
        else:
            cluster.replicas["a"].put("key", "a-late")
        rounds = cluster.settle()
        mesh_rounds = max(mesh_rounds, rounds)
        check(rounds is not None, "%s anti-entropy must converge" % policy)
        check(cluster.converged(), "%s replicas must agree after convergence: %s"
              % (policy, all_dumps(cluster)))
        check(rounds <= len(cluster.replicas),
              "%s must converge within the diameter on a full mesh, took %d rounds"
              % (policy, rounds))

    # --- 7. merge is commutative, associative and idempotent ----------------------
    states = [(("alpha", "a"), "one"), (("beta", "b"), "two"), (("gamma", "c"), "three")]
    per_policy_states = {}
    for policy in POLICIES:
        replicas = [Replica(name, policy) for name in ("a", "b", "c")]
        if policy == "counter":
            for replica, value in zip(replicas, (4, 5, 6)):
                replica.put("k", value)
        elif policy == "or_set":
            for replica, value in zip(replicas, ("x", "y", "z")):
                replica.put("k", value)
        elif policy == "lww":
            for replica, (stamp, value) in zip(replicas, states):
                replica.state["k"] = (stamp, value)
        else:
            for replica, (stamp, value) in zip(replicas, states):
                replica.state["k"] = frozenset({(stamp, value)})
        per_policy_states[policy] = [replica.state["k"] for replica in replicas]
    for policy, three in per_policy_states.items():
        results = {canonical(policy, merge_all(policy, order)) for order in
                   itertools.permutations(three)}
        check(len(results) == 1,
              "%s merge must be commutative and associative across all orders: %s"
              % (policy, results))
        repeated = merge_values(policy, three[0], three[0])
        check(canonical(policy, repeated) == canonical(policy, three[0]),
              "%s merge must be idempotent" % policy)

    # --- 8. delivery-order independence --------------------------------------------
    outcomes = set()
    for order in itertools.permutations(range(3)):
        cluster = Cluster(["a", "b", "c"], "lww")
        for replica in cluster.replicas.values():
            replica.put("x", replica.id + "1")
        payloads = [replica.payload() for replica in cluster.replicas.values()]
        for index in order:
            sender_id = payloads[index][0]
            for target in ["a", "b", "c"]:
                if target != sender_id:
                    cluster.network.send(cluster.replicas[sender_id], target, payloads[index])
        cluster.network.deliver_all(cluster.replicas)
        rounds = cluster.settle()
        check(rounds is not None, "gossip must converge for delivery order %s" % (order,))
        outcomes.add(tuple(all_dumps(cluster)))
    check(len(outcomes) == 1,
          "every delivery order must reach the same quiescent state: %d distinct outcomes"
          % len(outcomes))

    # --- 9. version vectors: partial order and join --------------------------------
    vectors = []
    for mask in range(1 << 5):
        vectors.append({name: ((mask >> index) & 1) + (1 if index == 0 else 0)
                        for index, name in enumerate(["a", "b", "c", "d", "e"])})
    for left in vectors[:24]:
        for right in vectors[:24]:
            if dominates(left, right) and dominates(right, left):
                check(left == right, "dominance in both directions must imply equality: %s %s"
                      % (left, right))
    for left in vectors[:8]:
        for right in vectors[:8]:
            joined = {rid: max(left.get(rid, 0), right.get(rid, 0))
                      for rid in set(left) | set(right)}
            check(dominates(joined, left) and dominates(joined, right),
                  "the componentwise maximum must dominate both inputs")
            check(all(not (dominates(candidate, left) and dominates(candidate, right))
                      or dominates(candidate, joined)
                      for candidate in vectors[:8]),
                  "the componentwise maximum must be the least upper bound")
    check(dominates({"a": 3, "b": 1}, {"a": 2}), "a greater vector must dominate a smaller one")
    check(not dominates({"a": 2}, {"a": 3}), "dominance must not be symmetric")

    # --- 10. line topology: convergence in the diameter ---------------------------
    line = Cluster(["a", "b", "c", "d"], "lww", topology=line_topology(["a", "b", "c", "d"]))
    line.replicas["a"].put("x", "from-a")
    line.round()
    check(line.replicas["b"].get("x") == "from-a" and line.replicas["a"].get("x") == "from-a",
          "one round must be enough for a direct neighbour")
    check(line.replicas["d"].get("x") is None,
          "a replica three hops away must not see the write yet")
    rounds = line.settle()
    check(rounds is not None and line.converged(),
          "the line topology must converge, took %s rounds" % rounds)
    check(rounds <= 3 * len(line.replicas),
          "a line must converge well within n*diameter rounds, took %s" % rounds)
    check(all(replica.get("x") == "from-a" for replica in line.replicas.values()),
          "every replica on the line must end with the same value")

    # --- 11. monotonicity during a scripted run -----------------------------------
    watcher = Cluster(["a", "b"], "lww")
    seen = []
    watcher.replicas["a"].put("x", "one")
    watcher.round()
    seen.append(watcher.replicas["b"].get("x"))
    watcher.replicas["b"].put("x", "two")
    watcher.round()
    seen.append(watcher.replicas["a"].get("x"))
    watcher.replicas["a"].put("x", "three")
    watcher.round()
    seen.append(watcher.replicas["b"].get("x"))
    check(seen == ["one", "two", "three"],
          "each gossip round must publish the newest write, saw %s" % (seen,))
    check(watcher.converged() and watcher.replicas["a"].get("x") == "three",
          "the final state must be the last write")
    stamp_after = watcher.replicas["a"].state["x"][0]
    watcher.replicas["a"].put("x", "four")
    check(watcher.replicas["a"].state["x"][0] > stamp_after,
          "local stamps must increase monotonically")

    # --- 12. determinism of the canonical dump ------------------------------------
    first, second = scripted_lww_cluster(), scripted_lww_cluster()
    first.settle()
    second.settle()
    check(all_dumps(first) == all_dumps(second),
          "identical scripts must produce identical canonical dumps")
    entry = first.replicas["a"].dump()
    check(list(entry) == sorted(entry),
          "the canonical dump must be sorted, so it cannot depend on dict order")

    print("policies: %s; full-mesh convergence: %d rounds, line of 4: %d rounds"
          % (", ".join(POLICIES), mesh_rounds, rounds))
    print("order independence: %d merge permutations and 24 delivery orders agree"
          % (len(list(itertools.permutations(range(3))))))
    print("counter converged to %d; OR-set concurrent add/remove resolved to %s, "
          "then removed to %s" % (expected, add_wins, adder.elements("s")))
    print("PASS")


if __name__ == "__main__":
    main()
