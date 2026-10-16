"""Task 631 - Implement gossip protocol for membership and failure detection.  (Tier: Expert, Ch 9 — System Design and Scalability)

Each node keeps a membership table mapping every peer to (incarnation, status, heartbeat). Gossip
rounds are push-pull: a node sends its table to a few peers, the receiver merges it, and the tables
come back in the reply, so one round spreads a change by the fanout. Merging is a monotone
least-upper-bound: an entry is only overwritten by one with a greater key (incarnation, status rank,
heartbeat), so a stale or lower-incarnation message can never roll a view back. That ordering is what
makes failure detection safe: silence for `timeout` rounds marks a peer suspect, more silence marks it
failed, and only a node that is actually alive can clear the mark - it replies with the *same* status
carried at a higher incarnation, which dominates the suspicion. A partitioned or crashed node stops
being heard from, so the rest of the cluster converges on "failed" without it, and when the partition
heals the node refutes and the whole cluster converges again.

Deterministic stand-ins, stated explicitly: peers are picked by a seeded LCG instead of randomness,
delivery is immediate within a round (so a round is one network delay), and the round counter drives
the timeouts instead of a wall clock. The properties verified are the real ones: agreement (all live
nodes converge on the same membership, with the same incarnation and status per member), bounded
convergence rounds, monotonicity of every view, exclusion of crashed/partitioned nodes, refutation of
false suspicion, and reproducibility of the whole run from the seed.

Checks: hand-checked merge ordering (alive < suspect < failed < alive at a higher incarnation), a
stale-digest no-rollback negative check, view convergence from scratch, the bounded round count,
suspicion then failure of a partitioned node, rejoining after the partition heals, refutation of a
false suspicion, exclusion of the failed node from the agreement, message-count bounds, seeded
determinism, and single-node/two-node boundaries.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


ALIVE, SUSPECT, FAILED = "alive", "suspect", "failed"
RANK = {ALIVE: 1, SUSPECT: 2, FAILED: 3}


def entry_key(entry):
    """The monotone merge key: a higher incarnation always wins, then the status rank, then the
    heartbeat. 'failed' outranks 'suspect' outranks 'alive' at the same incarnation, which is what
    lets a suspicion spread, and only a higher incarnation can clear it again."""
    incarnation, status, heartbeat = entry
    return (incarnation, RANK[status], heartbeat)


class MemberView:
    """One node's membership table."""

    def __init__(self, node, incarnation=0, heartbeat=0):
        self.entries = {node: (incarnation, ALIVE, heartbeat)}
        self.merges = 0
        self.ignored = 0

    def snapshot(self):
        return dict(self.entries)

    def set(self, node, incarnation, status, heartbeat):
        candidate = (incarnation, status, heartbeat)
        current = self.entries.get(node)
        if current is not None and entry_key(candidate) <= entry_key(current):
            self.ignored += 1
            return False
        self.entries[node] = candidate
        return True

    def merge(self, digest):
        """Take the maximum of every entry: monotone, so views can never roll back."""
        changed = False
        for node, entry in digest.items():
            if self.set(node, entry[0], entry[1], entry[2]):
                changed = True
        self.merges += 1
        return changed

    def status(self, node):
        entry = self.entries.get(node)
        return entry[1] if entry else None

    def incarnation(self, node):
        entry = self.entries.get(node)
        return entry[0] if entry else None

    def members(self, status=ALIVE):
        return sorted(node for node, entry in self.entries.items() if entry[1] == status)

    def signature(self):
        """What agreement means: the same incarnation and status for every member."""
        return {node: (entry[0], entry[1]) for node, entry in self.entries.items()}


def lcg(seed, modulus=65537):
    state = seed

    def nxt(bound):
        nonlocal state
        state = (state * 75 + 74) % modulus
        return state % bound

    return nxt


class GossipCluster:
    """Deterministic push-pull gossip with SWIM-style suspicion, failure and refutation."""

    def __init__(self, nodes, fanout=2, seed=11, timeout=3, suspect_timeout=2):
        self.nodes = list(nodes)
        self.fanout = fanout
        self.timeout = timeout
        self.suspect_timeout = suspect_timeout
        self.views = {node: MemberView(node) for node in self.nodes}
        self.incarnations = {node: 0 for node in self.nodes}
        self.heartbeats = {node: 0 for node in self.nodes}
        self.last_heard = {node: {peer: 0 for peer in self.nodes} for node in self.nodes}
        self.alive = set(self.nodes)
        self.partitions = set()
        self.round = 0
        self.messages = 0
        self.refutations = 0
        self.trace = []
        self.pick = lcg(seed)

    # ---- topology --------------------------------------------------------------------------
    def partition(self, groups):
        self.partitions = set()
        for index, group in enumerate(groups):
            for other in groups[index + 1:]:
                for first in group:
                    for second in other:
                        self.partitions.add(frozenset((first, second)))

    def heal(self):
        self.partitions = set()

    def crash(self, node):
        self.alive.discard(node)

    def reachable(self, first, second):
        return (first in self.alive and second in self.alive
                and frozenset((first, second)) not in self.partitions)

    def targets(self, node):
        """Fanout peers, chosen by the seeded generator so every run is reproducible."""
        candidates = [peer for peer in self.nodes if peer != node and self.reachable(node, peer)]
        chosen = []
        for _ in range(min(self.fanout, len(candidates))):
            chosen.append(candidates.pop(self.pick(len(candidates))))
        return chosen

    # ---- one round -------------------------------------------------------------------------
    def tick(self):
        self.round += 1
        for node in self.nodes:
            if node in self.alive:
                self.heartbeats[node] += 1
                self.views[node].set(node, self.incarnations[node], ALIVE, self.heartbeats[node])
        for node in self.nodes:
            if node not in self.alive:
                continue
            for target in self.targets(node):
                self.exchange(node, target)
        self.detect()
        return self.round

    def exchange(self, node, target):
        """Push the table, pull the reply; a suspicion the receiver holds is refuted in passing."""
        self.last_heard[target][node] = self.round
        self.messages += 1
        self.views[target].merge(self.views[node].snapshot())
        if target in self.alive and self.views[target].status(node) != ALIVE:
            self.refute(node, target)
        self.last_heard[node][target] = self.round
        self.messages += 1
        self.views[node].merge(self.views[target].snapshot())
        if node in self.alive and self.views[node].status(target) != ALIVE:
            self.refute(target, node)
        self.trace.append((self.round, node, target))

    def refute(self, node, tell):
        """Only a live node can do this, and only with a higher incarnation."""
        self.incarnations[node] += 1
        self.views[node].set(node, self.incarnations[node], ALIVE, self.heartbeats[node])
        self.views[tell].merge(self.views[node].snapshot())
        self.refutations += 1
        self.messages += 1
        self.trace.append((self.round, node, "refute"))

    def detect(self):
        """Local failure detector: silence first means suspect, then failed.

        It evaluates every peer it still believes alive or suspect - including peers that are
        crashed or cut off, which is exactly how failures are discovered in the first place.
        """
        for observer in self.nodes:
            if observer not in self.alive:
                continue
            for peer in self.nodes:
                if peer == observer or self.views[observer].status(peer) == FAILED:
                    continue
                silence = self.round - self.last_heard[observer][peer]
                known = self.views[observer].incarnation(peer) or 0
                if silence >= self.timeout + self.suspect_timeout:
                    self.views[observer].set(peer, known, FAILED, 0)
                elif silence >= self.timeout:
                    self.views[observer].set(peer, known, SUSPECT, 0)

    def run(self, rounds):
        for _ in range(rounds):
            self.tick()
        return self

    def run_until_agree(self, max_rounds=60, expect_alive=None, observers=None):
        """Run until every node in `observers` (live nodes by default) has the same
        (incarnation, status) map - optionally also requiring a particular live set."""
        watchers = observers if observers is not None else sorted(self.alive)
        for _ in range(max_rounds):
            self.tick()
            signatures = [self.views[node].signature() for node in watchers]
            if all(sig == signatures[0] for sig in signatures):
                if expect_alive is None:
                    return self.round
                if sorted(node for node, entry in signatures[0].items()
                          if entry[1] == ALIVE) == expect_alive:
                    return self.round
        return None

    def agreement(self, observers=None):
        watchers = sorted(observers if observers is not None else self.alive)
        signatures = [self.views[node].signature() for node in watchers]
        if not signatures or not all(sig == signatures[0] for sig in signatures):
            return None
        return signatures[0]

    def live_members(self, observers=None):
        agreed = self.agreement(observers)
        if agreed is None:
            return None
        return sorted(node for node, entry in agreed.items() if entry[1] == ALIVE)


def main():
    # --- 1. merge ordering, hand-checked ------------------------------------------------------
    view = MemberView("A")
    check(view.entries["A"] == (0, ALIVE, 0), "a fresh view knows only itself")
    check(view.set("A", 0, ALIVE, 1) and view.set("A", 0, SUSPECT, 0),
          "a suspicion overrides an alive entry at the same incarnation")
    check(view.status("A") == SUSPECT, "and the view now reads suspect")
    check(view.set("A", 0, FAILED, 0) and view.status("A") == FAILED,
          "failure outranks suspicion at the same incarnation")
    check(not view.set("A", 0, ALIVE, 9),
          "an alive entry cannot clear a failure at the same incarnation - not even with a heartbeat")
    check(view.set("A", 1, ALIVE, 0) and view.status("A") == ALIVE,
          "only a higher incarnation clears the failure: that is the refutation")
    check(entry_key((0, ALIVE, 5)) < entry_key((0, SUSPECT, 0)) < entry_key((0, FAILED, 0))
          < entry_key((1, ALIVE, 0)), "the documented merge order holds")
    check(view.ignored >= 1, "the ignored update was counted, not silently dropped")

    stale = {"A": (0, ALIVE, 3)}                 # a digest older than what the view already holds
    check(view.merge(stale) is False and view.status("A") == ALIVE
          and view.incarnation("A") == 1, "a stale digest cannot roll the view back")
    check(view.merge({"B": (0, ALIVE, 1)}) is True and view.members() == ["A", "B"],
          "but a new member is added")

    # --- 2. convergence from scratch -----------------------------------------------------------
    cluster = GossipCluster(list(range(8)), fanout=2, seed=2024)
    rounds = cluster.run_until_agree(max_rounds=40, expect_alive=list(range(8)))
    check(rounds is not None, "the cluster converges on a full membership")
    check(rounds <= 12, f"eight nodes with fanout two converge quickly (took {rounds} rounds)")
    check(cluster.live_members() == list(range(8)), "every node is alive in the agreed view")
    check(all(cluster.views[node].members(ALIVE) == list(range(8)) for node in cluster.alive),
          "each individual view lists exactly the live set")
    check(all(status == ALIVE for node in range(8)
              for status in [cluster.views[node].status(node)]), "each node sees itself alive")
    check(cluster.refutations == 0, "a healthy cluster needs no refutations")

    # --- 3. determinism from the seed -----------------------------------------------------------
    twin = GossipCluster(list(range(8)), fanout=2, seed=2024).run(rounds)
    check(twin.trace == cluster.trace, "the same seed reproduces the same gossip trace exactly")
    check(twin.messages == cluster.messages and twin.refutations == cluster.refutations,
          "and the same message and refutation counts")
    check(twin.agreement() == cluster.agreement(), "and the same agreed membership")

    # --- 4. partition: suspect, then failed, and excluded from the agreement --------------------
    cluster = GossipCluster(list(range(6)), fanout=2, seed=99, timeout=2, suspect_timeout=2)
    cluster.run_until_agree(max_rounds=40)
    cluster.partition([[0], [1, 2, 3, 4, 5]])
    cluster.run(cluster.timeout)
    majority_status = min(cluster.views[node].status(0) for node in range(1, 6))
    check(all(cluster.views[node].status(0) in (SUSPECT, FAILED) for node in range(1, 6)),
          "the majority starts to suspect the isolated node as soon as the timeout expires")
    cluster.run(cluster.suspect_timeout + 1)
    check(all(cluster.views[node].status(0) == FAILED for node in range(1, 6)),
          "more silence turns the suspicion into a failure on the majority side")
    majority = cluster.run_until_agree(max_rounds=40, expect_alive=[1, 2, 3, 4, 5],
                                       observers=[1, 2, 3, 4, 5])
    check(majority is not None, "the majority converges on the membership without the isolated node")
    check(cluster.live_members(observers=[1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5],
          "the isolated node is excluded from the agreed live set")
    check(cluster.views[0].status(0) == ALIVE,
          "the isolated node still thinks it is alive - it knows nothing about the partition")
    check(cluster.views[0].status(1) == FAILED,
          "and it suspects the whole other side, which is exactly the partition's asymmetry")

    # --- 5. healing: the node refutes and the cluster converges again ---------------------------
    refutations_before = cluster.refutations
    cluster.heal()
    rejoined = cluster.run_until_agree(max_rounds=40, expect_alive=list(range(6)))
    check(rejoined is not None, "after healing the cluster agrees again")
    check(cluster.live_members() == list(range(6)), "the isolated node is back in the live set")
    check(cluster.refutations > refutations_before,
          "rejoining required a refutation with a higher incarnation")
    check(cluster.incarnations[0] > 0 and cluster.views[1].incarnation(0) == cluster.incarnations[0],
          "the refuted incarnation is the one everybody now holds")
    check(all(cluster.views[node].status(0) == ALIVE for node in range(6)),
          "no stale failure survives the refutation")

    # --- 6. a false suspicion is refuted without a partition ------------------------------------
    cluster = GossipCluster(list(range(5)), fanout=2, seed=7, timeout=2, suspect_timeout=4)
    cluster.run_until_agree(max_rounds=40)
    cluster.partition([[3], [0, 1, 2, 4]])       # node 3 goes quiet, but it is very much alive
    cluster.run(cluster.timeout + 1)
    suspected = [node for node in range(5)
                 if node != 3 and cluster.views[node].status(3) == SUSPECT]
    check(suspected, "the silent node gets suspected by at least one peer")
    check(all(cluster.views[node].status(3) != FAILED for node in range(5) if node != 3),
          "but the suspicion has not hardened into a failure yet")
    check(cluster.views[3].status(0) == SUSPECT, "and the isolation works in both directions")
    cluster.heal()                               # ... and it comes back before the failure timeout
    check(cluster.run_until_agree(max_rounds=40, expect_alive=list(range(5))) is not None,
          "the cluster agrees again after the false suspicion")
    check(all(cluster.views[node].status(3) == ALIVE for node in range(5)),
          "the suspicion is refuted once the node is heard from again")
    check(cluster.incarnations[3] >= 1, "and the refutation bumped its incarnation")

    # --- 7. a crashed node never comes back on its own ------------------------------------------
    cluster = GossipCluster(list(range(5)), fanout=2, seed=5, timeout=2, suspect_timeout=2)
    cluster.run_until_agree(max_rounds=40)
    cluster.crash(4)
    check(cluster.run_until_agree(max_rounds=40, expect_alive=[0, 1, 2, 3]) is not None,
          "the survivors converge on a four-node membership")
    check(all(node not in cluster.live_members() for node in [4]),
          "the crashed node is dropped from every agreed view")
    cluster.run(3)
    check(cluster.live_members() == [0, 1, 2, 3], "and it stays dropped - nothing revives it")
    check(all(cluster.views[node].status(4) == FAILED for node in range(4)),
          "the survivors all hold the same failure entry")

    # --- 8. message bounds and boundaries -------------------------------------------------------
    cluster = GossipCluster(list(range(16)), fanout=6, seed=31, timeout=8)
    cluster.run(6)
    check(cluster.messages <= 6 * 16 * (2 * 6 + 1),
          "messages stay within rounds * nodes * (2 * fanout + refutations)")
    check(cluster.messages >= 6 * 16, "and each node does send something every round")
    check(cluster.refutations == 0, "inside the timeout nobody is suspected, so nobody refutes")
    check(cluster.live_members() == list(range(16)), "and the membership stays complete")

    solo = GossipCluster([0], seed=1)
    check(solo.run_until_agree(max_rounds=5, expect_alive=[0]) == 1,
          "a single node agrees with itself on the first round")
    check(solo.messages == 0, "it has nobody to gossip with")

    pair = GossipCluster([0, 1], fanout=1, seed=2)
    check(pair.run_until_agree(max_rounds=10, expect_alive=[0, 1]) == 1,
          "two nodes converge after a single exchange")
    check(all(len(peer_view.entries) == 2 for peer_view in pair.views.values()),
          "both know both, and nobody else")

    print(f"gossip: rounds-to-agree={rounds} refutations={cluster.refutations} "
          f"messages(16 nodes, 6 rounds)={cluster.messages} final-alive={cluster.live_members()}")


if __name__ == "__main__":
    main()
    print("PASS")
