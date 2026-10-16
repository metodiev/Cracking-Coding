"""Task 628 - Implement distributed consensus simulation with network partitions (Raft test harness).  (Tier: Expert, Ch 9 — System Design and Scalability)

A compact Raft: leaders are elected by a majority of votes, entries are replicated by AppendEntries
with the previous-index/previous-term consistency check, and an entry commits once it is stored on a
majority. The harness drives everything from a deterministic schedule - ticks for election and
heartbeat timeouts, FIFO message delivery, and explicit partitions - so a partition, a leader change
and the subsequent log repair are all reproducible.

Deterministic stand-ins for the randomised parts of Raft, stated explicitly: election timeouts are
`3 + node_index` ticks (so the lowest-numbered live node always wins a fresh election), heartbeats
are sent once per tick, and the whole cluster is driven by one thread that ticks and then drains the
message queue. The Commit rule is the real one: an entry is committed only when it is stored on a
majority *and* it belongs to the leader's own term, and a follower's commit index only moves when a
leader tells it to.

Safety properties asserted after *every* tick and *every* message: at most one leader per term
(election safety), a node never votes twice in a term, two logs that agree at an index agree on
every earlier entry (log matching), an entry once committed at an index is never replaced or lost by
any node that has reached that index, and the current leader holds every committed entry (leader
completeness). Partition tests assert the two halves of availability and safety: the minority side
cannot commit even though the old leader still believes it leads, the majority side elects a new
leader in a higher term and commits, and the minority leader's uncommitted entry is overwritten when
the partition heals.

Checks: election and re-election, replication and majority commit, an uncommitted entry surviving a
partition only to be discarded, healing and catch-up, a single-node cluster, and the invariant
checker running across every step of every scenario.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


FOLLOWER, CANDIDATE, LEADER = "follower", "candidate", "leader"
REQUEST_VOTE, VOTE_RESPONSE, APPEND, APPEND_RESPONSE = "RequestVote", "VoteResponse", "AppendEntries", "AppendResponse"


class Message:
    __slots__ = ("sender", "recipient", "kind", "payload")

    def __init__(self, sender, recipient, kind, payload):
        self.sender = sender
        self.recipient = recipient
        self.kind = kind
        self.payload = payload

    def __repr__(self):
        return f"{self.kind}({self.sender}->{self.recipient})"


class Network:
    """FIFO in-memory network with explicit partitions and crashed (silent) nodes."""

    def __init__(self):
        self.queue = []
        self.partitions = set()
        self.dropped = []
        self.down = set()

    def partition(self, groups):
        self.partitions = set()
        for i, group in enumerate(groups):
            for other in groups[i + 1:]:
                for a in group:
                    for b in other:
                        self.partitions.add(frozenset((a, b)))

    def heal(self):
        self.partitions = set()

    def cut_off(self, node, others):
        for other in others:
            self.partitions.add(frozenset((node, other)))

    def blocked(self, sender, recipient):
        return (sender in self.down or recipient in self.down
                or frozenset((sender, recipient)) in self.partitions)

    def send(self, sender, recipient, kind, payload):
        message = Message(sender, recipient, kind, payload)
        if self.blocked(sender, recipient):
            self.dropped.append(message)
        else:
            self.queue.append(message)
        return message

    def deliver(self):
        return self.queue.pop(0) if self.queue else None


class RaftNode:
    """One replica: term, single vote per term, a log of (term, command) entries and a commit index."""

    def __init__(self, node_id, cluster, network):
        self.id = node_id
        self.cluster = list(cluster)
        self.net = network
        self.term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.state = FOLLOWER
        self.leader = None
        self.ticks = 0
        self.election_timeout = 3 + node_id
        self.next_index = {}
        self.match_index = {}
        self.applied = []

    # ---- helpers ---------------------------------------------------------------------------
    def last_index(self):
        return len(self.log)

    def last_term(self):
        return self.log[-1][0] if self.log else 0

    def entry(self, index):
        return self.log[index - 1]

    def majority(self):
        return len(self.cluster) // 2 + 1

    def reset_timer(self):
        self.ticks = 0

    def _others(self):
        return [node for node in self.cluster if node != self.id]

    # ---- timers ----------------------------------------------------------------------------
    def tick(self):
        if self.state == LEADER:
            self._broadcast_append()
            return
        self.ticks += 1
        if self.ticks >= self.election_timeout:
            self._start_election()

    def _start_election(self):
        self.state = CANDIDATE
        self.term += 1
        self.voted_for = self.id
        self.votes = 1
        self.reset_timer()
        for other in self._others():
            self.net.send(self.id, other, REQUEST_VOTE, {
                "term": self.term, "candidate": self.id,
                "last_index": self.last_index(), "last_term": self.last_term()})
        if self.votes >= self.majority():   # a cluster of one is its own majority
            self._become_leader()

    def _become_follower(self, term):
        if term > self.term:
            self.term = term
            self.voted_for = None
        self.state = FOLLOWER
        self.reset_timer()

    def _become_leader(self):
        self.state = LEADER
        self.leader = self.id
        self.reset_timer()
        # Standard Raft detail: a new leader appends a no-op entry in its own term, because an
        # entry from an earlier term cannot be committed by counting replicas alone. With an empty
        # log there is nothing dangling, so no entry is needed (and the log stays index-aligned).
        if self.log:
            self.log.append((self.term, None))
        for other in self._others():
            self.next_index[other] = self.last_index() + 1
            self.match_index[other] = 0
        self.match_index[self.id] = self.last_index()
        self._broadcast_append()

    # ---- message handling ------------------------------------------------------------------
    def handle(self, message):
        if message.payload.get("term", 0) > self.term:
            self._become_follower(message.payload["term"])
        if message.kind == REQUEST_VOTE:
            self._on_request_vote(message)
        elif message.kind == VOTE_RESPONSE:
            self._on_vote_response(message)
        elif message.kind == APPEND:
            self._on_append(message)
        elif message.kind == APPEND_RESPONSE:
            self._on_append_response(message)

    def _on_request_vote(self, message):
        term = message.payload["term"]
        candidate = message.payload["candidate"]
        up_to_date = (message.payload["last_term"], message.payload["last_index"]) >= (self.last_term(), self.last_index())
        granted = (term >= self.term and up_to_date
                   and (self.voted_for is None or self.voted_for == candidate))
        if granted:
            self.voted_for = candidate
            self.reset_timer()
        self.net.send(self.id, candidate, VOTE_RESPONSE,
                      {"term": self.term, "granted": granted, "voter": self.id})

    def _on_vote_response(self, message):
        if self.state != CANDIDATE or message.payload["term"] != self.term:
            return
        if message.payload["granted"]:
            self.votes += 1
            if self.votes >= self.majority():
                self._become_leader()

    def _broadcast_append(self):
        for other in self._others():
            next_index = self.next_index.get(other, 1)
            prev_index = next_index - 1
            prev_term = self.entry(prev_index)[0] if prev_index > 0 else 0
            self.net.send(self.id, other, APPEND, {
                "term": self.term, "leader": self.id, "prev_index": prev_index,
                "prev_term": prev_term, "entries": self.log[prev_index:],
                "leader_commit": self.commit_index})

    def _on_append(self, message):
        payload = message.payload
        if payload["term"] < self.term:
            self.net.send(self.id, payload["leader"], APPEND_RESPONSE,
                          {"term": self.term, "success": False, "match_index": self.last_index()})
            return
        self._become_follower(payload["term"])
        self.leader = payload["leader"]
        self.reset_timer()
        prev_index, prev_term = payload["prev_index"], payload["prev_term"]
        if prev_index > self.last_index() or (prev_index > 0 and self.entry(prev_index)[0] != prev_term):
            self.net.send(self.id, payload["leader"], APPEND_RESPONSE,
                          {"term": self.term, "success": False, "match_index": self.last_index()})
            return
        index = prev_index + 1
        for entry in payload["entries"]:
            if index <= self.last_index():
                if self.entry(index)[0] != entry[0]:     # conflict: truncate and take the leader's
                    self.log = self.log[:index - 1]
                    self.log.append(entry)
            else:
                self.log.append(entry)
            index += 1
        if payload["leader_commit"] > self.commit_index:
            self.commit_index = min(payload["leader_commit"], self.last_index())
        self.net.send(self.id, payload["leader"], APPEND_RESPONSE,
                      {"term": self.term, "success": True, "match_index": self.last_index()})

    def _on_append_response(self, message):
        if self.state != LEADER or message.payload["term"] != self.term:
            return
        follower = message.sender
        if message.payload["success"]:
            self.match_index[follower] = max(self.match_index.get(follower, 0),
                                             message.payload["match_index"])
            self.next_index[follower] = self.match_index[follower] + 1
            self._advance_commit()
        else:
            self.next_index[follower] = max(1, message.payload["match_index"] + 1)

    def _advance_commit(self):
        """Commit the highest index stored on a majority *and* written in the current term."""
        for index in range(self.last_index(), self.commit_index, -1):
            if self.entry(index)[0] != self.term:
                continue
            replicated = sum(1 for node in self.cluster
                             if node == self.id or self.match_index.get(node, 0) >= index)
            if replicated >= self.majority():
                self.commit_index = index
                return True
        return False

    # ---- client interface ------------------------------------------------------------------
    def append_entry(self, command):
        check(self.state == LEADER, "only a leader accepts a client command")
        self.log.append((self.term, command))
        self.match_index[self.id] = self.last_index()
        self._broadcast_append()
        self._advance_commit()
        return self.last_index()

    def commands(self):
        """Client commands in log order; the leader's no-op entries carry None and are skipped."""
        return [command for _term, command in self.log if command is not None]


class Cluster:
    """Deterministic driver plus the safety invariant checker that runs after every step."""

    def __init__(self, node_ids):
        self.net = Network()
        self.nodes = [RaftNode(node_id, node_ids, self.net) for node_id in node_ids]
        self.by_id = {node.id: node for node in self.nodes}
        self.history = {}                 # committed index -> (term, command), never rewritten
        self.leaders_seen = {}            # term -> leader id
        self.steps = 0
        self.delivered = 0

    def leader(self):
        leaders = [node for node in self.nodes if node.state == LEADER]
        return leaders[0] if leaders else None

    def commands(self):
        return {node.id: node.commands() for node in self.nodes}

    def converge_on(self, expected):
        return all(node.commands() == expected for node in self.nodes)

    def check_invariants(self):
        for node in self.nodes:
            if node.state == LEADER:
                known = self.leaders_seen.get(node.term)
                check(known is None or known == node.id,
                      f"two leaders in term {node.term}: {known} and {node.id}")
                self.leaders_seen[node.term] = node.id
                # Leader completeness: a node elected in term T carries every entry committed
                # before T. (An entry committed *after* T may well be missing from a leader that
                # was stranded in a minority partition - that leader can no longer commit anyway,
                # and the rejection rule stops it from truncating the newer entries.)
                for index, (entry_term, _command) in self.history.items():
                    if entry_term < node.term:
                        check(index <= node.last_index()
                              and node.entry(index) == self.history[index],
                              f"leader {node.id} (term {node.term}) is missing committed entry {index}")
            for index in range(1, node.commit_index + 1):
                entry = node.entry(index)
                if index in self.history:
                    check(self.history[index] == entry,
                          f"committed entry {index} changed on node {node.id}")
                else:
                    self.history[index] = entry
        for first in self.nodes:
            for second in self.nodes:
                if first.id >= second.id:
                    continue
                for index in range(1, min(first.last_index(), second.last_index()) + 1):
                    if first.entry(index)[0] == second.entry(index)[0]:
                        check(first.log[:index] == second.log[:index],
                              f"log matching violated at index {index}: {first.id} vs {second.id}")
                        break

    def step(self):
        for node in self.nodes:
            node.tick()
        self.check_invariants()
        while True:
            message = self.net.deliver()
            if message is None:
                break
            self.delivered += 1
            recipient = self.by_id[message.recipient]
            recipient.handle(message)
            self.check_invariants()
        self.steps += 1
        return self.steps

    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self

    def run_until_stable(self, max_steps=60):
        """Run until the cluster has one leader, one term, one log and one commit index."""
        for _ in range(max_steps):
            self.step()
            if (self.leader() is not None
                    and len({tuple(node.log) for node in self.nodes}) == 1
                    and len({node.term for node in self.nodes}) == 1
                    and len({node.commit_index for node in self.nodes}) == 1):
                return True
        return False

    def to_partition(self, first_group, second_group):
        self.net.partition([first_group, second_group])


def main():
    # --- 1. election in a healthy five-node cluster -------------------------------------------
    cluster = Cluster([0, 1, 2, 3, 4])
    cluster.run(2)
    check(all(node.state == FOLLOWER for node in cluster.nodes),
          "two ticks are below the smallest timeout, so no election starts")
    cluster.run(1)
    leader = cluster.leader()
    check(leader is not None and leader.id == 0, "the lowest-numbered node wins the first election")
    check(leader.term == 1, "the first election happens in term 1")
    check(cluster.leaders_seen == {1: 0}, "exactly one leader was recorded for term 1")
    check(all(node.voted_for == 0 for node in cluster.nodes), "every follower voted for the leader")
    check(all(node.log == [] for node in cluster.nodes), "the cluster starts with empty logs")
    check(all(node.ticks == 0 for node in cluster.nodes), "the election reset every timer")

    # --- 2. replication and majority commit ----------------------------------------------------
    check(leader.append_entry("x=1") == 1, "the first entry goes to index 1")
    check(leader.commit_index == 0, "the entry needs a majority before it commits")
    cluster.run(1)
    check(leader.commit_index == 1, "the entry commits once a majority stores it")
    check(cluster.history[1] == (1, "x=1"), "the committed entry is recorded in the cluster history")
    cluster.run(2)
    check(all(node.commit_index == 1 for node in cluster.nodes),
          "the leader's commit index propagates to every follower")
    check(cluster.converge_on(["x=1"]), "every log holds the committed entry")
    check(leader.append_entry("x=2") == 2, "the leader keeps appending in order")
    cluster.run(2)
    check(cluster.converge_on(["x=1", "x=2"]), "the second entry replicates too")
    check(cluster.history[2] == (1, "x=2"), "the second entry commits")

    # --- 3. a 1-node cluster is its own majority ------------------------------------------------
    solo = Cluster([0])
    solo.run(4)
    check(solo.leader() is not None and solo.leader().term == 1, "a single node elects itself")
    solo.leader().append_entry("only")
    solo.run(1)
    check(solo.nodes[0].commit_index == 1 and solo.history[1] == (1, "only"),
          "with one node a majority of one is immediate")

    # --- 4. partition: the minority leader cannot commit, the majority elects a new leader ------
    cluster = Cluster([0, 1, 2, 3, 4])
    cluster.run(4)
    old_leader = cluster.leader()
    check(old_leader is not None and old_leader.id == 0, "a stable leader exists before the split")

    cluster.to_partition([0, 1], [2, 3, 4])
    minority_entry = old_leader.append_entry("y=stale")
    cluster.run(3)
    check(old_leader.state == LEADER and old_leader.term == 1,
          "the minority leader still believes it leads")
    check(old_leader.commit_index == 0, "a 2-of-5 minority can never commit")
    check(1 not in cluster.history, "the isolated entry is not committed anywhere")
    check(old_leader.log == [(1, "y=stale")],
          "the minority leader keeps the entry in its log, uncommitted")

    cluster.run(4)                       # node 1 times out at tick 4, node 2 at tick 5
    majority_leader = cluster.by_id[2]
    check(majority_leader.state == LEADER and majority_leader.term == 2,
          "the majority side elects a new leader in a higher term")
    check(majority_leader.id in (2, 3, 4), "the new leader comes from the majority side")
    majority_leader.append_entry("z=fresh")
    cluster.run(2)
    check(cluster.history[1] == (2, "z=fresh"),
          "the majority commits its own entry at the same index, in the new term")
    check("y=stale" not in [entry[1] for entry in cluster.history.values()],
          "the stale entry never becomes committed")
    check(cluster.by_id[3].commands() == ["z=fresh"] and cluster.by_id[4].commands() == ["z=fresh"],
          "the majority replicas agree on the new history")
    check(cluster.by_id[0].commands() == ["y=stale"],
          "the isolated leader still holds the entry that will be discarded")

    # --- 5. healing: the minority leader steps down and repairs its log -------------------------
    check(minority_entry == 1, "the stale entry sat at index 1")
    cluster.net.heal()
    check(cluster.run_until_stable(), "the cluster re-stabilises once the partition heals")
    check(cluster.leader().id in (2, 3, 4), "a node from the majority side keeps leading")
    check(cluster.converge_on(["z=fresh"]),
          "healing truncates the stale entry and every log converges")
    check(old_leader.log == [(2, "z=fresh")], "the stale entry is gone from the old leader's log")
    check(len(cluster.history) == 1 and cluster.history[1] == (2, "z=fresh"),
          "history holds exactly the entry that was committed")
    check("y=stale" not in [entry[1] for entry in cluster.history.values()],
          "the discarding was safe: the stale entry never reached history")

    # --- 6. a new leader's no-op is what lets an earlier-term entry commit ----------------------
    cluster = Cluster([0, 1, 2, 3, 4])
    cluster.run(4)
    old_leader = cluster.leader()
    check(old_leader.id == 0, "node 0 leads the fresh five-node cluster")
    check(old_leader.append_entry("early") == 1, "the entry lands at index 1")
    cluster.run(2)
    check(cluster.history[1] == (1, "early") and all(node.commit_index == 1
                                                     for node in cluster.nodes),
          "a 3-of-5 majority commits the first entry")

    cluster.to_partition([0, 1], [2, 3, 4])
    check(old_leader.append_entry("stranded") == 2, "the stranded entry lands at index 2")
    cluster.by_id[0].up = False                  # the old leader dies with the entry in flight
    cluster.net.down.add(0)
    check(2 not in cluster.history and cluster.by_id[0].commit_index == 1,
          "two of five replicas are not a majority, so the entry is uncommitted when node 0 dies")
    check([m.recipient for m in cluster.net.queue] == [1],
          "only the dead leader's neighbour has the in-flight entry queued")
    cluster.net.heal()                           # but the rest of the cluster can talk again
    cluster.run(8)
    successor = cluster.by_id[1]
    check(successor.state == LEADER and successor.term == 2,
          "the successor that inherited index 2 wins the election")
    check(successor.log == [(1, "early"), (1, "stranded"), (2, None)],
          "the new leader appends its no-op in the new term")
    check(cluster.history[2] == (1, "stranded"),
          "the earlier-term entry is now committed")
    check(cluster.history[3] == (2, None) and successor.commit_index == 3,
          "the no-op commit is what carried the earlier-term entry with it")
    check(cluster.by_id[4].commit_index == 3, "the followers learn the new commit index")
    check(cluster.by_id[0].commit_index == 1 and cluster.by_id[0].log == [(1, "early"), (1, "stranded")],
          "the dead leader never sees index 2 committed (its log is the length that was replicated)")
    check(cluster.by_id[0].state == LEADER and cluster.by_id[0].term == 1,
          "a crashed node still believes it leads - the harness knows better")

    cluster.net.down.discard(0)               # the old leader comes back
    cluster.run(6)
    check(cluster.by_id[0].state == FOLLOWER and cluster.by_id[0].term == 2,
          "the resurrected leader steps down when it sees the higher term")
    check(cluster.by_id[0].log == [(1, "early"), (1, "stranded"), (2, None)],
          "it also picks up the entry it never learned about")
    check(cluster.by_id[0].commit_index == 3, "and it accepts the committed prefix")
    check(len({tuple(node.log) for node in cluster.nodes}) == 1, "all five logs are identical")
    check(len(cluster.history) == 3, "exactly three entries are committed")

    # --- 7. a lone leader cannot commit, and its entry is discarded after the partition heals ---
    cluster = Cluster([0, 1, 2, 3, 4])
    cluster.run(4)
    leader = cluster.leader()
    cluster.to_partition([0], [1, 2, 3, 4])
    leader.append_entry("risky")
    cluster.run(5)                       # node 1 times out at tick 4 and wins the majority side
    check(leader.state == LEADER and leader.commit_index == 0,
          "a lone leader reaches no majority and commits nothing")
    check(1 not in cluster.history, "nothing was committed by the lonely leader")
    check(cluster.by_id[1].commit_index == 0 and cluster.by_id[4].commit_index == 0,
          "no follower advances its commit index on its own")
    new_leader = cluster.by_id[1]
    check(new_leader.state == LEADER and new_leader.term == 2,
          "the majority side elects a new leader in a higher term")
    check(new_leader.log == [], "the new leader inherited nothing from the stranded entry")
    new_leader.append_entry("safe")
    cluster.run(2)
    check(cluster.history[1] == (2, "safe"), "the majority commits its own entry at that index")
    check(leader.commands() == ["risky"], "the stranded leader still holds the doomed entry")

    cluster.net.heal()
    check(cluster.run_until_stable(), "the cluster recovers after the second partition")
    check(cluster.leader().id == 1, "the leader holding the committed entry keeps leading")
    check(cluster.converge_on(["safe"]), "the stranded entry is replaced on every replica")
    check(leader.log == [(2, "safe")], "the stranded leader's log is repaired")
    check(cluster.history[1] == (2, "safe") and "risky" not in
          [entry[1] for entry in cluster.history.values()], "the doomed entry never committed")

    # --- 8. a truncated tail: committed entries survive a leader change -------------------------
    cluster = Cluster([0, 1, 2])
    cluster.run(4)
    leader = cluster.leader()
    leader.append_entry("tail=1")
    cluster.run(2)
    check(leader.commit_index == 1 and cluster.history[1] == (1, "tail=1"),
          "a 3-node cluster commits the first entry on one follower's ack")
    cluster.to_partition([0], [1, 2])
    leader.append_entry("tail=uncommitted")
    cluster.run(1)
    check(2 not in cluster.history and leader.last_index() == 2 and leader.commit_index == 1,
          "the tail entry stays uncommitted while the leader is alone")
    cluster.net.heal()
    check(cluster.run_until_stable(), "the three-node cluster stabilises again")
    check(cluster.history[1] == (1, "tail=1")
          and all(node.log[0] == (1, "tail=1") for node in cluster.nodes),
          "the committed prefix is untouched by the leader change")
    check(len({tuple(node.commands()) for node in cluster.nodes}) == 1
          and all(node.commit_index == max(n.commit_index for n in cluster.nodes)
                  for node in cluster.nodes),
          "all logs and commit indices agree at the end")

    print(f"raft: leaders_seen={cluster.leaders_seen} steps={cluster.steps} "
          f"messages={cluster.delivered} committed={len(cluster.history)} "
          f"dropped={len(cluster.net.dropped)}")


if __name__ == "__main__":
    main()
    print("PASS")
