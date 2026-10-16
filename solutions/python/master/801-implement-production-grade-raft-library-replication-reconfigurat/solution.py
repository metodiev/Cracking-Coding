"""Task 801 - Implement production-grade Raft library (replication, reconfiguration, snapshots).  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A faithful in-process Raft.  Every server runs the full protocol state machine (follower,
candidate, leader) and all communication travels over an explicit deterministic in-memory
network with per-link one-way latency and FIFO ordering.  Every piece of the paper is present:

  * leader election - seeded election timeouts, RequestVote with the log-up-to-date test, the
    "recent leader" vote suppression rule, quorum = majority of the configuration in the log;
  * log replication - AppendEntries carrying prev-log-index/term, conflict truncation with a
    term-based backtrack hint, and commit only once an entry of the current term is stored on a
    quorum;
  * safety - terms are monotone, a server votes at most once per term, entries are applied in
    index order to a deterministic state machine, and the driver records every leadership claim
    and every apply so that violations would be *observed* rather than asserted away;
  * snapshots - the log is compacted up to the applied index, the term-run boundaries of the
    compacted prefix are kept so that term(i) stays answerable, and a lagging server is caught up
    with InstallSnapshot;
  * membership change - single-change joint consensus: the leader appends a joint configuration,
    quorums then need a majority of the old *and* the new set, and the final configuration is
    appended by the leader that applies the joint entry.

Verified invariants (checked against the recorded run history, never assumed):
  election safety      at most one leader per term in every run, including runs with leader
                       crashes and a 3 -> 5 -> 3 reconfiguration;
  state machine safety every server that applies index i applies the same (term, command), and
                       each server applies a contiguous prefix of indices in order;
  log matching         in every recorded snapshot of all logs, two logs holding the same
                       (index, term) agree on the whole preceding prefix;
  leader completeness  a committed command survives a leader crash and is present on every live
                       server afterwards; persisted hard state round-trips exactly;
  snapshot equivalence a compacted image equals a from-scratch replay of the committed prefix, and
                       a wiped server caught up only by InstallSnapshot ends identical to the
                       leader;
  liveness             a leader is elected and commands commit while a quorum is reachable; no
                       leader appears while a quorum is not reachable;
  quorum arithmetic    the quorum predicate accepts exactly the majority subsets of a plain and
                       of a joint configuration (brute force over all subsets);
  cost bound           messages per committed command stay within a small constant times n.

Simplified versus a production library (etcd/tikv-raft): no disk, no threads, no sockets - the
cluster is a single-threaded loop over a deterministic event queue; snapshot transfer is one
whole-image message instead of a chunked streaming install; servers added by a reconfiguration
start empty and catch up through ordinary replication to the union of the old and new member
sets, so the paper's "catch the new server up before the joint entry commits" bookkeeping is not
modelled; ReadIndex/lease reads, pre-vote and the client session layer are omitted.  Latencies and
election timeouts are fixed constants rather than wall-clock driven, which makes a run exactly
reproducible.
"""
import heapq

MASK64 = (1 << 64) - 1
TWO_POW_53 = 9007199254740992.0

FOLLOWER, CANDIDATE, LEADER = "follower", "candidate", "leader"

HEARTBEAT_INTERVAL = 4
ELECTION_TIMEOUT_BASE = 14
ELECTION_TIMEOUT_JITTER = 8
LATENCY_BASE = 1
LATENCY_JITTER = 2

# messages carrying a configuration entry are recognised by their command tag
CONFIG_TAGS = ("config", "config_final")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def has_quorum(granted, voters, joint=None):
    """True when `granted` holds a majority of the configuration (both halves in joint mode)."""
    granted = set(granted)
    if joint is None:
        voters = set(voters)
        return len(granted & voters) >= len(voters) // 2 + 1
    old, new = set(joint[0]), set(joint[1])
    return (len(granted & old) >= len(old) // 2 + 1
            and len(granted & new) >= len(new) // 2 + 1)


def replay(commands, upto=None):
    """Independent oracle: apply `commands` (index order) to a fresh key/value state machine."""
    machine = {}
    for index, (_, command) in enumerate(commands, start=1):
        if upto is not None and index > upto:
            break
        kind = command[0]
        if kind == "set":
            machine[command[1]] = command[2]
        elif kind == "incr":
            machine[command[1]] = machine.get(command[1], 0) + command[2]
        elif kind == "del":
            machine.pop(command[1], None)
    return machine


# --------------------------------------------------------------------------- network
class Network:
    """Deterministic message network: per-link one-way latency, FIFO, crash-aware."""

    def __init__(self, latency_base=LATENCY_BASE, latency_jitter=LATENCY_JITTER, clock=0x5EED):
        self.now = 0
        self.queue = []
        self.seq = 0
        self.clock = clock
        self.latency_base = latency_base
        self.latency_jitter = latency_jitter
        self.cluster = None
        self.sent = 0
        self.delivered = 0
        self.dropped = 0
        self.installs = 0

    def send(self, src, dst, payload):
        self.clock, raw = splitmix64(self.clock)
        latency = self.latency_base + raw % (self.latency_jitter + 1)
        self.seq += 1
        self.sent += 1
        heapq.heappush(self.queue, (self.now + latency, self.seq, src, dst, payload))

    def deliver_due(self):
        while self.queue and self.queue[0][0] <= self.now:
            _, _, src, dst, payload = heapq.heappop(self.queue)
            node = self.cluster.nodes.get(dst) if self.cluster is not None else None
            if node is None or not node.alive:
                self.dropped += 1
                continue
            self.delivered += 1
            node.receive(src, payload, self, self.cluster)

    def next_time(self):
        return self.queue[0][0] if self.queue else None


# ------------------------------------------------------------------------------ server
class RaftNode:
    def __init__(self, node_id, voters, seed, now=0):
        self.id = node_id
        self.base_voters = tuple(voters)
        self.base_joint = None
        self.config_voters = tuple(voters)
        self.joint = None
        self.snapshot_threshold = None
        self.state = FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log = []                 # (term, command) for indices > snapshot_index
        self.commit_index = 0
        self.last_applied = 0
        self.snapshot_index = 0
        self.snapshot_term = 0
        self.prefix_terms = []        # (index, term) term-run boundaries inside the prefix
        self.image = {}               # frozen state machine at snapshot_index
        self.machine = {}             # live state machine
        self.pending_final = None
        self.next_index = {}
        self.match_index = {}
        self.votes = set()
        self.seed = seed
        self.alive = True
        self.leader_id = None
        self.last_heartbeat = now
        self.election_deadline = now
        self._reset_deadline(now, 0)

    # ------------------------------------------------------------------ log helpers
    def last_log_index(self):
        return self.snapshot_index + len(self.log)

    def last_log_term(self):
        return self.log[-1][0] if self.log else self.snapshot_term

    def term_at(self, index):
        """Term of the entry at `index`, or None past the end of the log (0 for index 0)."""
        if index <= 0:
            return 0
        if index > self.last_log_index():
            return None
        if index <= self.snapshot_index:
            term = 0
            for boundary, value in self.prefix_terms:
                if boundary <= index:
                    term = value
                else:
                    break
            return term
        return self.log[index - self.snapshot_index - 1][0]

    def entry_at(self, index):
        if self.snapshot_index < index <= self.last_log_index():
            return self.log[index - self.snapshot_index - 1]
        return None

    def abstract_log(self):
        """{index: term} for everything this server knows, compacted prefix included."""
        terms = {}
        for position, (boundary, term) in enumerate(self.prefix_terms):
            end = (self.prefix_terms[position + 1][0] - 1
                   if position + 1 < len(self.prefix_terms) else self.snapshot_index)
            for index in range(boundary, end + 1):
                terms[index] = term
        for offset, (term, _) in enumerate(self.log):
            terms[self.snapshot_index + offset + 1] = term
        return terms

    # ------------------------------------------------------------------- config mgmt
    def _recompute_config(self):
        """The configuration in effect is the last one in the log, committed or not."""
        for _, command in reversed(self.log):
            if command[0] == "config":
                self.joint = (tuple(command[1][0]), tuple(command[1][1]))
                self.config_voters = tuple(command[1][1])
                return
            if command[0] == "config_final":
                self.joint = None
                self.config_voters = tuple(command[1])
                return
        self.config_voters = self.base_voters
        self.joint = self.base_joint

    def replication_targets(self):
        targets = set(self.config_voters) | set(self.next_index)
        if self.joint is not None:
            targets |= set(self.joint[0]) | set(self.joint[1])
        targets.discard(self.id)
        return sorted(targets)

    # ----------------------------------------------------------------------- timers
    def _reset_deadline(self, now, term):
        """Seeded jitter so that at most one candidate times out first in a given term."""
        seed = (self.seed * 0x9E3779B97F4A7C15 + term * 7919 + self.id * 104729) & MASK64
        _, raw = splitmix64(seed)
        timeout = ELECTION_TIMEOUT_BASE + int(((raw >> 11) / TWO_POW_53) * ELECTION_TIMEOUT_JITTER)
        self.election_deadline = now + timeout

    def next_event_time(self):
        if self.state == LEADER:
            return self.last_heartbeat + HEARTBEAT_INTERVAL
        return self.election_deadline

    def tick(self, now, cluster):
        if self.state == LEADER:
            if now - self.last_heartbeat >= HEARTBEAT_INTERVAL:
                self.last_heartbeat = now
                self._replicate(cluster)
        elif now >= self.election_deadline:
            self.start_election(now, cluster)

    # --------------------------------------------------------------------- election
    def start_election(self, now, cluster):
        self.state = CANDIDATE
        self.current_term += 1
        self.voted_for = self.id
        self.leader_id = None
        self.votes = {self.id}
        self._reset_deadline(now, self.current_term)
        if has_quorum(self.votes, self.config_voters, self.joint):
            self.become_leader(now, cluster)
            return
        request = ("request_vote", self.id, self.current_term, self.last_log_index(),
                   self.last_log_term())
        targets = set(self.config_voters) | set(self.replication_targets())
        for peer in sorted(targets):
            if peer != self.id:
                cluster.net.send(self.id, peer, request)

    def become_leader(self, now, cluster):
        self.state = LEADER
        self.leader_id = self.id
        self.last_heartbeat = now
        self.next_index = {peer: self.last_log_index() + 1 for peer in self.replication_targets()}
        self.match_index = {peer: 0 for peer in self.replication_targets()}
        cluster.note_leader(self.current_term, self.id)
        self._replicate(cluster)

    def step_down(self, term):
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
        self.state = FOLLOWER
        self.leader_id = None
        self.votes = set()
        self.next_index = {}
        self.match_index = {}

    # ------------------------------------------------------------------ replication
    def append_command(self, command, cluster=None):
        """Client entry point (leader only); returns the assigned index."""
        index = self.last_log_index() + 1
        self.log.append((self.current_term, command))
        if command[0] in CONFIG_TAGS:
            self._recompute_config()
        if cluster is not None and has_quorum({self.id}, self.config_voters, self.joint):
            self.commit_index = index
            self.apply_committed(cluster)
        return index

    def change_config(self, new_voters, cluster):
        """Single-server-style joint consensus: old -> (old+new) -> new."""
        old = tuple(self.config_voters) if self.joint is None else tuple(self.joint[1])
        joint = (old, tuple(new_voters))
        self.pending_final = tuple(new_voters)
        return self.append_command(("config", joint), cluster)

    def _replicate(self, cluster):
        for peer in self.replication_targets():
            next_index = self.next_index.get(peer, self.last_log_index() + 1)
            if next_index <= self.snapshot_index:
                self.next_index[peer] = self.snapshot_index
                cluster.net.installs += 1
                cluster.net.send(self.id, peer, ("install_snapshot", self.id, self.current_term,
                                                 self.snapshot_payload()))
                continue
            self.next_index[peer] = next_index
            prev_index = next_index - 1
            offset = prev_index - self.snapshot_index
            entries = list(self.log[offset:])
            cluster.net.send(self.id, peer, ("append_entries", self.id, self.current_term,
                                             prev_index, self.term_at(prev_index), entries,
                                             self.commit_index))

    def snapshot_payload(self):
        return {"index": self.snapshot_index, "term": self.snapshot_term,
                "image": dict(self.image), "prefix_terms": list(self.prefix_terms),
                "voters": self.config_voters, "joint": self.joint}

    def _advance_commit(self, cluster):
        """Commit the highest current-term entry stored on a quorum."""
        for index in range(self.last_log_index(), self.commit_index, -1):
            if self.term_at(index) != self.current_term:
                continue
            holders = {self.id} | {peer for peer, match in self.match_index.items()
                                   if match >= index}
            if has_quorum(holders, self.config_voters, self.joint):
                self.commit_index = index
                self.apply_committed(cluster)
                return

    # ----------------------------------------------------------------- message layer
    def receive(self, src, message, net, cluster):
        kind = message[0]
        if kind == "request_vote":
            self.handle_request_vote(src, message, net, cluster)
        elif kind == "request_vote_response":
            self.handle_request_vote_response(src, message, net, cluster)
        elif kind == "append_entries":
            self.handle_append_entries(src, message, net, cluster)
        elif kind == "append_entries_response":
            self.handle_append_entries_response(src, message, net, cluster)
        elif kind == "install_snapshot":
            self.handle_install_snapshot(src, message, net, cluster)
        elif kind == "install_snapshot_response":
            self.handle_install_snapshot_response(src, message, net, cluster)

    def handle_request_vote(self, src, message, net, cluster):
        _, candidate, term, last_index, last_term = message
        if self.leader_id is not None and net.now - self.last_heartbeat < ELECTION_TIMEOUT_BASE:
            return          # a recent leader exists: ignore the request without moving the term
        if term < self.current_term:
            net.send(self.id, candidate, ("request_vote_response", self.id, self.current_term,
                                          False))
            return
        if term > self.current_term:
            self.step_down(term)
        up_to_date = (last_term > self.last_log_term()
                      or (last_term == self.last_log_term() and last_index >= self.last_log_index()))
        granted = self.voted_for in (None, candidate) and up_to_date
        if granted:
            self.voted_for = candidate
            self._reset_deadline(net.now, term)
        net.send(self.id, candidate, ("request_vote_response", self.id, self.current_term,
                                      granted))

    def handle_request_vote_response(self, src, message, net, cluster):
        _, voter, term, granted = message
        if term > self.current_term:
            self.step_down(term)
            return
        if self.state != CANDIDATE or term < self.current_term or not granted:
            return
        self.votes.add(voter)
        if has_quorum(self.votes, self.config_voters, self.joint):
            self.become_leader(net.now, cluster)

    def handle_append_entries(self, src, message, net, cluster):
        _, leader, term, prev_index, prev_term, entries, leader_commit = message
        if term < self.current_term:
            net.send(self.id, leader, ("append_entries_response", self.id, self.current_term,
                                       False, self.last_log_index() + 1, self.last_log_index()))
            return
        if term > self.current_term:
            self.step_down(term)
        if self.state != FOLLOWER:
            self.state = FOLLOWER
            self.next_index = {}
            self.match_index = {}
        self.leader_id = leader
        self.last_heartbeat = net.now
        self._reset_deadline(net.now, term)

        if prev_index > self.last_log_index():
            net.send(self.id, leader, ("append_entries_response", self.id, self.current_term,
                                       False, self.last_log_index() + 1, self.last_log_index()))
            return
        if self.term_at(prev_index) != prev_term:
            conflicting = self.term_at(prev_index) or 0
            first = prev_index
            while first > 1 and self.term_at(first - 1) == conflicting:
                first -= 1
            net.send(self.id, leader, ("append_entries_response", self.id, self.current_term,
                                       False, first, self.last_log_index()))
            return

        appended = 0
        for offset, entry in enumerate(entries):
            index = prev_index + 1 + offset
            if index <= self.snapshot_index:
                continue
            if index <= self.last_log_index():
                if self.term_at(index) != entry[0]:
                    self.log = self.log[:index - self.snapshot_index - 1]
                    self._recompute_config()
                else:
                    appended += 1
                    continue
            if index == self.last_log_index() + 1:
                self.log.append(entry)
                appended += 1
                if entry[1][0] in CONFIG_TAGS:
                    self._recompute_config()

        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, self.last_log_index())
        self.apply_committed(cluster)
        net.send(self.id, leader, ("append_entries_response", self.id, self.current_term, True,
                                   prev_index + appended, self.last_log_index()))

    def handle_append_entries_response(self, src, message, net, cluster):
        _, follower, term, success, match, follower_last = message
        if term > self.current_term:
            self.step_down(term)
            return
        if self.state != LEADER or term < self.current_term:
            return
        if success:
            self.match_index[follower] = max(self.match_index.get(follower, 0), match)
            self.next_index[follower] = self.match_index[follower] + 1
            self._advance_commit(cluster)
            return
        hint = max(1, min(self.next_index.get(follower, 1) - 1, match))
        self.next_index[follower] = hint
        self.match_index[follower] = min(self.match_index.get(follower, 0), hint - 1)

    def handle_install_snapshot(self, src, message, net, cluster):
        _, leader, term, payload = message
        if term < self.current_term:
            net.send(self.id, leader, ("install_snapshot_response", self.id, self.current_term,
                                       False, self.snapshot_index))
            return
        if term > self.current_term:
            self.step_down(term)
        self.state = FOLLOWER
        self.leader_id = leader
        self.last_heartbeat = net.now
        self._reset_deadline(net.now, term)
        if payload["index"] > self.snapshot_index:
            self.image = dict(payload["image"])
            self.machine = dict(payload["image"])
            self.snapshot_index = payload["index"]
            self.snapshot_term = payload["term"]
            self.prefix_terms = list(payload["prefix_terms"])
            self.log = []
            self.commit_index = self.snapshot_index
            self.last_applied = self.snapshot_index
            self.base_voters = tuple(payload["voters"])
            self.base_joint = payload["joint"]
            self._recompute_config()
        net.send(self.id, leader, ("install_snapshot_response", self.id, self.current_term, True,
                                   self.snapshot_index))

    def handle_install_snapshot_response(self, src, message, net, cluster):
        _, follower, term, success, match = message
        if term > self.current_term:
            self.step_down(term)
            return
        if self.state != LEADER or not success:
            return
        self.match_index[follower] = max(self.match_index.get(follower, 0), match)
        self.next_index[follower] = self.match_index[follower] + 1
        self._advance_commit(cluster)

    # -------------------------------------------------------------- state machine
    def _apply_command(self, command, cluster):
        kind = command[0]
        if kind == "set":
            self.machine[command[1]] = command[2]
        elif kind == "incr":
            self.machine[command[1]] = self.machine.get(command[1], 0) + command[2]
        elif kind == "del":
            self.machine.pop(command[1], None)
        elif kind == "config":
            self._recompute_config()
            if self.state == LEADER and cluster is not None:
                self.append_command(("config_final", tuple(command[1][1])), cluster)
        elif kind == "config_final":
            self._recompute_config()
            if self.state == LEADER and self.id not in self.config_voters:
                self.step_down(self.current_term)

    def apply_committed(self, cluster):
        while self.last_applied < self.commit_index:
            index = self.last_applied + 1
            entry = self.entry_at(index)
            if entry is None:
                return
            term, command = entry
            self._apply_command(command, cluster)
            self.last_applied = index
            if cluster is not None:
                cluster.note_applied(index, term, command, self.id)
        if (self.snapshot_threshold is not None
                and self.last_applied - self.snapshot_index >= self.snapshot_threshold):
            self.compact()

    def compact(self):
        """Drop the applied prefix from the log, keeping term-run boundaries."""
        target = self.last_applied
        if target <= self.snapshot_index:
            return False
        term = self.term_at(target)
        dropped = self.log[:target - self.snapshot_index]
        index = self.snapshot_index + 1
        for entry_term, _ in dropped:
            if not self.prefix_terms or self.prefix_terms[-1][1] != entry_term:
                self.prefix_terms.append((index, entry_term))
            index += 1
        self.log = self.log[target - self.snapshot_index:]
        self.snapshot_index = target
        self.snapshot_term = term
        self.image = dict(self.machine)
        self.base_voters = self.config_voters
        self.base_joint = self.joint
        return True

    # ---------------------------------------------------------------- persistence
    def persist(self):
        return {"id": self.id, "term": self.current_term, "voted_for": self.voted_for,
                "log": list(self.log), "commit_index": self.commit_index,
                "last_applied": self.last_applied, "snapshot_index": self.snapshot_index,
                "snapshot_term": self.snapshot_term, "prefix_terms": list(self.prefix_terms),
                "image": dict(self.image), "machine": dict(self.machine),
                "voters": self.config_voters, "joint": self.joint}

    @classmethod
    def from_persisted(cls, state, voters, seed, now=0):
        node = cls(state["id"], voters, seed, now)
        node.current_term = state["term"]
        node.voted_for = state["voted_for"]
        node.log = [tuple(entry) for entry in state["log"]]
        node.commit_index = state["commit_index"]
        node.last_applied = state["last_applied"]
        node.snapshot_index = state["snapshot_index"]
        node.snapshot_term = state["snapshot_term"]
        node.prefix_terms = [tuple(pair) for pair in state["prefix_terms"]]
        node.image = dict(state["image"])
        node.machine = dict(state["machine"])
        node.base_voters = tuple(state["voters"])
        node.base_joint = state["joint"]
        node._recompute_config()
        return node


# ---------------------------------------------------------------------------- driver
class Cluster:
    """Single-threaded deterministic driver: tick, deliver, apply, advance the clock."""

    def __init__(self, ids, seed=1, snapshot_threshold=None, latency_jitter=LATENCY_JITTER):
        self.net = Network(latency_jitter=latency_jitter)
        self.nodes = {}
        self.snapshot_threshold = snapshot_threshold
        for node_id in ids:
            node = RaftNode(node_id, ids, seed + 101 * node_id)
            node.snapshot_threshold = snapshot_threshold
            self.nodes[node_id] = node
        self.net.cluster = self
        self.seed = seed
        self.leaders = {}          # term -> node id (a clash is a safety violation)
        self.applied = {}          # index -> (term, command)
        self.violations = []
        self.history = []          # optional snapshots of every log, for the log-matching test
        self.rounds = 0

    # ------------------------------------------------------------------- reporting
    def note_leader(self, term, node_id):
        known = self.leaders.get(term)
        if known is not None and known != node_id:
            self.violations.append(("election safety", term, known, node_id))
        else:
            self.leaders[term] = node_id

    def note_applied(self, index, term, command, node_id):
        known = self.applied.get(index)
        if known is not None and known != (term, command):
            self.violations.append(("state machine safety", index, known, (term, command), node_id))
        else:
            self.applied[index] = (term, command)

    def leader(self):
        leaders = [node for node in self.nodes.values() if node.alive and node.state == LEADER]
        return leaders[0] if len(leaders) == 1 else None

    def leader_count(self):
        return len([node for node in self.nodes.values() if node.alive and node.state == LEADER])

    def live(self):
        return [node for node in self.nodes.values() if node.alive]

    # ------------------------------------------------------------------ operations
    def crash(self, node_id):
        self.nodes[node_id].alive = False

    def revive(self, node_id):
        self.nodes[node_id].alive = True
        self.nodes[node_id].state = FOLLOWER
        self.nodes[node_id].leader_id = None
        self.nodes[node_id].next_index = {}
        self.nodes[node_id].match_index = {}

    def wipe(self, node_id):
        """Restart with total data loss: empty log, empty state machine, empty hard state."""
        node = self.nodes[node_id]
        node.alive = True
        node.state = FOLLOWER
        node.current_term = 0
        node.voted_for = None
        node.log = []
        node.commit_index = 0
        node.last_applied = 0
        node.snapshot_index = 0
        node.snapshot_term = 0
        node.prefix_terms = []
        node.image = {}
        node.machine = {}
        node.next_index = {}
        node.match_index = {}
        node.base_voters = tuple(self.nodes)
        node.base_joint = None
        node._recompute_config()

    def add_node(self, node_id, voters):
        node = RaftNode(node_id, voters, self.seed + 101 * node_id, self.net.now)
        node.snapshot_threshold = self.snapshot_threshold
        self.nodes[node_id] = node
        return node

    def submit(self, command):
        leaders = [node for node in self.live() if node.state == LEADER]
        if len(leaders) != 1:
            raise AssertionError("expected exactly one leader, found %d" % len(leaders))
        return leaders[0].append_command(command, self)

    def step(self):
        self.rounds += 1
        now = self.net.now
        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            if node.alive:
                node.tick(now, self)
        self.net.deliver_due()
        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            if node.alive:
                node.apply_committed(self)
        candidates = []
        pending = self.net.next_time()
        if pending is not None:
            candidates.append(pending)
        for node in self.live():
            candidates.append(node.next_event_time())
        self.net.now = min(candidates) if candidates else now + 1
        if self.net.now <= now:
            self.net.now = now + 1

    def run(self, rounds, record_every=0):
        for _ in range(rounds):
            self.step()
            if record_every and self.rounds % record_every == 0:
                self.record()

    def record(self):
        self.history.append({node_id: node.abstract_log() for node_id, node in self.nodes.items()
                             if node.alive})

    def wait_for_leader(self, rounds):
        for _ in range(rounds):
            if self.leader_count() == 1:
                return True
            self.step()
        return self.leader_count() == 1


# ------------------------------------------------------------------------ verifiers
def log_matching_holds(history):
    """Two logs holding the same (index, term) must agree on the whole preceding prefix."""
    for snapshot in history:
        ids = sorted(snapshot)
        for left_id in range(len(ids)):
            for right_id in range(left_id + 1, len(ids)):
                left, right = snapshot[ids[left_id]], snapshot[ids[right_id]]
                agreeing = [index for index in set(left) & set(right) if left[index] == right[index]]
                if not agreeing:
                    continue
                boundary = max(agreeing)
                for index in range(1, boundary + 1):
                    if index in left and index in right and left[index] != right[index]:
                        return False, (ids[left_id], ids[right_id], index, boundary)
    return True, None


def quorum_predicate_matches_bruteforce(voters, joint=None):
    """The predicate must accept exactly the subsets of size >= majority, for both halves."""
    universe = sorted(set(voters) | (set(joint[0]) | set(joint[1]) if joint else set()))
    for mask in range(1 << len(universe)):
        subset = {universe[i] for i in range(len(universe)) if mask >> i & 1}
        expected = len(subset & set(voters)) >= len(set(voters)) // 2 + 1
        if joint is not None:
            old, new = set(joint[0]), set(joint[1])
            expected = (len(subset & old) >= len(old) // 2 + 1
                        and len(subset & new) >= len(new) // 2 + 1)
        if has_quorum(subset, voters, joint) != expected:
            return False, subset
    return True, None


def main():
    # --- 1. quorum arithmetic (the basis of election safety) ------------------------
    for voters, joint in (((1, 2, 3), None), ((1, 2, 3, 4, 5), None),
                          ((1, 2, 3), ((1, 2, 3), (1, 2, 3, 4, 5))),
                          ((1, 2, 3, 4, 5), ((1, 2, 3, 4, 5), (2, 3, 4)))):
        ok, witness = quorum_predicate_matches_bruteforce(voters, joint)
        check(ok, "quorum predicate disagrees with brute force on %s: %s" % (joint, witness))
    check(has_quorum({1, 2}, (1, 2, 3)) and not has_quorum({1}, (1, 2, 3)),
          "a strict majority of three must be exactly two servers")
    check(has_quorum({1, 2, 3}, (1, 2, 3), ((1, 2, 3), (1, 2, 3, 4, 5)))
          and not has_quorum({1, 2}, (1, 2, 3), ((1, 2, 3), (1, 2, 3, 4, 5))),
          "a joint quorum must satisfy both halves of the configuration")

    # --- 2. a three-server cluster elects one leader and replicates -----------------
    cluster = Cluster([1, 2, 3], seed=7)
    check(cluster.wait_for_leader(200), "a three-server cluster must elect one leader")
    leader = cluster.leader()
    first_term = leader.current_term
    index = cluster.submit(("set", "x", 1))
    cluster.run(60)
    check(cluster.violations == [], "no safety violation expected: %s" % (cluster.violations,))
    check(all(cluster.nodes[i].machine.get("x") == 1 for i in (1, 2, 3)),
          "a committed command must reach every server: %s"
          % [cluster.nodes[i].machine for i in (1, 2, 3)])
    check(all(cluster.nodes[i].last_applied >= index for i in (1, 2, 3)),
          "every server must apply the committed index")
    check(all(cluster.nodes[i].commit_index >= index for i in (1, 2, 3)),
          "every server must learn the commit index")
    check(cluster.leader_count() == 1, "exactly one leader may exist at a time")
    check(cluster.nodes[leader.id].current_term >= first_term,
          "terms never move backwards for a leader")

    # --- 3. an unreachable leader is replaced and progress continues ----------------
    stale = cluster.leader()
    cluster.crash(stale.id)
    check(cluster.wait_for_leader(400), "the surviving majority must elect a new leader")
    replacement = cluster.leader()
    check(replacement.id != stale.id, "the crashed leader must be replaced, not resurrected")
    check(replacement.current_term > stale.current_term,
          "the replacement must win a strictly later term")
    cluster.submit(("set", "y", 2))
    cluster.run(120)
    live = [node for node in cluster.live()]
    check(all(node.machine.get("x") == 1 and node.machine.get("y") == 2 for node in live),
          "a committed command must survive the leader crash")
    check(cluster.violations == [], "crash-failover must not violate safety: %s"
          % (cluster.violations,))
    check(all(node.last_applied >= node.snapshot_index for node in live),
          "applied index must never trail the compacted prefix")

    # --- 4. no quorum, no leader (liveness is conditional on a majority) -----------
    dead = Cluster([1, 2, 3], seed=11)
    check(dead.wait_for_leader(200), "the cluster must elect a leader before the failures")
    dead.crash(dead.leader().id)
    dead.crash([node.id for node in dead.live()][0])
    dead.run(600)
    check(dead.leader() is None, "a cluster without a majority must not have a leader")
    check(dead.violations == [], "failed elections must not break election safety")

    # --- 5. log matching over the recorded history ---------------------------------
    traced = Cluster([1, 2, 3, 4, 5], seed=23)
    check(traced.wait_for_leader(300), "a five-server cluster must elect a leader")
    traced.record()
    for step in range(40):
        traced.submit(("incr", "counter", 1))
        traced.run(6, record_every=3)
    traced.run(80, record_every=6)
    traced.crash(traced.leader().id)
    traced.run(120, record_every=6)
    traced.record()
    check(traced.violations == [], "no violation in the traced run: %s" % (traced.violations,))
    holds, witness = log_matching_holds(traced.history)
    check(holds, "the log matching property must hold in every recorded state: %s" % (witness,))
    check(len(traced.history) > 20, "the history must contain enough states to be meaningful")
    counter = sum(1 for index in traced.applied if traced.applied[index][1][0] == "incr")
    check(counter == 40, "exactly the 40 submitted increments may commit, got %d" % counter)
    final = [node.machine.get("counter", 0) for node in traced.live()]
    check(final == [40] * len(final),
          "every live server must hold the same counter value: %s" % (final,))
    check(all(node.last_applied == node.commit_index for node in traced.live()),
          "every live server must have applied its whole committed prefix")

    # --- 6. snapshot equivalence and whole-image catch-up --------------------------
    snap = Cluster([1, 2, 3], seed=31, snapshot_threshold=4)
    check(snap.wait_for_leader(300), "the snapshot cluster must elect a leader")
    snap.crash(3)
    for value in range(24):
        snap.submit(("set", "k%02d" % value, value))
        snap.run(8)
    snap_leader = snap.leader()
    check(snap_leader.snapshot_index >= 4,
          "the leader must have compacted its log, snapshot at %d" % snap_leader.snapshot_index)
    check(len(snap_leader.log) <= 4 * 3,
          "compaction must bound the log length, got %d" % len(snap_leader.log))
    ordered = [snap.applied[index] for index in sorted(snap.applied)]
    image = replay(ordered, snap_leader.snapshot_index)
    check(image == snap_leader.image,
          "the state image must equal a replay of the committed prefix")
    check(snap_leader.term_at(snap_leader.snapshot_index) == snap_leader.snapshot_term,
          "term_at must stay answerable inside the compacted prefix")

    snap.wipe(3)
    snap.run(400)
    check(snap.net.installs >= 1,
          "catching up a wiped server must go through InstallSnapshot, sent %d"
          % snap.net.installs)
    check(snap.nodes[3].snapshot_index >= snap_leader.snapshot_index,
          "a wiped server must be caught up by InstallSnapshot, got %d"
          % snap.nodes[3].snapshot_index)
    check(snap.nodes[3].machine == snap.nodes[1].machine,
          "the snapshot image must restore the exact state machine")
    check(snap.nodes[3].last_applied == snap.nodes[3].commit_index,
          "a snapshot install must leave no unapplied committed prefix")
    check(snap.violations == [], "snapshot transfer must not break safety: %s" % (snap.violations,))

    # --- 7. reconfiguration 3 -> 5 -> 3 (joint consensus) --------------------------
    reconf = Cluster([1, 2, 3], seed=41, snapshot_threshold=6)
    check(reconf.wait_for_leader(300), "the cluster must elect a leader before reconfiguring")
    reconf.submit(("set", "before", 1))
    reconf.run(40)
    leader = reconf.leader()
    leader.change_config([1, 2, 3, 4, 5], reconf)
    reconf.run(40)
    for node_id in (4, 5):
        reconf.add_node(node_id, [1, 2, 3, 4, 5])
    reconf.run(400)
    check(reconf.violations == [], "joint consensus must keep safety: %s" % (reconf.violations,))
    check(tuple(reconf.leader().config_voters) == (1, 2, 3, 4, 5),
          "the new configuration must take effect, got %s" % (reconf.leader().config_voters,))
    check(all(node.machine.get("before") == 1 for node in reconf.live()),
          "the pre-reconfiguration state must survive on the new servers")
    reconf.submit(("set", "after", 2))
    reconf.run(80)
    check(all(node.machine.get("after") == 2 for node in reconf.live()),
          "the five-server configuration must commit new entries")
    leader = reconf.leader()
    leader.change_config([1, 3, 4], reconf)
    reconf.run(400)
    check(tuple(reconf.leader().config_voters) == (1, 3, 4),
          "shrinking the configuration must take effect, got %s" % (reconf.leader().config_voters,))
    check(all(node.machine.get("after") == 2 for node in reconf.live()
              if node.id in (1, 3, 4)),
          "the shrunk configuration must retain the state")
    check(reconf.violations == [], "shrinking must not break safety: %s" % (reconf.violations,))
    check(all(node.last_applied == node.commit_index for node in reconf.live()),
          "every live server must be fully caught up after reconfiguration")

    # --- 8. handler-level negatives ------------------------------------------------
    solo = Cluster([9], seed=51)
    solo.run(60)
    check(solo.nodes[9].state == LEADER and solo.net.sent == 0,
          "a single-server cluster must elect itself without exchanging any message")
    solo.submit(("set", "only", 1))
    check(solo.nodes[9].machine == {"only": 1},
          "a single-server cluster must commit locally")

    voter = RaftNode(2, (1, 2, 3), seed=61)
    voter.receive(1, ("request_vote", 1, 1, 5, 1), Network(), None)
    check(voter.voted_for == 1, "an up-to-date candidate must win the vote")
    voter.receive(3, ("request_vote", 3, 1, 5, 1), Network(), None)
    check(voter.voted_for == 1, "a server must not vote twice in the same term")
    receiver = RaftNode(4, (1, 2, 3), seed=62)
    receiver.log = [(1, ("set", "a", 1)), (1, ("set", "b", 2))]
    receiver._recompute_config()
    net = Network()
    receiver.receive(3, ("request_vote", 3, 3, 1, 1), net, None)
    check(receiver.voted_for is None, "a candidate with a shorter log must be refused")
    receiver.receive(3, ("request_vote", 3, 3, 2, 1), net, None)
    check(receiver.voted_for == 3, "an equally long, equally recent log must be accepted")
    receiver.receive(1, ("append_entries", 1, 1, 0, 0, [], 0), net, None)
    check(receiver.current_term == 3, "a stale-term AppendEntries must not move the term back")
    suppressed = RaftNode(5, (1, 2, 3), seed=63)
    suppressed.leader_id = 1
    suppressed.last_heartbeat = 0
    suppressed.receive(3, ("request_vote", 3, 9, 99, 99), net, None)
    check(suppressed.current_term == 0 and suppressed.voted_for is None,
          "a vote request arriving while a leader is live must be ignored entirely (no term bump)")

    # --- 9. persistence round-trip -------------------------------------------------
    original = snap.nodes[1]
    hard = original.persist()
    restarted = RaftNode.from_persisted(hard, (1, 2, 3), seed=original.seed)
    check(restarted.current_term == original.current_term
          and restarted.log == original.log
          and restarted.image == original.image
          and restarted.machine == original.machine
          and restarted.prefix_terms == original.prefix_terms,
          "persisted hard state must round-trip exactly")
    check(restarted.abstract_log() == original.abstract_log(),
          "a restarted server must answer term(i) identically")
    restarted.voted_for = 2
    restarted.current_term = original.current_term + 1
    restarted.receive(3, ("request_vote", 3, restarted.current_term, 99, 99), Network(), None)
    check(restarted.voted_for == 2,
          "a restarted server must honour the vote it already persisted")

    # --- 10. determinism and cost bound -------------------------------------------
    first = Cluster([1, 2, 3, 4, 5], seed=71)
    first.wait_for_leader(300)
    for value in range(12):
        first.submit(("set", "d%d" % value, value))
        first.run(10)
    second = Cluster([1, 2, 3, 4, 5], seed=71)
    second.wait_for_leader(300)
    for value in range(12):
        second.submit(("set", "d%d" % value, value))
        second.run(10)
    check(first.leaders == second.leaders,
          "the same seed must produce the same leadership history")
    check({node.id: node.machine for node in first.live()}
          == {node.id: node.machine for node in second.live()},
          "the same seed must produce the same state machines")
    check(first.net.sent == second.net.sent,
          "the same seed must produce the same message count")
    messages_per_command = first.net.sent / 12.0
    check(messages_per_command <= 8 * len(first.nodes),
          "replication cost must stay linear in n, got %.1f messages per command"
          % messages_per_command)
    check(all(node.last_applied == node.commit_index for node in first.live()),
          "the deterministic cluster must be fully caught up")

    print("runs: 3-server failover term %d -> %d, 40 increments on 5 servers, "
          "snapshot at index %d, 3->5->3 reconfiguration"
          % (stale.current_term, replacement.current_term, snap_leader.snapshot_index))
    print("recorded log states: %d, committed entries: %d, safety violations: %d"
          % (len(traced.history), len(traced.applied), len(reconf.violations)))
    print("messages for 12 commands on 5 servers: %d (%.1f per command, %d dropped)"
          % (first.net.sent, messages_per_command, first.net.dropped))
    print("PASS")


if __name__ == "__main__":
    main()
