"""Task 472 - Implement basic Raft consensus pseudo-simulation (leader election outline). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

Deterministic Raft: scripted message delivery (no wall-clock timing), explicit partition/crash
actions, and a 5-node cluster. The simulation implements RequestVote / AppendEntries with the
usual Raft rules (vote once per term, log up-to-date check, conflict truncation, commit only
after a majority matchIndex and only for current-term entries). Assertions: at most one leader
per term, log-matching safety across partitions, leader completeness (a new leader holds every
committed entry), committed entries never change, and elections after the leader crashes.
"""

N = 5
MAJORITY = 3


class Node:
    def __init__(self, i):
        self.id = i
        self.term = 0
        self.voted_for = -1
        self.state = "F"          # F follower, C candidate, L leader
        self.log = []             # list of (term, cmd)
        self.commit = 0
        self.voters = set()
        self.next_index = [0] * N
        self.match_index = [-1] * N


class RaftSim:
    def __init__(self):
        self.nodes = [Node(i) for i in range(N)]
        self.alive = [True] * N
        self.partition = None                 # None or list of node-id groups
        self.queue = []                       # (from, to, message tuples)
        self.leaders_by_term = {}
        self.committed = {}                   # index -> (term, cmd)
        self.committed_up_to = 0
        self.last_term = [0] * N
        self.last_commit = [0] * N

    # ---------- network ----------
    def _connected(self, a, b):
        if self.partition is None:
            return True
        return any(a in g and b in g for g in self.partition)

    def send(self, frm, to, msg):
        if not self.alive[frm] or not self.alive[to]:
            return
        if not self._connected(frm, to):
            return
        self.queue.append((frm, to, msg))

    def drain(self):
        while self.queue:
            frm, to, msg = self.queue.pop(0)
            if not self.alive[frm] or not self.alive[to] or not self._connected(frm, to):
                continue
            self._deliver(frm, to, msg)

    # ---------- actions ----------
    def crash(self, i):
        self.alive[i] = False

    def recover(self, i):
        self.alive[i] = True
        self.nodes[i].state = "F"

    def partition_off(self, groups):
        self.partition = [set(g) for g in groups]

    def heal(self):
        self.partition = None

    def timeout(self, i):
        if not self.alive[i]:
            return
        n = self.nodes[i]
        n.term += 1
        n.voted_for = i
        n.state = "C"
        n.voters = {i}
        last_idx = len(n.log) - 1
        last_term = n.log[-1][0] if n.log else 0
        for j in range(N):
            if j != i:
                self.send(i, j, ("RV", n.term, i, last_idx, last_term))

    def client(self, cmd):
        leaders = [n for n in self.nodes if n.state == "L" and self.alive[n.id]]
        if not leaders:
            return False
        n = max(leaders, key=lambda x: x.term)
        n.log.append((n.term, cmd))
        return True

    def replicate(self):
        for n in self.nodes:
            if n.state != "L" or not self.alive[n.id]:
                continue
            for f in range(N):
                if f == n.id:
                    continue
                nn = min(n.next_index[f], len(n.log))
                prev_idx = nn - 1
                prev_term = n.log[prev_idx][0] if prev_idx >= 0 else 0
                self.send(n.id, f, ("AE", n.term, n.id, prev_idx, prev_term,
                                    list(n.log[nn:]), n.commit))

    # ---------- message handling ----------
    def _deliver(self, frm, to, msg):
        n = self.nodes[to]
        kind = msg[0]
        if kind == "RV":
            _, mterm, cand, last_idx, last_term = msg
            if mterm < n.term:
                self.send(to, frm, ("RVResp", n.term, to, False))
                return
            if mterm > n.term:
                n.term = mterm
                n.voted_for = -1
                if n.state != "F":
                    n.state = "F"
            own_idx = len(n.log) - 1
            own_term = n.log[-1][0] if n.log else 0
            up_to_date = last_term > own_term or (last_term == own_term and last_idx >= own_idx)
            granted = n.voted_for in (-1, cand) and up_to_date
            if granted:
                n.voted_for = cand
            self.send(to, frm, ("RVResp", n.term, to, granted))
        elif kind == "RVResp":
            _, mterm, voter, granted = msg
            if mterm > n.term:
                n.term = mterm
                n.voted_for = -1
                n.state = "F"
                return
            if n.state == "C" and mterm == n.term and granted and voter not in n.voters:
                n.voters.add(voter)
                if len(n.voters) >= MAJORITY:
                    self._become_leader(n)
        elif kind == "AE":
            _, mterm, _leader, prev_idx, prev_term, entries, mcommit = msg
            if mterm < n.term:
                self.send(to, frm, ("AEResp", n.term, to, False, len(n.log)))
                return
            if mterm > n.term:
                n.term = mterm
                n.voted_for = -1
            n.state = "F"
            if prev_idx >= len(n.log) or (prev_idx >= 0 and n.log[prev_idx][0] != prev_term):
                self.send(to, frm, ("AEResp", n.term, to, False, len(n.log)))
                return
            idx = prev_idx + 1
            for entry in entries:
                if idx < len(n.log):
                    if n.log[idx][0] != entry[0]:
                        del n.log[idx:]
                        n.log.append(entry)
                else:
                    n.log.append(entry)
                idx += 1
            last_new = prev_idx + len(entries) + 1
            if mcommit > n.commit:
                new_commit = min(mcommit, last_new)
                for i in range(n.commit, new_commit):
                    self._record_commit(i, n.log[i])
                n.commit = max(n.commit, new_commit)
            self.send(to, frm, ("AEResp", n.term, to, True, last_new))
        elif kind == "AEResp":
            _, mterm, follower, success, hint = msg
            if mterm > n.term:
                n.term = mterm
                n.voted_for = -1
                n.state = "F"
                return
            if n.state != "L" or mterm != n.term:
                return
            if success:
                if hint > n.match_index[follower]:
                    n.match_index[follower] = hint
                    n.next_index[follower] = hint + 1
                new_commit = n.commit
                for n_ in range(len(n.log), n.commit, -1):
                    if n.log[n_ - 1][0] != n.term:
                        continue
                    votes = sum(1 for j in range(N)
                                if (j == n.id and len(n.log) >= n_) or n.match_index[j] >= n_)
                    if votes >= MAJORITY:
                        new_commit = n_
                        break
                if new_commit > n.commit:
                    for i in range(n.commit, new_commit):
                        self._record_commit(i, n.log[i])
                    n.commit = new_commit
            else:
                n.next_index[follower] = max(0, min(n.next_index[follower] - 1, hint))

    def _become_leader(self, n):
        n.state = "L"
        prev = self.leaders_by_term.get(n.term)
        assert prev is None or prev == n.id, ("two leaders in term", n.term, prev, n.id)
        self.leaders_by_term[n.term] = n.id
        n.next_index = [len(n.log)] * N
        n.match_index = [-1] * N
        for i in range(self.committed_up_to):
            assert i < len(n.log) and n.log[i] == self.committed[i], \
                ("elected leader lacks committed entry", n.id, i)

    def _record_commit(self, i, entry):
        prev = self.committed.get(i)
        assert prev is None or prev == entry, ("committed entry changed", i, prev, entry)
        self.committed[i] = entry
        while self.committed_up_to in self.committed:
            self.committed_up_to += 1

    # ---------- invariants ----------
    def check_invariants(self):
        for a in range(N):
            for b in range(a + 1, N):
                la, lb = self.nodes[a].log, self.nodes[b].log
                for i in range(min(len(la), len(lb))):
                    if la[i][0] == lb[i][0]:
                        assert la[:i + 1] == lb[:i + 1], ("log matching violated", a, b, i)
        for i, n in enumerate(self.nodes):
            assert n.commit <= len(n.log), ("commit beyond log", i)
            assert n.term >= self.last_term[i], ("term regression", i)
            assert n.commit >= self.last_commit[i], ("commit regression", i)
            self.last_term[i] = n.term
            self.last_commit[i] = n.commit

    def run(self, actions):
        for action in actions:
            kind = action[0]
            if kind == "timeout":
                self.timeout(action[1])
            elif kind == "client":
                self.client(action[1])
            elif kind == "replicate":
                self.replicate()
            elif kind == "drain":
                self.drain()
            elif kind == "partition":
                self.partition_off(action[1])
            elif kind == "heal":
                self.heal()
            elif kind == "crash":
                self.crash(action[1])
            elif kind == "recover":
                self.recover(action[1])
            self.check_invariants()


def lcg(seed):
    return (seed * 75 + 74) % 65537


def random_actions(count, seed):
    actions = []
    counter = 0
    for _ in range(count):
        seed = lcg(seed)
        a = seed % 8
        seed = lcg(seed)
        node = seed % N
        if a == 0:
            actions.append(("timeout", node))
        elif a == 1:
            counter += 1
            actions.append(("client", "c%d" % counter))
        elif a == 2:
            actions.append(("replicate",))
        elif a == 3:
            actions.append(("drain",))
        elif a == 4:
            seed = lcg(seed)
            mask = seed
            g0 = tuple(i for i in range(N) if (mask >> i) & 1)
            g1 = tuple(i for i in range(N) if not (mask >> i) & 1)
            actions.append(("partition", [g0, g1]))
        elif a == 5:
            actions.append(("heal",))
        elif a == 6:
            actions.append(("crash", node))
        else:
            actions.append(("recover", node))
        if a != 3:
            actions.append(("drain",))
    return actions


def main():
    sim = RaftSim()
    sim.run([("timeout", 0), ("drain",)])
    assert sim.leaders_by_term == {1: 0}
    sim.run([("client", "x=1"), ("replicate",), ("drain",), ("replicate",), ("drain",)])
    assert sim.committed_up_to == 1
    assert sim.committed[0] == (1, "x=1")

    # Leader failure: the remaining majority must elect a new leader in a higher term.
    sim.run([("crash", 0), ("timeout", 1), ("drain",)])
    assert sim.leaders_by_term[2] == 1
    sim.run([("client", "y=2"), ("replicate",), ("drain",), ("replicate",), ("drain",)])
    assert sim.committed_up_to == 2 and sim.committed[1] == (2, "y=2")

    # Partition: the leader [0,1,2] can replicate z=3 to node 2 but has no quorum.
    sim.run([("partition", [(0, 1, 2), (3, 4)]),
             ("client", "z=3"), ("replicate",), ("drain",), ("replicate",), ("drain",)])
    assert sim.committed_up_to == 2, "z=3 must not commit without a quorum"
    assert sim.nodes[1].log[-1] == (2, "z=3")
    assert sim.nodes[2].log[-1] == (2, "z=3")

    # The minority [3,4] can never elect a leader (needs 3 votes).
    sim.run([("timeout", 3), ("drain",)])
    assert 3 not in sim.leaders_by_term, "minority must not elect a leader"
    sim.run([("heal",), ("drain",)])

    # After healing, node 1 first campaigns in term 3 (blocked by stale votes) then wins term 4.
    sim.run([("timeout", 1), ("drain",)])
    assert 3 not in sim.leaders_by_term
    sim.run([("timeout", 1), ("drain",)])
    assert sim.leaders_by_term[4] == 1

    # The new leader commits a current-term entry, carrying z=3 along with it.
    sim.run([("client", "w=4"), ("replicate",), ("drain",),
             ("replicate",), ("drain",), ("replicate",), ("drain",)])
    assert sim.committed_up_to == 4
    assert sim.committed[2] == (2, "z=3")
    assert sim.committed[3] == (4, "w=4")
    expected = [(1, "x=1"), (2, "y=2"), (2, "z=3"), (4, "w=4")]
    for i in (1, 2, 3, 4):
        assert sim.nodes[i].log == expected, ("log mismatch", i)

    # Randomized churn: safety must hold under arbitrary timeouts/partitions/crashes.
    sim.run(random_actions(64, 24680))

    # Fair finale: heal, revive everything, elect the most up-to-date node, commit one more command.
    sim.run([("heal",)] + [("recover", i) for i in range(N)] +
            [("drain",), ("drain",)])
    best = max(range(N), key=lambda i: (sim.nodes[i].log[-1][0] if sim.nodes[i].log else 0,
                                        len(sim.nodes[i].log)))
    for _ in range(40):
        sim.run([("timeout", best), ("drain",)])
        if sim.nodes[best].state == "L":
            break
    assert sim.nodes[best].state == "L", "most up-to-date node must win the fair election"
    sim.run([("client", "end")])
    for _ in range(12):
        sim.run([("replicate",), ("drain",), ("replicate",), ("drain",)])
        if sim.committed_up_to == len(sim.nodes[best].log):
            break
    assert sim.committed_up_to == len(sim.nodes[best].log)
    assert (sim.nodes[best].term, "end") in sim.committed.values()
    print("PASS")


if __name__ == "__main__":
    main()
