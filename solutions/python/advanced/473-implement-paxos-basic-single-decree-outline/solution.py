"""Task 473 - Implement Paxos basic single-decree outline. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

Single-decree Paxos with 3 acceptors and 2 proposers, driven by a scripted message queue
(explicit deliver/drop actions). Proposers run textbook phase 1 / phase 2, including the
recovery rule: if any promise carries an accepted value, the proposer must re-propose the
value with the highest accepted round. Safety oracle: whenever a majority of acceptors has
accepted the same (round, value), that value counts as chosen; the simulation asserts that at
most one distinct value is ever chosen (across engine events and the acceptor-log oracle).
Liveness is shown in a fair (no-loss) scripted run and in a final fair segment.
"""

MAJORITY = 2  # of 3 acceptors


class Acceptor:
    def __init__(self, i):
        self.id = i
        self.promised = -1
        self.accepted_round = -1
        self.accepted_value = None
        self.history = []          # (round, value) in accept order


class Proposer:
    def __init__(self, i):
        self.id = i
        self.phase = "idle"        # idle | P1 | P2 | done
        self.round = -1
        self.own_value = None
        self.decided_value = None
        self.promises = []         # (acceptor id, accepted round, accepted value)
        self.accepts = set()       # acceptor ids that accepted the current round
        self.chosen = None         # (round, value)


class PaxosSim:
    def __init__(self):
        self.acceptors = [Acceptor(i) for i in range(3)]
        self.proposers = [Proposer(i) for i in range(2)]
        self.queue = []            # (("A", id) | ("P", id), message)
        self.chosen_events = []    # engine-reported (round, value)
        self.oracle_chosen = set()
        self.last_promised = [-1, -1, -1]

    # ---------- proposals / network ----------
    def propose(self, p, round_no, value):
        pr = self.proposers[p]
        pr.phase = "P1"
        pr.round = round_no
        pr.own_value = value
        pr.decided_value = None
        pr.promises = []
        pr.accepts = set()
        for a in range(3):
            self.queue.append((("A", a), ("prepare", round_no, p)))

    def _deliver_one(self):
        dest, msg = self.queue.pop(0)
        kind = msg[0]
        if dest[0] == "A":
            a = self.acceptors[dest[1]]
            if kind == "prepare":
                _, round_no, p = msg
                if round_no > a.promised:
                    a.promised = round_no
                    self.queue.append((("P", p), ("promise", round_no, a.id,
                                                  a.accepted_round, a.accepted_value)))
                else:
                    self.queue.append((("P", p), ("nack", round_no, a.id)))
            elif kind == "accept":
                _, round_no, value, proposer = msg
                if round_no >= a.promised:
                    a.promised = round_no
                    a.accepted_round = round_no
                    a.accepted_value = value
                    a.history.append((round_no, value))
                    self.queue.append((("P", proposer), ("accepted", round_no, a.id, value)))
                else:
                    self.queue.append((("P", proposer), ("nack", round_no, a.id)))
        else:
            pr = self.proposers[dest[1]]
            if kind == "promise":
                _, round_no, acc_id, acc_round, acc_value = msg
                if pr.phase != "P1" or round_no != pr.round:
                    return
                pr.promises.append((acc_id, acc_round, acc_value))
                if len(pr.promises) >= MAJORITY:
                    best_round = -1
                    best_value = pr.own_value
                    for _acc, acc_round, acc_value in pr.promises:
                        if acc_value is not None and acc_round > best_round:
                            best_round = acc_round
                            best_value = acc_value
                    pr.decided_value = best_value
                    pr.phase = "P2"
                    for a in range(3):
                        self.queue.append((("A", a), ("accept", round_no, best_value, pr.id)))
            elif kind == "accepted":
                _, round_no, acc_id, value = msg
                if pr.phase != "P2" or round_no != pr.round:
                    return
                pr.accepts.add(acc_id)
                if len(pr.accepts) >= MAJORITY:
                    pr.chosen = (round_no, value)
                    pr.phase = "done"
                    self.chosen_events.append((round_no, value))

    def deliver(self, k):
        for _ in range(k):
            if self.queue:
                self._deliver_one()

    def drain(self):
        while self.queue:
            self._deliver_one()

    def drop(self, k):
        for _ in range(k):
            if self.queue:
                self.queue.pop(0)

    # ---------- invariants ----------
    def check(self):
        oracle = set()
        for a in self.acceptors:
            for round_no, value in a.history:
                count = sum(1 for b in self.acceptors if (round_no, value) in b.history)
                if count >= MAJORITY:
                    oracle.add(value)
        self.oracle_chosen |= oracle
        chosen_values = {v for _r, v in self.chosen_events} | self.oracle_chosen
        assert len(chosen_values) <= 1, ("at most one value may be chosen", chosen_values)
        for a in self.acceptors:
            assert a.promised >= self.last_promised[a.id], ("promise regression", a.id)
            self.last_promised[a.id] = a.promised
            rounds = [r for r, _v in a.history]
            assert rounds == sorted(rounds), ("accept rounds not monotone", a.id)

    def run(self, actions):
        for action in actions:
            kind = action[0]
            if kind == "propose":
                self.propose(action[1], action[2], action[3])
            elif kind == "deliver":
                self.deliver(action[1])
            elif kind == "drain":
                self.drain()
            elif kind == "drop":
                self.drop(action[1])
            self.check()


def lcg(seed):
    return (seed * 75 + 74) % 65537


def random_actions(count, seed, start_round):
    actions = []
    rnd = start_round
    for _ in range(count):
        seed = lcg(seed)
        a = seed % 8
        seed = lcg(seed)
        p = seed % 2
        seed = lcg(seed)
        value = seed % 20
        seed = lcg(seed)
        k = seed % 3
        if a < 3:
            actions.append(("propose", p, rnd, value))
            rnd += 1
        elif a == 3:
            actions.append(("deliver", 1 + k))
        elif a < 6:
            actions.append(("drain",))
        elif a == 6:
            actions.append(("drop", k))
        else:
            actions.append(("deliver", 1))
    return actions


def main():
    sim = PaxosSim()

    # P0's value 7 is accepted by a majority (A0, A1) but all replies are dropped:
    # the value is *chosen* although P0 never learns that.
    sim.run([("propose", 0, 1000, 7), ("deliver", 3), ("deliver", 2), ("deliver", 3), ("drop", 3)])
    assert sim.chosen_events == []
    assert sim.oracle_chosen == {7}
    assert sim.acceptors[0].history == [(1000, 7)]
    assert sim.acceptors[1].history == [(1000, 7)]
    assert sim.acceptors[2].history == []

    # P1 proposes 9 with a higher round; the promises carry (1000, 7), so it must re-propose 7.
    sim.run([("propose", 1, 1001, 9), ("drain",)])
    assert sim.proposers[1].decided_value == 7, "P1 must adopt the chosen value, not its own 9"
    assert sim.proposers[1].chosen == (1001, 7)
    assert sim.chosen_events == [(1001, 7)]
    assert sim.oracle_chosen == {7}

    # P0 retries with round 1002 and its own value 8; it must also re-propose 7.
    sim.run([("propose", 0, 1002, 8), ("drain",)])
    assert sim.proposers[0].decided_value == 7, "P0 retry must also re-propose 7"
    assert sim.chosen_events == [(1001, 7), (1002, 7)]
    assert sim.acceptors[0].history == [(1000, 7), (1001, 7), (1002, 7)]
    assert sim.acceptors[1].history == [(1000, 7), (1001, 7), (1002, 7)]
    assert sim.acceptors[2].history == [(1001, 7), (1002, 7)]
    accepted_values = {v for a in sim.acceptors for _r, v in a.history}
    assert accepted_values == {7}, "values 9 and 8 must never be accepted anywhere"

    # Randomized churn with losses: safety (at most one chosen value) must hold throughout.
    sim.run(random_actions(30, 13579, 2000))

    # Fair finale: fresh proposals with all messages delivered; liveness must return.
    for p in range(2):
        sim.run([("propose", p, 3000 + p, 5 + p), ("drain",)])
    assert sim.chosen_events, "fair run must choose a value"
    assert {v for _r, v in sim.chosen_events} == {7}
    assert sim.oracle_chosen == {7}
    print("PASS")


if __name__ == "__main__":
    main()
