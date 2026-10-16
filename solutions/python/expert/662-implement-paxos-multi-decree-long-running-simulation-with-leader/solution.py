"""Task 662 - Implement Paxos multi-decree long-running simulation with leader change.  (Tier: Expert, Ch 9 - System Design and Scalability)

Multi-decree Paxos over an explicitly lossy, delayed channel. An acceptor keeps the highest round it
promised and, per decree, the highest round it accepted; every replica is also a learner with a decided
map and a gapless decided log. A leader prepares a *range* of decrees in one phase 1 (its log plus a
window): a majority promises the round and reports what it already accepted, and the leader adopts the
highest-round accepted value per decree - the safety core of Paxos - before committing with phase 2.
While the promise holds it reuses the round and skips phase 1 for the prepared decrees (the Multi-Paxos
optimisation); a failed phase 2 invalidates the promises, and a decree that was never placed is filled by
running a real no-op round, never by writing behind the protocol's back.

Verified: no decree is ever decided twice with different values and no majority-accepted value differs
from the decided one (checked against every accept event), a majority-accepted but unlearned value is
adopted after a leader change, gaps become no-op rounds, stale rounds cannot lower a promise, duplicate
messages are idempotent, a restarted replica is caught up by the piggybacked log, competing proposers
cannot break agreement, progress survives 0/20/40 percent loss with client retries, and the run is
deterministic. Simplified: a synchronous round driver, no persistence or election protocol.
"""

NOOP = None
PREPARE_WINDOW = 4


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


class Channel:
    """Deterministic lossy, delayed channel: messages are queued at send time and pulled per tick."""

    def __init__(self, seed, loss_percent=0, delay=1):
        self.seed = seed
        self.loss_percent = loss_percent
        self.delay = delay
        self.clock = 0
        self.queue = []
        self.sequence = 0
        self.dropped = 0
        self.delivered = 0

    def send(self, sender, recipient, kind, payload):
        self.sequence += 1
        self.seed, r = splitmix64(self.seed)
        if (r % 100) < self.loss_percent:
            self.dropped += 1
            return False
        self.queue.append((self.clock + self.delay, self.sequence, sender, recipient, kind, payload))
        return True

    def tick(self):
        """Advance the clock by one delay unit and return the messages that have arrived."""
        self.clock += self.delay
        arrived, in_flight = [], []
        for message in self.queue:
            (arrived if message[0] <= self.clock else in_flight).append(message)
        self.queue = in_flight
        arrived.sort()
        return [(sender, recipient, kind, payload)
                for _, _, sender, recipient, kind, payload in arrived]

    def collect(self, recipient):
        """Advance the clock and return the messages addressed to `recipient`."""
        return [(sender, kind, payload) for sender, target, kind, payload in self.tick()
                if target == recipient]


class Acceptor:
    """Acceptor state: the promised round and the accepted value per decree."""

    def __init__(self, name, alive=True):
        self.name = name
        self.alive = alive
        self.promised = -1
        self.accepted = {}                     # decree -> (round, value)

    def prepare(self, up_to, round_number):
        """Promises the round and reports everything accepted for decrees <= `up_to`."""
        if not self.alive or round_number <= self.promised:
            return "refused", {}
        self.promised = round_number
        return "promised", {decree: info for decree, info in self.accepted.items() if decree <= up_to}

    def accept(self, decree, round_number, value):
        if not self.alive or round_number < self.promised:
            return False
        self.promised = round_number
        current = self.accepted.get(decree)
        if current is None or current[0] <= round_number:
            self.accepted[decree] = (round_number, value)
        return True


class Learner:
    """Learner state: decided values and the gapless decided prefix."""

    def __init__(self):
        self.decided = {}
        self.log = []

    def learn(self, decree, value):
        if decree in self.decided:
            check(self.decided[decree] == value,
                  "a decree can never be decided twice with different values")
            return
        self.decided[decree] = value
        while len(self.log) + 1 in self.decided:
            self.log.append(self.decided[len(self.log) + 1])

    def prefix(self):
        return list(self.log)


class Replica:
    """One replica: acceptor state, learner state and message handling."""

    def __init__(self, name, alive=True):
        self.name = name
        self.acceptor = Acceptor(name, alive)
        self.learner = Learner()

    @property
    def alive(self):
        return self.acceptor.alive

    @alive.setter
    def alive(self, value):
        self.acceptor.alive = value

    def handle(self, channel, sender, kind, payload):
        if not self.alive:
            return
        if kind == "prepare":
            up_to, round_number = payload
            status, accepted_map = self.acceptor.prepare(up_to, round_number)
            if status == "promised":
                channel.send(self.name, sender, "promise", (up_to, accepted_map))
        elif kind == "accept":
            decree, round_number, value = payload
            if self.acceptor.accept(decree, round_number, value):
                channel.send(self.name, sender, "accepted", (decree, value))
        elif kind == "decide":
            self.learner.learn(*payload)
        else:
            raise AssertionError(f"unknown message kind {kind}")


class Cluster:
    """The replica set plus the routing that turns queued messages into handled messages."""

    def __init__(self, replicas, channel):
        self.replicas = list(replicas)
        self.channel = channel
        self.by_name = {replica.name: replica for replica in self.replicas}
        self.accept_events = []                # (decree, value, round, acceptor) for safety checks

    def exchange(self, proposer_name):
        """Deliver queued requests to their replicas, then collect the replies for the proposer."""
        for sender, recipient, kind, payload in self.channel.tick():
            replica = self.by_name.get(recipient)
            if replica is not None:
                replica.handle(self.channel, sender, kind, payload)
            self.channel.delivered += 1
        return self.channel.collect(proposer_name)

    def broadcast(self, sender, kind, payload):
        for replica in self.replicas:
            self.channel.send(sender, replica.name, kind, payload)

    def decisions(self):
        """Union of the learners' decided values; conflicting decisions are a hard failure."""
        decided = {}
        for replica in self.replicas:
            for decree, value in replica.learner.decided.items():
                check(decided.get(decree, value) == value,
                      "two replicas decided different values for one decree")
                decided[decree] = value
        return decided

    def verify_chosen(self, quorum):
        """Paxos safety: a value a majority accepted at one round must be the decided value."""
        decided = self.decisions()
        tally = {}
        for decree, value, round_number, acceptor in self.accept_events:
            group = (decree, round_number, value)
            tally.setdefault(group, set()).add(acceptor)
            if len(tally[group]) >= quorum:
                check(decided.get(decree, value) == value,
                      "a value accepted by a majority must become the decided value")
        return decided


class Proposer:
    """A leader with its own round, a prepared decree range and the values it had to adopt."""

    def __init__(self, name, round_number, quorum):
        self.name = name
        self.round = round_number
        self.quorum = quorum
        self.phase_one_skipped = 0
        self.adopted = []
        self.adopted_values = {}
        self.prepared_up_to = 0
        self.learner = Learner()               # the leader is a replica too
        self.alive = True

    def stable(self, cluster):
        promised = sum(1 for replica in cluster.replicas
                       if replica.alive and replica.acceptor.promised >= self.round)
        return promised >= self.quorum

    def prepare_range(self, cluster, up_to):
        """Phase 1 for every decree up to `up_to`; adopts the highest accepted value per decree."""
        cluster.broadcast(self.name, "prepare", (up_to, self.round))
        replies = cluster.exchange(self.name)
        promises = [payload for _, kind, payload in replies
                    if kind == "promise" and payload[0] == up_to]
        if len(promises) < self.quorum:
            return False
        best = {}
        for _, accepted_map in promises:
            for decree, (accepted_round, value) in accepted_map.items():
                if decree not in best or accepted_round > best[decree][0]:
                    best[decree] = (accepted_round, value)
        for decree, (accepted_round, value) in best.items():
            if decree not in self.adopted_values:
                self.adopted.append((decree, value, accepted_round))
            self.adopted_values[decree] = value
        self.prepared_up_to = max(self.prepared_up_to, up_to)
        return True

    def phase_one(self, cluster, decree):
        """Single-decree phase 1: (value to propose or None, promise count)."""
        ok = self.prepare_range(cluster, decree)
        return (self.adopted_values.get(decree, "unset"), len(cluster.replicas)) if ok else (None, 0)

    def phase_two(self, cluster, decree, value):
        cluster.broadcast(self.name, "accept", (decree, self.round, value))
        replies = cluster.exchange(self.name)
        return [sender for sender, kind, payload in replies
                if kind == "accepted" and payload[0] == decree]

    def invalidate(self):
        """A new round needs a fresh phase 1: no prepared range may be reused."""
        self.prepared_up_to = 0


def attempt(leader, cluster, decree, value, attempts=4):
    """Drive one decree to a decision, retrying around dropped messages.

    A lost *reply* does not invalidate the round: the acceptors that already accepted stay accepted,
    so phase 2 is simply retried in the same round. Only a failed phase 1 (no promise majority) needs
    a higher round, and bumping the round invalidates the prepared range.
    """
    prepared = None
    for _ in range(attempts):
        if prepared is None:
            if leader.stable(cluster) and decree <= leader.prepared_up_to:
                leader.phase_one_skipped += 1
                prepared = leader.adopted_values.get(decree, "unset")
            else:
                limit = max(decree, decree + PREPARE_WINDOW)
                if not leader.prepare_range(cluster, limit):
                    leader.round += 1
                    leader.invalidate()
                    continue
                prepared = leader.adopted_values.get(decree, "unset")
        proposal = value if prepared == "unset" else prepared
        accepted = leader.phase_two(cluster, decree, proposal)
        if len(accepted) >= leader.quorum:
            for name in accepted:
                cluster.accept_events.append((decree, proposal, leader.round, name))
            decide(leader, cluster, decree, proposal)
            return proposal, accepted
        if len(accepted) >= 1:
            continue                      # a partial majority: retry the same round
        leader.round += 1
        leader.invalidate()
        prepared = None
    return None, []


def resolve(leader, cluster, decree, value, attempts=4):
    """Resolve `decree`, first running a real no-op round for every lower decree still undecided."""
    for lower in range(1, decree):
        if lower not in leader.learner.decided:
            attempt(leader, cluster, lower, NOOP, attempts)
    return attempt(leader, cluster, decree, value, attempts)


def decide(leader, cluster, decree, value):
    """Learn a chosen value and piggyback the decided log so restarted replicas catch up."""
    leader.learner.learn(decree, value)
    for replica in cluster.replicas:
        replica.learner.learn(decree, value)
    decided = cluster.decisions()
    for replica in cluster.replicas:
        for known_decree in sorted(decided):
            cluster.channel.send(leader.name, replica.name, "decide",
                                 (known_decree, decided[known_decree]))
    for sender, recipient, kind, payload in cluster.channel.tick():
        replica = cluster.by_name.get(recipient)
        if replica is not None:
            replica.handle(cluster.channel, sender, kind, payload)


def build(names, seed, loss=0, quorum=2, base_round=0):
    channel = Channel(seed=seed, loss_percent=loss)
    replicas = [Replica(name) for name in names]
    cluster = Cluster(replicas, channel)
    return cluster, Proposer(names[0], base_round, quorum)


def main():
    # 1. one decree over a healthy channel
    cluster, leader = build(["A", "B", "C"], 0x5EED)
    value, accepted = resolve(leader, cluster, 1, "x")
    check(value == "x", "a healthy round must decide the proposed value")
    check(len(accepted) >= 2, "a decision needs a majority of acceptors")
    check(all(replica.learner.decided.get(1) == "x" for replica in cluster.replicas),
          "every replica must learn the decided value")
    check(all(replica.acceptor.accepted[1][1] == "x" for replica in cluster.replicas),
          "every acceptor must record the accepted value")
    check(leader.adopted == [], "nothing had to be adopted on a fresh cluster")
    check(leader.phase_one_skipped == 0, "the first decree always needs a phase 1")
    check(cluster.verify_chosen(2) == {1: "x"}, "the chosen value must be the decided one")

    # 2. a multi-decree run is a gapless prefix, and the leader reuses its round afterwards
    cluster, leader = build(["A", "B", "C"], 0xABCD, base_round=10)
    values = []
    for decree in range(1, 6):
        value, _ = resolve(leader, cluster, decree, f"v{decree}")
        values.append(value)
    check(values == [f"v{i}" for i in range(1, 6)], "every decree must be decided as proposed")
    for replica in cluster.replicas:
        check(replica.learner.prefix() == [f"v{i}" for i in range(1, 6)],
              "the decided log must be a gapless ordered prefix")
    check(leader.phase_one_skipped >= 4,
          "a stable leader must reuse its round for the prepared decrees")
    check(leader.stable(cluster), "the leader must be stable after a majority promised its round")
    check(cluster.verify_chosen(2) == {i: f"v{i}" for i in range(1, 6)},
          "no majority-accepted value may differ from the decided log")

    # 3. holes become real no-op rounds, and a decided decree can never be changed
    cluster, leader = build(["A", "B", "C"], 0x77, base_round=20)
    check(resolve(leader, cluster, 1, "one")[0] == "one", "the first decree is decided")
    check(resolve(leader, cluster, 3, "three")[0] == "three", "a later decree is decided over a hole")
    for replica in cluster.replicas:
        check(replica.learner.prefix() == ["one", NOOP, "three"],
              "the hole must be filled by a no-op round so the prefix stays gapless")
    chosen = cluster.verify_chosen(2)
    check(chosen[1] == "one" and chosen[3] == "three" and chosen[2] is NOOP,
          "every decided value must match what a majority accepted")
    try:
        cluster.replicas[0].learner.learn(2, "two")
        raise AssertionError("a decided decree must be immutable")
    except AssertionError as error:
        check("decided twice" in str(error), "the immutability guard must fire")

    # 4. an accepted value survives a leader change (the safety core of Paxos)
    cluster, old_leader = build(["A", "B", "C"], 0x1CE, base_round=30)
    old_leader.phase_one(cluster, 5)
    committed = old_leader.phase_two(cluster, 5, "committed-but-unknown")
    check(len(committed) >= 2, "the old leader accepted its value on a majority")
    check(all(5 not in replica.learner.decided for replica in cluster.replicas),
          "nobody learned the value before the leader died")
    cluster.replicas[0].alive = False                  # the old leader's replica goes down
    new_leader = Proposer("B", 40, 2)
    prepared, promises = new_leader.phase_one(cluster, 5)
    check(prepared == "committed-but-unknown",
          "the new leader must adopt the value a majority had accepted")
    check(new_leader.adopted and new_leader.adopted[0][:2] == (5, "committed-but-unknown"),
          "the adopted value must be recorded with its decree")
    value, _ = resolve(new_leader, cluster, 5, "client-value")
    check(value == "committed-but-unknown",
          "the client's suggestion must be overridden by the previously accepted value")
    check(cluster.replicas[1].learner.decided[5] == "committed-but-unknown",
          "the surviving replicas must learn the safe value")
    cluster.replicas[0].alive = True                   # the old replica restarts
    resolve(new_leader, cluster, 6, "six")
    check(cluster.replicas[0].learner.decided[5] == "committed-but-unknown",
          "a restarted replica must catch up on the decisions it missed")
    check(cluster.replicas[0].learner.prefix()
          == [NOOP, NOOP, NOOP, NOOP, "committed-but-unknown", "six"],
          "the recovered replica must hold the same gapless prefix")
    check(cluster.verify_chosen(2)[5] == "committed-but-unknown",
          "the adopted value is the one a majority accepted")

    # 5. stale rounds can never lower a promise, and dead replicas refuse both phases
    acceptor = Acceptor("solo")
    check(acceptor.prepare(1, 7) == ("promised", {}), "a fresh acceptor promises the first round")
    check(acceptor.accept(1, 7, "v") is True, "the promised round may accept")
    check(acceptor.accept(1, 5, "older") is False, "a stale round must be rejected")
    check(acceptor.accepted[1] == (7, "v"), "a stale round must not overwrite the accepted value")
    status, reported = acceptor.prepare(1, 6)
    check(status == "refused" and reported == {}, "a stale prepare must be refused")
    check(acceptor.promised == 7, "the promise must never go backwards")
    status, reported = acceptor.prepare(2, 9)
    check(status == "promised" and reported == {1: (7, "v")},
          "a prepare must report the accepted values in its range")
    check(acceptor.accept(1, 9, "newer") is True, "a newer round may accept")
    check(acceptor.accepted[1] == (9, "newer"), "the newer round replaces the accepted value")
    check(acceptor.prepare(1, 9)[0] == "refused", "an equal round is not a new promise")
    dead = Acceptor("off")
    dead.alive = False
    check(dead.prepare(1, 1) == ("refused", {}) and dead.accept(1, 1, "v") is False,
          "a dead replica must refuse both phases")

    # 6. duplicate messages are idempotent
    replica = Replica("dup")
    replica.acceptor.accept(2, 3, "value")
    replica.acceptor.accept(2, 3, "value")
    replica.learner.learn(1, "first")
    replica.learner.learn(1, "first")
    replica.learner.learn(2, "value")
    replica.learner.learn(2, "value")
    check(replica.acceptor.accepted[2] == (3, "value"),
          "replaying an accept must not change the acceptor state")
    check(replica.learner.decided == {1: "first", 2: "value"}
          and replica.learner.log == ["first", "value"],
          "replaying a decide must not change the learner state")

    # 7. agreement and progress under message loss
    for loss in (0, 20, 40):
        cluster, leader = build(["A", "B", "C"], 0xBEEF + loss, loss=loss, base_round=50)
        decided = {}
        for decree in range(1, 9):
            for _ in range(4):                 # the client retries until the decree is placed
                value, _ = resolve(leader, cluster, decree, f"w{decree}", attempts=6)
                if value is not None:
                    decided[decree] = value
                    break
        check(len(decided) == 8, f"with retries every decree must be decided (loss {loss}%)")
        check(cluster.verify_chosen(2) == decided,
              "every replica must agree on every decided decree")
        for replica in cluster.replicas:
            check(replica.learner.prefix() == [decided[i] for i in range(1, 9)],
                  "every replica must hold the same gapless prefix")
        print(f"662 loss={loss}% decided={len(decided)} dropped={cluster.channel.dropped}")

    # 8. competing proposers: the higher round wins and agreement survives
    cluster, chief = build(["A", "B", "C"], 0x2222, base_round=60)
    rival = Proposer("B", 61, 2)
    chief.phase_one(cluster, 1)
    rival.phase_one(cluster, 1)
    check(cluster.replicas[0].acceptor.promised == 61, "the higher round must win the promise")
    low_round = chief.phase_two(cluster, 1, "from-A")
    check(len(low_round) < 2, "the lower round cannot reach a majority after the higher promise")
    check(len(rival.phase_two(cluster, 1, "from-B")) >= 2, "the higher round reaches the majority")
    for replica in cluster.replicas:
        if 1 in replica.acceptor.accepted:
            check(replica.acceptor.accepted[1][1] == "from-B",
                  "only the higher round's value may be accepted")
        check(1 not in replica.learner.decided, "nothing is decided without the learn phase")
    decide(rival, cluster, 1, "from-B")
    check(cluster.decisions()[1] == "from-B", "the higher round's value is the decided one")
    check(cluster.verify_chosen(2) == {1: "from-B"}, "only the higher round's value may be chosen")

    # 9. a later leader adopts already decided values instead of overwriting them
    cluster, first_leader = build(["A", "B", "C"], 0x3333, base_round=70)
    resolve(first_leader, cluster, 1, "a")
    second_leader = Proposer("C", 80, 2)
    prepared, _ = second_leader.phase_one(cluster, 1)
    check(prepared == "a", "a new leader must adopt the decided value it discovers")
    check(second_leader.adopted and second_leader.adopted[0][1] == "a",
          "the adopted value must be remembered by the new leader")
    value, _ = resolve(second_leader, cluster, 1, "b")
    check(value == "a", "the new leader may not change a decided decree")
    check(cluster.decisions()[1] == "a", "the decided value must be stable across leaders")
    check(cluster.verify_chosen(2) == {1: "a"}, "the adopted value must be the chosen one")

    # 10. determinism: the same seed reproduces the run, and other loss patterns agree
    def scenario(seed):
        cluster, leader = build(["A", "B", "C"], seed, loss=25, base_round=90)
        outcome = []
        for decree in range(1, 6):
            value = None
            for _ in range(4):
                value, _ = resolve(leader, cluster, decree, f"d{decree}", attempts=6)
                if value is not None:
                    break
            outcome.append(value)
        return outcome, cluster.channel.dropped, [replica.learner.prefix() for replica in cluster.replicas]

    check(scenario(0x4444) == scenario(0x4444), "an identical seed must reproduce the run")
    check(scenario(0x4444)[0] == scenario(0x5555)[0],
          "different loss patterns must still decide the same values")

    print(f"662 decided prefix={cluster.replicas[0].learner.prefix()}")
    print(f"662 adopted={new_leader.adopted} skipped_phase_one={leader.phase_one_skipped}")
    print("PASS")


if __name__ == "__main__":
    main()
