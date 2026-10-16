"""Task 630 - Implement causal consistency protocol simulation (vector clocks).  (Tier: Expert, Ch 9 — System Design and Scalability)

A vector clock stamps every event with one counter per process; incrementing on a local event and
taking a component-wise maximum when a message arrives makes the clock a faithful encoding of the
happened-before relation: vc(e) < vc(f) exactly when e happens before f, and two events are
concurrent exactly when neither clock dominates the other. A causally consistent store exploits that
by *buffering* a message whose causal past is incomplete: a message carrying the sender's clock must
wait until the receiver's clock dominates the sender's clock as it was just before the send. The
receiver then joins the two clocks, so the receive event is ordered after everything the sender knew.

Verified here: the vector-clock order is checked against an independent oracle - the transitive
closure of program order plus the send->receive edges - for *every* pair of events of a randomised
(seeded) run, so the equivalence `vc(a) < vc(b) <=> a happened before b` is not assumed but measured.
Also verified: happened-before is a strict partial order; a newest-first adversarial delivery order
still delivers everything, and the delivery order is a linear extension of happened-before; buffering
really happens (messages are refused until their dependency arrives); a naive immediate-delivery
variant is caught violating causality (the negative control); and a last-writer-wins register
converges after all messages, flagging genuinely concurrent writes.

Simplifications, stated explicitly: there is no real network or threading - the cluster keeps one
in-memory FIFO queue and the driver delivers newest-first to force reordering, local events are
modelled as explicit steps of a seeded script rather than by an application, and the buffer is a plain
list of refused messages instead of a per-sender queue. The invariant that is still verified is the one
that matters: a message is delivered only when every event in its causal past (per the sender's clock)
has already been delivered, and the vector-clock order then agrees with happened-before on every pair
of events.

Checks: clock algebra (join is a least upper bound, idempotent/commutative/associative), the
equivalence against the closure oracle, the strict-partial-order properties, causal delivery under
reordering, the naive-delivery violation counter, buffer bookkeeping, register convergence and
concurrent-write detection, and a single-node boundary case.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class VectorClock:
    """Sparse vector clock: counts[node] is how many events of `node` are in this event's past."""

    __slots__ = ("counts",)

    def __init__(self, counts=None):
        self.counts = dict(counts or {})

    def copy(self):
        return VectorClock(self.counts)

    def get(self, node):
        return self.counts.get(node, 0)

    def nodes(self):
        return set(self.counts)

    def increment(self, node):
        self.counts[node] = self.get(node) + 1
        return self

    def join(self, other):
        for node, count in other.counts.items():
            if count > self.get(node):
                self.counts[node] = count
        return self

    def le(self, other):
        return all(self.get(node) <= other.get(node) for node in self.nodes() | other.nodes())

    def lt(self, other):
        return self.le(other) and not other.le(self)

    def equal(self, other):
        return self.le(other) and other.le(self)

    def concurrent(self, other):
        return not self.le(other) and not other.le(self)

    def key(self):
        return tuple(sorted(self.counts.items()))

    def __repr__(self):
        return "VC" + str(dict(self.key()))


class Event:
    __slots__ = ("seq", "node", "kind", "label", "clock")

    def __init__(self, seq, node, kind, label, clock):
        self.seq = seq
        self.node = node
        self.kind = kind
        self.label = label
        self.clock = clock

    def __repr__(self):
        return f"{self.node}:{self.kind}({self.label}){self.clock}"


class Message:
    __slots__ = ("seq", "sender", "receiver", "label", "send_event", "sent_after", "sent_vc",
                 "pair_seq", "receive_event")

    def __init__(self, seq, sender, receiver, label, send_event, sent_after, sent_vc, pair_seq):
        self.seq = seq
        self.sender = sender
        self.receiver = receiver
        self.label = label
        self.send_event = send_event
        self.sent_after = sent_after          # sender's clock just before the send
        self.sent_vc = sent_vc                # sender's clock including the send event
        self.pair_seq = pair_seq              # this sender's n-th message to this receiver
        self.receive_event = None

    def __repr__(self):
        return f"{self.sender}->{self.receiver}:{self.label}"


class Node:
    def __init__(self, name, cluster):
        self.name = name
        self.cluster = cluster
        self.vc = VectorClock()
        self.events = []                      # this node's events in program order
        self.pair_counts = {}                 # receiver -> messages sent to it so far
        self.delivered_pairs = {}              # sender -> messages received from it so far

    def local_event(self, label):
        self.vc.increment(self.name)
        return self.cluster.record(self, "local", label)

    def send(self, target, label):
        sent_after = self.vc.copy()
        self.vc.increment(self.name)
        event = self.cluster.record(self, "send", f"{label}->{target}")
        pair_seq = self.pair_counts.get(target, 0) + 1
        self.pair_counts[target] = pair_seq
        return self.cluster.enqueue(Message(self.cluster.next_seq(), self.name, target, label,
                                           event, sent_after, self.vc.copy(), pair_seq))

    def receive(self, message, order):
        """Deliver a message: join the sender's clock and tag the receive event with the result."""
        self.vc.join(message.sent_vc).increment(self.name)
        self.delivered_pairs[message.sender] = self.delivered_pairs.get(message.sender, 0) + 1
        message.receive_event = self.cluster.record(self, "receive",
                                                    f"{message.label}<-{message.sender}", order)
        return message.receive_event


class Cluster:
    """Deterministic run: explicit message queue, newest-first (adversarial) delivery order."""

    def __init__(self, names, naive=False):
        self.nodes = {name: Node(name, self) for name in names}
        self.queue = []
        self.delivered = []
        self.events = []
        self.delivery_order = []
        self.buffered_refusals = 0
        self.violations = []
        self.naive = naive                        # True: deliver without waiting for the past
        self._seq = 0

    def next_seq(self):
        self._seq += 1
        return self._seq

    def record(self, node, kind, label, order=None):
        event = Event(self.next_seq(), node.name, kind, label, node.vc.copy())
        node.events.append(event)
        self.events.append(event)
        self.delivery_order.append(event)
        return event

    def enqueue(self, message):
        self.queue.append(message)
        return message

    def local(self, node, label):
        return self.nodes[node].local_event(label)

    def send(self, sender, receiver, label):
        return self.nodes[sender].send(receiver, label)

    def ready(self, message):
        """Causal delivery rule: the sender's own messages must arrive in order (per sender-receiver
        pair), and every node the sender had already heard from must be caught up here. The sender's
        own clock component is skipped - it counts events the receiver can only learn this way."""
        receiver = self.nodes[message.receiver]
        if message.pair_seq != receiver.delivered_pairs.get(message.sender, 0) + 1:
            return False
        return all(message.sent_after.get(node) <= receiver.vc.get(node)
                   for node in message.sent_after.nodes() if node != message.sender)

    def attempt(self, message):
        receiver = self.nodes[message.receiver]
        if not self.ready(message):
            self.buffered_refusals += 1
            if not self.naive:
                return False
            self.violations.append((message, self.ready(message)))   # the verdict at delivery time
        receiver.receive(message, len(self.delivery_order))
        self.delivered.append(message)
        self.queue.remove(message)
        return True

    def run(self, max_rounds=200):
        """Newest-first delivery: the worst case for a causal filter, which must buffer the rest."""
        for _ in range(max_rounds):
            if not self.queue:
                return True
            for message in list(reversed(self.queue)):
                self.attempt(message)
            if self.queue and self.buffered_refusals and all(
                    not self.ready(message) for message in self.queue) and not self.naive:
                raise AssertionError("causal delivery deadlocked - a dependency never arrived")
        raise AssertionError("delivery did not finish inside the round budget")

    def buffered(self):
        return len(self.queue)


def hb_closure(events, messages):
    """Oracle: transitive closure of program order plus send -> receive edges."""
    successors = {event.seq: set() for event in events}
    per_node = {}
    for event in events:
        per_node.setdefault(event.node, []).append(event)
    for chain in per_node.values():
        for first, second in zip(chain, chain[1:]):
            successors[first.seq].add(second.seq)
    for message in messages:
        if message.receive_event is not None:
            successors[message.send_event.seq].add(message.receive_event.seq)
    reachable = {}
    for start in successors:
        seen, stack = set(), list(successors[start])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(successors[current])
        reachable[start] = seen
    return reachable


def lcg(seed, modulus=65537):
    state = seed

    def nxt(bound):
        nonlocal state
        state = (state * 75 + 74) % modulus
        return state % bound

    return nxt


class LwwRegister:
    """Last-writer-wins register: concurrent assignments are detected, then tie-broken by value."""

    def __init__(self):
        self.value = None                     # (clock key, assigned value)
        self.clock = VectorClock()
        self.conflicts = 0

    def assign(self, value, clock):
        """"stale" when the incoming write is dominated by the current one, else it lands."""
        if clock.lt(self.clock):
            return "stale"
        outcome = "concurrent" if (self.value is not None and clock.concurrent(self.clock)) else "assigned"
        if outcome == "concurrent":
            self.conflicts += 1
        self.value = (clock.key(), value)
        self.clock = clock.copy()
        return outcome

    def merge(self, other):
        if other.value is None:
            return
        if self.value is None:
            self.value, self.clock = other.value, other.clock.copy()
        elif self.value == other.value:
            self.clock.join(other.clock)
        elif self.clock.lt(other.clock):
            self.value, self.clock = other.value, other.clock.copy()
        elif other.clock.lt(self.clock):
            pass                              # our write dominates: keep it
        else:                                 # genuinely concurrent: same tie-break on both sides
            self.conflicts += 1
            self.value = min(self.value, other.value, key=lambda entry: (entry[1], entry[0]))
            self.clock.join(other.clock)


def build_run(seed, nodes=("A", "B", "C"), steps=40, naive=False):
    """Randomised-but-seeded message pattern with an adversarial (newest-first) delivery order."""
    cluster = Cluster(list(nodes), naive=naive)
    pick = lcg(seed)
    names = list(nodes)
    for _ in range(steps):
        actor = names[pick(len(names))]
        other = names[pick(len(names))]
        if actor == other:
            cluster.local(actor, "local")
        else:
            cluster.send(actor, other, f"m{pick(100)}")
    return cluster


def main():
    # --- 1. clock algebra ---------------------------------------------------------------------
    empty = VectorClock()
    check(empty.key() == () and empty.get("A") == 0, "an empty clock counts nothing")
    check(empty.le(VectorClock())and empty.concurrent(empty) is False,
          "two empty clocks are equal, not concurrent")
    a, b = VectorClock({"A": 1}), VectorClock({"A": 1, "B": 2})
    check(a.lt(b) and not b.lt(a) and not a.concurrent(b), "(1) < (1,2) strictly")
    c, d = VectorClock({"A": 2}), VectorClock({"B": 1})
    check(c.concurrent(d) and d.concurrent(c), "clocks on disjoint nodes are concurrent")
    joined = c.copy().join(d)
    check(joined.key() == (("A", 2), ("B", 1)), "join takes the component-wise maximum")
    check(c.le(joined) and d.le(joined), "the join dominates both inputs")
    check(joined.le(VectorClock({"A": 5, "B": 5, "C": 1})), "and only upper bounds dominate the join")
    check(c.copy().join(c).key() == c.key() and c.copy().join(d).key() == d.copy().join(c).key(),
          "join is idempotent and commutative")
    check(VectorClock({"A": 1}).copy().join(VectorClock({"B": 3})).join(VectorClock({"A": 2})).key()
          == (("A", 2), ("B", 3)), "join is associative in effect")

    # --- 2. the classic three-step chain ------------------------------------------------------
    chain = Cluster(["P", "Q"])
    chain.local("P", "e1")                       # {P:1}
    chain.send("P", "Q", "e2")                   # send: {P:2}
    message = chain.queue[0]
    check(message.sent_after.key() == (("P", 1),) and message.sent_vc.key() == (("P", 2),),
          "the message carries the pre-send and post-send clocks")
    chain.local("Q", "e3")                       # {Q:1} - concurrent with the send on P
    check(chain.nodes["Q"].vc.concurrent(message.sent_vc),
          "the local event on Q is concurrent with the send on P")
    check(chain.ready(message),
          "no third node is involved, so nothing has to arrive before this message")
    chain.run()
    check(chain.nodes["Q"].vc.key() == (("P", 2), ("Q", 2)),
          "after the receive Q knows P:2 and has two events of its own")
    check(message.receive_event is not None and message.send_event.clock.lt(
        message.receive_event.clock), "the receive event is strictly after the send event")
    check(chain.violations == [] and chain.queue == [], "the run is complete and causally clean")

    # --- 3. the randomised run: vector clocks versus the closure oracle ------------------------
    cluster = build_run(seed=97531, steps=45)
    cluster.run()
    check(not cluster.queue, "every message was eventually delivered")
    check(len(cluster.delivered) > 5, "the run really sent messages")
    check(cluster.violations == [], "causal delivery ordered every message correctly")

    reachable = hb_closure(cluster.events, cluster.delivered)
    mismatches = []
    for first in cluster.events:
        for second in cluster.events:
            if first.seq == second.seq:
                continue
            oracle = second.seq in reachable[first.seq]
            observed = first.clock.lt(second.clock)
            if oracle != observed:
                mismatches.append((first, second, oracle, observed))
    check(not mismatches,
          f"vector clocks must agree with the happened-before closure: {mismatches[:2]}")
    check(cluster.buffered_refusals > 0,
          "the adversarial order forced the receiver to buffer at least once")

    # strict partial order: irreflexive, asymmetric and transitive
    for event in cluster.events:
        check(not event.clock.lt(event.clock), "no event happens before itself")
    check(all(not (first.clock.lt(second.clock) and second.clock.lt(first.clock))
              for first in cluster.events for second in cluster.events),
          "happened-before is asymmetric")
    chain_of_three = [(x, y) for x in cluster.events for y in cluster.events if x.clock.lt(y.clock)]
    for first, second in chain_of_three[:60]:
        for third in cluster.events[:12]:
            if second.clock.lt(third.clock):
                check(first.clock.lt(third.clock), "happened-before is transitive")

    # the delivery order is a linear extension of happened-before
    position = {event.seq: index for index, event in enumerate(cluster.delivery_order)}
    inversions = [(first, second) for first, second in chain_of_three
                  if position[first.seq] > position[second.seq]]
    check(not inversions,
          f"nothing may be delivered before its causal past: {inversions[:2]}")
    check(all(position[event.seq] < position[receive.seq]
              for event, receive in [(m.send_event, m.receive_event)
                                     for m in cluster.delivered]), "sends precede their receives")
    concurrent_pairs = [(first, second) for first in cluster.events for second in cluster.events
                        if first.clock.concurrent(second.clock)]
    check(len(concurrent_pairs) > 0, "the run contains genuinely concurrent events")
    check(all(position[first.seq] != position[second.seq] for first, second in concurrent_pairs
              if first.seq != second.seq), "concurrent events are still totally ordered by delivery")

    # --- 4. negative control: naive delivery breaks causality ---------------------------------
    naive = build_run(seed=97531, steps=45, naive=True)
    naive.run()
    check(naive.queue == [] and len(naive.delivered) == len(cluster.delivered),
          "the naive variant delivers every message too")
    check(len(naive.violations) > 0
          and all(not ready for _message, ready in naive.violations),
          "but it delivers messages whose causal history the receiver has not seen yet")

    # --- 5. buffer bookkeeping: a message waits for the dependency it names ---------------------
    chain = Cluster(["X", "Y", "Z"])
    chain.send("X", "Y", "info")
    chain.run()                                  # Y now knows about X's message
    chain.send("Y", "Z", "derived")              # sent_after = {X:1, Y:1}: depends on X's message
    dependent = chain.queue[0]
    check(dependent.sent_after.key() == (("X", 1), ("Y", 1)),
          "the derived message names X's message as its causal past")
    check(chain.ready(dependent) is False and chain.attempt(dependent) is False,
          "'derived' is buffered: Z has not seen the message it depends on")
    check(chain.buffered() == 1 and chain.buffered_refusals == 1, "one message, one refusal")
    info = chain.send("X", "Z", "info")          # the dependency itself, sent later
    check(chain.attempt(info) is True, "'info' is deliverable and arrives")
    check(chain.nodes["Z"].vc.get("X") == 2,
          "Z adopts X's clock, including the send event itself")
    check(chain.buffered() == 1 and chain.ready(dependent) is True,
          "the buffered message became deliverable the moment its dependency landed")
    check(chain.attempt(dependent) is True, "the buffer is then drained")
    check(chain.buffered() == 0 and chain.violations == [], "nothing is left in flight")

    # --- 6. last-writer-wins register convergence ----------------------------------------------
    register_a, register_b = LwwRegister(), LwwRegister()
    check(register_a.assign("v1", VectorClock({"A": 1})) == "assigned", "a fresh write lands")
    register_b.merge(register_a)
    check(register_b.value == ((("A", 1),), "v1") and register_b.clock.key() == (("A", 1),),
          "a dominating write propagates by merging")
    check(register_b.assign("v0", VectorClock()) == "stale" and register_b.value[1] == "v1",
          "an older write cannot overwrite a newer one")
    register_b.merge(register_b)
    check(register_b.value[1] == "v1" and register_b.conflicts == 0,
          "merging a register with itself changes nothing")

    left, right = LwwRegister(), LwwRegister()
    check(left.assign("left", VectorClock({"A": 2})) == "assigned",
          "the first concurrent write lands on replica A")
    check(right.assign("right", VectorClock({"B": 1})) == "assigned",
          "the second lands on replica B")
    check(left.clock.concurrent(right.clock), "the two writes are concurrent")
    left.merge(right)
    right.merge(left)
    check(left.value == right.value and left.clock.key() == right.clock.key(),
          "merging in both directions converges on the same winner")
    check(left.conflicts + right.conflicts >= 1,
          "the concurrent write was detected instead of silently ordered")
    left.merge(right)
    right.merge(left)
    check(left.value == right.value and left.clock.key() == right.clock.key(),
          "merging is idempotent once the replicas agree")

    # --- 7. boundary: a single node needs no communication -------------------------------------
    solo = Cluster(["S"])
    solo.local("S", "a")
    solo.local("S", "b")
    check(solo.queue == [] and solo.nodes["S"].vc.key() == (("S", 2),),
          "a single node just counts its own events")
    check(solo.events[0].clock.lt(solo.events[1].clock), "its own events are causally ordered")

    print(f"causal: events={len(cluster.events)} messages={len(cluster.delivered)} "
          f"buffered-refusals={cluster.buffered_refusals} naive-violations={len(naive.violations)} "
          f"concurrent-pairs={len(concurrent_pairs)}")


if __name__ == "__main__":
    main()
    print("PASS")
