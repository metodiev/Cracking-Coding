"""Task 636 - Implement hybrid logical clocks and test ordering in distributed events.  (Tier: Expert, Ch 9 — System Design and Scalability)

A hybrid logical clock keeps a pair (l, c): `l` tracks physical time, and `c` is a counter that breaks
ties inside the same `l`. The update rules are: for a local (or send) event, l' = max(l, pt) and
c' = c + 1 when l' is unchanged, else 0; when a message stamped (lm, cm) arrives, l' = max(l, lm, pt)
and c' is max(c, cm) + 1, c + 1, cm + 1 or 0 depending on which of the three values produced l'. Two
consequences are what make HLC useful: an event's l is always *exactly* the largest physical time in
its causal history (so no sender can push a receiver's clock into the future), and (l, c) strictly
increases along every chain of happened-before, so sorting events by (l, c) is a linear extension of
causality - the property behind "physical-time-like" ordering in a distributed system.

Verified here against ground truth rather than by assertion: every event also carries a vector clock,
whose ordering provably coincides with happened-before (checked by comparing it against the transitive
closure of program order plus send->receive edges), and that oracle is then used to (a) confirm the
l component equals the maximum physical time over each event's causal past, (b) confirm the (l, c)
order is a linear extension of causality, (c) confirm per-node uniqueness of (l, c), and (d) count how
often a naive physical-time ordering violates causality (always > 0 in a skewed run, while HLC never
does). Skewed clocks, same-millisecond batches, and a round-trip wire format are covered as well.

Simplifications, stated explicitly: physical time is a deterministic integer that advances by a fixed
step per simulated event (no wall clock), and the network is an in-memory FIFO queue driven by the
test. Nothing about the clock algorithm itself is simplified.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ---------------------------------------------------------------------------------------------
# the ground-truth oracle: a vector clock
# ---------------------------------------------------------------------------------------------
class VectorClock:
    __slots__ = ("counts",)

    def __init__(self, counts=None):
        self.counts = dict(counts or {})

    def copy(self):
        return VectorClock(self.counts)

    def get(self, node):
        return self.counts.get(node, 0)

    def increment(self, node):
        self.counts[node] = self.get(node) + 1
        return self

    def join(self, other):
        for node, count in other.counts.items():
            if count > self.get(node):
                self.counts[node] = count
        return self

    def lt(self, other):
        nodes = set(self.counts) | set(other.counts)
        less = any(self.get(node) < other.get(node) for node in nodes)
        return less and all(self.get(node) <= other.get(node) for node in nodes)

    def key(self):
        return tuple(sorted(self.counts.items()))


# ---------------------------------------------------------------------------------------------
# hybrid logical clock
# ---------------------------------------------------------------------------------------------
class HLC:
    """The hybrid logical clock itself: (l, c) with the three timestamp update rules."""

    def __init__(self, node):
        self.node = node
        self.l = 0
        self.c = 0
        self.ticks = 0

    def stamp(self):
        return (self.l, self.c)

    def local(self, physical_time):
        """A local event (and a send is a local event that also transmits the stamp)."""
        previous = self.l
        self.l = max(self.l, physical_time)
        self.c = self.c + 1 if self.l == previous else 0
        self.ticks += 1
        return self.stamp()

    def receive(self, physical_time, remote):
        lm, cm = remote
        last_l = self.l
        self.l = max(last_l, lm, physical_time)
        if self.l == last_l == lm:
            self.c = max(self.c, cm) + 1
        elif self.l == last_l:
            self.c = self.c + 1
        elif self.l == lm:
            self.c = cm + 1
        else:
            self.c = 0
        self.ticks += 1
        return self.stamp()


def wire_format(clock):
    """Fixed-width text so a byte comparison follows the (l, c) order."""
    return f"{clock[0]:016d}:{clock[1]:08d}"


def parse_wire(text):
    left, right = text.split(":")
    return (int(left), int(right))


class Event:
    __slots__ = ("seq", "node", "kind", "label", "clock", "vector", "physical")

    def __init__(self, seq, node, kind, label, clock, vector, physical):
        self.seq = seq
        self.node = node
        self.kind = kind
        self.label = label
        self.clock = clock
        self.vector = vector
        self.physical = physical

    def __repr__(self):
        return f"{self.node}:{self.kind}({self.label}){self.clock}"


class Message:
    __slots__ = ("seq", "sender", "receiver", "label", "clock", "vector", "send_event", "receive_event")

    def __init__(self, seq, sender, receiver, label, clock, vector, send_event):
        self.seq = seq
        self.sender = sender
        self.receiver = receiver
        self.label = label
        self.clock = clock
        self.vector = vector
        self.send_event = send_event
        self.receive_event = None


class Node:
    def __init__(self, name, cluster, start_time, step=5):
        self.name = name
        self.cluster = cluster
        self.start_time = start_time
        self.step = step
        self.physical = start_time
        self.hlc = HLC(name)
        self.vector = VectorClock()

    def advance(self):
        self.physical += self.step
        return self.physical

    def local_event(self, label):
        self.vector.increment(self.name)
        clock = self.hlc.local(self.physical)
        return self.cluster.record(self, "local", label, clock)

    def send(self, target, label):
        self.vector.increment(self.name)
        clock = self.hlc.local(self.advance())
        event = self.cluster.record(self, "send", f"{label}->{target}", clock)
        message = Message(self.cluster.next_seq(), self.name, target, label, clock,
                          self.vector.copy(), event)
        return self.cluster.enqueue(message)

    def receive(self, message):
        self.vector.join(message.vector)
        self.vector.increment(self.name)
        clock = self.hlc.receive(self.advance(), message.clock)
        message.receive_event = self.cluster.record(self, "receive",
                                                   f"{message.label}<-{message.sender}", clock)
        return message.receive_event


class Cluster:
    """Deterministic run: FIFO delivery of HLC-stamped messages, plus the ground-truth vectors."""

    def __init__(self, starts):
        self.nodes = {name: Node(name, self, start) for name, start in starts.items()}
        self.queue = []
        self.events = []
        self.delivered = []
        self.delivery_order = []
        self._seq = 0

    def next_seq(self):
        self._seq += 1
        return self._seq

    def record(self, node, kind, label, clock):
        event = Event(self.next_seq(), node.name, kind, label, clock, node.vector.copy(),
                      node.physical)
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

    def deliver_all(self):
        while self.queue:
            message = self.queue.pop(0)
            self.nodes[message.receiver].receive(message)
            self.delivered.append(message)
        return self

    def run(self, script):
        for step in script:
            if step[0] == "local":
                self.local(step[1], step[2])
            elif step[0] == "send":
                self.send(step[1], step[2], step[3])
            else:
                self.deliver_all()
        return self.deliver_all()

    def causal_past(self, event):
        """Every event whose vector clock is strictly below this one (the hb closure, via the VC)."""
        return [other for other in self.events if other.vector.lt(event.vector)]


def hb_closure(events, messages):
    """Independent oracle: transitive closure of program order plus send -> receive edges."""
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


def build_script():
    """A fixed script with local events, sends and receives, including same-millisecond batches."""
    script = []
    for node in ("A", "B", "C"):
        for index in range(3):
            script.append(("local", node, f"l{index}"))
        script.append(("send", node, "B" if node != "B" else "C", f"m{node}"))
    script.append(("deliver",))
    for node in ("A", "B", "C"):
        script.append(("local", node, "after"))
    script.append(("send", "B", "A", "reply"))
    script.append(("deliver",))
    script.append(("local", "A", "post"))
    script.append(("deliver",))
    return script


def main():
    # --- 1. the clock rules, hand-checked -------------------------------------------------------
    clock = HLC("A")
    check(clock.stamp() == (0, 0), "a fresh clock reads (0, 0)")
    check(clock.local(10) == (10, 0), "the first local event takes the physical time with c = 0")
    check(clock.local(10) == (10, 1), "a second event in the same millisecond bumps c")
    check(clock.local(10) == (10, 2), "and again")
    check(clock.local(12) == (12, 0), "a later physical time resets the counter")
    check(clock.local(11) == (12, 1),
          "a stale physical clock cannot move the logical component backwards")
    check(clock.receive(30, (40, 7)) == (40, 8),
          "a message whose l dominates takes cm + 1: the receive follows the send immediately")
    check(clock.receive(41, (40, 7)) == (41, 0),
          "a physical time beyond both resets the counter to 0")
    check(clock.receive(41, (41, 7)) == (41, 8),
          "equal logical components take max(c, cm) + 1")
    clock = HLC("B")
    clock.local(100)
    check(clock.receive(50, (100, 0)) == (100, 1),
          "a receive keeps the local l when it is the largest and bumps c")
    clock = HLC("C")
    check(clock.receive(5, (0, 0)) == (5, 0), "a fresh clock with a fast physical time uses it")

    # --- 2. wire format round-trip and ordering --------------------------------------------------
    stamps = [(0, 0), (1, 0), (1, 99), (2, 0), (10, 5), (1234567890, 123)]
    check([parse_wire(wire_format(stamp)) for stamp in stamps] == stamps,
          "the wire format round-trips every stamp")
    check(sorted(stamps, key=wire_format) == sorted(stamps),
          "a byte comparison of the fixed-width form follows the (l, c) order")
    check(sorted(stamps, key=wire_format) == sorted(stamps, key=lambda stamp: stamp),
          "and agrees with the tuple order")

    # --- 3. a skewed cluster: causality, drift and ordering -------------------------------------
    starts = {"A": 1000, "B": 0, "C": 500}       # A's physical clock is far ahead of B's
    cluster = Cluster(starts)
    cluster.run(build_script())
    check(len(cluster.events) > 20, "the script produced a real run")
    check(all(event.vector.key() for event in cluster.events), "every event carries a vector clock")

    reachable = hb_closure(cluster.events, cluster.delivered)
    mismatches = [pair for pair in
                  [(first, second) for first in cluster.events for second in cluster.events
                   if first.seq != second.seq]
                  if (pair[1].seq in reachable[pair[0].seq]) != pair[0].vector.lt(pair[1].vector)]
    check(not mismatches, f"vector clocks agree with the closure oracle: {mismatches[:2]}")

    for event in cluster.events:                 # monotone per node
        earlier = [other for other in cluster.events
                   if other.node == event.node and other.seq < event.seq]
        check(all(earlier_event.clock < event.clock for earlier_event in earlier),
              f"a node's stamps strictly increase: {event}")

    for event in cluster.events:
        past = [other.physical for other in cluster.causal_past(event)] + [event.physical]
        check(event.clock[0] == max(past),
              f"l must be the largest physical time in the causal past of {event}: "
              f"{event.clock[0]} != {max(past)}")

    for first in cluster.events:                 # happened-before implies a smaller stamp
        for second in cluster.events:
            if first.vector.lt(second.vector):
                check(first.clock < second.clock,
                      f"causality must imply stamp order: {first} -> {second}")

    sorted_events = sorted(cluster.events, key=lambda event: event.clock)
    position = {event.seq: index for index, event in enumerate(sorted_events)}
    inversions = [(first, second) for first in cluster.events for second in cluster.events
                  if first.vector.lt(second.vector) and position[first.seq] > position[second.seq]]
    check(not inversions,
          f"the stamp order is a linear extension of happened-before: {inversions[:2]}")

    per_node = {}
    for event in cluster.events:
        per_node.setdefault(event.node, []).append(event.clock)
    check(all(len(stamps) == len(set(stamps)) for stamps in per_node.values()),
          "no node ever stamps two events with the same (l, c)")

    # --- 4. the negative control: physical time alone gets causality wrong ----------------------
    naive_violations = []
    for first in cluster.events:
        for second in cluster.events:
            if first.vector.lt(second.vector) and first.physical > second.physical:
                naive_violations.append((first, second))
    check(naive_violations,
          "with a skewed clock, ordering by physical time violates causality somewhere")
    check(len(naive_violations) > 0 and not inversions,
          "the hybrid clock fixes exactly what the naive ordering gets wrong")

    # --- 5. skew propagation, same-millisecond bursts and determinism ----------------------------
    fast = Cluster({"A": 5000, "B": 0})
    fast.local("B", "b0")
    before = fast.nodes["B"].hlc.stamp()
    fast.send("A", "B", "future")
    sender_stamp = fast.nodes["A"].hlc.stamp()
    fast.nodes["B"].advance()
    fast.nodes["B"].advance()
    fast.deliver_all()
    after = fast.nodes["B"].hlc.stamp()
    check(before < after and after == (sender_stamp[0], sender_stamp[1] + 1),
          f"the receiver's stamp becomes exactly the sender's l with the next counter: {after}")
    fast.local("B", "b1")
    check(fast.nodes["B"].hlc.stamp() == (sender_stamp[0], sender_stamp[1] + 2),
          "and the receiver keeps that l for its next local event, bumping c")
    check(fast.nodes["B"].physical < fast.nodes["B"].hlc.l,
          "so the receiver's HLC is ahead of its own physical clock - without inventing future time")

    burst = Cluster({"A": 7})
    stamps = []
    for index in range(5):
        burst.local("A", f"s{index}")
        stamps.append(burst.nodes["A"].hlc.stamp())
    check(stamps == [(7, 0), (7, 1), (7, 2), (7, 3), (7, 4)],
          "five events in the same millisecond get five consecutive counters")

    twin = Cluster(starts)
    twin.run(build_script())
    check([event.clock for event in twin.events] == [event.clock for event in cluster.events],
          "the same script reproduces the same stamps")
    check([event.vector.key() for event in twin.events]
          == [event.vector.key() for event in cluster.events], "and the same vector clocks")

    # --- 6. boundaries --------------------------------------------------------------------------
    single = Cluster({"S": 3})
    first_stamp = single.local("S", "only")
    check(first_stamp.clock == (3, 0) and len(single.events) == 1,
          "a single event on a single node is (physical, 0)")
    lonely = Cluster({"X": 0, "Y": 0})
    lonely.local("X", "x0")
    lonely.local("Y", "y0")
    check(lonely.nodes["X"].hlc.stamp() == lonely.nodes["Y"].hlc.stamp() == (0, 1),
          "concurrent events on different nodes may share the same stamp")
    lonely.send("X", "Y", "m")
    lonely.deliver_all()
    check(lonely.nodes["Y"].hlc.stamp() > (0, 1),
          "the shared stamp is broken by the message that follows it")
    empty = Cluster({"Z": 0})
    check(empty.nodes["Z"].hlc.stamp() == (0, 0) and empty.events == [],
          "a node with no events keeps its initial stamp")

    print(f"hlc: events={len(cluster.events)} messages={len(cluster.delivered)} "
          f"naive-order-violations={len(naive_violations)} hlc-order-violations={len(inversions)} "
          f"max-l={max(event.clock[0] for event in cluster.events)}")


if __name__ == "__main__":
    main()
    print("PASS")
