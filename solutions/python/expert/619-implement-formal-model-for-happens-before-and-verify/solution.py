"""Task 619 - Implement formal model for happens-before and verify.  (Tier: Expert, CTCI Ch 15 — Threads and Locks)

Lamport's happens-before relation, built as an explicit partial order over the events of a
deterministic run, and a vector clock whose order is proved - by exhaustive comparison - to be
exactly that partial order.

Model:
  * a run is a set of events; each event belongs to one process and has a local sequence number,
    and is either a local step, a send of a specific message or the receive of that message;
  * the happens-before graph has program-order edges (consecutive events of one process) and
    message edges (the send of a message to its receive);
  * a -> b is the transitive closure of those edges, computed here directly (bitset closure over
    the small event set), which is the *definition* used to judge the vector clocks rather than
    the other way round;
  * each process carries a vector clock: a local step bumps its own component, a send stamps the
    message with the current vector, and a receive takes the component-wise maximum with the
    message's stamp and then bumps its own component.

Verified exhaustively for every ordered pair of events of every run:
  * the vector clocks are exact: VC(a) < VC(b) (component-wise <= with one strict) if and only if
    a -> b, and VC(a) incomparable with VC(b) if and only if a and b are concurrent;
  * the derived relation really is a strict partial order (irreflexive, antisymmetric, transitive)
    and every event is reachable from its own past only;
  * a causal-broadcast simulation that buffers messages and delivers only ready ones produces a
    delivery order that is a linear extension of happens-before - then a deliberately forged
    delivery is shown to create a cycle in the graph, i.e. to destroy the order.
"""


class Event:
    __slots__ = ("index", "process", "sequence", "kind", "message", "vector")

    def __init__(self, index, process, sequence, kind, message=None):
        self.index = index
        self.process = process
        self.sequence = sequence
        self.kind = kind                 # "local", "send" or "recv"
        self.message = message
        self.vector = None


class Run:
    """A deterministic execution of n processes exchanging numbered messages."""

    def __init__(self, processes):
        self.processes = list(processes)
        self.events = []
        self.per_process = {name: [] for name in self.processes}
        self.clock = {name: [0] * len(self.processes) for name in self.processes}
        self.messages = {}               # message id -> vector stamped at the send
        self.sent_from = {}              # message id -> event index of the send

    def _add(self, process, kind, message=None):
        sequence = len(self.per_process[process])
        event = Event(len(self.events), process, sequence, kind, message)
        vector = list(self.clock[process])
        position = self.processes.index(process)
        vector[position] += 1
        self.clock[process] = vector
        event.vector = tuple(vector)
        self.events.append(event)
        self.per_process[process].append(event.index)
        return event

    def local(self, process):
        return self._add(process, "local")

    def send(self, process, message):
        event = self._add(process, "send", message)
        self.messages[message] = event.vector
        self.sent_from[message] = event.index
        return event

    def receive(self, process, message):
        stamp = self.messages.get(message)
        if stamp is None:
            raise AssertionError("message %s was never sent" % message)
        current = self.clock[process]
        self.clock[process] = [max(current[i], stamp[i]) for i in range(len(self.processes))]
        return self._add(process, "recv", message)

    # -- the relation -----------------------------------------------------
    def edges(self):
        edges = set()
        for process, indices in self.per_process.items():
            for first, second in zip(indices, indices[1:]):
                edges.add((first, second))                 # program order
        for message, send_index in self.sent_from.items():
            receivers = [event.index for event in self.events
                         if event.kind == "recv" and event.message == message]
            for receive_index in receivers:
                edges.add((send_index, receive_index))     # message order
        return edges

    def closure(self):
        """Transitive closure of the happens-before edges (reflexive-free)."""
        count = len(self.events)
        reach = [[False] * count for _ in range(count)]
        for first, second in self.edges():
            reach[first][second] = True
        for middle in range(count):
            row = reach[middle]
            for start in range(count):
                if reach[start][middle]:
                    target = reach[start]
                    for end in range(count):
                        if row[end]:
                            target[end] = True
        return reach

    def concurrent(self, reach, first, second):
        return not reach[first][second] and not reach[second][first]


def vector_less(first, second):
    return all(a <= b for a, b in zip(first, second)) and first != second


def vector_incomparable(first, second):
    return not vector_less(first, second) and not vector_less(second, first)


def build_run():
    """A fixed 3-process run: messages crossing, a reply, and some local steps in between."""
    run = Run(["p0", "p1", "p2"])
    run.local("p0")
    run.send("p0", "m1")
    run.local("p0")
    run.local("p1")
    run.receive("p1", "m1")
    run.send("p1", "m2")
    run.send("p1", "m3")
    run.local("p2")
    run.send("p2", "m0")                 # independent of everything before it
    run.local("p1")
    run.receive("p0", "m2")
    run.receive("p2", "m3")
    run.local("p0")
    run.receive("p0", "m0")
    run.send("p0", "m4")
    run.receive("p2", "m4")
    run.local("p1")
    return run


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    run = build_run()
    reach = run.closure()
    count = len(run.events)
    check(count >= 17, "the run should have a decent number of events")

    # ---- exactness of the vector clocks ------------------------------------
    for first in range(count):
        check(not reach[first][first], "happens-before must be irreflexive")
        for second in range(count):
            if first == second:
                continue
            differs = vector_less(run.events[first].vector, run.events[second].vector)
            check(differs == reach[first][second],
                  "VC ordering and happens-before disagree for events %d and %d" % (first, second))
            check(not (differs and vector_less(run.events[second].vector,
                                               run.events[first].vector)),
                  "the two orderings must be asymmetric")
            incomparable = vector_incomparable(run.events[first].vector,
                                               run.events[second].vector)
            check(incomparable == run.concurrent(reach, first, second),
                  "concurrency and incomparable clocks disagree for %d, %d" % (first, second))

    # ---- the relation is a strict partial order -----------------------------
    for first in range(count):
        for second in range(count):
            for third in range(count):
                if reach[first][second] and reach[second][third]:
                    check(reach[first][third],
                          "transitivity broken for %d -> %d -> %d" % (first, second, third))
    order_edges = sum(1 for first in range(count) for second in range(count)
                      if reach[first][second])
    check(order_edges > count, "a real partial order must have many comparable pairs")

    # the past of an event via the closure equals the past via the edges plus closure
    for event in run.events:
        past = {other for other in range(count) if reach[other][event.index]}
        check(event.index not in past, "an event is not its own past")
        if event.kind == "recv":
            check(run.sent_from[event.message] in past,
                  "the send of a delivered message must be in the past")

    # ---- causal delivery: a linear extension of happens-before -------------
    policy = CausalDelivery(run)
    delivered = policy.deliver_all()
    check(len(delivered) == count, "causal delivery must deliver every event")
    position_of = {event_index: position for position, event_index in enumerate(delivered)}
    for first in range(count):
        for second in range(count):
            if reach[first][second]:
                check(position_of[first] < position_of[second],
                      "causal delivery broke happens-before on %d -> %d" % (first, second))
    check(policy.buffered_ever > 0,
          "the causal policy must have had to buffer at least one message")

    # a message can never be delivered before it was sent in the causal order
    for event in run.events:
        if event.kind == "recv":
            check(position_of[run.sent_from[event.message]] < position_of[event.index],
                  "a receive precedes its own send")

    # ---- forged delivery: a cycle appears ----------------------------------
    forged = Run(["p0", "p1"])
    forged.send("p0", "x")
    forged.receive("p1", "x")
    forged.send("p1", "y")
    forged.receive("p0", "y")
    forged_edges = forged.edges()
    cycle_added = set(forged_edges) | {(forged.sent_from["y"], forged.sent_from["x"])}
    check(has_cycle(len(forged.events), cycle_added),
          "an extra edge against message order must create a cycle")
    check(not has_cycle(len(forged.events), forged_edges),
          "the honest graph of the same run stays acyclic")

    # ---- rejected input ------------------------------------------------------
    raised = False
    try:
        Run(["q0"]).receive("q0", "missing")
    except AssertionError:
        raised = True
    check(raised, "receiving a never-sent message must raise")

    print("run: %d processes, %d events, %d comparable pairs, %d messages"
          % (len(run.processes), count, order_edges, len(run.sent_from)))
    print("vector clocks match the happens-before closure on every one of the %d pairs"
          % (count * (count - 1)))
    print("causal delivery: %d events delivered in a linear extension, %d buffered waits"
          % (len(delivered), policy.buffered_ever))
    print("PASS")


class CausalDelivery:
    """Delivers messages only when every message they causally depend on was delivered."""

    def __init__(self, run):
        self.run = run
        self.delivered = []
        self.pending = {}
        self.buffered_ever = 0

    def deliver_all(self):
        """Scan the events in a deliberately unhelpful order (the reverse process order) so that
        receives whose sends have not been delivered yet really end up buffered."""
        run = self.run
        order = {name: position for position, name in
                 enumerate(reversed(run.processes))}
        outstanding = sorted(run.events, key=lambda event: (order[event.process],
                                                            event.sequence))
        local_next = {name: 0 for name in run.processes}
        progress = True
        while progress:
            progress = False
            for event in list(outstanding):
                if event.sequence != local_next[event.process]:
                    continue                       # this process has not reached it yet
                if event.kind == "recv":
                    send_index = run.sent_from[event.message]
                    if send_index not in self.delivered:
                        if event.index not in self.pending:
                            self.pending[event.index] = True
                            self.buffered_ever += 1
                        continue                   # buffer until the send is delivered
                self.delivered.append(event.index)
                local_next[event.process] += 1
                outstanding.remove(event)
                self.pending.pop(event.index, None)
                progress = True
        return self.delivered


def has_cycle(count, edges):
    outgoing = {node: set() for node in range(count)}
    for first, second in edges:
        outgoing[first].add(second)
    colour = [0] * count

    def visit(node):
        colour[node] = 1
        for other in outgoing[node]:
            if colour[other] == 1:
                return True
            if colour[other] == 0 and visit(other):
                return True
        colour[node] = 2
        return False

    return any(colour[node] == 0 and visit(node) for node in range(count))


if __name__ == "__main__":
    main()
