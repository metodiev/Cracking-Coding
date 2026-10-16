"""Task 616 - Implement distributed snapshot algorithm (Chandy-Lamport).  (Tier: Expert, Ch 9 — System Design and Scalability)

The Chandy-Lamport (1985) global snapshot, simulated over an explicit event schedule: processes,
directed FIFO channels and messages are plain data structures advanced by a list of
(process, action) entries, so there are no threads, sockets, sleeps or timing dependencies.

The algorithm as implemented - markers travel *in band*, through the same FIFO channels as the
data messages, which is what makes the cut consistent.  Tokens *move*: a send takes the amount
out of the sender and puts it into the channel, a receive puts it into the receiver, so the grand
total of process states plus in-flight messages is invariant and the recorded snapshot must
reproduce exactly that number:
  * the initiator records its own state, starts recording state on every incoming channel and
    then pushes one marker into every outgoing channel;
  * a process receiving a marker for the first time records its own state, records the delivering
    channel as empty (nothing sent before the sender's cut can still be behind the marker),
    starts recording every other incoming channel and forwards markers on all outgoing channels;
  * a process that is already recording closes the incoming channel that delivered the marker;
    data messages received on a still-recorded channel are appended to that channel's recorded
    state - they were sent before the sender's cut but delivered after this process's cut;
  * the snapshot is every process state plus the recorded state of every channel.

Verified here:
  * conservation - recorded process states plus recorded channel states sum to the seeded total;
  * cut consistency recomputed independently from the event logs: for each channel, the messages
    received after the receiver's cut must be a subset of the messages sent before the sender's
    cut (nothing is received before it is sent in the cut), and the recorded channel state must
    be exactly that intersection in FIFO order;
  * completeness - after the marker wave settles, every process in a strongly connected graph has
    recorded and closed every incoming channel, and this holds for a snapshot started at *each*
    process in turn, with the total conserved every time;
  * negative controls - a snapshot that freezes only the process states loses the message that
    is in flight at the cut (it is not a consistent cut), and in a graph where the initiator has
    no outgoing path to a process that process never joins the snapshot.
"""

MARKER = object()                 # travels inside the FIFO channels, exactly like a real marker


def all_links(names):
    return [(src, dst) for src in names for dst in names if src != dst]


class Process:
    def __init__(self, name):
        self.name = name
        self.tokens = 0
        self.recorded = None      # the Snapshot record once the marker wave reaches this process
        self.log = []             # (sequence, kind, detail) for the independent cut check


class Snapshot:
    """One process's part of the global snapshot: its state and its recorded channels."""

    def __init__(self, state, cut_sequence):
        self.state = state
        self.cut_sequence = cut_sequence
        self.channels = {}        # incoming source name -> [recorded message ids]
        self.closed = set()       # incoming channels already closed by their marker


class Simulator:
    def __init__(self, names, links=None):
        self.names = list(names)
        self.processes = {name: Process(name) for name in names}
        self.channels = {(src, dst): [] for src, dst in (links or all_links(names))}
        self.sequence = 0
        self.message_id = 0
        self.total = 0            # the conserved quantity: tokens held by processes

    # -- primitive operations ---------------------------------------------
    def _log(self, name, kind, detail):
        self.sequence += 1
        self.processes[name].log.append((self.sequence, kind, detail))
        return self.sequence

    def seed(self, name, tokens):
        self.processes[name].tokens += tokens
        self.total += tokens
        self._log(name, "internal", {"seeded": tokens})

    def internal(self, name, delta):
        self.processes[name].tokens += delta
        self.total += delta
        self._log(name, "internal", {"delta": delta})

    def send(self, src, dst, amount):
        """Move tokens out of src into an in-flight message on the FIFO channel src -> dst."""
        if (src, dst) not in self.channels:
            raise AssertionError("no channel %s -> %s" % (src, dst))
        if self.processes[src].tokens < amount:
            raise AssertionError("%s cannot send %d tokens" % (src, amount))
        self.message_id += 1
        self.processes[src].tokens -= amount
        self.channels[(src, dst)].append((self.message_id, amount))
        self._log(src, "send", {"dst": dst, "message": self.message_id, "amount": amount})

    def receive(self, name, src):
        if (src, name) not in self.channels:
            raise AssertionError("no channel %s -> %s" % (src, name))
        channel = self.channels[(src, name)]
        if not channel:
            raise AssertionError("channel %s -> %s is empty" % (src, name))
        message = channel.pop(0)
        process = self.processes[name]
        if message is MARKER:
            self._deliver_marker(name, src)
            return MARKER
        message_id, amount = message
        process.tokens += amount
        sequence = self._log(name, "recv", {"src": src, "message": message_id,
                                            "amount": amount})
        record = process.recorded
        if record is not None and src in record.channels and src not in record.closed:
            record.channels[src].append(message_id)
        return sequence

    def drain(self, limit=400):
        """Deliver every message, in a fixed channel order, until all channels are empty."""
        for _ in range(limit):
            progressed = False
            for (src, dst) in sorted(self.channels):
                if self.channels[(src, dst)]:
                    self.receive(dst, src)
                    progressed = True
            if not progressed:
                return
        raise AssertionError("the channel network did not settle")

    # -- the snapshot algorithm -------------------------------------------
    def start_snapshot(self, initiator):
        process = self.processes[initiator]
        if process.recorded is not None:
            raise AssertionError("a snapshot is already in flight")
        sequence = self._log(initiator, "snapshot", {"role": "initiator"})
        record = Snapshot({"tokens": process.tokens}, sequence)
        for (src, dst) in self.channels:
            if dst == initiator:
                record.channels[src] = []
        process.recorded = record
        for (src, dst) in sorted(self.channels):
            if src == initiator:
                self.channels[(src, dst)].append(MARKER)

    def _deliver_marker(self, name, src):
        process = self.processes[name]
        record = process.recorded
        if record is None:                                  # joining: first marker seen
            sequence = self._log(name, "snapshot", {"role": "joined", "via": src})
            record = Snapshot({"tokens": process.tokens}, sequence)
            record.channels[src] = []                       # the delivering channel is empty
            record.closed.add(src)
            for (a, b) in self.channels:
                if b == name and a != src:
                    record.channels[a] = []
            process.recorded = record
            for (a, b) in sorted(self.channels):
                if a == name:
                    self.channels[(a, b)].append(MARKER)
        else:                                               # already recording: close it
            record.closed.add(src)

    def records(self):
        return {name: process.recorded for name, process in self.processes.items()
                if process.recorded is not None}

    def is_complete(self, records):
        expected_incoming = len(self.names) - 1
        return (len(records) == len(self.names)
                and all(len(record.closed) == expected_incoming
                        for record in records.values()))

    def frozen(self, records):
        """Materialise the snapshot as (process states, channel states)."""
        states = {name: dict(record.state) for name, record in records.items()}
        channels = {(src, name): list(record.channels.get(src, []))
                    for name, record in records.items()
                    for src in record.channels}
        return states, channels

    def snapshot_total(self, records):
        states, channels = self.frozen(records)
        total = sum(state["tokens"] for state in states.values())
        for (src, dst), message_ids in channels.items():
            total += sum(self.amount_of(message_id, src) for message_id in message_ids)
        return total

    def amount_of(self, message_id, src):
        for _, kind, detail in self.processes[src].log:
            if kind == "send" and detail["message"] == message_id:
                return detail["amount"]
        raise AssertionError("unknown message id %d" % message_id)

    # -- independent cut verification --------------------------------------
    def verify_cut(self, records):
        """Recompute every recorded channel from the logs and check the cut property.

        For channel src -> dst: let S be the messages src sent at or before its cut.  A message
        that dst received at or before its own cut must be in S (the cut never shows a receive
        without its matching send), and the recorded channel state must be exactly the messages
        of S that dst received *after* its cut, in send order."""
        problems = []
        for dst, record in records.items():
            for src, message_ids in record.channels.items():
                sent_before_cut = [detail["message"]
                                   for sequence, kind, detail in self.processes[src].log
                                   if kind == "send" and detail["dst"] == dst
                                   and sequence <= records[src].cut_sequence]
                received_before_cut = {detail["message"]
                                       for sequence, kind, detail
                                       in self.processes[dst].log
                                       if kind == "recv" and detail["src"] == src
                                       and sequence <= record.cut_sequence}
                received_after_cut = {detail["message"]
                                      for sequence, kind, detail
                                      in self.processes[dst].log
                                      if kind == "recv" and detail["src"] == src
                                      and sequence > record.cut_sequence}
                if not received_before_cut <= set(sent_before_cut):
                    problems.append("channel %s->%s shows a receive inside the cut whose "
                                    "send is outside it" % (src, dst))
                expected = [message for message in sent_before_cut
                            if message in received_after_cut]
                if expected != list(message_ids):
                    problems.append("channel %s->%s recorded %s but the cut says %s"
                                    % (src, dst, list(message_ids), expected))
        return problems


def build(names, program, seed_tokens, links=None):
    simulator = Simulator(names, links)
    for name, tokens in seed_tokens.items():
        simulator.seed(name, tokens)
    for entry in program:
        kind, args = entry[0], entry[1:]
        if kind == "send":
            simulator.send(args[0], args[1], args[2])
        elif kind == "recv":
            simulator.receive(args[0], args[1])
        elif kind == "internal":
            simulator.internal(args[0], args[1])
        elif kind == "snapshot":
            simulator.start_snapshot(args[0])
        else:
            raise AssertionError("unknown action %r" % kind)
    return simulator


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    names = ["p0", "p1", "p2", "p3"]
    seed = {"p0": 10, "p1": 20, "p2": 5, "p3": 0}

    program = [
        ("send", "p1", "p2", 7),            # in flight while the snapshot starts
        ("send", "p2", "p0", 3),
        ("snapshot", "p0"),
        ("send", "p0", "p1", 4),            # sent after p0's cut: must be channel state
        ("recv", "p1", "p0"),               # p1 receives the marker and joins; 4 is behind it
        ("recv", "p2", "p1"),               # p2 consumes 7 before joining
        ("recv", "p0", "p2"),               # p0 receives 3 after its own cut
        ("recv", "p3", "p0"),               # p3 joins through p0's marker
    ]
    simulator = build(names, program, seed)
    simulator.drain()                       # let the marker wave finish
    records = simulator.records()
    check(simulator.is_complete(records), "every process must join and close every channel")
    main_records = records
    main_total = simulator.snapshot_total(records)
    check(main_total == simulator.total == 35, "the snapshot must conserve the tokens")
    check(simulator.verify_cut(records) == [], "independent cut verification")
    states, channels = simulator.frozen(main_records)
    check(sorted(states) == names, "all four processes recorded a state")
    for name, state in states.items():
        check(isinstance(state["tokens"], int), "states are plain values")
    held = {key: value for key, value in channels.items() if value}
    check(held, "some channel must hold an in-flight message")
    check(held.get(("p2", "p0")), "the message in flight from p2 to p0 at p0's cut is "
                                  "recorded as channel state rather than process state")
    check(all(message_id > 0 for values in channels.values() for message_id in values),
          "recorded channel entries are message ids")

    # ---- a snapshot started at each process in turn is consistent too ----
    spread = {"p0": 12, "p1": 12, "p2": 12, "p3": 12}
    for initiator in names:
        program = []
        for step in range(24):
            src = names[step % 4]
            dst = names[(step * 3 + 1) % 4]
            program.append(("send", src, dst, 1))
            if step % 5 == 4:
                program.append(("recv", dst, src))
        program.insert(len(program) // 2, ("snapshot", initiator))
        run = build(names, program, spread)
        run.drain()
        run_records = run.records()
        check(run.is_complete(run_records), "initiator %s left channels open" % initiator)
        check(run.snapshot_total(run_records) == run.total,
              "initiator %s produced an inconsistent total" % initiator)
        check(run.verify_cut(run_records) == [],
              "initiator %s cut problem: %s" % (initiator, run.verify_cut(run_records)))

    # ---- negative control: a process-states-only freeze is not a consistent cut --
    naive = Simulator(["a", "b"])
    naive.seed("a", 10)
    naive.send("a", "b", 5)                                  # 5 tokens are in flight
    naive_states_only = (naive.processes["a"].tokens + naive.processes["b"].tokens)
    check(naive_states_only == 5,
          "freezing only the process states silently loses the in-flight message")
    check(naive.total == 10, "the true total has to count messages in flight")
    check(naive_states_only != naive.total, "a states-only snapshot is inconsistent")

    proper = Simulator(["a", "b", "c"])
    proper.seed("a", 10)
    proper.seed("b", 0)
    proper.seed("c", 10)
    proper.send("c", "b", 4)                                 # in flight when a starts
    proper.start_snapshot("a")
    proper.receive("b", "a")                                 # b's cut is here
    proper.receive("c", "a")                                 # c's cut is after its send
    proper.receive("b", "c")                                 # the in-flight 4 arrives late
    proper.drain()                                           # finish the marker wave
    proper_records = proper.records()
    check(proper.is_complete(proper_records), "the three-process snapshot completes")
    check(proper.snapshot_total(proper_records) == 20 == proper.total,
          "Chandy-Lamport conserves the tokens including the in-flight one")
    states, channels = proper.frozen(proper_records)
    check(channels[("c", "b")], "the in-flight message is channel state, not process state")
    check(states["b"]["tokens"] == 0 and states["c"]["tokens"] == 6,
          "the process states are the ones frozen at their own cuts")
    check(proper.verify_cut(proper_records) == [], "the three-process cut is consistent")

    # ---- a process with no incoming path never joins the snapshot ------------
    weak = Simulator(["root", "island"], links=[("island", "root")])
    weak.seed("root", 3)
    weak.seed("island", 4)
    weak.start_snapshot("root")
    weak.drain()
    weak_records = weak.records()
    check(not weak.is_complete(weak_records),
          "a process the initiator cannot reach must never join")
    check(list(weak_records) == ["root"], "only the initiator recorded a state")
    check(weak.snapshot_total(weak_records) == 3,
          "the partial snapshot only covers the reachable part")

    # ---- rejected inputs ------------------------------------------------------
    broken = Simulator(["x", "y"], links=[("x", "y")])
    for thunk in (lambda: broken.receive("y", "x"), lambda: broken.send("y", "x", 1)):
        raised = False
        try:
            thunk()
        except AssertionError:
            raised = True
        check(raised, "an operation on a missing or empty channel must raise")
    once = Simulator(["x", "y"], links=[("x", "y")])
    once.start_snapshot("x")
    raised = False
    try:
        once.start_snapshot("x")
    except AssertionError:
        raised = True
    check(raised, "a second concurrent snapshot must be rejected")

    print("4-process snapshot: %d tokens conserved, %d channels recorded, cut verified"
          % (main_total, len(held)))
    print("snapshot started at each of the four processes: consistent every time")
    print("negative control: a states-only freeze reported %d tokens instead of %d"
          % (naive_states_only, naive.total))
    print("PASS")


if __name__ == "__main__":
    main()
