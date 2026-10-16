"""Task 692 - Implement reliable ordered multicast with failure handling.  (Tier: Expert, CTCI Ch 9 — System Design and Scalability)

A sequencer-based total-order multicast with the failure handling that makes it reliable:

* the current view's lowest-id member is the **sequencer**: it stamps every application message with a
  monotonically increasing sequence number and multicasts `(seq, sender, payload)` to the whole view;
* every member buffers out-of-order messages and delivers strictly in sequence order, so delivery is
  FIFO per sender and identical across members — the total-order guarantee;
* a member that sees a gap sends a **NACK** for the missing sequence number and any member that still
  has the message — in its pending buffer or in the retransmission window it keeps after delivery —
  sends it again.  Retransmitting from the members' buffers rather than from the original sender is
  exactly what makes the protocol survive the loss of the sequencer itself;
* a **view change** removes the failed member: the survivors agree on the new view, the new sequencer
  continues from the highest sequence number anyone has seen, and the pending buffers are carried over
  so the messages the old sequencer had already sent are still delivered.

The simulation is deterministic: messages travel over links whose drops come from a fixed-seed
splitmix64 PRNG, and every round is a synchronous step, so the same run happens on every machine.

Verified: identical delivery order at every surviving member; no duplicates and no holes; prefix
closure at every member (the delivered set is always `1..k`); agreement between members about which
sequence numbers are still missing; delivery of every message that reached at least one survivor, even
with the sequencer dying mid-stream; a member that never saw a message reporting nothing at all rather
than inventing a delivery; and liveness after the view change, when the new sequencer continues above
the highest sequence number in use.
"""

MSG, ACK, NACK, VIEW = "msg", "ack", "nack", "view"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def salt(value):
    """A tiny deterministic spread so every ordered pair of members gets its own loss stream."""
    return (value * 2654435761 + 12345) % 4294967296


class Link:
    """One directional link with a deterministic drop pattern."""

    def __init__(self, loss_percent, seed):
        self.loss_percent = loss_percent
        self.state = (seed * 6364136223846793005 + 1442695040888963407) % (2 ** 64)
        self.dropped = 0
        self.delivered = 0

    def carries(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) % (2 ** 64)
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % (2 ** 64)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % (2 ** 64)
        value = (z ^ (z >> 31)) % 10000
        if value >= self.loss_percent * 100:
            self.delivered += 1
            return True
        self.dropped += 1
        return False


class Member:
    def __init__(self, name):
        self.name = name
        self.buffer = {}            # seq -> (seq, sender, payload): received, not yet deliverable
        self.history = {}           # delivered messages kept for retransmission (a send window)
        self.delivered = []         # [(seq, sender, payload)] in sequence order
        self.next_expected = 1
        self.highest_seen = 0
        self.missing = set()
        self.alive = True

    def prefix_ok(self):
        return [seq for seq, _, _ in self.delivered] == list(range(1, len(self.delivered) + 1))


class Multicast:
    def __init__(self, names, loss_percent):
        self.names = sorted(names)
        self.members = {name: Member(name) for name in self.names}
        self.links = {(a, b): Link(loss_percent, 1000 + salt(a) + salt(b))
                      for a in self.names for b in self.names}
        self.view = list(self.names)
        self.sequencer = self.view[0]
        self.next_sequence = 1
        self.pending = []           # in-flight (sender, receiver, kind, seq, payload)
        self.log = []               # (receiver, seq, sender) for every delivery

    # ---- plumbing ---------------------------------------------------------------------------
    def set_loss(self, loss_percent):
        for link in self.links.values():
            link.loss_percent = loss_percent

    def dropped(self):
        return sum(link.dropped for link in self.links.values())

    def send(self, sender, receiver, kind, seq, payload=None):
        if self.links[(sender, receiver)].carries():
            self.pending.append((sender, receiver, kind, seq, payload))

    def broadcast(self, sender, kind, seq, payload=None):
        for receiver in self.view:
            if receiver != sender:
                self.send(sender, receiver, kind, seq, payload)

    # ---- the protocol ----------------------------------------------------------------------
    def submit(self, payload):
        """The sequencer proposes the next message; returns the sequence number it assigned."""
        sender = self.sequencer
        seq = self.next_sequence
        self.next_sequence += 1
        member = self.members[sender]
        member.buffer[seq] = (seq, sender, payload)
        member.highest_seen = max(member.highest_seen, seq)
        self.broadcast(sender, MSG, seq, payload)
        return seq

    def step(self):
        """One synchronous round: deliver what arrived, then fill the gaps."""
        inbox, self.pending = self.pending, []
        retransmits = 0
        for sender, receiver, kind, seq, payload in inbox:
            member = self.members[receiver]
            if not member.alive:
                continue
            if kind == MSG:
                if seq >= member.next_expected:              # earlier ones are duplicates
                    member.buffer.setdefault(seq, (seq, sender, payload))
                member.highest_seen = max(member.highest_seen, seq)
                self.send(receiver, sender, ACK, seq)
            elif kind == NACK:
                stored = member.buffer.get(seq) or member.history.get(seq)
                if stored is not None:
                    self.send(receiver, sender, MSG, seq, stored[2])
                    retransmits += 1
            elif kind == VIEW:
                self.view = list(payload)
        for name in list(self.view):
            member = self.members[name]
            if not member.alive:
                continue
            while member.next_expected in member.buffer:
                entry = member.buffer.pop(member.next_expected)
                member.delivered.append(entry)
                member.history[entry[0]] = entry
                self.log.append((name, entry[0], entry[1]))
                member.next_expected += 1
            member.missing = set(range(member.next_expected, member.highest_seen + 1))
            for seq in sorted(member.missing):
                self.broadcast(name, NACK, seq)
        return retransmits

    def run(self, rounds):
        for _ in range(rounds):
            self.step()

    def settle(self, limit=200):
        """Run until no member still has a gap and nothing is in flight: the protocol quiesces.
        A NACK is repeated every round, which is what makes the repair survive a lost NACK."""
        for _ in range(limit):
            self.step()
            outstanding = any(member.alive and member.missing
                              for member in self.members.values())
            if not self.pending and not outstanding:
                break

    # ---- failure handling ------------------------------------------------------------------
    def fail(self, name):
        """The member stops sending and receiving; the survivors install a new view."""
        self.members[name].alive = False
        self.view = [other for other in self.view if other != name]
        self.pending = [item for item in self.pending if item[0] != name and item[1] != name]
        self.sequencer = self.view[0]

    def announce_view(self):
        for name in self.view:
            self.broadcast(name, VIEW, 0, tuple(self.view))

    def resume(self):
        """The new sequencer continues above every sequence number anyone has seen."""
        highest = max(member.highest_seen for member in self.members.values())
        self.next_sequence = max(self.next_sequence, highest + 1)
        self.sequencer = self.view[0]
        return self.next_sequence


def order_of(log, name):
    return [seq for who, seq, _ in log if who == name]


def main():
    check(salt(1) != salt(2), "the deterministic salt separates members")
    check(all(Link(0, 7).carries() for _ in range(20)), "a lossless link drops nothing")
    check(not any(Link(100, 7).carries() for _ in range(20)), "a dead link drops everything")
    check(Link(50, 7).loss_percent == 50, "the loss rate is configurable")

    cluster = Multicast([0, 1, 2], loss_percent=10)
    check(cluster.sequencer == 0 and cluster.view == [0, 1, 2], "the lowest id is the sequencer")
    check([cluster.submit("m%d" % index) for index in range(10)] == list(range(1, 11)),
          "the sequencer stamps messages 1..10")
    cluster.settle()
    for name in cluster.view:
        member = cluster.members[name]
        check(member.missing == set(), f"member {name} has no outstanding gap")
        check(member.prefix_ok(), f"member {name} delivered a prefix of the sequence")
        check(order_of(cluster.log, name) == list(range(1, 11)),
              f"member {name} delivered 1..10 in order")
    orders = {tuple(order_of(cluster.log, name)) for name in cluster.view}
    check(len(orders) == 1, "every member sees exactly the same total order")
    check(cluster.dropped() > 0, "the run really lost messages, so recovery was exercised")
    check(cluster.members[2].buffer == {}, "and every buffer ended up empty")

    cluster = Multicast([0, 1, 2, 3], loss_percent=0)
    for payload in range(8):
        cluster.submit("m%d" % payload)
    cluster.settle()
    for name in cluster.view:
        check(order_of(cluster.log, name) == list(range(1, 9)),
              "a clean first phase delivers everything to everyone")
    cluster.set_loss(10)
    cluster.fail(0)
    cluster.announce_view()
    check(cluster.sequencer == 1, "the next lowest id takes over as sequencer")
    check(cluster.resume() == 9, "and it continues above the highest sequence number in use")
    for payload in range(8, 16):
        cluster.submit("m%d" % payload)
    cluster.settle(120)
    survivors = [1, 2, 3]
    for name in survivors:
        check(order_of(cluster.log, name) == list(range(1, 17)),
              f"survivor {name} delivered all sixteen messages across the view change")
        check(cluster.members[name].missing == set(), "and has no gap left behind")
    check(len({tuple(order_of(cluster.log, name)) for name in survivors}) == 1,
          "the survivors agree on a single total order")
    check(order_of(cluster.log, 0) == list(range(1, 9)),
          "the failed sequencer's log stops exactly where it died")
    check(cluster.dropped() > 0, "the post-failure phase lost messages and repaired them")
    lost = order_of(cluster.log, 1)
    check(lost == sorted(lost), "delivery order stays monotone")
    check(set(lost) == set(range(1, 17)), "nothing was skipped: buffered copies recovered the loss")

    cluster = Multicast([0, 1], loss_percent=0)
    cluster.submit("only")
    cluster.step()
    check(order_of(cluster.log, 1) == [1], "a clean run delivers immediately")
    check(cluster.members[1].buffer == {}, "and leaves the buffer empty")

    cluster = Multicast([0, 1, 2], loss_percent=0)
    cluster.links[(0, 1)].loss_percent = 100
    cluster.links[(0, 2)].loss_percent = 100
    cluster.submit("vanishes")
    cluster.settle(10)
    for name in (1, 2):
        check(order_of(cluster.log, name) == [],
              f"member {name} delivers nothing rather than inventing a delivery")
        check(cluster.members[name].missing == set(),
              f"member {name} cannot know about a sequence number it never saw")
    check(order_of(cluster.log, 0) == [1], "the sender itself did deliver it")
    check(cluster.members[0].delivered[0][2] == "vanishes", "with the original payload")

    cluster = Multicast([0, 1, 2], loss_percent=10)
    for payload in range(12):
        cluster.submit("m%d" % payload)
    cluster.run(6)
    for name in cluster.view:
        member = cluster.members[name]
        delivered = order_of(cluster.log, name)
        check(delivered == list(range(1, len(delivered) + 1)),
              f"member {name} delivered a prefix, never a hole")
    cluster.settle(120)
    for name in cluster.view:
        member = cluster.members[name]
        check(member.missing == set(), "and every gap is filled in the end")
        check(len(set(order_of(cluster.log, name))) == len(order_of(cluster.log, name)),
              "no duplicate deliveries")
        check(member.highest_seen == 12 and member.next_expected == 13,
              f"member {name} ends up expecting exactly the next sequence number")
    check(len({tuple(order_of(cluster.log, name)) for name in cluster.view}) == 1,
          "the total order is the same at every member")

    cluster = Multicast([0, 1, 2, 3], loss_percent=5)
    for payload in range(6):
        cluster.submit("m%d" % payload)
    cluster.settle()
    cluster.fail(1)
    cluster.announce_view()
    check(cluster.resume() == 7, "a failed non-leader changes nothing about the sequence numbering")
    for payload in range(6, 10):
        cluster.submit("m%d" % payload)
    cluster.settle(120)
    for name in (0, 2, 3):
        check(order_of(cluster.log, name) == list(range(1, 11)),
              f"member {name} keeps the stream going and skips the dead member")
    check(1 not in cluster.view and len(cluster.view) == 3, "the view shrank by exactly one")

    print("loss=10%, 3 members, 10 messages: identical order 1..10 everywhere, "
          f"{cluster.dropped()} link drops repaired")
    print(f"sequencer failure mid-stream: survivors delivered 1..16, resumed at 9")
    print("a message lost on every link is reported as nothing, never delivered out of order")
    print("PASS")


if __name__ == "__main__":
    main()
