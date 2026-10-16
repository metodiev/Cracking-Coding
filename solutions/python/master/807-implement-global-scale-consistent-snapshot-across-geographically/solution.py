"""Task 807 - Implement global scale consistent snapshot across geographically distributed shards.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A Chandy-Lamport global snapshot of a geo-distributed ledger.  Shards sit in different regions and
talk over FIFO channels with per-link one-way latency.  The initiator injects a marker into every
outgoing channel; a shard records its local state the first time it sees a marker, forwards markers
on its own outgoing channels and records every application message that arrives after it has
recorded (the classic in-transit set).  The result is a *consistent cut*: what every shard includes
is a prefix of that shard's history and, crucially, if a receive is inside the cut then its send is
inside the cut too - so the captured states plus the recorded in-transit messages form a global
state the system really passed through, even though shards keep serving transfers while the
snapshot runs and no clock is synchronised.

The workload is a ledger: a transfer debits one account and ships a credit message to the shard
that owns the destination account, so money is either on a shard or inside a channel.  That gives
the snapshot an invariant a wrong snapshot cannot satisfy: the sum of all captured balances plus
the amount of every recorded in-transit credit equals the opening total.  A "naive" snapshot that
grabs every shard at one wall-clock instant is implemented alongside, and the checks show it breaks
the invariant while the marker protocol does not.

Verified invariants:
  cut consistency     for every message, if the receiver applied it before its own capture then the
                      sender sent it before its capture - brute-forced over the entire event log;
  prefix property     each captured state equals an independent replay of that shard's logged
                      mutations up to its capture point;
  in-transit set      the recorded set is exactly the messages sent before the sender's capture and
                      received after the receiver's capture;
  money conservation  captured balances plus in-transit credits equal the opening total for every
                      snapshot, while the naive wall-clock grab loses the credits in flight;
  referential         in-transit credits are always positive, so no account is reported short,
   integrity          and a consistent snapshot never shows a negative balance;
  non-blocking        transfers keep flowing during the snapshot and the protocol adds exactly one
                      marker per directed edge and nothing else;
  termination bound   every snapshot completes within a few diameters after the injection, whatever
                      the interleaving and whoever initiates it;
  determinism         identical schedules produce identical snapshots, identical in-transit sets
                      and identical message counts.

Simplified versus a production snapshot service: one process, so the network is an explicit event
queue rather than sockets; channels are FIFO with a *fixed* per-link latency - a jittered
per-message latency would reorder messages on a link and break the marker protocol, which the
checks would immediately expose - and there is no loss, retransmission or view change; the ledger
has a few accounts per shard; the snapshot is held in memory rather than streamed to object
storage, and the "wall clock" used by the naive comparison is a simulated counter.
"""
import heapq

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class Shard:
    def __init__(self, name, accounts):
        self.name = name
        self.balances = dict(accounts)
        self.opening = dict(accounts)
        self.captured = False
        self.capture_seq = None
        self.captured_state = None
        self.in_transit = []           # (source shard, message) recorded after this shard captured
        self.log = []                  # (global seq, kind, payload) local event log
        self.marker_sent = set()

    def total(self):
        return sum(self.balances.values())

    def mutations_before(self, seq):
        """Replay this shard's logged mutations up to (but excluding) `seq` from its opening."""
        replayed = dict(self.opening)
        for event_seq, kind, payload in self.log:
            if event_seq >= seq:
                continue
            if kind == "debit":
                replayed[payload[0]] -= payload[1]
            elif kind == "credit":
                replayed[payload[0]] += payload[1]
        return replayed


class Network:
    """Deterministic FIFO network: per-link one-way latency, one global event sequence."""

    def __init__(self, shards, latency=None):
        self.shards = shards
        self.queue = []
        self.now = 0
        self.seq = 0
        self.clock = 0x5EED
        self.messages = 0
        self.markers = 0
        self.latency = latency or {}
        self.sent_log = {}             # message id -> (sender, seq, time, kind)
        self.recv_log = {}             # message id -> (receiver, seq, time)
        self.payloads = {}             # message id -> payload

    def link_latency(self, source, target):
        """One fixed latency per directed link: a jittered latency would break FIFO order, and
        with it the guarantee that a marker overtakes every message sent after it."""
        key = (source, target)
        if key not in self.latency:
            self.clock, raw = splitmix64(self.clock + 31 * len(source) + 17 * len(target))
            self.latency[key] = 1 + raw % 3
        return self.latency[key]

    def send(self, source, target, kind, payload=None, message_id=None):
        event_seq = self.next_seq()
        self.messages += 1
        if kind == "marker":
            self.markers += 1
        identity = message_id if message_id is not None else "m%d" % event_seq
        message = {"id": identity, "kind": kind, "from": source.name, "to": target.name,
                   "payload": payload, "seq": event_seq}
        self.sent_log[identity] = (source.name, event_seq, self.now, kind)
        self.payloads[identity] = payload
        source.log.append((event_seq, "send", (identity, kind, target.name)))
        heapq.heappush(self.queue, (self.now + self.link_latency(source.name, target.name),
                                    event_seq, identity, source.name, target.name, message))
        return message

    def next_seq(self):
        self.seq += 1
        return self.seq

    def deliver_due(self):
        delivered = 0
        while self.queue and self.queue[0][0] <= self.now:
            _, _, identity, source, target, message = heapq.heappop(self.queue)
            recv_seq = self.next_seq()
            self.recv_log[identity] = (target, recv_seq, self.now)
            receiver = self.shards[target]
            receiver.log.append((recv_seq, "recv", (identity, message["kind"], source)))
            delivered += 1
            if message["kind"] == "marker":
                receive_marker(self, receiver, source)
            else:
                receive_transfer(self, receiver, self.shards[source], message)
        return delivered

    def advance(self):
        self.now = self.queue[0][0] if self.queue else self.now + 1
        return self.now

    def in_flight_total(self):
        return sum(entry[5]["payload"][1] for entry in self.queue
                   if entry[5]["kind"] == "transfer")


# ------------------------------------------------------------------------ marker protocol
def capture(network, shard):
    shard.captured = True
    shard.capture_seq = network.next_seq()
    shard.captured_state = dict(shard.balances)
    shard.log.append((shard.capture_seq, "capture", None))
    for other in network.shards.values():
        if other.name == shard.name or other.name in shard.marker_sent:
            continue
        shard.marker_sent.add(other.name)
        network.send(shard, other, "marker",
                     message_id="marker-%s-%s" % (shard.name, other.name))


def receive_marker(network, shard, source):
    if not shard.captured:
        capture(network, shard)


def receive_transfer(network, shard, source_shard, message):
    account, amount = message["payload"]
    if shard.captured:
        shard.in_transit.append((source_shard.name, message))
    shard.balances[account] += amount
    shard.log.append((network.seq, "credit", (account, amount)))


class Snapshot:
    def __init__(self, network):
        self.network = network

    @staticmethod
    def begin(network, initiator):
        for shard in network.shards.values():       # a fresh snapshot starts from a clean state
            shard.captured = False
            shard.capture_seq = None
            shard.captured_state = None
            shard.in_transit = []
            shard.marker_sent = set()
        capture(network, network.shards[initiator])

    def pending_markers(self):
        return any(entry[5]["kind"] == "marker" for entry in self.network.queue)

    def complete(self):
        """All shards captured and no marker left in flight, so every channel is marked."""
        return (all(shard.captured for shard in self.network.shards.values())
                and not self.pending_markers())

    def state(self):
        return {name: dict(shard.captured_state or {})
                for name, shard in self.network.shards.items()}

    def in_transit_total(self):
        return sum(message["payload"][1] for shard in self.network.shards.values()
                   for _source, message in shard.in_transit)

    def in_transit_ids(self):
        return sorted(message["id"] for shard in self.network.shards.values()
                      for _source, message in shard.in_transit)


def naive_snapshot(network):
    """Grab every shard at one wall-clock instant: simple, and not a consistent cut."""
    return {name: dict(shard.balances) for name, shard in network.shards.items()}


def transfer(network, source_name, target_name, account_from, account_to, amount, tag):
    source = network.shards[source_name]
    source.balances[account_from] -= amount
    source.log.append((network.next_seq(), "debit", (account_from, amount)))
    return network.send(source, network.shards[target_name], "transfer", (account_to, amount),
                        message_id="t-" + tag)


def transfer_accounting(network):
    """Classify every delivered transfer against the cut, using only the event log."""
    inside = in_transit = absent = 0
    in_transit_ids, absent_ids, violations = [], [], []
    for identity, (sender_name, send_seq, _sent_at, kind) in sorted(network.sent_log.items()):
        if kind != "transfer" or identity not in network.recv_log:
            continue
        receiver_name, recv_seq, _recv_at = network.recv_log[identity]
        sender = network.shards[sender_name]
        receiver = network.shards[receiver_name]
        if not sender.captured or not receiver.captured:
            continue
        amount = network.payloads[identity][1]
        debit_inside = send_seq < sender.capture_seq
        credit_inside = recv_seq < receiver.capture_seq
        if debit_inside and credit_inside:
            inside += amount
        elif debit_inside:
            in_transit += amount
            in_transit_ids.append(identity)
        elif credit_inside:
            violations.append(identity)          # a credit inside the cut with no debit in it
        else:
            absent += amount
            absent_ids.append(identity)
    return {"inside": inside, "in_transit": in_transit, "absent": absent,
            "in_transit_ids": sorted(in_transit_ids), "absent_ids": sorted(absent_ids),
            "violations": sorted(violations)}


def drive(network, rounds, per_round=None, snapshot=None):
    for round_index in range(rounds):
        if per_round is not None:
            per_round(round_index)
        network.deliver_due()
        if snapshot is not None and snapshot.complete():
            return round_index + 1
        network.advance()
    return rounds


def cut_consistency_problems(network):
    """Brute force: a message applied inside the cut must have been sent inside the cut."""
    problems = []
    for identity, (sender_name, send_seq, _sent_at, kind) in sorted(network.sent_log.items()):
        if kind == "marker" or identity not in network.recv_log:
            continue
        receiver_name, recv_seq, _recv_at = network.recv_log[identity]
        sender = network.shards[sender_name]
        receiver = network.shards[receiver_name]
        if not sender.captured or not receiver.captured:
            continue
        if recv_seq < receiver.capture_seq and send_seq > sender.capture_seq:
            problems.append((identity, sender_name, receiver_name))
    return problems


def prefix_problems(network):
    """Each capture must equal a replay of that shard's logged mutations before the capture."""
    problems = []
    for shard in network.shards.values():
        if not shard.captured:
            problems.append((shard.name, "never captured"))
            continue
        if shard.mutations_before(shard.capture_seq) != shard.captured_state:
            problems.append((shard.name, "captured state is not the replayed prefix"))
    return problems


def in_transit_oracle(network):
    """Independent derivation: sent before the sender captured, received after the receiver did."""
    derived = set()
    for identity, (sender_name, send_seq, _sent_at, kind) in sorted(network.sent_log.items()):
        if kind != "transfer" or identity not in network.recv_log:
            continue
        receiver_name, recv_seq, _recv_at = network.recv_log[identity]
        sender = network.shards[sender_name]
        receiver = network.shards[receiver_name]
        if not sender.captured or not receiver.captured or sender_name == receiver_name:
            continue
        if send_seq < sender.capture_seq and recv_seq > receiver.capture_seq:
            derived.add(identity)
    return derived


def channel_record_oracle(network):
    """The set the algorithm should record: transfers applied after the receiver captured."""
    derived = set()
    for identity, (_sender, _seq, _sent_at, kind) in sorted(network.sent_log.items()):
        if kind != "transfer" or identity not in network.recv_log:
            continue
        receiver_name, recv_seq, _recv_at = network.recv_log[identity]
        receiver = network.shards[receiver_name]
        if receiver.captured and recv_seq > receiver.capture_seq:
            derived.add(identity)
    return derived


def build(shard_count=4, accounts=3, seed=11):
    names = ["eu", "us", "apac", "sa", "cn"][:shard_count]
    shards = {}
    state = seed
    for name in names:
        balances = {}
        for index in range(accounts):
            state, raw = splitmix64(state)
            balances["%s-a%d" % (name, index)] = 100 + raw % 50
        shards[name] = Shard(name, balances)
    return Network(shards)


def main():
    # --- 1. an idle system: the snapshot is the exact live state -------------------
    quiet = build()
    opening = sum(shard.total() for shard in quiet.shards.values())
    shot = Snapshot(quiet)
    Snapshot.begin(quiet, "eu")
    drive(quiet, 40, snapshot=shot)
    check(shot.complete(), "the marker protocol must reach every shard")
    check(shot.in_transit_ids() == [], "an idle system must have nothing in transit")
    check(shot.state() == {name: dict(shard.balances) for name, shard in quiet.shards.items()},
          "on an idle system the snapshot must equal the live state")
    check(sum(sum(row.values()) for row in shot.state().values()) == opening,
          "money must be conserved by the idle snapshot")
    check(cut_consistency_problems(quiet) == [], "the idle cut must be consistent")
    check(prefix_problems(quiet) == [], "each capture must equal a replay of its local log")

    # --- 2. transfers in flight: the in-transit set is recorded and used ------------
    busy = build(seed=13)
    total = sum(shard.total() for shard in busy.shards.values())
    plan = [("eu" if index % 2 else "us", "apac" if index % 3 else "sa", index * 3 + 1)
            for index in range(24)]
    state = 7

    def workload(round_index):
        nonlocal state
        if round_index >= len(plan):
            return
        source, target, amount = plan[round_index]
        state, raw = splitmix64(state)
        account = "%s-a%d" % (source, raw % 3)
        transfer(busy, source, target, account, target + "-a0", amount, "w%d" % round_index)

    shot = Snapshot(busy)
    Snapshot.begin(busy, "us")
    rounds = drive(busy, 200, per_round=workload, snapshot=shot)
    check(shot.complete(), "the snapshot must complete while traffic keeps flowing")
    check(rounds <= 12,
          "the snapshot must terminate within a few rounds of the injection, took %d" % rounds)
    check(cut_consistency_problems(busy) == [],
          "every message applied inside the cut must have been sent inside it: %s"
          % (cut_consistency_problems(busy)[:3],))
    check(prefix_problems(busy) == [],
          "each capture must equal a replay of that shard's local prefix: %s"
          % (prefix_problems(busy)[:3],))
    recorded = set(shot.in_transit_ids())
    check(recorded == channel_record_oracle(busy),
          "the recorded channel state must be exactly the transfers applied after the receiver "
          "captured: %s vs %s" % (sorted(recorded)[:4], sorted(channel_record_oracle(busy))[:4]))
    money_relevant = in_transit_oracle(busy)
    check(money_relevant <= recorded,
          "every transfer whose debit is inside the cut and credit outside must be recorded: %s"
          % (sorted(money_relevant - recorded)[:4],))
    accounting = transfer_accounting(busy)
    check(accounting["violations"] == [],
          "a credit may never be inside the cut without its debit: %s"
          % (accounting["violations"][:3],))
    check(accounting["in_transit_ids"] == sorted(money_relevant),
          "the classified in-transit transfers must be exactly the derivation: %s vs %s"
          % (accounting["in_transit_ids"][:4], sorted(money_relevant)[:4]))
    check(recorded - money_relevant == set(accounting["absent_ids"]),
          "the recorded extras must be exactly the transfers sent after the sender captured: %s"
          % (sorted(recorded - money_relevant)[:4],))
    captured_total = sum(sum(row.values()) for row in shot.state().values())
    check(captured_total == total - accounting["in_transit"],
          "captured balances must be the opening total minus the credits inside channels: "
          "%d vs %d - %d" % (captured_total, total, accounting["in_transit"]))
    check(captured_total + accounting["in_transit"] == total,
          "captured balances plus the in-transit amount must equal the opening total: %d + %d "
          "vs %d" % (captured_total, accounting["in_transit"], total))
    check(shot.in_transit_total() - accounting["absent"] == accounting["in_transit"],
          "the recorded channel state minus the transfers not yet sent at capture time must be "
          "the in-transit amount: %d - %d vs %d"
          % (shot.in_transit_total(), accounting["absent"], accounting["in_transit"]))
    check(len(money_relevant) > 0, "the schedule must leave credits inside channels")
    check(len(accounting["absent_ids"]) > 0,
          "some transfers must lie entirely outside the cut, which the accounting must classify")
    check(len({shard.capture_seq for shard in busy.shards.values()}) > 1,
          "shards must capture at different moments, otherwise the cut is trivial: %s"
          % [shard.capture_seq for shard in busy.shards.values()])
    late = [identity for identity, (sender, seq, _time, kind) in busy.sent_log.items()
            if kind == "transfer" and seq > busy.shards[sender].capture_seq]
    check(len(late) > 0,
          "some transfers must be sent after their sender captured, so the cut is a real cut")

    # --- 3. the naive wall-clock grab breaks the invariant ------------------------
    scenario = build(shard_count=2, accounts=2, seed=5)
    start_total = sum(shard.total() for shard in scenario.shards.values())
    transfer(scenario, "eu", "us", "eu-a0", "us-a0", 30, "n1")
    naive = naive_snapshot(scenario)
    check(sum(sum(row.values()) for row in naive.values()) == start_total - 30,
          "the naive grab must lose the credit that is still inside the channel")
    marker = Snapshot(scenario)
    Snapshot.begin(scenario, "us")
    drive(scenario, 40, snapshot=marker)
    check(marker.complete(), "the two-shard snapshot must complete")
    marker_total = sum(sum(row.values()) for row in marker.state().values())
    marker_accounting = transfer_accounting(scenario)
    check(marker_total + marker_accounting["in_transit"] == start_total,
          "the marker snapshot must conserve money: %d + %d vs %d"
          % (marker_total, marker_accounting["in_transit"], start_total))
    check((marker.in_transit_total() == 30) != (marker_total == start_total),
          "the in-flight credit must be accounted for on exactly one side of the cut: captured "
          "total %d, in transit %d, opening %d"
          % (marker_total, marker.in_transit_total(), start_total))
    check(cut_consistency_problems(scenario) == [],
          "the two-shard cut must be consistent even with a credit in flight")

    # --- 4. message cost ----------------------------------------------------------
    markers = busy.markers
    edges = len(busy.shards) * (len(busy.shards) - 1)
    transfers = len([identity for identity in busy.sent_log if identity.startswith("t-")])
    check(markers == edges,
          "markers must be one per directed edge: %d vs %d" % (markers, edges))
    check(busy.messages == markers + transfers,
          "the protocol must add nothing but markers: %d messages, %d markers, %d transfers"
          % (busy.messages, markers, transfers))
    check(transfers <= len(plan), "only the planned transfers may be sent: %d" % transfers)
    check(transfers > 0, "the workload must have sent transfers")

    # --- 5. ledger integrity -------------------------------------------------------
    ledger = build(shard_count=3, seed=29)
    ledger_opening = sum(shard.total() for shard in ledger.shards.values())

    def ledger_workload(round_index):
        source = ["eu", "us", "apac"][round_index % 3]
        target = ["apac", "eu", "us"][round_index % 3]
        transfer(ledger, source, target, "%s-a%d" % (source, round_index % 3),
                 target + "-a1", 5 + round_index, "l%d" % round_index)

    shot = Snapshot(ledger)
    Snapshot.begin(ledger, "apac")
    drive(ledger, 120, per_round=ledger_workload, snapshot=shot)
    check(shot.complete(), "the ledger snapshot must complete")
    captured_ledger = sum(sum(row.values()) for row in shot.state().values())
    ledger_accounting = transfer_accounting(ledger)
    check(captured_ledger + ledger_accounting["in_transit"] == ledger_opening,
          "the ledger snapshot must conserve money: %d + %d vs %d"
          % (captured_ledger, ledger_accounting["in_transit"], ledger_opening))
    check(ledger_accounting["violations"] == [],
          "no ledger credit may be inside the cut without its debit")
    check(cut_consistency_problems(ledger) == [], "the ledger cut must be consistent")
    check(prefix_problems(ledger) == [], "the ledger captures must equal local replays")
    reported = [value for row in shot.state().values() for value in row.values()]
    check(all(value >= 0 for value in reported),
          "no account may be reported negative in a consistent snapshot")
    check(all(message["payload"][1] > 0 for shard in ledger.shards.values()
              for _source, message in shard.in_transit),
          "every recorded in-transit credit must be a positive amount")

    # --- 6. repeated snapshots, each consistent -----------------------------------
    repeated = build(seed=37)
    live_opening = sum(shard.total() for shard in repeated.shards.values())
    results = []
    for round_index in range(12):
        source = ["eu", "us", "apac", "sa"][round_index % 4]
        target = ["sa", "apac", "us", "eu"][round_index % 4]
        transfer(repeated, source, target, source + "-a0", target + "-a0",
                 1 + round_index % 4, "r%d" % round_index)
        repeated.deliver_due()
        repeated.advance()
        if round_index in (3, 7, 11):
            shot = Snapshot(repeated)
            Snapshot.begin(repeated, "eu")
            drive(repeated, 60, snapshot=shot)
            hold = sum(sum(row.values()) for row in shot.state().values())
            accounting = transfer_accounting(repeated)
            results.append((round_index, shot.complete(),
                            hold + accounting["in_transit"] == live_opening,
                            accounting["violations"] == [],
                            cut_consistency_problems(repeated) == [],
                            prefix_problems(repeated) == []))
    check(len(results) == 3, "three snapshots must have been taken")
    check(all(complete and conserves and clean and consistent and prefix
              for _index, complete, conserves, clean, consistent, prefix in results),
          "every repeated snapshot must be complete, conservative and consistent: %s"
          % (results,))
    check(sum(shard.total() for shard in repeated.shards.values())
          + repeated.in_flight_total() == live_opening,
          "the live ledger plus its channels must conserve money")

    # --- 7. determinism -----------------------------------------------------------
    first = build(seed=41)
    second = build(seed=41)
    for target in (first, second):
        for index in range(10):
            transfer(target, "eu", "us", "eu-a0", "us-a0", index + 1, "d%d" % index)
    first_shot, second_shot = Snapshot(first), Snapshot(second)
    Snapshot.begin(first, "sa")
    Snapshot.begin(second, "sa")
    drive(first, 80, snapshot=first_shot)
    drive(second, 80, snapshot=second_shot)
    check(first_shot.state() == second_shot.state(),
          "identical schedules must produce identical snapshots")
    check(first_shot.in_transit_ids() == second_shot.in_transit_ids(),
          "identical schedules must record the same in-transit messages")
    check((first.messages, first.markers) == (second.messages, second.markers),
          "identical schedules must send the same messages")
    check(cut_consistency_problems(first) == [] and prefix_problems(first) == [],
          "the deterministic snapshot must be consistent")

    print("busy snapshot: %d shards, %d transfers, %d in-transit credits, %d markers over %d "
          "directed edges" % (len(busy.shards), transfers, len(recorded), markers, edges))
    print("conservation: captured %d + in transit %d = %d (opening %d); naive grab total %d"
          % (captured_total, shot.in_transit_total(), total, total,
             sum(sum(row.values()) for row in naive.values())))
    print("consistency problems: idle %d, busy %d, ledger %d; capture instants %s"
          % (len(cut_consistency_problems(quiet)), len(cut_consistency_problems(busy)),
             len(cut_consistency_problems(ledger)),
             [busy.shards[name].capture_seq for name in sorted(busy.shards)]))
    print("PASS")


if __name__ == "__main__":
    main()
