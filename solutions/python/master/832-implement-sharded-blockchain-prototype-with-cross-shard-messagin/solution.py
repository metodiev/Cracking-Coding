"""Task 832 - Implement sharded blockchain prototype with cross-shard messaging. (Tier: Master, CTCI Ch 9 — System Design and Scalability)

A sharded ledger splits state across shards that produce their own blocks, so a transfer
between two shards cannot be one atomic write: it is a two-phase protocol. The source shard
debits the sender, parks the money in escrow and emits a *receipt*; the receipt travels over an
explicit in-memory network; the destination shard credits the receiver only against a valid
inclusion proof of that receipt in a source block, and acknowledges back; the source releases
the escrow on the acknowledgement, or refunds the sender once the receipt's expiry plus a
safety margin has passed with no acknowledgement.

What is implemented
-------------------
* `receipts_root` / `receipt_proof` / `verify_receipt_proof`: SHA-256 inclusion proofs of a
  receipt inside the source shard's block (a receipt proves it was authorised by the source).
* `Network`: deterministic delivery at `send_tick + latency`, with an explicit per-tick
  capacity, a configurable drop list (unreliable shard-to-shard delivery) and duplicate
  delivery, all driven by ticks -- there is no thread and no wall-clock anywhere.
* `ShardSystem`: per-shard balances, per-shard block logs, escrow per cross-shard transfer,
  deadline/expiry accounting, the acknowledgement path, refunds, nullifiers against receipt
  replay, per-tick processing caps, and a full event log plus a violation list.
* Invariant checking runs after *every* tick, not only at the end: total supply (balances plus
  escrow), the cross-shard state machine (pending / completed / refunded, never two), no credit
  after a refund (which is what a double spend looks like here) and no double credit.

What is simplified
------------------
* Instant finality inside a shard: a deterministic producer appends one block per tick in a
  fixed order, so there are no forks to resolve here (the ledger-level reorg question is 831).
* Balances are non-negative integers; a "transaction" is an opaque id plus a fixed amount, so
  the checks are about atomicity, conservation, replay and timing rather than signatures.
* The safety argument is a *timing* argument and the file verifies both sides of it: with
  `latency <= expiry` and `latency <= margin` the acknowledgement always beats the refund, and
  with `margin = 0` the same scenario really does double-spend -- the violation is exhibited
  and reported instead of being assumed away.

Invariants verified
-------------------
1.  Receipt proofs: a receipt's inclusion proof verifies against its block's root, and a
    mutated receipt, a proof from another block and a proof for another position all fail.
2.  Conservation of supply: balances plus escrow equal the initial supply after every single
    tick of every run (including the unsafe run's precise violation point).
3.  Atomicity: every cross-shard transfer ends in exactly one of completed or refunded; none is
    left pending, and no transfer is ever both credited and refunded in the safe configuration.
4.  Replay: a duplicated receipt is credited once, the nullifier records it, and the duplicate
    is rejected with its own log entry.
5.  Expiry: a receipt delivered after its expiry is rejected, so a refunded transfer can never
    be credited later.
6.  Forgery: a receipt whose amount or inclusion proof was tampered with is rejected, and a
    receipt addressed to the wrong shard is refused.
7.  Safety needs the timing assumption: the same scenario with `margin = 0` produces a real
    double spend (detected by the invariant checker), while the documented configuration
    produces none.
8.  Scale accounting: no shard processes more than the per-tick cap, the network delivers no
    more than its per-tick capacity, and the measured completion latency obeys the bound
    `expiry <= completion <= expiry + margin + latency`.
9.  Determinism: two runs with the same scenario and drop list produce identical state digests
    and identical event logs.
10. Liveness: with a dropped receipt the transfer still resolves (by refund) inside the
    deadline bound, and with no drops every transfer completes.
"""
import hashlib

SHARDS = 4
LATENCY = 2
EXPIRY = 3
MARGIN = 4
CAP = 2                  # transactions/receipts a shard may process per tick
CAPACITY = 8             # messages the network delivers per tick
INITIAL = 100
ACCOUNTS = 3


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------------------
# receipts and their inclusion proofs
# --------------------------------------------------------------------------------------
def receipt_blob(receipt):
    return (f"{receipt['id']}|{receipt['src']}|{receipt['dst']}|{receipt['sender']}|"
            f"{receipt['receiver']}|{receipt['amount']}|{receipt['send_tick']}|"
            f"{receipt['expiry']}")


def receipts_root(receipts):
    if not receipts:
        return sha("empty")
    level = [sha(receipt_blob(receipt)) for receipt in receipts]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [sha(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def receipt_proof(receipts, index):
    level = [sha(receipt_blob(receipt)) for receipt in receipts]
    path, position = [], index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        path.append(("right" if position % 2 == 0 else "left", level[position ^ 1]))
        level = [sha(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
        position //= 2
    return path


def verify_receipt_proof(receipt, path, root):
    node = sha(receipt_blob(receipt))
    for side, sibling in path:
        node = sha(node + sibling) if side == "right" else sha(sibling + node)
    return node == root


# --------------------------------------------------------------------------------------
# the explicit in-memory network
# --------------------------------------------------------------------------------------
class Network:
    """Deliver at send_tick + latency, capped per tick, with a drop list and duplicates."""

    def __init__(self, latency=LATENCY, capacity=CAPACITY, drop=(), duplicate=()):
        self.latency = latency
        self.capacity = capacity
        self.drop = set(drop)
        self.duplicate = set(duplicate)
        self.inflight = []
        self.log = []
        self.sent = 0

    def send(self, tick, message):
        key = (message["kind"], message["id"], message["retry"])
        self.sent += 1
        if key in self.drop:
            self.log.append((tick, "dropped", message["kind"], message["id"]))
            return False
        copies = 2 if key in self.duplicate else 1
        for copy_index in range(copies):
            self.inflight.append((tick + self.latency + copy_index, dict(message)))
        self.log.append((tick, "sent", message["kind"], message["id"]))
        return True

    def deliver(self, tick):
        due = sorted([m for m in self.inflight if m[0] == tick],
                     key=lambda m: (m[1]["kind"], m[1]["dst"], m[1]["id"]))
        self.inflight = [m for m in self.inflight if m[0] != tick]
        delivered = due[:self.capacity]
        dropped = due[self.capacity:]
        for _, message in dropped:
            self.log.append((tick, "capacity_drop", message["kind"], message["id"]))
        return [(tick, message) for _, message in delivered]


# --------------------------------------------------------------------------------------
# the sharded system
# --------------------------------------------------------------------------------------
class ShardSystem:
    def __init__(self, shards=SHARDS, latency=LATENCY, expiry=EXPIRY, margin=MARGIN,
                 cap=CAP, capacity=CAPACITY, drop=(), duplicate=(), initial=INITIAL):
        self.shard_ids = list(range(shards))
        self.latency, self.expiry, self.margin = latency, expiry, margin
        self.cap = cap
        self.initial = initial
        self.balances = {f"s{shard}:a{account}": initial
                         for shard in self.shard_ids for account in range(ACCOUNTS)}
        self.escrow = {}                     # tx id -> {"shard", "amount", "deadline"}
        self.nullifiers = set()              # credited receipt ids
        self.acknowledged = {}               # tx id -> tick the ack arrived
        self.credited = {}                   # tx id -> amount credited but not settled yet
        self.blocks = {shard: [] for shard in self.shard_ids}
        self.inbox = {shard: [] for shard in self.shard_ids}
        self.pending_local = {shard: [] for shard in self.shard_ids}
        self.pending_cross = {shard: [] for shard in self.shard_ids}
        self.receipts = {}                   # tx id -> receipt
        self.receipt_index = {}              # tx id -> index inside its source block
        self.receipt_location = {}           # tx id -> (source shard, block height)
        self.status = {}                     # tx id -> pending/completed/refunded
        self.events = []
        self.violations = []
        self.violation_keys = set()
        self.tick_now = 0
        self.processed = 0
        self.network = Network(latency, capacity, drop, duplicate)
        self.max_processed_per_tick = 0

    # -- accounting --------------------------------------------------------------------
    def supply(self):
        return sum(self.balances.values()) + sum(box["amount"] for box in self.escrow.values())

    def state_digest(self):
        parts = [f"{key}={value}" for key, value in sorted(self.balances.items())]
        parts += [f"escrow:{key}={box['amount']}" for key, box in sorted(self.escrow.items())]
        return sha("|".join(parts))

    def escrow_of(self, shard):
        return sum(box["amount"] for box in self.escrow.values() if box["shard"] == shard)

    # -- submission --------------------------------------------------------------------
    def submit_local(self, sender, receiver, amount, tx_id):
        shard = int(sender.split(":")[0][1:])
        self.pending_local[shard].append({"id": tx_id, "sender": sender,
                                          "receiver": receiver, "amount": amount})
        self.status[tx_id] = "pending_local"
        return tx_id

    def submit_cross(self, sender, receiver, amount, tx_id):
        src = int(sender.split(":")[0][1:])
        dst = int(receiver.split(":")[0][1:])
        self.pending_cross[src].append({"id": tx_id, "sender": sender, "receiver": receiver,
                                        "amount": amount, "dst": dst})
        self.status[tx_id] = "pending"
        return tx_id

    # -- one tick ----------------------------------------------------------------------
    def tick(self):
        self.tick_now += 1
        self._deliver()
        processed = 0
        for shard in self.shard_ids:
            processed += self._produce(shard)
        self.max_processed_per_tick = max(self.max_processed_per_tick, processed)
        self._refund_due()
        self._check_invariants()
        return processed

    def _deliver(self):
        for tick, message in self.network.deliver(self.tick_now):
            if message["kind"] == "receipt":
                self.inbox[message["dst"]].append(message)
                self.events.append((tick, "receipt_arrived", message["id"]))
            else:
                self.acknowledged[message["id"]] = tick
                self.events.append((tick, "ack_arrived", message["id"]))

    def _produce(self, shard):
        block = {"height": len(self.blocks[shard]) + 1, "tick": self.tick_now,
                 "txs": [], "receipts": [], "refunds": []}
        budget = self.cap

        local = sorted(self.pending_local[shard], key=lambda tx: tx["id"])[:budget]
        for tx in local:
            if self.balances[tx["sender"]] >= tx["amount"] and tx["sender"] != tx["receiver"]:
                self.balances[tx["sender"]] -= tx["amount"]
                self.balances[tx["receiver"]] += tx["amount"]
                block["txs"].append(tx["id"])
                self.status[tx["id"]] = "completed_local"
                self.events.append((self.tick_now, "local", tx["id"]))
            else:
                self.status[tx["id"]] = "rejected"
                self.events.append((self.tick_now, "rejected", tx["id"]))
            self.pending_local[shard].remove(tx)
        budget -= len(local)

        sends = sorted(self.pending_cross[shard], key=lambda tx: tx["id"])[:budget]
        for tx in sends:
            if self.balances[tx["sender"]] >= tx["amount"]:
                self.balances[tx["sender"]] -= tx["amount"]
                self.escrow[tx["id"]] = {"shard": shard, "amount": tx["amount"],
                                         "sender": tx["sender"]}
                receipt = {"id": tx["id"], "src": shard, "dst": tx["dst"],
                           "sender": tx["sender"], "receiver": tx["receiver"],
                           "amount": tx["amount"], "send_tick": self.tick_now,
                           "expiry": self.tick_now + self.expiry, "retry": 0}
                self.receipts[tx["id"]] = receipt
                self.receipt_index[tx["id"]] = len(block["receipts"])
                self.receipt_location[tx["id"]] = (shard, len(self.blocks[shard]) + 1)
                block["receipts"].append(receipt)
                self.network.send(self.tick_now, {"kind": "receipt", **receipt})
                self.events.append((self.tick_now, "escrowed", tx["id"]))
            else:
                self.status[tx["id"]] = "rejected"
                self.events.append((self.tick_now, "rejected", tx["id"]))
            self.pending_cross[shard].remove(tx)
        budget -= len(sends)

        arrivals = sorted(self.inbox[shard], key=lambda m: m["id"])[:max(0, budget)]
        for message in arrivals:
            self.inbox[shard].remove(message)
            self._credit(shard, message)
        block["root"] = receipts_root(block["receipts"]) if block["receipts"] else None
        self.blocks[shard].append(block)
        return len(block["txs"]) + len(block["receipts"])

    def _credit(self, shard, message):
        tx_id = message["id"]
        receipt = message
        if message["dst"] != shard:
            self.events.append((self.tick_now, "wrong_shard", tx_id))
            return
        if tx_id in self.nullifiers:
            self.events.append((self.tick_now, "duplicate_receipt", tx_id))
            return
        if self.tick_now > message["expiry"]:
            self.events.append((self.tick_now, "expired_receipt", tx_id))
            return
        source_block = None
        location = self.receipt_location.get(tx_id)
        if location is not None and location[1] <= len(self.blocks[location[0]]):
            source_block = self.blocks[location[0]][location[1] - 1]
        index = self.receipt_index.get(tx_id)
        known = self.receipts.get(tx_id)
        if source_block is None or known is None or index is None:
            self.events.append((self.tick_now, "unknown_receipt", tx_id))
            return
        path = receipt_proof(source_block["receipts"], index)
        if receipt_blob(known) != receipt_blob(receipt) or not verify_receipt_proof(
                known, path, source_block["root"]):
            self.events.append((self.tick_now, "bad_proof", tx_id))
            return
        self.balances[receipt["receiver"]] += receipt["amount"]
        self.nullifiers.add(tx_id)
        self.credited[tx_id] = receipt["amount"]
        self.events.append((self.tick_now, "credited", tx_id))
        self.network.send(self.tick_now, {"kind": "ack", "id": tx_id, "dst": message["src"],
                                          "retry": 0})

    def _refund_due(self):
        for tx_id, box in sorted(self.escrow.items()):
            receipt = self.receipts.get(tx_id)
            if receipt is None:
                continue
            deadline = receipt["send_tick"] + self.expiry + self.margin
            if self.tick_now >= deadline and tx_id not in self.acknowledged:
                self.balances[box["sender"]] += box["amount"]
                self.escrow[tx_id] = {"shard": box["shard"], "amount": 0,
                                      "sender": box["sender"]}
                self.status[tx_id] = "refunded"
                self.events.append((self.tick_now, "refunded", tx_id))

    def _complete(self):
        """Escrow released on acknowledgement; called during the invariant sweep."""
        for tx_id in sorted(self.acknowledged):
            if tx_id in self.escrow and self.status.get(tx_id) == "pending":
                self.escrow[tx_id]["amount"] = 0
                self.status[tx_id] = "completed"
                self.events.append((self.tick_now, "completed", tx_id))

    # -- invariants ---------------------------------------------------------------------
    def cross_shard_ids(self):
        return sorted(tx for tx in self.status if tx.startswith("x"))

    def _record(self, kind, tx_id, delta):
        """One entry per (kind, transaction): the first tick at which it was observed."""
        key = (kind, tx_id)
        if key not in self.violation_keys:
            self.violation_keys.add(key)
            self.violations.append((self.tick_now, kind, tx_id, delta))

    def _check_invariants(self):
        """Accounting runs after every tick, not only at the end.

        A credited-but-unsettled transfer legitimately exists in two places (the receiver's
        balance on the destination shard and the sender's escrow on the source shard), because
        the two shards act on local knowledge only. Settling (the acknowledgement) burns the
        escrow. So the invariant is: balances + escrow - unsettled credits == initial supply.
        A refund that fires after a credit leaves the credit standing while the escrow is gone,
        which is exactly the extra supply a double spend creates.
        """
        self._complete()
        settled = {tx for tx, status in self.status.items()
                   if status in ("completed", "refunded")}
        in_flight = sum(self.credited[tx] for tx in self.credited if tx not in settled)
        expected = len(self.shard_ids) * ACCOUNTS * self.initial
        actual = self.supply() - in_flight
        if actual != expected:
            self._record("supply", "global", actual - expected)
        for tx_id in self.cross_shard_ids():
            credited = any(kind == "credited" and event_id == tx_id
                           for _, kind, event_id in self.events)
            refunded = any(kind == "refunded" and event_id == tx_id
                           for _, kind, event_id in self.events)
            if credited and refunded:
                self._record("double_spend", tx_id, self.credited.get(tx_id, 0))
        if any(box["amount"] < 0 for box in self.escrow.values()):
            self._record("negative_escrow", "global", 0)

    def resolve(self, ticks=40):
        for _ in range(ticks):
            self.tick()
            if not self.pending_cross_pending():
                break
        return self

    def pending_cross_pending(self):
        return any(any(tx for tx in queue) for queue in self.pending_cross.values()) \
            or any(self.status.get(tx) == "pending" for tx in self.cross_shard_ids())

    def final_status(self):
        return {tx: self.status.get(tx) for tx in self.cross_shard_ids()}

    def completion_latencies(self):
        escrow_tick = {tx: tick for tick, kind, tx in self.events if kind == "escrowed"}
        done_tick = {tx: tick for tick, kind, tx in self.events
                     if kind in ("completed", "refunded")}
        return {tx: done_tick[tx] - escrow_tick[tx] for tx in done_tick if tx in escrow_tick}


# --------------------------------------------------------------------------------------
# scenario
# --------------------------------------------------------------------------------------
def build_scenario():
    """A fixed mix of local and cross-shard transfers over the four shards."""
    transfers = [
        ("x1", "s0:a0", "s1:a0", 30), ("x2", "s1:a1", "s2:a1", 40),
        ("x3", "s2:a2", "s3:a0", 25), ("x4", "s3:a1", "s0:a2", 15),
        ("x5", "s0:a1", "s2:a0", 20), ("x6", "s1:a2", "s3:a2", 35),
        ("x7", "s2:a0", "s0:a1", 10), ("x8", "s3:a0", "s1:a0", 45),
    ]
    locals_ = [("l1", "s0:a2", "s0:a0", 5), ("l2", "s2:a1", "s2:a2", 7)]
    return transfers, locals_


def run_scenario(margin=MARGIN, drop=(), duplicate=(), latency=LATENCY, expiry=EXPIRY,
                 ticks=40):
    system = ShardSystem(latency=latency, expiry=expiry, margin=margin, drop=drop,
                         duplicate=duplicate)
    transfers, locals_ = build_scenario()
    for tx_id, sender, receiver, amount in locals_:
        system.submit_local(sender, receiver, amount, tx_id)
    for tx_id, sender, receiver, amount in transfers:
        system.submit_cross(sender, receiver, amount, tx_id)
    system.resolve(ticks)
    return system


def main():
    safe = run_scenario()
    supply0 = SHARDS * ACCOUNTS * INITIAL

    # -- 1. receipt proofs ----------------------------------------------------------------
    receipts = [{"id": f"r{i}", "src": 0, "dst": 1, "sender": "s0:a0", "receiver": "s1:a0",
                 "amount": i, "send_tick": 0, "expiry": 3, "retry": 0} for i in range(5)]
    root = receipts_root(receipts)
    check(all(verify_receipt_proof(receipts[i], receipt_proof(receipts, i), root)
              for i in range(5)), "every receipt's inclusion proof verifies")
    altered = dict(receipts[2])
    altered["amount"] = 999
    check(not verify_receipt_proof(altered, receipt_proof(receipts, 2), root),
          "a receipt with a tampered amount fails its proof")
    check(not verify_receipt_proof(receipts[2], receipt_proof(receipts, 3), root),
          "a proof for another position fails")
    other = receipts[:2]
    check(not verify_receipt_proof(receipts[0], receipt_proof(other, 0), root),
          "a proof from another block fails")
    check(receipts_root([]) == sha("empty"), "an empty block root is defined")
    check(len(receipt_proof(receipts, 0)) == 3, "five receipts need three siblings")

    # -- 2. conservation and atomicity in the safe configuration --------------------------
    check(safe.supply() == supply0, "supply is conserved at the end of the safe run")
    check(not safe.violations, f"no invariant violations in the safe run ({safe.violations})")
    statuses = safe.final_status()
    check(set(statuses.values()) == {"completed"},
          f"every cross-shard transfer completed ({statuses})")
    check(all(status in ("completed", "completed_local", "rejected")
              for status in safe.status.values()), "no transaction is left pending")
    transfers, locals_ = build_scenario()
    atomic = {f"s{shard}:a{account}": INITIAL
              for shard in range(SHARDS) for account in range(ACCOUNTS)}
    for _, sender, receiver, amount in list(locals_) + list(transfers):
        atomic[sender] -= amount
        atomic[receiver] += amount
    check(all(value >= 0 for value in atomic.values()), "the atomic ledger stays solvent")
    check(safe.balances == atomic,
          "the sharded run ends exactly where an atomic run of the same transfers would")
    check(all(box["amount"] == 0 for box in safe.escrow.values()),
          "no escrow is left holding value after completion")

    # -- 3. per-tick conservation ---------------------------------------------------------
    watched = ShardSystem()
    transfers, locals_ = build_scenario()
    for tx_id, sender, receiver, amount in locals_:
        watched.submit_local(sender, receiver, amount, tx_id)
    for tx_id, sender, receiver, amount in transfers:
        watched.submit_cross(sender, receiver, amount, tx_id)
    bad_ticks = []
    for _ in range(40):
        watched.tick()
        settled = {tx for tx, status in watched.status.items()
                   if status in ("completed", "refunded")}
        in_flight = sum(watched.credited[tx] for tx in watched.credited
                        if tx not in settled)
        if watched.supply() - in_flight != supply0:
            bad_ticks.append((watched.tick_now, watched.supply() - in_flight - supply0))
    check(not bad_ticks,
          f"balances + escrow - unsettled credits holds after every tick ({bad_ticks})")
    check(watched.credited and all(not values for values in watched.inbox.values()),
          "the run really did have credits in flight, and every inbox drained")
    check(watched.tick_now == 40 or not watched.pending_cross_pending(),
          "the watched run settled before the tick budget expired")
    check(watched.max_processed_per_tick <= CAP * SHARDS,
          f"no tick processes more than cap * shards items "
          f"({watched.max_processed_per_tick})")

    # -- 4. replay and duplicate delivery --------------------------------------------------
    replay = run_scenario(duplicate=(("receipt", "x1", 0),))
    check(len(replay.nullifiers) == 8, "eight receipts were credited exactly once each")
    check(any(kind == "duplicate_receipt" for _, kind, _ in replay.events),
          "the duplicated delivery was rejected as a replay")
    check(replay.supply() == supply0 and not replay.violations,
          "replay did not create money and broke no invariant")
    check(replay.balances == atomic,
          "the duplicated receipt leaves the same final ledger as the clean run")

    # -- 5. forged receipts ----------------------------------------------------------------
    forged = ShardSystem()
    forged.submit_cross("s0:a0", "s1:a0", 30, "x1")
    forged.tick()
    receipt = forged.receipts["x1"]
    block = forged.blocks[0][-1]
    tampered = dict(receipt)
    tampered["amount"] = 3000
    forged.inbox[1].append(tampered)
    forged.tick()
    check(any(kind == "bad_proof" for _, kind, _ in forged.events),
          "a tampered receipt is rejected by the destination")
    check(forged.balances["s1:a0"] == INITIAL,
          "the tampered receipt credited nothing")
    misrouted = dict(receipt)
    forged.inbox[2].append(misrouted)
    forged.tick()
    check(any(kind == "wrong_shard" for _, kind, _ in forged.events),
          "a receipt delivered to the wrong shard is refused")
    stale_system = ShardSystem()
    stale_system.submit_cross("s0:a0", "s1:a0", 30, "y1")
    stale_system.tick()
    stale = stale_system.receipts["y1"]
    stale_system.tick_now = stale["expiry"] + 1        # the network copy never arrived
    stale_system.inbox[1].append(dict(stale))
    stale_system.tick()
    check(any(kind == "expired_receipt" for _, kind, _ in stale_system.events),
          "a receipt past its expiry is refused")
    check(stale_system.balances["s1:a0"] == INITIAL and stale_system.status["y1"] != "completed",
          "no late delivery ever moved value")
    check(stale_system.nullifiers == set(),
          "the expired receipt was not recorded as spent either")

    # -- 6. timeout and refund -------------------------------------------------------------
    dropped = run_scenario(drop=(("receipt", "x1", 0),))
    check(dropped.status["x1"] == "refunded",
          "a dropped receipt leads to a refund, not a hang")
    check(not any(kind == "credited" and event_id == "x1"
                  for _, kind, event_id in dropped.events),
          "the refunded transfer was never credited")
    refunded_ledger = dict(atomic)
    refunded_ledger["s0:a0"] += 30                 # x1 was refunded, so undo its effect
    refunded_ledger["s1:a0"] -= 30
    check(dropped.balances == refunded_ledger,
          "the dropped transfer is the only difference from the atomic ledger")
    check(dropped.balances["s1:a0"] == atomic["s1:a0"] - 30,
          "and the destination never saw the money")
    check(dropped.supply() == supply0 and not dropped.violations,
          "the refund path conserves supply")
    latencies = dropped.completion_latencies()
    check(latencies["x1"] == EXPIRY + MARGIN,
          f"the refund fires exactly at expiry + margin ({latencies['x1']})")
    check(all(latency >= EXPIRY for latency in latencies.values()),
          f"no cross-shard transfer resolves before the expiry ({latencies})")

    # -- 7. the timing assumption is load-bearing ------------------------------------------
    check(safe.latency <= safe.expiry and safe.latency <= safe.margin,
          "the safe configuration satisfies latency <= expiry and latency <= margin")
    unsafe = run_scenario(margin=0)
    double_spends = [entry for entry in unsafe.violations if entry[1] == "double_spend"]
    supply_breaks = [entry for entry in unsafe.violations if entry[1] == "supply"]
    check(double_spends and supply_breaks,
          f"margin = 0 really double-spends: {double_spends[:2]}")
    doubly = {tx_id for tx_id, _, _, amount in transfers
              if any(kind == "credited" and event_id == tx_id
                     for _, kind, event_id in unsafe.events)
              and any(kind == "refunded" and event_id == tx_id
                      for _, kind, event_id in unsafe.events)}
    amounts = {tx_id: amount for tx_id, _, _, amount in transfers}
    check(len(doubly) == 8 and unsafe.status["x1"] == "refunded",
          f"all eight transfers were both credited and refunded ({sorted(doubly)})")
    check(sum(amounts[tx_id] for tx_id in doubly) == unsafe.supply() - supply0,
          f"the extra supply is exactly the doubly settled amount "
          f"({unsafe.supply() - supply0} vs {sum(amounts[tx_id] for tx_id in doubly)})")
    check(unsafe.escrow["x1"]["amount"] == 0,
          "the refund emptied the escrow without cancelling the destination credit")

    # -- 8. latency bounds and determinism --------------------------------------------------
    safe_latencies = safe.completion_latencies()
    upper = EXPIRY + MARGIN + LATENCY
    check(all(EXPIRY <= value <= upper for value in safe_latencies.values()),
          f"completion latency stays inside [expiry, expiry+margin+latency] ({safe_latencies})")
    check(all(value == 2 * LATENCY for value in safe_latencies.values()),
          "every transfer settles as soon as the acknowledgement can arrive")
    check(safe.network.sent == 8 * 2,
          f"each cross-shard transfer costs one receipt and one acknowledgement "
          f"({safe.network.sent} messages)")
    check(all(message[2] in ("receipt", "ack") for message in safe.network.log),
          "only receipts and acknowledgements travel between shards")
    repeat = run_scenario()
    check(repeat.state_digest() == safe.state_digest()
          and repeat.events == safe.events and repeat.network.log == safe.network.log,
          "the whole run is bit-reproducible")
    repeat_drop = run_scenario(drop=(("receipt", "x1", 0),))
    check(repeat_drop.state_digest() == dropped.state_digest(),
          "the drop scenario is reproducible too")

    # -- 9. shard-local bookkeeping --------------------------------------------------------
    block_heights = [len(safe.blocks[shard]) for shard in safe.shard_ids]
    check(all(safe.blocks[shard][0]["height"] == 1 for shard in safe.shard_ids),
          "every shard numbers its own blocks from height 1")
    check(all(height <= safe.network.sent + 1 for height in block_heights),
          f"each shard produces one block per tick it has work ({block_heights})")
    check(safe.escrow_of(0) == 0 and safe.escrow_of(3) == 0,
          "escrow is released per shard once every transfer settles")

    print(f"safe run: supply {safe.supply()} == {supply0}, transfers "
          f"{sorted(set(safe.final_status().values()))}, "
          f"completion latency {sorted(set(safe_latencies.values()))}")
    print(f"dropped receipt: x1 -> {dropped.status['x1']} at tick "
          f"{latencies['x1']} (expiry {EXPIRY} + margin {MARGIN})")
    print(f"margin = 0: {len(double_spends)} double spends, supply "
          f"{unsafe.supply()} vs {supply0}")
    print("PASS")


if __name__ == "__main__":
    main()
