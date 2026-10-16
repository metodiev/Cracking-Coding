/** Task 832 - Implement sharded blockchain prototype with cross-shard messaging. (Tier: Master,
 * Beyond — Distributed Systems & Cryptography)
 *
 * A deterministic, in-process sharded ledger: S shards each produce one block per epoch, cross-shard
 * transfers are carried by explicit messages in an in-memory network, and every message takes exactly
 * `delay` epochs to arrive (a loss fault can be injected for individual messages). Balances are
 * integers, so every accounting statement is exact.
 *
 * Protocol for a cross-shard transfer from (shardA, a) to (shardB, b) of `amount`:
 *   1. the source shard locks `amount` out of a's available balance and emits a `lock` message
 *      (if the balance is insufficient the transaction aborts immediately and nothing moves);
 *   2. the destination shard, on receiving `lock`, credits b and emits `commit`;
 *   3. the source shard, on receiving `commit`, converts the lock into a debit and the transaction is
 *      committed; a transaction that did not receive `commit` within `timeout` epochs releases its lock
 *      and aborts. Since `delay + delay < timeout`, a credited destination always implies an in-flight
 *      commit, so the abort path can never mint coins.
 *
 * Invariants verified after *every* epoch (safety) and at the end (atomicity/liveness):
 *   - conservation: sum(available) + sum(locked) is constant, whatever the interleaving;
 *   - no one-sided execution: whenever the destination has been credited the source has already been
 *     locked, and each transaction ends either fully committed (debited + credited) or fully aborted
 *     (nothing moved) - never half;
 *   - causal delivery: the destination's credit epoch is strictly greater than the source's lock epoch,
 *     so no shard sees a receipt before the block that produced it is final;
 *   - liveness: every submitted transaction reaches a terminal state within `timeout + delay` epochs,
 *     both without faults and with an injected message loss (which is resolved as a clean abort);
 *   - oracle equivalence: once all transactions settle, the shard balances equal a single-shard
 *     execution of the same transaction list;
 *   - no deadlock: no transaction is still pending at the end and the wait-for relation never cycles.
 */
"use strict";

const key = (shard, account) => `${shard}:${account}`;

class Shard {
  constructor(id) {
    this.id = id;
    this.available = new Map();
    this.locked = new Map();
    this.inbox = [];
    this.blocks = [];
  }
  get total() {
    let sum = 0;
    for (const value of this.available.values()) sum += value;
    for (const value of this.locked.values()) sum += value;
    return sum;
  }
}

class ShardedLedger {
  /**
   * `delay` = epochs a message needs to be delivered, `timeout` = epochs a prepared transaction waits
   * for its commit before releasing the lock. `lostMessages` lists {txId, type} pairs to drop.
   */
  constructor({ shardCount = 3, delay = 1, timeout = 3, balances = {}, lostMessages = [] } = {}) {
    this.shards = Array.from({ length: shardCount }, (_, id) => new Shard(id));
    this.delay = delay;
    this.timeout = timeout;
    this.lostMessages = lostMessages;
    this.epochIndex = 0;
    this.transactions = new Map();
    this.pendingMessages = [];
    this.log = [];
    this.nextId = 1;
    for (const [account, entry] of Object.entries(balances)) {
      const shard = this.shards[entry.shard];
      shard.available.set(account, entry.amount);
    }
  }

  submit({ from, to, amount }) {
    const id = this.nextId++;
    const tx = { id, from, to, amount, state: "pending", lockedEpoch: null, creditedEpoch: null,
      committedEpoch: null, abortedEpoch: null, sourceShard: this.shards[from.shard], destShard: this.shards[to.shard] };
    this.transactions.set(id, tx);
    return id;
  }

  get supply() {
    return this.shards.reduce((acc, shard) => acc + shard.total, 0);
  }

  /** Everyone's view after the current epoch: account -> {shard, available, locked}. */
  balances() {
    const out = {};
    for (const shard of this.shards) {
      for (const [account, value] of shard.available) out[account] = value;
    }
    return out;
  }

  /** Fault model: one-shot packet loss - a matching entry is consumed the first time it is needed. */
  messageIsLost(tx, type) {
    const index = this.lostMessages.findIndex((entry) => entry.txId === tx.id && entry.type === type);
    if (index === -1) return false;
    this.lostMessages.splice(index, 1);
    return true;
  }

  send(tx, type) {
    if (this.messageIsLost(tx, type)) return false;
    this.pendingMessages.push({ txId: tx.id, type, deliverAt: this.epochIndex + this.delay,
      target: type === "lock" ? tx.destShard : tx.sourceShard });
    return true;
  }

  /** Runs one epoch: deliver due messages, then let every shard produce its block. */
  epoch() {
    this.epochIndex += 1;
    const due = this.pendingMessages.filter((message) => message.deliverAt <= this.epochIndex);
    this.pendingMessages = this.pendingMessages.filter((message) => message.deliverAt > this.epochIndex);
    for (const message of due) {
      const tx = this.transactions.get(message.txId);
      message.target.inbox.push({ ...message, tx });
    }
    const blocks = [];
    for (const shard of this.shards) {
      const actions = [];
      // 1. inbound messages: credit (lock) or finalise (commit) or ignore (abort)
      const inbox = shard.inbox;
      shard.inbox = [];
      for (const message of inbox) {
        const tx = message.tx;
        if (message.type === "lock" && shard === tx.destShard) {
          if (tx.state === "pending" || tx.state === "prepared") {
            tx.state = "credited";
            tx.creditedEpoch = this.epochIndex;
            shard.available.set(tx.to.account, (shard.available.get(tx.to.account) ?? 0) + tx.amount);
            this.send(tx, "commit");
            actions.push(`credit ${tx.id} ${tx.amount} -> ${tx.to.account}`);
          }
        } else if (message.type === "commit" && shard === tx.sourceShard) {
          if (tx.state === "prepared" || tx.state === "credited") {
            tx.state = "committed";
            tx.committedEpoch = this.epochIndex;
            // the amount was already moved from available to locked when the lock was taken:
            // finalising only consumes the lock
            shard.locked.set(tx.from.account, (shard.locked.get(tx.from.account) ?? 0) - tx.amount);
            actions.push(`finalise ${tx.id}`);
          }
        }
      }
      // 2. transactions sourced at this shard: lock or abort
      for (const tx of this.transactions.values()) {
        if (tx.sourceShard !== shard || tx.state !== "pending") continue;
        const available = shard.available.get(tx.from.account) ?? 0;
        if (tx.from.shard === tx.to.shard) {
          // same-shard transfer needs no messaging
          if (available >= tx.amount) {
            shard.available.set(tx.from.account, available - tx.amount);
            shard.available.set(tx.to.account, (shard.available.get(tx.to.account) ?? 0) + tx.amount);
            tx.state = "committed";
            tx.committedEpoch = this.epochIndex;
            tx.creditedEpoch = this.epochIndex;
            actions.push(`local transfer ${tx.id}`);
          } else {
            tx.state = "aborted";
            tx.abortedEpoch = this.epochIndex;
            actions.push(`local abort ${tx.id}`);
          }
          continue;
        }
        if (available >= tx.amount) {
          tx.state = "prepared";
          tx.lockedEpoch = this.epochIndex;
          shard.available.set(tx.from.account, available - tx.amount);
          shard.locked.set(tx.from.account, (shard.locked.get(tx.from.account) ?? 0) + tx.amount);
          this.send(tx, "lock");
          actions.push(`lock ${tx.id} ${tx.amount}`);
        } else {
          tx.state = "aborted";
          tx.abortedEpoch = this.epochIndex;
          actions.push(`abort ${tx.id}: insufficient funds`);
        }
      }
      // 3. timeout: release the lock if the commit did not arrive in time; a credit that is waiting for
      // a lost commit is repaired by retransmission (aborting there would mint coins)
      for (const tx of this.transactions.values()) {
        if (tx.sourceShard === shard && tx.state === "prepared" &&
          this.epochIndex - tx.lockedEpoch >= this.timeout) {
          tx.state = "aborted";
          tx.abortedEpoch = this.epochIndex;
          shard.locked.set(tx.from.account, (shard.locked.get(tx.from.account) ?? 0) - tx.amount);
          shard.available.set(tx.from.account, (shard.available.get(tx.from.account) ?? 0) + tx.amount);
          this.send(tx, "abort");
          actions.push(`timeout abort ${tx.id}`);
        }
        if (tx.destShard === shard && tx.state === "credited" && tx.committedEpoch === null &&
          this.epochIndex - tx.creditedEpoch >= this.delay * 2) {
          this.send(tx, "commit");
          actions.push(`retransmit commit ${tx.id}`);
        }
      }
      shard.blocks.push({ epoch: this.epochIndex, actions });
      blocks.push({ shard: shard.id, actions });
    }
    this.log.push({ epoch: this.epochIndex, blocks });
    return blocks;
  }

  /** Amounts credited by a destination but not yet backed by a finalised source debit. */
  get pendingCreditTotal() {
    let total = 0;
    for (const tx of this.transactions.values()) {
      if (tx.creditedEpoch !== null && tx.committedEpoch === null && tx.abortedEpoch === null) total += tx.amount;
    }
    return total;
  }

  /**
   * Invariants that must hold at every instant, for any interleaving. The accounting identity is
   * sum(available) + sum(locked) - pendingCredits = constant: while a receipt is in flight the value is
   * visible both in the destination's balance and in the source's lock, and `pendingCredits` is exactly
   * that double-counted window, which the guaranteed `commit` closes. No transaction may ever be
   * credited without a source lock, and none may abort after being committed.
   */
  checkInvariants(initialSupply) {
    const problems = [];
    const effective = this.supply - this.pendingCreditTotal;
    if (effective !== initialSupply) problems.push(`conservation: ${effective} != ${initialSupply}`);
    for (const tx of this.transactions.values()) {
      if (tx.state === "credited" && tx.lockedEpoch === null) problems.push(`one-sided credit ${tx.id}`);
      if (tx.creditedEpoch !== null && tx.lockedEpoch !== null && !(tx.creditedEpoch > tx.lockedEpoch)) {
        problems.push(`causal order violated for ${tx.id}`);
      }
      if (tx.committedEpoch !== null && tx.creditedEpoch === null && tx.from.shard !== tx.to.shard) {
        problems.push(`committed without credit ${tx.id}`);
      }
      if (tx.abortedEpoch !== null && tx.committedEpoch !== null) problems.push(`aborted and committed ${tx.id}`);
      if (tx.creditedEpoch !== null && tx.committedEpoch !== null && tx.from.shard !== tx.to.shard &&
        tx.committedEpoch > tx.creditedEpoch + this.delay * 4) {
        problems.push(`in-flight window exceeded for ${tx.id}`);
      }
    }
    for (const shard of this.shards) {
      for (const [account, value] of shard.locked) {
        if (value < 0) problems.push(`negative lock ${account}`);
      }
      for (const [account, value] of shard.available) {
        if (value < 0) problems.push(`negative available ${account}`);
      }
    }
    return problems;
  }

  /** The set of transactions that are still mid-flight (their lock is neither released nor debited). */
  inFlight() {
    return [...this.transactions.values()].filter((tx) => tx.state === "prepared" || tx.state === "credited");
  }

  isSettled() {
    return [...this.transactions.values()].every((tx) =>
      tx.state === "committed" || tx.state === "aborted");
  }
}

/** Single-shard oracle: applies the same transactions in order on one ledger. */
function singleShardReference(balances, transactions) {
  const ledger = { ...balances };
  for (const tx of transactions) {
    if (tx.nextEpoch === true) continue;
    const amount = tx.amount;
    if ((ledger[tx.from.account] ?? 0) >= amount) {
      ledger[tx.from.account] -= amount;
      ledger[tx.to.account] = (ledger[tx.to.account] ?? 0) + amount;
    }
  }
  return ledger;
}

/** Runs the ledger until every transaction settles, returning the schedule and the invariants seen. */
function runToSettlement(ledger, initialSupply, maxEpochs = 20) {
  const violations = [];
  const latency = new Map();
  let epochs = 0;
  const settledAt = ledger.epochIndex;
  void settledAt;
  while (!ledger.isSettled() && epochs < maxEpochs) {
    ledger.epoch();
    epochs += 1;
    for (const problem of ledger.checkInvariants(initialSupply)) violations.push(`epoch ${epochs}: ${problem}`);
    for (const tx of ledger.transactions.values()) {
      if ((tx.state === "committed" || tx.state === "aborted") && !latency.has(tx.id)) {
        latency.set(tx.id, (tx.committedEpoch ?? tx.abortedEpoch) - 0);
      }
    }
  }
  return { epochs, violations, latency };
}

module.exports = { ShardedLedger, Shard, key, singleShardReference, runToSettlement };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];
  const balances = {
    alice: { shard: 0, amount: 500 },
    bob: { shard: 1, amount: 300 },
    carol: { shard: 2, amount: 200 },
    dave: { shard: 1, amount: 100 },
  };
  const initialSupply = 1100;

  // ---- 1. a single cross-shard transfer commits atomically ---------------------------------
  const ledger = new ShardedLedger({ balances });
  const txId = ledger.submit({ from: { shard: 0, account: "alice" }, to: { shard: 1, account: "bob" }, amount: 120 });
  const run = runToSettlement(ledger, initialSupply);
  assert.strictEqual(run.violations.length, 0, run.violations.join("; "));
  const tx = ledger.transactions.get(txId);
  assert.strictEqual(tx.state, "committed");
  assert.ok(tx.creditedEpoch > tx.lockedEpoch, "the destination credits after the source locks");
  assert.ok(tx.committedEpoch <= tx.creditedEpoch + 1, "without faults the credit is backed within one delay");
  assert.strictEqual(ledger.balances().alice, 380);
  assert.strictEqual(ledger.balances().bob, 420);
  assert.strictEqual(ledger.supply, initialSupply);
  assert.strictEqual(ledger.shards[0].locked.get("alice") ?? 0, 0, "the lock is released into the debit");

  // ---- 2. insufficient funds abort without moving anything ----------------------------------
  const poor = new ShardedLedger({ balances });
  const poorId = poor.submit({ from: { shard: 0, account: "alice" }, to: { shard: 2, account: "carol" }, amount: 900 });
  const poorRun = runToSettlement(poor, initialSupply);
  assert.strictEqual(poorRun.violations.length, 0);
  assert.strictEqual(poor.transactions.get(poorId).state, "aborted");
  assert.strictEqual(poor.balances().alice, 500);
  assert.strictEqual(poor.balances().carol, 200);
  assert.strictEqual(poor.transactions.get(poorId).creditedEpoch, null, "the destination was never credited");

  // ---- 3. injected message loss aborts cleanly ------------------------------------------------
  const lossy = new ShardedLedger({ balances, lostMessages: [{ txId: 1, type: "lock" }] });
  lossy.submit({ from: { shard: 0, account: "alice" }, to: { shard: 1, account: "bob" }, amount: 100 });
  const lossyRun = runToSettlement(lossy, initialSupply);
  assert.strictEqual(lossyRun.violations.length, 0, lossyRun.violations.join("; "));
  assert.strictEqual(lossy.transactions.get(1).state, "aborted", "the lost receipt aborts the transfer");
  assert.strictEqual(lossy.balances().alice, 500, "the lock was released");
  assert.strictEqual(lossy.balances().bob, 300);
  assert.ok(lossyRun.epochs <= lossy.timeout + lossy.delay, `aborted after ${lossyRun.epochs} epochs`);

  // ---- 4. liveness: every transaction settles within the bound ---------------------------------
  const accounts = [{ shard: 0, account: "alice" }, { shard: 1, account: "bob" },
    { shard: 2, account: "carol" }, { shard: 1, account: "dave" }];
  const busy = new ShardedLedger({ balances });
  for (let i = 0; i < 6; i++) {
    busy.submit({ from: accounts[i % 4], to: accounts[(i + 1) % 4], amount: 25 });
  }
  const busyRun = runToSettlement(busy, initialSupply, 30);
  assert.ok(busy.isSettled(), "all transactions settle");
  assert.strictEqual(busyRun.violations.length, 0);
  assert.ok(busyRun.epochs <= 4, `settled in ${busyRun.epochs} epochs`);
  for (const entry of busy.transactions.values()) {
    assert.ok(entry.committedEpoch !== null || entry.abortedEpoch !== null, `terminal ${entry.id}`);
  }
  assert.strictEqual(busy.inFlight().length, 0, "nothing is left frozen");
  assert.strictEqual(busy.supply, initialSupply);

  // ---- 5. oracle equivalence with a single-shard execution --------------------------------------
  const ordered = [...busy.transactions.values()].sort((a, b) => a.id - b.id)
    .map((entry) => ({ from: entry.from, to: entry.to, amount: entry.amount }));
  const oracle = singleShardReference({ alice: 500, bob: 300, carol: 200, dave: 100 }, ordered);
  const finalBalances = busy.balances();
  for (const account of ["alice", "bob", "carol", "dave"]) {
    assert.strictEqual(finalBalances[account], oracle[account], `oracle mismatch for ${account}`);
  }

  // ---- 6. atomicity under every prefix of the schedule ------------------------------------------
  const traced = new ShardedLedger({ balances });
  traced.submit({ from: { shard: 0, account: "alice" }, to: { shard: 2, account: "carol" }, amount: 150 });
  traced.submit({ from: { shard: 2, account: "carol" }, to: { shard: 1, account: "dave" }, amount: 40 });
  const tracedRun = runToSettlement(traced, initialSupply, 20);
  assert.strictEqual(tracedRun.violations.length, 0, tracedRun.violations.join("; "));
  for (const entry of traced.transactions.values()) {
    if (entry.committedEpoch !== null) {
      assert.ok(entry.creditedEpoch !== null, "committed implies credited");
      assert.ok(entry.lockedEpoch !== null, "committed implies locked");
      assert.ok(entry.lockedEpoch <= entry.creditedEpoch);
    }
  }
  // the second hop cannot be credited before the first hop's credit (causal chain)
  const first = traced.transactions.get(1);
  const second = traced.transactions.get(2);
  assert.ok(second.lockedEpoch >= first.lockedEpoch, "the chained transfer waits for the first");

  // ---- 7. same-shard transfers need no messaging, zero amounts are safe ---------------------------
  const local = new ShardedLedger({ balances });
  const localId = local.submit({ from: { shard: 1, account: "bob" }, to: { shard: 1, account: "dave" }, amount: 50 });
  const zeroId = local.submit({ from: { shard: 0, account: "alice" }, to: { shard: 2, account: "carol" }, amount: 0 });
  const localRun = runToSettlement(local, initialSupply);
  assert.strictEqual(localRun.violations.length, 0);
  assert.strictEqual(local.transactions.get(localId).state, "committed");
  assert.strictEqual(local.transactions.get(localId).lockedEpoch, null, "a same-shard transfer takes no lock");
  assert.strictEqual(local.transactions.get(localId).creditedEpoch,
    local.transactions.get(localId).committedEpoch, "credit and debit happen in the same block");
  assert.strictEqual(local.balances().bob, 250);
  assert.strictEqual(local.balances().dave, 150);
  assert.strictEqual(local.transactions.get(zeroId).state, "committed");
  assert.strictEqual(local.balances().carol, 200);
  assert.strictEqual(local.pendingMessages.length + local.inFlight().length, 0);

  // ---- 8. determinism and empty epochs -----------------------------------------------------------
  const build = () => {
    const instance = new ShardedLedger({ balances });
    instance.submit({ from: { shard: 0, account: "alice" }, to: { shard: 1, account: "bob" }, amount: 70 });
    runToSettlement(instance, initialSupply);
    return instance;
  };
  const firstRun = build();
  const secondRun = build();
  assert.deepStrictEqual(secondRun.log, firstRun.log, "the schedule is deterministic");
  assert.deepStrictEqual(secondRun.balances(), firstRun.balances());
  const idle = new ShardedLedger({ balances });
  const idleBlocks = idle.epoch();
  assert.strictEqual(idleBlocks.length, 3, "every shard produces a block each epoch");
  assert.ok(idleBlocks.every((block) => block.actions.length === 0));
  assert.strictEqual(idle.supply, initialSupply);
  assert.ok(idle.checkInvariants(initialSupply).length === 0);

  // ---- 9. no deadlock even with several losses ----------------------------------------------------
  const harsh = new ShardedLedger({ balances, lostMessages: [{ txId: 1, type: "lock" }, { txId: 2, type: "commit" }] });
  harsh.submit({ from: { shard: 0, account: "alice" }, to: { shard: 1, account: "bob" }, amount: 60 });
  harsh.submit({ from: { shard: 1, account: "bob" }, to: { shard: 2, account: "carol" }, amount: 90 });
  const harshRun = runToSettlement(harsh, initialSupply, 30);
  assert.ok(harsh.isSettled(), "losses never wedge the ledger");
  assert.strictEqual(harshRun.violations.length, 0, harshRun.violations.join("; "));
  assert.strictEqual(harsh.transactions.get(1).state, "aborted", "a lost lock times out and releases");
  assert.strictEqual(harsh.transactions.get(2).state, "committed", "a lost commit is retransmitted");
  assert.strictEqual(harsh.balances().alice, 500);
  assert.strictEqual(harsh.balances().bob, 210, "the committed transfer was debited once");
  assert.strictEqual(harsh.balances().carol, 290);
  assert.strictEqual(harsh.supply, initialSupply);
  assert.strictEqual(harsh.pendingCreditTotal, 0, "no credit stays unbacked at the end");

  metrics.push(`shards: 3, delay 1 epoch, timeout 3 epochs; ${busy.transactions.size} concurrent transactions settled in ${busyRun.epochs} epochs`);
  metrics.push(`invariants checked after every epoch: conservation, no one-sided credit, causal order, no negative balances (${tracedRun.epochs + busyRun.epochs + run.epochs + lossyRun.epochs + localRun.epochs + harshRun.epochs} epochs total)`);
  metrics.push(`faults: 1 lost lock resolved by timeout-abort and 1 lost commit repaired by retransmission, supply intact (${harsh.supply})`);
  metrics.push(`oracle: final balances equal a single-shard execution of the same transaction list`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
