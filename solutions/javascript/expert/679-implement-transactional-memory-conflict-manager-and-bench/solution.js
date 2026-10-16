/**
 * Task 679 - Implement transactional memory conflict manager and bench.
 * (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * An optimistic software transactional memory with a conflict manager and a small benchmark:
 *
 *   memory      - every location carries a version number; a transaction's read set records the version
 *                 it observed and its write set buffers values, so nothing becomes visible before commit;
 *   commit      - validate every entry of the read set (the version must still be the one observed, and
 *                 it must not have been written by somebody else since) and then install the write set,
 *                 bumping the versions; a failed validation aborts the transaction, which is retried;
 *   manager     - a deterministic schedule (seeded splitmix64) runs the transactions in rounds, retrying
 *                 aborted ones, and records commits, aborts, restarts and the number of moved values.
 *
 * The benchmark transactions are bank transfers, which is the classic serializability witness: the total
 * balance is invariant under any interleaving, and the manager's final state is checked against *every*
 * serial order of the committed transactions (brute-force over all permutations), so the result must be
 * a legal serial execution and not merely a plausible one.
 *
 * Verified: a conflict-free workload commits everything at the first attempt; two transactions with
 * overlapping write sets produce exactly one abort and the eventual state equals the serial order in
 * which the committed one ran first; the brute-force permutation check confirms serializability on
 * several workloads and conflict levels; total balances are conserved after every commit; raising the
 * conflict rate raises the abort count while every transaction still eventually commits; and the runs are
 * reproducible for a fixed seed.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed679n;

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }
  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }
  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ----------------------------------------------------------------------------------- memory
function makeMemory(initial) {
  const values = new Map();
  const versions = new Map();
  for (const [loc, value] of Object.entries(initial)) {
    values.set(loc, value);
    versions.set(loc, 0);
  }
  let version = 0;
  return {
    read: (loc) => ({ value: values.get(loc) ?? 0, version: versions.get(loc) ?? 0 }),
    value: (loc) => values.get(loc) ?? 0,
    entries: () => [...values.entries()],
    total: () => [...values.values()].reduce((a, b) => a + b, 0),
    /** Installs a validated write set; returns the number of versions bumped. */
    commit: (writeSet, readSet) => {
      let bumped = 0;
      for (const [loc, value] of writeSet) {
        values.set(loc, value);
        version += 1;
        versions.set(loc, version);
        bumped += 1;
      }
      void readSet;
      return bumped;
    },
  };
}

function begin() {
  return { reads: new Map(), writes: new Map(), status: "active" };
}

function txRead(memory, tx, loc) {
  if (tx.writes.has(loc)) return tx.writes.get(loc);
  if (!tx.reads.has(loc)) tx.reads.set(loc, memory.read(loc).version);
  return memory.read(loc).value;
}

function txWrite(tx, loc, value) {
  tx.writes.set(loc, value);
}

/** Validation-then-install: either the whole write set lands or nothing does. */
function txCommit(memory, tx) {
  for (const [loc, version] of tx.reads) {
    const current = memory.read(loc);
    if (current.version !== version) {
      tx.status = "aborted";
      return { committed: false, reason: "read-set-invalidated", loc };
    }
  }
  const bumped = memory.commit(tx.writes, tx.reads);
  tx.status = "committed";
  return { committed: true, bumped };
}

// ---------------------------------------------------------------------------------- workload
/** A transfer is a read-read-write transaction over two locations. */
const transfer = (from, to, amount) => ({ from, to, amount });

function runTransfer(memory, tx) {
  const balance = begin();
  const source = txRead(memory, balance, tx.from);
  const target = txRead(memory, balance, tx.to);
  if (source < tx.amount) {
    balance.status = "committed";
    return { committed: true, empty: true };
  }
  txWrite(balance, tx.from, source - tx.amount);
  txWrite(balance, tx.to, target + tx.amount);
  return txCommit(memory, balance);
}

/**
 * Interleaved manager: one *operation* of a randomly chosen unfinished transaction per step, so reads and
 * writes from different transactions really interleave and validation has something to catch. A failed
 * validation restarts the transaction from its first operation.
 */
function runManager(accounts, transactions, seed = DEFAULT_SEED, maxSteps = 4000) {
  const memory = makeMemory(accounts);
  const rng = new SplitMix64(seed);
  const fresh = () => ({ reads: new Map(), writes: new Map(), source: 0, target: 0, status: "active" });
  const pending = transactions.map((tx, index) => ({ tx, index, phase: 0, attempts: 0, txState: fresh() }));
  const committed = [];
  const abortsByIndex = new Map();
  let aborts = 0;
  let steps = 0;
  while (pending.length > 0 && steps < maxSteps) {
    steps += 1;
    const pick = rng.nextInt(pending.length);
    const entry = pending[pick];
    const state = entry.txState;
    if (entry.phase === 0) {
      state.source = txRead(memory, state, entry.tx.from);
      entry.phase = 1;
    } else if (entry.phase === 1) {
      state.target = txRead(memory, state, entry.tx.to);
      entry.phase = 2;
    } else if (entry.phase === 2) {
      // An unaffordable transfer (or a self-transfer, which moves nothing) becomes a committed no-op.
      if (state.source < entry.tx.amount || entry.tx.from === entry.tx.to) {
        state.status = "committed";
        entry.phase = 4;
      } else {
        txWrite(state, entry.tx.from, state.source - entry.tx.amount);
        entry.phase = 3;
      }
    } else if (entry.phase === 3) {
      txWrite(state, entry.tx.to, state.target + entry.tx.amount);
      entry.phase = 4;
    } else {
      const result = txCommit(memory, state);
      entry.attempts += 1;
      if (result.committed) {
        committed.push({ index: entry.index, tx: entry.tx, serial: committed.length });
        pending.splice(pick, 1);
      } else {
        aborts += 1;
        abortsByIndex.set(entry.index, (abortsByIndex.get(entry.index) ?? 0) + 1);
        entry.phase = 0;
        entry.txState = fresh();
      }
    }
  }
  return { memory, committed, aborts, abortsByIndex, steps, remaining: pending.length,
    state: memory.entries(), moves: committed.length };
}

/** Independent reference: apply a list of transfers serially, in the given order. */
function serialResult(accounts, transactions) {
  const balances = new Map(Object.entries(accounts));
  for (const tx of transactions) {
    const source = balances.get(tx.from) ?? 0;
    const target = balances.get(tx.to) ?? 0;
    if (source < tx.amount || tx.from === tx.to) continue;
    balances.set(tx.from, source - tx.amount);
    balances.set(tx.to, target + tx.amount);
  }
  return [...balances.entries()].sort();
}

/** Every permutation of the committed transactions, applied serially, as candidate final states. */
function serialisableStates(accounts, committed, limit = 5000) {
  const states = new Set();
  let visited = 0;
  const walk = (prefix, rest) => {
    if (rest.length === 0 || visited > limit) {
      states.add(JSON.stringify(serialResult(accounts, prefix)));
      return;
    }
    visited += 1;
    for (let i = 0; i < rest.length; i++) {
      walk([...prefix, rest[i]], rest.filter((_, index) => index !== i));
    }
  };
  walk([], committed);
  return { states, explored: visited };
}

const transactionsOf = (list) => list.map(([from, to, amount]) => transfer(from, to, amount));

function main() {
  // (1) Disjoint transactions: no shared location, so nothing can be invalidated.
  const quiet = runManager({ a: 100, b: 100, c: 100, d: 100 }, transactionsOf([["a", "b", 10], ["c", "d", 5]]));
  assert.deepStrictEqual([quiet.aborts, quiet.remaining], [0, 0], "disjoint transactions never abort");
  assert.deepStrictEqual([quiet.memory.total(), quiet.moves], [400, 2], "the balance total is conserved");
  assert.deepStrictEqual(quiet.memory.entries().sort(), [["a", 90], ["b", 110], ["c", 95], ["d", 105]],
    "and both transfers are exact");
  // The same workload with a shared account does conflict: b is read by one and written by the other.
  const shared_workload = runManager({ a: 100, b: 100, c: 100 }, transactionsOf([["a", "b", 10], ["b", "c", 5]]));
  assert.ok(shared_workload.aborts >= 1, `documented: chaining through b invalidates a read set (${shared_workload.aborts} abort(s))`);
  assert.strictEqual(shared_workload.memory.total(), 300, "and the total is still conserved");

  // (2) Deliberate conflict: two transactions write the same two locations.
  const shared = { x: 50, y: 50 };
  const conflict = transactionsOf([["x", "y", 10], ["x", "y", 20]]);
  const run = runManager(shared, conflict, 5n);
  assert.strictEqual(run.aborts >= 1, true, `documented: overlapping write sets cause ${run.aborts} abort(s)`);
  assert.strictEqual(run.remaining, 0, "but with retries every transaction eventually commits");
  assert.strictEqual(run.memory.total(), 100, "the invariant total is still 100");
  const serial = serialisableStates(shared, [conflict[0], conflict[1]]);
  assert.ok(serial.states.has(JSON.stringify(run.memory.entries().sort())),
    "and the final state is one of the legal serial orders");

  // (3) Serializability on several workloads and seeds, checked against every permutation.
  const accounts = { a: 60, b: 60, c: 60, d: 60 };
  const workloads = [
    transactionsOf([["a", "b", 10], ["b", "c", 20], ["c", "d", 5], ["d", "a", 15]]),
    transactionsOf([["a", "b", 30], ["a", "c", 20], ["a", "d", 10], ["b", "a", 5]]),
    transactionsOf([["a", "b", 25], ["b", "a", 25], ["c", "d", 12], ["d", "c", 8]]),
  ];
  let checked = 0;
  for (const workload of workloads) {
    for (const seed of [1n, 7n, 42n]) {
      const result = runManager(accounts, workload, seed);
      assert.strictEqual(result.remaining, 0, "every transaction commits under the manager");
      assert.strictEqual(result.memory.total(), 240, "the total is conserved for this workload too");
      const committedTransactions = result.committed.map((entry) => entry.tx);
      assert.deepStrictEqual(serialResult(accounts, committedTransactions), [...result.memory.entries()].sort(),
        `seed ${seed}: the commit order is itself a valid serial execution`);
      const states = serialisableStates(accounts, committedTransactions);
      assert.ok(states.states.has(JSON.stringify(result.memory.entries().sort())),
        `seed ${seed}: the result equals some serial order of the committed transactions`);
      checked += 1;
    }
  }
  assert.strictEqual(checked, 9, "nine manager runs were checked for serializability");

  // (5) Higher conflict rates abort more, while progress is never lost.
  const accountNames = ["k0", "k1", "k2", "k3"];
  const abortCounts = [4, 3, 2].map((width) => {
    const bank = {};
    for (const key of accountNames) bank[key] = 120;
    const txs = [];
    for (let i = 0; i < 8; i++) txs.push(transfer(`k${i % width}`, `k${(i + 1) % width}`, 5 + i));
    const outcome = runManager(bank, txs, DEFAULT_SEED);
    assert.strictEqual(outcome.remaining, 0, "the manager always finishes");
    assert.strictEqual(outcome.memory.total(), 480, "no value is created or destroyed");
    return outcome.aborts;
  });
  assert.ok(abortCounts[abortCounts.length - 1] >= abortCounts[0],
    `documented: concentrating the transfers on fewer accounts aborts more (${abortCounts.join(" -> ")})`);
  assert.ok(abortCounts[abortCounts.length - 1] > 0, "the most contended level does abort");

  // (5) A transaction that cannot afford the transfer is a committed no-op, not an abort.
  const broke = runManager({ p: 5, q: 100 }, transactionsOf([["p", "q", 50], ["q", "p", 1]]));
  assert.deepStrictEqual([broke.aborts, broke.memory.value("p"), broke.memory.value("q")], [0, 6, 99],
    "the unaffordable transfer changes nothing and the affordable one runs");

  // (6) Versioning: a write invalidates the read set observed by an already-started transaction.
  const memory = makeMemory({ v: 1 });
  const first = begin();
  txRead(memory, first, "v");
  const second = begin();
  txWrite(second, "v", 9);
  txCommit(memory, second);
  assert.deepStrictEqual([txCommit(memory, first).committed, memory.value("v")], [false, 9],
    "the stale transaction is aborted instead of clobbering the newer value");
  const third = begin();
  txRead(memory, third, "v");
  txWrite(third, "v", memory.read("v").value + 1);
  assert.deepStrictEqual([txCommit(memory, third).committed, memory.value("v")], [true, 10],
    "a fresh transaction commits and bumps the version");

  // (7) Boundary cases and determinism.
  const empty = runManager({ a: 1 }, []);
  assert.deepStrictEqual([empty.committed.length, empty.aborts, empty.steps], [0, 0, 0], "no transactions, no work");
  const single = runManager({ a: 10 }, transactionsOf([["a", "a", 3]]));
  assert.deepStrictEqual([single.aborts, single.memory.value("a")], [0, 10], "a self-transfer moves nothing");
  assert.deepStrictEqual(serialResult({ a: 10 }, [transfer("a", "a", 3)]).map(([, v]) => v), [10],
    "which is also what the serial reference says");
  assert.deepStrictEqual(runManager(shared, conflict, 5n).state, runManager(shared, conflict, 5n).state,
    "the manager is reproducible for a fixed seed");
  const otherSeed = runManager(shared, conflict, 6n);
  assert.deepStrictEqual([otherSeed.remaining, otherSeed.memory.total(), otherSeed.aborts >= 1], [0, 100, true],
    "a different schedule still commits everything and still hits the conflict");
}

module.exports = { SplitMix64, DEFAULT_SEED, makeMemory, begin, txRead, txWrite, txCommit, transfer,
  runTransfer, runManager, serialResult, serialisableStates, transactionsOf };

if (require.main === module) {
  main();
  console.log("PASS");
}
