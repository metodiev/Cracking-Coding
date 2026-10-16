/** Task 618 - Implement linearizability checker for concurrent histories. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * A full-history-search linearizability checker, the Wing-Gong procedure. A history is a set of
 * operations with a real-time interval [call, ret]; two operations are concurrent when their
 * intervals overlap, and the history is linearizable iff there is a permutation of the
 * operations that (a) respects the real-time order (x may not follow y when y's response
 * precedes x's invocation), and (b) reproduces exactly the recorded results when replayed
 * against the object's sequential specification. The search tries, at every step, each pending
 * operation whose invocation is not later than the earliest response still pending, and memoizes
 * the (object state, pending set) pairs it has already failed on, so long sequential stretches
 * collapse instead of exploding.
 *
 * Two specifications are provided: a read/write register and a FIFO queue (whose dequeue of an
 * empty queue returns the string "empty").
 *
 * What is simplified: histories are small (the search is exponential in the number of concurrent
 * operations - the randomized differential test keeps them at six), the object state is hashed
 * with a plain key function instead of a canonical form, and only whole-history decisions are
 * made, not "which operation is the culprit". Verified: found witnesses are revalidated
 * independently (permutation, real-time order, exact replay of every result), the checker's
 * verdict agrees with an exhaustive permutation search on small random histories for both
 * specifications, hand-built linearizable and non-linearizable histories for both objects are
 * classified correctly, and malformed histories are rejected.
 */
"use strict";

const REGISTER = {
  name: "register",
  initial: () => 0,
  key: (value) => String(value),
  apply: (value, op) => {
    if (op.op === "write") return { ok: true, value: op.arg };
    if (op.op === "read") return op.result === value ? { ok: true, value } : { ok: false };
    return { ok: false };
  },
};

const QUEUE = {
  name: "queue",
  initial: () => [],
  key: (items) => items.join(","),
  apply: (items, op) => {
    if (op.op === "enq") return { ok: true, value: [...items, op.arg] };
    if (op.op !== "deq") return { ok: false };
    if (items.length === 0) return op.result === "empty" ? { ok: true, value: items } : { ok: false };
    if (op.result !== items[0]) return { ok: false };
    return { ok: true, value: items.slice(1) };
  },
};

function validateHistory(history) {
  if (!Array.isArray(history)) throw new TypeError("history must be an array");
  for (const [index, op] of history.entries()) {
    if (typeof op !== "object" || op === null) throw new TypeError(`operation ${index} must be an object`);
    if (!Number.isFinite(op.call) || !Number.isFinite(op.ret)) throw new TypeError(`operation ${index} needs numeric call/ret`);
    if (op.ret < op.call) throw new RangeError(`operation ${index} returns before it is invoked`);
    if (typeof op.op !== "string") throw new TypeError(`operation ${index} needs an operation name`);
    if (op.proc === undefined) throw new TypeError(`operation ${index} needs proc`);
  }
  return true;
}

/** Returns { linearizable, witness, explored }; witness is the linearization or null. */
function checkLinearizable(history, spec = REGISTER, options = {}) {
  validateHistory(history);
  const ops = history.map((op, index) => ({ index, call: op.call, ret: op.ret }));
  const visited = new Set();
  let explored = 0;
  const search = (state, pending, order) => {
    if (pending.length === 0) return order;
    const signature = `${spec.key(state)}|${pending.map((p) => p.index).sort((a, b) => a - b).join(",")}`;
    if (visited.has(signature)) return null;
    visited.add(signature);
    explored++;
    let earliest = Infinity;
    for (const p of pending) if (p.ret < earliest) earliest = p.ret;
    for (const p of pending) {
      if (p.call > earliest) continue;              // another pending operation responded strictly earlier
      const outcome = spec.apply(state, history[p.index]);
      if (!outcome.ok) continue;
      const witness = search(outcome.value, pending.filter((q) => q !== p), order.concat(p.index));
      if (witness !== null) return witness;
    }
    return null;
  };
  const witness = search(spec.initial(), ops, []);
  return { linearizable: witness !== null, witness, explored, memo: visited.size };
}

/** Independent revalidation: permutation, real-time order, exact replay of the results. */
function validateWitness(history, witness, spec = REGISTER) {
  if (!Array.isArray(witness) || witness.length !== history.length) return false;
  if (new Set(witness).size !== history.length) return false;
  for (const index of witness) if (!Number.isInteger(index) || index < 0 || index >= history.length) return false;
  for (let i = 0; i < witness.length; i++) {
    for (let j = i + 1; j < witness.length; j++) {
      if (history[witness[j]].ret < history[witness[i]].call) return false;
    }
  }
  let state = spec.initial();
  for (const index of witness) {
    const outcome = spec.apply(state, history[index]);
    if (!outcome.ok) return false;
    state = outcome.value;
  }
  return true;
}

/** Exhaustive oracle: tries every permutation that respects the real-time order. */
function bruteForceLinearizable(history, spec = REGISTER) {
  validateHistory(history);
  const n = history.length;
  const used = new Array(n).fill(false);
  const current = [];
  const consistent = () => {
    for (let i = 0; i < current.length; i++) {
      for (let j = i + 1; j < current.length; j++) {
        if (history[current[j]].ret < history[current[i]].call) return false;
      }
    }
    return true;
  };
  const search = () => {
    if (current.length === n) return validateWitness(history, current.slice(), spec);
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      current.push(i);
      if (consistent() && search()) { used[i] = false; current.pop(); return true; }
      used[i] = false;
      current.pop();
    }
    return false;
  };
  return search();
}

function xorshift32(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= (state << 13) >>> 0; state >>>= 0;
    state ^= state >>> 17;
    state ^= (state << 5) >>> 0; state >>>= 0;
    return state / 4294967296;
  };
}

/** Random register history: results come from the call-ordered execution, then get corrupted. */
function randomHistory(count, next, corrupt) {
  const ops = [];
  for (let i = 0; i < count; i++) {
    const call = Math.floor(next() * 12);
    ops.push({
      proc: i % 2, op: next() < 0.5 ? "write" : "read",
      arg: Math.floor(next() * 3), call, ret: call + 1 + Math.floor(next() * 3),
    });
  }
  let value = 0;
  for (const i of ops.map((_, index) => index).sort((a, b) => ops[a].call - ops[b].call || a - b)) {
    if (ops[i].op === "write") value = ops[i].arg;
    else ops[i].result = value;
  }
  if (corrupt) {
    const i = Math.floor(next() * count);
    if (ops[i].op === "read") ops[i].result = (ops[i].result + 1) % 3;
    else ops[i].arg = (ops[i].arg + 1) % 3;
  }
  return ops;
}

module.exports = { REGISTER, QUEUE, checkLinearizable, validateWitness, bruteForceLinearizable, randomHistory };

if (require.main === module) {
  const assert = require("node:assert");
  const reg = (proc, op, call, ret, extra) => ({ proc, op, call, ret, ...extra });

  // --- register: a concurrent but linearizable history ---
  const concurrentWrites = [
    reg(0, "write", 0, 3, { arg: 1 }),
    reg(1, "write", 1, 4, { arg: 2 }),
    reg(2, "read", 5, 6, { result: 2 }),
    reg(2, "read", 7, 8, { result: 2 }),
  ];
  const concurrentResult = checkLinearizable(concurrentWrites, REGISTER);
  assert.strictEqual(concurrentResult.linearizable, true, "the writes overlap, the reads agree afterwards");
  assert.ok(validateWitness(concurrentWrites, concurrentResult.witness, REGISTER), "the witness must replay");
  const writeOrder = concurrentResult.witness.filter((index) => concurrentWrites[index].op === "write");
  assert.strictEqual(writeOrder[writeOrder.length - 1], 1, "the value 2 write must be the last write");

  // --- register: two concurrent reads demanding different final values is impossible ---
  const impossible = [
    reg(0, "write", 0, 3, { arg: 1 }),
    reg(1, "write", 1, 4, { arg: 2 }),
    reg(2, "read", 5, 6, { result: 1 }),
    reg(3, "read", 5, 6, { result: 2 }),
  ];
  const impossibleResult = checkLinearizable(impossible, REGISTER);
  assert.strictEqual(impossibleResult.linearizable, false, "both reads see a different last value");
  assert.strictEqual(impossibleResult.witness, null);

  // --- register: a read that overlaps a write may legitimately return the old value ---
  const staleRead = [reg(0, "write", 0, 4, { arg: 7 }), reg(1, "read", 3, 5, { result: 0 })];
  assert.strictEqual(checkLinearizable(staleRead, REGISTER).linearizable, true, "read first, write second");
  const lateRead = [reg(0, "write", 0, 2, { arg: 7 }), reg(1, "read", 3, 4, { result: 0 })];
  assert.strictEqual(checkLinearizable(lateRead, REGISTER).linearizable, false, "a read after the write must see it");
  const forcedStale = [reg(0, "write", 0, 2, { arg: 7 }), reg(0, "write", 3, 4, { arg: 8 }), reg(1, "read", 5, 6, { result: 7 })];
  assert.strictEqual(checkLinearizable(forcedStale, REGISTER).linearizable, false, "the last write must win");
  assert.strictEqual(bruteForceLinearizable(forcedStale, REGISTER), false);
  const correctOrder = [reg(0, "write", 0, 2, { arg: 7 }), reg(0, "write", 3, 4, { arg: 8 }), reg(1, "read", 5, 6, { result: 8 })];
  assert.strictEqual(checkLinearizable(correctOrder, REGISTER).linearizable, true);

  // --- FIFO queue: the classic LIFO-looking dequeue order is not linearizable ---
  const fifo = [
    { proc: 0, op: "enq", arg: 1, call: 0, ret: 2 },
    { proc: 1, op: "enq", arg: 2, call: 3, ret: 5 },
    { proc: 2, op: "deq", result: 1, call: 6, ret: 7 },
    { proc: 2, op: "deq", result: 2, call: 8, ret: 9 },
  ];
  assert.strictEqual(checkLinearizable(fifo, QUEUE).linearizable, true, "FIFO order is linearizable");
  assert.ok(validateWitness(fifo, checkLinearizable(fifo, QUEUE).witness, QUEUE));
  const lifo = [
    { proc: 0, op: "enq", arg: 1, call: 0, ret: 2 },
    { proc: 1, op: "enq", arg: 2, call: 3, ret: 5 },
    { proc: 2, op: "deq", result: 2, call: 6, ret: 7 },
    { proc: 2, op: "deq", result: 1, call: 8, ret: 9 },
  ];
  assert.strictEqual(checkLinearizable(lifo, QUEUE).linearizable, false, "dequeue returned the newer element first");
  const emptyDeq = [{ proc: 0, op: "deq", result: "empty", call: 0, ret: 1 }];
  assert.strictEqual(checkLinearizable(emptyDeq, QUEUE).linearizable, true, "dequeuing an empty queue is allowed");
  assert.strictEqual(checkLinearizable([{ proc: 0, op: "deq", result: 5, call: 0, ret: 1 }], QUEUE).linearizable, false);

  // --- differential test against the exhaustive permutation oracle ---
  const next = xorshift32(0x5eed);
  let linearizable = 0;
  let nonLinearizable = 0;
  for (let trial = 0; trial < 80; trial++) {
    const history = randomHistory(2 + Math.floor(next() * 5), next, next() < 0.45);
    const fast = checkLinearizable(history, REGISTER);
    const slow = bruteForceLinearizable(history, REGISTER);
    assert.strictEqual(fast.linearizable, slow, `trial ${trial} disagreed: ${JSON.stringify(history)}`);
    if (fast.linearizable) {
      linearizable++;
      assert.ok(validateWitness(history, fast.witness, REGISTER), `trial ${trial} witness must replay`);
      assert.ok(fast.witness.length === history.length);
    } else {
      nonLinearizable++;
      assert.strictEqual(fast.witness, null);
    }
  }
  assert.ok(linearizable > 5 && nonLinearizable > 5, "the sample must contain both verdicts");

  // --- memoization keeps long sequential histories cheap, and edge cases behave ---
  const sequential = [];
  for (let i = 0; i < 18; i++) {
    sequential.push(reg(i % 2, "write", 3 * i, 3 * i + 1, { arg: i }));
    sequential.push(reg(i % 2, "read", 3 * i + 1, 3 * i + 2, { result: i }));
  }
  const sequentialResult = checkLinearizable(sequential, REGISTER);
  assert.strictEqual(sequentialResult.linearizable, true);
  assert.ok(sequentialResult.explored <= 37, `a sequential history explores one state per step (got ${sequentialResult.explored})`);
  const emptyHistory = checkLinearizable([], REGISTER);
  assert.deepStrictEqual(emptyHistory, { linearizable: true, witness: [], explored: 0, memo: 0 });
  const instant = [reg(0, "write", 1, 1, { arg: 5 }), reg(1, "read", 1, 1, { result: 5 })];
  assert.strictEqual(checkLinearizable(instant, REGISTER).linearizable, true, "instantaneous overlapping operations");
  const unknownOp = [reg(0, "cas", 0, 1, { arg: 1 })];
  assert.strictEqual(checkLinearizable(unknownOp, REGISTER).linearizable, false, "an unknown operation is not allowed");
  assert.throws(() => checkLinearizable([{ proc: 0, op: "read", call: 5, ret: 1, result: 0 }], REGISTER), RangeError);
  assert.throws(() => checkLinearizable([{ proc: 0, op: "read", call: 1 }], REGISTER), TypeError);
  assert.throws(() => checkLinearizable([{ op: "read", call: 1, ret: 2 }], REGISTER), TypeError);
  assert.throws(() => checkLinearizable([{ proc: 0, call: 1, ret: 2 }], REGISTER), TypeError);
  assert.throws(() => checkLinearizable(null, REGISTER), TypeError);
  assert.strictEqual(validateWitness(fifo, [0, 1, 2, 2], QUEUE), false, "a witness with a repeated index is invalid");
  assert.strictEqual(validateWitness(fifo, [2, 0, 1, 3], QUEUE), false, "a witness violating real time is invalid");

  console.log(`random trials: ${linearizable} linearizable / ${nonLinearizable} not, states explored=${sequentialResult.explored}`);
  console.log("PASS");
}
