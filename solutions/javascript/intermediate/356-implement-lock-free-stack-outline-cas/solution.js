/**
 * Task 356 - Implement lock-free stack outline (CAS). (Tier: Intermediate, CTCI Ch 15)
 *
 * JavaScript runs single-threaded, so this is the documented task-195 simulation: the CAS
 * on the head pointer is a plain compare + store on a holder, trivially atomic with respect
 * to the event loop - the same indivisible-instruction abstraction the threaded languages
 * get from hardware CAS. Four virtual pusher coroutines interleave at every await, and
 * because each push is atomic no node can be lost; the final snapshot must hold exactly
 * 1000 values. The ABA hazard of pointer CAS is documented but cannot occur here: nodes
 * are never recycled during a run, and any lost CAS retries with a freshly read head.
 * The empty-stack sentinel is null (pushed values are numbers and can never be null).
 */
"use strict";

const THREADS = 4;
const PUSHES_PER_THREAD = 250;

class Node {
  constructor(value, next = null) {
    this.value = value;
    this.next = next;
  }
}

class LockFreeStack {
  constructor() {
    this.head = null;
    this.casAttempts = 0;
    this.casFailures = 0;
  }

  compareAndSwapHead(expected, newHead) {
    this.casAttempts += 1;
    if (this.head === expected) {
      this.head = newHead;
      return true;
    }
    this.casFailures += 1;
    return false;
  }

  push(value) {
    let retries = 0;
    for (;;) {
      const current = this.head;
      const node = new Node(value, current);
      if (this.compareAndSwapHead(current, node)) return retries;
      retries += 1;
    }
  }

  pop() {
    for (;;) {
      const current = this.head;
      if (current === null) return null;             // documented empty-stack sentinel
      if (this.compareAndSwapHead(current, current.next)) return current.value;
    }
  }

  isEmpty() {
    return this.head === null;
  }

  snapshotValues() {
    const values = [];
    for (let node = this.head; node !== null; node = node.next) values.push(node.value);
    return values;
  }
}

async function main() {
  const assert = require("node:assert");

  // Single-threaded script: LIFO order and the empty sentinel.
  const stack = new LockFreeStack();
  assert.strictEqual(stack.pop(), null, "pop on an empty stack returns the sentinel");
  for (let value = 1; value <= 5; value++) stack.push(value);
  const drained = [];
  for (let i = 0; i < 5; i++) drained.push(stack.pop());
  assert.deepStrictEqual(drained, [5, 4, 3, 2, 1], "LIFO order");
  assert.strictEqual(stack.pop(), null, "empty again after draining");
  assert.strictEqual(stack.casFailures, 0, "single-threaded script never loses a CAS");

  // Simulated concurrent pushes: no node is lost.
  const shared = new LockFreeStack();
  const total = THREADS * PUSHES_PER_THREAD;
  const pushers = [];
  for (let threadIndex = 0; threadIndex < THREADS; threadIndex++) {
    pushers.push((async () => {
      for (let offset = 0; offset < PUSHES_PER_THREAD; offset++) {
        shared.push(threadIndex * PUSHES_PER_THREAD + offset);
        if (offset % 32 === 31) await Promise.resolve();       // interleave with other pushers
      }
    })());
  }
  await Promise.all(pushers);

  const values = shared.snapshotValues();
  assert.strictEqual(values.length, total, "no node lost under concurrent pushes");
  assert.deepStrictEqual([...values].sort((a, b) => a - b),
    Array.from({ length: total }, (_, i) => i), "every pushed value present exactly once");

  // Simulated concurrent pops: no node is lost or duplicated.
  const poppedPerThread = new Array(THREADS).fill(null).map(() => []);
  const poppers = [];
  for (let threadIndex = 0; threadIndex < THREADS; threadIndex++) {
    poppers.push((async () => {
      for (;;) {
        const value = shared.pop();
        if (value === null) break;
        poppedPerThread[threadIndex].push(value);
        if (poppedPerThread[threadIndex].length % 32 === 0) await Promise.resolve();
      }
    })());
  }
  await Promise.all(poppers);

  const allPopped = poppedPerThread.flat();
  assert.strictEqual(allPopped.length, total, "no node lost under concurrent pops");
  assert.deepStrictEqual([...allPopped].sort((a, b) => a - b),
    Array.from({ length: total }, (_, i) => i), "every pushed value popped exactly once");
  assert.strictEqual(shared.pop(), null, "stack is empty again");
  assert.strictEqual(shared.isEmpty(), true, "empty state after the concurrent phase");

  // Exact CAS accounting: every successful push/pop is one attempt, every retry one
  // failure. The poppers that stopped with the sentinel made no CAS attempt at all.
  assert.ok(shared.casFailures >= 0, "CAS failures are non-negative (they may be 0)");
  assert.strictEqual(shared.casAttempts, 2 * total + shared.casFailures,
    "attempts == successful operations + failures");
}

module.exports = { LockFreeStack, Node };

if (require.main === module) {
  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
