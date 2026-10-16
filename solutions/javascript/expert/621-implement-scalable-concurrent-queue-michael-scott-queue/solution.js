/** Task 621 - Implement scalable concurrent queue (Michael-Scott queue). (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * The Michael-Scott non-blocking queue, simulated on a single thread. There are no threads and no
 * timers: every field is an `Atomic` with load and compareAndSwap, each operation is a generator
 * that yields once per atomic access, and a *schedule* of thread ids decides who takes the next
 * step. That makes the interesting interleavings reproducible - pausing an enqueue between reading
 * `tail.next` and its CAS, letting another thread enqueue, then resuming makes the first CAS fail
 * exactly as it would on hardware (the "enq 1 jumped behind enq 2" case is checked below) - and
 * pausing after a successful link but before the tail CAS exercises the helping branch that
 * advances a lagging tail. ABA cannot happen because nodes are created fresh and never reused,
 * which is the standard reason this queue needs no tagged pointer.
 *
 * Invariants validated after *every single step*: the tail is reachable from the head, link orders
 * along the chain strictly increase (that is the FIFO order, and dequeues must follow it), the
 * head is the first node, and no cycle exists; at rest the tail lags the last node by at most one
 * and an empty queue is exactly one dummy node. On top of that: conservation (dequeued + remaining
 * == linked), no duplicate dequeue, and an empty dequeue only when the dummy's next is null.
 *
 * What is simplified: one thread runs at a time (so a CAS is trivially atomic) - this is a faithful
 * interleaving simulation of the algorithm rather than a kernel test - and there is no memory
 * reclamation story (hazard pointers / epochs are a different task).
 */
"use strict";

class Atomic {
  constructor(value) {
    this.value = value;
    this.casAttempts = 0;
    this.casFailures = 0;
  }

  load() { return this.value; }

  compareAndSwap(expected, replacement, queue) {
    this.casAttempts++;
    if (this.value === expected) {
      this.value = replacement;
      return true;
    }
    this.casFailures++;
    if (queue !== undefined) queue.casFailures++;
    return false;
  }
}

class MsQueue {
  constructor() {
    this.dummy = { value: undefined, next: new Atomic(null), linkOrder: 0 };
    this.head = new Atomic(this.dummy);
    this.tail = new Atomic(this.dummy);
    this.links = 0;
    this.casFailures = 0;
    this.helpingAdvances = 0;
    this.dequeued = [];
    this.steps = 0;
  }

  chain() {
    const nodes = [];
    for (let node = this.head.load(); node !== null; node = node.next.load()) nodes.push(node);
    return nodes;
  }

  isEmpty() { return this.head.load().next.load() === null; }

  contents() { return this.chain().slice(1).map((node) => node.value); }

  /** Structural invariants; throws with the broken invariant instead of returning false. */
  validate(strict = false) {
    const nodes = this.chain();
    if (nodes[0] !== this.head.load()) throw new Error("the head is not the first node of the chain");
    const seen = new Set();
    for (let i = 0; i < nodes.length; i++) {
      if (seen.has(nodes[i])) throw new Error("the chain contains a cycle");
      seen.add(nodes[i]);
      if (i > 0 && !(nodes[i - 1].linkOrder < nodes[i].linkOrder)) throw new Error("link order is not increasing (FIFO broken)");
      if (i + 1 < nodes.length && nodes[i].next.load() !== nodes[i + 1]) throw new Error("broken next pointer");
    }
    const tail = this.tail.load();
    const index = nodes.indexOf(tail);
    if (index === -1) throw new Error("the tail is not reachable from the head");
    if (strict) {
      if (nodes.length - 1 - index > 1) throw new Error("the tail lags the last node by more than one");
      if (this.isEmpty() && (nodes.length !== 1 || tail !== this.head.load())) throw new Error("an empty queue must be one dummy node");
    }
    return true;
  }
}

/** enqueue(x): link behind the tail, retry on a stale read, help a lagging tail. */
function* enqueueOp(queue, value) {
  const node = { value, next: new Atomic(null), linkOrder: 0 };
  let tail = queue.tail.load();
  yield "read-tail";
  while (true) {
    const next = tail.next.load();
    yield "read-tail-next";
    if (tail !== queue.tail.load()) { tail = queue.tail.load(); yield "reread-tail"; continue; }
    yield "check";
    if (next === null) {
      if (tail.next.compareAndSwap(null, node, queue)) {
        node.linkOrder = ++queue.links;             // assigned at the linking moment, before any yield
        yield "link";
        break;
      }
      yield "link-failed";
    } else if (queue.tail.compareAndSwap(tail, next, queue)) {
      queue.helpingAdvances++;
      yield "help-tail";
    } else {
      yield "help-tail-failed";
    }
    tail = queue.tail.load();
    yield "read-tail";
  }
  queue.tail.compareAndSwap(tail, node, queue);
  yield "advance-tail";
  return { kind: "enq", value, linkOrder: node.linkOrder };
}

/** dequeue(): the queue is empty only when the dummy's next is null, otherwise CAS the head. */
function* dequeueOp(queue) {
  while (true) {
    const head = queue.head.load();
    const tail = queue.tail.load();
    const next = head.next.load();
    yield "read";
    if (head !== queue.head.load()) continue;
    if (next === null) { yield "empty"; return { kind: "deq", empty: true }; }
    if (head === tail) {
      queue.tail.compareAndSwap(tail, next, queue);
      yield "help-tail";
      continue;
    }
    const value = next.value;
    yield "before-cas-head";
    if (queue.head.compareAndSwap(head, next, queue)) {
      queue.dequeued.push(next);
      yield "cas-head";
      return { kind: "deq", value, linkOrder: next.linkOrder };
    }
    yield "cas-head-failed";
  }
}

/** Advances one thread by one atomic step and validates the structure at every step boundary. */
function runSchedule(queue, plans, order, options = {}) {
  const limit = options.limit ?? 200000;
  const threads = plans.map((plan) => ({ plan, index: 0, current: null, results: [] }));
  const isActive = (thread) => thread.current !== null || thread.index < thread.plan.length;
  const active = () => threads.some(isActive);
  let steps = 0;
  const timeline = [];
  while (active()) {
    if (steps > limit) throw new Error("schedule did not converge");
    const wanted = order(steps, threads);
    const id = isActive(threads[wanted]) ? wanted : threads.findIndex(isActive);
    const thread = threads[id];
    if (thread.current === null) {
      const operation = thread.plan[thread.index];
      if (operation.kind === "enq") thread.current = enqueueOp(queue, operation.value);
      else if (operation.kind === "deq") thread.current = dequeueOp(queue);
      else throw new Error(`unknown operation ${operation.kind}`);
    }
    const outcome = thread.current.next();
    queue.steps = ++steps;
    if (outcome.done) {
      thread.results.push(outcome.value);
      timeline.push(outcome.value);
      thread.index++;
      thread.current = null;
    }
    queue.validate();
  }
  queue.validate(true);                              // at rest the tail has caught up
  return { steps, results: threads.map((thread) => thread.results), timeline };
}

const roundRobin = (step, threads) => (step % threads.length);

/** Runs the phases in order: [{ thread, steps }], Infinity meaning "until that thread finishes". */
const phaseOrder = (phases) => {
  let index = 0;
  let used = 0;
  return (step, threads) => {
    while (index < phases.length && used >= phases[index].steps) { index++; used = 0; }
    if (index >= phases.length) return 0;
    used++;
    return phases[index].thread;
  };
};

function xorshift32(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= (state << 13) >>> 0; state >>>= 0;
    state ^= state >>> 17;
    state ^= (state << 5) >>> 0; state >>>= 0;
    return state / 4294967296;
  };
}

module.exports = { MsQueue, Atomic, enqueueOp, dequeueOp, runSchedule, roundRobin, phaseOrder };

if (require.main === module) {
  const assert = require("node:assert");
  const dequeueOrder = (results) => results.filter((r) => r.kind === "deq" && !r.empty).map((r) => r.linkOrder);
  const checkFifo = (results) => {
    const orders = dequeueOrder(results);
    for (let i = 1; i < orders.length; i++) assert.ok(orders[i - 1] < orders[i], "dequeues must follow link order");
    return orders.length;
  };

  // --- single thread: plain FIFO behaviour ---
  const single = new MsQueue();
  const singleRun = runSchedule(single, [[
    { kind: "enq", value: 1 }, { kind: "enq", value: 2 }, { kind: "enq", value: 3 },
    { kind: "deq" }, { kind: "deq" }, { kind: "enq", value: 4 }, { kind: "deq" }, { kind: "deq" }, { kind: "deq" },
  ]], roundRobin);
  checkFifo(singleRun.results.flat());
  assert.deepStrictEqual(singleRun.results[0].filter((r) => r.kind === "deq" && !r.empty).map((r) => r.value), [1, 2, 3, 4]);
  assert.strictEqual(singleRun.results[0][8].empty, true, "the last dequeue finds an empty queue");
  assert.deepStrictEqual(single.contents(), []);
  assert.strictEqual(single.casFailures, 0, "a single thread never loses a CAS");

  // --- two producers interleaved round-robin: conservation and link order ---
  const pair = new MsQueue();
  runSchedule(pair, [
    [{ kind: "enq", value: 10 }, { kind: "enq", value: 30 }],
    [{ kind: "enq", value: 20 }, { kind: "enq", value: 40 }],
  ], roundRobin);
  assert.strictEqual(pair.contents().length, 4);
  const drainRun = runSchedule(pair, [[{ kind: "deq" }, { kind: "deq" }, { kind: "deq" }, { kind: "deq" }]], roundRobin);
  const drained = drainRun.results[0].map((r) => r.value);
  assert.deepStrictEqual([...drained].sort((a, b) => a - b), [10, 20, 30, 40], "no value is lost or duplicated");
  assert.strictEqual(new Set(drained).size, 4);
  assert.strictEqual(pair.contents().length, 0, "the queue drained completely");

  // --- a paused enqueue loses its link CAS, retries and still links exactly once ---
  const raced = new MsQueue();
  const racedRun = runSchedule(raced, [
    [{ kind: "enq", value: 1 }, { kind: "enq", value: 3 }],
    [{ kind: "enq", value: 2 }],
  ], phaseOrder([{ thread: 0, steps: 3 }, { thread: 1, steps: 4 }, { thread: 0, steps: Infinity }]));
  assert.ok(raced.casFailures > 0, "the paused thread must lose a CAS");
  assert.strictEqual(raced.links, 3, "each node is linked exactly once");
  assert.deepStrictEqual([...raced.contents()].sort((a, b) => a - b), [1, 2, 3]);
  assert.deepStrictEqual(racedRun.results.map((r) => r.length), [2, 1]);

  // --- a paused dequeue cannot remove a node twice ---
  const doubling = new MsQueue();
  runSchedule(doubling, [[{ kind: "enq", value: 7 }, { kind: "enq", value: 8 }]], roundRobin);
  const doubleRun = runSchedule(doubling, [
    [{ kind: "deq" }, { kind: "deq" }],
    [{ kind: "deq" }],
  ], phaseOrder([{ thread: 0, steps: 2 }, { thread: 1, steps: Infinity }]));
  const values = doubleRun.results.flat().filter((r) => r.kind === "deq" && !r.empty).map((r) => r.value);
  assert.strictEqual(new Set(values).size, values.length, "no node may be dequeued twice");
  assert.deepStrictEqual([...values].sort((a, b) => a - b), [7, 8]);
  assert.ok(doubling.casFailures > 0, "the competing dequeues must lose a CAS");

  // --- a thread paused after linking is helped by the next thread (tail advance) ---
  const helping = new MsQueue();
  const helpedRun = runSchedule(helping, [
    [{ kind: "enq", value: 1 }],
    [{ kind: "enq", value: 2 }],
  ], phaseOrder([{ thread: 1, steps: 4 }, { thread: 0, steps: Infinity }]));
  assert.ok(helping.helpingAdvances >= 1, "the other thread must advance the lagging tail");
  assert.deepStrictEqual([...helping.contents()].sort((a, b) => a - b), [1, 2]);
  assert.deepStrictEqual(helpedRun.results.flat().map((r) => r.value), [1, 2]);

  // --- randomised interleavings, validated after every single step ---
  const next = xorshift32(0xbeef);
  for (let trial = 0; trial < 6; trial++) {
    const queue = new MsQueue();
    const plans = [];
    let value = 100 * trial;
    for (let thread = 0; thread < 4; thread++) {
      const plan = [];
      for (let i = 0; i < 8; i++) {
        if (next() < 0.6) plan.push({ kind: "enq", value: ++value });
        else plan.push({ kind: "deq" });
      }
      plans.push(plan);
    }
    const run = runSchedule(queue, plans, (step, threads) => (Math.floor(next() * threads.length)));
    // removals are recorded at the successful head CAS, so queue.dequeued is the linearization order
    const removedOrders = queue.dequeued.map((node) => node.linkOrder);
    for (let i = 1; i < removedOrders.length; i++) {
      assert.ok(removedOrders[i - 1] < removedOrders[i], `trial ${trial}: removals must follow link order`);
    }
    const returned = run.timeline.filter((r) => r.kind === "deq" && !r.empty).map((r) => r.value).sort((a, b) => a - b);
    const removed = queue.dequeued.map((node) => node.value).sort((a, b) => a - b);
    assert.deepStrictEqual(returned, removed, `trial ${trial}: every removal is reported exactly once`);
    assert.strictEqual(queue.dequeued.length + queue.contents().length, queue.links, `trial ${trial}: values are conserved`);
    assert.strictEqual(new Set(removed).size, removed.length, `trial ${trial}: no duplicate removal`);
    for (const node of queue.dequeued) assert.ok(!queue.contents().includes(node.value), `trial ${trial}: removed values are off the queue`);
  }

  // --- determinism, argument and structure errors ---
  const build = () => {
    const queue = new MsQueue();
    const run = runSchedule(queue, [
      [{ kind: "enq", value: 1 }, { kind: "deq" }],
      [{ kind: "enq", value: 2 }, { kind: "deq" }],
    ], roundRobin);
    return { contents: queue.contents(), steps: run.steps, failures: queue.casFailures };
  };
  assert.deepStrictEqual(build(), build(), "identical schedules replay identically");
  assert.throws(() => runSchedule(new MsQueue(), [[{ kind: "push", value: 1 }]], roundRobin), /unknown operation/);
  assert.throws(() => runSchedule(new MsQueue(), [[{ kind: "enq", value: 1 }]], roundRobin, { limit: 0 }), /did not converge/);
  assert.throws(() => {
    const broken = new MsQueue();
    broken.tail = new Atomic({ value: 9, next: new Atomic(null), linkOrder: 9 });
    broken.validate();
  }, /tail is not reachable/);

  console.log(`steps=${single.steps + pair.steps} casFailures=${raced.casFailures + doubling.casFailures} helping=${helping.helpingAdvances}`);
  console.log("PASS");
}
