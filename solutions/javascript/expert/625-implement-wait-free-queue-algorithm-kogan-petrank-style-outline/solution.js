/** Task 625 - Implement wait-free queue algorithm (Kogan-Petrank style outline). (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * The Kogan-Petrank queue makes every operation *wait-free*: a thread announces its operation in
 * a shared phase array, then performs a bounded number of steps in which it both advances its own
 * operation and *helps* any other pending operation it finds. Because any thread can finish any
 * announced operation, a thread that stalls (or is descheduled forever) cannot prevent the others
 * from completing theirs. The queue itself is a linked list of bounded segments; `enqIdx` and
 * `deqIdx` are fetch-and-increment counters that assign slot indices, and each slot holds the
 * enqueue operation record that claimed that index.
 *
 * What is simplified (the task asks for an outline): no threads, no atomics, no timers. The
 * paper's CAS loops are replaced by an explicit, deterministic step scheduler: `help` performs
 * exactly what one helping round of the algorithm does, and the fetch-and-increment claims are
 * sequential because the schedule never lets two threads claim on the same instruction. The
 * CAS *checks* are kept: `completeEnqueue` verifies that a slot never ends up holding a foreign
 * operation, and `help` is idempotent because it re-reads the record's state first.
 *
 * Verified invariants: (1) FIFO order - the dequeued values are exactly a prefix of the slot-claim
 * order; (2) no value is lost or duplicated, and every announced operation reaches a decision;
 * (3) with helping, each operation completes within `2 * threads + 4` scheduling steps, and a
 * thread that stops stepping still has its operation completed by a helper; (4) without helping
 * the same schedule starves - the stalled enqueue is never completed and a dequeue wrongly
 * reports the queue empty, which is what the waiting deadline in the real algorithm exists for.
 */
"use strict";

const PENDING = "pending";
const SUCCESS = "success";
const FAIL = "fail";

/** A bounded segment: shared counters plus one operation record per claimed slot. */
class Segment {
  constructor(capacity, id) {
    this.capacity = capacity;
    this.id = id;
    this.slots = new Array(capacity).fill(null);
    this.enqIdx = 0;
    this.deqIdx = 0;
    this.next = null;
  }
}

class WaitFreeQueue {
  constructor(threads, { helping = true } = {}) {
    if (threads < 1) throw new Error("the queue needs at least one thread");
    this.threads = threads;
    this.helping = helping;
    this.capacity = threads + 2;
    this.head = new Segment(this.capacity, 0);
    this.tail = this.head;
    this.phase = new Array(threads).fill(null);
    this.segments = 1;
    this.unlinked = 0;
    this.starvations = 0;
    this.helpRounds = 0;
    this.records = [];
    this.claimOrder = [];
    this.completions = [];
  }

  announce(thread, record) {
    const current = this.phase[thread];
    if (current !== null && current.state === PENDING) throw new Error("one outstanding operation per thread");
    record.thread = thread;
    record.state = PENDING;
    record.steps = 0;
    record.helpers = 0;
    this.phase[thread] = record;
    this.records.push(record);
    return record;
  }

  pendingRecords() {
    return this.phase.filter((record) => record !== null && record.state === PENDING);
  }

  /** Fetch-and-increment on the tail segment's enqIdx, linking a fresh segment when it is full. */
  claimSlot() {
    if (this.tail.enqIdx === this.tail.capacity) {
      const next = new Segment(this.capacity, this.segments++);
      this.tail.next = next;
      this.tail = next;
    }
    const segment = this.tail;
    return { segment, index: segment.enqIdx++ };
  }

  completeEnqueue(record) {
    const { segment, index } = this.claimSlot();
    if (segment.slots[index] === null) segment.slots[index] = record;
    else if (segment.slots[index] !== record) throw new Error("slot already holds a foreign operation");
    this.claimOrder.push(record.value);
    record.state = SUCCESS;
    return record;
  }

  /** Take the head slot; a slot whose enqueue has not been placed yet must be helped first. */
  completeDequeue(record, actor) {
    for (let guard = 0; guard < 8 * this.threads + 8; guard++) {
      const segment = this.head;
      if (segment.deqIdx < segment.enqIdx) {
        const slot = segment.slots[segment.deqIdx];
        if (slot === null) {
          if (!this.helping) return this.starve(record);
          const pending = this.pendingRecords().filter((other) => other.type === "enq");
          if (pending.length === 0) return this.starve(record);
          this.help(pending[0], actor);
          continue;
        }
        segment.deqIdx++;
        if (slot.state === PENDING) this.help(slot, actor);
        record.value = slot.value;
        record.state = slot.state === SUCCESS ? SUCCESS : FAIL;
        return record;
      }
      // the segment is fully consumed: unlink it and continue in the next one
      if (segment.next === null) {
        const pending = this.pendingRecords().filter((other) => other !== record);
        if (pending.length === 0) {
          record.state = FAIL; // definitively empty: nothing is in flight
          return record;
        }
        if (!this.helping) return this.starve(record);
        this.help(pending[0], actor);
        continue;
      }
      this.head = segment.next;
      this.unlinked++;
    }
    return this.starve(record);
  }

  starve(record) {
    this.starvations++;
    record.state = FAIL;
    return record;
  }

  help(record, actor = record.thread) {
    if (record.state !== PENDING) return record; // idempotent: a decided record is never redone
    if (record.type === "enq") this.completeEnqueue(record);
    else this.completeDequeue(record, actor);
    if (record.state !== PENDING && record.thread !== actor) record.helpers++;
    return record;
  }

  /** One helping round: complete the lowest-index pending operation. Every still-pending
   *  operation counts the round, which is what the wait-free step bound is measured in. */
  helpFirstPending(actor) {
    this.helpRounds++;
    const pending = this.pendingRecords();
    for (const record of pending) record.steps++;
    if (pending.length === 0) return null;
    this.help(pending[0], actor);
    return pending[0];
  }

  /** One scheduling step of `thread`: finish its outstanding operation, else announce the next. */
  step(thread, script) {
    const outstanding = this.phase[thread];
    if (outstanding !== null) {
      if (outstanding.state !== PENDING) return this.retire(thread, outstanding);
      if (this.helping) this.helpFirstPending(thread);
      else this.help(outstanding, thread);
      if (outstanding.state === PENDING) return "waiting";
      return this.retire(thread, outstanding);
    }
    const operation = script[thread].shift();
    if (operation === undefined) {
      if (this.helping) this.helpFirstPending(thread);
      return "idle";
    }
    const record = operation.type === "enq"
      ? this.announce(thread, { type: "enq", value: operation.value })
      : this.announce(thread, { type: "deq" });
    if (this.helping) this.helpFirstPending(thread);
    else this.help(record, thread);
    if (record.state === PENDING) return "waiting";
    return this.retire(thread, record);
  }

  retire(thread, record) {
    this.phase[thread] = null;
    this.completions.push({ thread, type: record.type, value: record.value, steps: record.steps });
    return "completed";
  }

  /** Convenience wrappers for the single-threaded checks (thread 0 acts alone). */
  syncEnqueue(value) {
    const record = this.announce(0, { type: "enq", value });
    this.help(record, 0);
    this.phase[0] = null;
    return record;
  }

  syncDequeue() {
    const record = this.announce(0, { type: "deq" });
    this.help(record, 0);
    this.phase[0] = null;
    return record;
  }

  dequeuedValues() {
    return this.completions.filter((entry) => entry.type === "deq" && entry.value !== undefined).map((entry) => entry.value);
  }
}

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

function buildScripts(seed, threads, opsPerThread) {
  const scripts = [];
  let value = 0;
  for (let t = 0; t < threads; t++) {
    const script = [];
    for (let i = 0; i < opsPerThread; i++) {
      seed = lcg(seed);
      script.push(seed % 100 < 55 ? { type: "enq", value: `v${value++}` } : { type: "deq" });
    }
    scripts.push(script);
  }
  return { scripts, seed, submitted: value };
}

/** Deterministic round-robin micro-step schedule; a stalled thread is simply never scheduled. */
function runSchedule(queue, scripts, { stalled = null } = {}) {
  const threads = scripts.length;
  const lists = scripts.map((list) => list.slice());
  const idleLimit = 4 * threads + 8;
  let idle = 0;
  for (let tick = 0; tick < 4000; tick++) {
    if (lists.every((list) => list.length === 0) && queue.phase.every((record) => record === null)) break;
    const thread = tick % threads;
    if (thread === stalled) continue; // this thread announced its operation and then stopped running
    const before = queue.phase[thread];
    const outcome = queue.step(thread, lists);
    if (outcome === "idle" && before === null) {
      if (++idle > idleLimit) break;
    } else {
      idle = 0;
    }
  }
  return { queue, lists };
}

module.exports = {
  PENDING, SUCCESS, FAIL, Segment, WaitFreeQueue, buildScripts, runSchedule, lcg,
};

if (require.main === module) {
  const assert = require("node:assert");
  // --- single thread: FIFO order and the empty boundary --------------------------------
  const solo = new WaitFreeQueue(1);
  assert.strictEqual(solo.syncDequeue().state, FAIL, "an empty queue reports empty");
  for (const value of ["a", "b", "c"]) assert.strictEqual(solo.syncEnqueue(value).state, SUCCESS);
  assert.deepStrictEqual(["a", "b", "c"].map((v) => solo.syncDequeue().value), ["a", "b", "c"]);
  assert.strictEqual(solo.syncDequeue().state, FAIL, "empty again");
  assert.strictEqual(solo.starvations, 0, "no starvation on a single thread");
  assert.strictEqual(solo.claimOrder.length, 3);
  // --- segments grow, are chained and are unlinked once drained ------------------------
  const growing = new WaitFreeQueue(1); // capacity = 3 slots per segment
  for (let i = 0; i < 8; i++) growing.syncEnqueue(i);
  assert.ok(growing.segments > 1, "the queue must chain a second segment");
  assert.deepStrictEqual(Array.from({ length: 8 }, () => growing.syncDequeue().value), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.ok(growing.unlinked >= 1, "fully consumed segments are unlinked");
  assert.strictEqual(growing.syncDequeue().state, FAIL, "empty after draining");
  // --- a stalled thread cannot block the others (helping) ------------------------------
  const helpedQueue = new WaitFreeQueue(3, { helping: true });
  const stalledRecord = helpedQueue.announce(0, { type: "enq", value: "lost" }); // then thread 0 stops
  const helped = runSchedule(helpedQueue, [[], [{ type: "deq" }], [{ type: "deq" }]], { stalled: 0 });
  assert.strictEqual(stalledRecord.state, SUCCESS, "a helper completed the stalled enqueue");
  assert.ok(stalledRecord.helpers >= 1, "the completing thread is recorded as a helper");
  assert.deepStrictEqual(helped.queue.dequeuedValues(), ["lost"], "the value is still delivered in FIFO order");
  assert.strictEqual(helped.queue.starvations, 0);
  // --- the same schedule without helping starves ---------------------------------------
  const starvedQueue = new WaitFreeQueue(3, { helping: false });
  const lostRecord = starvedQueue.announce(0, { type: "enq", value: "lost" });
  const starved = runSchedule(starvedQueue, [[], [{ type: "deq" }], [{ type: "deq" }]], { stalled: 0 });
  assert.strictEqual(lostRecord.state, PENDING, "without helping the stalled operation never completes");
  assert.ok(starved.queue.starvations > 0, "a dequeue wrongly reports the queue empty");
  assert.deepStrictEqual(starved.queue.dequeuedValues(), [], "the stalled value is never delivered");
  // --- randomized interleavings: FIFO, no loss, no duplication, bounded steps -----------
  let seed = 424242;
  let totalOperations = 0;
  let maxSteps = 0;
  for (let trial = 0; trial < 18; trial++) {
    seed = lcg(seed);
    const threads = 2 + (seed % 3);
    const built = buildScripts(seed, threads, 10);
    seed = built.seed;
    const queue = new WaitFreeQueue(threads, { helping: true });
    runSchedule(queue, built.scripts);
    assert.strictEqual(queue.starvations, 0, `trial ${trial}: no operation may starve`);
    assert.deepStrictEqual(queue.pendingRecords(), [], `trial ${trial}: every announced operation is decided`);
    assert.strictEqual(new Set(queue.claimOrder).size, queue.claimOrder.length, `trial ${trial}: duplicated claim`);
    assert.strictEqual(new Set(queue.dequeuedValues()).size, queue.dequeuedValues().length, `trial ${trial}: value delivered twice`);
    assert.deepStrictEqual(queue.dequeuedValues(), queue.claimOrder.slice(0, queue.dequeuedValues().length),
      `trial ${trial}: dequeues must follow the slot-claim order (FIFO)`);
    assert.deepStrictEqual(queue.claimOrder.slice().sort(), Array.from({ length: built.submitted }, (_, i) => `v${i}`).sort(),
      `trial ${trial}: the claimed values are exactly the submitted ones`);
    for (const entry of queue.completions) {
      assert.ok(entry.steps <= 2 * threads + 4, `trial ${trial}: operation took ${entry.steps} steps`);
      maxSteps = Math.max(maxSteps, entry.steps);
    }
    const enqueues = queue.records.filter((record) => record.type === "enq");
    assert.strictEqual(enqueues.length, built.submitted, `trial ${trial}: enqueue records`);
    for (const record of enqueues) assert.strictEqual(record.state, SUCCESS, `trial ${trial}: an enqueue must not fail`);
    totalOperations += queue.completions.length;
    assert.ok(totalOperations > 0);
  }
  // --- helping is idempotent ------------------------------------------------------------
  const idem = new WaitFreeQueue(2);
  const first = idem.announce(0, { type: "enq", value: "once" });
  idem.help(first, 1);
  assert.strictEqual(first.helpers, 1, "the helped record notes that another thread completed it");
  idem.help(first, 1);
  idem.help(first, 1);
  assert.strictEqual(idem.claimOrder.length, 1, "a decided operation is never enqueued twice");
  assert.strictEqual(first.helpers, 1, "later helping rounds are no-ops");
  assert.strictEqual(first.state, SUCCESS);
  idem.phase[0] = null;
  const second = idem.announce(1, { type: "deq" });
  idem.help(second, 1);
  assert.strictEqual(second.state, SUCCESS);
  assert.strictEqual(second.value, "once", "the dequeue reads the same slot the helper filled");
  // --- operations announced before anybody helps must wait, but stay bounded --------------
  const batched = new WaitFreeQueue(3, { helping: true });
  const announced = ["x", "y", "z"].map((value, thread) => batched.announce(thread, { type: "enq", value }));
  for (let round = 0; round < 20 && batched.pendingRecords().length > 0; round++) batched.helpFirstPending(round % 3);
  assert.deepStrictEqual(announced.map((record) => record.state), [SUCCESS, SUCCESS, SUCCESS], "all three decided");
  assert.deepStrictEqual(batched.claimOrder, ["x", "y", "z"], "helping preserves the announcement order");
  assert.deepStrictEqual(announced.map((record) => record.steps), [1, 2, 3], "each op counts the rounds it waited");
  assert.ok(announced[2].steps <= 2 * 3 + 4, "waiting stays within the wait-free bound");
  // --- the schedule is deterministic ----------------------------------------------------
  const replayA = runSchedule(new WaitFreeQueue(3, { helping: true }), buildScripts(77, 3, 9).scripts);
  const replayB = runSchedule(new WaitFreeQueue(3, { helping: true }), buildScripts(77, 3, 9).scripts);
  assert.deepStrictEqual(replayA.queue.dequeuedValues(), replayB.queue.dequeuedValues(), "identical schedules");
  assert.ok(replayA.queue.dequeuedValues().length > 0);
  // --- degenerate inputs ----------------------------------------------------------------
  assert.throws(() => new WaitFreeQueue(0), Error, "the queue needs a thread");
  const single = new WaitFreeQueue(2);
  const record = single.announce(0, { type: "enq", value: 1 });
  assert.throws(() => single.announce(0, { type: "enq", value: 2 }), Error, "one outstanding operation per thread");
  single.help(record, 0);
  single.phase[0] = null;
  assert.strictEqual(single.syncDequeue().value, 1);

  console.log(`wait-free ops=${totalOperations} max-steps=${maxSteps} batched-steps=${announced[2].steps} starved-without-helping=${starved.queue.starvations}`); console.log("PASS");
}
