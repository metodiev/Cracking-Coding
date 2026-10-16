/** Task 626 - Implement work-stealing deque (Chase-Lev) for task scheduler. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * The Chase-Lev deque gives a task scheduler the two access patterns it needs: the *owner* pushes
 * and pops at the bottom (LIFO - the hottest task runs first, which is what makes work stealing
 * cache friendly) while *thieves* take from the top (FIFO - the oldest work, which keeps the
 * stolen subtrees large). Both ends are unbounded counters `bottom`/`top` over a circular array;
 * the owner alone moves `bottom`, so its pushes and pops need no CAS except in the single-element
 * case, where the owner and a thief race for the same task and the owner settles it with a CAS on
 * `top`. That race is the whole reason the structure is subtle, so the deque below exposes
 * `stealBegin`/`stealCommit` to make the window schedulable.
 *
 * What is simplified: no threads and no timers. The interleaving is explicit - a check notes down
 * a thief's `stealBegin` handle, lets the owner run to completion, then calls `stealCommit`, which
 * must fail its CAS. The owner is assumed to be single-threaded (only one thread pushes/pops), and
 * the circular array is grown by the owner alone, which is what the real algorithm assumes too
 * (grow before publishing the new bottom).
 *
 * Verified invariants: (1) a task is handed out at most once, whatever the interleaving; (2) the
 * owner observes LIFO and thieves observe FIFO order - cross-checked against an array model that
 * pops at the back and shifts at the front; (3) the single-element race is resolved exactly once
 * and the loser gets an empty result instead of a duplicate task; (4) growth preserves the logical
 * contents and every element is accounted for.
 */
"use strict";

/** Circular array addressed by the deque's absolute, monotonically increasing indices. */
class CircularArray {
  constructor(logSize, segment) {
    this.logSize = logSize;
    this.capacity = 1 << logSize;
    this.segment = segment;
  }

  static empty() {
    return new CircularArray(1, [null, null]);
  }

  get(index) {
    return this.segment[index & (this.capacity - 1)];
  }

  put(index, value) {
    this.segment[index & (this.capacity - 1)] = value;
  }

  /** Double the array and copy the live range [top, bottom) into the new one. */
  grow(bottom, top) {
    const next = new CircularArray(this.logSize + 1, new Array(this.capacity * 2).fill(null));
    for (let index = top; index < bottom; index++) next.put(index, this.get(index));
    return next;
  }
}

class WorkStealingDeque {
  constructor() {
    this.top = 0;
    this.bottom = 0;
    this.array = CircularArray.empty();
    this.resizes = 0;
    this.conflicts = 0;
    this.pushes = 0;
    this.pops = 0;
    this.steals = 0;
  }

  /** Owner-only. Grows first if the array would overflow, then publishes the new bottom. */
  push(task) {
    const bottom = this.bottom;
    const top = this.top;
    if (bottom - top > this.array.capacity - 1) {
      this.array = this.array.grow(bottom, top);
      this.resizes++;
    }
    this.array.put(bottom, task);
    this.bottom = bottom + 1;
    this.pushes++;
    return task;
  }

  /** Owner-only, LIFO. The single-element case races with thieves and is settled by CAS(top). */
  pop() {
    const bottom = this.bottom - 1;
    this.bottom = bottom;
    const top = this.top;
    if (bottom < top) {
      this.bottom = top; // the deque was empty: undo the speculative decrement
      return undefined;
    }
    const task = this.array.get(bottom);
    if (bottom > top) {
      this.pops++;
      return task;
    }
    if (this.compareAndSetTop(top, top + 1)) {
      this.bottom = top + 1;
      this.pops++;
      return task;
    }
    this.conflicts++; // a thief took the last task while we were deciding
    this.bottom = top + 1;
    return undefined;
  }

  /** A thief's first half: read both ends and the task, then decide. */
  stealBegin() {
    const top = this.top;
    const bottom = this.bottom;
    if (top >= bottom) return null;
    return { index: top, task: this.array.get(top) };
  }

  /** A thief's second half: only the CAS on `top` may hand the task out. */
  stealCommit(handle) {
    if (handle === null) return undefined;
    if (!this.compareAndSetTop(handle.index, handle.index + 1)) {
      this.conflicts++;
      return undefined;
    }
    this.steals++;
    return handle.task;
  }

  steal() {
    return this.stealCommit(this.stealBegin());
  }

  compareAndSetTop(expected, next) {
    if (this.top !== expected) return false;
    this.top = next;
    return true;
  }

  size() {
    return Math.max(0, this.bottom - this.top);
  }
}

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

/** Alternates owner and thief operations against an array model: `pop` is model.pop and `steal`
 *  is model.shift, so any deviation from the two-ends contract fails the check. */
function runSchedule(seed, steps, assert) {
  const deque = new WorkStealingDeque();
  const model = [];
  const handedOut = new Set();
  let s = seed;
  for (let i = 0; i < steps; i++) {
    s = lcg(s);
    const choice = s % 100;
    if (choice < 50) {
      const task = `t${i}`;
      deque.push(task);
      model.push(task);
    } else if (choice < 75) {
      const got = deque.pop();
      const expected = model.pop();
      assert.strictEqual(got, expected, `step ${i}: the owner must pop the youngest task`);
      if (got !== undefined) {
        assert.ok(!handedOut.has(got), `step ${i}: ${got} was handed out twice`);
        handedOut.add(got);
      }
    } else {
      const got = deque.steal();
      const expected = model.shift();
      assert.strictEqual(got, expected, `step ${i}: a thief must take the oldest task`);
      if (got !== undefined) {
        assert.ok(!handedOut.has(got), `step ${i}: ${got} was handed out twice`);
        handedOut.add(got);
      }
    }
    assert.strictEqual(deque.size(), model.length, `step ${i}: the model and the deque disagree on size`);
  }
  return { deque, model, handedOut };
}

module.exports = { CircularArray, WorkStealingDeque, runSchedule, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  // --- the two ends -----------------------------------------------------------------------
  const deque = new WorkStealingDeque();
  assert.strictEqual(deque.pop(), undefined, "popping an empty deque yields nothing");
  assert.strictEqual(deque.steal(), undefined, "stealing from an empty deque yields nothing");
  assert.strictEqual(deque.stealBegin(), null);
  for (const task of ["a", "b", "c"]) deque.push(task);
  assert.strictEqual(deque.size(), 3);
  assert.strictEqual(deque.steal(), "a", "thieves take the oldest task");
  assert.strictEqual(deque.pop(), "c", "the owner takes the youngest task");
  assert.strictEqual(deque.steal(), "b");
  assert.strictEqual(deque.size(), 0);
  assert.strictEqual(deque.pop(), undefined, "empty again");

  // --- the single-element race, owner wins ------------------------------------------------
  const racy = new WorkStealingDeque();
  racy.push("only");
  const thief = racy.stealBegin();
  assert.deepStrictEqual(thief, { index: 0, task: "only" });
  assert.strictEqual(racy.pop(), "only", "the owner settles the last-element race");
  assert.strictEqual(racy.stealCommit(thief), undefined, "the thief's stale CAS must fail");
  assert.strictEqual(racy.conflicts, 1, "the failed CAS is counted");
  assert.strictEqual(racy.size(), 0);
  assert.strictEqual(racy.steals, 0);

  // --- the same race with the thief winning ----------------------------------------------
  const racy2 = new WorkStealingDeque();
  racy2.push("only");
  const thief2 = racy2.stealBegin();
  assert.strictEqual(racy2.stealCommit(thief2), "only", "a thief may win the race instead");
  assert.strictEqual(racy2.pop(), undefined, "the owner then finds nothing");
  assert.strictEqual(racy2.size(), 0);
  assert.strictEqual(racy2.pops, 0);

  // --- growth preserves contents ----------------------------------------------------------
  const growing = new WorkStealingDeque();
  for (let i = 0; i < 64; i++) growing.push(`task${i}`);
  assert.ok(growing.resizes >= 4, `expected repeated growth, got ${growing.resizes}`);
  assert.strictEqual(growing.size(), 64);
  const stolen = [];
  for (let i = 0; i < 40; i++) stolen.push(growing.steal());
  assert.deepStrictEqual(stolen, Array.from({ length: 40 }, (_, i) => `task${i}`), "FIFO across resizes");
  const popped = [];
  for (let i = 0; i < 24; i++) popped.push(growing.pop());
  assert.deepStrictEqual(popped, Array.from({ length: 24 }, (_, i) => `task${63 - i}`), "LIFO across resizes");
  assert.strictEqual(growing.size(), 0);
  assert.strictEqual(growing.resizes, 5, "64 tasks in 2-slot arrays need five doublings");
  assert.strictEqual(growing.push("after-growth"), "after-growth");
  assert.strictEqual(growing.pop(), "after-growth");

  // --- randomized schedules against the two-ends model -----------------------------------
  let seed = 20241007;
  let handedOut = 0;
  let conflicts = 0;
  for (let trial = 0; trial < 20; trial++) {
    seed = lcg(seed);
    const result = runSchedule(seed, 150, assert);
    handedOut += result.handedOut.size;
    conflicts += result.deque.conflicts;
    assert.strictEqual(result.handedOut.size, result.deque.pops + result.deque.steals,
      `trial ${trial}: hand-out accounting`);
    assert.strictEqual(result.deque.pushes, result.handedOut.size + result.model.length,
      `trial ${trial}: every pushed task is either handed out or still queued`);
  }
  assert.ok(handedOut > 500, "the schedules must hand out plenty of tasks");
  assert.strictEqual(conflicts, 0, "sequential schedules never race, so no CAS may fail");

  // --- boundary: the absolute indices keep counting across drain and refill --------------
  const recycled = new WorkStealingDeque();
  recycled.push("x");
  assert.strictEqual(recycled.steal(), "x");
  recycled.push("y");
  assert.strictEqual(recycled.pop(), "y");
  assert.strictEqual(recycled.size(), 0);
  assert.strictEqual(recycled.top, recycled.bottom, "an empty deque has top === bottom");
  for (let i = 0; i < 5; i++) recycled.push(i);
  assert.deepStrictEqual([0, 1, 2, 3, 4].map(() => recycled.steal()), [0, 1, 2, 3, 4], "still FIFO at the top");

  console.log(`chase-lev tasks handed out=${handedOut} resizes=${growing.resizes} race conflicts=${racy.conflicts}`);
  console.log("PASS");
}
