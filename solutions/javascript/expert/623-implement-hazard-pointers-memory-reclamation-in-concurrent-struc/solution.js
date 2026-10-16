/** Task 623 - Implement hazard pointers memory reclamation in concurrent structures. (Tier: Expert, CTCI Ch 15 — Threads and Locks)
 *
 * Hazard pointers (Maged Michael, 2004) make a lock-free structure safe against the classic
 * "read a pointer that another thread has already recycled" bug. Before dereferencing a node a
 * thread publishes it into its own hazard slot; a thread that retires a node may return it to the
 * allocator only if no hazard slot currently holds it. Because there are H slots, a scan over R
 * retired nodes frees at least R - H of them, which bounds the retire list and therefore memory
 * use.
 *
 * Model: there are no threads, timers or clock reads here. All concurrency lives in an explicit
 * `MemoryModel` plus a step scheduler that interleaves two threads' push/pop *micro-steps*
 * (a pop is deliberately split into `beginPop` - load head + publish hazard - and `finishPop` -
 * dereference + validating CAS - so the schedule can suspend a thread between the two).
 * `MemoryModel.load` raises a *use-after-free fault* whenever a reclaimed block is dereferenced;
 * that fault counter is the safety property the checks assert.
 *
 * Verified invariants:
 *   1. the hazard protocol never dereferences a reclaimed block (0 faults) on the crafted
 *      interleaving *and* on randomized schedules, while the identical schedule with protection
 *      disabled faults - the control experiment that gives the check teeth;
 *   2. a block that a hazard slot holds is never handed back to the allocator (the scan counts
 *      the deferrals it performed, and the crafted schedule requires at least one);
 *   3. after every thread passes a quiescent state the retire list drains empty (no leak), and
 *      the retire list never exceeds the threshold (bounded memory).
 */
"use strict";

const POISON = Symbol("use-after-free");
const RETRY = "RETRY";
const EMPTY = "EMPTY";
const FAULT = "FAULT";

/** Allocator + dereference checker. `load` is the only way to touch a block, so a block that was
 *  already returned to the allocator is impossible to read without being noticed. */
class MemoryModel {
  constructor() {
    this.blocks = new Map();
    this.reclaimed = new Set();
    this.nextId = 1;
    this.allocations = 0;
    this.faults = [];
  }
  alloc(value, next) {
    const id = this.nextId++;
    this.blocks.set(id, { id, value, next });
    this.allocations++;
    return id;
  }
  load(id) {
    if (this.reclaimed.has(id) || !this.blocks.has(id)) {
      this.faults.push(id);
      return POISON;
    }
    return this.blocks.get(id);
  }
  free(id) {
    this.reclaimed.add(id);
  }

  /** Blocks the stack can still reach; used for the "everything is eventually reclaimed" check. */
  liveChain(head) {
    const live = [];
    let id = head;
    while (id !== null) {
      const block = this.load(id);
      if (block === POISON) return null;
      live.push(id);
      id = block.next;
    }
    return live;
  }
}

/** One hazard slot per thread (slot 0) plus the retire list and the threshold scan. */
class HazardPointers {
  constructor(threads, threshold) {
    this.slots = Array.from({ length: threads }, () => new Array(2).fill(null));
    this.retired = [];
    this.threshold = threshold;
    this.peakRetired = 0;
    this.reclaimed = 0;
    this.deferred = 0;
    this.scans = 0;
  }
  protect(thread, slot, id) {
    this.slots[thread][slot] = id;
    return id !== null;
  }
  clearSlot(thread, slot) {
    this.slots[thread][slot] = null;
  }
  clear(thread) {
    this.slots[thread].fill(null);
  }
  isProtected(id) {
    for (const slot of this.slots) {
      if (slot[0] === id || slot[1] === id) return true;
    }
    return false;
  }
  retire(id) {
    this.retired.push(id);
    if (this.retired.length > this.peakRetired) this.peakRetired = this.retired.length;
  }

  /** One full scan: free everything nobody protects, keep the rest for the next scan. */
  scan(mm) {
    this.scans++;
    const deferred = [];
    for (const id of this.retired) {
      if (this.isProtected(id)) {
        deferred.push(id);
        this.deferred++;
      } else {
        mm.free(id);
        this.reclaimed++;
      }
    }
    this.retired = deferred;
    return this.reclaimed;
  }
  maybeScan(mm) {
    if (this.retired.length >= this.threshold) this.scan(mm);
  }
  drain(mm) {
    this.scan(mm);
  }
}

/** Treiber stack whose pop is split into two schedulable micro-steps. */
class TreiberStack {
  constructor(mm, hp) {
    this.mm = mm;
    this.hp = hp;
    this.head = null;
    this.pushes = 0;
    this.pops = 0;
  }
  push(value) {
    const id = this.mm.alloc(value, this.head);
    this.head = id;
    this.pushes++;
    return id;
  }

  /** Step 1: load the head and (if protected) publish it; re-validate before committing to it. */
  beginPop(thread, protect) {
    const head = this.head;
    if (protect) this.hp.protect(thread, 0, head);
    if (this.head !== head) {
      if (protect) this.hp.clearSlot(thread, 0);
      return { retry: true, node: null };
    }
    return { retry: false, node: head };
  }

  /** Step 2: dereference the protected node, then CAS the head away from it. The dereference is
   *  the operation hazard pointers exist to make safe: without protection the node may already be
   *  back in the allocator, which `MemoryModel.load` reports as a fault. */
  finishPop(thread, handle, protect) {
    if (handle.retry) return RETRY;
    const node = handle.node;
    if (node === null) {
      if (protect) this.hp.clear(thread);
      return EMPTY;
    }
    const block = this.mm.load(node);
    if (block === POISON) {
      if (protect) this.hp.clear(thread);
      return FAULT;
    }
    const next = block.next;
    if (this.head === node) {
      this.head = next;
      this.hp.retire(node);
      this.hp.maybeScan(this.mm);
      if (protect) this.hp.clear(thread);
      this.pops++;
      return block.value;
    }
    if (protect) this.hp.clearSlot(thread, 0);
    return RETRY;
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
      script.push(seed % 100 < 55 ? { type: "push", value: `v${value++}` } : { type: "pop" });
    }
    scripts.push(script);
  }
  return { scripts, seed, nextValue: value };
}

/** Interleaves push/pop micro-steps of `threads` threads in a deterministic order. */
function runInterleaved(seed, threads, opsPerThread, protect, threshold) {
  const mm = new MemoryModel();
  const hp = new HazardPointers(threads, threshold);
  const stack = new TreiberStack(mm, hp);
  const built = buildScripts(seed, threads, opsPerThread);
  const state = built.scripts.map((list) => ({ list, pending: null, retry: false, attempts: 0, done: false }));
  const pushed = [];
  const popped = [];
  let empties = 0;
  let gaveUp = 0;
  let s = built.seed;
  for (let guard = 0; guard < 4000; guard++) {
    if (state.every((st) => st.done)) break;
    s = lcg(s);
    const t = s % threads;
    const st = state[t];
    if (st.done) continue;
    if (st.pending) {
      const result = stack.finishPop(t, st.pending, protect);
      st.pending = null;
      if (result === RETRY) {
        st.attempts++;
        if (st.attempts > 16) {
          st.done = true;
          gaveUp++;
        } else {
          st.retry = true;
        }
      } else {
        if (result === EMPTY) empties++;
        else popped.push(result);
        st.done = st.list.length === 0;
      }
      continue;
    }
    if (st.retry) {
      st.pending = stack.beginPop(t, protect);
      st.retry = false;
      continue;
    }
    const op = st.list.shift();
    if (!op) {
      st.done = true;
      continue;
    }
    if (op.type === "push") {
      pushed.push(op.value);
      stack.push(op.value);
    } else {
      st.pending = stack.beginPop(t, protect);
    }
  }
  return { mm, hp, stack, pushed, popped, empties, gaveUp };
}

/** Same two-thread micro-step skeleton as `runInterleaved`, but the interleaving is fixed so the
 *  preemption window is guaranteed to be hit: T0 loads the head, T1 pops and reclaims it, T0
 *  resumes and dereferences the node it loaded. */
function craftedPreemption(protect, threshold = 1) {
  const mm = new MemoryModel();
  const hp = new HazardPointers(2, threshold);
  const stack = new TreiberStack(mm, hp);
  stack.push("a"); // block 1
  stack.push("b"); // block 2, the head
  const t0 = stack.beginPop(0, protect); // T0 loads block 2 and is suspended here
  const t1 = stack.beginPop(1, protect);
  const t1Result = stack.finishPop(1, t1, protect); // T1 pops block 2 and retires it
  const t0Result = stack.finishPop(0, t0, protect); // T0 resumes: dereference of block 2 + CAS
  let retryResult = null;
  if (t0Result === RETRY) retryResult = stack.finishPop(0, stack.beginPop(0, protect), protect);
  hp.clear(0);
  hp.clear(1);
  hp.drain(mm);
  return { mm, hp, stack, t0Result, t1Result, retryResult };
}

module.exports = {
  POISON, RETRY, EMPTY, FAULT, MemoryModel, HazardPointers, TreiberStack,
  buildScripts, runInterleaved, craftedPreemption, lcg,
};

if (require.main === module) {
  const assert = require("node:assert");
  // --- crafted preemption: protection on (safe) vs off (use-after-free) ---------------------
  const safe = craftedPreemption(true);
  assert.strictEqual(safe.t1Result, "b", "T1 must pop the head block");
  assert.strictEqual(safe.mm.faults.length, 0, "hazard protocol must not dereference a freed block");
  assert.strictEqual(safe.t0Result, RETRY, "T0 must notice that the head moved on and retry");
  assert.ok(safe.hp.deferred >= 1, "the scan must defer the block T0 still protects");
  assert.strictEqual(safe.retryResult, "a", "the retried pop must return the remaining value");
  assert.strictEqual(safe.mm.reclaimed.size, 2, "both popped blocks are reclaimable after quiescence");
  assert.strictEqual(safe.hp.retired.length, 0, "drain must empty the retire list");
  const unsafe = craftedPreemption(false);
  assert.ok(unsafe.mm.faults.length >= 1, "without hazard pointers the same schedule must fault");
  assert.strictEqual(unsafe.t0Result, FAULT, "the unprotected pop dereferences a reclaimed block");
  assert.strictEqual(unsafe.hp.deferred, 0, "no protection means no deferral");
  // --- trivial boundaries ------------------------------------------------------------------
  const emptyStack = new TreiberStack(new MemoryModel(), new HazardPointers(1, 2));
  assert.strictEqual(emptyStack.finishPop(0, emptyStack.beginPop(0, true), true), EMPTY);
  assert.strictEqual(emptyStack.head, null, "popping an empty stack leaves it empty");
  const single = new TreiberStack(new MemoryModel(), new HazardPointers(1, 2));
  single.push(42);
  assert.strictEqual(single.finishPop(0, single.beginPop(0, true), true), 42, "single element round trip");
  assert.strictEqual(single.finishPop(0, single.beginPop(0, true), true), EMPTY, "stack is empty again");
  // --- randomized interleavings with the hazard protocol ------------------------------------
  let seed = 20240719;
  let maxPeak = 0;
  for (let trial = 0; trial < 24; trial++) {
    seed = lcg(seed);
    const threads = 2 + (seed % 3);
    const run = runInterleaved(seed, threads, 12, true, threads * 2);
    assert.strictEqual(run.mm.faults.length, 0, `trial ${trial}: use-after-free in the protected run`);
    assert.strictEqual(run.gaveUp, 0, `trial ${trial}: pop retry budget exhausted`);
    const remaining = run.mm.liveChain(run.stack.head);
    assert.ok(remaining !== null, `trial ${trial}: stack chain must be dereferenceable`);
    assert.strictEqual(run.popped.length + remaining.length, run.pushed.length, `trial ${trial}: lost or duplicated values`);
    const seen = new Set(run.popped.concat(remaining.map((id) => run.mm.blocks.get(id).value)));
    assert.strictEqual(seen.size, run.pushed.length, `trial ${trial}: duplicated value delivered`);
    assert.ok(run.hp.peakRetired <= threads * 2 + 1, `trial ${trial}: retire list must stay bounded`);
    for (let t = 0; t < run.hp.slots.length; t++) run.hp.clear(t); // every thread passes a quiescent state
    run.hp.drain(run.mm);
    assert.strictEqual(run.mm.reclaimed.size, run.mm.allocations - remaining.length,
      `trial ${trial}: every block off the stack must be reclaimed and no live block may be`);
    assert.strictEqual(run.hp.retired.length, 0, `trial ${trial}: retire list must drain`);
    assert.ok(run.hp.deferred > 0, `trial ${trial}: the schedule exercised at least one deferral`);
    maxPeak = Math.max(maxPeak, run.hp.peakRetired);
  }
  // --- the unprotected control over randomized schedules -----------------------------------
  seed = lcg(seed);
  const unsafeRun = runInterleaved(seed, 3, 14, false, 6);
  assert.strictEqual(unsafeRun.hp.deferred, 0, "unprotected runs never defer a block");
  assert.strictEqual(unsafeRun.mm.faults.length > 0, true, "unprotected runs expose the race");
  console.log(`hazard scans=${safe.hp.scans} crafted-deferrals=${safe.hp.deferred} peak-retired=${maxPeak}`);
  console.log(`protected-window faults=0 unprotected-window faults=${unsafeRun.mm.faults.length}`);
  console.log("PASS");
}
