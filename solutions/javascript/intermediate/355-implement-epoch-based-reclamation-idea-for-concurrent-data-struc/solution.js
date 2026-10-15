/**
 * Task 355 - Implement epoch-based reclamation idea for concurrent data structure safe delete. (Tier: Intermediate, CTCI Ch 15)
 *
 * Deterministic single-threaded simulation of epoch-based reclamation. The essence of EBR
 * is the pin/epoch bookkeeping, so a scripted interleaving (no real threads needed) is a
 * faithful representation of the algorithm: an object retired in epoch r may be freed only
 * when no thread is pinned in an epoch <= r - i.e. r <= minPinnedEpoch - 1, and with no
 * pinned threads every retired object is safe. collect() first tries to advance the epoch
 * (an advance requires every pinned thread to sit in the current epoch), then drains every
 * bucket entry the rule allows and keeps the rest on the deferred list. Real EBR frees the
 * bucket from two epochs back; this conservative rule is strictly safer. The self-test
 * scripts pin/unpin/retire/collect phases plus an LCG-driven interleaving and asserts the
 * free invariant at every free, deferral while old pins exist, and a full drain at rest.
 */
"use strict";

const EPOCHS = 3;
const SCRIPT_STEPS = 120;

class EbrManager {
  constructor(numThreads) {
    this.globalEpoch = 0;
    this.pinEpochs = new Array(numThreads).fill(null);
    this.buckets = [];
    for (let i = 0; i < EPOCHS; i++) this.buckets.push([]);   // [objectId, retirementEpoch]
    this.freeLog = [];                                        // [objectId, retirementEpoch, epochAtFree]
    this.violations = 0;
    this.deferredEntries = 0;
    this.advances = 0;
  }

  pin(thread) {
    if (this.pinEpochs[thread] !== null) throw new Error(`thread ${thread} already pinned`);
    this.pinEpochs[thread] = this.globalEpoch;
  }

  unpin(thread) {
    if (this.pinEpochs[thread] === null) throw new Error(`thread ${thread} is not pinned`);
    this.pinEpochs[thread] = null;
  }

  isPinned(thread) {
    return this.pinEpochs[thread] !== null;
  }

  retire(thread, objectId) {
    if (this.pinEpochs[thread] === null) throw new Error("retire requires a pinned thread");
    this.buckets[this.globalEpoch % EPOCHS].push([objectId, this.globalEpoch]);
  }

  /** Advance only when every pinned thread sits in the current epoch. */
  tryAdvance() {
    for (const epoch of this.pinEpochs) {
      if (epoch !== null && epoch !== this.globalEpoch) return false;
    }
    this.globalEpoch += 1;
    this.advances += 1;
    return true;
  }

  minPinnedEpoch() {
    let min = null;
    for (const epoch of this.pinEpochs) {
      if (epoch !== null) min = min === null ? epoch : Math.min(min, epoch);
    }
    return min;
  }

  /** Drain every safe deferred object; returns the objects freed by this call. */
  collect() {
    this.tryAdvance();
    const minPinned = this.minPinnedEpoch();
    const safeEpoch = minPinned === null ? null : minPinned - 1;
    const freed = [];
    for (const bucket of this.buckets) {
      const kept = [];
      for (const [objectId, retiredEpoch] of bucket) {
        if (safeEpoch === null || retiredEpoch <= safeEpoch) {
          this.recordFree(objectId, retiredEpoch);
          freed.push(objectId);
        } else {
          kept.push([objectId, retiredEpoch]);
        }
      }
      this.deferredEntries += kept.length;
      bucket.length = 0;
      bucket.push(...kept);
    }
    return freed;
  }

  /**
   * The safety invariant, checked at the moment of every free: no thread may be pinned
   * in an epoch <= the object's retirement epoch.
   */
  recordFree(objectId, retiredEpoch) {
    for (const epoch of this.pinEpochs) {
      if (epoch !== null && epoch <= retiredEpoch) this.violations += 1;
    }
    this.freeLog.push([objectId, retiredEpoch, this.globalEpoch]);
  }
}

function main() {
  const assert = require("node:assert");

  // Phase 1: retirement in epoch 0; the pin that predates an advance blocks the free.
  const mgr = new EbrManager(3);
  mgr.pin(0);
  mgr.retire(0, "A");                                 // A retired in epoch 0
  assert.deepStrictEqual(mgr.collect(), [], "A must not be freed while thread 0 pins epoch 0");

  mgr.pin(1);                                         // thread 1 pins the current epoch 1
  assert.deepStrictEqual(mgr.collect(), [], "old pin (epoch 0) keeps A deferred");
  assert.strictEqual(mgr.violations, 0, "no free while a thread pinned an old epoch");

  mgr.unpin(0);
  assert.deepStrictEqual(mgr.collect(), ["A"], "A frees once no thread pins an epoch <= 0");
  assert.strictEqual(mgr.violations, 0, "freeing A respected the pin rule");

  // Phase 2: a thread pinned in the retirement epoch itself blocks the free.
  mgr.pin(0);                                         // thread 0 pins epoch 2
  mgr.retire(0, "B");                                 // B retired in epoch 2
  assert.deepStrictEqual(mgr.collect(), [], "B deferred while thread 1 still pins epoch 1");
  mgr.unpin(1);
  assert.deepStrictEqual(mgr.collect(), [], "B still deferred: thread 0 pins its retirement epoch 2");
  mgr.unpin(0);
  assert.deepStrictEqual(mgr.collect(), ["B"], "B frees at quiescence");
  assert.strictEqual(mgr.violations, 0, "freeing B respected the pin rule");
  for (const entry of mgr.freeLog) {
    assert.ok(entry[1] <= entry[2], `free log for ${entry[0]} records a retirement epoch not in the future`);
  }

  // Phase 3: LCG-driven interleaving of pin / unpin / retire / collect across 3 threads.
  const scripted = new EbrManager(3);
  let seed = 4242;
  let nextObject = 100;
  const retired = [];
  for (let step = 0; step < SCRIPT_STEPS; step++) {
    seed = (seed * 75 + 74) % 65537;
    const thread = seed % 3;
    seed = (seed * 75 + 74) % 65537;
    const action = seed % 4;
    if (action === 0 && !scripted.isPinned(thread)) {
      scripted.pin(thread);
    } else if (action === 1 && scripted.isPinned(thread)) {
      scripted.unpin(thread);
    } else if (action >= 2 && scripted.isPinned(thread)) {
      scripted.retire(thread, nextObject);
      retired.push(nextObject);
      nextObject += 1;
    }
    scripted.collect();
    assert.strictEqual(scripted.violations, 0, `free-of-pinned violation at step ${step}`);
  }

  assert.ok(retired.length >= 10, "the interleaving retired a useful number of objects");
  assert.ok(scripted.deferredEntries > 0, "old pins actually deferred some objects");

  // Quiescence: unpin everything, then the deferred list must drain completely.
  for (let thread = 0; thread < 3; thread++) {
    if (scripted.isPinned(thread)) scripted.unpin(thread);
  }
  scripted.collect();
  for (const bucket of scripted.buckets) {
    assert.deepStrictEqual(bucket, [], "deferred lists drained");
  }

  const freedIds = scripted.freeLog.map((entry) => entry[0]).sort((a, b) => a - b);
  const sortedRetired = [...retired].sort((a, b) => a - b);
  assert.deepStrictEqual(freedIds, sortedRetired, "every retired object freed exactly once");
  assert.strictEqual(new Set(freedIds).size, freedIds.length, "no double frees");
  assert.strictEqual(scripted.violations, 0, "the whole run respected the pin rule");
  assert.ok(scripted.advances >= 1, "epochs actually advanced");
}

module.exports = { EbrManager, EPOCHS };

if (require.main === module) {
  try {
    main();
    console.log("PASS");
  } catch (err) {
    console.error(err);
    process.exit(1);
  }
}
