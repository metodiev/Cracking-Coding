/** Task 481 - Implement optimistic checkpointing/restart simulation. (Tier: Advanced, CTCI Ch 14)
 *
 * Model (documented semantics):
 * - A job applies one effect per step, step ids 0..n-1. Effects are idempotent: replaying a step
 *   never duplicates effect id `step` (apply() is a set insert).
 * - Optimistic checkpointing: after every `interval` completed steps (progress % interval == 0)
 *   the job snapshots (progress, effects) into a single checkpoint slot, overwriting the old one.
 * - A crash when "about to start step k" means steps 0..k-1 executed and the checkpoint slot is
 *   the last one taken among them. Recovery rolls back to that slot: effects of steps that were
 *   never checkpointed are lost (optimistic, uncommitted), while everything at or before the last
 *   checkpoint survives: no committed effect is ever lost.
 * - Recovered progress is exactly floor(k / interval) * interval, so redo work is < interval.
 */
"use strict";

class OptimisticJob {
  constructor(n, interval) {
    this.n = n;
    this.interval = interval;
    this.progress = 0;
    this.effects = [];
    this.checkpointProgress = 0;
    this.checkpointEffects = [];
    this.checkpointCount = 0;
    this.executions = 0;
  }

  apply(step) {
    this.executions += 1;
    if (!this.effects.includes(step)) this.effects.push(step);
  }

  // Execute steps 0..k-1, then crash while trying to start step k.
  runUntil(k) {
    while (this.progress < k) {
      const step = this.progress;
      this.apply(step);
      this.progress = step + 1;
      if (this.progress % this.interval === 0) {
        this.checkpointProgress = this.progress;
        this.checkpointEffects = this.effects.slice();
        this.checkpointCount += 1;
      }
    }
  }

  recover() {
    this.progress = this.checkpointProgress;
    this.effects = this.checkpointEffects.slice();
  }

  finish() {
    this.runUntil(this.n);
  }

  redone() {
    return this.executions - this.effects.length;
  }
}

function crashPoints(count, seed, limit) {
  const points = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    points.push(seed % limit);
  }
  return points;
}

function rangeList(n) {
  return Array.from({ length: n }, (_, i) => i);
}

module.exports = { OptimisticJob, crashPoints };

if (require.main === module) {
  const assert = require("node:assert");

  const sorted = (list) => list.slice().sort((a, b) => a - b);
  const eq = (a, b) => assert.deepStrictEqual(a, b);

  const verifyCrash = (n, interval, k) => {
    const job = new OptimisticJob(n, interval);
    job.runUntil(k);
    assert.strictEqual(job.progress, k, `progress at crash ${k}`);
    job.recover();
    const expected = Math.floor(k / interval) * interval;
    assert.strictEqual(job.progress, expected, `recovered progress ${k}`);
    assert.ok(job.progress <= k);
    assert.ok(k - job.progress < interval);
    eq(sorted(job.effects), rangeList(expected));
    job.finish();
    assert.strictEqual(job.progress, n);
    eq(sorted(job.effects), rangeList(n));
    assert.strictEqual(job.redone(), k - expected, `redone ${k}`);
    return job;
  };

  const job = new OptimisticJob(25, 4);
  job.runUntil(7);
  assert.strictEqual(job.progress, 7);
  eq(sorted(job.effects), rangeList(7));
  job.recover();
  assert.strictEqual(job.progress, 4);
  eq(sorted(job.effects), rangeList(4));
  job.runUntil(16);
  job.recover();
  assert.strictEqual(job.progress, 16);
  eq(sorted(job.effects), rangeList(16));
  job.runUntil(23);
  job.recover();
  assert.strictEqual(job.progress, 20);
  eq(sorted(job.effects), rangeList(20));
  job.finish();
  assert.strictEqual(job.progress, 25);
  eq(sorted(job.effects), rangeList(25));
  assert.strictEqual(job.executions, 7 + 12 + 7 + 5);
  assert.strictEqual(job.redone(), (7 - 4) + (23 - 20));
  assert.strictEqual(job.checkpointCount, 6);
  assert.strictEqual(job.checkpointProgress, 24);
  eq(job.checkpointEffects, rangeList(24));

  for (const k of [0, 3, 7, 8, 16, 23]) {
    assert.strictEqual(verifyCrash(25, 4, k).progress, 25, `scripted finish ${k}`);
  }
  assert.deepStrictEqual(
    [3, 7, 8, 16, 23].map((k) => [k, Math.floor(k / 4) * 4]),
    [[3, 0], [7, 4], [8, 8], [16, 16], [23, 20]]
  );

  let checked = 0;
  for (const k of crashPoints(20, 481, 26)) {
    const expected = Math.floor(k / 4) * 4;
    assert.strictEqual(verifyCrash(25, 4, k).progress, 25, `lcg finish ${k}`);
    assert.strictEqual(expected, Math.floor(k / 4) * 4);
    checked += 1;
  }
  assert.strictEqual(checked, 20);

  for (const interval of [1, 3, 5]) {
    for (let k = 0; k <= 20; k++) {
      const verified = verifyCrash(20, interval, k);
      assert.strictEqual(verified.redone(), k - Math.floor(k / interval) * interval);
      assert.ok(verified.redone() < interval);
    }
  }
  console.log("PASS");
}
