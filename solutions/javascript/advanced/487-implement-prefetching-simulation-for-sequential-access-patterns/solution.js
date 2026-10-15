/** Task 487 - Implement prefetching simulation for sequential access patterns. (Tier: Advanced, CTCI Ch 14)
 *
 * Simulated memory with documented semantics:
 * - Address space is [0, bound); every access and every prefetched address must be inside it - the
 *   prefetcher clips at bound-1 so it can never read past the array.
 * - Sequential pattern detector: the first access of a run is only recorded; every following access
 *   with addr == last + 1 continues the run and issues speculative prefetches for
 *   addr + 1 .. min(addr + depth, bound - 1). Any non-consecutive access restarts the detector.
 * - A prefetch "hit" is an access to an address sitting in the prefetched set (one entry per
 *   address); a hit consumes the entry. Documented lower bound: a cold run of length L >= 2 yields
 *   at least L - 2 prefetch hits (the run's first access and its immediate successor cannot be
 *   covered), so a fresh run of length L scores (L-2)/L (0.9375 at L = 32).
 *
 * The test pins hand cases, then asserts on sequential traces that the per-segment prefetch hit
 * ratio is >= 0.9 (documented threshold) and that no simulated read ever leaves [0, bound).
 */
"use strict";

class Prefetcher {
  constructor(bound, depth) {
    this.bound = bound;
    this.depth = depth;
    this.last = -1;
    this.started = false;
    this.pending = new Set();
    this.hits = 0;
    this.accesses = 0;
    this.issued = 0;
    this.maxPrefetch = -1;
  }

  access(addr) {
    if (addr < 0 || addr >= this.bound) throw new Error(`FAILED: access out of bounds ${addr}`);
    this.accesses += 1;
    const hit = this.pending.delete(addr);
    if (hit) this.hits += 1;
    const continues = this.started && addr === this.last + 1;
    this.last = addr;
    this.started = true;
    if (continues) {
      for (let nxt = addr + 1; nxt <= Math.min(addr + this.depth, this.bound - 1); nxt++) {
        this.pending.add(nxt);
        this.issued += 1;
        this.maxPrefetch = Math.max(this.maxPrefetch, nxt);
      }
    }
    return hit;
  }
}

function lcgValues(count, seed, limit) {
  const values = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    values.push(seed % limit);
  }
  return values;
}

module.exports = { Prefetcher };

if (require.main === module) {
  const assert = require("node:assert");

  let prefetcher = new Prefetcher(10, 3);
  const observed = [];
  for (let addr = 0; addr < 7; addr++) observed.push(prefetcher.access(addr));
  assert.deepStrictEqual(observed, [false, false, true, true, true, true, true]);
  assert.strictEqual(prefetcher.hits, 5);
  assert.deepStrictEqual(Array.from(prefetcher.pending).sort((a, b) => a - b), [7, 8, 9]);
  assert.strictEqual(prefetcher.maxPrefetch, 9);
  assert.strictEqual(prefetcher.issued, 18);
  assert.strictEqual(prefetcher.access(9), true);
  assert.strictEqual(prefetcher.hits, 6);
  assert.deepStrictEqual(Array.from(prefetcher.pending).sort((a, b) => a - b), [7, 8]);
  assert.strictEqual(prefetcher.access(4), false);
  assert.strictEqual(prefetcher.hits, 6);
  assert.strictEqual(prefetcher.issued, 18);

  prefetcher = new Prefetcher(10, 3);
  const clip = [];
  for (let addr = 0; addr < 10; addr++) clip.push(prefetcher.access(addr));
  assert.deepStrictEqual(clip, [false, false, true, true, true, true, true, true, true, true]);
  assert.strictEqual(prefetcher.hits, 8);
  assert.strictEqual(prefetcher.maxPrefetch, 9);
  assert.strictEqual(prefetcher.pending.size, 0);

  prefetcher = new Prefetcher(64, 4);
  for (let addr = 0; addr < 40; addr++) prefetcher.access(addr);
  assert.strictEqual(prefetcher.hits, 38);
  assert.strictEqual(prefetcher.accesses, 40);
  assert.ok(prefetcher.hits / prefetcher.accesses >= 0.9);
  assert.strictEqual(prefetcher.maxPrefetch, 43);

  prefetcher = new Prefetcher(128, 4);
  const segmentHits = [];
  for (const jump of lcgValues(4, 487, 80)) {
    const before = prefetcher.hits;
    for (let offset = 0; offset < 32; offset++) prefetcher.access(jump + offset);
    const got = prefetcher.hits - before;
    segmentHits.push(got);
    assert.ok(got >= 30, `segment hits ${jump} ${got}`);
    assert.ok(got / 32 >= 0.9, `segment ratio ${jump}`);
  }
  assert.deepStrictEqual(segmentHits, [30, 30, 30, 30]);
  assert.strictEqual(prefetcher.maxPrefetch < 128, true);
  assert.ok(prefetcher.hits <= prefetcher.accesses);
  assert.ok(prefetcher.hits / prefetcher.accesses >= 0.9);

  prefetcher = new Prefetcher(64, 5);
  const trace = [];
  for (const value of lcgValues(30, 488, 64)) {
    trace.push(value);
    if (value + 2 < 64) trace.push(value + 1, value + 2);
  }
  for (const addr of trace) {
    assert.ok(addr >= 0 && addr < 64, `trace bounds ${addr}`);
    prefetcher.access(addr);
  }
  assert.ok(prefetcher.maxPrefetch <= 63);
  assert.ok(prefetcher.hits <= prefetcher.accesses);
  assert.strictEqual(prefetcher.accesses, trace.length);

  prefetcher = new Prefetcher(128, 4);
  for (const jump of lcgValues(4, 487, 80)) {
    for (let offset = 0; offset < 32; offset++) prefetcher.access(jump + offset);
  }
  assert.strictEqual(prefetcher.accesses, 128);
  assert.strictEqual(prefetcher.issued, 496);
  assert.strictEqual(prefetcher.hits, 120);
  console.log("PASS");
}
