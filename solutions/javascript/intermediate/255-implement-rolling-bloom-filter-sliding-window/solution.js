/** Task 255 - Implement rolling bloom filter sliding window. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

const SLICES = 4;
const BITS_PER_SLICE = 64;
const HASHES = 3;
// Deterministic expectation for the "loaded" scenario below (cross-language check).
const EXPECTED_HITS = 4;

function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    h = (h ^ text.charCodeAt(i)) >>> 0;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

function fmix32(x) {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x = (x ^ (x >>> 13)) >>> 0;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x;
}

function positions(item, bits, hashes) {
  const h1 = fmix32(fnv1a(item));
  const h2 = (fmix32(fnv1a(item + "#")) | 1) >>> 0;
  const out = [];
  for (let i = 0; i < hashes; i += 1) out.push((h1 + i * h2) % bits);
  return out;
}

/**
 * Sliding-window bloom filter: every insert writes its probes into ALL slices; every step()
 * clears exactly one slice, rotating. check() ORs the slices, so an item inserted at step t
 * is guaranteed visible for the next S-1 steps (some slice still holds its bits) and is
 * guaranteed gone once S clears have happened. False positives are expected; false
 * negatives inside the window are impossible.
 */
class RollingBloom {
  constructor(slices = SLICES, bits = BITS_PER_SLICE, hashes = HASHES) {
    this.slices = Array.from({ length: slices }, () => new Uint8Array(bits));
    this.bits = bits;
    this.hashes = hashes;
    this.current = 0;
  }

  insert(item) {
    const probes = positions(item, this.bits, this.hashes);
    for (const slice of this.slices) for (const p of probes) slice[p] = 1;
  }

  step() {
    this.current = (this.current + 1) % this.slices.length;
    this.slices[this.current] = new Uint8Array(this.bits);
  }

  contains(item) {
    const probes = positions(item, this.bits, this.hashes);
    return this.slices.some((slice) => probes.every((p) => slice[p] === 1));
  }

  popcount(index) {
    let count = 0;
    for (let i = 0; i < this.slices[index].length; i += 1) count += this.slices[index][i];
    return count;
  }
}

module.exports = { SLICES, BITS_PER_SLICE, HASHES, fnv1a, fmix32, positions, RollingBloom };

if (require.main === module) {
  const assert = require("node:assert");

  const rb = new RollingBloom();
  assert.strictEqual(rb.contains("alpha"), false);       // empty window: guaranteed miss

  rb.insert("alpha");
  assert.strictEqual(rb.contains("alpha"), true);
  for (let i = 0; i < SLICES; i += 1) assert.strictEqual(rb.popcount(i), HASHES);

  rb.step();
  assert.strictEqual(rb.contains("alpha"), true);        // age 1: still inside the window
  let zeroed = 0;
  for (let i = 0; i < SLICES; i += 1) if (rb.popcount(i) === 0) zeroed += 1;
  assert.strictEqual(zeroed, 1);                         // exactly one rotation

  const ghost = new RollingBloom();
  ghost.insert("solo");
  for (let i = 0; i < SLICES; i += 1) ghost.step();
  assert.strictEqual(ghost.contains("solo"), false);     // window passed: expired
  for (let i = 0; i < SLICES; i += 1) assert.strictEqual(ghost.popcount(i), 0);

  const keep = new RollingBloom();
  keep.insert("keep");
  for (let i = 0; i < SLICES - 1; i += 1) keep.step();
  assert.strictEqual(keep.contains("keep"), true);       // age 3: still inside the window
  keep.step();
  assert.strictEqual(keep.contains("keep"), false);      // age 4: every slice was cleared

  const timeline = new RollingBloom();
  const history = [];
  let stepNo = 0;
  const addNow = (item) => {
    timeline.insert(item);
    history.push({ step: stepNo, item });
  };
  const assertNoFalseNegative = () => {
    for (const { step, item } of history) {
      if (stepNo - step <= SLICES - 1) {
        assert.strictEqual(timeline.contains(item), true, `in-window miss for ${item}`);
      }
    }
  };
  addNow("alpha");
  for (let i = 1; i <= 8; i += 1) {
    timeline.step();
    stepNo += 1;
    addNow(`w${i}`);
    assertNoFalseNegative();
  }
  assert.strictEqual(timeline.contains("alpha"), false); // 8 steps old: fully evicted

  const loaded = new RollingBloom();
  for (let i = 0; i < 8; i += 1) loaded.insert(`member-${i}`);
  for (let i = 0; i < 8; i += 1) assert.strictEqual(loaded.contains(`member-${i}`), true);
  let hits = 0;
  for (let i = 0; i < 100; i += 1) if (loaded.contains(`q${i}`)) hits += 1;
  assert.strictEqual(hits, EXPECTED_HITS);
  assert.ok(hits >= 1);                                  // false positives do happen: expected
  console.log("PASS");
}
