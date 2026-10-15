/** Task 165 - Reservoir sampling for k items from stream. (Tier: Beginner, CTCI Ch 6) */
"use strict";

/** Deterministic seeded PRNG (mulberry32) so tests are reproducible. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function reservoirSample(stream, k, rng) {
  const sample = [];
  for (let i = 0; i < stream.length; i++) {
    if (i < k) {
      sample.push(stream[i]);
    } else {
      const j = Math.floor(rng() * (i + 1));
      if (j < k) sample[j] = stream[i];
    }
  }
  return sample;
}

module.exports = { reservoirSample, mulberry32 };

if (require.main === module) {
  const assert = require("node:assert");
  const stream = Array.from({ length: 100 }, (_, i) => i);

  const sample = reservoirSample(stream, 5, mulberry32(7));
  assert.strictEqual(sample.length, 5);
  assert.strictEqual(new Set(sample).size, 5);
  assert.deepStrictEqual(sample, reservoirSample(stream, 5, mulberry32(7)));
  assert.notDeepStrictEqual(sample, reservoirSample(stream, 5, mulberry32(8)));

  assert.deepStrictEqual(reservoirSample([1, 2, 3], 5, mulberry32(1)), [1, 2, 3]);
  assert.deepStrictEqual(reservoirSample([], 3, mulberry32(1)), []);

  const trials = 4000;
  let hits = 0;
  const rng = mulberry32(123);
  for (let t = 0; t < trials; t++) {
    if (reservoirSample(stream, 5, rng).includes(0)) hits++;
  }
  const ratio = hits / trials;
  assert.ok(ratio > 0.03 && ratio < 0.08, "inclusion ratio ~0.05, got " + ratio);
  console.log("PASS");
}
