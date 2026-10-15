/** Task 449 - Implement global min-cut (Karger randomized algorithm). (Tier: Advanced, CTCI Ch 4) */
"use strict";

class Lcg {
  constructor(seed) {
    this.state = seed % 65537;
  }

  next() {
    this.state = (this.state * 75 + 74) % 65537;
    return this.state;
  }

  below(bound) {
    return this.next() % bound;
  }
}

function kargerMinCut(n, edges, seed) {
  const rng = new Lcg(seed);
  const label = Array.from({ length: n }, (_, i) => i);
  let remaining = edges.map((edge) => [edge[0], edge[1]]);
  let components = n;
  while (components > 2) {
    const pick = remaining[rng.below(remaining.length)];
    const a = label[pick[0]];
    const b = label[pick[1]];
    for (let i = 0; i < n; i++) {
      if (label[i] === b) label[i] = a;
    }
    remaining = remaining.filter((edge) => label[edge[0]] !== label[edge[1]]);
    components--;
  }
  const first = label[0];
  const sideA = [];
  const sideB = [];
  for (let i = 0; i < n; i++) {
    if (label[i] === first) sideA.push(i);
    else sideB.push(i);
  }
  return [remaining.length, sideA, sideB];
}

function bruteMinCut(n, edges) {
  let best = null;
  for (let mask = 1; mask < 1 << (n - 1); mask++) {
    let cut = 0;
    for (const [u, v] of edges) {
      const bu = u > 0 ? (mask >> (u - 1)) & 1 : 0;
      const bv = v > 0 ? (mask >> (v - 1)) & 1 : 0;
      if (bu !== bv) cut++;
    }
    if (best === null || cut < best) best = cut;
  }
  return best;
}

function checkWitness(n, edges, value, sideA, sideB) {
  if (sideA.length === 0 || sideB.length === 0) throw new Error("FAILED: empty side");
  if (sideA.length + sideB.length !== n) throw new Error("FAILED: not a partition");
  const onA = new Set(sideA);
  const crossing = edges.filter(([u, v]) => onA.has(u) !== onA.has(v)).length;
  if (crossing !== value) throw new Error(`FAILED: crossing ${crossing} != ${value}`);
}

function kargerBestOf(n, edges, baseSeed, trials, expected) {
  let best = null;
  let successes = 0;
  for (let t = 0; t < trials; t++) {
    const seed = (baseSeed * 31 + t * 17 + 1) % 65537;
    const [value, sideA, sideB] = kargerMinCut(n, edges, seed);
    checkWitness(n, edges, value, sideA, sideB);
    if (value < expected) throw new Error(`FAILED: cut ${value} below minimum ${expected}`);
    if (value === expected) successes++;
    if (best === null || value < best) best = value;
  }
  return [best, successes];
}

function lcgGraphs(count, seed, maxN) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    const n = 3 + (seed % (maxN - 2));
    const edges = [];
    for (let node = 1; node < n; node++) {
      seed = (seed * 75 + 74) % 65537;
      const parent = seed % node;
      edges.push([node, parent]);
    }
    const extras = seed % (n + 2);
    for (let e = 0; e < extras; e++) {
      seed = (seed * 75 + 74) % 65537;
      const u = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const v = seed % n;
      if (u !== v) edges.push([u, v]);
    }
    graphs.push([n, edges]);
  }
  return graphs;
}

module.exports = { Lcg, kargerMinCut, bruteMinCut, checkWitness, kargerBestOf, lcgGraphs };

if (require.main === module) {
  const assert = require("node:assert");

  const triangle = [[0, 1], [1, 2], [0, 2]];
  assert.strictEqual(bruteMinCut(3, triangle), 2);
  for (const seed of [1, 2, 3, 65536]) {
    const [value, sideA, sideB] = kargerMinCut(3, triangle, seed);
    checkWitness(3, triangle, value, sideA, sideB);
    assert.strictEqual(value, 2);
  }

  const k4 = [];
  for (let u = 0; u < 4; u++) {
    for (let v = u + 1; v < 4; v++) k4.push([u, v]);
  }
  assert.strictEqual(bruteMinCut(4, k4), 3);
  let [best, successes] = kargerBestOf(4, k4, 42, 60, 3);
  assert.strictEqual(best, 3);
  assert.ok(successes > 0);

  let totalTrials = 0;
  for (const [n, edges] of lcgGraphs(3, 449, 6)) {
    const expected = bruteMinCut(n, edges);
    for (const baseSeed of [1, 42, 1234, 65536]) {
      [best, successes] = kargerBestOf(n, edges, baseSeed, 400, expected);
      assert.strictEqual(best, expected, `n=${n} seed=${baseSeed}`);
      assert.ok(successes > 0, `n=${n} seed=${baseSeed}`);
      totalTrials += 400;
    }
  }
  assert.strictEqual(totalTrials, 4800);

  const k6 = [];
  for (let u = 0; u < 6; u++) {
    for (let v = u + 1; v < 6; v++) k6.push([u, v]);
  }
  assert.strictEqual(bruteMinCut(6, k6), 5);
  [best, successes] = kargerBestOf(6, k6, 1234, 400, 5);
  assert.strictEqual(best, 5);
  assert.ok(successes > 0);

  const parallel = [[0, 1], [1, 0], [0, 1]];
  assert.strictEqual(bruteMinCut(2, parallel), 3);
  [best, successes] = kargerBestOf(2, parallel, 7, 25, 3);
  assert.strictEqual(best, 3);
  assert.strictEqual(successes, 25);
  console.log("PASS");
}
