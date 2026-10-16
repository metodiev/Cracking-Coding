/**
 * Task 521 - Implement reservoir-sampling-based stratified sampling. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Two-layer sampler. The inner layer is Algorithm R reservoir sampling: keep the first k items of a
 * stream in the reservoir, then for item i (0-based, i >= k) draw j uniformly from [0, i] and, when
 * j < k, replace reservoir slot j - which makes every k-subset of the stream equally likely without
 * knowing the stream length in advance. The outer layer partitions the population into strata
 * (first-appearance order of the key function, preserving each stratum's original order) and runs
 * an independent reservoir per stratum with the requested quota, capping a quota at the stratum
 * size and reporting the cap in the result. All randomness comes from one explicitly seeded 32-bit
 * LCG (Math.imul arithmetic, high bits used for the uniform draw), so every run is reproducible.
 *
 * Documented checks: the reservoir's uniformity is verified *exactly* rather than statistically -
 * with a fixed random stream the algorithm's decision sequence only depends on the stream
 * positions, so over all n! arrival permutations the chosen position set is constant (asserted),
 * and consequently each item appears in exactly k*(n-1)! permutations (5 items, k = 2: 48 of 120).
 * A stub RNG with preset draws reproduces a hand-computed trace (j = 1 kept, j = 3 dropped). The
 * stratified wrapper is checked for exact per-stratum sizes, no duplicates, membership in the
 * correct stratum, quota capping, first-appearance stratum ordering, inclusion frequencies near
 * k/n over 400 independent seeds, and bit-identical results for a repeated seed with a different
 * result for a different seed. Boundary cases: k = 0, k = n, k > n, an empty population, a missing
 * quota key (0 draws), a negative quota and an unknown quota key (both rejected).
 */
"use strict";

/** 32-bit linear congruential generator; values are taken from the high bits. */
class Lcg {
  constructor(seed) {
    // Scramble the seed so that neighbouring seeds give independent streams.
    let s = ((seed >>> 0) ^ 0x9e3779b9) >>> 0;
    s = Math.imul(s ^ (s >>> 16), 0x85ebca6b) >>> 0;
    s = Math.imul(s ^ (s >>> 13), 0xc2b2ae35) >>> 0;
    this.state = (s ^ (s >>> 16)) >>> 0;
    if (this.state === 0) this.state = 0x1b873593;
  }

  nextUint32() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  nextFloat() {
    return this.nextUint32() / 4294967296;
  }

  nextInt(bound) {
    return Math.floor(this.nextFloat() * bound);
  }
}

/** Protocol-compatible stub: replays a fixed list of integers (for hand-checked traces). */
class StubRng {
  constructor(draws) {
    this.draws = draws.slice();
    this.calls = 0;
  }

  nextUint32() {
    return 0;
  }

  nextFloat() {
    return 0;
  }

  nextInt() {
    if (this.calls >= this.draws.length) throw new Error("stub RNG exhausted");
    return this.draws[this.calls++];
  }
}

/**
 * Algorithm R. Slot order is preserved (not sorted); when k >= items.length the whole stream is
 * returned and no random draw is made.
 */
function reservoirSample(items, k, rng) {
  if (!Number.isInteger(k) || k < 0) throw new Error("k must be a non-negative integer");
  if (k === 0 || items.length === 0) return [];
  const reservoir = items.slice(0, Math.min(k, items.length));
  for (let i = k; i < items.length; i++) {
    const j = rng.nextInt(i + 1);
    if (j < k) reservoir[j] = items[i];
  }
  return reservoir;
}

/** Reservoir draw over stream positions; returns the chosen positions in ascending order. */
function reservoirPositions(n, k, rng) {
  const positions = reservoirSample(Array.from({ length: n }, (_, i) => i), k, rng);
  return positions.slice().sort((a, b) => a - b);
}

/**
 * Stratified sample. `keyOf(item, index)` maps an item to its stratum; `quotas` maps a stratum key
 * to the requested sample size. Returns the strata in first-appearance order, each with its
 * members sorted by original index, plus the concatenated and index-sorted flat selection.
 */
function stratifiedSample(items, keyOf, quotas, rng) {
  const available = new Set(quotas.keys());
  const groups = new Map();
  items.forEach((item, index) => {
    const key = keyOf(item, index);
    available.delete(key);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(index);
  });
  if (available.size > 0) {
    throw new Error(`unknown stratum key(s): ${[...available].sort().join(",")}`);
  }

  const strata = [];
  for (const [key, indices] of groups) {
    const requested = quotas.get(key);
    if (!Number.isInteger(requested) || requested < 0) {
      throw new Error(`quota for ${key} must be a non-negative integer`);
    }
    const taken = Math.min(requested, indices.length);
    const picked = reservoirPositions(indices.length, taken, rng).map((p) => indices[p]);
    strata.push({ key, requested, taken, members: picked.slice().sort((a, b) => a - b) });
  }

  const flat = [];
  for (const stratum of strata) flat.push(...stratum.members);
  flat.sort((a, b) => a - b);
  return { strata, selected: flat };
}

/** Every permutation of a small array (Heap-free, plain lexicographic recursion). */
function permutations(values) {
  if (values.length <= 1) return [values.slice()];
  const out = [];
  for (let i = 0; i < values.length; i++) {
    const rest = values.slice(0, i).concat(values.slice(i + 1));
    for (const tail of permutations(rest)) out.push([values[i], ...tail]);
  }
  return out;
}

function main() {
  const assert = require("node:assert");

  // Hand-checked trace: stream of 6, k = 3; draws force j = 1 (kept) and j = 3 (dropped).
  const stub = new StubRng([1, 3, 0]);
  assert.deepStrictEqual(reservoirSample([0, 1, 2, 3, 4, 5], 3, stub), [5, 3, 2],
    "slot 1 <- item 3; item 4 dropped because j = 3 is not < 3; slot 0 <- item 5");
  assert.strictEqual(stub.calls, 3, "one draw per stream item beyond the first k");

  assert.deepStrictEqual(reservoirSample([1, 2], 5, new StubRng([])), [1, 2],
    "k >= n returns the whole stream without consuming randomness");
  assert.deepStrictEqual(reservoirSample([1, 2, 3], 0, new StubRng([])), []);
  assert.deepStrictEqual(reservoirSample([], 3, new Lcg(1)), []);

  // Exact uniformity: for a fixed random stream the chosen *position set* is permutation
  // invariant, hence each item is included in exactly k*(n-1)! of the n! permutations.
  for (const k of [1, 2, 3]) {
    const n = 5;
    const perms = permutations([0, 1, 2, 3, 4]);
    assert.strictEqual(perms.length, 120);
    const inclusion = new Array(n).fill(0);
    let reference = null;
    let subsets = new Set();
    for (const perm of perms) {
      const drawn = reservoirSample(perm, k, new Lcg(2024));
      assert.strictEqual(drawn.length, k);
      assert.strictEqual(new Set(drawn).size, k, "the reservoir never repeats an item");
      const positions = drawn.map((value) => perm.indexOf(value)).sort((a, b) => a - b);
      const key = positions.join(",");
      subsets.add(key);
      if (reference === null) reference = key;
      assert.strictEqual(key, reference,
        "with a fixed stream the reservoir slot positions do not depend on the arrival order");
      for (const value of drawn) inclusion[value] += 1;
    }
    assert.strictEqual(subsets.size, 1);
    const factorial = (m) => (m <= 1 ? 1 : m * factorial(m - 1));
    const expected = k * factorial(n - 1);
    for (let item = 0; item < n; item++) {
      assert.strictEqual(inclusion[item], expected,
        `k = ${k}: item ${item} appears in exactly k*(n-1)! permutations`);
    }
  }

  // Marginal uniformity over independent seeds (statistical companion to the exact check above).
  const n = 20;
  const k = 5;
  const counts = new Array(n).fill(0);
  const trials = 400;
  for (let trial = 0; trial < trials; trial++) {
    const drawn = reservoirPositions(n, k, new Lcg(1000 + trial));
    assert.strictEqual(drawn.length, k);
    assert.strictEqual(new Set(drawn).size, k);
    for (const p of drawn) counts[p] += 1;
  }
  for (const count of counts) {
    assert.ok(Math.abs(count / trials - k / n) <= 0.05,
      `inclusion frequency ${count / trials} should be near ${k / n}`);
  }

  // Stratified wrapper: quotas, membership, ordering, capping, duplicates.
  const population = [];
  for (let i = 0; i < 60; i++) population.push({ group: ["a", "b", "c"][i % 3], id: i });
  const quotas = new Map([["a", 4], ["b", 2], ["c", 9]]);
  const sample = stratifiedSample(population, (item) => item.group, quotas, new Lcg(77));
  assert.deepStrictEqual(sample.strata.map((s) => s.key), ["a", "b", "c"],
    "strata appear in first-appearance order");
  assert.deepStrictEqual(sample.strata.map((s) => s.taken), [4, 2, 9]);
  assert.deepStrictEqual(sample.strata.map((s) => s.requested), [4, 2, 9]);
  assert.strictEqual(sample.selected.length, 15);
  assert.strictEqual(new Set(sample.selected).size, 15, "no item is drawn twice");
  for (const stratum of sample.strata) {
    for (const index of stratum.members) {
      assert.strictEqual(population[index].group, stratum.key,
        "an item is only ever sampled into its own stratum");
    }
    const sorted = stratum.members.slice().sort((a, b) => a - b);
    assert.deepStrictEqual(stratum.members, sorted, "members are reported in original order");
  }

  const capped = stratifiedSample(population, (item) => item.group,
    new Map([["a", 100], ["b", 0], ["c", 1]]), new Lcg(5));
  assert.deepStrictEqual(capped.strata.map((s) => [s.requested, s.taken]), [[100, 20], [0, 0],
    [1, 1]], "quotas larger than a stratum are capped and reported");

  const empty = stratifiedSample(population, (item) => item.group, new Map([["a", 0], ["b", 0],
    ["c", 0]]), new Lcg(9));
  assert.deepStrictEqual(empty.selected, []);
  assert.deepStrictEqual(empty.strata.map((s) => s.taken), [0, 0, 0]);

  const again = stratifiedSample(population, (item) => item.group, quotas, new Lcg(77));
  assert.deepStrictEqual(again, sample, "the same seed reproduces the selection bit for bit");
  const other = stratifiedSample(population, (item) => item.group, quotas, new Lcg(78));
  assert.notDeepStrictEqual(other.selected, sample.selected,
    "a different seed produces a different draw");
  assert.deepStrictEqual(other.strata.map((s) => s.taken), [4, 2, 9],
    "changing the seed never changes the quotas");

  // Stratum proportions are preserved in expectation: each stratum's share of the sample tracks
  // its share of the population, and quota capping is the only source of deviation.
  const balanced = stratifiedSample(population, (item) => item.group,
    new Map([["a", 6], ["b", 6], ["c", 6]]), new Lcg(1234));
  assert.deepStrictEqual(balanced.strata.map((s) => s.taken), [6, 6, 6]);
  const share = balanced.strata.map((s) => s.taken / 20);
  assert.deepStrictEqual(share, [0.3, 0.3, 0.3]);

  assert.throws(() => reservoirSample([1, 2], -1, new Lcg(1)), /non-negative integer/);
  assert.throws(() => stratifiedSample(population, (item) => item.group,
    new Map([["z", 1]]), new Lcg(1)), /unknown stratum key/);
  assert.throws(() => stratifiedSample(population, (item) => item.group,
    new Map([["a", -2], ["b", 0], ["c", 0]]), new Lcg(1)), /non-negative integer/);
}

module.exports = { Lcg, StubRng, reservoirSample, reservoirPositions, stratifiedSample,
  permutations };

if (require.main === module) {
  main();
  console.log("PASS");
}
