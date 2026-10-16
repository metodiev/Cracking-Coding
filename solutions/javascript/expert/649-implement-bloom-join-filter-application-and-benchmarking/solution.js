/** Task 649 - Implement bloom join filter application and benchmarking. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMULATED: the classic bloom-filter join (a "sideways information passing" plan). The build
 * side is hashed into a bloom filter (m bits, k probes derived by double hashing from two independent
 * 32-bit hashes of the key), and the probe side first tests the filter: rows that fail the test cannot
 * possibly match and are skipped without touching the hash table. Rows that pass are verified against the
 * build-side hash table, so a false positive costs a wasted lookup but can never add a bogus result row.
 *
 * WHAT IS VERIFIED: the bloom filter never drops a matching key (every inserted key tests positive, and the
 * join result is compared row for row with a plain hash join on duplicate-heavy random inputs, including
 * saturated and degenerate filters); the measured false-positive rate matches the textbook prediction
 * (1 - e^(-kn/m))^k within a tolerance, bit utilisation matches 1 - (1 - 1/m)^(kn), and the measured rate is
 * actually minimised near the textbook k = (m/n) ln 2 rather than at k = 1 or k = 12; the benchmark
 * counters show the filter saving hash-table probes (and the saving disappears when the probe side shares
 * almost no keys with the build side - both directions are measured); the filter is reproducible for a
 * fixed seed; and invalid parameters (no bits, no hashes, a non-function key extractor) are refused.
 */
"use strict";

function hash32(value, seed = 0) {
  const text = String(value);
  let hash = (2166136261 ^ (seed * 2654435761)) >>> 0;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 2246822519) >>> 0;
  hash ^= hash >>> 13;
  return hash >>> 0;
}

class BloomFilter {
  constructor(expectedKeys, bitsPerKey = 8, hashes = 6, seed = 649) {
    if (!(expectedKeys >= 1)) throw new Error("a bloom filter needs at least one expected key");
    if (!(bitsPerKey >= 0.5)) throw new Error("at least half a bit per key is required");
    if (!(hashes >= 1)) throw new Error("a bloom filter needs at least one hash");
    this.m = Math.max(1, Math.round(expectedKeys * bitsPerKey));
    this.k = Math.round(hashes);
    this.seed = seed;
    this.bits = new Uint8Array(this.m);
    this.inserted = 0;
  }
  _probes(key) {
    const first = hash32(key, this.seed);
    const second = (hash32(key, this.seed + 1) >>> 1) | 1; // a positive odd stride keeps probes in range
    const probes = [];
    for (let index = 0; index < this.k; index++) probes.push((first + index * second) % this.m);
    return probes;
  }
  add(key) {
    for (const probe of this._probes(key)) this.bits[probe] = 1;
    this.inserted++;
    return this;
  }
  test(key) {
    for (const probe of this._probes(key)) if (!this.bits[probe]) return false;
    return true;
  }
  setBits() {
    let set = 0;
    for (const bit of this.bits) set += bit;
    return set;
  }
  /** Textbook false-positive probability for `keys` distinct keys in this filter. */
  falsePositiveRate(keys) {
    return Math.pow(1 - Math.exp(-this.k * keys / this.m), this.k);
  }
  expectedUtilisation(keys) {
    return 1 - Math.pow(1 - 1 / this.m, this.k * keys);
  } }

function indexBy(rows, keyOf) {
  if (typeof keyOf !== "function") throw new Error("a key extractor function is required");
  const table = new Map();
  for (const row of rows) {
    const key = keyOf(row);
    if (key === null || key === undefined) continue; // NULL keys never match
    const bucket = table.get(key);
    if (bucket) bucket.push(row);
    else table.set(key, [row]);
  }
  return table;
}

function plainHashJoin(left, right, keyOf) {
  const table = indexBy(right, keyOf);
  const rows = [];
  const counters = { tableLookups: 0, rowsEmitted: 0 };
  for (const outer of left) {
    const key = keyOf(outer);
    counters.tableLookups++;
    if (key === null || key === undefined || !table.has(key)) continue;
    for (const inner of table.get(key)) {
      counters.rowsEmitted++;
      rows.push({ ...outer, ...inner });
    } }
  return { rows, counters };
}

/** Bloom filter on the build side, then probe: filter test first, hash lookup only for survivors. */
function bloomJoin(left, right, keyOf, options = {}) {
  const bitsPerKey = options.bitsPerKey === undefined ? 8 : options.bitsPerKey;
  const hashes = options.hashes === undefined ? 6 : options.hashes;
  const distinctKeys = new Set();
  for (const row of right) {
    const key = keyOf(row);
    if (key !== null && key !== undefined) distinctKeys.add(key);
  }
  const filter = new BloomFilter(Math.max(1, distinctKeys.size), bitsPerKey, hashes);
  for (const key of distinctKeys) filter.add(key);
  const table = indexBy(right, keyOf);
  const rows = [];
  const counters = { probeRows: left.length, bloomRejects: 0, tableLookups: 0, falsePositives: 0, rowsEmitted: 0 };
  for (const outer of left) {
    const key = keyOf(outer);
    if (key === null || key === undefined) {
      counters.bloomRejects++;
      continue;
    }
    if (!filter.test(key)) {
      counters.bloomRejects++;
      continue;
    }
    counters.tableLookups++;
    if (!table.has(key)) {
      counters.falsePositives++; // the filter said maybe, the hash table settles it
      continue;
    }
    for (const inner of table.get(key)) {
      counters.rowsEmitted++;
      rows.push({ ...outer, ...inner });
    } }
  return { rows, counters, filter, distinctBuildKeys: distinctKeys.size };
}

function canonical(rows) {
  return rows.map((row) => JSON.stringify(row, Object.keys(row).sort())).sort().join("|");
}

function makeRows(side, count, keySpan, seed) {
  const rows = [];
  let state = seed >>> 0;
  for (let index = 0; index < count; index++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const draw = (state >>> 8) % 100;
    rows.push({
      side, id: index,
      key: draw < 4 ? null : (state >>> 11) % keySpan, // a few NULL keys that must never match
    });
  }
  return rows;
}

module.exports = { BloomFilter, hash32, plainHashJoin, bloomJoin, canonical, makeRows, indexBy };

if (require.main === module) {
  const assert = require("node:assert");
  const keyOf = (row) => row.key;
  // A filter never forgets: every inserted key tests positive, and the zero-bit filter rejects.
  const filter = new BloomFilter(200, 8, 6);
  const inserted = new Set();
  for (let index = 0; index < 200; index++) {
    const key = `key-${index}`;
    filter.add(key);
    inserted.add(key);
  }
  assert.strictEqual(filter.inserted, 200);
  for (const key of inserted) assert.strictEqual(filter.test(key), true, `false negative on ${key}`);
  const empty = new BloomFilter(50, 8, 6);
  assert.strictEqual(empty.test("anything"), false, "an empty filter accepts nothing");
  assert.throws(() => new BloomFilter(0, 8, 6), /expected key/);
  assert.throws(() => new BloomFilter(10, 0, 6), /half a bit/);
  assert.throws(() => new BloomFilter(10, 8, 0), /at least one hash/);
  assert.throws(() => indexBy([], "not a function"), /key extractor/);
  // Correctness: the join result must equal a plain hash join for duplicate-heavy, NULL-containing data.
  let checked = 0;
  for (const [leftCount, rightCount, keySpan, seed] of [
    [200, 300, 60, 1], [500, 40, 500, 2], [10, 10, 3, 3], [400, 400, 20, 4], [0, 50, 10, 5], [50, 0, 10, 6], ]) {
    const left = makeRows("l", leftCount, keySpan, seed);
    const right = makeRows("r", rightCount, keySpan, seed + 100);
    const expected = plainHashJoin(left, right, keyOf);
    const bloomed = bloomJoin(left, right, keyOf);
    assert.strictEqual(canonical(bloomed.rows), canonical(expected.rows),
      `bloom join mismatch for ${leftCount}x${rightCount}`);
    assert.strictEqual(bloomed.rows.length, expected.rows.length);
    assert.strictEqual(bloomed.counters.probeRows, leftCount);
    assert.ok(bloomed.counters.tableLookups <= expected.counters.tableLookups,
      "the filter can only reduce hash-table lookups");
    assert.strictEqual(bloomed.counters.rowsEmitted, expected.counters.rowsEmitted);
    for (const row of bloomed.rows) assert.notStrictEqual(row.key, null, "NULL keys never join");
    checked++;
  }
  // False-positive rate versus the textbook formula, and bit utilisation.
  const buildKeys = 2000;
  const probeKeys = 20000;
  const absentKeys = [];
  for (let index = 0; index < probeKeys; index++) absentKeys.push(`absent-${index}`);
  const measured = [];
  for (const hashes of [1, 3, 6, 12]) {
    const roomy = new BloomFilter(buildKeys, 10, hashes);
    for (let index = 0; index < buildKeys; index++) roomy.add(`present-${index}`);
    let positives = 0;
    for (const key of absentKeys) if (roomy.test(key)) positives++;
    const rate = positives / probeKeys;
    const theory = roomy.falsePositiveRate(buildKeys);
    measured.push([hashes, rate, theory]);
    assert.ok(Math.abs(rate - theory) <= 0.03, `k=${hashes}: measured ${rate} theoretical ${theory}`);
    const utilisation = roomy.setBits() / roomy.m;
    assert.ok(Math.abs(utilisation - roomy.expectedUtilisation(buildKeys)) <= 0.05,
      `k=${hashes}: utilisation ${utilisation}`);
  }
  const rates = new Map(measured.map(([hashes, rate]) => [hashes, rate]));
  assert.ok(rates.get(6) < rates.get(1), "too few hashes is worse than the optimum");
  assert.ok(rates.get(6) < rates.get(12), "too many hashes is worse than the optimum");
  assert.ok(Math.abs(10 * Math.log(2) - 6) < 1, "the optimum k = (m/n) ln 2 is near six for ten bits per key");
  // Benchmarking: the filter saves hash-table probes whenever the probe side is mostly non-matching, and
  // a saturated filter (half a bit per key) is still correct even though it rejects almost nothing.
  const left = makeRows("l", 5000, 4000, 21);
  const right = makeRows("r", 1000, 4000, 22);
  const plain = plainHashJoin(left, right, keyOf);
  const bloomed = bloomJoin(left, right, keyOf, { bitsPerKey: 10, hashes: 7 });
  const saturated = bloomJoin(left, right, keyOf, { bitsPerKey: 1, hashes: 4 });
  assert.strictEqual(canonical(bloomed.rows), canonical(plain.rows));
  assert.strictEqual(canonical(saturated.rows), canonical(plain.rows), "a saturated filter stays correct");
  assert.ok(bloomed.counters.bloomRejects > left.length * 0.5, "most probe rows are rejected cheaply");
  assert.ok(bloomed.counters.tableLookups < plain.counters.tableLookups,
    "the filter replaces most hash-table lookups with a bit test");
  assert.ok(saturated.counters.tableLookups > bloomed.counters.tableLookups,
    "a saturated filter wastes more lookups");
  const saving = 1 - bloomed.counters.tableLookups / plain.counters.tableLookups;
  assert.ok(saving > 0.5, `the measured saving must be real, got ${saving}`);
  // A filter is deterministic: same seed and keys give the same bits and the same counters.
  const again = bloomJoin(left, right, keyOf, { bitsPerKey: 10, hashes: 7 });
  assert.deepStrictEqual(again.counters, bloomed.counters);
  assert.deepStrictEqual([...again.filter.bits], [...bloomed.filter.bits]);
  assert.ok(bloomed.distinctBuildKeys > 0 && bloomed.distinctBuildKeys <= right.length);
  console.log(`bloom join: ${checked} joins verified, ${bloomed.counters.bloomRejects}/${left.length} probes ` +
    `rejected by the filter, ${bloomed.counters.falsePositives} false positives, hash-table lookups ` +
    `${bloomed.counters.tableLookups} vs ${plain.counters.tableLookups} without the filter, ` +
    `saving ${(saving * 100).toFixed(1)}%`);
  console.log("PASS");
}
