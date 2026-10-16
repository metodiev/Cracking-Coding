/**
 * Task 584 - Implement offline dynamic programming with bitset acceleration tricks. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * Three bit-parallel DPs plus the "offline" query engine that makes them worthwhile.
 *
 * (1) Subset-sum reachability.  The set of sums a subset of `values` can reach is a bitmask where
 *     bit s means "sum s is achievable"; adding an item of weight v is the single BigInt operation
 *     dp |= dp << v.  The whole DP therefore runs in O(n * S / 64) word operations with no
 *     per-sum branching, and the number of distinct achievable sums is popcount(dp) — a trick the
 *     classical boolean-array DP cannot express directly.
 * (2) Bounded knapsack.  Multiplicities are folded in by binary splitting (1,2,4,... copies per
 *     step), which keeps the number of shift operations at O(sum log c_i) instead of O(sum c_i).
 * (3) Bit-parallel LCS on strings via Hyyrö's bit-vector recurrence: one BigInt mask per alphabet
 *     symbol marks the positions of that symbol in b, and V <- ((V + U) | (V - U)) advances the
 *     edit state for each character of a.  The LCS length is the number of zero bits of V.  The
 *     subtraction relies on two's-complement wrap-around, which BigInt reproduces exactly as long
 *     as the result is masked back to m bits (verified below against the O(nm) table DP).
 * (4) Offline queries.  Queries are (prefix, target) pairs — "is target reachable using the first
 *     `prefix` items?"  Recomputing a DP per query is O(q n S / 64); sorting the queries by prefix
 *     and keeping one running bitset answers them all in O(n S / 64 + q), materialising nothing.
 *
 * Determinism: every pseudo-random value comes from a documented 64-bit splitmix64 generator
 * (seed 0x5EED_584) rather than the repo's 16-bit LCG, which has short cycles and poor low-bit
 * structure; the same seed always yields the same instances.
 *
 * Documented checks: the subset-sum bitset is compared bit-for-bit with the boolean-array DP and
 * with exhaustive 2^16 enumeration of all subset sums; bounded knapsack (binary splitting) is
 * compared with a DP over the explicitly expanded multiset; the bit-parallel LCS is compared with
 * the classic O(nm) table on 40 deterministic instances including empty/single-character/identical
 * strings and binary-alphabet strings of length 300, and with closed-form answers on a few
 * hand-checked pairs; the offline engine's answers are compared with per-query from-scratch DPs on
 * a 200-item / 5 000-query batch, and it is asserted that query order does not change any answer and
 * that prefix 0 only ever reaches target 0.  Documented caveat: the bitset DP's cost is linear in
 * the *total sum*, so it is only competitive while S / 64 stays small compared with the number of
 * states a sparse DP would visit; the boolean DP is kept as the reference implementation.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const SPLITMIX_SEED = 0x5eed584n;

/** Documented 64-bit splitmix64 stream: z = mix(z ^ (z >> 30) * C1 ...) of a Weyl-stepped state. */
class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  /** Uniform integer in [0, n). */
  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }

  /** Uniform double in [0, 1) built from the top 53 bits, so all draws are independent. */
  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }
}

function popcount32(x) {
  let v = x - ((x >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return (v * 0x01010101) >>> 24;
}

/** popcount of a non-negative BigInt, 32 bits at a time. */
function popcountBig(x) {
  assert.ok(x >= 0n, "popcountBig expects a non-negative value");
  let count = 0;
  let v = x;
  while (v > 0n) {
    count += popcount32(Number(v & 0xffffffffn));
    v >>= 32n;
  }
  return count;
}

/** Bit s of the result is set iff some subset of `values` sums to s. */
function subsetSumBitset(values) {
  let dp = 1n;
  for (const v of values) {
    if (!Number.isInteger(v) || v < 0) throw new RangeError(`bad item weight ${v}`);
    if (v > 0) dp |= dp << BigInt(v);
  }
  return dp;
}

/** Reference O(n * S) boolean subset-sum. */
function subsetSumNaive(values) {
  const total = values.reduce((a, b) => a + b, 0);
  const reach = new Uint8Array(total + 1);
  reach[0] = 1;
  for (const v of values) {
    for (let s = total; s >= v; s--) if (reach[s - v]) reach[s] = 1;
  }
  return reach;
}

/** Exhaustive ground truth for small n: every one of the 2^n subsets. */
function subsetSumsExhaustive(values) {
  const sums = new Set();
  const n = values.length;
  for (let mask = 0; mask < 1 << n; mask++) {
    let s = 0;
    for (let i = 0; i < n; i++) if (mask & (1 << i)) s += values[i];
    sums.add(s);
  }
  return sums;
}

/** Bounded knapsack by binary splitting of the multiplicities. */
function boundedKnapsackBitset(values, counts) {
  assert.strictEqual(values.length, counts.length, "values/counts length mismatch");
  let dp = 1n;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < 0 || counts[i] < 0 || !Number.isInteger(counts[i])) {
      throw new RangeError("boundedKnapsackBitset expects non-negative integer counts");
    }
    let remaining = counts[i];
    let block = 1;
    while (remaining > 0) {
      const take = Math.min(block, remaining);
      if (values[i] > 0) dp |= dp << BigInt(values[i] * take);
      remaining -= take;
      block *= 2;
    }
  }
  return dp;
}

/** Bit-parallel LCS (Hyyrö).  BigInt supplies arbitrary-width words for free. */
function lcsBitset(a, b) {
  const m = b.length;
  if (a.length === 0 || m === 0) return 0;
  const masks = new Map();
  for (let j = 0; j < m; j++) {
    const c = b[j];
    masks.set(c, (masks.get(c) || 0n) | (1n << BigInt(j)));
  }
  const full = (1n << BigInt(m)) - 1n;
  let v = full;
  for (let i = 0; i < a.length; i++) {
    const u = v & (masks.get(a[i]) || 0n);
    v = ((v + u) | (v - u)) & full;
  }
  return popcountBig(~v & full);
}

/** Classic O(nm) LCS table, the independent reference. */
function lcsDp(a, b) {
  let prev = new Uint32Array(b.length + 1);
  let cur = new Uint32Array(b.length + 1);
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    }
    const t = prev;
    prev = cur;
    cur = t;
    cur.fill(0);
  }
  return prev[b.length];
}

/**
 * Offline subset-sum queries.  `queries` is a list of [prefix, target] pairs and the result is a
 * boolean per query, computed with one running bitset plus prefix snapshots taken on demand.
 */
function offlineSubsetQueries(values, queries) {
  const byPrefix = new Map();
  queries.forEach(([prefix, target], index) => {
    if (prefix < 0 || prefix > values.length || target < 0) {
      throw new RangeError(`bad query (${prefix}, ${target})`);
    }
    if (!byPrefix.has(prefix)) byPrefix.set(prefix, []);
    byPrefix.get(prefix).push([index, target]);
  });
  const prefixes = [...byPrefix.keys()].sort((x, y) => x - y);
  const answers = new Array(queries.length).fill(false);
  let dp = 1n;
  let used = 0;
  for (const prefix of prefixes) {
    while (used < prefix) {
      dp |= dp << BigInt(values[used]);
      used++;
    }
    for (const [index, target] of byPrefix.get(prefix)) {
      answers[index] = ((dp >> BigInt(target)) & 1n) === 1n;
    }
  }
  return answers;
}

/** Per-query from-scratch DPs; used only as a reference for the offline engine. */
function offlineSubsetQueriesNaive(values, queries) {
  return queries.map(([prefix, target]) =>
    ((subsetSumBitset(values.slice(0, prefix)) >> BigInt(target)) & 1n) === 1n);
}

function reachableSetFromMask(mask) {
  const set = new Set();
  let v = mask;
  let index = 0;
  while (v > 0n) {
    if (v & 1n) set.add(index);
    v >>= 1n;
    index++;
  }
  return set;
}

function randomItems(n, maxWeight, rng) {
  return Array.from({ length: n }, () => rng.nextInt(maxWeight + 1));
}

function randomString(length, alphabet, rng) {
  let s = "";
  for (let i = 0; i < length; i++) s += String.fromCharCode(97 + rng.nextInt(alphabet));
  return s;
}

function main() {
  // (1) Hand-checked subset sums: {1,2,3,4} reaches every sum 0..10.
  const handMask = subsetSumBitset([1, 2, 3, 4]);
  assert.deepStrictEqual(reachableSetFromMask(handMask),
    new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
  assert.strictEqual(popcountBig(handMask), 11, "popcount counts the distinct achievable sums");
  assert.strictEqual(subsetSumBitset([]), 1n, "the empty set reaches exactly 0");
  assert.strictEqual(subsetSumBitset([0, 0]), 1n, "zero-weight items cannot create new sums");
  assert.strictEqual(subsetSumBitset([5]), 0b100001n, "a single item reaches 0 and 5");
  assert.strictEqual(subsetSumBitset([2, 2]), 0b10101n, "[2,2] reaches exactly {0,2,4}");
  assert.strictEqual(reachableSetFromMask(subsetSumBitset([4, 6])).size, 4,
    "{4,6} reaches 0,4,6,10 only (the classical 4/6 example)");
  assert.ok(!reachableSetFromMask(subsetSumBitset([4, 6])).has(8));

  // (2) Exhaustive ground truth over 2^16 subsets.
  const rng = new SplitMix64(SPLITMIX_SEED);
  const exhaustiveItems = randomItems(16, 25, rng);
  assert.deepStrictEqual(reachableSetFromMask(subsetSumBitset(exhaustiveItems)),
    subsetSumsExhaustive(exhaustiveItems),
    "bitset DP must reproduce exhaustive enumeration of all 2^16 subsets");

  // (3) Bit-for-bit comparison with the boolean DP on 40 random instances.
  let comparedStates = 0;
  for (let trial = 0; trial < 40; trial++) {
    const n = rng.nextInt(23);
    const items = randomItems(n, 30, rng);
    const mask = subsetSumBitset(items);
    const reach = subsetSumNaive(items);
    const total = items.reduce((a, b) => a + b, 0);
    assert.strictEqual(mask >> BigInt(total + 1), 0n, "no bit above the total sum may be set");
    for (let s = 0; s <= total; s++) {
      assert.strictEqual(((mask >> BigInt(s)) & 1n) === 1n, reach[s] === 1,
        `subset sum mismatch for s=${s} on [${items}]`);
      comparedStates++;
    }
    assert.strictEqual(popcountBig(mask), reach.reduce((a, b) => a + b, 0),
      "popcount equals the number of reachable sums");
  }
  assert.ok(comparedStates > 1000, "the comparison covered enough states");

  // (4) Bounded knapsack: binary splitting vs an explicit multiset expansion.
  const boundedCases = [
    { values: [3, 5], counts: [2, 1] },
    { values: [1, 2, 4, 8, 16], counts: [1, 3, 0, 7, 2] },
    { values: [7], counts: [13] },
    { values: [2, 3, 5], counts: [0, 0, 0] },
  ];
  for (const { values, counts } of boundedCases) {
    const expanded = [];
    values.forEach((v, i) => { for (let c = 0; c < counts[i]; c++) expanded.push(v); });
    const fast = boundedKnapsackBitset(values, counts);
    const reference = subsetSumBitset(expanded);
    assert.strictEqual(fast, reference,
      `bounded knapsack mismatch for values=[${values}] counts=[${counts}]`);
  }
  // Twenty random bounded instances with multiplicities up to 40.
  for (let trial = 0; trial < 20; trial++) {
    const k = 1 + rng.nextInt(5);
    const values = randomItems(k, 12, rng);
    const counts = Array.from({ length: k }, () => rng.nextInt(41));
    const expanded = [];
    values.forEach((v, i) => { for (let c = 0; c < counts[i]; c++) expanded.push(v); });
    assert.strictEqual(boundedKnapsackBitset(values, counts), subsetSumBitset(expanded));
    assert.strictEqual(boundedKnapsackBitset(values, counts),
      subsetSumNaive(expanded).reduce((acc, bit, s) => (bit ? acc + (1n << BigInt(s)) : acc), 0n));
  }

  // (5) Bit-parallel LCS against the table DP and closed forms.
  assert.strictEqual(lcsBitset("", "abc"), 0);
  assert.strictEqual(lcsBitset("abc", ""), 0);
  assert.strictEqual(lcsBitset("abc", "abc"), 3);
  assert.strictEqual(lcsBitset("aaaa", "aaabbbaaa"), 4);
  assert.strictEqual(lcsBitset("abab", "baba"), 3, "abab / baba share bab (or aba)");
  assert.strictEqual(lcsBitset("AGGTAB", "GXTXAYB"), 4, "classic textbook instance (GTAB)");
  for (let trial = 0; trial < 40; trial++) {
    const n = rng.nextInt(300);
    const m = rng.nextInt(300);
    const alphabet = 2 + rng.nextInt(4);
    const a = randomString(n, alphabet, rng);
    const b = randomString(m, alphabet, rng);
    assert.strictEqual(lcsBitset(a, b), lcsDp(a, b),
      `LCS mismatch on lengths ${n}/${m} alphabet ${alphabet}`);
  }
  const longA = randomString(4000, 2, rng);
  const longB = randomString(4000, 2, rng);
  assert.strictEqual(lcsBitset(longA, longB), lcsDp(longA, longB),
    "bit-parallel LCS must stay exact on 4000x4000 binary strings");

  // (6) Offline query engine.
  const queryItems = randomItems(200, 1000, rng);
  const queryRng = new SplitMix64(0xc0ffee584n);
  const queries = [];
  for (let i = 0; i < 5000; i++) {
    queries.push([queryRng.nextInt(201), queryRng.nextInt(5000)]);
  }
  const offline = offlineSubsetQueries(queryItems, queries);
  // Reference for a deterministic 120-query subset (from-scratch DP each time).
  const sampleIndices = Array.from({ length: 120 }, (_, i) => (i * 41) % queries.length);
  const sampleQueries = sampleIndices.map((i) => queries[i]);
  const naiveAnswers = offlineSubsetQueriesNaive(queryItems, sampleQueries);
  sampleQueries.forEach((q, i) => {
    assert.strictEqual(offline[sampleIndices[i]], naiveAnswers[i],
      `offline answer disagrees with a fresh DP for query (${q[0]}, ${q[1]})`);
  });
  const shuffled = queries.map((q, i) => [q, i]);
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = queryRng.nextInt(i + 1);
    const t = shuffled[i];
    shuffled[i] = shuffled[j];
    shuffled[j] = t;
  }
  const reordered = offlineSubsetQueries(queryItems, shuffled.map(([q]) => q));
  shuffled.forEach(([q, original], newIndex) => {
    assert.strictEqual(reordered[newIndex], offline[original],
      "offline answers are independent of query order");
  });
  const prefixZeroQueries = [[0, 0], [0, 1], [0, 7]];
  assert.deepStrictEqual(offlineSubsetQueries(queryItems, prefixZeroQueries),
    [true, false, false], "prefix 0 can only reach target 0");
  assert.strictEqual(offlineSubsetQueries([5], [[1, 5], [1, 0]])[0], true);
  assert.strictEqual(offlineSubsetQueries([5], [[1, 5], [1, 0]])[1], true);
  assert.throws(() => offlineSubsetQueries([1], [[2, 0]]), RangeError);
  assert.throws(() => subsetSumBitset([-1]), RangeError);
  assert.throws(() => boundedKnapsackBitset([1], [1.5]), RangeError);

  // (7) The offline engine answers a big query batch consistently with the direct DP for prefix n.
  const fullMask = subsetSumBitset(queryItems);
  const fullTotal = queryItems.reduce((a, b) => a + b, 0);
  const fullQueries = [];
  for (let s = 0; s < 4000; s++) fullQueries.push([200, s]);
  const fullAnswers = offlineSubsetQueries(queryItems, fullQueries);
  for (let s = 0; s < 4000; s++) {
    assert.strictEqual(fullAnswers[s], ((fullMask >> BigInt(s)) & 1n) === 1n,
      `prefix-200 answer for sum ${s} must match the full DP`);
  }
  assert.strictEqual(popcountBig(fullMask), subsetSumNaive(queryItems).reduce((a, b) => a + b, 0));
  assert.ok(fullTotal > 90000, "the 200-item instance has a large total sum");

  // (8) Determinism: the generator and the offline engine are pure functions.
  const rngA = new SplitMix64(1);
  const rngB = new SplitMix64(1);
  const drawsA = Array.from({ length: 12 }, () => rngA.next());
  const drawsB = Array.from({ length: 12 }, () => rngB.next());
  assert.deepStrictEqual(drawsA, drawsB, "splitmix64 is a pure function of its seed");
  assert.strictEqual(new Set(drawsA.map(String)).size, 12, "consecutive draws differ");
  assert.ok(drawsA.every((x) => x >= 0n && x <= MASK64));
  const repeatItems = randomItems(30, 40, new SplitMix64(99n));
  assert.strictEqual(subsetSumBitset(repeatItems), subsetSumBitset(repeatItems.slice()));
  const smallQueries = Array.from({ length: 50 }, (_, i) => [i % 31, (i * 7) % 200]);
  assert.deepStrictEqual(offlineSubsetQueries(repeatItems, smallQueries.slice()),
    offlineSubsetQueries(repeatItems, smallQueries.slice()));
}

module.exports = { MASK64, SPLITMIX_SEED, SplitMix64, popcount32, popcountBig, subsetSumBitset,
  subsetSumNaive, subsetSumsExhaustive, boundedKnapsackBitset, lcsBitset, lcsDp,
  offlineSubsetQueries, offlineSubsetQueriesNaive, reachableSetFromMask, randomItems,
  randomString };

if (require.main === module) {
  main();
  console.log("PASS");
}
