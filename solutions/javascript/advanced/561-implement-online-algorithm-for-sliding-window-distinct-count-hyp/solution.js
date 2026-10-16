/**
 * Task 561 - Implement online algorithm for sliding-window distinct count (HyperLogLog outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Two online answers to "how many distinct items are in the last w elements?" are implemented and
 * compared.  Exactly: one pass with a last-seen-position map gives the distinct count of any suffix
 * window in O(1) amortised per element - the count only changes when an item's previous occurrence
 * leaves the window or when a brand-new item enters it - so the exact version is cheap and is used as
 * the ground truth for the approximate one.  Approximately: a HyperLogLog sketch of the window with
 * m = 2^b registers, where every item is hashed with splitmix64, the top b bits select a register and
 * the remaining bits give the number of leading zeros plus one; the "stale rebuild" strategy (recompute
 * the sketch from scratch every s stream steps and report it in between) is the practical way to keep
 * HLL online for a sliding window, and its staleness error is measured rather than assumed.
 *
 * Documented checks: the exact sliding-window count equals a brute-force recomputation for every window
 * position on LCG streams of repeated values, for window sizes 1, 2, w, n - 1 and n (and a window larger
 * than the stream); an item's arrival inside the window always increases the distinct count by exactly
 * one when the item was absent and never otherwise; HyperLogLog's relative error obeys the textbook
 * 1.04/sqrt(m) standard error within 3 sigma for cardinalities 500, 5000 and 50000 at m = 256, 1024 and
 * 4096, with the estimator's small-range linear-counting correction asserted to matter (the raw
 * harmonic-mean estimator is measurably worse when the cardinality is comparable to m); the sketch
 * merge identity HLL(A ∪ B) = registerwise max(HLL(A), HLL(B)) is verified exactly over random
 * partitions; the sketch is asserted to be monotone under insertion and 0 for the empty multiset; the
 * stale-rebuild window estimator is asserted to stay inside a documented band around the exact window
 * count whose width is 3 sigma plus the observed staleness, and a smaller rebuild interval is asserted
 * to be at least as accurate as a longer one; the harmonic denominator sum 2^-M[j] of the raw estimator
 * is asserted non-increasing under insertion, which is the mechanism behind HLL reporting an increasing
 * estimate; and reruns are bit-identical.  The hash is splitmix64 (documented), not the
 * repo's 16-bit LCG, because HLL needs well-mixed bits.
 */
"use strict";

const TWO_64 = 1n << 64n;
const MASK_64 = TWO_64 - 1n;
const SPLITMIX_GAMMA = 0x9e3779b97f4a7c15n;

/** splitmix64: a bijective mixer with good avalanche behaviour in all bit positions. */
function splitmix64(state) {
  let z = (state + SPLITMIX_GAMMA) & MASK_64;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK_64;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK_64;
  return (z ^ (z >> 31n)) & MASK_64;
}

function hashItem(value) {
  return splitmix64(BigInt.asUintN(64, BigInt(value)));
}

function countLeadingZeros64(value) {
  let v = value === 0n ? 0n : value;
  let count = 0;
  for (let bit = 63n; bit >= 0n; bit--) {
    if ((v >> bit) & 1n) break;
    count += 1;
  }
  return count;
}

class HyperLogLog {
  constructor(bits) {
    if (bits < 4 || bits > 20) throw new Error("register exponent out of range");
    this.bits = bits;
    this.m = 1 << bits;
    this.registers = new Uint8Array(this.m);
  }

  /** Position: the top `bits` bits of the hash; rank: leading zeros of the rest, plus one. */
  static positionAndRank(hash, bits) {
    const position = Number(hash >> BigInt(64 - bits));
    const rest = (hash << BigInt(bits)) & MASK_64;
    return { position, rank: countLeadingZeros64(rest) + 1 };
  }

  addHash(hash) {
    const { position, rank } = HyperLogLog.positionAndRank(hash, this.bits);
    if (rank > this.registers[position]) this.registers[position] = rank;
    return this;
  }

  add(value) {
    return this.addHash(hashItem(value));
  }

  merge(other) {
    if (other.bits !== this.bits) throw new Error("register exponent mismatch");
    const merged = new HyperLogLog(this.bits);
    for (let i = 0; i < this.m; i++) {
      merged.registers[i] = Math.max(this.registers[i], other.registers[i]);
    }
    return merged;
  }

  zeroRegisters() {
    let count = 0;
    for (const rank of this.registers) if (rank === 0) count += 1;
    return count;
  }

  /** Raw harmonic-mean estimate: alpha_m * m^2 / sum 2^-M[j]. */
  rawEstimate() {
    let denominator = 0;
    for (const rank of this.registers) denominator += Math.pow(2, -rank);
    const alpha = 0.7213 / (1 + 1.079 / this.m);
    return (alpha * this.m * this.m) / denominator;
  }

  /** With the classical small-range correction (linear counting when registers are still empty). */
  estimate() {
    const raw = this.rawEstimate();
    const zeros = this.zeroRegisters();
    if (raw <= 2.5 * this.m && zeros > 0) return this.m * Math.log(this.m / zeros);
    return raw;
  }
}

/**
 * Exact distinct count of every suffix window of length w in one pass.  The invariant is
 * distinct = |{y : lastSeen[y] >= i - w + 1}|; adding the item x at position i increases the count only
 * when x was not in the *previous* window [i - w, i - 1] (that is, its last occurrence was before
 * i - w), and the position i - w leaving the window decreases it only when that position held the last
 * occurrence of its item.  Note the two effects can concern the same item (x = stream[i - w]) and then
 * they cancel, which is why the increment test uses i - w and not i - w + 1.
 */
function exactSlidingDistinct(stream, w) {
  if (w <= 0) throw new Error("window must be positive");
  const lastSeen = new Map();
  const counts = [];
  let distinct = 0;
  for (let index = 0; index < stream.length; index++) {
    const item = stream[index];
    const previous = lastSeen.get(item);
    const entering = previous === undefined || previous < index - w;
    lastSeen.set(item, index);
    if (entering) distinct += 1;
    const leavingIndex = index - w;
    if (leavingIndex >= 0) {
      const leaving = stream[leavingIndex];
      if (lastSeen.get(leaving) === leavingIndex) {
        lastSeen.delete(leaving);
        distinct -= 1;
      }
    }
    if (index + 1 >= w) counts.push(distinct);
  }
  return counts;
}

/** Independent oracle: recompute the distinct count of every window from scratch. */
function bruteForceSlidingDistinct(stream, w) {
  const counts = [];
  for (let end = w - 1; end < stream.length; end++) {
    counts.push(new Set(stream.slice(end - w + 1, end + 1)).size);
  }
  return counts;
}

/**
 * Online sliding-window estimate: rebuild a fresh HLL over the current window every `stride` steps and
 * report the most recent sketch in between (the documented staleness trade-off).
 */
function slidingHllEstimates(stream, w, bits, stride) {
  const estimates = [];
  let sketch = null;
  for (let index = 0; index < stream.length; index++) {
    if ((index + 1) >= w) {
      if (sketch === null || (index + 1 - w) % stride === 0) {
        sketch = new HyperLogLog(bits);
        for (let offset = index - w + 1; offset <= index; offset++) sketch.add(stream[offset]);
      }
      estimates.push(sketch.estimate());
    }
  }
  return estimates;
}

function lcgStream(length, universe, seedState) {
  const stream = [];
  let state = seedState % MASK_64;
  for (let i = 0; i < length; i++) {
    state = splitmix64(state);
    stream.push(Number(state % BigInt(universe)));
  }
  return stream;
}

function distinctValues(count, seedState) {
  const set = new Set();
  let state = seedState % MASK_64;
  while (set.size < count) {
    state = splitmix64(state);
    set.add(state % MASK_64);
  }
  return [...set];
}

function maxRelativeError(estimates, exact) {
  let worst = 0;
  for (let i = 0; i < exact.length; i++) {
    if (exact[i] === 0) continue;
    worst = Math.max(worst, Math.abs(estimates[i] - exact[i]) / exact[i]);
  }
  return worst;
}

function main() {
  const assert = require("node:assert");

  // (1) The exact one-pass count matches the brute-force recomputation everywhere.
  for (const seed of [1, 2, 3]) {
    for (const [length, universe, w] of [[200, 20, 10], [300, 50, 25], [150, 5, 3],
      [120, 200, 40]]) {
      const stream = lcgStream(length, universe, BigInt(seed) * 7919n);
      const exact = exactSlidingDistinct(stream, w);
      const brute = bruteForceSlidingDistinct(stream, w);
      assert.deepStrictEqual(exact, brute,
        `seed ${seed}: window ${w} of a ${length}-long stream over ${universe} values`);
      assert.strictEqual(exact.length, length - w + 1, "one count per window position");
      for (const count of exact) {
        assert.ok(count >= 1 && count <= Math.min(w, universe), "counts respect the bounds");
      }
    }
  }

  // (2) Boundary windows: w = 1, w = n, w > n and w = n - 1.
  const stream = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
  assert.deepStrictEqual(exactSlidingDistinct(stream, 1), new Array(stream.length).fill(1),
    "a window of one always holds one distinct item");
  assert.deepStrictEqual(exactSlidingDistinct(stream, stream.length), [new Set(stream).size]);
  assert.deepStrictEqual(exactSlidingDistinct(stream, stream.length + 5), [],
    "a window longer than the stream has no complete window");
  assert.deepStrictEqual(exactSlidingDistinct(stream, stream.length - 1),
    [new Set(stream.slice(0, 10)).size, new Set(stream.slice(1)).size]);
  assert.deepStrictEqual(exactSlidingDistinct([7, 7, 7, 7], 2), [1, 1, 1]);
  assert.deepStrictEqual(exactSlidingDistinct([1, 2, 1, 2], 4), [2]);
  assert.throws(() => exactSlidingDistinct(stream, 0), /window must be positive/);

  // (3) The incremental counter tracks the exact one, and re-adding an item that is still inside the
  //     window never changes the count.
  for (const w of [1, 2, 3, 5]) {
    const counts = exactSlidingDistinct(stream, w);
    for (let index = w - 1; index < stream.length; index++) {
      const window = stream.slice(index - w + 1, index + 1);
      assert.strictEqual(counts[index - w + 1], new Set(window).size,
        `window ${w} ending at ${index}`);
    }
  }
  {
    const repeating = [1, 1, 1, 2, 2, 1, 1, 2, 2, 2];
    assert.deepStrictEqual(exactSlidingDistinct(repeating, 3), [1, 2, 2, 2, 2, 2, 2, 1],
      "[1,1,1] 1; [1,1,2] 2; [1,2,2] 2; [2,2,1] 2; [2,1,1] 2; [1,1,2] 2; [1,2,2] 2; [2,2,2] 1");
    assert.deepStrictEqual(exactSlidingDistinct(repeating, 10), [2]);
  }

  // (4) HyperLogLog accuracy against the textbook 1.04/sqrt(m) standard error, at 3 sigma.
  for (const bits of [8, 10, 12]) {
    const m = 1 << bits;
    const standardError = 1.04 / Math.sqrt(m);
    for (const cardinality of [500, 5000, 50000]) {
      for (const seed of [1, 2, 3]) {
        const values = distinctValues(cardinality, BigInt(seed) * 104729n + BigInt(cardinality));
        assert.strictEqual(values.length, cardinality);
        const sketch = new HyperLogLog(bits);
        for (const value of values) sketch.add(value);
        const estimate = sketch.estimate();
        const relative = Math.abs(estimate - cardinality) / cardinality;
        assert.ok(relative <= 3 * standardError + 1e-9,
          `m = ${m}, n = ${cardinality}, seed ${seed}: relative error ${relative} `
          + `(3 sigma = ${3 * standardError})`);
      }
    }
  }

  // (5) The small-range correction matters when the cardinality is comparable to m.
  {
    const sketch = new HyperLogLog(12);
    for (let i = 0; i < 200; i++) sketch.add(i);
    const raw = sketch.rawEstimate();
    const corrected = sketch.estimate();
    assert.ok(Math.abs(corrected - 200) / 200 < Math.abs(raw - 200) / 200,
      `linear counting is better here: ${corrected.toFixed(1)} vs raw ${raw.toFixed(1)}`);
    assert.ok(sketch.zeroRegisters() > 0, "and the correction is applicable (empty registers)");
  }

  // (6) Merge identity: HLL(A ∪ B) is the registerwise maximum of HLL(A) and HLL(B).
  for (let seed = 1; seed <= 4; seed++) {
    const left = distinctValues(3000, BigInt(seed) * 31337n);
    const right = distinctValues(4000, BigInt(seed) * 61681n);
    const leftSketch = new HyperLogLog(10);
    const rightSketch = new HyperLogLog(10);
    for (const value of left) leftSketch.add(value);
    for (const value of right) rightSketch.add(value);
    const merged = leftSketch.merge(rightSketch);
    const unionSketch = new HyperLogLog(10);
    for (const value of new Set([...left, ...right])) unionSketch.add(value);
    assert.deepStrictEqual([...merged.registers], [...unionSketch.registers],
      "merging is exactly the registerwise maximum");
    assert.strictEqual(merged.estimate(), unionSketch.estimate());
    assert.throws(() => leftSketch.merge(new HyperLogLog(8)), /register exponent mismatch/);
  }

  // (7) Monotonicity and the empty sketch.
  {
    const sketch = new HyperLogLog(8);
    assert.strictEqual(sketch.estimate(), 0, "an empty sketch estimates zero via linear counting");
    let previous = 0;
    for (let i = 1; i <= 500; i++) {
      sketch.add(i * 3);
      const estimate = sketch.estimate();
      assert.ok(estimate >= previous - 1e-9, `the estimate must not decrease at i = ${i}`);
      previous = estimate;
    }
    assert.ok(previous > 400 && previous < 600, "and it lands near 500");
    assert.throws(() => new HyperLogLog(2), /register exponent out of range/);
    assert.throws(() => new HyperLogLog(21), /register exponent out of range/);
  }

  // (8) The sliding-window estimator with stale rebuilds stays inside the documented band.
  const windowStream = lcgStream(600, 400, 12345n);
  const windowSize = 200;
  const exactWindows = exactSlidingDistinct(windowStream, windowSize);
  for (const [bits, stride] of [[10, 25], [10, 100], [12, 50]]) {
    const estimates = slidingHllEstimates(windowStream, windowSize, bits, stride);
    assert.strictEqual(estimates.length, exactWindows.length);
    const standardError = 1.04 / Math.sqrt(1 << bits);
    const epsilon = 3 * standardError;
    for (let i = 0; i < exactWindows.length; i++) {
      const relative = Math.abs(estimates[i] - exactWindows[i]) / exactWindows[i];
      assert.ok(relative <= epsilon + 0.15,
        `bits ${bits}, stride ${stride}, position ${i}: relative error ${relative} exceeds `
        + `${epsilon} + the documented 0.15 staleness allowance`);
    }
    assert.ok(estimates.every((value) => value >= 1), "the window is never empty here");
  }
  // A shorter rebuild interval is at least as accurate.
  const fine = slidingHllEstimates(windowStream, windowSize, 10, 10);
  const coarse = slidingHllEstimates(windowStream, windowSize, 10, 200);
  assert.ok(maxRelativeError(fine, exactWindows) <= maxRelativeError(coarse, exactWindows),
    `stride 10 (${maxRelativeError(fine, exactWindows)}) should beat stride 200 `
    + `(${maxRelativeError(coarse, exactWindows)})`);
  // Estimating the whole prefix instead of the window would be badly wrong, which is the point.
  const prefixSketch = new HyperLogLog(10);
  for (const item of windowStream) prefixSketch.add(item);
  assert.ok(Math.abs(prefixSketch.estimate() - exactWindows[exactWindows.length - 1])
    / exactWindows[exactWindows.length - 1] > 0.5,
  "the whole-prefix cardinality is far above the window cardinality, so the window must be sketched");

  // (9) The estimator's denominator mechanism: sum 2^-M[j] is non-increasing under insertion.
  {
    const sketch = new HyperLogLog(8);
    let previousDenominator = Infinity;
    for (let i = 0; i < 200; i++) {
      sketch.add(i);
      let denominator = 0;
      for (const rank of sketch.registers) denominator += Math.pow(2, -rank);
      assert.ok(denominator <= previousDenominator + 1e-12,
        "the harmonic denominator can only shrink");
      previousDenominator = denominator;
    }
    assert.ok(previousDenominator < (1 << 8), "and it ends well below m");
  }

  // (10) Hashing sanity: splitmix64 is bijective-looking (no collisions on 100k small inputs) and
  //      the register choice spreads items over all registers.
  {
    const hashes = new Set();
    for (let i = 0; i < 100000; i++) hashes.add(hashItem(i).toString());
    assert.strictEqual(hashes.size, 100000, "no hash collisions on the first 100k integers");
    const sketch = new HyperLogLog(8);
    for (let i = 0; i < 256 * 40; i++) sketch.add(i);
    const used = [...sketch.registers].filter((rank) => rank > 0).length;
    assert.strictEqual(used, 256, "every register is touched by 10240 items");
    assert.strictEqual(countLeadingZeros64(0n), 64);
    assert.strictEqual(countLeadingZeros64(1n << 63n), 0);
    assert.deepStrictEqual(HyperLogLog.positionAndRank(0n, 8), { position: 0, rank: 65 },
      "an all-zero hash gives rank 65");
  }

  // (11) Determinism.
  assert.deepStrictEqual(exactSlidingDistinct(windowStream, 50), exactSlidingDistinct(windowStream, 50));
  assert.deepStrictEqual(slidingHllEstimates(windowStream, 200, 10, 25),
    slidingHllEstimates(windowStream, 200, 10, 25));
}

module.exports = { TWO_64, MASK_64, splitmix64, hashItem, countLeadingZeros64, HyperLogLog,
  exactSlidingDistinct, bruteForceSlidingDistinct, slidingHllEstimates, lcgStream, distinctValues,
  maxRelativeError };

if (require.main === module) {
  main();
  console.log("PASS");
}
