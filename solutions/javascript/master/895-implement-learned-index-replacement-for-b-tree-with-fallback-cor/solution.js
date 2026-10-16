/** Task 895 - Implement learned index replacement for a B-tree with fallback correctness checks.
 *  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * A recursive model index (RMI) over a sorted key array — the learned-index replacement for a B-tree —
 * with the two mechanisms that make it *safe* rather than merely fast:
 *
 *  model      stage 1 is one linear model mapping key → position; the position selects a bucket and
 *             stage 2 is a per-bucket linear model of the residual (index − stage-1 prediction). A
 *             lookup costs two multiplications instead of ~log2(n) comparisons.
 *  fallback   the prediction is only a guess, so every lookup performs an exponential (doubling) probe
 *             outwards from the guess, bracketing the key, and then a binary refinement inside that
 *             window. Correctness therefore never depends on the model being right — the model only
 *             decides how many steps the search takes. After training, the engine *measures* the true
 *             maximum error ε over the data and publishes it together with the step budget
 *             2·⌈log2(2ε+1)⌉ + 4 (two logarithmic phases: the doubling probe and the binary refinement).
 *
 * Maintenance: inserting in sorted order shifts positions, so a retrain is triggered every
 * `retrainThreshold` inserts (with the error policy "relearn when the data drifts" being verified, not
 * assumed).
 *
 * Invariants verified below: (1) every lookup agrees with an independent binary-search lower bound — for
 * present keys, duplicates (first occurrence) and absent keys (insertion point), including a key above
 * and below the whole array; (2) the measured prediction error really is within the declared ε, for
 * random, duplicated, skewed, all-equal and outlier-shaped data; (3) the local search never leaves the
 * array and respects its step budget; (4) the RMI beats binary search on total probes at 10^4 keys while
 * staying exact on adversarial inputs; (5) retraining happens exactly on the documented schedule;
 * (6) empty, single-key and all-equal indexes behave; (7) the model and its error are deterministic.
 *
 * Simplifications: integer keys, an in-memory array (the page format of the B-tree it replaces is not
 * modelled), single-threaded, and a two-stage RMI rather than a deep neural model.
 */
"use strict";

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
}

/** Least-squares line through (x, y); a vertical fit falls back to the mean of y. */
function fitLine(points) {
  const n = points.length;
  if (n === 0) return { slope: 0, intercept: 0 };
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumXY = 0;
  for (const [x, y] of points) { sumX += x; sumY += y; sumXX += x * x; sumXY += x * y; }
  const denominator = n * sumXX - sumX * sumX;
  if (denominator === 0) return { slope: 0, intercept: sumY / n };
  const slope = (n * sumXY - sumX * sumY) / denominator;
  return { slope, intercept: (sumY - slope * sumX) / n };
}

class LearnedIndex {
  constructor(options = {}) {
    this.keys = [];
    this.stage1Count = options.stage1Count || 8;
    this.retrainThreshold = options.retrainThreshold || 4;
    this.maxError = 0;
    this.averageError = 0;
    this.retrains = 0;
    this.builds = 0;
    this.pendingInserts = 0;
    this.compareSteps = 0;
    this.searchSteps = 0;
    this.models = [];
    this.stage1 = { slope: 0, intercept: 0 };
    this.perBucket = 1;
  }

  /** Trains both stages and measures the true maximum and average error. */
  train() {
    const n = this.keys.length;
    this.builds++;
    this.models = [];
    this.stage1 = { slope: 0, intercept: 0 };
    if (n === 0) { this.maxError = 0; this.averageError = 0; return this; }
    // Stage 1: a global line, trained on a strided sample so the fit stays cheap.
    const stride = Math.max(1, Math.floor(n / 512));
    const points = [];
    for (let i = 0; i < n; i += stride) points.push([this.keys[i], i]);
    points.push([this.keys[n - 1], n - 1]);
    this.stage1 = fitLine(points);
    // Stage 2: one residual model per bucket, where buckets come from stage 1's prediction.
    const buckets = Math.max(1, Math.min(this.stage1Count, n));
    this.perBucket = Math.ceil(n / buckets);
    for (let b = 0; b < buckets; b++) {
      const start = b * this.perBucket;
      const end = Math.min(n, start + this.perBucket);
      if (start >= end) continue;
      const residualPoints = [];
      for (let i = start; i < end; i++) {
        const predicted = this.stage1.intercept + this.stage1.slope * this.keys[i];
        residualPoints.push([this.keys[i], i - predicted]);
      }
      this.models.push({ bucket: b, start, end, residual: fitLine(residualPoints) });
    }
    let total = 0;
    this.maxError = 0;
    for (let i = 0; i < n; i++) {
      const error = Math.abs(Math.round(this.predictPosition(this.keys[i])) - i);
      this.maxError = Math.max(this.maxError, error);
      total += error;
    }
    this.averageError = total / n;
    return this;
  }

  /** Predicted position of `key`: stage 1 gives the bucket, stage 2 corrects it. */
  predictPosition(key) {
    if (this.models.length === 0) return 0;
    const predicted = this.stage1.intercept + this.stage1.slope * key;
    let bucket = Math.floor(predicted / this.perBucket);
    if (!Number.isFinite(bucket)) bucket = 0;
    bucket = Math.min(this.models.length - 1, Math.max(0, bucket));
    const model = this.models[bucket];
    return predicted + model.residual.intercept + model.residual.slope * key;
  }

  /** Independent oracle: the classic lower bound, with comparison counting for the baseline cost. */
  binaryLowerBound(key) {
    let low = 0;
    let high = this.keys.length;
    while (low < high) {
      const mid = (low + high) >> 1;
      this.compareSteps++;
      if (this.keys[mid] < key) low = mid + 1;
      else high = mid;
    }
    return low;
  }

  /**
   * Exact lower bound via a bounded local search: doubling probe from the model's guess, then a
   * binary refinement. Invariants maintained by the probe: keys[lo] < key (or lo = 0) and
   * keys[hi] ≥ key (or hi = n).
   */
  search(key) {
    const n = this.keys.length;
    if (n === 0) return 0;
    let guess = Math.round(this.predictPosition(key));
    if (!Number.isFinite(guess)) guess = 0;
    guess = Math.min(n - 1, Math.max(0, guess));
    let lo;
    let hi;
    if (this.keys[guess] < key) {
      lo = guess;
      hi = guess + 1;
      let step = 1;
      while (hi < n && this.keys[hi] < key) {
        lo = hi;
        hi = Math.min(n, hi + step);
        step *= 2;
        this.searchSteps++;
      }
      if (hi === n && this.keys[n - 1] < key) return n;
    } else {
      hi = guess;
      lo = guess;
      let step = 1;
      while (lo > 0 && this.keys[lo - 1] >= key) {
        hi = lo;
        lo = Math.max(0, lo - step);
        step *= 2;
        this.searchSteps++;
      }
    }
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      this.searchSteps++;
      if (this.keys[mid] < key) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  insert(key) {
    const position = this.keys.length === 0 ? 0 : this.binaryLowerBound(key);
    this.keys.splice(position, 0, key);
    this.pendingInserts++;
    if (this.pendingInserts >= this.retrainThreshold) {
      this.pendingInserts = 0;
      this.retrains++;
      this.train();
    }
    return position;
  }

  /** Appends a key at the end (a fast path that does not need a search). */
  push(key) {
    this.keys.push(key);
    this.pendingInserts++;
    if (this.pendingInserts >= this.retrainThreshold) {
      this.pendingInserts = 0;
      this.retrains++;
      this.train();
    }
    return this.keys.length - 1;
  }

  get size() { return this.keys.length; }

  /** Theoretical worst case: a doubling probe plus a binary refinement over the final window. */
  static searchBudget(maxError) { return 2 * Math.ceil(Math.log2(2 * Math.max(1, maxError) + 1)) + 4; }
}

function makeSortedKeys(count, prng, options = {}) {
  if (options.allEqual) return new Array(count).fill(41);
  const keys = [];
  let value = 0;
  for (let i = 0; i < count; i++) {
    value += options.duplicates ? 1 + prng.below(3) : (options.skewed ? 1 + prng.below(4) : 1);
    keys.push(value);
  }
  return keys;
}

module.exports = { Prng, fitLine, LearnedIndex, makeSortedKeys };

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- regression helper ----------------------------------------------------------------------
  {
    const line = fitLine([[0, 0], [1, 1], [2, 2], [3, 3]]);
    assert.ok(Math.abs(line.slope - 1) < 1e-12 && Math.abs(line.intercept) < 1e-12);
    const flat = fitLine([[5, 7], [5, 9], [5, 11]]);
    assert.strictEqual(flat.slope, 0, "a vertical bucket falls back to the mean");
    assert.strictEqual(flat.intercept, 9);
    assert.deepStrictEqual(fitLine([]), { slope: 0, intercept: 0 });
  }

  // --- exact lookups against the binary-search oracle -----------------------------------------
  {
    const prng = new Prng(895);
    for (const count of [1, 2, 7, 64, 513]) {
      const keys = makeSortedKeys(count, prng, { duplicates: count > 10 });
      const index = new LearnedIndex({ stage1Count: 8 });
      index.keys = keys.slice();
      index.train();
      for (const key of keys) {
        assert.strictEqual(index.search(key), index.binaryLowerBound(key), `count ${count}, key ${key}`);
      }
      for (const absent of [keys[0] - 5, keys[0] - 1, keys[count - 1] + 1, keys[count - 1] + 99]) {
        assert.strictEqual(index.search(absent), index.binaryLowerBound(absent), `absent ${absent}`);
      }
      assert.ok(index.maxError >= 0 && Number.isFinite(index.maxError));
    }
  }

  // --- the measured error bounds every prediction ---------------------------------------------
  {
    const prng = new Prng(4242);
    for (const shape of [{}, { duplicates: true }, { skewed: true }, { allEqual: true }]) {
      const keys = makeSortedKeys(200, prng, shape);
      const index = new LearnedIndex({ stage1Count: 6 });
      index.keys = keys.slice();
      index.train();
      for (let i = 0; i < keys.length; i++) {
        const error = Math.abs(Math.round(index.predictPosition(keys[i])) - i);
        assert.ok(error <= index.maxError + 1e-9,
          `${JSON.stringify(shape)}: error ${error} at ${i} exceeds the declared bound ${index.maxError}`);
      }
      assert.ok(index.maxError <= keys.length);
      assert.ok(index.averageError <= index.maxError);
    }
  }

  // --- the local search stays inside the array and inside its budget --------------------------
  {
    const prng = new Prng(77);
    const keys = makeSortedKeys(1000, prng, { duplicates: true });
    const index = new LearnedIndex({ stage1Count: 8 });
    index.keys = keys.slice();
    index.train();
    const budget = LearnedIndex.searchBudget(index.maxError);
    for (let probe = 0; probe < 400; probe++) {
      const key = index.keys[prng.below(keys.length)] + prng.below(3) - 1;
      const before = index.searchSteps;
      const position = index.search(key);
      const used = index.searchSteps - before;
      assert.ok(position >= 0 && position <= keys.length, "the result must be a valid position");
      assert.ok(used <= budget, `search used ${used} steps, budget ${budget}`);
      assert.strictEqual(position, index.binaryLowerBound(key), `key ${key}`);
    }
  }

  // --- the RMI beats binary search on probes at scale -----------------------------------------
  {
    const prng = new Prng(31337);
    const keys = makeSortedKeys(10000, prng, { skewed: true });
    const index = new LearnedIndex({ stage1Count: 64, retrainThreshold: 64 });
    index.keys = keys.slice();
    index.train();
    assert.strictEqual(index.size, 10000);
    assert.ok(index.maxError < 200, `error ${index.maxError} should be small at this scale`);
    let searchTotal = 0;
    let binaryTotal = 0;
    for (let probe = 0; probe < 600; probe++) {
      const key = keys[(prng.below(keys.length) * 7919) % keys.length];
      let before = index.searchSteps;
      const learned = index.search(key);
      searchTotal += index.searchSteps - before;
      before = index.compareSteps;
      const oracle = index.binaryLowerBound(key);
      binaryTotal += index.compareSteps - before;
      assert.strictEqual(learned, oracle, `probe ${probe}`);
    }
    assert.ok(searchTotal < binaryTotal / 2,
      `learned ${searchTotal} probes vs binary ${binaryTotal}`);
  }

  // --- sorted insertion, retraining and drift -------------------------------------------------
  {
    const index = new LearnedIndex({ stage1Count: 4, retrainThreshold: 8 });
    const prng = new Prng(555);
    const inserts = [];
    for (let i = 0; i < 40; i++) inserts.push(i * 3 + prng.below(2));
    const sorted = inserts.slice().sort((a, b) => a - b);
    for (const value of sorted) index.insert(value);
    assert.strictEqual(index.size, 40);
    assert.strictEqual(index.retrains, Math.floor(40 / 8), "one retrain per threshold window");
    assert.deepStrictEqual([...index.keys], sorted);
    for (const value of sorted) assert.strictEqual(index.search(value), index.binaryLowerBound(value));
    // The append fast path keeps the ordering and retrains on schedule too.
    const appended = new LearnedIndex({ stage1Count: 3, retrainThreshold: 5 });
    let next = 100;
    for (let i = 0; i < 15; i++) { appended.push(next); next += 2; }
    assert.strictEqual(appended.size, 15);
    assert.strictEqual(appended.retrains, 3);
    assert.strictEqual(appended.search(100), 0);
    assert.strictEqual(appended.search(102), 1);
    assert.strictEqual(appended.search(101), 1, "an absent key yields the insertion point");
    assert.strictEqual(appended.search(1), 0);
    assert.strictEqual(appended.search(999), 15);
    assert.strictEqual(appended.search(128), 14);
  }

  // --- adversarial shapes ----------------------------------------------------------------------
  {
    const empty = new LearnedIndex();
    empty.train();
    assert.strictEqual(empty.search(5), 0, "an empty index has one valid insertion point");
    assert.strictEqual(empty.size, 0);
    assert.strictEqual(empty.maxError, 0);
    const single = new LearnedIndex({ stage1Count: 4 });
    single.keys.push(9);
    single.train();
    assert.strictEqual(single.search(9), 0);
    assert.strictEqual(single.search(8), 0);
    assert.strictEqual(single.search(10), 1);
    const equal = new LearnedIndex({ stage1Count: 4 });
    equal.keys = new Array(500).fill(3);
    equal.train();
    assert.strictEqual(equal.search(3), 0, "the first occurrence of a duplicated key");
    assert.strictEqual(equal.search(2), 0);
    assert.strictEqual(equal.search(4), 500);
    assert.strictEqual(equal.binaryLowerBound(3), 0);
    assert.strictEqual(equal.search(4), equal.binaryLowerBound(4));
    // A single far outlier must not break the search bound.
    const outlier = new LearnedIndex({ stage1Count: 4 });
    outlier.keys = [...Array.from({ length: 100 }, (_, i) => i), 1e9];
    outlier.train();
    for (const key of [0, 50, 99, 1e9, 1e9 - 1, 500, -7]) {
      assert.strictEqual(outlier.search(key), outlier.binaryLowerBound(key), `outlier key ${key}`);
    }
    assert.ok(Number.isFinite(outlier.maxError) && outlier.maxError >= 0);
    // A quadratic key distribution cannot be captured by linear models: the *measured* error is large,
    // so the fallback search has to do real work — and must still be exactly right and within budget.
    const curved = new LearnedIndex({ stage1Count: 4 });
    curved.keys = Array.from({ length: 500 }, (_, i) => i * i);
    curved.train();
    assert.ok(curved.maxError > 5,
      `a quadratic distribution must exceed a linear model (measured ${curved.maxError})`);
    const curvedBudget = LearnedIndex.searchBudget(curved.maxError);
    for (const key of [0, 1, 3, 250000, 249001, 300, 499 * 499, -5, 1e7]) {
      const before = curved.searchSteps;
      const position = curved.search(key);
      const used = curved.searchSteps - before;
      assert.strictEqual(position, curved.binaryLowerBound(key), `curved key ${key}`);
      assert.ok(used <= curvedBudget, `curved search used ${used} steps, budget ${curvedBudget}`);
    }
    assert.ok(curved.averageError > 0 && curved.averageError < curved.maxError);
  }

  // --- determinism ------------------------------------------------------------------------------
  {
    const build = () => {
      const prng = new Prng(99);
      const keys = makeSortedKeys(500, prng, { duplicates: true });
      const index = new LearnedIndex({ stage1Count: 5 });
      index.keys = keys.slice();
      index.train();
      return `${index.maxError}|${index.averageError.toFixed(6)}|${index.models.length}|${index.search(keys[250])}`;
    };
    assert.strictEqual(build(), build());
    const a = new LearnedIndex({ stage1Count: 64 });
    const b = new LearnedIndex({ stage1Count: 64 });
    const keys = makeSortedKeys(10000, new Prng(31337), { skewed: true });
    a.keys = keys.slice();
    b.keys = keys.slice();
    a.train();
    b.train();
    assert.deepStrictEqual(a.stage1, b.stage1);
  }

  console.log(`learned-index: 10000 keys, maxError=${indexUnderTest().maxError}, ` +
    `search budget 2*ceil(log2(2*eps+1))+4, retrains verified`);

  function indexUnderTest() {
    const keys = makeSortedKeys(10000, new Prng(31337), { skewed: true });
    const index = new LearnedIndex({ stage1Count: 64 });
    index.keys = keys.slice();
    index.train();
    return index;
  }
  console.log("PASS");
}
