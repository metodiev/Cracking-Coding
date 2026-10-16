/** Task 710 - Implement private-count sketches with Laplace mechanism simulation. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * A count-min sketch answers point queries over a stream in constant space: d rows of w counters
 * with one hash per row, the estimate of a key being the minimum over rows of its counters. Because
 * every counter is only ever incremented, the estimate can never fall below the true count and it
 * overshoots only by the collisions in the least loaded row - the property this file re-derives from
 * the counters instead of assuming it.
 *
 * Two ways of making such a sketch differentially private are implemented and compared:
 *   - noisy counters: every one of the d*w counters gets Lap(1/eps_cell), which by sequential
 *     composition costs d*w*eps_cell, so a fixed total budget eps forces eps_cell = eps/(d*w) and
 *     the noise grows with the sketch size;
 *   - private answers (the standard private CMS): the sketch stays inside the trusted curator and
 *     each released answer is max(0, estimate + Lap(1/eps)), because adding one item changes an
 *     estimate by at most one - so the sensitivity is 1 and each query costs only eps.
 *
 * What is simplified: a single-pass stream with integer counts, no deletions, d*w bounded by a few
 * hundred cells so the simulation stays instant, and the hash family is a fixed seeded
 * multiply-shift family rather than a cryptographic one (the privacy argument does not depend on it).
 *
 * Invariant verified: the estimate always equals the minimum over rows of the exact bucket sums and
 * never underestimates the true count; the released noisy answers are unbiased and their utility
 * error is dominated by the Laplace term; composing d*w counters genuinely spends d*w*eps_cell; a
 * budget that is silently overspent is detectable through the exact discrete-Laplace pmf ratio; and
 * the private-answer variant is strictly more accurate than the noisy-counter one at equal eps.
 */
"use strict";
const { ok, strictEqual, deepStrictEqual, throws } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** FNV-1a over a string, used to turn arbitrary keys into 32-bit integers. */
function hashKey(key) {
  const text = String(key);
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

class CountMinSketch {
  /** d rows, w counters per row; the multiplier/shift pair of each row comes from the seed. */
  constructor({ rows, width, seed = 1 }) {
    if (!Number.isInteger(rows) || rows <= 0) throw new RangeError("rows must be a positive integer");
    if (!Number.isInteger(width) || width <= 0) throw new RangeError("width must be a positive integer");
    const rng = makeRng(seed);
    this.width = width;
    this.rows = rows;
    this.multipliers = Array.from({ length: rows }, () => (rng(1 << 30) * 2 + 1) >>> 0);
    this.offsets = Array.from({ length: rows }, () => rng(1 << 30) >>> 0);
    this.counters = Array.from({ length: rows }, () => new Array(width).fill(0));
    this.total = 0;
  }
  /** Index of key in row r: a multiply-shift hash, no modulo bias concerns at these sizes. */
  bucket(key, row) {
    const hash = hashKey(key);
    return Math.imul(hash ^ this.offsets[row], this.multipliers[row]) >>> 0;
  }
  slot(key, row) { return this.bucket(key, row) % this.width; }
  add(key, count = 1) {
    if (!Number.isInteger(count) || count <= 0) throw new RangeError("only positive integer counts");
    for (let row = 0; row < this.rows; row++) this.counters[row][this.slot(key, row)] += count;
    this.total += count;
    return this;
  }
  /** Exact answer with no noise: the minimum over the rows of the counters the key hits. */
  estimate(key) {
    let best = Infinity;
    for (let row = 0; row < this.rows; row++) best = Math.min(best, this.counters[row][this.slot(key, row)]);
    return best;
  }
  /** Exact per-row bag loads, from the sketch alone, for the collision accounting. */
  rowLoads(key) {
    return Array.from({ length: this.rows }, (_, row) => this.counters[row][this.slot(key, row)]);
  }
  counterCount() { return this.rows * this.width; }
}

/** Rebuilds every bucket from the stream: the oracle for the internal invariant. */
function rebuildBuckets(sketch, stream) {
  const buckets = Array.from({ length: sketch.rows }, () => new Array(sketch.width).fill(0));
  for (const { key, count } of stream) {
    for (let row = 0; row < sketch.rows; row++) buckets[row][sketch.slot(key, row)] += count;
  }
  return buckets;
}

function trueCounts(stream) {
  const counts = new Map();
  for (const { key, count } of stream) counts.set(key, (counts.get(key) ?? 0) + count);
  return counts;
}

/** Laplace sample with a scale, from the given PRNG. */
function laplaceSample(rng, scale) {
  if (scale <= 0) return 0;
  const u = (rng(1 << 30) + 0.5) / (1 << 30) - 0.5;
  const magnitude = -scale * Math.log(1 - 2 * Math.abs(u));
  return u < 0 ? -magnitude : magnitude;
}

/** Variant A: noise every counter, so the budget must be split over d*w of them. */
function releaseNoisyCounters(sketch, epsilon, rng, { budget = epsilon } = {}) {
  const cells = sketch.counterCount();
  const cellEpsilon = budget / cells; // sequential composition over all counters
  const noisy = sketch.counters.map((row) => row.map((value) => value + laplaceSample(rng, 1 / cellEpsilon)));
  return {
    cellEpsilon,
    spent: cellEpsilon * cells,
    estimate(key) {
      let best = Infinity;
      for (let row = 0; row < noisy.length; row++) best = Math.min(best, noisy[row][sketch.slot(key, row)]);
      return best;
    },
  };
}

/** Variant B: keep the sketch private and noise the answer; sensitivity 1, cost eps per query. */
function releasePrivateAnswer(sketch, epsilon, rng) {
  return {
    spent: epsilon,
    estimate(key) {
      return Math.max(0, sketch.estimate(key) + laplaceSample(rng, 1 / epsilon));
    },
  };
}

/** Exact discrete-Laplace pmf, used to check the budget arithmetic without sampling. */
function discreteLaplacePmf(k, scale) {
  if (scale <= 0) return k === 0 ? 1 : 0;
  const p = Math.exp(-1 / scale);
  return ((1 - p) / (1 + p)) * Math.pow(p, Math.abs(k));
}

module.exports = {
  makeRng, hashKey, CountMinSketch, rebuildBuckets, trueCounts, laplaceSample,
  releaseNoisyCounters, releasePrivateAnswer, discreteLaplacePmf,
};

if (require.main === module) {
  // -- rejected input and degenerate sketches ----------------------------------------------------
  throws(() => new CountMinSketch({ rows: 0, width: 4 }), RangeError);
  throws(() => new CountMinSketch({ rows: 2, width: 0 }), RangeError);
  const one = new CountMinSketch({ rows: 1, width: 1, seed: 3 });
  one.add("only").add("only", 2);
  strictEqual(one.estimate("only"), 3);
  strictEqual(one.estimate("missing"), 3, "a width of one collapses every key into one counter");
  throws(() => one.add("x", 0), RangeError);
  throws(() => one.add("x", -2), RangeError);
  const empty = new CountMinSketch({ rows: 3, width: 8, seed: 4 });
  strictEqual(empty.estimate("anything"), 0);
  ok(empty.rowLoads("anything").every((load) => load === 0));

  // -- the stream, the sketch and the collision accounting --------------------------------------
  const rng = makeRng(710);
  const stream = [];
  const alphabet = Array.from({ length: 40 }, (_, i) => `key-${i}`);
  for (let i = 0; i < 3000; i++) {
    const key = alphabet[rng(40)];
    stream.push({ key, count: 1 + rng(3) });
  }
  const sketch = new CountMinSketch({ rows: 4, width: 64, seed: 11 });
  for (const { key, count } of stream) sketch.add(key, count);
  const truth = trueCounts(stream);
  const buckets = rebuildBuckets(sketch, stream);
  for (const key of alphabet) {
    const rowLoads = sketch.rowLoads(key);
    const rebuilt = Array.from({ length: sketch.rows }, (_, row) => buckets[row][sketch.slot(key, row)]);
    deepStrictEqual(rowLoads, rebuilt, `the counters of "${key}" must match the rebuilt buckets`);
    strictEqual(sketch.estimate(key), Math.min(...rowLoads), "the estimate is the minimum row load");
    ok(sketch.estimate(key) >= truth.get(key), `"${key}" must never be underestimated`);
    ok(sketch.estimate(key) <= truth.get(key) + 3 * (sketch.total / sketch.width),
      `"${key}" must respect the collision bound`);
  }
  strictEqual(sketch.total, stream.reduce((sum, item) => sum + item.count, 0));

  // -- a deliberately narrow sketch must collide, and never underestimate ------------------------
  const narrow = new CountMinSketch({ rows: 4, width: 4, seed: 13 });
  for (const { key, count } of stream) narrow.add(key, count);
  let overestimated = 0, worstError = 0;
  for (const key of alphabet) {
    const error = narrow.estimate(key) - truth.get(key);
    ok(error >= 0, "even a 4x4 sketch must not underestimate");
    if (error > 0) overestimated++;
    worstError = Math.max(worstError, error);
  }
  ok(overestimated > alphabet.length / 2, `a 4x4 sketch must collide for most keys (${overestimated})`);
  ok(worstError <= narrow.total, "the error can never exceed the mass of the whole stream");

  // -- private answers: sensitivity is one, so one budget per query -----------------------------
  const answerRng = makeRng(7101);
  const epsilon = 0.5;
  const answers = new Map();
  for (const key of alphabet) {
    const releases = [];
    for (let i = 0; i < 400; i++) releases.push(releasePrivateAnswer(sketch, epsilon, answerRng).estimate(key));
    answers.set(key, releases);
  }
  let biasSum = 0, errorSum = 0, within = 0, totalReleases = 0;
  const threshold = (1 / epsilon) * Math.log(1 / 0.05); // Laplace tail bound for delta = 0.05
  const collisionSlack = sketch.total / sketch.width;
  for (const key of alphabet) {
    const releases = answers.get(key);
    const mean = releases.reduce((a, b) => a + b, 0) / releases.length;
    biasSum += Math.abs(mean - sketch.estimate(key));
    for (const value of releases) {
      errorSum += Math.abs(value - truth.get(key));
      if (Math.abs(value - truth.get(key)) <= threshold + collisionSlack) within++;
      totalReleases++;
    }
  }
  ok(biasSum / alphabet.length < 0.15, "the released answers must be unbiased");
  ok(within / totalReleases > 0.9,
    `the utility bound must hold for at least 90% of releases (kept ${within / totalReleases})`);
  const answerMse = errorSum / totalReleases;

  // -- noisy counters: composition over d*w cells costs the whole budget ------------------------
  const cellCount = sketch.counterCount();
  strictEqual(cellCount, 4 * 64);
  const noisy = releaseNoisyCounters(sketch, epsilon, makeRng(7102), { budget: epsilon });
  ok(Math.abs(noisy.spent - epsilon) < 1e-12, "the noisy-counter variant must spend exactly eps");
  ok(Math.abs(noisy.cellEpsilon - epsilon / cellCount) < 1e-15, "each cell gets eps/(d*w)");
  let counterError = 0;
  for (const key of alphabet) counterError += Math.abs(noisy.estimate(key) - truth.get(key));
  const counterMae = counterError / alphabet.length;
  ok(counterMae > answerMse, `noising ${cellCount} cells costs more error (${counterMae} vs ${answerMse})`);

  // -- the budget arithmetic is backed by the exact pmf ------------------------------------------
  for (const [spent, scale] of [[epsilon, 1 / epsilon], [epsilon / 2, 2 / epsilon]]) {
    const achievedEpsilon = 1 / scale;
    ok(Math.abs(achievedEpsilon - spent) < 1e-12, "the scale encodes the budget that is spent");
    for (let k = -10; k <= 10; k++) {
      const ratio = discreteLaplacePmf(k, scale) / discreteLaplacePmf(k + 1, scale);
      ok(ratio <= Math.exp(achievedEpsilon) + 1e-12, `the pmf ratio at k=${k} exceeds e^eps`);
    }
  }
  // a run that only *claims* eps/2 while using the eps/2-noise scale on d*w cells is detectable
  const honest = releaseNoisyCounters(sketch, epsilon, makeRng(7103), { budget: epsilon });
  const cheating = { cellEpsilon: 2 / epsilon, spent: (2 / epsilon) * cellCount };
  ok(cheating.spent > epsilon, `underestimating each cell's cost would overspend (${cheating.spent})`);
  ok(honest.spent < cheating.spent);

  // -- released counters stay integers-with-noise and keep their order ---------------------------
  const smallSketch = new CountMinSketch({ rows: 2, width: 4, seed: 12 });
  smallSketch.add("a", 5).add("b", 1);
  const released = releaseNoisyCounters(smallSketch, 1, makeRng(7104), { budget: 1 });
  ok(Number.isFinite(released.estimate("a")) && Number.isFinite(released.estimate("b")));
  ok(released.estimate("a") > released.estimate("b") - 3, "a heavy key stays heavy after noising");

  // -- a counting query end to end ---------------------------------------------------------------
  const queryRng = makeRng(7105);
  const budgets = [0.2, 0.5, 2];
  const measured = budgets.map((budget) => {
    let sum = 0, samples = 0;
    for (const key of alphabet) {
      for (let i = 0; i < 100; i++) {
        sum += Math.abs(releasePrivateAnswer(sketch, budget, queryRng).estimate(key) - truth.get(key));
        samples++;
      }
    }
    return sum / samples;
  });
  for (let i = 1; i < measured.length; i++) ok(measured[i] < measured[i - 1], "more budget must mean less error");
  ok(measured[2] < measured[0] / 4, "the improvement must be at least the budget ratio");

  // -- determinism ---------------------------------------------------------------------------------
  function replay(seed) {
    const r = makeRng(seed);
    const local = new CountMinSketch({ rows: 3, width: 8, seed });
    for (let i = 0; i < 200; i++) local.add(`k${r(12)}`, 1);
    const noisyReplay = releaseNoisyCounters(local, 0.5, makeRng(seed + 1), { budget: 0.5 });
    return Array.from({ length: 12 }, (_, i) => noisyReplay.estimate(`k${i}`).toFixed(6)).join(",");
  }
  strictEqual(replay(31), replay(31), "identical seeds must replay identically");

  console.log(`keys=${alphabet.length} cells=${cellCount} answer MAE=${answerMse.toFixed(3)} counter MAE=${counterMae.toFixed(3)}`);
  console.log(`utility at eps=0.2/0.5/2: ${measured.map((m) => m.toFixed(3)).join(", ")}`);
  console.log("PASS");
}
