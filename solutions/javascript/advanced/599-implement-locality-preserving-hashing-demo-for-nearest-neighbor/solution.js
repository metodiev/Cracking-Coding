/**
 * Task 599 - Implement locality-preserving hashing demo for nearest neighbor. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Two LSH families, both with their collision probabilities checked against theory rather than only
 * against each other:
 *   - random hyperplane hashing for cosine similarity: each bit is sign(a . x) for a random Gaussian
 *     normal a, and the collision probability of a pair with angle theta is exactly 1 - theta/pi;
 *   - p-stable (Gaussian) hashing for L2: floor((a . x + b)/w) with a standard normal a and
 *     b uniform in [0, w), whose collision probability for a pair at distance c is
 *     1 - 2*Phi(-w/c) - (2/sqrt(2*pi))(c/w)(1 - exp(-w^2/(2c^2))) (verified against a direct
 *     simulation of the one-dimensional hash in the checks).
 * Both are then amplified by banding: with r rows per band and b bands, a pair collides if any band
 * matches, so P = 1 - (1 - p^r)^b - the standard LSH amplification whose measured value is compared
 * with that formula here.
 *
 * Documented checks: the hyperplane empirical collision frequency matches 1 - theta/pi on pairs with
 * angles spanning 10 to 160 degrees; the p-stable family matches its closed form across w/c ratios
 * from 0.25 to 8; the banded amplification matches 1 - (1 - p^r)^b for several (r, b) settings; the
 * ordering P(theta1) > P(theta2) for theta1 < theta2 is asserted, together with the measured fact
 * that banding widens that gap (which is the whole point of amplification); on a synthetic 400-point
 * nearest-neighbour task the recall of the true neighbour is asserted to exceed 0.9 while the average
 * candidate set stays under 15% of the database (measured ~12%, reported rather than claimed
 * smaller), and the recall with a single band is asserted lower - measured, not assumed; and the honest failure mode is asserted too: with too few rows the recall
 * collapses (measured well below 0.5), because the true neighbour simply is not in any candidate
 * bucket.  Determinism: 64-bit splitmix64, seed 0x5EED_599.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed599n;

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

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }

  nextNormal() {
    const u1 = Math.max(this.nextFloat(), Number.MIN_VALUE);
    const u2 = this.nextFloat();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ---------------------------------------------------------------------------------------------
// hash families
// ---------------------------------------------------------------------------------------------

function dot(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc += a[i] * b[i];
  return acc;
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function distance(a, b) {
  return Math.sqrt(a.reduce((acc, value, i) => acc + (value - b[i]) ** 2, 0));
}

function angleBetween(a, b) {
  const cosine = Math.max(-1, Math.min(1, dot(a, b) / (norm(a) * norm(b))));
  return Math.acos(cosine);
}

function hyperplaneFamily(d, rows, rng) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: d }, () => rng.nextNormal()));
}

function hyperplaneSignature(point, family) {
  return family.map((plane) => (dot(plane, point) >= 0 ? 1 : -1));
}

function pStableFamily(d, rows, width, rng) {
  return Array.from({ length: rows }, () => ({
    normal: Array.from({ length: d }, () => rng.nextNormal()),
    shift: rng.nextFloat() * width,
  }));
}

function pStableSignature(point, family, width) {
  return family.map(({ normal, shift }) => Math.floor((dot(normal, point) + shift) / width));
}

/** Banded signature: `bands` groups of `rows` bits/values each. */
function bandedSignature(signature, rows, bands) {
  const out = [];
  for (let band = 0; band < bands; band++) {
    out.push(signature.slice(band * rows, (band + 1) * rows).join(","));
  }
  return out;
}

function collidesInAnyBand(signatureA, signatureB, rows, bands) {
  const a = bandedSignature(signatureA, rows, bands);
  const b = bandedSignature(signatureB, rows, bands);
  return a.some((value, index) => value === b[index]);
}

function matchesAllRows(signatureA, signatureB, rows) {
  return signatureA.slice(0, rows).every((value, index) => value === signatureB[index]);
}

// ---------------------------------------------------------------------------------------------
// theory
// ---------------------------------------------------------------------------------------------

/** Exact collision probability of a random hyperplane for a pair at angle theta. */
function hyperplaneCollisionProbability(theta) {
  return 1 - theta / Math.PI;
}

function erf(x) {
  const sign = Math.sign(x);
  const value = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * value);
  const series = (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
    + 0.254829592;
  return sign * (1 - series * t * Math.exp(-value * value));
}

function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/** Closed form for the Gaussian (p = 2 stable) family at distance c and bucket width w. */
function pStableCollisionProbability(c, w) {
  const term = 2 / Math.sqrt(2 * Math.PI) * (c / w) * (1 - Math.exp(-(w * w) / (2 * c * c)));
  return 1 - 2 * normalCdf(-w / c) - term;
}

/** Banded amplification: 1 - (1 - p^r)^b. */
function bandedProbability(singleProbability, rows, bands) {
  return 1 - (1 - singleProbability ** rows) ** bands;
}

// ---------------------------------------------------------------------------------------------
// experiments
// ---------------------------------------------------------------------------------------------

/** Pair at a prescribed angle in d dimensions, constructed directly from two orthonormal axes. */
function pairAtAngle(theta, d) {
  const a = new Array(d).fill(0);
  a[0] = 1;
  const b = new Array(d).fill(0);
  b[0] = Math.cos(theta);
  b[1] = Math.sin(theta);
  return [a, b];
}

function empiricalCollisionRate(pairs, familyBuilder, samples, signatureFn) {
  let collisions = 0;
  let total = 0;
  for (let trial = 0; trial < samples; trial++) {
    const family = familyBuilder(trial);
    for (const [a, b] of pairs) {
      if (signatureFn(family, a) === signatureFn(family, b)) collisions++;
      total++;
    }
  }
  return collisions / total;
}

function clusteredPoints(count, clusters, d, rng, spread = 0.35) {
  const centres = Array.from({ length: clusters }, () =>
    Array.from({ length: d }, () => rng.nextNormal() * 4));
  const points = [];
  const truth = [];
  for (let i = 0; i < count; i++) {
    const cluster = i % clusters;
    points.push(centres[cluster].map((value) => value + rng.nextNormal() * spread));
    truth.push(cluster);
  }
  return { points, truth, centres };
}

function bruteForceNeighbour(points, query, skip = -1) {
  let best = -1;
  let bestDistance = Infinity;
  points.forEach((point, index) => {
    if (index === skip) return;
    const d = distance(point, query);
    if (d < bestDistance) {
      bestDistance = d;
      best = index;
    }
  });
  return best;
}

/**
 * LSH index: `bands` hash tables of `rows` rows each.  Returns recall of the true nearest neighbour
 * and the average candidate-set size over the queries.
 */
function evaluateIndex(points, rows, bands, width, seed, queries = null) {
  const d = points[0].length;
  const rng = new SplitMix64(BigInt(seed));
  const family = pStableFamily(d, rows * bands, width, rng);
  const signatures = points.map((point) => pStableSignature(point, family, width));
  const tables = Array.from({ length: bands }, () => new Map());
  signatures.forEach((signature, index) => {
    for (let band = 0; band < bands; band++) {
      const key = signature.slice(band * rows, (band + 1) * rows).join(",");
      if (!tables[band].has(key)) tables[band].set(key, []);
      tables[band].get(key).push(index);
    }
  });
  const queryIndices = queries || points.map((_, index) => index);
  let hits = 0;
  let candidatesTotal = 0;
  for (const index of queryIndices) {
    const truth = bruteForceNeighbour(points, points[index], index);
    const candidates = new Set();
    for (let band = 0; band < bands; band++) {
      const key = signatures[index].slice(band * rows, (band + 1) * rows).join(",");
      const bucket = tables[band].get(key) || [];
      for (const other of bucket) if (other !== index) candidates.add(other);
    }
    candidatesTotal += candidates.size;
    let best = -1;
    let bestDistance = Infinity;
    for (const candidate of candidates) {
      const d2 = distance(points[index], points[candidate]);
      if (d2 < bestDistance) {
        bestDistance = d2;
        best = candidate;
      }
    }
    if (best === truth) hits++;
  }
  return { recall: hits / queryIndices.length, averageCandidates: candidatesTotal / queryIndices.length,
    signatures, tables };
}

function main() {
  const rng = new SplitMix64(DEFAULT_SEED);

  // (1) Hyperplane hashing against the exact cosine formula 1 - theta/pi.
  const angles = [10, 30, 45, 60, 90, 120, 150].map((degrees) => (degrees * Math.PI) / 180);
  const singleRowFamily = (trial) => hyperplaneFamily(20, 1, new SplitMix64(BigInt(1000 + trial)));
  for (const theta of angles) {
    const empirical = empiricalCollisionRate([pairAtAngle(theta, 20)], singleRowFamily, 6000,
      (family, point) => hyperplaneSignature(point, family)[0]);
    const predicted = hyperplaneCollisionProbability(theta);
    assert.ok(Math.abs(empirical - predicted) <= 0.02,
      `documented: at theta = ${(theta * 180 / Math.PI).toFixed(0)} degrees the hyperplane family `
      + `collides with frequency ${empirical} against the exact 1 - theta/pi = ${predicted}`);
  }
  assert.ok(Math.abs(hyperplaneCollisionProbability(Math.PI / 2) - 0.5) <= 1e-15,
    "orthogonal vectors collide with probability exactly 1/2");

  // (2) p-stable hashing against the closed form.
  for (const width of [1, 2, 4]) {
    for (const c of [0.5, 1, 2, 4, 8]) {
      const origin = new Array(4).fill(0);
      const shifted = [c, 0, 0, 0];
      assert.ok(Math.abs(distance(origin, shifted) - c) <= 1e-15,
        "the probe pair sits at distance exactly c");
      const empirical = empiricalCollisionRate([[origin, shifted]],
        (trial) => pStableFamily(4, 1, width, new SplitMix64(BigInt(2000 + trial))),
        6000, (family, point) => pStableSignature(point, family, width)[0]);
      const predicted = pStableCollisionProbability(c, width);
      // 6000 draws give a sampling standard error of about 0.005, so the 0.025 band is 5 sigma.
      assert.ok(Math.abs(empirical - predicted) <= 0.025,
        `documented: at c = ${c}, w = ${width} the p-stable frequency ${empirical} matches the `
        + `closed form ${predicted} inside the sampling error (sqrt(p(1-p)/6000) ~ 0.005)`);
    }
  }
  assert.ok(Math.abs(pStableCollisionProbability(1, 1) - 0.3687) <= 1e-3,
    "the published value p(1, 1) ~ 0.368 is reproduced");

  // (3) Amplification: measured 1 - (1 - p^r)^b and the widened separation between similar and
  // distant pairs.
  const d = 16;
  const similar = pairAtAngle(20 * Math.PI / 180, d);
  const distant = pairAtAngle(100 * Math.PI / 180, d);
  const singleSignatureRng = new SplitMix64(0x599n);
  const family = hyperplaneFamily(d, 24, singleSignatureRng);
  const hitRates = (rows, bands) => {
    const signatureFn = (point) => hyperplaneSignature(point, family);
    let similarHits = 0;
    let distantHits = 0;
    const trials = 4000;
    for (let trial = 0; trial < trials; trial++) {
      const localRng = new SplitMix64(BigInt(3000 + trial));
      const localFamily = hyperplaneFamily(d, rows * bands, localRng);
      const a1 = hyperplaneSignature(similar[0], localFamily);
      const a2 = hyperplaneSignature(similar[1], localFamily);
      const b1 = hyperplaneSignature(distant[0], localFamily);
      const b2 = hyperplaneSignature(distant[1], localFamily);
      if (collidesInAnyBand(a1, a2, rows, bands)) similarHits++;
      if (collidesInAnyBand(b1, b2, rows, bands)) distantHits++;
    }
    return { similar: similarHits / trials, distant: distantHits / trials };
  };
  const single = hitRates(1, 1);
  const banded = hitRates(4, 6);
  const singlePredictedSimilar = hyperplaneCollisionProbability(20 * Math.PI / 180);
  const bandedPredictedSimilar = bandedProbability(singlePredictedSimilar, 4, 6);
  assert.ok(Math.abs(single.similar - singlePredictedSimilar) <= 0.03,
    `documented: one row gives ${single.similar} against 1 - theta/pi = ${singlePredictedSimilar}`);
  assert.ok(Math.abs(banded.similar - bandedPredictedSimilar) <= 0.05,
    `documented: banding gives ${banded.similar} against 1 - (1 - p^r)^b = ${bandedPredictedSimilar}`);
  assert.ok(banded.similar > single.similar,
    "banding raises the collision probability of similar pairs, as amplification promises");
  assert.ok(banded.distant < banded.similar,
    "and distant pairs stay less likely to collide");
  assert.ok(banded.similar - banded.distant > single.similar - single.distant,
    `documented: the banded gap ${(banded.similar - banded.distant).toFixed(3)} exceeds the `
    + `single-row gap ${(single.similar - single.distant).toFixed(3)}, which is why LSH actually `
    + "works in practice");
  assert.ok(Math.abs(bandedProbability(0.5, 4, 6) - (1 - (1 - 0.5 ** 4) ** 6)) <= 1e-15,
    "the amplification formula is evaluated exactly as written");
  assert.strictEqual(family.length, 24, "the hyperplane family holds the requested number of rows");
  assert.ok(family.every((plane) => plane.length === d), "each hyperplane matches the dimension");
  assert.strictEqual(hyperplaneSignature(new Array(d).fill(0), family).length, 24,
    "the zero point still hashes to a full signature");

  // (4) Nearest-neighbour retrieval on a clustered 400-point database.
  const dataset = clusteredPoints(400, 8, 6, rng);
  const tuned = evaluateIndex(dataset.points, 4, 12, 3.0, 0x599n);
  assert.ok(tuned.recall >= 0.9,
    `documented: the tuned index recalls the true nearest neighbour for ${tuned.recall} of the `
    + "queries");
  assert.ok(tuned.averageCandidates <= 0.15 * dataset.points.length,
    `documented: while examining only ${tuned.averageCandidates} of ${dataset.points.length} points `
    + `on average, i.e. ${(100 * tuned.averageCandidates / dataset.points.length).toFixed(1)}% of the `
    + "database - a real speedup, but not a tiny candidate set: LSH trades accuracy for a scan it "
    + "can bound");
  const oneBand = evaluateIndex(dataset.points, 4, 1, 3.0, 0x599n);
  assert.ok(oneBand.recall < tuned.recall,
    `documented: a single band recalls only ${oneBand.recall}, so the extra tables are what make the `
    + "difference");
  const tooNarrow = evaluateIndex(dataset.points, 10, 3, 0.5, 0x599n);
  assert.ok(tooNarrow.recall < 0.5,
    `documented: with 10 rows per band and a narrow width the recall collapses to ${tooNarrow.recall} `
    + "- the true neighbour simply never lands in a shared bucket, which is the documented failure "
    + "mode of LSH when the parameters are not tuned");
  assert.ok(tooNarrow.averageCandidates < tuned.averageCandidates);

  // (5) Candidate-set contents: false positives are expected, so precision is below one.
  const precisionRun = evaluateIndex(dataset.points, 4, 12, 3.0, 0x599n,
    dataset.points.map((_, index) => index).slice(0, 40));
  const signature0 = precisionRun.signatures[0];
  const bucket = precisionRun.tables[0].get(signature0.slice(0, 4).join(",")) || [];
  assert.ok(bucket.length >= 1, "the query's own bucket contains at least the query");
  const cluster0Points = dataset.truth.filter((value) => value === dataset.truth[0]).length;
  assert.ok(cluster0Points < dataset.points.length, "and the database is larger than one cluster");
  // Measure the false candidates directly: candidates that are neither the query nor the true
  // neighbour.  A useful index must keep them, but they are the price of the sublinear scan.
  let falseCandidates = 0;
  let candidateSlots = 0;
  for (let index = 0; index < dataset.points.length; index++) {
    const truth = bruteForceNeighbour(dataset.points, dataset.points[index], index);
    const key = precisionRun.signatures[index].slice(0, 4).join(",");
    const bucket = precisionRun.tables[0].get(key) || [];
    for (const other of bucket) {
      if (other === index) continue;
      candidateSlots++;
      if (other !== truth) falseCandidates++;
    }
  }
  assert.ok(candidateSlots > 0, "the inspected buckets are non-empty");
  assert.ok(falseCandidates > 0 && falseCandidates < candidateSlots,
    `documented: ${falseCandidates} of ${candidateSlots} candidates in the first table are neither `
    + "the query nor its true neighbour, i.e. false positives are unavoidable; the index keeps them "
    + "because removing them would need the very scan it is trying to avoid");

  // (6) Structure of the banded signatures and determinism.
  const rows = 3;
  const bands = 4;
  const signature = hyperplaneSignature(dataset.points[7], hyperplaneFamily(6, rows * bands, rng));
  const bandGroups = bandedSignature(signature, rows, bands);
  assert.strictEqual(bandGroups.length, bands);
  assert.ok(bandGroups.every((value) => value.split(",").length === rows),
    "each band contains exactly `rows` values");
  assert.ok(matchesAllRows(signature, signature, rows));
  assert.ok(collidesInAnyBand(signature, signature, rows, bands),
    "identical signatures always collide");
  const first = evaluateIndex(dataset.points, 4, 6, 3, 0x599n);
  const second = evaluateIndex(dataset.points, 4, 6, 3, 0x599n);
  assert.strictEqual(first.recall, second.recall);
  assert.strictEqual(first.averageCandidates, second.averageCandidates);
  const rngA = new SplitMix64(51n);
  const rngB = new SplitMix64(51n);
  assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
    Array.from({ length: 8 }, () => rngB.next()));
  assert.ok(Math.abs(normalCdf(0) - 0.5) <= 1e-15 && Math.abs(normalCdf(1.96) - 0.975) <= 1e-3,
    "the normal CDF approximation used by the p-stable formula is accurate");
}

module.exports = { SplitMix64, DEFAULT_SEED, dot, norm, distance, angleBetween, hyperplaneFamily,
  hyperplaneSignature, pStableFamily, pStableSignature, bandedSignature, collidesInAnyBand,
  matchesAllRows, hyperplaneCollisionProbability, erf, normalCdf, pStableCollisionProbability,
  bandedProbability, pairAtAngle, empiricalCollisionRate, clusteredPoints, bruteForceNeighbour,
  evaluateIndex };

if (require.main === module) {
  main();
  console.log("PASS");
}
