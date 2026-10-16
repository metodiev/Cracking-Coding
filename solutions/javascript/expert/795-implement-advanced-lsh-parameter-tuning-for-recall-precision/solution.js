/** Task 795 - Implement advanced LSH parameter tuning for recall/precision. (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
 *
 * Random-hyperplane LSH for cosine similarity: a hyperplane through the origin defined by a random
 * Gaussian vector p hashes a point to sign(p . x), and for two points separated by angle theta the
 * two bits agree with probability
 *   p(theta) = 1 - theta / pi
 * (the classic Goemans-Williamson/SRP fact, verified empirically below against many sampled
 * hyperplanes). A banding scheme of b bands with r rows each makes a pair a *candidate* when at least
 * one of its b r-bit signatures matches, so the candidate probability is
 *   1 - (1 - p(theta)^r)^b.
 * Recall is then measured against brute-force near neighbours (cosine >= threshold); precision is the
 * fraction of candidates that really are near neighbours. Tuning means choosing (b, r) under a
 * b*r-hyperplane budget that reaches a target recall with the fewest candidate pairs, and the search
 * below verifies both that the winner reaches the target and that no cheaper configuration does.
 *
 * Invariants verified: per-bit collision frequencies match the closed form 1 - theta/pi, measured
 * recall/precision are monotone in b and r respectively, the predicted candidate probability tracks the
 * measured one, identical points always collide while orthogonal ones collide half the time, tuning
 * returns a minimal-cost feasible configuration, and empty/single-point inputs are handled.
 *
 * What is simplified: an in-memory banded table over a few dozen points (no disk buckets, no
 * multi-probe or LSH forest), and a single hash family. A seeded splitmix32 supplies every hyperplane,
 * so the tables are deterministic and all reported rates are measured rather than assumed.
 */
"use strict";

const { ok, throws } = require("node:assert");

/** Deterministic splitmix32 uniforms in [0, 1). */
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return ((z ^ (z >>> 15)) >>> 0) / 4294967296;
  };
}

/** Standard normal from two uniforms (Box-Muller). */
const gaussian = (random) => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

const dot = (a, b) => a.reduce((acc, v, i) => acc + v * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));
const cosine = (a, b) => dot(a, b) / (norm(a) * norm(b));

/** Exact collision probability of one random hyperplane for angular separation theta. */
const collisionProbability = (theta) => 1 - theta / Math.PI;

/** Builds b*r Gaussian hyperplanes from a seed (the single source of randomness in this file). */
function makeHyperplanes(count, dimension, seed) {
  const random = makeRng(seed);
  return Array.from({ length: count }, () => Array.from({ length: dimension }, () => gaussian(random)));
}

/** Banded signatures: signatures[i] holds b integers, each built from r hyperplane signs. */
function signatures(vectors, hyperplanes, { bands, rows }) {
  return vectors.map((vector) => {
    const bits = hyperplanes.map((plane) => (dot(plane, vector) >= 0 ? 1 : 0));
    return Array.from({ length: bands }, (_, band) => {
      let value = 0;
      for (let k = 0; k < rows; k++) value = value * 2 + bits[band * rows + k];
      return value;
    });
  });
}

/** Candidate pairs (i < j) sharing at least one band bucket. */
function candidatePairs(signaturesList, { bands }) {
  const candidates = new Set();
  for (let band = 0; band < bands; band++) {
    const buckets = new Map();
    signaturesList.forEach((signature, i) => {
      const key = signature[band];
      if (!buckets.has(key)) buckets.set(key, []);
      buckets.get(key).push(i);
    });
    for (const bucket of buckets.values()) {
      for (let a = 0; a < bucket.length; a++) for (let b = a + 1; b < bucket.length; b++) candidates.add(`${bucket[a]}|${bucket[b]}`);
    }
  }
  return candidates;
}

/** Brute-force near-neighbour pairs: cosine similarity at least `threshold`. */
function neighbourPairs(vectors, threshold) {
  const pairs = new Set();
  for (let i = 0; i < vectors.length; i++) {
    for (let j = i + 1; j < vectors.length; j++) {
      if (cosine(vectors[i], vectors[j]) >= threshold) pairs.add(`${i}|${j}`);
    }
  }
  return pairs;
}

/** Retrieval quality of one configuration, measured against the brute-force answers. */
function evaluate(vectors, neighbours, { bands, rows, seed }) {
  const hyperplanes = makeHyperplanes(bands * rows, vectors[0] ? vectors[0].length : 1, seed);
  const candidates = candidatePairs(signatures(vectors, hyperplanes, { bands, rows }), { bands });
  let hits = 0;
  for (const pair of candidates) if (neighbours.has(pair)) hits += 1;
  const recall = neighbours.size === 0 ? 1 : hits / neighbours.size;
  const precision = candidates.size === 0 ? 1 : hits / candidates.size;
  const theta = Math.acos(0.6);
  const predicted = 1 - (1 - collisionProbability(0) ** rows) ** bands;
  void theta;
  return { recall, precision, candidates: candidates.size, hits, recallPredicted: predicted };
}

/** Predicted candidate probability for a pair at angle theta under (b, r). */
const predictedCandidateRate = (theta, { bands, rows }) => 1 - (1 - collisionProbability(theta) ** rows) ** bands;

/**
 * Tunes (b, r) under a hyperplane budget: the feasible configuration with the fewest candidate pairs.
 * Returns every trial so the caller can check the minimality claim.
 */
function tune(vectors, neighbours, { budget = 48, targetRecall = 0.9, seed = 11, grid = null } = {}) {
  if (vectors.length === 0) throw new RangeError("tuning needs at least one vector");
  if (!(budget >= 1)) throw new RangeError("budget must be at least one hyperplane");
  const trials = [];
  const cells = grid ?? Array.from({ length: budget }, (_, k) => k + 1).flatMap((rows) => {
    const bands = Math.floor(budget / rows);
    return bands >= 1 ? [{ bands, rows }] : [];
  });
  for (const { bands, rows } of cells) {
    if (bands * rows > budget) continue;
    const result = evaluate(vectors, neighbours, { bands, rows, seed });
    trials.push({ bands, rows, ...result, cost: result.candidates });
  }
  const feasible = trials.filter((trial) => trial.recall >= targetRecall);
  if (feasible.length === 0) throw new Error("no configuration within the budget reaches the target recall");
  const best = feasible.reduce((acc, trial) => (trial.cost < acc.cost || (trial.cost === acc.cost && trial.bands * trial.rows < acc.bands * acc.rows) ? trial : acc));
  return { best, trials, feasible };
}

/** Deterministic cloud: three clusters of unit vectors in a low-dimensional space. */
function makeCloud(seed, perCluster = 20, dimension = 8) {
  const random = makeRng(seed);
  const directions = [
    [1, 0.2, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0.15, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0.25, 0, 0],
  ];
  const cloud = [];
  for (const direction of directions) {
    for (let i = 0; i < perCluster; i++) {
      cloud.push(Array.from({ length: dimension }, (_, d) => direction[d] + gaussian(random) * 0.35));
    }
  }
  return cloud;
}

function main() {
  const notes = [];
  const dimension = 8;
  const cloud = makeCloud(20251016);
  const neighbours = neighbourPairs(cloud, 0.6);

  // 1. Per-bit collision frequency must match the closed form 1 - theta/pi.
  const random = makeRng(99);
  const probes = [];
  for (let k = 0; k < 6; k++) {
    const a = Array.from({ length: dimension }, () => gaussian(random));
    const b = Array.from({ length: dimension }, () => gaussian(random));
    probes.push([a, b]);
  }
  const hyperplanes = makeHyperplanes(20000, dimension, 4242);
  let worstRateDeviation = 0;
  const rateReport = probes.map(([a, b]) => {
    const theta = Math.acos(Math.min(1, Math.max(-1, cosine(a, b))));
    let collisions = 0;
    for (const plane of hyperplanes) if ((dot(plane, a) >= 0) === (dot(plane, b) >= 0)) collisions += 1;
    const measured = collisions / hyperplanes.length;
    const theory = collisionProbability(theta);
    worstRateDeviation = Math.max(worstRateDeviation, Math.abs(measured - theory));
    return `${theta.toFixed(3)} rad: ${measured.toFixed(4)} vs ${theory.toFixed(4)}`;
  });
  notes.push(`per-bit collision rate (20000 hyperplanes): ${rateReport.join("; ")}`);
  ok(worstRateDeviation < 0.01, `measured collision rates must match 1 - theta/pi (worst deviation ${worstRateDeviation.toFixed(5)})`);

  // 2. Orientation sanity: identical points always collide, orthogonal points half the time.
  const identicalRate = predictedCandidateRate(0, { bands: 3, rows: 4 });
  const orthogonalRate = predictedCandidateRate(Math.PI / 2, { bands: 3, rows: 4 });
  notes.push(`theory: identical points collide with probability ${identicalRate.toFixed(6)}, orthogonal pairs with ${orthogonalRate.toFixed(4)} (bands 3, rows 4)`);
  ok(identicalRate > 1 - 1e-12, "identical points must always share a band");
  ok(Math.abs(orthogonalRate - (1 - (1 - 0.5 ** 4) ** 3)) < 1e-15, "orthogonal pairs must use p = 0.5");

  // 3. Recall is monotone in the number of bands, precision in the rows per band.
  const bandSweep = [1, 2, 4, 8].map((bands) => evaluate(cloud, neighbours, { bands, rows: 4, seed: 11 }).recall);
  const rowSweep = [1, 2, 4, 8].map((rows) => evaluate(cloud, neighbours, { bands: 8, rows, seed: 11 }).precision);
  notes.push(`band sweep (rows 4) recall: ${bandSweep.map((r) => r.toFixed(3)).join(" -> ")}; row sweep (bands 8) precision: ${rowSweep.map((p) => p.toFixed(3)).join(" -> ")}`);
  ok(bandSweep.every((r, i) => i === 0 || r >= bandSweep[i - 1] - 1e-12), "more bands must never reduce recall");
  ok(rowSweep.every((p, i) => i === 0 || p >= rowSweep[i - 1] - 1e-12), "longer bands must never reduce precision");
  ok(bandSweep.at(-1) > bandSweep[0] || bandSweep.at(-1) === 1, "the band sweep must actually change recall");

  // 4. Predicted candidate rate versus measured: use the empirical angle distribution.
  const angles = [];
  for (let i = 0; i < cloud.length; i++) {
    for (let j = i + 1; j < cloud.length; j++) angles.push(Math.acos(Math.min(1, Math.max(-1, cosine(cloud[i], cloud[j])))));
  }
  const configuration = { bands: 6, rows: 4 };
  const measured = evaluate(cloud, neighbours, { ...configuration, seed: 11 });
  const predictedCandidates = angles.reduce((acc, theta) => acc + predictedCandidateRate(theta, configuration), 0);
  notes.push(`candidate-rate prediction for bands 6, rows 4: ${predictedCandidates.toFixed(1)} expected vs ${measured.candidates} measured pairs; recall ${measured.recall.toFixed(3)}, precision ${measured.precision.toFixed(3)}`);
  ok(Math.abs(predictedCandidates - measured.candidates) / Math.max(1, measured.candidates) < 0.25, "the analytical candidate count must track the measured one");

  // 5. Tuning must return a feasible configuration that no cheaper one can replace.
  const tuning = tune(cloud, neighbours, { budget: 32, targetRecall: 0.9, seed: 11 });
  notes.push(`tuning (budget 32 hyperplanes, target recall 0.9): bands ${tuning.best.bands}, rows ${tuning.best.rows} -> recall ${tuning.best.recall.toFixed(3)}, precision ${tuning.best.precision.toFixed(3)}, ${tuning.best.candidates} candidate pairs of ${(cloud.length * (cloud.length - 1)) / 2} possible`);
  ok(tuning.best.recall >= 0.9, `the tuned configuration must reach the target recall (got ${tuning.best.recall})`);
  ok(tuning.trials.filter((t) => t.cost < tuning.best.cost).every((t) => t.recall < 0.9), "no cheaper configuration may also reach the target");
  ok(tuning.best.candidates <= (cloud.length * (cloud.length - 1)) / 2, "candidates can never exceed the number of pairs");
  ok(tuning.best.bands * tuning.best.rows <= 32, "the tuned configuration must respect the hyperplane budget");

  // 6. A tighter target must cost at least as much as a looser one.
  const looseTarget = tune(cloud, neighbours, { budget: 32, targetRecall: 0.7, seed: 11 });
  const strictTarget = tune(cloud, neighbours, { budget: 48, targetRecall: 0.97, seed: 11 });
  notes.push(`target sweep: recall >= 0.7 costs ${looseTarget.best.candidates} pairs, recall >= 0.97 costs ${strictTarget.best.candidates} pairs`);
  ok(strictTarget.best.candidates >= looseTarget.best.candidates, "a stricter recall target must not get cheaper");
  ok(looseTarget.best.recall >= 0.7 && strictTarget.best.recall >= 0.97, "each target must actually be met");

  // 7. Boundary cases: an empty cloud, a single point, and duplicate points.
  const emptyNeighbours = neighbourPairs([], 0.5);
  const emptyEval = evaluate([], emptyNeighbours, { bands: 2, rows: 2, seed: 1 });
  ok(emptyEval.candidates === 0 && emptyEval.recall === 1, "an empty cloud must give no candidates and a vacuous recall of 1");
  ok(evaluate([[1, 0]], new Set(), { bands: 2, rows: 2, seed: 1 }).candidates === 0, "a single point must give no candidates");
  const duplicates = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]];
  const duplicatePairs = candidatePairs(signatures(duplicates, makeHyperplanes(8, 4, 5), { bands: 2, rows: 4 }), { bands: 2 });
  notes.push(`duplicate points: ${duplicatePairs.size} of 3 possible pairs collide (identical signatures always share every band)`);
  ok(duplicatePairs.size === 3, "identical points must collide in every band");

  // 8. Validation and determinism.
  throws(() => tune([], new Set(), { budget: 4, targetRecall: 0.99 }), RangeError, "an empty cloud must throw before hashing");
  const repeat = evaluate(cloud, neighbours, { bands: 6, rows: 4, seed: 11 });
  ok(repeat.recall === measured.recall && repeat.candidates === measured.candidates, "the same seed must reproduce the tables exactly");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { makeRng, gaussian, dot, cosine, collisionProbability, makeHyperplanes, signatures, candidatePairs, neighbourPairs, evaluate, predictedCandidateRate, tune, makeCloud };

if (require.main === module) main();
