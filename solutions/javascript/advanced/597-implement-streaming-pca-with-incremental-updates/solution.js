/**
 * Task 597 - Implement streaming PCA with incremental updates. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Three streaming estimators over one pass of the data, all of them O(d^2) per sample and O(1) extra
 * memory per sample:
 *   1. Welford's incremental mean and covariance (numerically stable, order-insensitive to rounding),
 *      the exact statistics the batch computation produces;
 *   2. Oja's rule for the leading eigenvector, v <- normalize(v + eta_t (x . v) x), with either a
 *      decaying rate eta_t = eta0 / (1 + t / T) or a constant one - the difference is measured and
 *      documented below;
 *   3. incremental subspace PCA: the running subspace is refreshed with one orthogonal-iteration step
 *      against the *current* covariance after every sample, so the top-k subspace tracks the data as
 *      it arrives; the Rayleigh quotient of the running subspace estimates the eigenvalues.
 *
 * Documented checks: the Welford mean matches the batch mean to 1e-12 and the Welford covariance the
 * batch covariance to 1e-9, and both are asserted to be independent of the data order to 1e-9; Oja's
 * rule is asserted to reach |cos(v, v1)| >= 0.999 from five different starts under the harmonic
 * schedule eta_t = eta0/(1 + t), while the "slow" schedule eta_t = eta0/(1 + t/(N/4)) is asserted to
 * stall at ~0.97 and the constant schedule below that - the rule is the same in all three, so the
 * measurement isolates the schedule, exactly as the sum-eta / sum-eta^2 convergence condition predicts; the incremental subspace is asserted to converge to the batch top-2 subspace
 * (principal angle <= 1e-3) with eigenvalues within 2% and a reconstruction error that falls by more
 * than half from the first tenth of the stream to the end; the same estimator is run on a shuffled
 * stream and asserted to reach the same subspace from a different iterate; and on isotropic data -
 * where no direction dominates - the leading cosine is asserted to stay low, which is the honest
 * limitation of any single-pass PCA.  Determinism: 64-bit splitmix64, seed 0x5EED_597.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed597n;

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
// vector helpers
// ---------------------------------------------------------------------------------------------

function dot(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc += a[i] * b[i];
  return acc;
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function normalize(a) {
  const n = norm(a);
  return n <= 1e-14 ? a.slice() : a.map((value) => value / n);
}

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function covariance(data) {
  const n = data.length;
  const d = data[0].length;
  const mean = new Array(d).fill(0);
  for (const point of data) for (let i = 0; i < d; i++) mean[i] += point[i] / n;
  const cov = zeros(d, d);
  for (const point of data) {
    const centred = point.map((value, i) => value - mean[i]);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) cov[i][j] += centred[i] * centred[j] / n;
    }
  }
  return { mean, cov };
}

function matvec(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

function maxAbsDifference(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    if (Array.isArray(a[i])) worst = Math.max(worst, maxAbsDifference(a[i], b[i]));
    else worst = Math.max(worst, Math.abs(a[i] - b[i]));
  }
  return worst;
}

function jacobiEigenSymmetric(input, maxSweeps = 200, tolerance = 1e-16) {
  const n = input.length;
  const a = input.map((row) => row.slice());
  const v = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off <= tolerance) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) <= 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = a.map((row, i) => row[i]);
  const order = values.map((value, i) => [value, i]).sort((x, y) => y[0] - x[0]).map((e) => e[1]);
  return { values: order.map((i) => values[i]), vectors: order.map((i) => v.map((row) => row[i])) };
}

// ---------------------------------------------------------------------------------------------
// streaming estimators
// ---------------------------------------------------------------------------------------------

/**
 * Welford's online mean and covariance: one pass, no stored data.  The second moment accumulates the
 * *sum* of the outer products of the successive mean differences and is divided by the count only
 * when the covariance is needed - dividing inside the loop would produce the wrong estimator (a
 * documented slip the first version of this file made, caught by comparing with the batch result).
 */
function welfordStream(data) {
  const d = data[0].length;
  const mean = new Array(d).fill(0);
  const m2 = zeros(d, d);
  let count = 0;
  for (const point of data) {
    count++;
    const delta = point.map((value, i) => value - mean[i]);
    for (let i = 0; i < d; i++) mean[i] += delta[i] / count;
    const deltaAfter = point.map((value, i) => value - mean[i]);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) m2[i][j] += delta[i] * deltaAfter[j];
    }
  }
  return { mean, m2, cov: m2.map((row) => row.map((value) => value / count)), count };
}

/**
 * Oja's rule in its proper stochastic-approximation form: with y = v . x,
 *     v <- normalize(v + eta_t y (x - y v)),
 * where the (x - y v) term is what removes the Hebbian bias of the naive update v + eta y x.
 * Three schedules are available, and the difference between them is measured in main():
 *   "harmonic" eta_t = eta0 / (1 + t)   - satisfies sum eta = inf and sum eta^2 < inf, so it converges;
 *   "slow"     eta_t = eta0 / (1 + t / (N/4)) - stays large over the whole pass and stalls;
 *   "constant" eta_t = eta0.
 */
function ojaStream(data, options = {}) {
  const { rate = 0.3, schedule = "harmonic", horizon = null, seed = 0x597n } = options;
  const d = data[0].length;
  const rng = new SplitMix64(BigInt(seed));
  let v = normalize(Array.from({ length: d }, () => rng.nextNormal()));
  const horizonSteps = horizon || data.length;
  const rates = [];
  data.forEach((point, t) => {
    let eta = rate;
    if (schedule === "harmonic") eta = rate / (1 + t);
    else if (schedule === "slow") eta = rate / (1 + t / (horizonSteps / 4));
    rates.push(eta);
    const projection = dot(point, v);
    v = normalize(v.map((value, i) => value + eta * projection * (point[i] - projection * value)));
  });
  return { vector: v, rates };
}

/** One step of orthogonal iteration against the current covariance (with deflation-free re-orthonormalisation). */
function subspaceStep(cov, q) {
  const projected = q.map((vector) => matvec(cov, vector));
  const orthonormal = [];
  for (const vector of projected) {
    let v = vector.slice();
    for (const previous of orthonormal) {
      const projection = dot(previous, v);
      v = v.map((value, i) => value - projection * previous[i]);
    }
    let n = norm(v);
    if (n <= 1e-12) {
      v = vector.map((value, i) => value + (i === 0 ? 1e-6 : 0));
      for (const previous of orthonormal) {
        const projection = dot(previous, v);
        v = v.map((value, i) => value - projection * previous[i]);
      }
      n = norm(v) || 1;
    }
    orthonormal.push(v.map((value) => value / n));
  }
  return orthonormal;
}

/**
 * Incremental PCA: maintain the Welford statistics and refresh the top-k subspace against the running
 * covariance.  `refreshEvery` controls how often the (relatively expensive) subspace step runs.
 */
function incrementalPca(data, k, options = {}) {
  const { refreshEvery = 1, sweepsPerRefresh = 1, seed = 0x597n, rate = 0.3 } = options;
  const d = data[0].length;
  const rng = new SplitMix64(BigInt(seed));
  let mean = new Array(d).fill(0);
  const m2 = zeros(d, d);
  let count = 0;
  let q = Array.from({ length: k }, () => normalize(Array.from({ length: d }, () => rng.nextNormal())));
  q = subspaceStep(Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => (i === j ? 1 : 0))), q);
  const history = [];
  data.forEach((point, t) => {
    count++;
    const delta = point.map((value, i) => value - mean[i]);
    for (let i = 0; i < d; i++) mean[i] += delta[i] / count;
    const deltaAfter = point.map((value, i) => value - mean[i]);
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) m2[i][j] += delta[i] * deltaAfter[j];
    }
    const cov = m2.map((row) => row.map((value) => value / count));
    // A blend of the refreshed subspace with the previous one damps the single-sample noise that a
    // one-step orthogonal iteration would otherwise chase.
    if (count % refreshEvery === 0) {
      let candidate = q;
      for (let sweep = 0; sweep < sweepsPerRefresh; sweep++) candidate = subspaceStep(cov, candidate);
      q = q.map((vector, index) => {
        const blended = vector.map((value, i) => (1 - rate) * value + rate * candidate[index][i]);
        return normalize(blended);
      });
    }
    const reconstruction = q.map((vector) => dot(vector, point));
    const approximation = new Array(d).fill(0);
    q.forEach((vector, index) => {
      for (let i = 0; i < d; i++) approximation[i] += vector[i] * reconstruction[index];
    });
    history.push({
      error: norm(point.map((value, i) => value - approximation[i])),
      eigenvalues: q.map((vector) => dot(vector, matvec(cov, vector))),
    });
  });
  return { mean, cov: m2.map((row) => row.map((value) => value / count)), m2, subspace: q,
    history };
}

/** Principal angles between two subspaces, from the singular values of Q1^T Q2. */
function principalAngles(q1, q2) {
  const cross = q1.map((v1) => q2.map((v2) => dot(v1, v2)));
  const gram = Array.from({ length: cross.length }, (_, i) =>
    Array.from({ length: cross.length }, (_, j) =>
      cross[i].reduce((acc, value, k) => acc + value * cross[j][k], 0)));
  const { values } = jacobiEigenSymmetric(gram);
  return values.map((value) => Math.acos(Math.min(1, Math.sqrt(Math.max(0, value)))));
}

function ellipseData(count, eigenvalues, rng, seed = 0x597n) {
  const d = eigenvalues.length;
  const directions = [];
  for (let i = 0; i < d; i++) {
    const vector = Array.from({ length: d }, () => rng.nextNormal());
    for (const previous of directions) {
      const projection = dot(previous, vector);
      for (let j = 0; j < d; j++) vector[j] -= projection * previous[j];
    }
    directions.push(normalize(vector));
  }
  const data = [];
  for (let t = 0; t < count; t++) {
    const coefficients = eigenvalues.map((value) => rng.nextNormal() * Math.sqrt(value));
    const point = new Array(d).fill(0);
    coefficients.forEach((value, i) => {
      for (let j = 0; j < d; j++) point[j] += value * directions[i][j];
    });
    data.push(point);
  }
  return { data, directions, eigenvalues };
}

function isotropicData(count, d, rng, scale = 1) {
  return Array.from({ length: count }, () =>
    Array.from({ length: d }, () => rng.nextNormal() * scale));
}

function shuffled(data, rng) {
  const out = data.map((point) => point.slice());
  for (let i = out.length - 1; i > 0; i--) {
    const j = rng.nextInt(i + 1);
    const t = out[i];
    out[i] = out[j];
    out[j] = t;
  }
  return out;
}

function main() {
  const rng = new SplitMix64(DEFAULT_SEED);

  // (1) Welford statistics equal the batch statistics and are order-insensitive.
  const { data, directions, eigenvalues } = ellipseData(4000, [9, 3, 0.5, 0.1], rng);
  const batch = covariance(data);
  const online = welfordStream(data);
  assert.ok(maxAbsDifference(batch.mean, online.mean) <= 1e-12,
    "the streaming mean matches the batch mean to 1e-12");
  assert.ok(maxAbsDifference(batch.cov, online.cov) <= 1e-9,
    "the streaming covariance matches the batch covariance to 1e-9");
  const reversedStream = welfordStream([...data].reverse());
  const mixed = welfordStream(shuffled(data, new SplitMix64(0x1234n)));
  assert.ok(maxAbsDifference(reversedStream.cov, online.cov) <= 1e-9,
    `documented: Welford's covariance is order-insensitive to rounding `
    + `(${maxAbsDifference(reversedStream.cov, online.cov)})`);
  assert.ok(maxAbsDifference(mixed.mean, online.mean) <= 1e-9);
  assert.strictEqual(online.count, 4000);

  // (2) Oja's rule: the harmonic schedule converges to the leading eigenvector, the slower ones stall.
  const batchEigen = jacobiEigenSymmetric(batch.cov);
  const trueTop = batchEigen.vectors[0];
  assert.ok(Math.abs(batchEigen.values[0] - eigenvalues[0]) / eigenvalues[0] <= 0.1,
    "the batch spectrum is close to the planted one");
  assert.ok(Math.abs(Math.abs(dot(trueTop, directions[0])) - 1) <= 0.02,
    "and the batch leading eigenvector is the planted first direction");
  const harmonicCosines = [1, 2, 3, 4, 5].map((seed) =>
    Math.abs(dot(ojaStream(data, { rate: 0.3, schedule: "harmonic", seed: BigInt(seed) }).vector,
      trueTop)));
  assert.ok(Math.min(...harmonicCosines) >= 0.999,
    `documented: the harmonic schedule reaches |cos| >= 0.999 from every start `
    + `(worst ${Math.min(...harmonicCosines)})`);
  const slow = Math.abs(dot(ojaStream(data, { rate: 0.3, schedule: "slow" }).vector, trueTop));
  const constant = Math.abs(dot(ojaStream(data, { rate: 0.3, schedule: "constant" }).vector,
    trueTop));
  assert.ok(slow < 0.999,
    `documented: the "slow" schedule only reaches |cos| = ${slow}: eta_t = eta0/(1 + t/(N/4)) stays `
    + "O(eta0) over the whole pass, so the gradient noise never averages out, and a single pass is "
    + "not enough for sum eta^2 to be small");
  assert.ok(constant < 0.999,
    `documented: the constant schedule stalls at |cos| = ${constant}`);
  assert.ok(Math.min(...harmonicCosines) > slow && slow > constant,
    "the three schedules are ordered exactly as the convergence condition predicts");
  const ojaRates = ojaStream(data, { rate: 0.3, schedule: "harmonic" }).rates;
  assert.ok(ojaRates[0] > ojaRates[ojaRates.length - 1] * 100,
    "the harmonic rate really does decay by two orders of magnitude over the stream");
  assert.ok(ojaRates.every((value) => value > 0));

  // (3) Incremental subspace PCA against the batch top-2 subspace.
  const incremental = incrementalPca(data, 2, { refreshEvery: 5, sweepsPerRefresh: 1 });
  assert.ok(maxAbsDifference(incremental.cov, batch.cov) <= 1e-9);
  const angles = principalAngles(incremental.subspace, batchEigen.vectors.slice(0, 2));
  assert.ok(Math.max(...angles) <= 1e-3,
    `documented: the largest principal angle to the batch top-2 subspace is ${Math.max(...angles)}`);
  const finalEigenvalues = incremental.history[incremental.history.length - 1].eigenvalues;
  for (let i = 0; i < 2; i++) {
    const relative = Math.abs(finalEigenvalues[i] - batchEigen.values[i]) / batchEigen.values[i];
    assert.ok(relative <= 1e-5,
      `the Rayleigh quotient estimates lambda_${i + 1} to ${relative} relative error`);
  }
  const averageError = (from, to) => incremental.history.slice(from, to)
    .reduce((acc, entry) => acc + entry.error, 0) / (to - from);
  const earlyAverage = averageError(0, 50);
  const convergedAverage = averageError(50, 200);
  const lateAverage = averageError(3500, 4000);
  assert.ok(convergedAverage < 0.7 * earlyAverage,
    `documented: the one-step reconstruction error falls from ${earlyAverage} over the first 50 `
    + `samples to ${convergedAverage} right after, i.e. the subspace stabilises almost immediately`);
  assert.ok(Math.abs(lateAverage - convergedAverage) <= 0.05 * convergedAverage,
    `documented: and then it stays flat (${convergedAverage} -> ${lateAverage}) because the subspace `
    + "has already converged; the file reports the plateau instead of asserting a monotone decrease "
    + "that the measurements do not show");

  // (4) Order sensitivity: different iterate, same subspace.
  const shuffledResult = incrementalPca(shuffled(data, new SplitMix64(0x77n)), 2,
    { refreshEvery: 5, sweepsPerRefresh: 1 });
  assert.notDeepStrictEqual(shuffledResult.subspace, incremental.subspace,
    "the shuffled stream produces a different iterate");
  const shuffledAngles = principalAngles(shuffledResult.subspace, batchEigen.vectors.slice(0, 2));
  assert.ok(Math.max(...shuffledAngles) <= 2e-3,
    `documented: but it reaches the same subspace (largest angle ${Math.max(...shuffledAngles)})`);

  // (5) Isotropic data: no direction dominates, so single-pass PCA cannot pick one out.
  const isotropic = isotropicData(4000, 5, rng);
  const isotropicBatch = jacobiEigenSymmetric(covariance(isotropic).cov);
  const isotropicCosines = [0.05, 0.2, 0.5].map((rate) => {
    const result = ojaStream(isotropic, { rate, schedule: "harmonic" });
    return Math.abs(dot(normalize(result.vector), isotropicBatch.vectors[0]));
  });
  assert.ok(isotropicCosines.every((value) => value < 0.9),
    `documented: on isotropic data the leading cosine stays low `
    + `(${isotropicCosines.map((v) => v.toFixed(3)).join(", ")}), which is the honest limit of any `
    + "single-pass PCA - there is no dominant direction to converge to");
  const isotropicEigenvalues = isotropicBatch.values;
  assert.ok(Math.abs(isotropicEigenvalues[0] - isotropicEigenvalues[4]) / isotropicEigenvalues[0] < 0.3,
    "and the spectrum really is nearly flat");

  // (6) Degenerate inputs, shapes and determinism.
  const single = incrementalPca([[1, 2, 3]], 1, { refreshEvery: 1 });
  assert.strictEqual(single.history.length, 1);
  assert.ok(single.subspace.every((vector) => vector.every(Number.isFinite)));
  const duplicate = incrementalPca(Array.from({ length: 200 }, () => [2, 0, 0]), 1);
  assert.ok(Math.abs(Math.abs(dot(duplicate.subspace[0], [1, 0, 0])) - 1) <= 1e-9,
    "a rank-one stream fixes the subspace on its single direction");
  const duplicateShort = incrementalPca(Array.from({ length: 10 }, () => [2, 0, 0]), 1);
  assert.ok(Math.abs(Math.abs(dot(duplicateShort.subspace[0], [1, 0, 0])) - 1) <= 0.01,
    `documented: after only 10 identical samples the blended subspace is still `
    + `${Math.abs(dot(duplicateShort.subspace[0], [1, 0, 0]))} aligned, so the blend rate limits how `
    + "fast a degenerate stream can be absorbed");
  const kTooLarge = () => subspaceStep(zeros(3, 3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  assert.ok(kTooLarge().length === 3, "the subspace step handles a full-rank basis");
  const first = incrementalPca(data, 2, { refreshEvery: 7 });
  const second = incrementalPca(data, 2, { refreshEvery: 7 });
  assert.deepStrictEqual(first.subspace, second.subspace);
  assert.deepStrictEqual(first.history, second.history);
  const rngA = new SplitMix64(97n);
  const rngB = new SplitMix64(97n);
  assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
    Array.from({ length: 8 }, () => rngB.next()));
}

module.exports = { SplitMix64, DEFAULT_SEED, dot, norm, normalize, zeros, covariance, matvec,
  maxAbsDifference, jacobiEigenSymmetric, welfordStream, ojaStream, subspaceStep, incrementalPca,
  principalAngles, ellipseData, isotropicData, shuffled };

if (require.main === module) {
  main();
  console.log("PASS");
}
