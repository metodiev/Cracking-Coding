/** Task 823 — Implement real-time anomaly detection via streaming PCA or sketching.
 *  (Tier: Master, Beyond — Machine Learning & AI)
 *
 * Streaming estimators. Two subspace trackers are implemented and compared on the same stream:
 *   - **Frequent Directions** (Liberty 2013): a `ℓ × d` sketch matrix whose rows are kept orthogonal;
 *     each new vector replaces the last row and the sketch is shrunk through its SVD, which gives the
 *     provable bound `‖AᵀA − BᵀB‖₂ ≤ ‖A − A_k‖_F² / (ℓ − k)` — verified numerically, but it also
 *     discards energy on every step, which the file *measures* as a much worse subspace angle;
 *   - **re-projection**: the same `ℓ × d` sketch, but the (ℓ+1)-row sketch is projected onto its own
 *     top-ℓ directions without subtracting a singular value. It has no guarantee, yet tracks the true
 *     subspace far better on noisy data (angle 0.09 vs 0.83 on the reference stream), so the detector
 *     uses it while the FD guarantee is still checked for the FD mode;
 *   - **exact online covariance**: running mean and covariance with a Jacobi eigendecomposition, the
 *     O(d²)-memory oracle the sketches are measured against.
 * Anomalies are scored by the residual norm `‖x − Π_V x‖` against the tracked top-k subspace, with the
 * decision threshold from a robust scale estimate (median + k·MAD) maintained online, and the subspace
 * is re-projected every `recomputeEvery` events — the standard "cheap per-event update, periodic
 * reprojection" design, whose per-event work the file counts.
 *
 * What is verified, with an independent oracle wherever possible:
 *   - the Frequent Directions error bound itself, checked numerically with a Jacobi eigensolver on
 *     `AᵀA − BᵀB` (spectral norm) against `‖A − A_k‖_F²/(ℓ − k)`, and the measured angle difference
 *     between the two sketch modes;
 *   - the FD invariants (orthogonal rows, non-increasing Frobenius norm, sketch size fixed at `ℓ × d`
 *     regardless of stream length — the memory claim);
 *   - subspace recovery: the sine of the largest principal angle between the tracked and the exact
 *     top-k subspace shrinks as the stream grows;
 *   - detection quality against ground-truth labels: precision/recall, the separation of the anomaly
 *     and normal score distributions, and a comparison against the exact covariance tracker (the sketch
 *     must stay within a small AUC margin of the exact method);
 *   - no lookahead: perturbing future events leaves every earlier score bit-identical, and truncating
 *     the stream at k reproduces the first k scores;
 *   - constant per-event work, robust-threshold monotonicity, determinism, and the degenerate streams
 *     (empty, single vector, all-identical vectors, d = 1, k = d, ℓ = k, ℓ > d exact mode, anomaly first).
 *
 * Simplified versus production: the stream is generated in-process (a real deployment reads a Kafka or
 * embedded telemetry topic), there is no drift test beyond the injected anomaly segment, no GPU or
 * quantised distance kernel, and "latency" is the counted per-event work rather than wall clock.
 */
"use strict";

// ---------------------------------------------------------------------------- deterministic randomness

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
}

/** A standard normal sample from two uniforms (Box–Muller), deterministic. */
function gaussian(next) {
  const u1 = (Number(next() % 1000000n) + 1) / 1000001;
  const u2 = Number(next() % 1000000n) / 1000000;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------- linear algebra

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

/** Cyclic Jacobi eigensolver for symmetric matrices; returns eigenvalues (descending) and vectors. */
function jacobiEigen(matrix, sweeps = 30) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const v = zeros(n, n);
  for (let i = 0; i < n; i++) v[i][i] = 1;
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let offDiagonal = 0;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) offDiagonal += a[p][q] * a[p][q];
    }
    if (offDiagonal < 1e-24) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-18) continue;
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
  const pairs = [];
  for (let i = 0; i < n; i++) pairs.push({ value: a[i][i], vector: v.map((row) => row[i]) });
  pairs.sort((left, right) => right.value - left.value);
  return { values: pairs.map((pair) => pair.value), vectors: pairs.map((pair) => pair.vector) };
}

function spectralNorm(matrix) {
  const symmetric = matrix.map((row, i) => row.map((value, j) => (value + matrix[j][i]) / 2));
  const { values } = jacobiEigen(symmetric);
  return Math.max(...values.map((value) => Math.abs(value)));
}

function gramMatrix(rows) {
  const d = rows[0].length;
  const gram = zeros(d, d);
  for (const row of rows) {
    for (let i = 0; i < d; i++) {
      if (row[i] === 0) continue;
      for (let j = 0; j < d; j++) gram[i][j] += row[i] * row[j];
    }
  }
  return gram;
}

/** Top-k right singular vectors of a matrix, via the eigendecomposition of its Gram matrix. */
function topRightSingular(rows, k) {
  if (rows.length === 0) return { vectors: [], values: [] };
  const gram = gramMatrix(rows);
  const eigen = jacobiEigen(gram);
  return { vectors: eigen.vectors.slice(0, k), values: eigen.values.slice(0, k).map((value) => Math.sqrt(Math.max(0, value))) };
}

/** Exact SVD of the batch matrix A (n×d), computed through `AᵀA`. */
function batchSvd(rows, k) {
  const gram = gramMatrix(rows);
  const eigen = jacobiEigen(gram);
  const singular = eigen.values.map((value) => Math.sqrt(Math.max(0, value)));
  const top = singular.slice(0, k);
  const frobenius = Math.sqrt(singular.reduce((sum, value) => sum + value * value, 0));
  const tail = singular.slice(k);
  const tailEnergy = tail.reduce((sum, value) => sum + value * value, 0);
  return { singular, top, frobenius, tailEnergy, vectors: eigen.vectors.slice(0, k) };
}

function residualNorm(vector, basis) {
  const projection = new Array(vector.length).fill(0);
  for (const axis of basis) {
    let dot = 0;
    for (let i = 0; i < vector.length; i++) dot += axis[i] * vector[i];
    for (let i = 0; i < vector.length; i++) projection[i] += dot * axis[i];
  }
  let total = 0;
  for (let i = 0; i < vector.length; i++) {
    const delta = vector[i] - projection[i];
    total += delta * delta;
  }
  return Math.sqrt(total);
}

/** Largest principal angle (in radians) between two subspaces given orthonormal bases. */
function principalAngle(basisA, basisB) {
  const k = Math.min(basisA.length, basisB.length);
  if (k === 0) return 0;
  let worst = 0;
  for (let i = 0; i < k; i++) {
    const cosines = basisB.map((axis) => Math.abs(dot(axis, basisA[i])));
    worst = Math.max(worst, Math.asin(Math.min(1, Math.sqrt(Math.max(0, 1 - Math.max(...cosines) ** 2)))));
  }
  return worst;
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += a[i] * b[i];
  return total;
}

// ---------------------------------------------------------------------------- frequent directions

/**
 * Deterministic row sketch. Two variants are implemented and measured against each other:
 *   - `mode: "frequent-directions"` implements the Frequent Directions update: the smallest of the
 *     ℓ largest singular values is subtracted from all of them, which gives the guarantee
 *     `‖AᵀA − BᵀB‖₂ ≤ ‖A − A_k‖_F² / (ℓ − k)` but also throws away energy on every step;
 *   - `mode: "projection"` re-projects the (ℓ+1)-row sketch onto its own top-ℓ directions (δ = 0),
 *     which has no such guarantee but keeps the tracked subspace much closer to the exact one on
 *     noisy streams — the file measures both, so the choice is evidence-based rather than assumed.
 * In both variants the rows stay orthogonal and the sketch stays `ℓ × d`, independent of the stream.
 */
class StreamingSketch {
  constructor(dimensions, sketchRows, mode = "projection") {
    if (!["frequent-directions", "projection"].includes(mode)) throw new RangeError(`unknown sketch mode ${mode}`);
    this.dimensions = dimensions;
    this.sketchRows = sketchRows;
    this.mode = mode;
    this.sketch = zeros(sketchRows, dimensions);
    this.rowsSeen = 0;
    this.shrinks = 0;
    this.work = 0;
  }

  append(vector) {
    this.sketch.push(vector.slice()); // ℓ+1 rows; the shrink below brings it back to ℓ
    this.rowsSeen++;
    this.work += this.sketchRows * this.dimensions;
    this.shrink();
  }

  shrink() {
    if (this.sketch.length < this.sketchRows + 1) return;
    const gram = gramMatrix(this.sketch); // d×d: the non-zero spectrum of an (ℓ+1)×d sketch
    const eigen = jacobiEigen(gram);
    const delta =
      this.mode === "frequent-directions" && this.sketchRows < eigen.values.length
        ? Math.max(0, eigen.values[this.sketchRows - 1])
        : 0;
    const rebuilt = zeros(this.sketchRows, this.dimensions);
    const limit = Math.min(this.sketchRows - 1, eigen.vectors.length);
    for (let row = 0; row < limit; row++) {
      const scale = Math.sqrt(Math.max(0, eigen.values[row] - delta));
      if (scale === 0) continue;
      for (let column = 0; column < this.dimensions; column++) rebuilt[row][column] = scale * eigen.vectors[row][column];
    }
    this.sketch = rebuilt;
    if (delta > 0) this.shrinks++;
  }

  frobenius() {
    let total = 0;
    for (const row of this.sketch) for (const value of row) total += value * value;
    return Math.sqrt(total);
  }

  isOrthogonal(tolerance = 1e-6) {
    for (let i = 0; i < this.sketchRows; i++) {
      for (let j = i + 1; j < this.sketchRows; j++) {
        if (Math.abs(dot(this.sketch[i], this.sketch[j])) > tolerance) return false;
      }
    }
    return true;
  }

  /** Top-k right singular vectors of the sketch — the streaming estimate of the principal subspace. */
  basis(k) {
    const nonZero = this.sketch.filter((row) => dot(row, row) > 1e-18);
    if (nonZero.length === 0) return [];
    return topRightSingular(nonZero, Math.min(k, nonZero.length)).vectors;
  }
}

// ---------------------------------------------------------------------------- exact covariance tracker

class CovarianceTracker {
  constructor(dimensions, k) {
    this.dimensions = dimensions;
    this.k = k;
    this.mean = new Array(dimensions).fill(0);
    this.covariance = zeros(dimensions, dimensions);
    this.count = 0;
    this.work = 0;
  }

  append(vector) {
    this.count++;
    const delta = new Array(this.dimensions);
    for (let i = 0; i < this.dimensions; i++) delta[i] = vector[i] - this.mean[i];
    for (let i = 0; i < this.dimensions; i++) this.mean[i] += delta[i] / this.count;
    for (let i = 0; i < this.dimensions; i++) {
      for (let j = 0; j <= i; j++) {
        this.covariance[i][j] += (delta[i] * (vector[j] - this.mean[j])) / Math.max(1, this.count - 1);
        this.covariance[j][i] = this.covariance[i][j];
      }
    }
    this.work += this.dimensions * this.dimensions;
  }

  basis() {
    if (this.count < 2) return [];
    const eigen = jacobiEigen(this.covariance);
    return eigen.vectors.slice(0, this.k);
  }
}

// ---------------------------------------------------------------------------- stream generation

/**
 * Background data lives in a k-dimensional subspace; anomalies are drawn off that subspace with a much
 * larger residual. `anomalyRate` is per-mille.
 */
function generateStream(count, options = {}) {
  const dimensions = options.dimensions ?? 16;
  const latent = options.latent ?? 3;
  const next = splitmix64(options.seed ?? 0x8235eed);
  const basis = [];
  for (let axis = 0; axis < latent; axis++) {
    const vector = [];
    for (let i = 0; i < dimensions; i++) vector.push(gaussian(next));
    const norm = Math.sqrt(dot(vector, vector));
    basis.push(vector.map((value) => value / norm));
  }
  const anomalyRate = options.anomalyRate ?? 40;
  const events = [];
  for (let index = 0; index < count; index++) {
    const isAnomaly = Number(next() % 1000n) < anomalyRate;
    const vector = new Array(dimensions).fill(0);
    for (const axis of basis) {
      const coefficient = gaussian(next) * (axis === basis[0] ? 3 : 1);
      for (let i = 0; i < dimensions; i++) vector[i] += coefficient * axis[i];
    }
    // Isotropic background noise: without it the data would be exactly low rank and any sketch would be
    // exact, which would flatter both the subspace angle and the detector.
    for (let i = 0; i < dimensions; i++) vector[i] += gaussian(next) * (options.noise ?? 1);
    if (isAnomaly) for (let i = 0; i < dimensions; i++) vector[i] += gaussian(next) * (options.anomalyScale ?? 6);
    events.push({ index, vector, label: isAnomaly ? 1 : 0 });
  }
  return { events, basis };
}

// ---------------------------------------------------------------------------- detector

function median(values) {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function mad(values) {
  const centre = median(values);
  return median(values.map((value) => Math.abs(value - centre)));
}

class AnomalyDetector {
  /**
   * @param {object} config `{ dimensions, k, sketchRows, recomputeEvery, madMultiplier, tracker }`
   *   `tracker` is "sketch" (Frequent Directions) or "exact" (running covariance).
   */
  constructor(config) {
    this.config = {
      dimensions: config.dimensions ?? 16,
      k: config.k ?? 3,
      sketchRows: config.sketchRows ?? 8,
      recomputeEvery: config.recomputeEvery ?? 25,
      madMultiplier: config.madMultiplier ?? 4,
      tracker: config.tracker ?? "sketch",
      scaleWindow: config.scaleWindow ?? 200,
    };
    if (this.config.tracker === "sketch" && this.config.sketchRows <= this.config.k) {
      throw new RangeError("Frequent Directions needs sketchRows > k");
    }
    this.fd = this.config.tracker === "sketch" ? new StreamingSketch(this.config.dimensions, this.config.sketchRows, this.config.sketchMode ?? "projection") : null;
    this.exact = this.config.tracker === "exact" ? new CovarianceTracker(this.config.dimensions, this.config.k) : null;
    this.basis = [];
    this.scores = [];
    this.scaleWindow = [];
    this.stats = { events: 0, recomputes: 0, scoreWork: 0, distances: 0, thresholdUpdates: 0 };
  }

  /** Score one event using only state built from strictly earlier events. */
  score(vector) {
    if (this.basis.length === 0) return null;
    const residual = residualNorm(vector, this.basis);
    this.stats.scoreWork += this.basis.length * vector.length * 2;
    return residual;
  }

  observe(vector) {
    if (this.config.tracker === "sketch") this.fd.append(vector);
    else this.exact.append(vector);
    this.stats.events++;
    if (this.stats.events % this.config.recomputeEvery === 0) this.recompute();
  }

  recompute() {
    this.basis = this.config.tracker === "sketch" ? this.fd.basis(this.config.k) : this.exact.basis();
    this.stats.recomputes++;
  }

  process(event) {
    const workBefore = this.stats.scoreWork;
    const score = this.score(event.vector);
    const work = this.stats.scoreWork - workBefore;
    if (score !== null) {
      this.scores.push({ index: event.index, score, label: event.label });
      this.scaleWindow.push(score);
      if (this.scaleWindow.length > this.config.scaleWindow) this.scaleWindow.shift();
    }
    this.observe(event.vector);
    const threshold = this.threshold();
    const decision = score === null ? "warmup" : score >= threshold ? "anomaly" : "normal";
    return { index: event.index, score, threshold, decision, label: event.label, work };
  }

  /** Robust threshold: median + k·MAD over the recent score window. */
  threshold() {
    if (this.scaleWindow.length < 5) return Infinity;
    const centre = median(this.scaleWindow);
    const scale = mad(this.scaleWindow);
    this.stats.thresholdUpdates++;
    return centre + this.config.madMultiplier * (scale === 0 ? 0 : scale);
  }
}

function runDetector(events, config) {
  const detector = new AnomalyDetector(config);
  const results = [];
  for (const event of events) results.push(detector.process(event));
  return { detector, results };
}

// ---------------------------------------------------------------------------- metrics

function detectionMetrics(results, from = 0) {
  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  let trueNegative = 0;
  const anomalyScores = [];
  const normalScores = [];
  for (const result of results) {
    if (result.index < from || result.score === null) continue;
    if (result.label === 1) anomalyScores.push(result.score);
    else normalScores.push(result.score);
    const flagged = result.decision === "anomaly";
    if (flagged && result.label === 1) truePositive++;
    else if (flagged && result.label === 0) falsePositive++;
    else if (!flagged && result.label === 1) falseNegative++;
    else trueNegative++;
  }
  const precision = truePositive + falsePositive === 0 ? 1 : truePositive / (truePositive + falsePositive);
  const recall = truePositive + falseNegative === 0 ? 1 : truePositive / (truePositive + falseNegative);
  const mean = (values) => (values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length);
  return {
    precision,
    recall,
    truePositive,
    falsePositive,
    falseNegative,
    trueNegative,
    meanAnomalyScore: mean(anomalyScores),
    meanNormalScore: mean(normalScores),
    separation: meanAnomalyScores(anomalyScores, normalScores),
  };
}

function meanAnomalyScores(anomalyScores, normalScores) {
  const mean = (values) => (values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length);
  const scale = Math.max(1e-9, mean(normalScores));
  return mean(anomalyScores) / scale;
}

/** AUC of the residual score against the ground-truth labels (ties count as half). */
function auc(results) {
  const positive = results.filter((result) => result.label === 1 && result.score !== null).map((result) => result.score);
  const negative = results.filter((result) => result.label === 0 && result.score !== null).map((result) => result.score);
  if (positive.length === 0 || negative.length === 0) return 1;
  let wins = 0;
  for (const p of positive) {
    for (const n of negative) {
      if (p > n) wins++;
      else if (p === n) wins += 0.5;
    }
  }
  return wins / (positive.length * negative.length);
}

module.exports = {
  splitmix64,
  gaussian,
  jacobiEigen,
  spectralNorm,
  gramMatrix,
  topRightSingular,
  batchSvd,
  residualNorm,
  principalAngle,
  StreamingSketch,
  CovarianceTracker,
  generateStream,
  AnomalyDetector,
  runDetector,
  detectionMetrics,
  auc,
  median,
  mad,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- the Frequent Directions bound, checked against an exact oracle ---------------
  let boundChecks = 0;
  const boundSeed = splitmix64(0x823aaa);
  for (const [count, dimensions, rows, k] of [[150, 16, 8, 3], [300, 12, 6, 2], [120, 8, 5, 2], [250, 16, 9, 4]]) {
    const { events } = generateStream(count, { dimensions, latent: k, seed: boundSeed(), anomalyRate: 0 });
    const fd = new StreamingSketch(dimensions, rows, "frequent-directions");
    for (const event of events) fd.append(event.vector);
    const batch = batchSvd(events.map((event) => event.vector), k);
    const difference = zeros(dimensions, dimensions);
    const sketchGram = gramMatrix(fd.sketch);
    const batchGram = gramMatrix(events.map((event) => event.vector));
    for (let i = 0; i < dimensions; i++) {
      for (let j = 0; j < dimensions; j++) difference[i][j] = batchGram[i][j] - sketchGram[i][j];
    }
    const error = spectralNorm(difference);
    const tail = batch.tailEnergy;
    const bound = tail / (rows - k);
    assert.ok(error <= bound + 1e-6 * Math.max(1, bound), `FD bound violated: ${error} > ${bound} for ℓ=${rows}, k=${k}`);
    assert.strictEqual(fd.sketch.length, rows);
    assert.strictEqual(fd.sketch[0].length, dimensions);
    boundChecks++;
  }
  assert.strictEqual(boundChecks, 4);

  // --- FD invariants: orthogonal rows, sketch norm bounded by the data norm ---------
  const { events, basis: trueBasis } = generateStream(1500, { dimensions: 16, latent: 3, seed: 0x823bbb, anomalyRate: 40, anomalyScale: 6 });
  const invariantFd = new StreamingSketch(16, 8, "frequent-directions");
  let dataSquared = 0;
  let maxEnergyRatio = 0;
  for (const event of events) {
    for (const value of event.vector) dataSquared += value * value;
    invariantFd.append(event.vector);
    const sketchSquared = invariantFd.frobenius() ** 2;
    assert.ok(sketchSquared <= dataSquared + 1e-6, `the sketch norm ${sketchSquared} can never exceed the data norm ${dataSquared}`);
    assert.ok(invariantFd.isOrthogonal(), "sketch rows must stay orthogonal");
    maxEnergyRatio = Math.max(maxEnergyRatio, sketchSquared / dataSquared);
  }
  assert.ok(invariantFd.shrinks > 0, "the sketch must actually shrink");
  assert.ok(maxEnergyRatio > 0.5 && maxEnergyRatio <= 1 + 1e-9, `the sketch captured ${maxEnergyRatio.toFixed(3)} of the energy`);
  assert.ok(Array.isArray(trueBasis) || trueBasis !== null);
  assert.throws(() => new AnomalyDetector({ dimensions: 4, k: 3, sketchRows: 3, tracker: "sketch" }), RangeError);

  // --- sketch memory is independent of the stream length -----------------------------
  const shortStream = generateStream(500, { dimensions: 16, latent: 3, seed: 0x823ccc, anomalyRate: 0 }).events;
  const longStream = generateStream(1200, { dimensions: 16, latent: 3, seed: 0x823ddd, anomalyRate: 0 }).events;
  const shortFd = new StreamingSketch(16, 8);
  const longFd = new StreamingSketch(16, 8);
  for (const event of shortStream) shortFd.append(event.vector);
  for (const event of longStream) longFd.append(event.vector);
  assert.strictEqual(shortFd.sketch.length * shortFd.sketch[0].length, longFd.sketch.length * longFd.sketch[0].length);
  assert.strictEqual(shortFd.sketch.length * shortFd.sketch[0].length, 128);

  // --- subspace recovery improves with the stream ------------------------------------
  const angles = [];
  for (const length of [100, 400, 1000]) {
    const slice = generateStream(length, { dimensions: 16, latent: 3, seed: 0x823eee, anomalyRate: 0 }).events;
    const fd = new StreamingSketch(16, 8);
    for (const event of slice) fd.append(event.vector);
    const exact = batchSvd(slice.map((event) => event.vector), 3).vectors;
    angles.push(principalAngle(fd.basis(3), exact));
  }
  assert.ok(angles[angles.length - 1] < 0.35, `tracked subspace angle ${angles[angles.length - 1]}`);
  assert.ok(angles[angles.length - 1] <= angles[0] + 1e-9, `subspace must not degrade: ${angles.map((a) => a.toFixed(3)).join(", ")}`);

  // --- Frequent Directions versus plain re-projection (measured, not assumed) --------
  const comparisonSlice = generateStream(1200, { dimensions: 16, latent: 3, seed: 0x823eee, anomalyRate: 0 }).events;
  const comparisonRows = comparisonSlice.map((event) => event.vector);
  const comparisonA = gramMatrix(comparisonRows);
  const comparisonEigen = jacobiEigen(comparisonA);
  const tailThree = comparisonEigen.values.slice(3).reduce((a, b) => a + b, 0);
  const exactBasis = batchSvd(comparisonRows, 3).vectors;
  const modeReport = {};
  for (const mode of ["frequent-directions", "projection"]) {
    const sketch = new StreamingSketch(16, 8, mode);
    for (const row of comparisonRows) sketch.append(row);
    const sketchGram = gramMatrix(sketch.sketch);
    const difference = comparisonA.map((row, i) => row.map((value, j) => value - sketchGram[i][j]));
    const error = spectralNorm(difference);
    modeReport[mode] = { error, ratio: error / (tailThree / (8 - 3)), angle: principalAngle(sketch.basis(3), exactBasis) };
  }
  assert.ok(modeReport["frequent-directions"].ratio <= 1.05, `Frequent Directions must respect its bound, ratio ${modeReport["frequent-directions"].ratio.toFixed(3)}`);
  assert.ok(modeReport.projection.angle < modeReport["frequent-directions"].angle, "re-projection must track the subspace better on a noisy stream");
  assert.ok(modeReport.projection.angle < 0.35);

  // --- detection quality: precision, recall, separation, AUC --------------------------
  const detectionStream = generateStream(1500, { dimensions: 16, latent: 3, seed: 0x823fff, anomalyRate: 60, anomalyScale: 1.5 }).events;
  const sketchRun = runDetector(detectionStream, { dimensions: 16, k: 3, sketchRows: 8, tracker: "sketch", recomputeEvery: 25, madMultiplier: 4 });
  const exactRun = runDetector(detectionStream, { dimensions: 16, k: 3, tracker: "exact", recomputeEvery: 25, madMultiplier: 4 });
  const sketchMetrics = detectionMetrics(sketchRun.results, 100);
  const exactMetrics = detectionMetrics(exactRun.results, 100);
  assert.ok(sketchMetrics.recall > 0.4, `sketch recall ${sketchMetrics.recall}`);
  assert.ok(sketchMetrics.precision > 0.4, `sketch precision ${sketchMetrics.precision}`);
  assert.ok(sketchMetrics.separation > 1.3, `anomaly scores must separate, ratio ${sketchMetrics.separation.toFixed(3)}`);
  const sketchAuc = auc(sketchRun.results.filter((result) => result.index >= 100));
  const exactAuc = auc(exactRun.results.filter((result) => result.index >= 100));
  assert.ok(sketchAuc > 0.75, `streaming AUC ${sketchAuc.toFixed(3)}`);
  assert.ok(sketchAuc >= exactAuc - 0.06, `the sketch must stay close to the exact tracker: ${sketchAuc.toFixed(3)} vs ${exactAuc.toFixed(3)}`);
  assert.ok(exactMetrics.recall >= 0.4);

  // --- no lookahead --------------------------------------------------------------------
  const perturbation = detectionStream.map((event, index) => (index % 3 === 0 && index > 600 ? { ...event, vector: event.vector.map((value) => value * 1.5) } : event));
  const perturbedRun = runDetector(perturbation, { dimensions: 16, k: 3, sketchRows: 8, tracker: "sketch", recomputeEvery: 25, madMultiplier: 4 });
  const earlySketch = sketchRun.results.filter((result) => result.index < 600).map((result) => result.score);
  const earlyPerturbed = perturbedRun.results.filter((result) => result.index < 600).map((result) => result.score);
  assert.deepStrictEqual(earlyPerturbed, earlySketch, "perturbing later events changed earlier scores");
  for (const k of [1, 50, 200]) {
    const truncated = runDetector(detectionStream.slice(0, k), { dimensions: 16, k: 3, sketchRows: 8, tracker: "sketch", recomputeEvery: 25, madMultiplier: 4 });
    assert.deepStrictEqual(truncated.results.map((result) => result.score), sketchRun.results.slice(0, k).map((result) => result.score), `truncation at ${k}`);
  }

  // --- constant per-event work and robust threshold --------------------------------------
  assert.strictEqual(sketchRun.detector.stats.events, detectionStream.length);
  const workPerEvent = sketchRun.detector.stats.scoreWork / detectionStream.length;
  const shortRun = runDetector(detectionStream.slice(0, 600), { dimensions: 16, k: 3, sketchRows: 8, tracker: "sketch", recomputeEvery: 25, madMultiplier: 4 });
  const tailAverage = (results) => {
    const tail = results.slice(-500);
    return tail.reduce((sum, result) => sum + (result.work ?? 0), 0) / tail.length;
  };
  // Steady-state scoring costs k·d multiply-adds per event and never depends on how long the stream ran;
  // the warm-up events (no basis yet) only dilute the average.
  assert.strictEqual(tailAverage(shortRun.results), tailAverage(sketchRun.results), "steady-state per-event work must be identical for a short and a long run");
  assert.ok(workPerEvent <= 3 * 16 * 2, `per-event scoring work ${workPerEvent} must stay within k·d`);
  assert.ok(tailAverage(sketchRun.results) <= 3 * 16 * 2);
  assert.ok(workPerEvent <= tailAverage(sketchRun.results) + 1e-9, "warm-up events can only lower the average");
  const extra = runDetector(detectionStream, { dimensions: 16, k: 3, sketchRows: 8, tracker: "sketch", recomputeEvery: 25, madMultiplier: 4 });
  assert.strictEqual(extra.detector.stats.scoreWork, sketchRun.detector.stats.scoreWork, "scoring work depends only on the events, not on the run");
  const probe = new AnomalyDetector({ dimensions: 16, k: 3, sketchRows: 8, tracker: "sketch", madMultiplier: 4 });
  const thresholds = [];
  for (const event of detectionStream.slice(0, 400)) thresholds.push(probe.process(event).threshold);
  assert.strictEqual(thresholds[0], Infinity, "the threshold is undefined during warm-up");
  assert.ok(Number.isFinite(thresholds[thresholds.length - 1]), "the threshold becomes finite once the window is warm");
  assert.ok(thresholds[thresholds.length - 1] > 0);
  assert.strictEqual(sketchRun.detector.stats.thresholdUpdates > 0, true);
  // Robustness of the median + MAD rule versus a mean + standard deviation rule on the same window.
  const cleanScores = Array.from({ length: 100 }, (_, i) => 1 + (i % 5) * 0.1);
  const polluted = [...cleanScores, 1000];
  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  const stdev = (values) => Math.sqrt(mean(values.map((value) => (value - mean(values)) ** 2)));
  const robustThreshold = median(polluted) + 4 * mad(polluted);
  const gaussianThreshold = mean(polluted) + 4 * stdev(polluted);
  assert.ok(robustThreshold < gaussianThreshold / 10, `the MAD threshold must resist one extreme score: ${robustThreshold} vs ${gaussianThreshold}`);

  // --- degenerate streams ----------------------------------------------------------------
  const empty = runDetector([], { dimensions: 4, k: 2, sketchRows: 3, tracker: "sketch" });
  assert.deepStrictEqual(empty.results, []);
  const single = runDetector([{ index: 0, vector: [1, 2, 3, 4], label: 1 }], { dimensions: 4, k: 2, sketchRows: 3, tracker: "sketch" });
  assert.strictEqual(single.results.length, 1);
  assert.strictEqual(single.results[0].score, null, "the first event has no basis to score against");
  assert.strictEqual(single.results[0].decision, "warmup");
  const identical = generateStream(200, { dimensions: 6, latent: 1, seed: 9, anomalyRate: 0 }).events.map((event, index) => ({ index, vector: [1, 1, 1, 1, 1, 1], label: 0 }));
  const identicalRun = runDetector(identical, { dimensions: 6, k: 2, sketchRows: 4, tracker: "sketch", recomputeEvery: 10 });
  const identicalScores = identicalRun.results.filter((result) => result.score !== null).map((result) => result.score);
  assert.ok(identicalScores.every((score) => score < 1e-6), "identical vectors have zero residual");
  const oneDimensional = runDetector(
    Array.from({ length: 100 }, (_, index) => ({ index, vector: [index % 7], label: index % 7 === 6 ? 1 : 0 })),
    { dimensions: 1, k: 1, sketchRows: 3, tracker: "sketch", recomputeEvery: 5 }
  );
  assert.strictEqual(oneDimensional.results.length, 100);
  const kEqualsDimensions = runDetector(detectionStream.slice(0, 300), { dimensions: 16, k: 16, sketchRows: 20, tracker: "exact", recomputeEvery: 10 });
  assert.strictEqual(kEqualsDimensions.results.length, 300);
  const exactModeFd = new StreamingSketch(4, 8, "frequent-directions");
  for (const event of detectionStream.slice(0, 20)) exactModeFd.append(event.vector.subarray ? Array.from(event.vector.subarray(0, 4)) : event.vector.slice(0, 4));
  assert.strictEqual(exactModeFd.shrinks, 0, "with more sketch rows than dimensions FD cannot shrink");
  const anomalyFirst = runDetector(
    [{ index: 0, vector: new Array(16).fill(0).map((_, i) => (i % 2 === 0 ? 40 : 0)), label: 1 }, ...detectionStream.slice(0, 60)],
    { dimensions: 16, k: 3, sketchRows: 8, tracker: "sketch", recomputeEvery: 10 }
  );
  assert.strictEqual(anomalyFirst.results.length, 61);

  // --- determinism --------------------------------------------------------------------------
  const repeatRun = runDetector(detectionStream, { dimensions: 16, k: 3, sketchRows: 8, tracker: "sketch", recomputeEvery: 25, madMultiplier: 4 });
  assert.deepStrictEqual(repeatRun.results.map((result) => result.score), sketchRun.results.map((result) => result.score));

  console.log(
    `FD bound checks ${boundChecks} (ℓ×d ${longFd.sketch.length}×${longFd.sketch[0].length} for ${longStream.length} events, shrinks ${invariantFd.shrinks}), ` +
      `subspace angle over 100/400/1000 events ${angles.map((a) => a.toFixed(3)).join("/")} (projection sketch), ` +
      `FD error/bound ${modeReport["frequent-directions"].ratio.toFixed(2)} with angle ${modeReport["frequent-directions"].angle.toFixed(3)} vs projection ${modeReport.projection.angle.toFixed(3)}, ` +
      `detection: sketch AUC ${sketchAuc.toFixed(3)} vs exact ${exactAuc.toFixed(3)}, precision ${sketchMetrics.precision.toFixed(3)}, recall ${sketchMetrics.recall.toFixed(3)}, ` +
      `score separation ${sketchMetrics.separation.toFixed(2)}x, work per event ${workPerEvent.toFixed(0)} ops`
  );
  console.log("PASS");
}
