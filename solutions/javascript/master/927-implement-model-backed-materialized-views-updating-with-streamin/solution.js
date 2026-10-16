/** Task 927 - Implement model-backed materialized views updating with streaming features.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 * A materialized view whose rows are *predictions* of an online model, refreshed from a feature
 * stream:
 *   RidgeModel        streaming ridge regression. It keeps only the sufficient statistics
 *                     A = sum(x x') and b = sum(y x), so one sample costs O(d^2) and the
 *                     coefficients are solve(A + lambda I, b) whenever they are asked for. A batch
 *                     refit builds the same normal equations in a single pass, and for one feature
 *                     the closed form w = sum(xy) / (sum(x^2) + lambda) is an exact oracle.
 *   StreamingStats    Welford mean/variance for feature normalization, cross-checked against a
 *                     two-pass batch computation over the same stream.
 *   ModelBackedView   materializes one prediction per entity, marks itself stale as soon as the
 *                     model changes, and rebuilds either everything (refresh) or only the entities
 *                     whose features changed (refreshDirty) - the two paths must agree.
 * Simplification: linear ridge regression, in-memory entities, no persistence or versioning of the
 * model. Verified: incremental statistics equal a batch refit, the single-feature closed form,
 * Welford versus two-pass statistics, stale/dirty refresh equivalence, degenerate and empty
 * streams, invalid input and determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    return (this.state = x);
  }
  float() { return this.next() / 4294967296; }
  sym(scale) { return (this.float() * 2 - 1) * scale; }
}

/** Gauss-Jordan solve with partial pivoting; null when the system is singular. */
function solveLinear(A, rhs) {
  const n = rhs.length;
  const M = A.map((row, i) => row.concat([rhs[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    if (Math.abs(M[pivot][col]) < 1e-14) return null;
    const swap = M[col]; M[col] = M[pivot]; M[pivot] = swap;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

class RidgeModel {
  constructor(dimension, lambda) {
    assert.ok(Number.isInteger(dimension) && dimension > 0, "the dimension must be positive");
    assert.ok(lambda > 0, "a positive ridge term is required to keep the system solvable");
    this.dimension = dimension;
    this.lambda = lambda;
    this.A = Array.from({ length: dimension }, () => new Array(dimension).fill(0));
    this.b = new Array(dimension).fill(0);
    this.samples = 0;
    this.version = 0;
  }
  observe(features, target) {
    assert.strictEqual(features.length, this.dimension, "feature vector has the wrong dimension");
    assert.ok(Number.isFinite(target), "the target must be finite");
    for (const value of features) assert.ok(Number.isFinite(value), "features must be finite");
    for (let i = 0; i < this.dimension; i++) {
      this.b[i] += target * features[i];
      for (let j = 0; j < this.dimension; j++) this.A[i][j] += features[i] * features[j];
    }
    this.samples++;
    this.version++;
    return this;
  }
  coefficients() {
    const regularized = this.A.map((row, i) => row.map((v, j) => (i === j ? v + this.lambda : v)));
    const solution = solveLinear(regularized, this.b);
    assert.ok(solution, "the regularized system must be solvable");
    return solution;
  }
  predict(features) {
    assert.strictEqual(features.length, this.dimension, "feature vector has the wrong dimension");
    const weights = this.coefficients();
    return features.reduce((acc, value, i) => acc + value * weights[i], 0);
  }
  static fit(samples, dimension, lambda) {
    const model = new RidgeModel(dimension, lambda);
    for (const sample of samples) model.observe(sample.features, sample.target);
    return model;
  }
}

/** Welford's online mean/variance. */
class StreamingStats {
  constructor() { this.count = 0; this.mean = 0; this.m2 = 0; }
  observe(value) {
    assert.ok(Number.isFinite(value), "the observation must be finite");
    this.count++;
    const delta = value - this.mean;
    this.mean += delta / this.count;
    this.m2 += delta * (value - this.mean);
    return this;
  }
  get variance() { return this.count > 1 ? this.m2 / (this.count - 1) : 0; }
}

/** Two-pass batch variance, the independent oracle for the streaming statistics. */
function batchStats(values) {
  assert.ok(values.length > 0, "the batch must not be empty");
  const mean = values.reduce((a, v) => a + v, 0) / values.length;
  const variance = values.length > 1
    ? values.reduce((a, v) => a + (v - mean) ** 2, 0) / (values.length - 1) : 0;
  return { mean, variance };
}

class ModelBackedView {
  constructor(model, entities) {
    this.model = model;
    this.entities = entities instanceof Map ? new Map(entities) : new Map(Object.entries(entities));
    this.materialized = new Map();
    this.dirty = new Set(this.entities.keys());
    this.modelVersion = -1;
    this.recomputations = 0;
  }
  get stale() { return this.modelVersion !== this.model.version; }
  refresh() {
    this.materialized.clear();
    for (const [id, features] of this.entities) {
      this.materialized.set(id, this.model.predict(features));
      this.recomputations++;
    }
    this.dirty.clear();
    this.modelVersion = this.model.version;
    return this.materialized;
  }
  /** Refresh only the entities whose features were rewritten (the streaming path). */
  refreshDirty() {
    assert.ok(!this.stale || this.dirty.size > 0 || this.materialized.size === 0,
      "a stale view with no dirty entity must be fully refreshed");
    for (const id of this.dirty) {
      if (!this.entities.has(id)) continue;
      this.materialized.set(id, this.model.predict(this.entities.get(id)));
      this.recomputations++;
    }
    this.dirty.clear();
    this.modelVersion = this.model.version;
    return this.materialized;
  }
  updateFeatures(id, features) {
    assert.ok(this.entities.has(id), `unknown entity ${id}`);
    this.entities.set(id, features);
    this.dirty.add(id);
    return this;
  }
  get(id) {
    assert.ok(!this.stale, "the view is stale: refresh before reading");
    return this.materialized.get(id);
  }
}

function main() {
  const prng = new Prng(20250927);
  const dimension = 3;
  const lambda = 0.5;

  // 1. Streaming sufficient statistics reproduce a batch refit of the same normal equations.
  const stream = new RidgeModel(dimension, lambda);
  const samples = [];
  for (let k = 0; k < 120; k++) {
    const features = [1, prng.sym(2), prng.sym(2)];
    const target = 0.5 + 1.2 * features[0] - 0.7 * features[1] + 0.3 * features[2] + prng.sym(0.1);
    features[1] = features[0] * features[1];           // a mildly nonlinear feature interaction
    samples.push({ features, target });
    stream.observe(features, target);
  }
  const batch = RidgeModel.fit(samples, dimension, lambda);
  const streamingWeights = stream.coefficients();
  const batchWeights = batch.coefficients();
  assert.ok(streamingWeights.every((w, i) => Math.abs(w - batchWeights[i]) < 1e-9),
    "streaming and batch coefficients disagree");
  assert.strictEqual(stream.samples, samples.length);
  assert.strictEqual(batch.samples, samples.length);
  const half = RidgeModel.fit(samples.slice(0, 60), dimension, lambda).coefficients();
  assert.ok(half.some((w, i) => Math.abs(w - batchWeights[i]) > 1e-6),
    "the model must actually change as more samples arrive");

  // 2. Single-feature closed form: w = sum(x y) / (sum(x^2) + lambda), an exact oracle.
  {
    const one = new RidgeModel(1, 2);
    let sumXY = 0, sumXX = 0;
    for (let k = 0; k < 50; k++) {
      const x = prng.sym(3), y = 1.5 * x + prng.sym(0.2);
      sumXY += x * y;
      sumXX += x * x;
      one.observe([x], y);
    }
    const closed = sumXY / (sumXX + 2);
    assert.ok(Math.abs(one.coefficients()[0] - closed) < 1e-12, "the closed form disagrees");
    assert.ok(Math.abs(one.predict([4]) - 4 * closed) < 1e-9, "prediction must be linear in x");
  }

  // 3. Welford statistics match a two-pass batch computation on the same stream.
  {
    const stats = new StreamingStats();
    const values = [];
    for (let k = 0; k < 90; k++) {
      const value = 3 + prng.sym(1.5) * (1 + (k % 7));
      values.push(value);
      stats.observe(value);
    }
    const reference = batchStats(values);
    assert.ok(Math.abs(stats.mean - reference.mean) < 1e-12, "streaming mean drifted");
    assert.ok(Math.abs(stats.variance - reference.variance) < 1e-9, "streaming variance drifted");
    assert.strictEqual(stats.count, values.length);
  }

  // 4. The materialized view stays consistent with the model through streaming updates.
  const model = new RidgeModel(dimension, lambda);
  const entities = { a: [1, 0.5, -0.5], b: [2, -1, 0.25], c: [0.5, 0.75, 1] };
  const view = new ModelBackedView(model, entities);
  view.refresh();
  assert.strictEqual(view.recomputations, 3);
  for (const [id, features] of Object.entries(entities)) {
    assert.ok(Math.abs(view.get(id) - model.predict(features)) < 1e-12, `view row ${id} is wrong`);
  }
  for (const sample of samples.slice(0, 40)) model.observe(sample.features, sample.target);
  assert.ok(view.stale, "the view must notice that the model changed");
  assert.throws(() => view.get("a"), /stale/);
  const fullRefresh = [...view.refresh()];
  const featuresB = [2, -0.4, 0.1];
  view.updateFeatures("b", featuresB);
  view.refreshDirty();
  const dirtyValue = view.get("b");
  assert.ok(Math.abs(dirtyValue - model.predict(featuresB)) < 1e-12);
  assert.ok(!view.stale);
  const fullAgain = new ModelBackedView(model, view.entities).refresh();
  assert.deepStrictEqual([...fullAgain], [...view.materialized],
    "a full refresh and a dirty refresh must agree");
  assert.strictEqual(fullRefresh.length, 3);

  // 5. Boundary cases: an empty stream predicts from the prior, one sample is still stable.
  {
    const empty = new RidgeModel(2, 1);
    assert.strictEqual(empty.samples, 0);
    assert.deepStrictEqual(empty.coefficients(), [0, 0]);
    assert.strictEqual(empty.predict([5, -5]), 0);
    const single = new RidgeModel(2, 1);
    single.observe([2, 0], 6);
    const w = single.coefficients();
    assert.ok(w.every(Number.isFinite), "one sample must not make the solve singular");
    assert.ok(Math.abs(w[0] - 12 / 5) < 1e-12, "the high-variance feature must dominate");
    const constant = new RidgeModel(1, 3);
    for (let k = 0; k < 10; k++) constant.observe([1], 2);
    assert.ok(Math.abs(constant.coefficients()[0] - 2 / 1.3) < 1e-12,
      "a constant feature needs the ridge term");
    const statsZero = new StreamingStats();
    assert.strictEqual(statsZero.variance, 0);
    statsZero.observe(4);
    assert.strictEqual(statsZero.variance, 0, "a single observation has no variance");
    assert.strictEqual(statsZero.mean, 4);
  }

  // 6. Invalid input is rejected instead of silently corrupting the statistics.
  {
    const strict = new RidgeModel(2, 1);
    assert.throws(() => strict.observe([1], 1), /wrong dimension/);
    assert.throws(() => strict.observe([1, Number.NaN], 1), /finite/);
    assert.throws(() => strict.observe([1, 1], Infinity), /finite/);
    assert.throws(() => new RidgeModel(0, 1), /positive/);
    assert.throws(() => new RidgeModel(2, 0), /positive ridge/);
    assert.throws(() => new ModelBackedView(strict, {}).updateFeatures("x", [1, 1]), /unknown entity/);
    assert.throws(() => batchStats([]), /must not be empty/);
  }

  // 7. Determinism: the same seed rebuilds the same samples, weights and view.
  const rerun = () => {
    const p = new Prng(20250927);
    const local = new RidgeModel(2, 0.25);
    const localSamples = [];
    for (let k = 0; k < 30; k++) {
      const features = [1, p.sym(2)];
      const target = 1 + 0.5 * features[1] + p.sym(0.05);
      localSamples.push({ features, target });
      local.observe(features, target);
    }
    return { weights: local.coefficients(), batch: RidgeModel.fit(localSamples, 2, 0.25).coefficients() };
  };
  assert.deepStrictEqual(rerun(), rerun());
  assert.deepStrictEqual(rerun().weights, rerun().batch, "the two paths must agree exactly");

  console.log(`view: ${stream.samples} samples, ${dimension} features, ridge ${lambda}; ` +
    `streaming weights ${streamingWeights.map((w) => w.toFixed(4)).join(",")}; ` +
    `view rows ${view.materialized.size}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, RidgeModel, StreamingStats, batchStats, ModelBackedView, solveLinear };
