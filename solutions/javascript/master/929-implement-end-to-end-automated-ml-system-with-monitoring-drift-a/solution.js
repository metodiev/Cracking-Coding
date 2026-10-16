/** Task 929 - Implement end-to-end automated ML system with monitoring, drift alerts, retraining.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 * An automated loop over a drifting stream:
 *   trainer     ridge regression through the normal equations, so the fit is the exact solution of
 *               (X'X + lambda I) w = X'y - an independent oracle is the closed-form single-feature
 *               slope and the comparison against the mean predictor;
 *   monitor     a rolling window of recent (prediction error, input mean) statistics; a window is
 *               alarming when the rolling error exceeds the reference error by `errorSlack` or the
 *               input mean has moved by `meanSlack`, and an alert needs `minSamples` observations;
 *   automl      train -> serve -> monitor -> alert -> retrain, with a versioned model registry:
 *               every retrain bumps the version and rollback restores the previous weights exactly.
 * Simplification: one numeric target, no hyperparameter search beyond the ridge term, a single
 * in-memory registry and no feature store integration. Verified: the exact-fit oracle, the alert
 * firing under drift and staying silent on a stationary stream, performance recovery after
 * retraining, registry versioning and rollback, boundary windows, invalid input and determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    return (this.state = x >>> 0);
  }
  float() { return this.next() / 4294967296; }
  sym(scale) { return (this.float() * 2 - 1) * scale; }
  normal(scale) { return (this.float() + this.float() + this.float() - 1.5) * 2 * scale; }
}

/** Gauss-Jordan solve with partial pivoting. */
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

/** Ridge regression with an explicit bias term; the fit is exact, not approximate. */
class RidgeTrainer {
  constructor(lambda) {
    assert.ok(lambda > 0, "a positive ridge term keeps the system solvable");
    this.lambda = lambda;
  }
  fit(samples, dimension) {
    assert.ok(Array.isArray(samples) && samples.length > 0, "the training set is empty");
    const width = dimension + 1; // last column is the constant bias feature
    const A = Array.from({ length: width }, () => new Array(width).fill(0));
    const b = new Array(width).fill(0);
    for (const sample of samples) {
      assert.strictEqual(sample.features.length, dimension, "a sample has the wrong dimension");
      const row = [...sample.features, 1];
      for (let i = 0; i < width; i++) {
        b[i] += row[i] * sample.target;
        for (let j = 0; j < width; j++) A[i][j] += row[i] * row[j];
      }
    }
    for (let i = 0; i < dimension; i++) A[i][i] += this.lambda; // the intercept stays unpenalized
    const weights = solveLinear(A, b);
    assert.ok(weights, "the regularized normal equations must be solvable");
    return { weights: weights.slice(0, dimension), bias: weights[dimension], lambda: this.lambda,
      samples: samples.length };
  }
}

const meanSquaredError = (samples, model) => samples.reduce((acc, sample) => {
  const prediction = model.weights.reduce((sum, w, i) => sum + w * sample.features[i], model.bias);
  return acc + (prediction - sample.target) ** 2;
}, 0) / samples.length;

/** Stream generator: a linear concept that changes slope, offset and input scale at `driftAt`. */
function makeStream(prng, options = {}) {
  const count = options.count ?? 240;
  const driftAt = options.driftAt ?? Math.floor(count / 2);
  const drift = options.drift ?? { weight: 3, bias: 8, scale: 1.5 };
  const samples = [];
  for (let k = 0; k < count; k++) {
    const shifted = k >= driftAt;
    const features = [prng.normal(1) * (shifted ? drift.scale : 1), prng.normal(0.5)];
    const target = (shifted ? drift.weight : 1.5) * features[0] + (shifted ? drift.bias : 0)
      + 0.5 * features[1] + prng.normal(0.05);
    samples.push({ features, target, index: k, shifted });
  }
  return { samples, driftAt };
}

class Monitor {
  constructor(options = {}) {
    this.windowSize = options.windowSize ?? 20;
    this.errorSlack = options.errorSlack ?? 1.0;
    this.meanSlack = options.meanSlack ?? 0.8;
    this.minSamples = options.minSamples ?? 10;
    assert.ok(Number.isInteger(this.windowSize) && this.windowSize > 0, "windowSize must be positive");
    assert.ok(Number.isInteger(this.minSamples) && this.minSamples > 0, "minSamples must be positive");
    this.window = [];
    this.referenceError = null;
    this.referenceMean = null;
    this.alerts = [];
    this.readings = 0;
  }
  /** The reference is measured on the data the deployed model was trained on. */
  setReference(labeled, model) {
    this.referenceError = meanSquaredError(labeled, model);
    this.referenceMean = labeled.reduce((a, s) => a + s.features[0], 0) / labeled.length;
    this.window = [];
    return this;
  }
  observe(sample, model) {
    const prediction = model.weights.reduce((sum, w, i) => sum + w * sample.features[i], model.bias);
    this.window.push({ error: (prediction - sample.target) ** 2, mean: sample.features[0] });
    if (this.window.length > this.windowSize) this.window.shift();
    this.readings++;
    if (this.window.length < this.minSamples) return { status: "warming-up", window: this.window.length };
    const error = this.window.reduce((a, r) => a + r.error, 0) / this.window.length;
    const mean = this.window.reduce((a, r) => a + r.mean, 0) / this.window.length;
    if (this.referenceError === null) return { status: "warming-up", error, mean };
    const errorRatio = error / Math.max(1e-9, this.referenceError);
    const meanShift = Math.abs(mean - this.referenceMean);
    const alert = errorRatio > 1 + this.errorSlack || meanShift > this.meanSlack;
    if (alert) this.alerts.push({ error, mean, errorRatio, meanShift });
    return { status: alert ? "alarm" : "ok", error, mean, errorRatio, meanShift };
  }
  /** A new model resets the window and the reference. */
  rebase(model) {
    this.window = [];
    this.referenceError = null;
    this.referenceMean = null;
    return this;
  }
}

class AutoMlSystem {
  constructor(options = {}) {
    this.trainer = new RidgeTrainer(options.lambda ?? 0.05);
    this.dimension = options.dimension ?? 2;
    this.monitor = new Monitor(options.monitor ?? {});
    this.registry = [];
    this.current = null;
    this.retrains = 0;
    this.cursor = 0;
  }
  train(labeled) {
    const model = this.trainer.fit(labeled, this.dimension);
    this.registry.push(model);
    this.current = model;
    this.monitor.setReference(labeled, model);
    return model;
  }
  /** Promote the newest model; version 0 is the first one and the registry is append-only. */
  get version() { return this.registry.length - 1; }
  rollback(version) {
    assert.ok(Number.isInteger(version) && version >= 0 && version < this.registry.length,
      `unknown model version ${version}`);
    this.current = this.registry[version];
    this.monitor.rebase(this.current);
    return this.current;
  }
  get alerts() { return this.monitor.alerts; }
}

function main() {
  const prng = new Prng(20250929);

  // 1. The trainer is exact: its coefficients match the closed-form solution.
  {
    const trainer = new RidgeTrainer(0.5);
    let sumXY = 0, sumXX = 0;
    const rows = [];
    for (let k = 0; k < 60; k++) {
      const x = prng.sym(2), y = 2.5 * x - 1 + prng.sym(0.01);
      rows.push({ features: [x], target: y });
      sumXY += x * y;
      sumXX += x * x;
    }
    const fitted = trainer.fit(rows, 1);
    const closed = (sumXY - rows.reduce((a, r) => a + r.features[0], 0)
      * rows.reduce((a, r) => a + r.target, 0) / rows.length)
      / (sumXX - rows.reduce((a, r) => a + r.features[0], 0) ** 2 / rows.length + 0.5);
    assert.ok(Math.abs(fitted.weights[0] - closed) < 1e-9, "the ridge fit disagrees with the closed form");
    assert.ok(meanSquaredError(rows, fitted) < 1e-3, "a near-noiseless fit must be accurate");
    assert.ok(meanSquaredError(rows, fitted) < meanSquaredError(rows,
      { weights: [0], bias: rows.reduce((a, r) => a + r.target, 0) / rows.length }),
    "the model must beat the mean predictor");
  }

  // 2. Happy path: train on the first half, then the stationary half keeps the error stable.
  const stationary = makeStream(new Prng(3), { count: 200, drift: { weight: 1.5, bias: 0, scale: 1 } });
  const system = new AutoMlSystem({ dimension: 2, lambda: 0.05, monitor: { windowSize: 40, minSamples: 20 } });
  const firstHalf = stationary.samples.slice(0, 100);
  const initial = system.train(firstHalf);
  assert.strictEqual(system.version, 0);
  assert.ok(meanSquaredError(firstHalf, initial) < 0.05, "the initial fit should be tight");
  let statuses = [];
  for (const sample of stationary.samples.slice(100)) {
    statuses.push(system.monitor.observe(sample, system.current).status);
  }
  assert.strictEqual(system.alerts.length, 0, "a stationary stream must not raise alerts");
  assert.ok(statuses.includes("warming-up") && statuses.includes("ok"),
    "the monitor must warm up and then report ok");

  // 3. Drift: the monitor alarms, retraining recovers performance, and the version advances.
  const drifting = makeStream(new Prng(3), { count: 200 });
  const auto = new AutoMlSystem({ dimension: 2, lambda: 0.05, monitor: { windowSize: 40, minSamples: 20 } });
  const baseline = auto.train(drifting.samples.slice(0, drifting.driftAt));
  let errorBeforeAlert = null;
  let alertIndex = -1;
  for (let k = drifting.driftAt; k < drifting.samples.length; k++) {
    const reading = auto.monitor.observe(drifting.samples[k], auto.current);
    if (reading.status === "alarm" && alertIndex < 0) {
      alertIndex = k;
      errorBeforeAlert = reading.error;
    }
  }
  assert.ok(alertIndex >= 0, "drift must raise an alert");
  assert.ok(alertIndex - drifting.driftAt <= 60, `the alert took ${alertIndex - drifting.driftAt} samples`);
  const retrained = auto.train(drifting.samples.slice(alertIndex - 20, alertIndex));
  auto.retrains++;
  assert.strictEqual(auto.version, 1, "retraining must bump the registry version");
  const after = drifting.samples.slice(alertIndex + 20);
  assert.ok(meanSquaredError(after, retrained) < errorBeforeAlert * 0.5,
    `error ${meanSquaredError(after, retrained)} did not recover from ${errorBeforeAlert}`);
  assert.ok(meanSquaredError(after, retrained) < meanSquaredError(after, baseline) / 5,
    "the retrained model must beat the stale one by a wide margin");

  // 4. Alert contents are meaningful: the error ratio or the input mean actually moved.
  const alert = auto.alerts[0];
  assert.ok(alert.errorRatio > 1.25 || alert.meanShift > 0.4, "an alert needs a real excursion");
  assert.ok(auto.alerts.length >= 1);

  // 5. Boundary cases: too few samples, an empty window, a constant target, and an empty registry.
  const quiet = new Monitor({ windowSize: 10, minSamples: 5 });
  const dummy = { weights: [0, 0], bias: 0 };
  for (let k = 0; k < 4; k++) {
    assert.strictEqual(quiet.observe({ features: [0, 0], target: 0 }, dummy).status, "warming-up");
  }
  assert.strictEqual(quiet.alerts.length, 0, "an undersized window must never alert");
  const fresh = new AutoMlSystem({ dimension: 1 });
  assert.strictEqual(fresh.registry.length, 0);
  assert.strictEqual(fresh.current, null);
  const constantTrainer = new RidgeTrainer(1);
  const constantModel = constantTrainer.fit([{ features: [0], target: 7 }, { features: [1], target: 7 }], 1);
  assert.ok(Math.abs(constantModel.bias - 7) < 0.4, "a constant target is fit by the bias");
  assert.ok(Math.abs(constantModel.weights[0]) < 0.6, "a constant target leaves no slope");

  // 6. Registry: rollback restores the previous model exactly, and versions stay stable.
  const roll = new AutoMlSystem({ dimension: 2 });
  const v0 = roll.train(drifting.samples.slice(0, 40));
  const v1 = roll.train(drifting.samples.slice(40, 80));
  assert.strictEqual(roll.version, 1);
  assert.notDeepStrictEqual(v0.weights, v1.weights, "a retrain must change the weights");
  roll.rollback(0);
  assert.deepStrictEqual(roll.current, v0);
  const probe = drifting.samples[0];
  const predict = (model) => model.weights.reduce((sum, w, i) => sum + w * probe.features[i], model.bias);
  assert.strictEqual(predict(roll.current), predict(v0), "rollback must not alter the model");
  assert.throws(() => roll.rollback(7), /unknown model version/);
  assert.deepStrictEqual(roll.registry[1], v1, "the registry is append-only");

  // 7. Invalid input is rejected.
  assert.throws(() => new RidgeTrainer(0), /positive ridge/);
  assert.throws(() => new RidgeTrainer(1).fit([], 2), /empty/);
  assert.throws(() => new RidgeTrainer(1).fit([{ features: [1], target: 1 }], 2), /wrong dimension/);
  assert.throws(() => new Monitor({ windowSize: 0 }), /windowSize/);
  assert.throws(() => new Monitor({ minSamples: 0 }), /minSamples/);

  // 8. Determinism: the same seed rebuilds the same stream, alerts and model.
  const rerun = () => {
    const p = new Prng(11);
    const stream = makeStream(p, { count: 120 });
    const local = new AutoMlSystem({ dimension: 2, monitor: { windowSize: 15, minSamples: 8 } });
    local.train(stream.samples.slice(0, 60));
    for (const sample of stream.samples.slice(60)) local.monitor.observe(sample, local.current);
    return { alerts: local.alerts, weights: local.current.weights };
  };
  assert.deepStrictEqual(rerun(), rerun());

  console.log(`automl: alert after ${alertIndex - drifting.driftAt} drifting samples, ` +
    `error ${errorBeforeAlert.toFixed(3)} -> ${meanSquaredError(after, retrained).toFixed(3)}, ` +
    `versions ${auto.registry.length}, alerts ${auto.alerts.length}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, RidgeTrainer, makeStream, Monitor, AutoMlSystem, meanSquaredError, solveLinear };
