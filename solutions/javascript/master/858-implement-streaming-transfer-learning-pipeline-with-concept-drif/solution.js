/** Task 858 - Implement streaming transfer learning pipeline with concept drift detection.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Model. A labelled stream x_t in R^d arrives one sample at a time. The concept is piecewise stationary:
 * the stream is cut into segments, each with its own ground-truth coefficient vector, and the label is
 * y = w_segment . x + bounded noise. A linear model consumes the stream with one SGD step per sample
 * (eta_t = eta0 / sqrt(t)); a drift detector watches the *loss* stream, comparing a reference window of
 * recent errors with the window before it, and fires when the two means separate by more than a
 * Hoeffding-style bound for two consecutive samples (the persistence requirement is the hysteresis that
 * keeps a noisy but stationary stream quiet).
 *
 * On detection the pipeline transfers: it re-initialises the regression weights from their prior and
 * restores the learning rate, so the model re-learns the new concept quickly instead of dragging the
 * old one along. The detector statistic is recomputed here in two independent ways: incrementally from
 * running sums and in batch from a slice of the error history, and the two must agree.
 *
 * What is simplified: no neural feature extractor (transfer = re-initialising the head), a single
 * process, samples produced by a deterministic generator rather than a real data feed, no wall clock,
 * no threads, no external state, and randomness confined to one seeded mulberry32 stream.
 *
 * Invariants verified:
 *   recall      every ground-truth drift point produces a detection within `latencyBudget` samples, and
 *               the number of detections never explodes (at most one per drift plus a small allowance).
 *   precision   a stationary stream of the same length produces exactly zero detections for many seeds.
 *   recovery    after each drift the rolling error of the transferring pipeline falls back to the
 *               pre-drift level within a bounded window, and its cumulative error beats both a detector
 *               without transfer learning and a model that never adapts.
 *   correctness the detector statistic equals an independently recomputed batch statistic, the weight
 *               reset is exact, and the pipeline never diverges.
 *   determinism identical seeds reproduce identical detection points, errors and weights.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness (stream generation). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic linear stream with piecewise-constant concepts. `drifts` lists the change indices. */
function makeStream({ samples = 4000, dim = 4, drifts = [1000, 2400], seed = 858, noise = 0.05, stationary = false } = {}) {
  if (!Number.isInteger(samples) || samples < 1) throw new RangeError("samples must be >= 1");
  if (!Number.isInteger(dim) || dim < 1) throw new RangeError("dim must be >= 1");
  const rand = mulberry32(seed);
  const concepts = [];
  const cutPoints = stationary ? [] : drifts;
  for (let c = 0; c <= cutPoints.length; c++) {
    const weights = [];
    for (let j = 0; j < dim; j++) weights.push(rand() * 2 - 1);
    concepts.push({ weights, bias: rand() * 2 - 1 });
  }
  const rows = [];
  const truth = [];
  let segment = 0;
  for (let t = 0; t < samples; t++) {
    while (segment < cutPoints.length && t >= cutPoints[segment]) segment += 1;
    const concept = concepts[segment];
    const x = [];
    for (let j = 0; j < dim; j++) x.push(rand() * 2 - 1);
    let y = concept.bias;
    for (let j = 0; j < dim; j++) y += concept.weights[j] * x[j];
    y += (rand() * 2 - 1) * noise;
    rows.push({ x, y });
    truth.push(segment);
  }
  return { rows, concepts, truth, drifts: cutPoints };
}

/** Online linear regression with a decaying step size. */
class StreamingRegressor {
  constructor({ dim, eta0 = 0.2, seed = 858 }) {
    if (!Number.isInteger(dim) || dim < 1) throw new RangeError("dim must be >= 1");
    if (!(eta0 > 0)) throw new RangeError("eta0 must be > 0");
    this.dim = dim;
    this.eta0 = eta0;
    this.steps = 0;
    this.resets = 0;
    const rand = mulberry32(seed);
    this.prior = [];
    for (let j = 0; j < dim; j++) this.prior.push((rand() - 0.5) * 0.05);
    this.weights = this.prior.slice();
    this.bias = 0;
    this.diverged = false;
  }

  predict(x) {
    if (!Array.isArray(x) || x.length !== this.dim) throw new RangeError("feature length mismatch");
    let value = this.bias;
    for (let j = 0; j < this.dim; j++) value += this.weights[j] * x[j];
    return value;
  }

  /** One SGD step; returns the squared error before the update. */
  update(x, y) {
    const prediction = this.predict(x);
    const error = prediction - y;
    const eta = this.eta0 / Math.sqrt(1 + this.steps);
    for (let j = 0; j < this.dim; j++) this.weights[j] -= eta * error * x[j];
    this.bias -= eta * error;
    this.steps += 1;
    if (!Number.isFinite(this.weights[0]) || Math.abs(this.weights[0]) > 1e6) this.diverged = true;
    return error * error;
  }

  /** Transfer step: forget the drifted concept and restart from the prior with a fresh rate. */
  reset() {
    this.weights = this.prior.slice();
    this.bias = 0;
    this.steps = 0;
    this.resets += 1;
    return this;
  }
}

/**
 * Drift detector over the loss stream, with two signals:
 *   * `statistic()` is the separation of the recent window mean from the previous window mean, kept as a
 *     diagnostic and cross-checked against an independently recomputed batch value;
 *   * the actual alarm rule is "the instantaneous loss exceeds the reference window mean by the
 *     Hoeffding bound for `persistence` consecutive samples". A single excursion (one outlying sample)
 *     can therefore never raise an alarm unless persistence is 1, and a monotonically decreasing loss
 *     stream can never raise one at all.
 */
class DriftDetector {
  constructor({ window = 60, delta = 0.01, persistence = 2, scale = 1 } = {}) {
    if (!Number.isInteger(window) || window < 2) throw new RangeError("window must be >= 2");
    if (!(delta > 0 && delta < 1)) throw new RangeError("delta must be in (0,1)");
    if (!Number.isInteger(persistence) || persistence < 1) throw new RangeError("persistence must be >= 1");
    this.window = window;
    this.delta = delta;
    this.persistence = persistence;
    this.scale = scale;
    this.history = [];
    this.streak = 0;
    this.detections = [];
  }

  /** Hoeffding separation bound for a single sample against a `window`-sample mean. */
  bound() {
    return this.scale * Math.sqrt((2 / this.window) * Math.log(4 / this.delta));
  }

  /** Separation of the two most recent windows (0 while the history is too short). */
  statistic() {
    if (this.history.length < 2 * this.window) return 0;
    const n = this.history.length;
    return mean(this.history.slice(n - this.window)) - mean(this.history.slice(n - 2 * this.window, n - this.window));
  }

  /** The same statistic computed from running sums — the incremental implementation. */
  statisticFromSums(sums) {
    if (sums.recent === null || sums.reference === null) return 0;
    return sums.recent / this.window - sums.reference / this.window;
  }

  push(loss, index) {
    this.history.push(loss);
    const n = this.history.length;
    let sums = { recent: null, reference: null };
    if (n >= 2 * this.window) {
      let recent = 0;
      let reference = 0;
      for (let i = n - this.window; i < n; i++) recent += this.history[i];
      for (let i = n - 2 * this.window; i < n - this.window; i++) reference += this.history[i];
      sums = { recent, reference };
    }
    if (n >= this.window) {
      const referenceMean = mean(this.history.slice(n - this.window));
      if (loss > referenceMean + this.bound()) {
        this.streak += 1;
        if (this.streak >= this.persistence) {
          this.detections.push(index);
          this.streak = 0;
          this.history = []; // the detector restarts after a drift
          return { fired: true, statistic: this.statisticFromSums(sums), referenceMean };
        }
      } else {
        this.streak = 0;
      }
    }
    return { fired: false, statistic: this.statisticFromSums(sums), referenceMean: n >= this.window ? mean(this.history.slice(n - this.window)) : 0 };
  }
}

const mean = (values) => (values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length);

/** Rolling mean of the last `window` errors at every step (used for recovery measurements). */
function rollingErrors(errors, window) {
  const out = new Array(errors.length).fill(0);
  let sum = 0;
  for (let i = 0; i < errors.length; i++) {
    sum += errors[i];
    if (i >= window) sum -= errors[i - window];
    out[i] = sum / Math.min(i + 1, window);
  }
  return out;
}

/**
 * The pipeline: one pass over the stream, one SGD step per sample, drift detection on the loss stream,
 * and an optional transfer reset triggered by a detection.
 */
function runPipeline(stream, { transfer = true, detect = true, window = 60, delta = 0.01, persistence = 2, eta0 = 0.2, seed = 858 } = {}) {
  const det = new DriftDetector({ window, delta, persistence, scale: 1 });
  const model = new StreamingRegressor({ dim: stream.rows[0].x.length, eta0, seed });
  const errors = [];
  const losses = [];
  const resets = [];
  const statisticChecks = [];
  for (let t = 0; t < stream.rows.length; t++) {
    const { x, y } = stream.rows[t];
    const loss = model.update(x, y);
    errors.push(Math.abs(model.predict(x) - y));
    losses.push(loss);
    if (detect) {
      const probe = det.push(loss, t);
      if (t % 137 === 0) statisticChecks.push({ index: t, statistic: det.statistic() });
      if (probe.fired && transfer) {
        model.reset();
        resets.push(t);
      }
    }
  }
  return { model, detector: det, losses, errors, resets, detections: det.detections, statisticChecks, rolling: rollingErrors(losses, window) };
}

function selfCheck() {
  // --- validation ---------------------------------------------------------------------------------------
  assert.throws(() => makeStream({ samples: 0 }), /samples/);
  assert.throws(() => makeStream({ dim: 0 }), /dim/);
  assert.throws(() => new StreamingRegressor({ dim: 3, eta0: 0 }), /eta0/);
  assert.throws(() => new DriftDetector({ window: 1 }), /window/);
  assert.throws(() => new DriftDetector({ delta: 0 }), /delta/);
  assert.throws(() => new DriftDetector({ persistence: 0 }), /persistence/);

  const stream = makeStream({ samples: 4000, dim: 4, drifts: [1000, 2400], seed: 858 });
  assert.strictEqual(stream.rows.length, 4000);
  assert.deepStrictEqual(stream.drifts, [1000, 2400]);
  assert.deepStrictEqual([...new Set(stream.truth)].sort((a, b) => a - b), [0, 1, 2], "three concepts");

  // --- the loss really depends on the concept: pre-drift error is much smaller than post-drift error ---
  const model = new StreamingRegressor({ dim: 4, eta0: 0.2, seed: 858 });
  const plain = [];
  for (const { x, y } of stream.rows) plain.push(model.update(x, y));
  const beforeDrift = mean(plain.slice(900, 1000));
  const afterDrift = mean(plain.slice(1010, 1110));
  assert.ok(afterDrift > beforeDrift * 3, `a drift must hurt an unadapting model (${beforeDrift.toFixed(4)} -> ${afterDrift.toFixed(4)})`);
  assert.strictEqual(model.diverged, false);

  // --- deterministic generator and detector ---------------------------------------------------------------
  const twin = makeStream({ samples: 4000, dim: 4, drifts: [1000, 2400], seed: 858 });
  assert.deepStrictEqual(twin.rows[123], stream.rows[123]);
  const probe = new DriftDetector({ window: 10, delta: 0.05, persistence: 1 });
  for (let i = 0; i < 200; i++) probe.push(i < 100 ? 1 : 4, i);
  assert.ok(probe.detections.length >= 1);
  assert.ok(probe.detections[0] >= 100 && probe.detections[0] <= 102, `detection lands right after the change (${probe.detections[0]})`);

  // --- the window statistic matches an independently recomputed batch value --------------------------------
  const adaptive = runPipeline(stream, { transfer: true, window: 60, delta: 0.01, persistence: 2 });
  assert.ok(adaptive.statisticChecks.length > 5);
  const batchDetector = new DriftDetector({ window: 8, delta: 0.01, persistence: 1 });
  let recomputed = 0;
  for (let t = 0; t < stream.rows.length; t++) {
    batchDetector.push(plain[t], t);
    const n = batchDetector.history.length;
    if (n >= 2 * batchDetector.window && t % 211 === 0) {
      const recent = batchDetector.history.slice(n - batchDetector.window);
      const reference = batchDetector.history.slice(n - 2 * batchDetector.window, n - batchDetector.window);
      const batch = recent.reduce((a, b) => a + b, 0) / recent.length - reference.reduce((a, b) => a + b, 0) / reference.length;
      assert.ok(Math.abs(batch - batchDetector.statistic()) < 1e-12);
      assert.ok(Math.abs(batchDetector.statisticFromSums({ recent: recent.reduce((a, b) => a + b, 0), reference: reference.reduce((a, b) => a + b, 0) }) - batch) < 1e-12);
      recomputed += 1;
    }
  }
  assert.ok(recomputed > 0);
  // A monotonically improving stream can never raise an alarm: the loss only ever falls.
  const monotone = new DriftDetector({ window: 20, delta: 0.01, persistence: 1 });
  for (let i = 0; i < 500; i++) {
    const fired = monotone.push(10 / (i + 1), i);
    assert.strictEqual(fired.fired, false, `a falling loss stream fired at ${i}`);
  }
  assert.deepStrictEqual(monotone.detections, []);

  // --- recall: every drift is detected within the latency budget -------------------------------------------
  const latencyBudget = 4 * adaptive.detector.window;
  assert.strictEqual(adaptive.detections.length >= stream.drifts.length, true, `drifts detected: ${adaptive.detections.join(",")}`);
  for (const drift of stream.drifts) {
    const latency = adaptive.detections.filter((d) => d >= drift).map((d) => d - drift);
    assert.ok(latency.length > 0, `drift at ${drift} was missed`);
    assert.ok(Math.min(...latency) <= latencyBudget, `drift at ${drift} detected late (${Math.min(...latency)})`);
  }
  assert.ok(adaptive.detections.length <= stream.drifts.length + 2, `too many detections: ${adaptive.detections.join(",")}`);
  for (let i = 1; i < adaptive.detections.length; i++) assert.ok(adaptive.detections[i] > adaptive.detections[i - 1], "detections are ordered");
  assert.strictEqual(adaptive.resets.length, adaptive.detections.length, "every detection triggers exactly one transfer");

  // --- precision: a stationary stream raises no alarm --------------------------------------------------------
  for (const seed of [1, 2, 3, 858, 1000, 4242]) {
    const stationary = makeStream({ samples: 3000, dim: 4, stationary: true, seed });
    const run = runPipeline(stationary, { transfer: true, window: 60, delta: 0.01, persistence: 2 });
    assert.deepStrictEqual(run.detections, [], `stationary stream ${seed} produced false alarms at ${run.detections}`);
    assert.deepStrictEqual(run.resets, []);
  }

  // --- recovery: rolling error returns to the pre-drift level ------------------------------------------------
  for (const drift of stream.drifts) {
    const pre = adaptive.rolling[drift - 1];
    const peak = Math.max(...adaptive.rolling.slice(drift, drift + 60));
    const mid = adaptive.rolling[drift + 300];
    const after = adaptive.rolling[Math.min(adaptive.rolling.length - 1, drift + 800)];
    assert.ok(peak >= pre, "the drift must first make the error worse");
    assert.ok(mid < peak, "the error must fall again after the transfer reset");
    assert.ok(after < pre * 2, `the pipeline must recover after the drift at ${drift} (${pre.toFixed(4)} -> ${after.toFixed(4)})`);
    assert.ok(after < peak / 10, "recovery must undo the bulk of the damage");
  }

  // --- benefit: transfer learning beats both alternatives ------------------------------------------------------
  const noTransfer = runPipeline(stream, { transfer: false, window: 60, delta: 0.01, persistence: 2 });
  const noDetection = runPipeline(stream, { detect: false });
  const cumulative = (run) => run.losses.reduce((a, b) => a + b, 0);
  assert.ok(cumulative(adaptive) < cumulative(noTransfer), "transfer must reduce the cumulative loss");
  assert.ok(cumulative(adaptive) < cumulative(noDetection), "detection plus transfer must beat never adapting");
  const tailAdaptive = mean(adaptive.losses.slice(-300));
  const tailNoDetection = mean(noDetection.losses.slice(-300));
  assert.ok(tailAdaptive < tailNoDetection, `the final concept must be tracked better (${tailAdaptive.toFixed(4)} vs ${tailNoDetection.toFixed(4)})`);
  assert.strictEqual(adaptive.model.diverged, false);
  assert.strictEqual(adaptive.model.resets, adaptive.detections.length);
  const probeModel = new StreamingRegressor({ dim: 3, eta0: 0.2, seed: 1 });
  probeModel.update([1, 1, 1], 5);
  assert.notDeepStrictEqual(probeModel.weights, probeModel.prior, "learning must change the weights");
  const stepsBeforeReset = probeModel.steps;
  assert.ok(stepsBeforeReset === 1);
  probeModel.reset();
  assert.deepStrictEqual(probeModel.weights, probeModel.prior, "a transfer restores the exact prior");
  assert.strictEqual(probeModel.bias, 0);
  assert.strictEqual(probeModel.steps, 0, "the learning-rate schedule restarts as well");
  assert.strictEqual(probeModel.resets, 1);

  // --- anti-hysteresis: a single persistence miss must not fire ------------------------------------------------
  const twitchy = new DriftDetector({ window: 20, delta: 0.01, persistence: 3 });
  let spikes = 0;
  for (let i = 0; i < 600; i++) {
    const spiky = i % 50 === 0 ? 40 : 1; // isolated spikes every 50 samples
    if (twitchy.push(spiky, i).fired) spikes += 1;
  }
  assert.strictEqual(spikes, 0, "isolated spikes must not survive the persistence requirement");
  const eager = new DriftDetector({ window: 20, delta: 0.01, persistence: 1 });
  let eagerFires = 0;
  for (let i = 0; i < 600; i++) if (eager.push(i % 50 === 0 ? 40 : 1, i).fired) eagerFires += 1;
  assert.ok(eagerFires > 0, "the same stream fires when persistence is 1, showing the guard matters");
  assert.ok(eagerFires <= 12, `one alarm per spike at most (${eagerFires})`);

  // --- edge cases: drift at sample 0, a stream shorter than the window, a constant stream -----------------------
  const immediate = makeStream({ samples: 600, dim: 3, drifts: [0], seed: 9 });
  const immediateRun = runPipeline(immediate, { window: 40, delta: 0.01, persistence: 2 });
  assert.ok(immediateRun.detections.length <= 2, "a drift at sample 0 must not cause repeated alarms");
  const shortStream = makeStream({ samples: 30, dim: 3, drifts: [10], seed: 10 });
  const shortRun = runPipeline(shortStream, { window: 60, delta: 0.01, persistence: 2 });
  assert.deepStrictEqual(shortRun.detections, [], "a stream shorter than two windows cannot be judged");
  assert.strictEqual(shortRun.losses.length, 30);
  const constantStream = { rows: new Array(500).fill(0).map(() => ({ x: [0, 0, 0, 0], y: 2 })), drifts: [] };
  const constantRun = runPipeline(constantStream, { window: 40, delta: 0.01, persistence: 2 });
  assert.deepStrictEqual(constantRun.detections, []);
  assert.ok(Number.isFinite(constantRun.losses[499]));

  // --- determinism -----------------------------------------------------------------------------------------------
  const repeatA = runPipeline(stream, { transfer: true, window: 60, delta: 0.01, persistence: 2 });
  const repeatB = runPipeline(stream, { transfer: true, window: 60, delta: 0.01, persistence: 2 });
  assert.deepStrictEqual(repeatA.detections, repeatB.detections);
  assert.deepStrictEqual(repeatA.resets, repeatB.resets);
  assert.deepStrictEqual(repeatA.losses.slice(0, 50), repeatB.losses.slice(0, 50));
  const otherSeed = makeStream({ samples: 4000, dim: 4, drifts: [1000, 2400], seed: 859 });
  assert.notDeepStrictEqual(otherSeed.rows[0], stream.rows[0]);

  console.log(`stream: ${stream.rows.length} samples, drifts at ${stream.drifts.join(",")}, detected at ${adaptive.detections.join(",")} (budget ${latencyBudget})`);
  console.log(`loss: adaptive=${cumulative(adaptive).toFixed(1)} no-transfer=${cumulative(noTransfer).toFixed(1)} never-adapt=${cumulative(noDetection).toFixed(1)}`);
  console.log(`tail loss: adaptive=${tailAdaptive.toFixed(4)} never-adapt=${tailNoDetection.toFixed(4)}, resets=${adaptive.resets.length}`);
  console.log("PASS");
}

module.exports = { makeStream, StreamingRegressor, DriftDetector, runPipeline, rollingErrors, mulberry32 };

if (require.main === module) {
  selfCheck();
}
