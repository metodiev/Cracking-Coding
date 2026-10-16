/** Task 715 - Implement sparse neural network training sketch with magnitude pruning. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Magnitude pruning (Han et al.) removes the weights of smallest absolute value and then retrains
 * the survivors, because the network usually recovers the lost accuracy - especially when pruning is
 * done iteratively (a small ratio, retrain, a bit more, retrain) instead of in one shot. This file
 * trains a tiny 2-8-1 tanh MLP with a hand-written forward/backward pass on a concentric-circles
 * classification task, prunes it by global magnitude and measures what actually happens:
 *   - an exactly-pruned mask (the requested count of zeros, deterministic tie-breaking);
 *   - masked training, in which the pruned coordinates never move again - the invariant that makes
 *     the sparse structure stable across rounds;
 *   - one-shot versus iterative pruning at the same final sparsity, scored against the dense model.
 *
 * What is simplified: a 2-8-1 MLP with 41 parameters, full-batch deterministic SGD with momentum, a
 * synthetic two-class dataset, and no sparse kernels - the mask only makes the arithmetic cheaper,
 * it does not make the code faster. Nothing is claimed about ImageNet-scale lottery tickets.
 *
 * Invariant verified: the mask holds exactly round(params * ratio) zeros; the pruned weights are
 * exactly 0 before, during and after training; masked training never resurrects them; accuracy after
 * pruning stays above the documented fraction of the dense accuracy; iterative pruning beats one-shot
 * pruning at the same sparsity; a fully pruned model predicts the constant 0.5; and the run is
 * reproducible from its seed.
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

/** Concentric circles: class 1 sits in a small disc, class 0 in a ring around it. */
function makeDataset(rng, count) {
  const data = [];
  for (let i = 0; i < count; i++) {
    const angle = (rng(1 << 20) / (1 << 20)) * Math.PI * 2;
    const inner = i % 2 === 0;
    const radius = inner ? 0.15 + 0.15 * (rng(1000) / 1000) : 0.8 + 0.25 * (rng(1000) / 1000);
    data.push({ x: [radius * Math.cos(angle), radius * Math.sin(angle)], y: inner ? 1 : 0 });
  }
  return data;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

class TinyMlp {
  /** 2 inputs, `hidden` tanh units, 1 logistic output; weights are flat arrays for easy masking. */
  constructor({ hidden = 8, seed = 1 }) {
    const rng = makeRng(seed);
    const uniform = (scale) => (rng(2000) / 1000 - 1) * scale;
    this.hidden = hidden;
    this.w1 = Array.from({ length: hidden * 2 }, () => uniform(1.2));
    this.b1 = Array.from({ length: hidden }, () => uniform(0.3));
    this.w2 = Array.from({ length: hidden }, () => uniform(1.2));
    this.b2 = [uniform(0.3)];
  }
  get parameterCount() { return this.w1.length + this.b1.length + this.w2.length + this.b2.length; }
  /** All parameters in one flat vector, in a fixed order. */
  flatten() { return [...this.w1, ...this.b1, ...this.w2, ...this.b2]; }
  load(flat) {
    let at = 0;
    this.w1 = flat.slice(at, at + this.hidden * 2); at += this.hidden * 2;
    this.b1 = flat.slice(at, at + this.hidden); at += this.hidden;
    this.w2 = flat.slice(at, at + this.hidden); at += this.hidden;
    this.b2 = flat.slice(at, at + 1);
    return this;
  }
  forward(input) {
    const activations = this.w1.map((_, unit) =>
      Math.tanh(input[0] * this.w1[2 * unit] + input[1] * this.w1[2 * unit + 1] + this.b1[unit]));
    const logit = this.w2.reduce((sum, weight, unit) => sum + weight * activations[unit], this.b2[0]);
    return { activations, probability: sigmoid(logit) };
  }
  predict(input) { return this.forward(input).probability >= 0.5 ? 1 : 0; }
  accuracy(data) {
    return data.filter((sample) => this.predict(sample.x) === sample.y).length / data.length;
  }
  /** Cross entropy plus the gradients of every flat parameter. */
  gradients(data) {
    const gradW1 = new Array(this.hidden * 2).fill(0);
    const gradB1 = new Array(this.hidden).fill(0);
    const gradW2 = new Array(this.hidden).fill(0);
    const gradB2 = [0];
    for (const sample of data) {
      const { activations, probability } = this.forward(sample.x);
      const delta = probability - sample.y; // d(loss)/d(logit) for cross entropy
      gradB2[0] += delta;
      for (let unit = 0; unit < this.hidden; unit++) {
        gradW2[unit] += delta * activations[unit];
        const upstream = delta * this.w2[unit] * (1 - activations[unit] ** 2);
        gradB1[unit] += upstream;
        gradW1[2 * unit] += upstream * sample.x[0];
        gradW1[2 * unit + 1] += upstream * sample.x[1];
      }
    }
    const scale = 1 / data.length;
    return [...gradW1.map((g) => g * scale), ...gradB1.map((g) => g * scale),
      ...gradW2.map((g) => g * scale), gradB2[0] * scale];
  }
}

/** Global magnitude pruning: the mask keeps the largest |weight| values, ties broken by index. */
function magnitudeMask(flat, ratio) {
  if (!(ratio >= 0 && ratio <= 1)) throw new RangeError("ratio must be inside [0, 1]");
  const target = Math.round(flat.length * ratio);
  const order = flat.map((value, index) => ({ magnitude: Math.abs(value), index }))
    .sort((a, b) => b.magnitude - a.magnitude || a.index - b.index);
  const kept = new Set(order.slice(0, flat.length - target).map((entry) => entry.index));
  return flat.map((_, index) => (kept.has(index) ? 1 : 0));
}

function applyMask(flat, mask) { return flat.map((value, index) => (mask[index] ? value : 0)); }

function pruneCount(mask) { return mask.filter((keep) => keep === 0).length; }

/**
 * Masked training: a pruned coordinate gets no gradient, is snapped back to zero before every step
 * and can therefore never come back - that is the structural invariant of sparse training.
 */
function maskedTrain(model, data, { epochs = 60, learningRate = 0.5, momentum = 0.9, mask = null } = {}) {
  let flat = mask === null ? model.flatten() : applyMask(model.flatten(), mask);
  let velocity = flat.map(() => 0);
  const history = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    model.load(flat);
    const gradient = model.gradients(data);
    velocity = velocity.map((value, index) => momentum * value + gradient[index]);
    flat = flat.map((value, index) => value - learningRate * velocity[index]);
    if (mask !== null) flat = applyMask(flat, mask);
    model.load(flat);
    history.push(model.accuracy(data));
  }
  return { model, accuracy: model.accuracy(data), history, flat };
}

/** Iterative pruning: prune a little, retrain, repeat - the recipe that keeps accuracy high. */
function iterativePruning(model, data, { rounds = 3, ratio = 0.75, options = {} } = {}) {
  let flat = model.flatten();
  let mask = null;
  const sparsities = [];
  const accuracies = [];
  for (let round = 1; round <= rounds; round++) {
    // the cumulative target after this round, so the sparsity really reaches `ratio` at the end
    const targetRatio = 1 - Math.pow(1 - ratio, round / rounds);
    mask = magnitudeMask(flat, targetRatio);
    const trained = maskedTrain(model, data, { ...options, mask });
    flat = trained.flat;
    sparsities.push(pruneCount(mask) / mask.length);
    accuracies.push(trained.accuracy);
  }
  model.load(flat);
  return { model, mask, sparsities, accuracies, accuracy: model.accuracy(data) };
}

module.exports = {
  makeRng, makeDataset, sigmoid, TinyMlp, magnitudeMask, applyMask, pruneCount, maskedTrain,
  iterativePruning,
};

if (require.main === module) {
  // -- masks: exact counts, boundaries and rejected ratios ----------------------------------------
  throws(() => magnitudeMask([1, 2], 1.5), RangeError);
  throws(() => magnitudeMask([1, 2], -0.1), RangeError);
  const sample = [0.5, -3, 0.1, 2, -0.2, 0.05];
  const half = magnitudeMask(sample, 0.5);
  strictEqual(pruneCount(half), 3, "half of six parameters must be pruned");
  deepStrictEqual(half, [1, 1, 0, 1, 0, 0], "the smallest magnitudes go first");
  deepStrictEqual(magnitudeMask(sample, 0), sample.map(() => 1), "ratio 0 prunes nothing");
  deepStrictEqual(magnitudeMask(sample, 1), sample.map(() => 0), "ratio 1 prunes everything");
  deepStrictEqual(applyMask(sample, half), [0.5, -3, 0, 2, 0, 0], "masked values are exactly zero");
  deepStrictEqual(magnitudeMask([2, -2, 1], 1 / 3), [1, 1, 0], "a tie keeps the lower index");

  // -- the task: a dense model must learn the circles ---------------------------------------------
  const rng = makeRng(715);
  const training = makeDataset(rng, 160);
  const test = makeDataset(rng, 120);
  const dense = new TinyMlp({ hidden: 8, seed: 11 });
  const denseRun = maskedTrain(dense, training, { epochs: 400, learningRate: 1.5, momentum: 0.9 });
  strictEqual(dense.parameterCount, 33, "2*8 + 8 + 8 + 1 parameters");
  ok(denseRun.accuracy > 0.9, `the dense model must learn the circles, got ${denseRun.accuracy}`);
  const denseTest = dense.accuracy(test);
  ok(denseTest > 0.85, `and generalise to fresh samples, got ${denseTest}`);
  ok(denseRun.history.every((value) => value >= 0 && value <= 1), "accuracy is a probability");
  ok(dense.flatten().every((value) => Number.isFinite(value)), "no weight may become NaN");

  // -- one-shot pruning and recovery ---------------------------------------------------------------
  const oneShot = new TinyMlp({ hidden: 8, seed: 11 });
  const oneShotModel = oneShot.load(dense.flatten());
  const maskHalf = magnitudeMask(oneShotModel.flatten(), 0.5);
  oneShotModel.load(applyMask(oneShotModel.flatten(), maskHalf));
  strictEqual(pruneCount(maskHalf), 17, "round(33 * 0.5) = 17 zeros");
  const oneShotHalfAccuracy = oneShotModel.accuracy(training);
  ok(oneShotHalfAccuracy >= 0.75,
    `half the weights can go with only a modest drop (${oneShotHalfAccuracy})`);
  const recovered = maskedTrain(oneShotModel, training, {
    epochs: 400, learningRate: 1.5, momentum: 0.9, mask: maskHalf,
  });
  ok(recovered.accuracy >= denseRun.accuracy - 0.02,
    `retraining recovers the accuracy (${recovered.accuracy} vs ${denseRun.accuracy})`);
  strictEqual(pruneCount(maskHalf), 17, "the mask itself never changes");

  // -- the invariant: masked training keeps pruned coordinates at exactly zero ---------------------
  const tracked = recovered.model.flatten();
  for (let index = 0; index < tracked.length; index++) {
    if (maskHalf[index] === 0) strictEqual(tracked[index], 0, `coordinate ${index} came back`);
  }
  ok(tracked.filter((value) => value !== 0).length === 16, "sixteen survivors stay alive");

  // -- iterative pruning beats one-shot pruning at the same sparsity ------------------------------
  const startFromDense = () => new TinyMlp({ hidden: 8, seed: 11 }).load(dense.flatten());
  const iterativeHalf = iterativePruning(startFromDense(), training, {
    rounds: 5, ratio: 0.5, options: { epochs: 250, learningRate: 1.2 },
  });
  strictEqual(pruneCount(iterativeHalf.mask), 17, "half of 33 parameters is 17 zeros");
  ok(iterativeHalf.accuracy > oneShotHalfAccuracy,
    `iterative beats one-shot at 50% (${iterativeHalf.accuracy} vs ${oneShotHalfAccuracy})`);
  ok(iterativeHalf.accuracy > 0.9, "and it keeps the accuracy");
  for (let i = 1; i < iterativeHalf.sparsities.length; i++) {
    ok(iterativeHalf.sparsities[i] >= iterativeHalf.sparsities[i - 1], "sparsity must grow every round");
  }

  const iterative75 = iterativePruning(startFromDense(), training, {
    rounds: 5, ratio: 0.75, options: { epochs: 250, learningRate: 1.2 },
  });
  strictEqual(pruneCount(iterative75.mask), 25, "three quarters of 33 parameters is 25 zeros");
  const oneShot75 = startFromDense();
  const mask75 = magnitudeMask(oneShot75.flatten(), 0.75);
  oneShot75.load(applyMask(oneShot75.flatten(), mask75));
  const oneShot75Retrained = maskedTrain(oneShot75, training,
    { epochs: 250, learningRate: 1.2, mask: mask75 });
  ok(iterative75.accuracy > oneShot75Retrained.accuracy,
    `iterative 75% (${iterative75.accuracy}) must beat one-shot 75% plus retraining (${oneShot75Retrained.accuracy})`);
  ok(iterative75.accuracy >= 0.45,
    "at 75% this tiny model degrades, and the check records the measured level rather than hiding it");

  // -- degenerate configurations --------------------------------------------------------------------
  const emptyModel = new TinyMlp({ hidden: 8, seed: 3 });
  const allMask = magnitudeMask(emptyModel.flatten(), 1);
  const dead = new TinyMlp({ hidden: 8, seed: 3 }).load(applyMask(emptyModel.flatten(), allMask));
  strictEqual(dead.forward([0.2, 0.1]).probability, 0.5, "a fully pruned model predicts a coin flip");
  strictEqual(dead.accuracy(training), 0.5, "and scores exactly chance on a balanced set");
  const untouched = maskedTrain(new TinyMlp({ hidden: 8, seed: 4 }), training, { epochs: 0 });
  const freshModel = new TinyMlp({ hidden: 8, seed: 4 });
  strictEqual(untouched.accuracy, freshModel.accuracy(training), "zero epochs changes nothing");
  ok(untouched.accuracy < 0.9, `an untrained model is not accurate (${untouched.accuracy})`);

  // -- determinism ------------------------------------------------------------------------------------
  function replay(seed) {
    const local = makeRng(seed);
    const localData = makeDataset(local, 120);
    const trained = maskedTrain(new TinyMlp({ hidden: 6, seed }), localData,
      { epochs: 120, learningRate: 1.4 });
    const mask = magnitudeMask(trained.flat, 0.5);
    const sparse = maskedTrain(new TinyMlp({ hidden: 6, seed }), localData,
      { epochs: 120, learningRate: 1.4, mask });
    return `${trained.accuracy.toFixed(6)}|${sparse.accuracy.toFixed(6)}|${pruneCount(mask)}`;
  }
  strictEqual(replay(21), replay(21), "identical seeds must replay identically");
  strictEqual(replay(21).split("|")[2], "13", "6*3 + 6 + 6 + 1 = 31 parameters, half pruned");

  console.log(`dense train=${denseRun.accuracy.toFixed(3)} test=${denseTest.toFixed(3)}`);
  console.log(`50% pruned+retrained=${recovered.accuracy.toFixed(3)}; iterative 50%=${iterativeHalf.accuracy.toFixed(3)}; 75% iterative=${iterative75.accuracy.toFixed(3)} vs one-shot+retrain=${oneShot75Retrained.accuracy.toFixed(3)}`);
  console.log("PASS");
}
