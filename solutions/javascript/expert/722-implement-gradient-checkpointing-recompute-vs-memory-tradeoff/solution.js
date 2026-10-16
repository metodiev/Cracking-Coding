/** Task 722 - Implement gradient checkpointing (recompute vs memory tradeoff). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Gradient checkpointing (Chen et al., "Training Deep Nets with Sublinear Memory Cost") trades
 * compute for memory: instead of keeping every forward activation alive until its layer's
 * backward, the forward pass stores only the input of every k-th layer (the *segment
 * boundaries*) and throws the rest away; during the backward pass the k activations of the
 * current segment are recomputed locally, used immediately and released again.
 *
 * Because the forward function is pure and deterministic, recomputing an activation returns
 * the very same float64 bits, so the gradients are unchanged -- this file proves that
 * claim by comparing them with deepStrictEqual against a store-everything reference, and
 * measures the peak memory of both schedules with a real allocator instead of trusting the
 * textbook formula:
 *
 *   live memory                     store-all          checkpoint(segment k)
 *   forward                         all L + 1 acts     boundaries (L/k + 1) + 1 transient
 *   backward                        free               boundaries + k recomputed acts
 *   extra compute                   0                  exactly one extra forward pass
 *
 * The peak of the checkpointed schedule is proportional to L/k + k, minimised near
 * k = sqrt(L): that is the sublinear O(sqrt(L)) memory result the demo below measures, and
 * the plan for a memory budget is chosen by an independent exhaustive scan over k.
 *
 * Verified: bit-exact gradients (store-all vs checkpointed, several k), equality with a
 * central-difference oracle on a small network, the measured store-all peak against the
 * analytic sum, the recompute count (one extra forward pass for every k), the sqrt-scaling
 * of the peak, the exhaustive-vs-bisection planner agreement under a memory budget,
 * infeasible budgets, degenerate networks (L = 0 and L = 1) and deterministic replay.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

/** Deterministic splitmix64 -> [0, 1). */
function makeRng(seed) {
  let state = (BigInt(seed >>> 0) + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
  return function next() {
    state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    z ^= z >> 31n;
    return Number(z >> 11n) / 9007199254740992;
  };
}

const BYTES_PER_ACTIVATION = 8;               // float64 activations

/** Allocator that records the high-water mark: the metric the whole file is about. */
class MemoryTracker {
  constructor() {
    this.live = new Map();
    this.bytes = 0;
    this.peak = 0;
    this.serial = 0;
  }

  alloc(size) {
    const name = `block${this.serial++}`;
    this.live.set(name, size);
    this.bytes += size;
    if (this.bytes > this.peak) this.peak = this.bytes;
    return name;
  }

  free(name) {
    const size = this.live.get(name);
    strictEqual(typeof size, "number", `unknown or double-freed block ${name}`);
    this.bytes -= size;
    this.live.delete(name);
  }

  get liveBlocks() {
    return this.live.size;
  }
}

/** Deterministic affine layer: a = tanh(W x + b). Each row is (weights, bias). */
function makeNetwork(rng, layers, width) {
  return Array.from({ length: layers }, () => ({
    weights: Array.from({ length: width }, () =>
      Array.from({ length: width }, () => (rng() * 2 - 1) * 0.5)),
    bias: Array.from({ length: width }, () => (rng() * 2 - 1) * 0.5),
  }));
}

function forwardLayer(layer, input) {
  return layer.bias.map((bias, row) => {
    let pre = bias;
    for (let column = 0; column < input.length; column++) pre += layer.weights[row][column] * input[column];
    return Math.tanh(pre);
  });
}

/** Loss = 0.5 ||a_L||^2, so dLoss/da_L = a_L. */
function lossOf(activation) {
  let loss = 0;
  for (const value of activation) loss += 0.5 * value * value;
  return loss;
}

/**
 * One backward step of a layer. dz = dL/da * (1 - a^2) for the tanh; the weight gradient is
 * the outer product of dz with the layer input, and the input gradient is W^T dz.
 */
function backwardLayer(layer, input, activation, upstream) {
  const width = activation.length;
  const dz = activation.map((a, row) => upstream[row] * (1 - a * a));
  const weightGrad = dz.map((value, row) => input.map((x) => value * x));
  const biasGrad = dz.slice();
  const inputGrad = input.map((_, column) => {
    let total = 0;
    for (let row = 0; row < width; row++) total += dz[row] * layer.weights[row][column];
    return total;
  });
  return { weightGrad, biasGrad, inputGrad };
}

/** Reference schedule: every activation stays live until the backward pass consumes it. */
function runStoreAll(network, input) {
  const tracker = new MemoryTracker();
  const handles = [tracker.alloc(input.length * BYTES_PER_ACTIVATION)];
  const activations = [input.slice()];
  for (let i = 0; i < network.length; i++) {
    const activation = forwardLayer(network[i], activations[i]);
    activations.push(activation);
    handles.push(tracker.alloc(activation.length * BYTES_PER_ACTIVATION));
  }
  const weightGrads = new Array(network.length);
  const biasGrads = new Array(network.length);
  let upstream = activations[network.length].slice();
  for (let i = network.length - 1; i >= 0; i--) {
    const step = backwardLayer(network[i], activations[i], activations[i + 1], upstream);
    weightGrads[i] = step.weightGrad;
    biasGrads[i] = step.biasGrad;
    upstream = step.inputGrad;
    tracker.free(handles[i + 1]);              // the activation is dead once its layer is done
  }
  tracker.free(handles[0]);
  return { weightGrads, biasGrads, inputGrad: upstream, peak: tracker.peak, recomputed: 0 };
}

/**
 * Checkpointed schedule: keep only segment boundaries during the forward pass, recompute the
 * k activations of a segment during its backward step, release them immediately afterwards.
 */
function runCheckpointed(network, input, segment) {
  const layers = network.length;
  const k = Math.max(1, Math.min(segment, Math.max(1, layers)));
  const tracker = new MemoryTracker();
  const stored = new Map();
  const inputHandle = tracker.alloc(input.length * BYTES_PER_ACTIVATION);
  stored.set(0, { values: input.slice(), handle: inputHandle });
  let current = input.slice();
  for (let i = 0; i < layers; i++) {
    const transient = tracker.alloc(current.length * BYTES_PER_ACTIVATION);
    const activation = forwardLayer(network[i], current);
    if ((i + 1) % k === 0 || i + 1 === layers) {
      stored.set(i + 1, { values: activation, handle: transient });   // a segment boundary
    } else {
      tracker.free(transient);
    }
    current = activation;
  }
  const weightGrads = new Array(layers);
  const biasGrads = new Array(layers);
  let upstream = stored.get(layers).values.slice();
  let recomputed = 0;
  const boundaries = [...stored.keys()].sort((a, b) => a - b);
  for (let b = boundaries.length - 1; b > 0; b--) {
    const segmentStart = boundaries[b - 1];
    const segmentEnd = boundaries[b];
    const segmentInput = stored.get(segmentStart).values;   // kept live from the forward pass
    // Recompute the chain of activations of this segment and keep them until its backward
    // walk releases them: that is the whole point of the trade.
    const handles = [tracker.alloc(segmentInput.length * BYTES_PER_ACTIVATION)];
    const chain = [segmentInput];
    for (let i = segmentStart; i < segmentEnd; i++) {
      const activation = forwardLayer(network[i], chain[chain.length - 1]);
      chain.push(activation);
      handles.push(tracker.alloc(activation.length * BYTES_PER_ACTIVATION));
      recomputed++;
    }
    for (let i = segmentEnd - 1; i >= segmentStart; i--) {
      const step = backwardLayer(network[i], chain[i - segmentStart], chain[i - segmentStart + 1], upstream);
      weightGrads[i] = step.weightGrad;
      biasGrads[i] = step.biasGrad;
      upstream = step.inputGrad;
      tracker.free(handles[i - segmentStart + 1]);
    }
    tracker.free(handles[0]);
  }
  for (const block of stored.values()) tracker.free(block.handle);
  return { weightGrads, biasGrads, inputGrad: upstream, peak: tracker.peak, recomputed };
}

const totalSize = (network, width) => (network.length + 1) * width * BYTES_PER_ACTIVATION;

/** Closed-form peak of the checkpointed schedule, used as the model for the budget planner. */
function modelPeak(layers, width, segment) {
  const k = Math.max(1, Math.min(segment, Math.max(1, layers)));
  const boundaries = Math.ceil(layers / k) + 1;
  return (boundaries + k + 1) * width * BYTES_PER_ACTIVATION;
}

/** Oracle planner: scan every feasible segment size. */
function bestSegmentForBudget(layers, width, budget) {
  for (let k = 1; k <= Math.max(1, layers); k++) {
    if (modelPeak(layers, width, k) <= budget) return k;
  }
  return null;
}

/**
 * Bisection planner. The model peak is V-shaped in k (it grows as L/k for small segments and
 * as k for large ones), so the search first locates the minimum on the decreasing branch and
 * then bisects there, which is where the smallest feasible segment size lives.
 */
function bisectSegmentForBudget(layers, width, budget) {
  const feasible = (k) => modelPeak(layers, width, k) <= budget;
  const span = Math.max(1, layers);
  let minAt = 1;
  for (let k = 2; k <= span; k++) if (modelPeak(layers, width, k) < modelPeak(layers, width, minAt)) minAt = k;
  if (!feasible(minAt)) return null;
  let low = 1;
  let high = minAt;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (feasible(mid)) high = mid; else low = mid + 1;
  }
  return low;
}

module.exports = {
  makeRng, MemoryTracker, makeNetwork, forwardLayer, lossOf, backwardLayer, runStoreAll,
  runCheckpointed, totalSize, modelPeak, bestSegmentForBudget, bisectSegmentForBudget,
};

if (require.main === module) {
  const rng = makeRng(20260722);
  const width = 4;
  const depth = 64;
  const network = makeNetwork(rng, depth, width);
  const input = Array.from({ length: width }, () => rng() * 2 - 1);

  // -- the recomputed gradients are bit-identical to the store-everything reference -----
  const reference = runStoreAll(network, input);
  strictEqual(reference.peak, totalSize(network, width),
    "storing everything keeps all L+1 activations live at the peak");
  strictEqual(reference.recomputed, 0, "the reference schedule never recomputes");
  for (const k of [1, 2, 4, 8, 16, 32, 64]) {
    const checkpointed = runCheckpointed(network, input, k);
    deepStrictEqual(checkpointed.inputGrad, reference.inputGrad,
      `segment ${k}: the input gradient must be bit-identical after recomputation`);
    deepStrictEqual(checkpointed.weightGrads, reference.weightGrads,
      `segment ${k}: weight gradients must be bit-identical`);
    deepStrictEqual(checkpointed.biasGrads, reference.biasGrads,
      `segment ${k}: bias gradients must be bit-identical`);
    strictEqual(checkpointed.recomputed, depth,
      `segment ${k}: every layer is recomputed exactly once (one extra forward pass)`);
  }
  console.log(`store-all peak=${reference.peak} B for ${depth} layers of ${width} float64 activations`);

  // -- independent oracle: central differences must agree with the analytic gradient -----
  const small = makeNetwork(rng, 3, 3);
  const smallInput = [0.3, -0.4, 0.7];
  const analytic = runStoreAll(small, smallInput).inputGrad;
  const lossAt = (x) => {
    let activation = x;
    for (const layer of small) activation = forwardLayer(layer, activation);
    return lossOf(activation);
  };
  const step = 1e-6;
  for (let i = 0; i < smallInput.length; i++) {
    const plus = smallInput.slice();
    const minus = smallInput.slice();
    plus[i] += step;
    minus[i] -= step;
    const numeric = (lossAt(plus) - lossAt(minus)) / (2 * step);
    ok(Math.abs(numeric - analytic[i]) < 1e-6 * Math.max(1, Math.abs(numeric)),
      `central differences disagree with the analytic input gradient at ${i}: ${numeric} vs ${analytic[i]}`);
  }
  const smallCheckpoint = runCheckpointed(small, smallInput, 2);
  deepStrictEqual(smallCheckpoint.inputGrad, analytic, "checkpointing a tiny network is exact too");

  // -- measured peak: sqrt scaling, and the optimum sits near sqrt(L) -------------------
  const measured = [];
  for (const k of [1, 2, 4, 8, 16, 32, 64]) {
    const run = runCheckpointed(network, input, k);
    measured.push({ k, peak: run.peak });
    strictEqual(run.peak, modelPeak(depth, width, k),
      `segment ${k}: the measured peak must match the model (boundaries + k + 1 blocks)`);
    ok(k > 1 && k < depth ? run.peak < reference.peak : run.peak >= reference.peak,
      `segment ${k}: only interior segments beat store-all (the extremes keep every activation)`);
  }
  const optimum = measured.reduce((best, entry) => (entry.peak < best.peak ? entry : best));
  ok(Math.abs(Math.log2(optimum.k) - Math.log2(Math.sqrt(depth))) <= 1,
    `the best measured segment (${optimum.k}) must be within a factor 2 of sqrt(L)=${Math.sqrt(depth)}`);
  const atOptimum = measured.find((entry) => entry.k === optimum.k);
  ok(reference.peak > atOptimum.peak * 3,
    `sqrt checkpointing must cut peak memory at least 3x (${reference.peak} vs ${atOptimum.peak})`);
  ok(atOptimum.peak < 4 * Math.sqrt(depth) * width * BYTES_PER_ACTIVATION,
    "the peak must stay in the O(sqrt(L)) regime");
  console.log(`peaks: ${measured.map((m) => `k=${m.k}:${m.peak}`).join(" ")}`);

  // -- the two planners agree, and infeasible budgets are reported as infeasible --------
  const minModelPeak = Array.from({ length: depth }, (_, i) => modelPeak(depth, width, i + 1))
    .reduce((a, b) => Math.min(a, b));
  for (const budget of [minModelPeak, minModelPeak + 64, modelPeak(depth, width, 16), 5000]) {
    const oracle = bestSegmentForBudget(depth, width, budget);
    strictEqual(bisectSegmentForBudget(depth, width, budget), oracle,
      `budget ${budget}: bisection must agree with the exhaustive scan`);
    ok(oracle !== null && modelPeak(depth, width, oracle) <= budget,
      `budget ${budget}: the chosen segment must fit the budget`);
  }
  strictEqual(bestSegmentForBudget(depth, width, minModelPeak - 1), null,
    "a budget below the achievable minimum is infeasible");
  strictEqual(bisectSegmentForBudget(depth, width, minModelPeak - 1), null,
    "the bisection planner must report the same infeasibility");
  strictEqual(bestSegmentForBudget(depth, width, 5000), 1,
    "a generous budget needs no checkpointing at all");
  const tightest = bestSegmentForBudget(depth, width, minModelPeak);
  ok(Math.abs(Math.log2(tightest) - Math.log2(Math.sqrt(depth))) <= 1,
    `the tightest feasible budget must pick a segment near sqrt(L), got ${tightest}`);

  // -- degenerate networks ---------------------------------------------------------------
  const flat = runStoreAll([], input);
  strictEqual(flat.peak, input.length * BYTES_PER_ACTIVATION, "no layers means one live activation");
  deepStrictEqual(flat.inputGrad, input, "with no layers the gradient is just a_L");
  const flatCheckpoint = runCheckpointed([], input, 8);
  strictEqual(flatCheckpoint.recomputed, 0, "nothing to recompute");
  deepStrictEqual(flatCheckpoint.inputGrad, flat.inputGrad);
  const single = makeNetwork(rng, 1, 3);
  const singleInput = [0.25, -0.5, 0.75];
  deepStrictEqual(runCheckpointed(single, singleInput, 99).inputGrad,
    runStoreAll(single, singleInput).inputGrad, "segment larger than the network is clamped");
  deepStrictEqual(runCheckpointed(single, singleInput, 0).inputGrad,
    runStoreAll(single, singleInput).inputGrad, "segment 0 clamps to 1");
  throws(() => new MemoryTracker().free("nope"), /double-freed|unknown/);

  // -- deterministic replay -------------------------------------------------------------
  const replay = () => {
    const run = runCheckpointed(network, input, 7);
    return { grad: run.inputGrad, peak: run.peak, recomputed: run.recomputed };
  };
  deepStrictEqual(replay(), replay(), "identical inputs must replay identically");
  console.log(`planner: budget ${modelPeak(depth, width, 8)} B -> segment ${bestSegmentForBudget(depth, width, modelPeak(depth, width, 8))}`);
  console.log("PASS");
}
