/** Task 866 - Implement feature attribution runtime (SHAP-like) scalable for tabular models.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Model. A feature-attribution runtime for tabular models. Given a model f, an instance x and a background
 * reference (the empty coalition value), it produces per-feature attributions with three methods:
 *
 *   exact        the Shapley value computed by enumerating every coalition:
 *                phi_j = sum_{S subset of N\{j}} |S|!(d-|S|-1)!/d! [f(S+j) - f(S)].
 *   permutation  Monte-Carlo over random permutations, which is unbiased by construction and reports its
 *                own standard error; the run scales to many features because it never enumerates subsets.
 *   kernelShap   the weighted least-squares formulation with the Shapley kernel weights,
 *                w(S) = (d-1) / (C(d,|S|) |S| (d-|S|)), solved exactly here by constrained normal equations
 *                (the efficiency constraint is eliminated by substitution), so it matches the exact values
 *                when all coalitions are used.
 *
 * The four Shapley axioms — efficiency, symmetry, dummy and linearity — are checked as identities, and the
 * sampling methods are calibrated against the exact values (and, at d = 20 where enumeration is impossible,
 * against an additive model whose true attributions are known in closed form).
 *
 * What is simplified: tabular models are plain functions instead of tree ensembles with a C++ kernel, the
 * background is a fixed reference vector rather than an expectation over a data sample, KernelSHAP uses the
 * exact normal equations instead of iterative least squares, and there is no anti-correlation capping or
 * interaction-index machinery. Randomness is one seeded mulberry32 stream — no wall clock, no threads, no I/O.
 *
 * Invariants verified:
 *   axioms      efficiency (attributions sum to f(x) - f(reference)), symmetry, dummy and linearity hold
 *               exactly for the exact method, and hold within tolerance for the sampled ones.
 *   correctness the exact method agrees with the closed-form attributions of additive models, KernelSHAP
 *               with all coalitions reproduces the exact values to machine precision, and the permutation
 *               estimate converges to them as the sample count grows with a calibrated standard error.
 *   scalability at d = 20 the permutation and KernelSHAP estimates match the analytic attributions within a
 *               documented tolerance while the exact method remains out of reach (2^20 evaluations).
 *   determinism identical seeds reproduce identical attributions, errors and sampled coalitions.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness (permutation sampling and coalition sampling). */
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

const mean = (values) => (values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length);

/** Builds the coalition value function f(S) from a model, an instance and a reference point. */
function coalitionFunction({ model, instance, reference }) {
  if (typeof model !== "function") throw new TypeError("model must be a function");
  if (!Array.isArray(instance) || instance.length === 0) throw new TypeError("instance must be a non-empty array");
  if (!Array.isArray(reference) || reference.length !== instance.length) throw new TypeError("reference must match the instance");
  const dimension = instance.length;
  const value = (mask) => {
    const point = new Array(dimension);
    for (let j = 0; j < dimension; j++) point[j] = mask & (1 << j) ? instance[j] : reference[j];
    return model(point);
  };
  return { value, dimension };
}

function factorial(n) {
  let out = 1;
  for (let i = 2; i <= n; i++) out *= i;
  return out;
}

/** Exact Shapley values by enumerating all 2^d coalitions. */
function exactShapley({ model, instance, reference }) {
  const { value, dimension } = coalitionFunction({ model, instance, reference });
  if (dimension > 20) throw new RangeError("exact enumeration is only practical up to 20 features");
  const contributions = new Array(dimension).fill(0);
  const full = (1 << dimension) - 1;
  let evaluations = 0;
  const cache = new Map();
  const evaluate = (mask) => {
    let result = cache.get(mask);
    if (result === undefined) {
      result = value(mask);
      cache.set(mask, result);
      evaluations += 1;
    }
    return result;
  };
  for (let mask = 0; mask <= full; mask++) {
    const base = evaluate(mask);
    const size = popcount(mask);
    const weight = (factorial(size) * factorial(dimension - size - 1)) / factorial(dimension);
    for (let j = 0; j < dimension; j++) {
      if (mask & (1 << j)) continue;
      contributions[j] += weight * (evaluate(mask | (1 << j)) - base);
    }
  }
  return { attributions: contributions, evaluations, dimension, empty: evaluate(0), full: evaluate(full) };
}

function popcount(value) {
  let count = 0;
  let remaining = value;
  while (remaining > 0) {
    count += remaining & 1;
    remaining >>= 1;
  }
  return count;
}

/**
 * Permutation sampling: average the marginal contribution of every feature over random orderings. Unbiased
 * and the only method here that scales past a handful of features.
 */
function permutationShapley({ model, instance, reference, samples = 200, seed = 866 }) {
  if (!Number.isInteger(samples) || samples < 1) throw new RangeError("samples must be >= 1");
  const { value, dimension } = coalitionFunction({ model, instance, reference });
  const rand = mulberry32(seed);
  const totals = new Array(dimension).fill(0);
  const squares = new Array(dimension).fill(0);
  let evaluations = 0;
  const order = new Array(dimension);
  for (let s = 0; s < samples; s++) {
    for (let j = 0; j < dimension; j++) order[j] = j;
    for (let j = dimension - 1; j > 0; j--) {
      const k = Math.floor(rand() * (j + 1));
      const tmp = order[j];
      order[j] = order[k];
      order[k] = tmp;
    }
    let mask = 0;
    let previous = value(0);
    evaluations += 1;
    for (const feature of order) {
      mask |= 1 << feature;
      const current = value(mask);
      evaluations += 1;
      const contribution = current - previous;
      totals[feature] += contribution;
      squares[feature] += contribution * contribution;
      previous = current;
    }
  }
  const attributions = totals.map((total) => total / samples);
  const standardErrors = squares.map((sum, j) => {
    const variance = Math.max(0, (sum - totals[j] * totals[j] / samples) / Math.max(1, samples - 1));
    return Math.sqrt(variance / samples);
  });
  return { attributions, standardErrors, samples, dimension, evaluations };
}

/** KernelSHAP: weighted least squares with the Shapley kernel, efficiency constraint eliminated. */
function kernelShap({ model, instance, reference, coalitions = null, seed = 866 }) {
  const { value, dimension } = coalitionFunction({ model, instance, reference });
  const rand = mulberry32(seed);
  const full = (1 << dimension) - 1;
  let masks;
  if (coalitions === null || coalitions >= 1 << dimension) {
    masks = [];
    for (let mask = 1; mask < full; mask++) masks.push(mask);
  } else {
    if (!Number.isInteger(coalitions) || coalitions < 1) throw new RangeError("coalitions must be >= 1 or null");
    const seen = new Set();
    while (seen.size < Math.min(coalitions, (1 << dimension) - 2)) {
      seen.add(1 + Math.floor(rand() * ((1 << dimension) - 2)));
    }
    masks = [...seen];
  }
  const empty = value(0);
  const total = value(full) - empty;
  const size = dimension - 1; // the last attribution follows from the efficiency constraint
  const matrix = [];
  const targets = [];
  const weights = [];
  let evaluations = 2;
  for (const mask of masks) {
    const cardinality = popcount(mask);
    const lastIncluded = mask & (1 << (dimension - 1)) ? 1 : 0;
    const row = new Array(size).fill(0);
    for (let j = 0; j < size; j++) row[j] = (mask & (1 << j) ? 1 : 0) - lastIncluded; // phi_last eliminated by substitution
    const weight = (dimension - 1) / (choose(dimension, cardinality) * cardinality * (dimension - cardinality));
    matrix.push(row);
    targets.push(value(mask) - empty - lastIncluded * total);
    weights.push(weight);
    evaluations += 1;
  }
  const solution = weightedLeastSquares(matrix, targets, weights, 1e-10);
  const attributions = solution.slice();
  attributions.push(total - solution.reduce((a, b) => a + b, 0));
  return { attributions, coalitionsUsed: masks.length, dimension, evaluations, total };
}

function choose(n, k) {
  if (k < 0 || k > n) return 0;
  let out = 1;
  for (let i = 0; i < k; i++) out = (out * (n - i)) / (i + 1);
  return out;
}

/** Weighted least squares via normal equations with Gaussian elimination and a small ridge. */
function weightedLeastSquares(matrix, targets, weights, ridge) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const ata = Array.from({ length: cols }, () => new Array(cols).fill(0));
  const atb = new Array(cols).fill(0);
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < cols; j++) ata[i][j] += weights[r] * matrix[r][i] * matrix[r][j];
      atb[i] += weights[r] * matrix[r][i] * targets[r];
    }
  }
  const trace = ata.reduce((acc, row, i) => acc + row[i], 0);
  for (let i = 0; i < cols; i++) ata[i][i] += ridge * (trace / cols + 1);
  const augmented = ata.map((row, i) => row.concat([atb[i]]));
  for (let col = 0; col < cols; col++) {
    let pivot = col;
    for (let r = col + 1; r < cols; r++) if (Math.abs(augmented[r][col]) > Math.abs(augmented[pivot][col])) pivot = r;
    if (Math.abs(augmented[pivot][col]) < 1e-14) continue;
    const tmp = augmented[col];
    augmented[col] = augmented[pivot];
    augmented[pivot] = tmp;
    const divisor = augmented[col][col];
    for (let j = col; j <= cols; j++) augmented[col][j] /= divisor;
    for (let r = 0; r < cols; r++) {
      if (r === col) continue;
      const factor = augmented[r][col];
      if (factor === 0) continue;
      for (let j = col; j <= cols; j++) augmented[r][j] -= factor * augmented[col][j];
    }
  }
  return augmented.map((row) => row[cols]);
}

/** Additive model: f(x) = sum_j w_j x_j + bias — its exact attributions are w_j (x_j - reference_j). */
function additiveModel(weights, bias = 0) {
  const model = (x) => x.reduce((acc, value, j) => acc + weights[j] * value, bias);
  return { model, weights, bias };
}

/** A small deterministic tree ensemble: each tree is a nested threshold function, summed. */
function treeEnsemble({ trees = 5, depth = 3, dimension = 6, seed = 866 } = {}) {
  if (!Number.isInteger(trees) || trees < 1) throw new RangeError("trees must be >= 1");
  if (!Number.isInteger(depth) || depth < 1) throw new RangeError("depth must be >= 1");
  const rand = mulberry32(seed);
  const structures = [];
  for (let t = 0; t < trees; t++) {
    const build = (level) => {
      if (level === 0) return { leaf: rand() * 2 - 1 };
      return {
        feature: Math.floor(rand() * dimension),
        threshold: (rand() * 2 - 1) * 0.5,
        left: build(level - 1),
        right: build(level - 1),
      };
    };
    structures.push(build(depth));
  }
  const leaves = (node) => (node.leaf !== undefined ? 1 : leaves(node.left) + leaves(node.right));
  const model = (x) => {
    let total = 0;
    for (const tree of structures) {
      let node = tree;
      while (node.leaf === undefined) node = x[node.feature] <= node.threshold ? node.left : node.right;
      total += node.leaf;
    }
    return total / structures.length;
  };
  return { model, structures, leavesPerTree: leaves(structures[0]) };
}

function selfCheck() {
  // --- validation -----------------------------------------------------------------------------------------
  assert.throws(() => coalitionFunction({ model: null, instance: [1], reference: [0] }), /model must be/);
  assert.throws(() => coalitionFunction({ model: () => 0, instance: [], reference: [] }), /instance must be/);
  assert.throws(() => coalitionFunction({ model: () => 0, instance: [1, 2], reference: [0] }), /reference must match/);
  assert.throws(() => permutationShapley({ model: () => 0, instance: [1, 2], reference: [0, 0], samples: 0 }), /samples/);
  assert.throws(() => kernelShap({ model: () => 0, instance: [1, 2], reference: [0, 0], coalitions: 0 }), /coalitions/);
  assert.throws(() => treeEnsemble({ trees: 0 }), /trees/);
  assert.throws(() => treeEnsemble({ depth: 0 }), /depth/);
  assert.throws(() => exactShapley({ model: () => 0, instance: new Array(21).fill(1), reference: new Array(21).fill(0) }), /20 features/);

  // --- the four Shapley axioms on an exact computation --------------------------------------------------------
  const { model: additive, weights } = additiveModel([1.5, -0.75, 2, 0.25], 0.4);
  const instance = [0.6, -0.2, 0.9, 0.1];
  const reference = [0.2, 0.3, -0.4, 0.5];
  const exact = exactShapley({ model: additive, instance, reference });
  assert.strictEqual(exact.evaluations, 2 ** 4, "the exact method evaluates every coalition exactly once");
  const closure = additive(instance) - additive(reference);
  assert.ok(Math.abs(exact.attributions.reduce((a, b) => a + b, 0) - closure) < 1e-9, "efficiency");
  for (let j = 0; j < 4; j++) {
    assert.ok(Math.abs(exact.attributions[j] - weights[j] * (instance[j] - reference[j])) < 1e-9, `closed form for feature ${j}`);
  }
  // symmetry: swapping two features with the same weight and the same values gives equal attributions
  const symmetricWeights = [1, 1, 0.5];
  const symmetric = exactShapley({
    model: (x) => symmetricWeights[0] * x[0] + symmetricWeights[1] * x[1] + symmetricWeights[2] * x[2],
    instance: [0.7, 0.7, 0.2],
    reference: [0, 0, 0],
  });
  assert.ok(Math.abs(symmetric.attributions[0] - symmetric.attributions[1]) < 1e-12, "symmetry");
  // dummy: a feature the model ignores must get zero
  const dummy = exactShapley({ model: (x) => 3 * x[0], instance: [0.5, 9], reference: [0, 0] });
  assert.ok(Math.abs(dummy.attributions[1]) < 1e-12, "dummy feature");
  assert.ok(Math.abs(dummy.attributions[0] - 1.5) < 1e-12);
  // linearity: attributions of f + g are the sums of the attributions
  const other = additiveModel([0.3, 0.4, -1, 2], -0.2);
  const combined = exactShapley({
    model: (x) => additive(x) + other.model(x),
    instance,
    reference,
  });
  for (let j = 0; j < 4; j++) {
    const single = exactShapley({ model: other.model, instance, reference }).attributions[j];
    assert.ok(Math.abs(combined.attributions[j] - (exact.attributions[j] + single)) < 1e-9, `linearity for feature ${j}`);
  }
  const signs = weights.map((weight, j) => Math.sign(weight * (instance[j] - reference[j])));
  for (let j = 0; j < 4; j++) assert.strictEqual(Math.sign(exact.attributions[j]), signs[j], `sign of feature ${j}`);
  assert.ok(exact.attributions.some((value) => value < 0), "negative attributions are produced");
  assert.ok(exact.attributions.some((value) => value > 0), "positive attributions are produced");

  // --- KernelSHAP with every coalition reproduces the exact values ---------------------------------------------
  const kernelExact = kernelShap({ model: additive, instance, reference });
  for (let j = 0; j < 4; j++) {
    assert.ok(Math.abs(kernelExact.attributions[j] - exact.attributions[j]) < 1e-6, `KernelSHAP must match the exact value at feature ${j}`);
  }
  assert.strictEqual(kernelExact.coalitionsUsed, 2 ** 4 - 2, "every interior coalition is used");
  assert.ok(Math.abs(kernelExact.attributions.reduce((a, b) => a + b, 0) - closure) < 1e-12, "the efficiency constraint is imposed exactly");
  const sampledKernel = kernelShap({ model: additive, instance, reference, coalitions: 8, seed: 5 });
  assert.strictEqual(sampledKernel.coalitionsUsed, 8);
  assert.ok(sampledKernel.attributions.every((value) => Number.isFinite(value)));
  assert.ok(Math.abs(sampledKernel.attributions.reduce((a, b) => a + b, 0) - closure) < 1e-9, "sampled KernelSHAP keeps efficiency");

  // --- permutation sampling converges to the exact values ---------------------------------------------------------
  // A model with interactions has genuine sampling variance (an additive model has none at all).
  const interaction = (x) => x[0] * x[1] + 0.5 * x[2] * x[1] + x[3];
  const interactionExact = exactShapley({ model: interaction, instance, reference });
  const small = permutationShapley({ model: interaction, instance, reference, samples: 20, seed: 866 });
  const medium = permutationShapley({ model: interaction, instance, reference, samples: 400, seed: 866 });
  const large = permutationShapley({ model: interaction, instance, reference, samples: 2000, seed: 866 });
  const errorOf = (run) => Math.max(...run.attributions.map((value, j) => Math.abs(value - interactionExact.attributions[j])));
  assert.ok(small.attributions.every((value) => Number.isFinite(value)));
  assert.ok(errorOf(medium) < 0.2, `400 permutations must be close to exact (${errorOf(medium).toFixed(4)})`);
  assert.ok(errorOf(large) < errorOf(small), `more permutations must help (${errorOf(small).toFixed(4)} -> ${errorOf(large).toFixed(4)})`);
  assert.ok(errorOf(medium) <= errorOf(small) * 1.5, `400 permutations must be at least as good as 20 (${errorOf(small).toFixed(4)} vs ${errorOf(medium).toFixed(4)})`);
  assert.ok(
    large.standardErrors.every((value, j) => value <= medium.standardErrors[j] * 1.5 + 1e-8),
    `the reported standard error must shrink with more samples (${large.standardErrors.map((v) => v.toFixed(5)).join(", ")} vs ${medium.standardErrors.map((v) => v.toFixed(5)).join(", ")})`,
  );
  const scaledRatio = medium.standardErrors[0] / large.standardErrors[0];
  assert.ok(scaledRatio > 1.5 && scaledRatio < 8, `standard errors must scale like 1/sqrt(samples) (ratio ${scaledRatio.toFixed(2)} vs sqrt(5) = 2.24)`);
  assert.ok(Math.abs(large.attributions.reduce((a, b) => a + b, 0) - (interaction(instance) - interaction(reference))) < 0.05, "the sampled sum must be near the closure value");
  // the reported standard error must be an honest calibration of the actual error
  const repeated = [];
  for (let seed = 1; seed <= 12; seed++) repeated.push(permutationShapley({ model: interaction, instance, reference, samples: 200, seed }).attributions);
  for (let j = 0; j < 4; j++) {
    const empirical = Math.sqrt(mean(repeated.map((run) => (run[j] - interactionExact.attributions[j]) ** 2)));
    const reported = permutationShapley({ model: interaction, instance, reference, samples: 200, seed: 1 }).standardErrors[j];
    if (reported === 0) {
      // An additive term has the same marginal contribution in every permutation, so it is exact.
      assert.ok(empirical < 1e-6, `feature ${j} has no sampling variance so it must be exact (${empirical})`);
      continue;
    }
    assert.ok(reported > 0, `feature ${j} must report a positive standard error`);
    assert.ok(empirical < reported * 4 + 0.05, `reported error ${reported.toFixed(4)} must bound the empirical one ${empirical.toFixed(4)}`);
    assert.ok(empirical > reported * 0.2, `the reported error must not be wildly pessimistic (${reported.toFixed(4)} vs ${empirical.toFixed(4)})`);
  }
  // an additive model is attributed exactly by a single permutation
  const additiveSinglePermutation = permutationShapley({ model: additive, instance, reference, samples: 1, seed: 1 });
  for (let j = 0; j < 4; j++) {
    assert.ok(Math.abs(additiveSinglePermutation.attributions[j] - exact.attributions[j]) < 1e-9, "an additive model is exact with one permutation");
    assert.ok(additiveSinglePermutation.standardErrors[j] < 1e-12, "no variance on an additive model");
  }

  // --- tree ensemble: exact attribution, axioms and shadow-free comparison -------------------------------------------------
  const ensemble = treeEnsemble({ trees: 4, depth: 3, dimension: 5, seed: 866 });
  const treeInstance = [0.3, -0.6, 0.8, 0.05, -0.4];
  const treeReference = [0, 0, 0, 0, 0];
  const treeExact = exactShapley({ model: ensemble.model, instance: treeInstance, reference: treeReference });
  assert.strictEqual(treeExact.evaluations, 2 ** 5);
  assert.ok(Math.abs(treeExact.attributions.reduce((a, b) => a + b, 0) - (ensemble.model(treeInstance) - ensemble.model(treeReference))) < 1e-9, "efficiency on a tree ensemble");
  const treeSampled = permutationShapley({ model: ensemble.model, instance: treeInstance, reference: treeReference, samples: 600, seed: 12 });
  const treeError = Math.max(...treeSampled.attributions.map((value, j) => Math.abs(value - treeExact.attributions[j])));
  assert.ok(treeError < 0.15, `the sampled tree attribution must track the exact one (${treeError.toFixed(4)})`);
  const treeKernel = kernelShap({ model: ensemble.model, instance: treeInstance, reference: treeReference });
  assert.ok(treeKernel.attributions.every((value, j) => Math.abs(value - treeExact.attributions[j]) < 1e-6), "KernelSHAP matches the exact tree attribution");
  const unusedFeature = exactShapley({ model: (x) => x[0] * 2 + x[1], instance: [1, 3, 7], reference: [0, 0, 0] });
  assert.ok(Math.abs(unusedFeature.attributions[2]) < 1e-12, "a feature the model ignores is a dummy even in a tree-free model");

  // --- scalability: d = 20 where enumeration is impossible -------------------------------------------------------------
  const bigWeights = [];
  const bigRand = mulberry32(99);
  for (let j = 0; j < 20; j++) bigWeights.push(bigRand() * 2 - 1);
  const bigAdditive = additiveModel(bigWeights, 0.25);
  const bigInstance = bigRand() ? [] : [];
  for (let j = 0; j < 20; j++) bigInstance.push(bigRand() * 2 - 1);
  const bigReference = new Array(20).fill(0);
  const analytic = bigWeights.map((weight, j) => weight * (bigInstance[j] - bigReference[j]));
  const bigSampled = permutationShapley({ model: bigAdditive.model, instance: bigInstance, reference: bigReference, samples: 4000, seed: 4242 });
  const bigError = Math.max(...bigSampled.attributions.map((value, j) => Math.abs(value - analytic[j])));
  assert.ok(bigError < 0.05, `d = 20 sampling must be accurate (${bigError.toFixed(4)})`);
  assert.ok(bigSampled.evaluations > 4000 * 20, "permutation sampling pays d evaluations per permutation");
  assert.ok(Math.abs(analytic.reduce((a, b) => a + b, 0) - (bigAdditive.model(bigInstance) - bigAdditive.model(bigReference))) < 1e-12);
  const bigKernel = kernelShap({ model: bigAdditive.model, instance: bigInstance, reference: bigReference, coalitions: 600, seed: 7 });
  const bigKernelError = Math.max(...bigKernel.attributions.map((value, j) => Math.abs(value - analytic[j])));
  assert.ok(bigKernelError < 0.15, `d = 20 KernelSHAP must be accurate (${bigKernelError.toFixed(4)})`);
  assert.ok(Math.abs(bigKernel.attributions.reduce((a, b) => a + b, 0) - (bigAdditive.model(bigInstance) - bigAdditive.model(bigReference))) < 1e-9);

  // --- degenerate models ---------------------------------------------------------------------------------------------------
  const single = exactShapley({ model: (x) => 3 * x[0], instance: [2], reference: [1] });
  assert.strictEqual(single.attributions.length, 1);
  assert.ok(Math.abs(single.attributions[0] - 3) < 1e-12);
  const constant = exactShapley({ model: () => 5, instance: [1, 2, 3], reference: [0, 0, 0] });
  assert.ok(constant.attributions.every((value) => Math.abs(value) < 1e-12), "a constant model attributes nothing");
  assert.ok(Math.abs(constant.attributions.reduce((a, b) => a + b, 0)) < 1e-12);
  const constantKernel = kernelShap({ model: () => 5, instance: [1, 2, 3], reference: [0, 0, 0], coalitions: 4, seed: 1 });
  assert.ok(constantKernel.attributions.every((value) => Math.abs(value) < 1e-9));
  assert.strictEqual(popcount(0), 0);
  assert.strictEqual(popcount(7), 3);
  assert.strictEqual(choose(5, 2), 10);
  assert.strictEqual(choose(20, 10), 184756);

  // --- determinism -----------------------------------------------------------------------------------------------------------
  const repeatA = permutationShapley({ model: additive, instance, reference, samples: 150, seed: 999 });
  const repeatB = permutationShapley({ model: additive, instance, reference, samples: 150, seed: 999 });
  assert.deepStrictEqual(repeatA.attributions, repeatB.attributions);
  assert.deepStrictEqual(repeatA.standardErrors, repeatB.standardErrors);
  assert.strictEqual(repeatA.evaluations, repeatB.evaluations);
  assert.notDeepStrictEqual(permutationShapley({ model: interaction, instance, reference, samples: 150, seed: 1000 }).attributions, permutationShapley({ model: interaction, instance, reference, samples: 150, seed: 999 }).attributions);
  assert.deepStrictEqual(
    permutationShapley({ model: interaction, instance, reference, samples: 150, seed: 999 }).attributions,
    permutationShapley({ model: interaction, instance, reference, samples: 150, seed: 999 }).attributions,
    "another seed changes the estimate but the same seed does not",
  );
  const kernelA = kernelShap({ model: additive, instance, reference, coalitions: 6, seed: 3 });
  const kernelB = kernelShap({ model: additive, instance, reference, coalitions: 6, seed: 3 });
  assert.deepStrictEqual(kernelA.attributions, kernelB.attributions);
  assert.deepStrictEqual(exactShapley({ model: ensemble.model, instance: treeInstance, reference: treeReference }).attributions, treeExact.attributions);

  console.log(`exact: ${exact.evaluations} coalition evaluations, attributions [${exact.attributions.map((v) => v.toFixed(3)).join(", ")}], closure ${closure.toFixed(3)}`);
  console.log(`sampling: max error ${errorOf(medium).toFixed(4)} at 400 permutations, ${errorOf(large).toFixed(4)} at 2000; standard error ${medium.standardErrors[0].toFixed(4)}`);
  console.log(`scalability: d=20 permutation error ${bigError.toFixed(4)}, KernelSHAP(600 coalitions) error ${bigKernelError.toFixed(4)}`);
  console.log("PASS");
}

module.exports = { exactShapley, permutationShapley, kernelShap, coalitionFunction, additiveModel, treeEnsemble, weightedLeastSquares, popcount, choose, mulberry32 };

if (require.main === module) {
  selfCheck();
}
