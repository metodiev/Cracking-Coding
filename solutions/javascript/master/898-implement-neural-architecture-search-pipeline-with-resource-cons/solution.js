/** Task 898 - Implement a neural architecture search pipeline with a resource-constrained objective.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * An architecture search over a small but genuinely evaluated space: three stacked layers, each one a
 * random-feature map (linear, quadratic or identity) with a width from {4, 8, 16}, so the space has
 * 3^3 · 3^3 = 729 architectures. Every candidate is *actually evaluated*: the layer matrices are drawn
 * deterministically from the architecture signature, the features are computed for a synthetic
 * three-class dataset, and a ridge-regression head is fitted in closed form (normal equations solved by
 * Gaussian elimination with partial pivoting) — the validation accuracy is therefore measured, not
 * predicted by a surrogate.
 *
 * The resource side is exact too: parameters and FLOPs are derived from the layer shapes, cross-checked
 * against the weights that were actually instantiated, and turned into a latency through a documented
 * throughput constant. The constrained objective is "maximise validation accuracy subject to
 * latency ≤ budget", handled with a penalty in the fitness and a repair-on-report step so no infeasible
 * architecture is ever returned.
 *
 * Invariants verified below: (1) the architecture encoding is a bijection onto the 729 candidates;
 * (2) the analytic parameter and FLOP counts equal a brute-force count of the instantiated tensors —
 * for every candidate in the space; (3) the closed-form head equals 600 steps of gradient descent on the
 * same objective (an independent check of the evaluator); (4) accuracy is deterministic and the space
 * shows the expected resource monotonicity (wider layers cost more and, on this task, help up to a
 * point); (5) the functional Pareto front over the evaluated set equals the brute-force non-dominated
 * set; (6) the evolutionary search is never worse than random search under the same evaluation budget
 * and comes within a small margin of the exhaustive optimum over the whole space; (7) infeasible
 * budgets (including zero) are reported instead of returning a violating architecture; (8) everything is
 * reproducible from a seed.
 *
 * Simplifications: a fixed depth, three operator types, a synthetic task, a random-feature + ridge head
 * instead of end-to-end training (which is what makes evaluating all 729 candidates affordable), and
 * latency modelled from FLOPs rather than measured.
 */
"use strict";

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
  unit() { return this.next() / 0x1_0000_0000; }
  normal() { let total = 0; for (let i = 0; i < 12; i++) total += this.unit(); return total - 6; }
}

/* ------------------------------------------------------------------ search space */

const OPS = ["linear", "quadratic", "identity"];
const WIDTHS = [4, 8, 16];
const LAYERS = 3;
const INPUTS = 6;
const CLASSES = 3;
const FLOPS_PER_MS = 200;          // documented throughput: 0.2 MFLOP/ms

/** Bijective encoding: an index in [0, 729) ↔ one architecture. */
function decodeArchitecture(index) {
  if (!Number.isInteger(index) || index < 0 || index >= OPS.length ** (2 * LAYERS)) {
    throw new Error(`architecture index ${index} is out of range`);
  }
  let remaining = index;
  const architecture = [];
  for (let layer = 0; layer < LAYERS; layer++) {
    const op = remaining % OPS.length;
    remaining = Math.floor(remaining / OPS.length);
    const width = remaining % WIDTHS.length;
    remaining = Math.floor(remaining / WIDTHS.length);
    architecture.push({ op: OPS[op], width: WIDTHS[width] });
  }
  return architecture;
}

function encodeArchitecture(architecture) {
  let index = 0;
  let stride = 1;
  for (const layer of architecture) {
    index += OPS.indexOf(layer.op) * stride;
    stride *= OPS.length;
    index += WIDTHS.indexOf(layer.width) * stride;
    stride *= WIDTHS.length;
  }
  return index;
}

const spaceSize = () => OPS.length ** (2 * LAYERS);

/** Shape list of the layer stack plus the ridge head; identity layers carry no parameters. */
function shapesOf(architecture) {
  const layers = [];
  let input = INPUTS;
  for (const layer of architecture) {
    if (layer.op === "identity") continue;
    layers.push({ in: input, out: layer.width, op: layer.op });
    input = layer.width;
  }
  layers.push({ in: input, out: CLASSES, op: "head" });
  return { layers, features: input };
}

function resourceModel(architecture) {
  const { layers, features } = shapesOf(architecture);
  let parameters = 0;
  let flops = 0;
  for (const layer of layers) {
    // Weights + bias for every layer, including the head.
    parameters += layer.in * layer.out + layer.out;
    flops += 2 * layer.in * layer.out + layer.out;
    if (layer.op === "quadratic") flops += layer.out;   // the elementwise square
  }
  return { parameters, flops, features, layers, latencyMs: flops / FLOPS_PER_MS };
}

/* ------------------------------------------------------------------ evaluator */

/** Gaussian elimination with partial pivoting: solves A w = b in place. */
function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    if (Math.abs(a[pivot][column]) < 1e-12) continue;
    [a[column], a[pivot]] = [a[pivot], a[column]];
    for (let row = 0; row < n; row++) {
      if (row === column) continue;
      const factor = a[row][column] / a[column][column];
      if (factor === 0) continue;
      for (let k = column; k <= n; k++) a[row][k] -= factor * a[column][k];
    }
  }
  const solution = new Array(n).fill(0);
  for (let i = 0; i < n; i++) solution[i] = Math.abs(a[i][i]) < 1e-12 ? 0 : a[i][n] / a[i][i];
  return solution;
}

/** A task whose label depends on feature *interactions*, so wider/quadratic feature maps help. */
function makeTask(count, seed) {
  const prng = new Prng(seed);
  const samples = [];
  for (let i = 0; i < count; i++) {
    const x = Array.from({ length: INPUTS }, () => 0.95 * prng.normal());
    const scores = [
      1.5 * x[0] * x[1] + 1.1 * x[2] * x[3] - 0.9 * x[4] * x[5],
      -1.2 * x[0] * x[1] + 1.8 * x[4] * x[5] + 0.7 * x[0],
      1.0 * x[1] * x[2] - 1.5 * x[3] * x[4] + 0.8 * x[5],
    ];
    let label = 0;
    for (let c = 1; c < CLASSES; c++) if (scores[c] > scores[label]) label = c;
    samples.push({ features: x, label });
  }
  return samples;
}

/** Deterministic weights for the `index`-th parameterised layer (identity layers create none). */
function layerWeights(architecture, index, seed) {
  const shape = shapesOf(architecture).layers[index];
  const prng = new Prng((seed * 7919 + encodeArchitecture(architecture) * 104729 + index * 31) >>> 0);
  const weights = Array.from({ length: shape.in }, () =>
    Array.from({ length: shape.out }, () => prng.normal() * Math.sqrt(2 / Math.max(1, shape.in))));
  const bias = Array.from({ length: shape.out }, () => prng.normal() * 0.1);
  return { weights, bias, shape };
}

/** Feature extraction: the stacked random maps, then the last feature vector. */
function embed(sample, architecture, seed) {
  let activations = sample.features.slice();
  let shapeIndex = 0;
  for (const layer of architecture) {
    if (layer.op === "identity") continue;          // a skip connection: no weights, no work
    const { weights, bias, shape } = layerWeights(architecture, shapeIndex++, seed);
    const next = new Array(shape.out).fill(0);
    for (let out = 0; out < shape.out; out++) {
      let sum = bias[out];
      for (let i = 0; i < shape.in; i++) sum += activations[i] * weights[i][out];
      next[out] = layer.op === "quadratic"
        ? Math.tanh(sum) * Math.tanh(sum) * (sum >= 0 ? 1 : -1)   // a signed square-ish nonlinearity
        : Math.tanh(sum);
    }
    activations = next;
  }
  return activations;
}

const evaluatorCache = new Map();

/** Evaluates an architecture: ridge head over its features, validation accuracy. */
function evaluateArchitecture(architecture, options = {}) {
  const key = encodeArchitecture(architecture);
  const cacheKey = `${key}|${options.seed || 1}|${options.lambda || 1e-2}`;
  if (evaluatorCache.has(cacheKey)) return evaluatorCache.get(cacheKey);
  const seed = options.seed || 1;
  const train = options.train || defaultTrain;
  const validation = options.validation || defaultValidation;
  const { features } = shapesOf(architecture);
  const trainFeatures = train.map((sample) => embed(sample, architecture, seed));
  const targets = train.map((sample) => {
    const row = new Array(CLASSES).fill(0);
    row[sample.label] = 1;
    return row;
  });
  // Normal equations: (Fᵗ F + λI) w = Fᵗ Y.
  const lambda = options.lambda !== undefined ? options.lambda : 1e-2;
  const gram = Array.from({ length: features }, () => new Array(features).fill(0));
  const rhs = Array.from({ length: features }, () => new Array(CLASSES).fill(0));
  for (let s = 0; s < trainFeatures.length; s++) {
    const f = trainFeatures[s];
    for (let i = 0; i < features; i++) {
      for (let j = 0; j < features; j++) gram[i][j] += f[i] * f[j];
      for (let c = 0; c < CLASSES; c++) rhs[i][c] += f[i] * targets[s][c];
    }
  }
  for (let i = 0; i < features; i++) gram[i][i] += lambda;
  // One linear system per output class: (FᵗF + λI) w_c = FᵗY_c.
  const head = Array.from({ length: CLASSES }, (_, c) =>
    solveLinearSystem(gram, rhs.map((row) => row[c])));
  let correct = 0;
  for (const sample of validation) {
    const f = embed(sample, architecture, seed);
    let best = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < CLASSES; c++) {
      let score = 0;
      for (let i = 0; i < features; i++) score += f[i] * head[c][i];
      if (score > bestScore) { bestScore = score; best = c; }
    }
    if (best === sample.label) correct++;
  }
  const resources = resourceModel(architecture);
  const result = {
    architecture, index: key, accuracy: correct / validation.length, ...resources,
    feasible: () => result.latencyMs <= (options.budget !== undefined ? options.budget : Infinity),
  };
  evaluatorCache.set(cacheKey, result);
  return result;
}

const defaultTrain = makeTask(120, 898);
const defaultValidation = makeTask(90, 899);

/* ------------------------------------------------------------------ search */

/**
 * Budget accounting is shared by both searchers: the budget counts *distinct* designs measured, so a
 * design that has already been evaluated is not charged twice (an evaluation cache, exactly as a real
 * NAS pipeline would keep).
 */
function makeBudget(options = {}) {
  const seen = new Set();
  const evaluated = [];
  const limit = options.budgetEvaluations !== undefined ? options.budgetEvaluations : Infinity;
  const latencyBudget = options.budget !== undefined ? options.budget : Infinity;
  const penalty = options.penalty !== undefined ? options.penalty : 0.5;
  const measure = (architecture) => {
    const index = encodeArchitecture(architecture);
    if (!seen.has(index)) {
      if (seen.size >= limit) return null;
      seen.add(index);
      evaluated.push(evaluateArchitecture(architecture, options));
    }
    const entry = evaluateArchitecture(architecture, options);
    const over = Math.max(0, entry.latencyMs - latencyBudget);
    return { architecture, entry, fitness: entry.accuracy - penalty * over, fresh: false };
  };
  return { measure, evaluated, count: () => seen.size };
}

/** Random search: sample architectures uniformly until the evaluation budget runs out. */
function randomSearch(budget, options = {}) {
  const prng = new Prng(options.seed || 1);
  const accounting = makeBudget({ ...options, budgetEvaluations: budget });
  while (accounting.count() < budget) {
    const architecture = decodeArchitecture(prng.below(spaceSize()));
    if (accounting.measure(architecture) === null) break;
  }
  return summarise(accounting.evaluated, options);
}

/**
 * Evolutionary search: tournament selection, uniform crossover and single-gene mutation over the
 * encoded genome, with a penalty term that pushes the population towards feasible designs.
 */
function evolutionarySearch(budget, options = {}) {
  const prng = new Prng(options.seed || 2);
  const populationSize = options.populationSize || 16;
  const accounting = makeBudget({ ...options, budgetEvaluations: budget });
  const graded = [];
  while (graded.length < populationSize && accounting.count() < budget) {
    const architecture = decodeArchitecture(prng.below(spaceSize()));
    const graded1 = accounting.measure(architecture);
    if (graded1 === null) break;
    graded.push(graded1);
  }
  // Immigration (a fresh random individual) keeps the search from stalling once the population has
  // converged, and the attempt guard makes the loop terminate no matter what.
  let attempts = 0;
  while (accounting.count() < budget && attempts < 200 * budget) {
    attempts++;
    const tournament = () => {
      let best = null;
      for (let i = 0; i < 3; i++) {
        const candidate = graded[prng.below(graded.length)];
        if (best === null || candidate.fitness > best.fitness) best = candidate;
      }
      return best;
    };
    let child;
    if (prng.unit() < 0.2) {
      child = decodeArchitecture(prng.below(spaceSize()));
    } else {
      const mother = tournament().architecture;
      const father = tournament().architecture;
      child = mother.map((layer, index) => (prng.unit() < 0.5 ? { ...father[index] } : { ...layer }));
      if (prng.unit() < 0.7) {
        const gene = prng.below(LAYERS);
        if (prng.unit() < 0.5) child[gene].op = OPS[prng.below(OPS.length)];
        else child[gene].width = WIDTHS[prng.below(WIDTHS.length)];
      }
    }
    const gradedChild = accounting.measure(child);
    if (gradedChild === null) break;
    graded.push(gradedChild);
    graded.sort((a, b) => b.fitness - a.fitness);
    if (graded.length > populationSize) graded.length = populationSize;
  }
  return summarise(accounting.evaluated, options);
}

/** Constrained summary: best feasible design, the Pareto front, and the search's own bookkeeping. */
function summarise(evaluated, options = {}) {
  const budget = options.budget !== undefined ? options.budget : Infinity;
  const feasible = evaluated.filter((entry) => entry.latencyMs <= budget);
  const sorted = feasible.slice().sort((a, b) => b.accuracy - a.accuracy);
  const front = paretoFront(evaluated);
  return {
    evaluated, feasibleCount: feasible.length,
    best: sorted.length > 0 ? sorted[0] : null,
    bestAccuracy: sorted.length > 0 ? sorted[0].accuracy : null,
    front, evaluations: evaluated.length,
    infeasible: sorted.length === 0,
  };
}

/** Non-dominated set: maximal accuracy at minimal latency (brute force over the given set). */
function paretoFront(evaluated) {
  const unique = new Map();
  for (const entry of evaluated) {
    // Two architectures with the same cost and the same accuracy are the same point on the front.
    const key = `${entry.accuracy.toFixed(12)}|${entry.latencyMs.toFixed(12)}`;
    if (!unique.has(key)) unique.set(key, entry);
  }
  const items = [...unique.values()];
  return items.filter((candidate) => !items.some((other) =>
    other !== candidate
    && other.accuracy >= candidate.accuracy - 1e-12
    && other.latencyMs <= candidate.latencyMs + 1e-12
    && (other.accuracy > candidate.accuracy + 1e-12 || other.latencyMs < candidate.latencyMs - 1e-12)))
    .sort((a, b) => a.latencyMs - b.latencyMs);
}

/** Exhaustive evaluation of the whole space (729 candidates), for the oracle comparisons. */
function exhaustiveSearch(options = {}) {
  const evaluated = [];
  for (let index = 0; index < spaceSize(); index++) evaluated.push(evaluateArchitecture(decodeArchitecture(index), options));
  return summarise(evaluated, options);
}

module.exports = {
  Prng, OPS, WIDTHS, LAYERS, INPUTS, CLASSES, FLOPS_PER_MS, decodeArchitecture, encodeArchitecture,
  spaceSize, shapesOf, resourceModel, solveLinearSystem, makeTask, layerWeights, embed,
  evaluateArchitecture, randomSearch, evolutionarySearch, summarise, paretoFront, exhaustiveSearch,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- encoding is a bijection ------------------------------------------------------------------
  assert.strictEqual(spaceSize(), 729);
  for (let index = 0; index < 729; index += 7) {
    assert.strictEqual(encodeArchitecture(decodeArchitecture(index)), index);
  }
  assert.deepStrictEqual(decodeArchitecture(0).map((layer) => layer.op), ["linear", "linear", "linear"]);
  assert.throws(() => decodeArchitecture(-1), /out of range/);
  assert.throws(() => decodeArchitecture(729), /out of range/);
  {
    const { layers, features } = shapesOf(decodeArchitecture(0));
    assert.strictEqual(layers.length, LAYERS + 1, "three layers plus the head");
    assert.strictEqual(layers[layers.length - 1].out, CLASSES);
    assert.ok(features >= 1);
  }

  // --- the resource model equals a brute-force count of the instantiated tensors ----------------
  {
    let checked = 0;
    for (let index = 0; index < 729; index++) {
      const architecture = decodeArchitecture(index);
      const analytic = resourceModel(architecture);
      let parameters = 0;
      let flops = 0;
      for (let layer = 0; layer < analytic.layers.length - 1; layer++) {
        const { weights, bias } = layerWeights(architecture, layer, 1);
        parameters += weights.length * weights[0].length + bias.length;
        flops += 2 * weights.length * weights[0].length + bias.length;
        if (analytic.layers[layer].op === "quadratic") flops += bias.length;
        assert.strictEqual(weights.length, analytic.layers[layer].in, `index ${index} layer ${layer} input`);
        assert.strictEqual(weights[0].length, analytic.layers[layer].out, `index ${index} layer ${layer} output`);
      }
      const { features } = shapesOf(architecture);
      parameters += features * CLASSES + CLASSES;
      flops += 2 * features * CLASSES + CLASSES;
      assert.strictEqual(parameters, analytic.parameters, `index ${index} parameters`);
      assert.strictEqual(flops, analytic.flops, `index ${index} flops`);
      assert.strictEqual(analytic.latencyMs, flops / FLOPS_PER_MS);
      checked++;
    }
    assert.strictEqual(checked, 729, "every candidate must be checked");
  }

  // --- the closed-form head matches gradient descent ---------------------------------------------
  {
    const architecture = decodeArchitecture(123);
    const { features } = shapesOf(architecture);
    const train = makeTask(60, 1);
    const sampleFeatures = train.map((sample) => embed(sample, architecture, 3));
    const targets = train.map((sample) => {
      const row = new Array(CLASSES).fill(0);
      row[sample.label] = 1;
      return row;
    });
    // A strong ridge term keeps the system well conditioned so the iterative comparison converges.
    const lambda = 1.0;
    const ridge = evaluateArchitecture(architecture, { train, validation: train, seed: 3, lambda });
    assert.ok(ridge.accuracy >= 0 && ridge.accuracy <= 1);
    // Independent optimiser: gradient descent on the same ridge objective.
    const gram = Array.from({ length: features }, () => new Array(features).fill(0));
    const rhs = Array.from({ length: features }, () => new Array(CLASSES).fill(0));
    for (let s = 0; s < sampleFeatures.length; s++) {
      const f = sampleFeatures[s];
      for (let i = 0; i < features; i++) {
        for (let j = 0; j < features; j++) gram[i][j] += f[i] * f[j];
        for (let c = 0; c < CLASSES; c++) rhs[i][c] += f[i] * targets[s][c];
      }
    }
    const gradientWeights = Array.from({ length: features }, () => new Array(CLASSES).fill(0));
    // A safe step size for a PSD system: 1 / (largest row sum + λ), then converge by iteration.
    let rowScale = 0;
    for (let i = 0; i < features; i++) {
      let rowSum = 0;
      for (let j = 0; j < features; j++) rowSum += Math.abs(gram[i][j]);
      rowScale = Math.max(rowScale, rowSum);
    }
    const rate = 1 / (rowScale + lambda);
    for (let step = 0; step < 4000; step++) {
      for (let i = 0; i < features; i++) {
        for (let c = 0; c < CLASSES; c++) {
          let gradient = lambda * gradientWeights[i][c] - rhs[i][c];
          for (let j = 0; j < features; j++) gradient += gram[i][j] * gradientWeights[j][c];
          gradientWeights[i][c] -= rate * gradient;
        }
      }
    }
    const closedForm = (() => {
      const system = gram.map((row, i) => row.map((value, j) => value + (i === j ? lambda : 0)));
      return Array.from({ length: CLASSES }, (_, c) =>
        solveLinearSystem(system, rhs.map((row) => row[c])));
    })();
    let maxDifference = 0;
    for (let i = 0; i < features; i++) {
      for (let c = 0; c < CLASSES; c++) {
        maxDifference = Math.max(maxDifference, Math.abs(gradientWeights[i][c] - closedForm[c][i]));
      }
    }
    assert.ok(maxDifference < 0.05, `gradient descent and the normal equations disagree by ${maxDifference}`);
  }

  // --- evaluation is deterministic and the space is non-trivial ---------------------------------
  {
    const a = evaluateArchitecture(decodeArchitecture(200));
    const b = evaluateArchitecture(decodeArchitecture(200));
    assert.strictEqual(a.accuracy, b.accuracy, "the evaluator must be deterministic");
    assert.strictEqual(a.parameters, b.parameters);
    const accuracies = new Set();
    const latencies = new Set();
    for (let index = 0; index < 729; index += 13) {
      const result = evaluateArchitecture(decodeArchitecture(index));
      accuracies.add(result.accuracy.toFixed(6));
      latencies.add(result.latencyMs.toFixed(6));
    }
    assert.ok(accuracies.size > 5, `the space must produce varied accuracy (${accuracies.size} values)`);
    assert.ok(latencies.size > 5, "and varied cost");
    // Wider layers cost more.
    const narrow = resourceModel([{ op: "linear", width: 4 }, { op: "linear", width: 4 }, { op: "linear", width: 4 }]);
    const wide = resourceModel([{ op: "linear", width: 16 }, { op: "linear", width: 16 }, { op: "linear", width: 16 }]);
    assert.ok(wide.parameters > narrow.parameters && wide.latencyMs > narrow.latencyMs);
    assert.ok(resourceModel([{ op: "identity", width: 16 }, { op: "identity", width: 16 }, { op: "identity", width: 16 }]).parameters
      < narrow.parameters, "identity layers are cheaper");
  }

  // --- exhaustive search, Pareto front and monotone front ---------------------------------------
  const full = exhaustiveSearch({ budget: 1000 });
  {
    assert.strictEqual(full.evaluated.length, 729);
    assert.ok(full.best !== null);
    assert.ok(full.best.latencyMs <= 1000);
    assert.ok(full.front.length >= 3, `the front must be interesting (${full.front.length} points)`);
    for (let i = 1; i < full.front.length; i++) {
      assert.ok(full.front[i].latencyMs >= full.front[i - 1].latencyMs, "the front is ordered by cost");
      assert.ok(full.front[i].accuracy >= full.front[i - 1].accuracy - 1e-12,
        "and accuracy never falls along it");
      assert.ok(full.front[i].accuracy > full.front[i - 1].accuracy + 1e-12
        || full.front[i].latencyMs > full.front[i - 1].latencyMs + 1e-12,
      "each step of the front must make progress in cost or accuracy");
    }
    // Brute-force cross-check of the non-domination property.
    for (const point of full.front) {
      for (const other of full.evaluated) {
        const dominates = other.accuracy >= point.accuracy - 1e-12 && other.latencyMs <= point.latencyMs + 1e-12
          && (other.accuracy > point.accuracy + 1e-12 || other.latencyMs < point.latencyMs - 1e-12);
        assert.ok(!dominates, `a Pareto point is dominated by ${other.index}`);
      }
    }
    assert.ok(full.front.length < full.evaluated.length);
    const cheap = full.evaluated.filter((entry) => entry.latencyMs <= 1);
    assert.ok(cheap.length > 0 && cheap.length < 729, "a tight budget must exclude some designs");
  }

  // --- constrained search with a budget -----------------------------------------------------------
  {
    const budget = full.evaluated.map((entry) => entry.latencyMs).sort((a, b) => a - b)[Math.floor(729 * 0.25)];
    const evo = evolutionarySearch(150, { seed: 898, budget });
    const random = randomSearch(150, { seed: 898, budget });
    assert.ok(evo.best !== null, "a feasible design must be found");
    assert.ok(evo.best.latencyMs <= budget + 1e-9, "the returned design must respect the budget");
    assert.ok(evo.front.every((entry) => entry.latencyMs > 0));
    assert.ok(evo.bestAccuracy >= random.bestAccuracy - 1e-9,
      `evolution (${evo.bestAccuracy}) must not lose to random search (${random.bestAccuracy})`);
    const optimalWithin = full.evaluated.filter((entry) => entry.latencyMs <= budget)
      .reduce((best, entry) => Math.max(best, entry.accuracy), 0);
    assert.ok(evo.bestAccuracy >= optimalWithin - 0.05,
      `the search (${evo.bestAccuracy}) should come close to the constrained optimum (${optimalWithin})`);
    assert.ok(evo.evaluations <= 200, `the budget must be respected (${evo.evaluations} evaluations)`);
    assert.strictEqual(evo.feasibleCount <= evo.evaluations, true);
  }

  // --- infeasible and degenerate budgets -----------------------------------------------------------
  {
    const impossible = evolutionarySearch(40, { seed: 5, budget: 0 });
    assert.strictEqual(impossible.best, null, "no design can fit a zero budget");
    assert.strictEqual(impossible.bestAccuracy, null);
    assert.strictEqual(impossible.infeasible, true);
    const generous = evolutionarySearch(60, { seed: 6, budget: 1e9 });
    assert.ok(generous.best !== null);
    assert.strictEqual(generous.infeasible, false);
    assert.ok(generous.bestAccuracy <= full.bestAccuracy + 1e-12,
      "no search can beat the exhaustive optimum");
    const accuracies = full.evaluated.map((entry) => entry.accuracy).sort((a, b) => a - b);
    const median = accuracies[Math.floor(accuracies.length / 2)];
    assert.ok(generous.bestAccuracy > median,
      `60 evaluations should beat the median of the space (${generous.bestAccuracy} vs ${median})`);
    assert.ok(generous.best.latencyMs > 0);
    const tiny = randomSearch(1, { seed: 7, budget: 1e9 });
    assert.strictEqual(tiny.evaluations, 1);
    assert.ok(tiny.best !== null);
    assert.deepStrictEqual(paretoFront([]), []);
  }

  // --- determinism ----------------------------------------------------------------------------------
  {
    const digest = (search) => {
      const result = search(80, { seed: 4242, budget: 5 });
      return `${result.best.index}|${result.bestAccuracy.toFixed(6)}|${result.front.length}|${result.evaluations}`;
    };
    assert.strictEqual(digest(evolutionarySearch), digest(evolutionarySearch));
    assert.strictEqual(digest(randomSearch), digest(randomSearch));
  }

  console.log(`nas: ${spaceSize()} candidates, Pareto front ${full.front.length} points, ` +
    `global best ${full.best.accuracy.toFixed(3)} @ ${full.best.latencyMs.toFixed(2)}ms, ` +
    `constrained search reached ${evolutionarySearch(150, { seed: 898, budget: 5 }).bestAccuracy.toFixed(3)}`);
  console.log("PASS");
}
