/**
 * Task 598 - Implement online SVM perceptron with kernel trick (simple). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Two online learners over a Gram matrix, so the "kernel trick" is explicitly the only thing that
 * changes between the linear and non-linear cases:
 *   - the kernel perceptron: one dual update alpha_i <- alpha_i + 1 on every mistake, decision
 *     f(x) = sum_i alpha_i y_i k(x_i, x), so the feature map is never materialised;
 *   - a Pegasos-style online linear SVM: subgradient steps on
 *     F(w) = (lambda/2)||w||^2 + (1/n) sum_i max(0, 1 - y_i w.x_i) with the 1/(lambda t) step,
 *     plus the standard averaging of the iterates.
 * Three kernels are used: linear, homogeneous polynomial of degree 2 (with offset, so the Gram matrix
 * is positive semi-definite), and the Gaussian RBF kernel exp(-gamma ||x - x'||^2).
 *
 * Documented checks: the Gram matrix of every kernel is verified to be positive semi-definite by
 * checking z^T K z >= 0 on deterministic probe vectors; the dual kernel perceptron is asserted to
 * produce exactly the same decision values as a primal implementation of the same updates on the
 * linear kernel (1e-12), which is what "the kernel trick changes nothing but the inner product"
 * means concretely; the mistake count on a planted, linearly separable problem is asserted to respect
 * Novikoff's bound mistakes <= (R/gamma)^2 with R = max ||x_i|| and gamma the planted margin, and the
 * bound is reported alongside the measurement; on XOR the linear kernel is asserted to *fail* (it is
 * not linearly separable) while the RBF kernel classifies every training point; Pegasos is asserted
 * to reach F(w_T) <= F(0) = 1 and to satisfy the margin y_i f(x_i) >= 1 for at least 90% of the
 * points, with the measured final objective and hinge reported; and both learners are deterministic
 * (64-bit splitmix64, seed 0x5EED_598).
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed598n;

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
// kernels
// ---------------------------------------------------------------------------------------------

function dot(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc += a[i] * b[i];
  return acc;
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

const KERNELS = {
  linear: () => (a, b) => dot(a, b),
  polynomial: (degree = 2, offset = 1) => (a, b) => (dot(a, b) + offset) ** degree,
  rbf: (gamma = 0.5) => (a, b) =>
    Math.exp(-gamma * a.reduce((acc, value, i) => acc + (value - b[i]) ** 2, 0)),
};

function gramMatrix(points, kernel) {
  return points.map((a) => points.map((b) => kernel(a, b)));
}

function isPositiveSemiDefinite(matrix, probes) {
  let worst = Infinity;
  for (const z of probes) {
    let value = 0;
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix.length; j++) value += z[i] * matrix[i][j] * z[j];
    }
    worst = Math.min(worst, value);
  }
  return worst;
}

// ---------------------------------------------------------------------------------------------
// kernel perceptron (dual) and the primal reference
// ---------------------------------------------------------------------------------------------

function kernelPerceptron(k, labels, options = {}) {
  const { passes = 50 } = options;
  const n = labels.length;
  const alpha = new Array(n).fill(0);
  const mistakesPerPass = [];
  let totalMistakes = 0;
  for (let pass = 0; pass < passes; pass++) {
    let mistakes = 0;
    for (let i = 0; i < n; i++) {
      let decision = 0;
      for (let j = 0; j < n; j++) {
        if (alpha[j] === 0) continue;
        decision += alpha[j] * labels[j] * k[j][i];
      }
      if (labels[i] * decision <= 0) {
        alpha[i] += 1;
        mistakes++;
        totalMistakes++;
      }
    }
    mistakesPerPass.push(mistakes);
    if (mistakes === 0) break;
  }
  return { alpha, mistakesPerPass, totalMistakes,
    decisions: (index) => labels.reduce((acc, label, j) =>
      acc + alpha[j] * label * k[j][index], 0) };
}

/** Primal perceptron on explicitly given points; used to check the dual on the linear kernel. */
function primalPerceptron(points, labels, options = {}) {
  const { passes = 50 } = options;
  const d = points[0].length;
  const w = new Array(d).fill(0);
  let bias = 0;
  let totalMistakes = 0;
  const mistakesPerPass = [];
  const n = points.length;
  for (let pass = 0; pass < passes; pass++) {
    let mistakes = 0;
    for (let i = 0; i < n; i++) {
      const decision = dot(w, points[i]);
      if (labels[i] * decision <= 0) {
        for (let j = 0; j < d; j++) w[j] += labels[i] * points[i][j];
        mistakes++;
        totalMistakes++;
      }
    }
    mistakesPerPass.push(mistakes);
    if (mistakes === 0) break;
  }
  assert.strictEqual(bias, 0);
  return { w, mistakesPerPass, totalMistakes };
}

function decisionFromAlpha(points, labels, alpha, point, kernel) {
  return labels.reduce((acc, label, j) =>
    acc + alpha[j] * label * kernel(points[j], point), 0);
}

function accuracy(decisions, labels) {
  let correct = 0;
  decisions.forEach((value, i) => {
    if (Math.sign(value) === labels[i]) correct++;
  });
  return correct / labels.length;
}

// ---------------------------------------------------------------------------------------------
// Pegasos
// ---------------------------------------------------------------------------------------------

function hingeLoss(w, points, labels) {
  return points.reduce((acc, point, i) =>
    acc + Math.max(0, 1 - labels[i] * dot(w, point)), 0) / points.length;
}

function pegasosObjective(w, points, labels, lambda) {
  return 0.5 * lambda * dot(w, w) + hingeLoss(w, points, labels);
}

/**
 * Pegasos: at step t pick one example at random, take the subgradient step with 1/(lambda t), and
 * keep the running average of the iterates (the average is what the bounds apply to).
 */
function pegasos(points, labels, lambda, iterations, seed = DEFAULT_SEED) {
  const d = points[0].length;
  const rng = new SplitMix64(BigInt(seed));
  let w = new Array(d).fill(0);
  const average = new Array(d).fill(0);
  const objectives = [];
  const hinges = [];
  for (let t = 1; t <= iterations; t++) {
    const index = rng.nextInt(points.length);
    const margin = labels[index] * dot(w, points[index]);
    const rate = 1 / (lambda * t);
    if (margin < 1) {
      w = w.map((value, j) => (1 - rate * lambda) * value
        + rate * labels[index] * points[index][j]);
    } else {
      w = w.map((value) => (1 - rate * lambda) * value);
    }
    const weight = t === 1 ? 1 : 2 / (t + 1);
    for (let j = 0; j < d; j++) {
      average[j] = (1 - weight) * average[j] + weight * w[j];
    }
    objectives.push(pegasosObjective(w, points, labels, lambda));
    hinges.push(hingeLoss(w, points, labels));
  }
  return { w, average, objectives, hinges };
}

// ---------------------------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------------------------

/** Linearly separable blobs around +-direction with margin `margin` and small noise. */
function separableData(count, d, rng, margin = 1, noise = 0.15) {
  const direction = (() => {
    let vector = Array.from({ length: d }, () => rng.nextNormal());
    const length = norm(vector);
    vector = vector.map((value) => value / length);
    return vector;
  })();
  const points = [];
  const labels = [];
  for (let i = 0; i < count; i++) {
    const label = i % 2 === 0 ? 1 : -1;
    const point = direction.map((value) =>
      label * margin * value + rng.nextNormal() * noise);
    points.push(point);
    labels.push(label);
  }
  return { points, labels, direction };
}

function xorData(rng, scale = 0.2) {
  const points = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    .map(([a, b]) => [a + rng.nextNormal() * scale, b + rng.nextNormal() * scale]);
  const labels = [1, -1, -1, 1];
  return { points, labels };
}

function main() {
  const rng = new SplitMix64(DEFAULT_SEED);
  const probes = Array.from({ length: 5 }, () =>
    Array.from({ length: 40 }, () => rng.nextNormal()));

  // (1) Every kernel's Gram matrix is positive semi-definite.
  const cloud = separableData(40, 3, rng);
  const kernels = {
    linear: KERNELS.linear(),
    polynomial: KERNELS.polynomial(2, 1),
    rbf: KERNELS.rbf(0.5),
    rbfSharp: KERNELS.rbf(4),
  };
  for (const [name, kernel] of Object.entries(kernels)) {
    const k = gramMatrix(cloud.points, kernel);
    const worst = isPositiveSemiDefinite(k, probes);
    assert.ok(worst >= -1e-12, `the ${name} Gram matrix must be PSD (worst z^T K z = ${worst})`);
    for (let i = 0; i < k.length; i++) {
      assert.ok(Math.abs(k[i][i] - kernel(cloud.points[i], cloud.points[i])) <= 1e-12);
    }
  }
  const linearGram = gramMatrix(cloud.points, kernels.linear);
  assert.ok(Math.abs(linearGram[0][1] - dot(cloud.points[0], cloud.points[1])) <= 1e-12);

  // (2) The dual kernel perceptron equals the primal implementation on the linear kernel.
  const separable = separableData(60, 2, new SplitMix64(0x598n), 1, 0.2);
  const separableGram = gramMatrix(separable.points, kernels.linear);
  const dual = kernelPerceptron(separableGram, separable.labels);
  const primal = primalPerceptron(separable.points, separable.labels);
  assert.strictEqual(dual.totalMistakes, primal.totalMistakes,
    `documented: dual and primal make the same mistake count (${dual.totalMistakes})`);
  assert.deepStrictEqual(dual.mistakesPerPass, primal.mistakesPerPass);
  for (let i = 0; i < separable.points.length; i++) {
    const dualDecision = decisionFromAlpha(separable.points, separable.labels, dual.alpha,
      separable.points[i], kernels.linear);
    const primalDecision = dot(primal.w, separable.points[i]);
    assert.ok(Math.abs(dualDecision - primalDecision) <= 1e-12,
      `dual and primal decisions agree at point ${i}`);
  }
  assert.strictEqual(dual.mistakesPerPass[dual.mistakesPerPass.length - 1], 0,
    "the perceptron separates the planted data on a final pass");

  // (3) Novikoff's bound on the mistake count.
  const radius = Math.max(...separable.points.map((point) => norm(point)));
  const direction = separable.direction;
  const margin = Math.min(...separable.points.map((point, i) =>
    separable.labels[i] * dot(direction, point)));
  assert.ok(margin > 0, `the planted margin is ${margin}`);
  const novikoffBound = (radius / margin) ** 2;
  assert.ok(dual.totalMistakes <= novikoffBound,
    `documented: mistakes ${dual.totalMistakes} respect Novikoff's bound (R/gamma)^2 = `
    + `${novikoffBound} with R = ${radius} and gamma = ${margin}`);
  assert.ok(novikoffBound > 1, "and the bound is a meaningful number here");
  const scaled = separableData(60, 2, new SplitMix64(0x599n), 2.5, 0.2);
  const scaledDual = kernelPerceptron(gramMatrix(scaled.points, kernels.linear), scaled.labels);
  const scaledRadius = Math.max(...scaled.points.map((point) => norm(point)));
  const scaledMargin = Math.min(...scaled.points.map((point, i) =>
    scaled.labels[i] * dot(scaled.direction, point)));
  assert.ok(scaledDual.totalMistakes <= (scaledRadius / scaledMargin) ** 2,
    `documented: a wider margin (${scaledMargin} vs ${margin}) needs ${scaledDual.totalMistakes} `
    + `mistakes instead of ${dual.totalMistakes}, and the bound tracks that`);

  // (4) XOR: the linear kernel cannot do it, the RBF kernel can.
  const xor = xorData(new SplitMix64(0x598n));
  const xorLinear = kernelPerceptron(gramMatrix(xor.points, kernels.linear), xor.labels,
    { passes: 200 });
  const xorLinearDecisions = xor.points.map((_, i) =>
    decisionFromAlpha(xor.points, xor.labels, xorLinear.alpha, xor.points[i], kernels.linear));
  assert.ok(accuracy(xorLinearDecisions, xor.labels) < 1,
    `documented: the linear kernel cannot separate XOR (accuracy `
    + `${accuracy(xorLinearDecisions, xor.labels)}) - it is not linearly separable`);
  const rbfXorKernel = KERNELS.rbf(1.0);
  const xorRbf = kernelPerceptron(gramMatrix(xor.points, rbfXorKernel), xor.labels, { passes: 200 });
  const xorRbfDecisions = xor.points.map((_, i) =>
    decisionFromAlpha(xor.points, xor.labels, xorRbf.alpha, xor.points[i], rbfXorKernel));
  assert.strictEqual(accuracy(xorRbfDecisions, xor.labels), 1,
    "the RBF kernel separates XOR with no mistakes");
  assert.ok(xorRbf.totalMistakes < xorLinear.totalMistakes + 200);
  assert.ok(Math.abs(xorRbfDecisions[0]) > 0.05 && Math.abs(xorRbfDecisions[1]) > 0.05,
    "the RBF decision values are well away from zero, not numerical noise");

  // (5) Pegasos: objective, margin condition and the hinge decay.
  const lambda = 0.01;
  const training = separableData(80, 2, new SplitMix64(0x5a8n), 1.2, 0.15);
  const run = pegasos(training.points, training.labels, lambda, 400, 0x598n);
  const zeroObjective = pegasosObjective(new Array(2).fill(0), training.points, training.labels,
    lambda);
  assert.ok(Math.abs(zeroObjective - 1) <= 1e-12,
    "F(0) = 1 exactly, since every training hinge at w = 0 is 1");
  const finalObjective = run.objectives[run.objectives.length - 1];
  assert.ok(finalObjective <= zeroObjective,
    `documented: Pegasos ends at F(w_T) = ${finalObjective} <= F(0) = ${zeroObjective}`);
  const positiveHingeSteps = run.hinges.filter((value) => value > 0).length;
  assert.ok(positiveHingeSteps > 0 && positiveHingeSteps < run.hinges.length,
    `documented: ${positiveHingeSteps} of ${run.hinges.length} iterates still had slack; once the `
    + "classifier separates the set the hinge is exactly zero and the step only shrinks w");
  assert.ok(run.hinges[run.hinges.length - 1] <= 1e-12,
    `and the final hinge is ${run.hinges[run.hinges.length - 1]}, i.e. no slack at all`);
  const finalDecisions = training.points.map((point) => dot(run.average, point));
  assert.ok(accuracy(finalDecisions, training.labels) >= 0.99,
    `Pegasos reaches accuracy ${accuracy(finalDecisions, training.labels)} on the separable set`);
  const marginSatisfied = training.points.filter((point, i) =>
    training.labels[i] * dot(run.average, point) >= 1).length / training.points.length;
  assert.ok(marginSatisfied >= 0.9,
    `documented: ${marginSatisfied} of the points satisfy y_i f(x_i) >= 1, the usual margin `
    + "condition up to slack");
  assert.ok(run.average.every((value) => Number.isFinite(value)));
  assert.ok(norm(run.average) > 0);
  const earlyAverageObjective = run.objectives.slice(0, 20)
    .reduce((acc, value) => acc + value, 0) / 20;
  const lateAverageObjective = run.objectives.slice(-20)
    .reduce((acc, value) => acc + value, 0) / 20;
  assert.ok(lateAverageObjective <= earlyAverageObjective,
    `documented: the objective average drops from ${earlyAverageObjective} to ${lateAverageObjective}`);
  assert.ok(lateAverageObjective <= 0.01 * earlyAverageObjective,
    `documented: by two orders of magnitude, from ${earlyAverageObjective} to `
    + `${lateAverageObjective}, i.e. the subgradient steps plus the regularisation reach a small `
    + "objective quickly on a separable problem");

  // (6) Kernel perceptron on an inseparable problem: mistakes keep coming, as documented.
  const rng2 = new SplitMix64(0x1234n);
  const noisy = { points: [], labels: [] };
  for (let i = 0; i < 40; i++) {
    const label = i % 2 === 0 ? 1 : -1;
    noisy.points.push([label * 0.1 + rng2.nextNormal() * 1.2, rng2.nextNormal() * 1.2]);
    noisy.labels.push(label);
  }
  const noisyRun = kernelPerceptron(gramMatrix(noisy.points, kernels.linear), noisy.labels,
    { passes: 30 });
  assert.strictEqual(noisyRun.mistakesPerPass.length, 30,
    "an inseparable problem never reaches a mistake-free pass");
  assert.ok(noisyRun.mistakesPerPass[noisyRun.mistakesPerPass.length - 1] > 0,
    `documented: the final pass still makes ${noisyRun.mistakesPerPass[noisyRun.mistakesPerPass.length - 1]} `
    + "mistakes; the perceptron only converges when the data is separable");

  // (7) Determinism.
  const first = kernelPerceptron(separableGram, separable.labels);
  const second = kernelPerceptron(separableGram, separable.labels);
  assert.deepStrictEqual(first.alpha, second.alpha);
  assert.deepStrictEqual(first.mistakesPerPass, second.mistakesPerPass);
  assert.deepStrictEqual(pegasos(training.points, training.labels, lambda, 50, 7n).w,
    pegasos(training.points, training.labels, lambda, 50, 7n).w);
  assert.notDeepStrictEqual(pegasos(training.points, training.labels, lambda, 50, 8n).w,
    pegasos(training.points, training.labels, lambda, 50, 7n).w,
    "a different sampling seed gives a different iterate");
  const rngA = new SplitMix64(31n);
  const rngB = new SplitMix64(31n);
  assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
    Array.from({ length: 8 }, () => rngB.next()));
}

module.exports = { SplitMix64, DEFAULT_SEED, dot, norm, KERNELS, gramMatrix, isPositiveSemiDefinite,
  kernelPerceptron, primalPerceptron, decisionFromAlpha, accuracy, hingeLoss, pegasosObjective,
  pegasos, separableData, xorData };

if (require.main === module) {
  main();
  console.log("PASS");
}
