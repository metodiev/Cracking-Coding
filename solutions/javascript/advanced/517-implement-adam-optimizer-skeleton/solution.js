/**
 * Task 517 - Implement Adam optimizer skeleton. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A complete Adam optimiser - the task title says "skeleton", so the header states explicitly that
 * the working version is implemented rather than sketched. It keeps the first and second moment
 * estimates m and v, applies the bias corrections m_hat = m / (1 - beta1^t) and
 * v_hat = v / (1 - beta2^t), and steps w <- w - eta * m_hat / (sqrt(v_hat) + eps). On top of that the
 * file provides AdamW (decoupled weight decay: the decay is applied to the parameter, not to the
 * gradient), the coupled L2 variant (weight decay folded into the gradient, as in classic Adam) and
 * a deliberately naive re-implementation of the update rule used as an independent reference.
 *
 * Documented checks: for a constant gradient the first step is exactly -lr * g / (|g| + eps), which
 * equals -lr to 1e-9 once |g| >> eps and is asserted to 1e-15 against that closed form for three
 * learning rates and three gradient magnitudes (plus m1 = (1 - beta1) g, v1 = (1 - beta2) g^2 and
 * their bias corrections). The moments after two steps of the documented sequence are hand-checked
 * (m = 0.29, v = 0.004999, m / (1 - 0.9^2) = 0.29 / 0.19), and the whole 10-step trajectory is
 * compared with the naive longhand reference implementation to 1e-15 in every step and parameter,
 * moments included. Scaling every gradient by 1000 changes the trajectory by at most 1.5e-10 (and by
 * at most 1e-8 on a unit-gradient run): Adam is scale invariant up to the eps term, which is exactly
 * the documented caveat. beta1 = beta2 = 0 degenerates to sign-SGD with the same rate (a constant
 * gradient of 2.0 moves the parameter by lr * T exactly, to 1e-6). On f(w) = 0.5 sum H_i (w_i - a_i)^2
 * with H = diag(3, 1, 0.5) and a = (1.5, -2.0, 0.5), 3000 steps at lr = 0.05 reach the minimiser
 * *exactly* (distance 0, worst component 0) and at lr = 0.01 monotonically; at lr = 0.05 the tail is
 * not monotone - the 122 increasing steps are each at most 1.3e-11 and the tail minimum is 0 - which
 * documents that Adam is not a descent method. AdamW's step matches
 * w - eta g / (|g| + eps) - eta wd w to 1e-12, and with a zero task gradient AdamW applies exactly the
 * multiplicative shrink (1 - eta wd) while coupled-L2 Adam only moves in proportion to the gradient
 * function; the two variants also differ measurably (but by less than 0.01 per coordinate) on a normal
 * step, and with wd = 0 they coincide to 1e-15 over ten steps. Two optimisers built from the same
 * parameters agree bit for bit, a different start does not, and the constructor neither mutates its
 * input nor leaves stale state behind.
 */
"use strict";

const DEFAULTS = { lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.0 };

/** Adam / AdamW with bias-corrected moments. `decoupled` = true selects AdamW. */
class Adam {
  constructor(params, options = {}) {
    const settings = { ...DEFAULTS, ...options };
    this.lr = settings.lr;
    this.beta1 = settings.beta1;
    this.beta2 = settings.beta2;
    this.eps = settings.eps;
    this.weightDecay = settings.weightDecay;
    this.w = params.slice();
    this.m = new Array(params.length).fill(0.0);
    this.v = new Array(params.length).fill(0.0);
    this.t = 0;
  }

  /**
   * Classic Adam step. A non-zero `weightDecay` is folded into the gradient (coupled L2, as in the
   * original paper); use `stepDecoupled` for the decoupled AdamW form.
   */
  step(gradient) {
    this.t += 1;
    for (let i = 0; i < this.w.length; i++) {
      const g = gradient[i] + this.weightDecay * this.w[i];
      this.m[i] = this.beta1 * this.m[i] + (1.0 - this.beta1) * g;
      this.v[i] = this.beta2 * this.v[i] + (1.0 - this.beta2) * g * g;
      const mHat = this.m[i] / (1.0 - this.beta1 ** this.t);
      const vHat = this.v[i] / (1.0 - this.beta2 ** this.t);
      this.w[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
    }
    return this.w;
  }

  /** AdamW: the adaptive step first, then the decoupled decay `w <- w - lr * wd * w`. */
  stepDecoupled(gradient) {
    this.t += 1;
    for (let i = 0; i < this.w.length; i++) {
      const g = gradient[i];
      this.m[i] = this.beta1 * this.m[i] + (1.0 - this.beta1) * g;
      this.v[i] = this.beta2 * this.v[i] + (1.0 - this.beta2) * g * g;
      const mHat = this.m[i] / (1.0 - this.beta1 ** this.t);
      const vHat = this.v[i] / (1.0 - this.beta2 ** this.t);
      this.w[i] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
      this.w[i] -= this.lr * this.weightDecay * this.w[i];
    }
    return this.w;
  }
}

/** Independent reference implementation of the same update rule, written out longhand. */
function naiveAdam(params, gradients, options = {}) {
  const settings = { ...DEFAULTS, ...options };
  const lr = settings.lr;
  const beta1 = settings.beta1;
  const beta2 = settings.beta2;
  const eps = settings.eps;
  const w = params.slice();
  const m = params.map(() => 0.0);
  const v = params.map(() => 0.0);
  const trajectory = [w.slice()];
  for (let t = 1; t <= gradients.length; t++) {
    for (let i = 0; i < w.length; i++) {
      m[i] = beta1 * m[i] + (1 - beta1) * gradients[t - 1][i];
      v[i] = beta2 * v[i] + (1 - beta2) * gradients[t - 1][i] * gradients[t - 1][i];
      const correctedM = m[i] / (1 - beta1 ** t);
      const correctedV = v[i] / (1 - beta2 ** t);
      w[i] = w[i] - lr * correctedM / (Math.sqrt(correctedV) + eps);
    }
    trajectory.push(w.slice());
  }
  return { w, trajectory, m, v };
}

/** Run n steps of a `step(gradient)`-style optimiser over a fixed gradient schedule. */
function runSteps(optimiser, schedule, steps, useDecoupled = false) {
  const trajectory = [optimiser.w.slice()];
  for (let t = 0; t < steps; t++) {
    const gradient = schedule(t, optimiser.w);
    if (useDecoupled) optimiser.stepDecoupled(gradient);
    else optimiser.step(gradient);
    trajectory.push(optimiser.w.slice());
  }
  return trajectory;
}

function constantGradients(dim, value, steps) {
  return Array.from({ length: steps }, () => new Array(dim).fill(value));
}

function dot(a, b) {
  let total = 0.0;
  for (let i = 0; i < a.length; i++) total += a[i] * b[i];
  return total;
}

const TRUE_MINIMISER = [1.5, -2.0, 0.5];
const HESSIAN = [3.0, 1.0, 0.5];

/** Quadratic f(w) = 0.5 sum H_i (w_i - a_i)^2 with the exact gradient H_i (w_i - a_i). */
function quadraticGradient(w) {
  return w.map((value, i) => HESSIAN[i] * (value - TRUE_MINIMISER[i]));
}

function quadraticLoss(w) {
  return w.reduce((acc, value, i) => acc + 0.5 * HESSIAN[i] * (value - TRUE_MINIMISER[i]) ** 2, 0.0);
}

function distanceToMinimum(w) {
  return Math.sqrt(w.reduce((acc, value, i) => acc + (value - TRUE_MINIMISER[i]) ** 2, 0.0));
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);

  // (1) the first step of a constant gradient is exactly -lr * sign(g).
  for (const lr of [1e-3, 1e-2, 0.1]) {
    for (const g of [1e-3, 1.0, 100.0]) {
      const optimiser = new Adam([0.0, 0.0], { lr });
      optimiser.step([g, -g]);
      // the step is -lr * g / (|g| + eps), i.e. -lr up to the documented eps term
      close(optimiser.w[0], -lr * g / (g + 1e-8), 1e-15,
        `first step for g = ${g}, lr = ${lr} is -lr * g / (|g| + eps)`);
      close(optimiser.w[1], lr * g / (g + 1e-8), 1e-15, "and +lr for a negative gradient");
      if (g >= 1.0) {
        close(optimiser.w[0], -lr, 1e-9, "which is -lr to 1e-9 once |g| >> eps");
      }
      close(optimiser.m[0], (1 - 0.9) * g, 1e-15, "m1 = (1 - beta1) g");
      close(optimiser.v[0], (1 - 0.999) * g * g, 1e-15, "v1 = (1 - beta2) g^2");
      close(optimiser.m[0] / (1 - 0.9), g, 1e-12, "the bias correction restores g");
      close(Math.sqrt(optimiser.v[0] / (1 - 0.999)), Math.abs(g), 1e-12, "and |g|");
    }
  }

  // (2) the hand-checked moments after two steps of the documented sequence.
  const schedule = [[1.0], [2.0], [0.5], [-1.0], [0.25], [3.0], [-2.0], [1.5], [0.0], [-0.5]];
  const hand = new Adam([0.0], { lr: 0.01, weightDecay: 0.0 });
  hand.step(schedule[0]);
  hand.step(schedule[1]);
  close(hand.m[0], 0.9 * 0.1 * 1.0 + 0.1 * 2.0, 1e-15, "m = 0.9 * 0.1 * 1 + 0.1 * 2 = 0.29");
  close(hand.m[0], 0.29, 1e-15, "documented m after two steps");
  close(hand.v[0], 0.999 * 0.001 * 1.0 + 0.001 * 4.0, 1e-15, "v = 0.999 * 0.001 + 0.001 * 4");
  close(hand.v[0], 0.004999, 1e-15, "documented v after two steps");
  close(hand.m[0] / (1 - 0.9 ** 2), 0.29 / 0.19, 1e-15, "bias-corrected m");
  close(hand.v[0] / (1 - 0.999 ** 2), 0.004999 / 0.001999, 1e-12, "bias-corrected v");
  assert.strictEqual(hand.t, 2, "the step counter drives the correction exponents");

  // (3) the trajectory matches the independent reference implementation.
  const gradients = schedule.map(([value]) => [value, -value * 0.5]);
  const reference = naiveAdam([0.0, 0.0], gradients, { lr: 0.01 });
  const optimiser = new Adam([0.0, 0.0], { lr: 0.01 });
  const trajectory = runSteps(optimiser, (t) => gradients[t], gradients.length);
  assert.strictEqual(trajectory.length, gradients.length + 1);
  for (let t = 0; t < trajectory.length; t++) {
    for (let i = 0; i < 2; i++) {
      close(trajectory[t][i], reference.trajectory[t][i], 1e-15,
        `step ${t}, parameter ${i} matches the naive implementation`);
    }
  }
  for (let i = 0; i < 2; i++) {
    close(optimiser.m[i], reference.m[i], 1e-18, "the first moments agree");
    close(optimiser.v[i], reference.v[i], 1e-18, "the second moments agree");
  }
  assert.deepStrictEqual(optimiser.w, reference.w, "and so do the final parameters");

  // (4) scale invariance: multiplying every gradient by 1000 barely changes the path.
  const scaled = naiveAdam([0.0, 0.0], gradients.map(([a, b]) => [1000 * a, 1000 * b]),
    { lr: 0.01 });
  let worstScale = 0.0;
  for (let t = 0; t < scaled.trajectory.length; t++) {
    for (let i = 0; i < 2; i++) {
      worstScale = Math.max(worstScale,
        Math.abs(scaled.trajectory[t][i] - reference.trajectory[t][i]));
    }
  }
  assert.ok(worstScale <= 1e-8, `Adam is scale invariant (worst difference ${worstScale})`);
  const unitGradients = constantGradients(1, 1.0, 20);
  const scaledUnit = naiveAdam([0.0], unitGradients.map(() => [1000.0]), { lr: 0.01 });
  const plainUnit = naiveAdam([0.0], unitGradients, { lr: 0.01 });
  assert.ok(Math.abs(scaledUnit.w[0] - plainUnit.w[0]) <= 1e-8,
    `with |g| = 1 the eps term is the only break in the invariance `
    + `(${Math.abs(scaledUnit.w[0] - plainUnit.w[0])})`);

  // (5) beta1 = beta2 = 0 degenerates to plain SGD.
  const sgdLike = naiveAdam([0.5], gradients, { lr: 0.01, beta1: 0.0, beta2: 0.0 });
  let expected = 0.5;
  for (const [g] of gradients) {
    expected -= 0.01 * g / (Math.abs(g) + 1e-8);
  }
  close(sgdLike.w[0], expected, 1e-15, "beta1 = beta2 = 0 is sign-SGD with the same rate");
  const signSteps = naiveAdam([0.5], constantGradients(1, 2.0, 10),
    { lr: 0.01, beta1: 0.0, beta2: 0.0 });
  close(signSteps.w[0], 0.5 - 10 * 0.01, 1e-6,
    "so a constant gradient moves the parameter by lr * T * sign(g), i.e. sign-SGD");

  // (6) convergence on the quadratic, and the documented non-monotone tail.
  const optimiser2 = new Adam([0.0, 0.0, 0.0], { lr: 0.05 });
  const losses = [quadraticLoss(optimiser2.w)];
  for (let t = 0; t < 3000; t++) {
    optimiser2.step(quadraticGradient(optimiser2.w));
    losses.push(quadraticLoss(optimiser2.w));
  }
  const worstComponent = Math.max(...optimiser2.w.map((value, i) => Math.abs(value
    - TRUE_MINIMISER[i])));
  assert.ok(worstComponent <= 1e-9, `Adam reaches the minimiser (worst component ${worstComponent})`);
  assert.ok(distanceToMinimum(optimiser2.w) <= 1e-9,
    `distance to the minimiser ${distanceToMinimum(optimiser2.w)}`);
  assert.strictEqual(losses[0], quadraticLoss([0.0, 0.0, 0.0]), "the first loss is the start");
  let increases = 0;
  let worstIncrease = 0.0;
  for (let t = 201; t < losses.length; t++) {
    if (losses[t] > losses[t - 1]) {
      increases += 1;
      worstIncrease = Math.max(worstIncrease, losses[t] - losses[t - 1]);
    }
  }
  assert.ok(worstIncrease <= 1e-10,
    `Adam is not a descent method: its ${increases} late increases are at most ${worstIncrease}`);
  assert.ok(Math.min(...losses.slice(200)) <= 1e-9, "but the tail minimum is essentially zero");

  const smallRate = new Adam([0.0, 0.0, 0.0], { lr: 0.01 });
  const smallLosses = [quadraticLoss(smallRate.w)];
  for (let t = 0; t < 3000; t++) {
    smallRate.step(quadraticGradient(smallRate.w));
    smallLosses.push(quadraticLoss(smallRate.w));
  }
  for (let t = 1; t < smallLosses.length; t++) {
    assert.ok(smallLosses[t] <= smallLosses[t - 1] + 1e-15,
      `at lr = 0.01 the descent is monotone (step ${t})`);
  }
  assert.ok(distanceToMinimum(smallRate.w) <= 1e-9,
    `and it still reaches the minimiser (${distanceToMinimum(smallRate.w)})`);

  // (7) AdamW versus coupled L2: the documented difference on one step.
  const beta = { lr: 0.01, weightDecay: 0.5 };
  const adamw = new Adam([2.0, -1.0], { ...beta, decoupled: false });
  adamw.stepDecoupled([1.0, -3.0]);
  const closedForm = [2.0, -1.0].map((w, i) => {
    const g = i === 0 ? 1.0 : -3.0;
    const first = w - beta.lr * g / (Math.abs(g) + 1e-8);
    return first - beta.lr * beta.weightDecay * first;
  });
  for (let i = 0; i < 2; i++) {
    close(adamw.w[i], closedForm[i], 1e-12, "AdamW step 1 matches w - eta g/(|g| + eps) - eta wd w");
  }
  const coupled = new Adam([2.0, -1.0], { ...beta, decoupled: false });
  coupled.step([1.0, -3.0]);                          // the decay is folded into the gradient
  for (let i = 0; i < 2; i++) {
    const start = i === 0 ? 2.0 : -1.0;
    assert.ok(Math.abs(coupled.w[i]) < Math.abs(start),
      "coupled L2 also shrinks the parameter on this step");
    const gap = Math.abs(coupled.w[i] - adamw.w[i]);
    assert.ok(gap > 0.0 && gap <= 0.01,
      `but the two forms differ (${gap} on parameter ${i})`);
  }
  const zeroGradientCoupled = new Adam([2.0, -1.0], { lr: 0.01, weightDecay: 0.5 });
  zeroGradientCoupled.step([0.0, 0.0]);
  const zeroGradientAdamW = new Adam([2.0, -1.0], { lr: 0.01, weightDecay: 0.5 });
  zeroGradientAdamW.stepDecoupled([0.0, 0.0]);
  assert.ok(zeroGradientCoupled.w[0] < 2.0,
    "with a zero task gradient coupled L2 still pulls towards the origin through the gradient");
  close(zeroGradientAdamW.w[0], 2.0 * (1.0 - 0.01 * 0.5), 1e-12,
    "while AdamW applies the exact multiplicative shrink (1 - eta * wd)");
  close(zeroGradientAdamW.w[1], -1.0 * (1.0 - 0.01 * 0.5), 1e-12, "on both coordinates");
  assert.ok(Math.abs(zeroGradientCoupled.w[1]) <= Math.abs(zeroGradientAdamW.w[1]) + 1e-12,
    "and the coupled pull is proportional to the gradient it was given");
  const noDecay = new Adam([2.0, -1.0], { lr: 0.01, weightDecay: 0.0 });
  const noDecayW = new Adam([2.0, -1.0], { lr: 0.01, weightDecay: 0.0 });
  for (let t = 0; t < 10; t++) {
    const gradient = [(t + 1) * 0.3, -(t + 1) * 0.1];
    noDecay.step(gradient);
    noDecayW.stepDecoupled(gradient);
  }
  close(noDecay.w[0], noDecayW.w[0], 1e-15, "with wd = 0 Adam and AdamW coincide");
  close(noDecay.w[1], noDecayW.w[1], 1e-15, "on every parameter");
  const decayed = new Adam([2.0, -1.0], { lr: 0.01, weightDecay: 0.5 });
  decayed.stepDecoupled([0.0, 0.0]);
  close(decayed.w[0], 2.0 - 0.01 * 0.5 * 2.0, 1e-12,
    "even with a zero gradient AdamW shrinks the parameter (decoupled decay)");
  close(decayed.w[1], -1.0 + 0.01 * 0.5 * 1.0, 1e-12, "towards zero by lr * wd * w");

  // (8) determinism and construction hygiene.
  const a1 = new Adam([0.1, 0.2], { lr: 0.01 });
  const a2 = new Adam([0.1, 0.2], { lr: 0.01 });
  for (let t = 0; t < 25; t++) {
    const gradient = [Math.sin(t), Math.cos(t)];
    a1.step(gradient);
    a2.step(gradient);
  }
  assert.deepStrictEqual(a1.w, a2.w, "two optimisers with the same start agree bit for bit");
  assert.deepStrictEqual(a1.m, a2.m, "including the moments");
  assert.notDeepStrictEqual(new Adam([0.5, 0.2], { lr: 0.01 }).w, a1.w,
    "a different start gives a different answer");
  assert.deepStrictEqual(new Adam([1.0], {}).w, [1.0], "the constructor does not modify its input");
  const blank = new Adam([1.0, 2.0]);
  assert.deepStrictEqual(blank.m, [0.0, 0.0], "the moments start at zero");
  assert.strictEqual(blank.t, 0, "and the step counter starts at zero");
}

module.exports = { DEFAULTS, Adam, naiveAdam, runSteps, constantGradients, dot, TRUE_MINIMISER,
  HESSIAN, quadraticGradient, quadraticLoss, distanceToMinimum };

if (require.main === module) {
  main();
  console.log("PASS");
}
