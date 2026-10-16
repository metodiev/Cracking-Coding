/** Task 907 - Implement differentiable programming demo for physical simulation parameter fitting.
 *  (Tier: Master, CTCI Beyond — Differentiable Programming & Scientific Computing)
 *
 * A tiny reverse-mode autodiff engine (Var nodes with a value, an accumulated gradient and a local
 * backward rule) is used to differentiate *through* a damped-oscillator simulation
 *     y'' = -k y - c y'
 * integrated with semi-implicit Euler for 200 steps of dt = 0.05. Fitting k and c to a synthetic
 * trajectory is then ordinary gradient descent on the simulated loss, and the checks confirm the whole
 * pipeline: the gradients match central finite differences, the fit recovers the ground-truth physics,
 * gradient descent beats a coarse grid search, the degenerate cases (no steps, a free particle at rest,
 * a zero learning rate) behave, and the run is reproducible.
 * Simplification: scalars only (no tensors), one ODE, and a gradient step with a halving line search
 * rather than a production optimiser — the point is that the gradient flows through the *simulator*,
 * which a numerical-differentiation demo cannot show.
 */
"use strict";

const assert = require("node:assert");
/* -------------------------------------------------------------------- reverse-mode autodiff */

class Var {
  constructor(value) {
    this.value = value;
    this.grad = 0;
    this.parents = [];
    this.backward = () => {};
  }
}

const isVar = (x) => x instanceof Var;
const valueOf = (x) => (isVar(x) ? x.value : x);

function link(value, parents, rule) {
  const node = new Var(value);
  node.parents = parents;
  node.backward = rule;
  return node;
}

const add = (a, b) => (isVar(a) || isVar(b)
  ? link(valueOf(a) + valueOf(b), [a, b].filter(isVar), (g) => {
    if (isVar(a)) a.grad += g;
    if (isVar(b)) b.grad += g;
  })
  : a + b);

const mul = (a, b) => (isVar(a) || isVar(b)
  ? link(valueOf(a) * valueOf(b), [a, b].filter(isVar), (g) => {
    if (isVar(a)) a.grad += g * valueOf(b);
    if (isVar(b)) b.grad += g * valueOf(a);
  })
  : a * b);

const sub = (a, b) => add(a, mul(-1, b));
const sin = (a) => (isVar(a) ? link(Math.sin(a.value), [a], (g) => { a.grad += g * Math.cos(a.value); }) : Math.sin(a));
const cos = (a) => (isVar(a) ? link(Math.cos(a.value), [a], (g) => { a.grad -= g * Math.sin(a.value); }) : Math.cos(a));
const exp = (a) => (isVar(a) ? link(Math.exp(a.value), [a], (g) => { a.grad += g * Math.exp(a.value); }) : Math.exp(a));
const tanh = (a) => (isVar(a)
  ? link(Math.tanh(a.value), [a], (g) => { a.grad += g * (1 - Math.tanh(a.value) ** 2); })
  : Math.tanh(a));
/** Reverse pass over the (topologically ordered) graph rooted at `output`. */
function backward(output) {
  const order = [];
  const seen = new Set();
  const visit = (node) => {
    if (!isVar(node) || seen.has(node)) return;
    seen.add(node);
    for (const parent of node.parents) visit(parent);
    order.push(node);
  };
  visit(output);
  output.grad = 1;
  for (let i = order.length - 1; i >= 0; i--) order[i].backward(order[i].grad);
}
/* -------------------------------------------------------------------- the physical model */

const DEFAULTS = { dt: 0.05, steps: 200, y0: 1.0, v0: 0.0 };
const TRUTH = { k: 4.0, c: 0.35 };
/** Semi-implicit Euler for y'' = -k y - c y'; works for numbers and for Vars alike. */
function integrate(k, c, params = DEFAULTS) {
  let y = params.y0;
  let v = params.v0;
  const trace = [y];
  for (let step = 0; step < params.steps; step++) {
    const accel = sub(sub(0, mul(k, y)), mul(c, v));
    v = add(v, mul(params.dt, accel));
    y = add(y, mul(params.dt, v));
    trace.push(y);
  }
  return trace;
}
/** Mean squared deviation from a target trajectory — the fitting objective.
 *  The residual is wired through both parameters, so the graph exists (with zero gradient) even when the
 *  trace is static, exactly like requires_grad in a tensor framework. */
function squaredLoss(k, c, targets, params = DEFAULTS) {
  const trace = integrate(k, c, params);
  let total = 0;
  for (let i = 0; i < trace.length; i++) total = add(total, mul(sub(trace[i], targets[i]), sub(trace[i], targets[i])));
  return mul(add(total, mul(add(k, c), 0)), 1 / trace.length);
}

const numericTrace = (k, c, params = DEFAULTS) => integrate(k, c, params);
/** Central finite differences of the loss: the independent oracle for the autodiff gradient. */
function finiteDifferenceGradient(k, c, targets, params = DEFAULTS, h = 1e-6) {
  const kPlus = squaredLoss(k + h, c, targets, params);
  const kMinus = squaredLoss(k - h, c, targets, params);
  const cPlus = squaredLoss(k, c + h, targets, params);
  const cMinus = squaredLoss(k, c - h, targets, params);
  return { k: (kPlus - kMinus) / (2 * h), c: (cPlus - cMinus) / (2 * h) };
}
/** Gradient descent with backtracking line search through the simulator.
 *  The trial step is halved until the objective actually decreases, so the reported loss sequence is
 *  non-increasing by construction — an interval check the tests rely on. */
function fit(targets, options, params = DEFAULTS) {
  const { iterations, learningRate, k0, c0 } = options;
  let k = new Var(k0);
  let c = new Var(c0);
  let step = learningRate;
  const history = [];
  for (let iteration = 0; iteration < iterations; iteration++) {
    k.grad = 0;
    c.grad = 0;
    const objective = squaredLoss(k, c, targets, params);
    const current = objective.value;
    history.push(current);
    backward(objective);
    let trial = step;
    let moved = false;
    for (let attempt = 0; attempt < 30 && !moved; attempt++) {
      const candidateK = k.value - trial * k.grad;
      const candidateC = c.value - trial * c.grad;
      const candidate = valueOf(squaredLoss(candidateK, candidateC, targets, params));
      if (candidate < current) {
        k = new Var(candidateK);
        c = new Var(candidateC);
        step = Math.min(learningRate, trial * 1.5);
        moved = true;
      } else {
        trial /= 2;
      }
    }
    if (!moved) step = learningRate;                  // at the optimum: keep the nominal step and stand still
  }
  return { k: k.value, c: c.value, history };
}

const relativeError = (a, b) => Math.abs(a - b) / Math.max(1e-12, Math.abs(a) + Math.abs(b));

function main() {
  const targets = numericTrace(TRUTH.k, TRUTH.c);
  const options = { iterations: 150, learningRate: 20, k0: 1.0, c0: 1.5 };
  const fitted = fit(targets, options);
  // 1. gradient check — a composite expression differentiated by autodiff vs central differences.
  {
    const expression = (k, c) => add(mul(exp(mul(sub(0, mul(k, c)), 1)), sin(add(k, mul(2, c)))),
      mul(mul(k, k), tanh(add(c, 1))));
    const k = new Var(0.7);
    const c = new Var(1.3);
    const output = expression(k, c);
    backward(output);
    const numeric = (evalK, evalC) => valueOf(expression(evalK, evalC));
    const h = 1e-6;
    const dk = (numeric(0.7 + h, 1.3) - numeric(0.7 - h, 1.3)) / (2 * h);
    const dc = (numeric(0.7, 1.3 + h) - numeric(0.7, 1.3 - h)) / (2 * h);
    assert.ok(relativeError(k.grad, dk) < 1e-6, `d/dk ${k.grad} vs ${dk}`);
    assert.ok(relativeError(c.grad, dc) < 1e-6, `d/dc ${c.grad} vs ${dc}`);
    assert.ok(Number.isFinite(output.value), "the forward value is finite");
  }
  // 2. gradient through the simulator — autodiff vs finite differences on the actual fitting loss.
  {
    const k = new Var(3.1);
    const c = new Var(0.9);
    backward(squaredLoss(k, c, targets));
    const numeric = finiteDifferenceGradient(3.1, 0.9, targets);
    assert.ok(relativeError(k.grad, numeric.k) < 1e-5, `dloss/dk ${k.grad} vs ${numeric.k}`);
    assert.ok(relativeError(c.grad, numeric.c) < 1e-5, `dloss/dc ${c.grad} vs ${numeric.c}`);
    assert.ok(Math.abs(k.grad) > 1e-3 && Math.abs(c.grad) > 1e-3, "the gradients are not trivially zero");
  }
  // 3. the fit recovers the physics, with a monotonically decreasing objective.
  {
    assert.ok(fitted.history[fitted.history.length - 1] < 1e-9,
      `final loss ${fitted.history[fitted.history.length - 1]} must vanish`);
    for (let i = 1; i < fitted.history.length; i++) {
      assert.ok(fitted.history[i] <= fitted.history[i - 1] + 1e-12, `loss must not increase at step ${i}`);
    }
    assert.ok(fitted.history[0] > fitted.history[fitted.history.length - 1], "the loss must actually drop");
    assert.ok(Math.abs(fitted.k - TRUTH.k) < 1e-4, `recovered k ${fitted.k} vs ${TRUTH.k}`);
    assert.ok(Math.abs(fitted.c - TRUTH.c) < 1e-4, `recovered c ${fitted.c} vs ${TRUTH.c}`);
  }
  // 4. independent oracle — gradient descent is at least as good as a coarse grid search.
  {
    let bestGrid = Infinity;
    let bestPair = null;
    for (let k = 1; k <= 7; k += 0.25) {
      for (let c = 0; c <= 2.0001; c += 0.1) {
        const loss = valueOf(squaredLoss(k, c, targets));
        if (loss < bestGrid) { bestGrid = loss; bestPair = { k, c }; }
      }
    }
    const fittedLoss = valueOf(squaredLoss(fitted.k, fitted.c, targets));
    assert.ok(fittedLoss <= bestGrid, `fitted loss ${fittedLoss} vs grid best ${bestGrid}`);
    assert.ok(Math.abs(bestPair.k - TRUTH.k) <= 0.3 && Math.abs(bestPair.c - TRUTH.c) <= 0.15,
      "the grid oracle lands near the truth too");
  }
  // 5. boundary — no integration steps, a free particle at rest, and a zero learning rate.
  {
    const instant = { dt: 0.05, steps: 0, y0: 1, v0: 0 };
    const k = new Var(2.5);
    const c = new Var(0.5);
    const loss = squaredLoss(k, c, [1], instant);
    assert.ok(Math.abs(loss.value) < 1e-12, "a zero-length simulation has zero residual");
    backward(loss);
    assert.ok(Math.abs(k.grad) < 1e-12 && Math.abs(c.grad) < 1e-12, "no dynamics means no parameter gradient");
    const rest = { dt: 0.05, steps: 20, y0: 0, v0: 0 };
    const resting = integrate(4, 0.35, rest);
    assert.ok(resting.every((y) => Math.abs(y) < 1e-15), "a system at rest stays at rest");
    const frozen = fit(targets, { ...options, learningRate: 0 });
    assert.strictEqual(frozen.k, options.k0, "a zero learning rate cannot move k");
    assert.strictEqual(frozen.c, options.c0, "a zero learning rate cannot move c");
    assert.strictEqual(frozen.history.length, options.iterations);
  }
  // 6. the simulator itself — energy must decrease for a damped, passive oscillator.
  {
    const trace = numericTrace(TRUTH.k, TRUTH.c);
    const energy = (i) => 0.5 * trace[i] ** 2;
    assert.ok(energy(trace.length - 1) <= energy(0), "the oscillation must lose energy");
    assert.ok(Math.max(...trace) <= DEFAULTS.y0 + 1e-9, "a passive system cannot overshoot its release");
    const stiff = numericTrace(20, 0.1);
    assert.ok(Math.max(...stiff.map(Math.abs)) <= 1 + 1e-9, "even a stiff system stays bounded");
  }
  // 7. determinism — the fit and the gradients replay exactly.
  {
    const again = fit(targets, options);
    assert.deepStrictEqual(again.history, fitted.history);
    assert.strictEqual(again.k, fitted.k);
    assert.deepStrictEqual(finiteDifferenceGradient(3.1, 0.9, targets), finiteDifferenceGradient(3.1, 0.9, targets));
    const other = fit(targets, { ...options, k0: 2.0 });
    assert.notDeepStrictEqual(other.history, fitted.history);
  }
  console.log(`differentiable-sim: fitted k=${fitted.k.toFixed(6)} c=${fitted.c.toFixed(6)} ` +
    `(truth ${TRUTH.k}/${TRUTH.c}), loss ${fitted.history[0].toFixed(4)} -> ` +
    `${fitted.history[fitted.history.length - 1].toExponential(2)} in ${options.iterations} steps`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  Var, add, mul, sub, sin, cos, exp, tanh, backward, DEFAULTS, TRUTH, integrate, squaredLoss,
  finiteDifferenceGradient, fit, numericTrace, };
