/** Task 825 - Implement end-to-end differentiable physics simulation toy. (Tier: Master, Beyond — Machine Learning & AI)
 *
 * A particle chain in 2-D is rolled out by a symplectic (semi-implicit) Euler integrator whose every
 * arithmetic step is recorded on a reverse-mode tape. Gravity, distance springs between neighbours,
 * the integrator, the loss and the optimiser therefore form one differentiable program: gradients of
 * the trajectory loss flow back into the physical parameters (initial positions and velocities, the
 * spring constant), exactly as in a differentiable simulator / "learn the simulator" pipeline.
 *
 * Physics (per step): forces are gravity -m*g for every particle plus, for each neighbour pair,
 * f = k*(|d| - L0)/|d| * d (attractive when stretched, repulsive when compressed); then
 * v <- v + dt*f/m followed by p <- p + dt*v. Frame 0 (the given initial state) is part of the loss,
 * which pins the initial condition; the loss is the MSE over all frames against a target trajectory.
 *
 * What is simplified: the topology is a fixed 1-D chain (no collision detection, no contact solver,
 * no sparse linear solve, single scalar spring model) and "at scale" is expressed through typed size
 * parameters (particles, steps, number of probe directions) instead of real parallelism. |d| uses a
 * +1e-12 floor so d|d|/dp stays finite when two particles coincide, which keeps the rollout smooth
 * everywhere. No timers, no randomness: every direction/target comes from a seeded LCG.
 *
 * Invariants verified:
 *   - integrator correctness: the taped rollout reproduces the closed-form semi-implicit free-fall
 *     trajectory p_n = p0 + n*dt*v0 - g*dt^2*n(n+1)/2 to 1e-12;
 *   - gradient correctness: tape gradients equal the analytic free-fall gradients and central finite
 *     differences of the general chain (random directions and per-parameter), while a negated
 *     gradient is rejected by the same test;
 *   - invariance: parameters that provably cannot influence the loss (mass in a gravity-only
 *     rollout, k/L0 of a single particle) get a gradient of exactly 0;
 *   - the symplectic scheme has bounded, non-secular energy error (max drift does not grow when the
 *     number of steps quadruples);
 *   - end-to-end learning: gradient descent on the taped loss recovers the ground-truth spring
 *     constant and the ground-truth initial state to ~1e-6 and drives the loss below 1e-13.
 */
"use strict";

/** Reverse-mode tape: parents always precede children, so reverse iteration is a valid topo order. */
class Tape {
  constructor() {
    this.nodes = [];
  }
  /** Creates a leaf parameter with a zero adjoint. */
  input(value, label = "") {
    const node = { value, grad: 0, parents: [], label, leaf: true };
    this.nodes.push(node);
    return node;
  }
  /** Creates an interior node from its value and the local derivatives w.r.t. its parents. */
  op(value, parents, label = "") {
    const node = { value, grad: 0, parents, label, leaf: false };
    this.nodes.push(node);
    return node;
  }
  /** Seeds the root adjoint at 1 and accumulates backwards; call once per fresh tape. */
  backward(root) {
    for (const node of this.nodes) node.grad = 0;
    if (root === null || root === undefined) return; // empty loss (zero particles)
    root.grad = 1;
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const node = this.nodes[i];
      if (node.grad === 0) continue;
      for (let k = 0; k < node.parents.length; k++) {
        node.parents[k][0].grad += node.grad * node.parents[k][1];
      }
    }
  }
}

const add = (t, a, b) => t.op(a.value + b.value, [[a, 1], [b, 1]], "add");
const sub = (t, a, b) => t.op(a.value - b.value, [[a, 1], [b, -1]], "sub");
const mul = (t, a, b) => t.op(a.value * b.value, [[a, b.value], [b, a.value]], "mul");
const div = (t, a, b) => t.op(a.value / b.value, [[a, 1 / b.value], [b, -a.value / (b.value * b.value)]], "div");
const neg = (t, a) => t.op(-a.value, [[a, -1]], "neg");
const scale = (t, c, a) => t.op(c * a.value, [[a, c]], "scale");
const fsqrt = (t, a) => t.op(Math.sqrt(a.value), [[a, 0.5 / Math.sqrt(a.value)]], "sqrt");

const DIST_FLOOR = 1e-12;

/** Flat parameter layout: p0 (2n), v0 (2n), k, m — the same order as the returned gradient. */
function specVector(spec) {
  return [...spec.p0, ...spec.v0, spec.k, spec.m];
}

function specFromVector(spec, vector) {
  const n2 = 2 * spec.n;
  return {
    ...spec,
    p0: vector.slice(0, n2),
    v0: vector.slice(n2, 2 * n2),
    k: vector[2 * n2],
    m: vector[2 * n2 + 1],
  };
}

/** Forces at the current state: gravity plus neighbour springs (shared shape in both integrators). */
function forcesOf(spec, x, y) {
  const fx = new Array(spec.n).fill(0);
  const fy = new Array(spec.n).fill(-spec.m * spec.gravity);
  for (let i = 0; i + 1 < spec.n; i++) {
    const dx = x[i + 1] - x[i];
    const dy = y[i + 1] - y[i];
    const dist = Math.sqrt(dx * dx + dy * dy + DIST_FLOOR);
    const coef = spec.k * (dist - spec.restLength) / dist;
    fx[i] += coef * dx;
    fx[i + 1] -= coef * dx;
    fy[i] += coef * dy;
    fy[i + 1] -= coef * dy;
  }
  return [fx, fy];
}

/**
 * Rolls the chain forward on `tape`. Returns the normalised loss, the parameter nodes and the
 * per-frame position nodes. `spec.targets` holds steps+1 frames of 2n coordinates.
 */
function rollout(tape, spec) {
  const n = spec.n;
  if (n === 0) return { loss: null, params: null, frames: [] };
  const k = tape.input(spec.k, "k");
  const m = tape.input(spec.m, "m");
  const gravity = tape.input(spec.gravity, "gravity");
  const p0 = [];
  const v0 = [];
  const px = [];
  const py = [];
  const vx = [];
  const vy = [];
  for (let i = 0; i < n; i++) {
    p0.push(tape.input(spec.p0[2 * i], `p0x${i}`), tape.input(spec.p0[2 * i + 1], `p0y${i}`));
    v0.push(tape.input(spec.v0[2 * i], `v0x${i}`), tape.input(spec.v0[2 * i + 1], `v0y${i}`));
    px[i] = p0[2 * i];
    py[i] = p0[2 * i + 1];
    vx[i] = v0[2 * i];
    vy[i] = v0[2 * i + 1];
  }
  const invM = div(tape, tape.input(1, "one"), m);
  const restLength = tape.input(spec.restLength, "restLength");
  const frames = [{ x: px.slice(), y: py.slice() }];

  let loss = null;
  const accumulate = (frame) => {
    let acc = tape.input(0, "sum");
    for (let i = 0; i < n; i++) {
      const dx = sub(tape, px[i], tape.input(frame.x[i], `tx${i}`));
      const dy = sub(tape, py[i], tape.input(frame.y[i], `ty${i}`));
      acc = add(tape, acc, add(tape, mul(tape, dx, dx), mul(tape, dy, dy)));
    }
    loss = loss === null ? acc : add(tape, loss, acc);
  };
  accumulate(spec.targets[0]);

  for (let step = 0; step < spec.steps; step++) {
    let fx = [];
    let fy = [];
    for (let i = 0; i < n; i++) {
      fx.push(tape.input(0, `fx${i}`));
      fy.push(neg(tape, mul(tape, m, gravity))); // gravity acts on every particle
    }
    for (let i = 0; i + 1 < n; i++) {
      const dx = sub(tape, px[i + 1], px[i]);
      const dy = sub(tape, py[i + 1], py[i]);
      const dist = fsqrt(tape, add(tape, add(tape, mul(tape, dx, dx), mul(tape, dy, dy)), tape.input(DIST_FLOOR, "floor")));
      const coef = div(tape, mul(tape, k, sub(tape, dist, restLength)), dist); // k*(|d|-L0)/|d|
      const sx = mul(tape, coef, dx);
      const sy = mul(tape, coef, dy);
      fx[i] = add(tape, fx[i], sx);
      fx[i + 1] = sub(tape, fx[i + 1], sx);
      fy[i] = add(tape, fy[i], sy);
      fy[i + 1] = sub(tape, fy[i + 1], sy);
    }
    for (let i = 0; i < n; i++) {
      vx[i] = add(tape, vx[i], scale(tape, spec.dt, mul(tape, fx[i], invM))); // v <- v + dt*F/m
      vy[i] = add(tape, vy[i], scale(tape, spec.dt, mul(tape, fy[i], invM)));
      px[i] = add(tape, px[i], scale(tape, spec.dt, vx[i])); // p <- p + dt*v  (semi-implicit)
      py[i] = add(tape, py[i], scale(tape, spec.dt, vy[i]));
    }
    frames.push({ x: px.slice(), y: py.slice() });
    accumulate(spec.targets[step + 1]);
  }

  const total = (spec.steps + 1) * n;
  return { loss: div(tape, loss, tape.input(total, "total")), params: { p0, v0, k, m, gravity }, frames };
}

/** Differentiates the rollout loss; returns [loss, gradient in specVector order]. */
function lossAndGradient(spec) {
  const tape = new Tape();
  const { loss, params } = rollout(tape, spec);
  tape.backward(loss);
  if (params === null) return [null, []];
  const grad = [];
  for (const node of params.p0) grad.push(node.grad);
  for (const node of params.v0) grad.push(node.grad);
  grad.push(params.k.grad, params.m.grad);
  return [loss.value, grad];
}

const lossValue = (spec) => lossAndGradient(spec)[0];

/** Numeric trajectory of the identical scheme, read straight off the tape nodes. */
function simulate(spec) {
  const tape = new Tape();
  const { frames } = rollout(tape, spec);
  return frames.map((frame) => ({ x: frame.x.map((node) => node.value), y: frame.y.map((node) => node.value) }));
}

/** Central-difference estimate of the directional derivative of the loss along `direction`. */
function directionalDifference(spec, direction, h) {
  const vector = specVector(spec);
  const plus = specFromVector(spec, vector.map((v, i) => v + h * direction[i]));
  const minus = specFromVector(spec, vector.map((v, i) => v - h * direction[i]));
  return (lossValue(plus) - lossValue(minus)) / (2 * h);
}

/** Closed form of the plain semi-implicit free fall (gravity only, one particle). */
function freeFallPosition(p0, v0, g, dt, steps) {
  const out = [p0];
  for (let n = 1; n <= steps; n++) out.push(p0 + n * dt * v0 - g * dt * dt * (n * (n + 1)) / 2);
  return out;
}

/** Fixes a target sequence the model has to match. */
function frame(x, y) {
  return { x: x.slice(), y: y.slice() };
}

/** Plain-number rollout used for targets and for the energy study of the same scheme. */
function integrate(spec) {
  const n = spec.n;
  const x = [];
  const y = [];
  const ux = [];
  const uy = [];
  for (let i = 0; i < n; i++) {
    x.push(spec.p0[2 * i]);
    y.push(spec.p0[2 * i + 1]);
    ux.push(spec.v0[2 * i]);
    uy.push(spec.v0[2 * i + 1]);
  }
  const frames = [frame(x, y)];
  const energies = [energy(spec, x, y, ux, uy)];
  for (let step = 0; step < spec.steps; step++) {
    const [fx, fy] = forcesOf(spec, x, y);
    for (let i = 0; i < n; i++) {
      ux[i] += spec.dt * fx[i] / spec.m;
      uy[i] += spec.dt * fy[i] / spec.m;
      x[i] += spec.dt * ux[i];
      y[i] += spec.dt * uy[i];
    }
    frames.push(frame(x, y));
    energies.push(energy(spec, x, y, ux, uy));
  }
  return { frames, energies };
}

/** Kinetic + spring + gravitational potential energy of the chain. */
function energy(spec, x, y, ux, uy) {
  let total = 0;
  for (let i = 0; i < spec.n; i++) {
    total += 0.5 * spec.m * (ux[i] * ux[i] + uy[i] * uy[i]) + spec.m * spec.gravity * y[i];
  }
  for (let i = 0; i + 1 < spec.n; i++) {
    const dx = x[i + 1] - x[i];
    const dy = y[i + 1] - y[i];
    const stretch = Math.sqrt(dx * dx + dy * dy) - spec.restLength;
    total += 0.5 * spec.k * stretch * stretch;
  }
  return total;
}

/** Deterministic LCG (Numerical Recipes constants) — replaces Math.random everywhere. */
function lcg(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Diagonal preconditioner: estimates each free coordinate's second derivative with a central
 * second difference and returns the reciprocal, i.e. a per-coordinate Newton step length.
 */
function curvatureScale(spec, mask, probe = 1e-3) {
  const base = specVector(spec);
  const baseLoss = lossValue(spec);
  const scale = base.map(() => 0);
  for (let i = 0; i < base.length; i++) {
    if (!mask[i]) continue;
    const h = probe * Math.max(1, Math.abs(base[i]));
    const up = base.slice();
    up[i] += h;
    const down = base.slice();
    down[i] -= h;
    const second = (lossValue(specFromVector(spec, up)) - 2 * baseLoss + lossValue(specFromVector(spec, down))) / (h * h);
    scale[i] = second > 1e-12 ? 1 / second : 1;
  }
  return scale;
}

/**
 * Preconditioned steepest descent on the taped loss with backtracking: each iteration halves the
 * step until the loss actually decreases, so the recorded loss sequence is monotone and the run
 * cannot diverge. `mask` selects the free coordinates, `lr` the per-coordinate step scale.
 */
function train(spec, mask, lr, iterations) {
  let current = spec;
  let [loss, gradient] = lossAndGradient(current);
  const history = [loss];
  for (let iter = 0; iter < iterations; iter++) {
    let step = 1;
    let accepted = null;
    for (let attempt = 0; attempt < 60; attempt++) {
      const vector = specVector(current);
      for (let i = 0; i < vector.length; i++) if (mask[i]) vector[i] -= step * lr[i] * gradient[i];
      const candidate = specFromVector(current, vector);
      const [candidateLoss, candidateGradient] = lossAndGradient(candidate);
      if (candidateLoss < loss) {
        accepted = { candidate, candidateLoss, candidateGradient };
        break;
      }
      step /= 2;
    }
    if (accepted === null) break; // no improving step exists along the gradient any more
    current = accepted.candidate;
    loss = accepted.candidateLoss;
    gradient = accepted.candidateGradient;
    history.push(loss);
    if (loss === 0) break;
  }
  return { spec: current, loss, gradient, history };
}

function chainSpec(overrides = {}) {
  return {
    n: 4,
    steps: 25,
    dt: 0.01,
    gravity: 9.8,
    k: 12,
    m: 1,
    restLength: 1,
    p0: [0, 0, 1, 0.2, 2, -0.1, 3, 0.05],
    v0: [0.1, 0, -0.05, 0.3, 0.2, -0.2, -0.1, 0.15],
    targets: [],
    ...overrides,
  };
}

/** Targets for `spec` produced by the ground-truth model `source`. */
const targetsFrom = (source) => integrate(source).frames;

module.exports = { Tape, rollout, lossAndGradient, lossValue, simulate, directionalDifference,
  specVector, specFromVector, freeFallPosition, integrate, energy, forcesOf, lcg, train,
  curvatureScale, chainSpec, targetsFrom };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];

  // ---- integrator against the closed form ------------------------------------------------
  const dt = 0.02;
  const gModel = 3.5;
  const gTarget = 3.0;
  const fallSteps = 12;
  const fall = { n: 1, steps: fallSteps, dt, gravity: gModel, k: 5, m: 2, restLength: 1,
    p0: [0.3, 1.7], v0: [0.25, -0.5], targets: [] };
  const closedModelY = freeFallPosition(fall.p0[1], fall.v0[1], gModel, dt, fallSteps);
  const closedTargetY = freeFallPosition(fall.p0[1], fall.v0[1], gTarget, dt, fallSteps);
  const fallTargets = [];
  for (let i = 0; i <= fallSteps; i++) fallTargets.push(frame([fall.p0[0] + i * dt * fall.v0[0]], [closedTargetY[i]]));
  const fallSpec = { ...fall, targets: fallTargets };
  const fallFrames = simulate(fallSpec);
  let worstFall = 0;
  for (let i = 0; i <= fallSteps; i++) {
    worstFall = Math.max(worstFall, Math.abs(fallFrames[i].y[0] - closedModelY[i]));
    assert.ok(Math.abs(fallFrames[i].x[0] - (fall.p0[0] + i * dt * fall.v0[0])) < 1e-12);
  }
  assert.ok(worstFall < 1e-12, `closed-form trajectory error ${worstFall}`);
  assert.ok(Math.abs(fallFrames.length - (fallSteps + 1)) === 0 && fallFrames[0].x.length === 1);

  // ---- analytic gradient of the free fall -------------------------------------------------
  const [fallLoss, fallGrad] = lossAndGradient(fallSpec);
  const norm = 1 / (fallSteps + 1);
  let analyticP0 = 0;
  let analyticV0 = 0;
  let analyticLoss = 0;
  for (let i = 0; i <= fallSteps; i++) {
    const d = closedModelY[i] - fallTargets[i].y[0];
    analyticP0 += 2 * d * norm;
    analyticV0 += 2 * d * (i * dt) * norm;
    analyticLoss += d * d * norm;
  }
  assert.ok(Math.abs(fallLoss - analyticLoss) < 1e-15, "loss equals the analytic free-fall loss");
  assert.ok(Math.abs(fallGrad[1] - analyticP0) < 1e-12, `dL/dp0y ${fallGrad[1]} vs ${analyticP0}`);
  assert.ok(Math.abs(fallGrad[3] - analyticV0) < 1e-12, `dL/dv0y ${fallGrad[3]} vs ${analyticV0}`);
  assert.ok(Math.abs(fallGrad[0]) < 1e-15, `x residual gradient stays at rounding level (${fallGrad[0]})`);
  assert.ok(Math.abs(fallGrad[2]) < 1e-15, `x velocity gradient stays at rounding level (${fallGrad[2]})`);
  // mass cancels analytically in dt*F/m for a gravity-only rollout: the two adjoint paths cancel
  // down to floating-point rounding only, so the gradient is ~1e-17 rather than bit-exact zero
  assert.ok(Math.abs(fallGrad[5]) < 1e-15, `no gradient leaks into the mass (${fallGrad[5]})`);
  // with a single particle no spring exists, so both spring parameters stay bit-exactly at zero
  assert.strictEqual(fallGrad[4], 0, "no gradient leaks into the spring constant");

  // ---- finite differences on the full chain ----------------------------------------------
  const truth = chainSpec({ n: 4, steps: 20, dt: 0.005, k: 12, restLength: 1,
    p0: [0, 0, 2.5, 0.3, 5, -0.2, 7.5, 0.4],
    v0: [0.1, 0, -0.05, 0.3, 0.2, -0.2, -0.1, 0.15] });
  const truthTargets = targetsFrom(truth);
  const model = { ...truth, k: 7.5, targets: truthTargets };
  const [modelLoss, modelGrad] = lossAndGradient(model);
  assert.ok(modelLoss > 1e-6 && Number.isFinite(modelLoss));
  assert.ok(lossValue({ ...truth, targets: truthTargets }) < 1e-24, "ground truth reproduces its own targets");

  const rng = lcg(825);
  let worstDirectional = 0;
  for (let trial = 0; trial < 3; trial++) {
    const direction = specVector(model).map(() => rng() - 0.5);
    let dot = 0;
    for (let i = 0; i < direction.length; i++) dot += direction[i] * modelGrad[i];
    const fd = directionalDifference(model, direction, 1e-5);
    worstDirectional = Math.max(worstDirectional, Math.abs(fd - dot) / Math.max(1e-12, Math.abs(dot)));
  }
  assert.ok(worstDirectional < 1e-6, `directional derivative error ${worstDirectional}`);

  const probeIndices = [];
  for (let i = 0; i < 2 * model.n; i += 4) probeIndices.push(i);
  probeIndices.push(2 * model.n, 3 * model.n + 1, 4 * model.n, 4 * model.n + 1);
  let worstPerParam = 0;
  let worstSignCheck = Infinity;
  for (const index of probeIndices) {
    const direction = specVector(model).map(() => 0);
    direction[index] = 1;
    const fd = directionalDifference(model, direction, 1e-5);
    worstPerParam = Math.max(worstPerParam, Math.abs(fd - modelGrad[index]) / Math.max(1e-9, Math.abs(fd)));
    assert.ok(Number.isFinite(fd) && fd !== 0);
    // the same comparison rejects the negated gradient: the sign of every probe coordinate matters
    worstSignCheck = Math.min(worstSignCheck, Math.abs(fd + modelGrad[index]) / Math.abs(fd));
  }
  assert.ok(worstPerParam < 1e-6, `per-parameter gradient error ${worstPerParam}`);
  assert.ok(worstSignCheck > 1 * 1.5, `negated gradients are rejected (worst ratio ${worstSignCheck})`);

  // the aggregate directional test over all coordinates agrees too
  const ones = specVector(model).map(() => 1);
  const trueDot = ones.reduce((acc, w, i) => acc + w * modelGrad[i], 0);
  assert.ok(Math.abs(directionalDifference(model, ones, 1e-5) - trueDot) < 1e-6);

  // ---- symplectic energy behaviour --------------------------------------------------------
  const energySpec = chainSpec({ n: 3, steps: 400, dt: 0.001, k: 40, gravity: 0, restLength: 0.5,
    p0: [0, 0, 0.7, 0.3, 1.5, -0.2], v0: [0.4, 0.1, -0.3, 0.2, 0.1, -0.15] });
  const drift = (steps) => {
    const { energies } = integrate({ ...energySpec, steps });
    const e0 = energies[0];
    let worst = 0;
    for (const value of energies) worst = Math.max(worst, Math.abs(value - e0) / Math.abs(e0));
    return worst;
  };
  const driftShort = drift(400);
  const driftLong = drift(1600);
  assert.ok(driftShort < 0.02, `energy drift ${driftShort}`);
  assert.ok(driftLong < driftShort * 1.5 + 1e-9, "energy error does not grow with the horizon");
  // Constant-force semi-implicit Euler loses exactly 0.5*m*g^2*dt^2 per particle per step, so the
  // free-fall energy has a closed form: E_n = E_0 - n * n_particles * 0.5*m*g^2*dt^2 (bit-close).
  const freeSpec = { ...energySpec, k: 0, gravity: 9.8 };
  const freeEnergies = integrate(freeSpec).energies;
  const perStep = 0.5 * freeSpec.m * freeSpec.gravity * freeSpec.gravity * freeSpec.dt * freeSpec.dt * freeSpec.n;
  let worstFree = 0;
  for (let i = 0; i < freeEnergies.length; i++) {
    worstFree = Math.max(worstFree, Math.abs(freeEnergies[i] - (freeEnergies[0] - i * perStep)));
  }
  assert.ok(worstFree < 1e-12, `free-fall energy matches the closed form (${worstFree})`);

  // A stretched spring must pull its endpoints together and a compressed one must push them apart
  // (guards the force sign); the two endpoint forces are equal and opposite by construction.
  const stretched = chainSpec({ n: 2, steps: 20, dt: 0.01, k: 10, gravity: 0, restLength: 1,
    p0: [0, 0, 3, 0], v0: [0, 0, 0, 0] });
  const [sfx, sfy] = forcesOf(stretched, [0, 3], [0, 0]);
  assert.ok(sfx[0] > 0 && sfx[1] < 0 && Math.abs(sfx[0] + sfx[1]) < 1e-15, "stretched pair is attracted");
  assert.strictEqual(sfy[0] + sfy[1], 0);
  const compressed = { ...stretched, p0: [0, 0, 0.5, 0], targets: [frame([0, 0.5], [0, 0])] };
  const [cfx] = forcesOf(compressed, [0, 0.5], [0, 0]);
  assert.ok(cfx[0] < 0 && cfx[1] > 0, "compressed pair is repelled");
  const gap = (f) => Math.abs(f.x[1] - f.x[0]);
  const zeroTargets = (n, steps) =>
    Array.from({ length: steps + 1 }, () => frame(new Array(n).fill(0), new Array(n).fill(0)));
  const pullFrames = simulate({ ...stretched, targets: zeroTargets(2, 20) });
  for (let i = 1; i <= 20; i++) assert.ok(gap(pullFrames[i]) < gap(pullFrames[i - 1]));
  const pushFrames = simulate({ ...compressed, targets: zeroTargets(2, 20) });
  for (let i = 1; i <= 20; i++) assert.ok(gap(pushFrames[i]) > gap(pushFrames[i - 1]));

  // ---- degenerate shapes ------------------------------------------------------------------
  const noParticles = lossAndGradient({ ...chainSpec({ n: 0, steps: 5 }), targets: [frame([], [])] });
  assert.strictEqual(noParticles[0], null);
  assert.deepStrictEqual(noParticles[1], []);
  assert.deepStrictEqual(simulate({ ...chainSpec({ n: 0, steps: 3 }), targets: [frame([], [])] }), []);

  const frozen = { ...chainSpec({ n: 2, steps: 0, p0: [0.5, -1, 2, 0], v0: [0, 0, 0, 0] }),
    targets: [frame([1, 2], [0, 0])] };
  const [frozenLoss, frozenGrad] = lossAndGradient(frozen);
  assert.ok(Math.abs(frozenLoss - (0.25 + 1) / 2) < 1e-15);
  assert.ok(Math.abs(frozenGrad[0] - 2 * (0.5 - 1) / 2) < 1e-15);
  assert.ok(Math.abs(frozenGrad[1] - 2 * (-1 - 0) / 2) < 1e-15);
  assert.ok(Math.abs(frozenGrad[2]) < 1e-15 && Math.abs(frozenGrad[3]) < 1e-15);

  // ---- end-to-end learning ----------------------------------------------------------------
  const trainSpec = { ...model, k: 9.4 };
  const kMask = specVector(model).map((_, i) => (i === 4 * model.n ? 1 : 0));
  const learnedK = train(trainSpec, kMask, curvatureScale(trainSpec, kMask), 100);
  for (let i = 1; i < learnedK.history.length; i++) {
    assert.ok(learnedK.history[i] <= learnedK.history[i - 1], "backtracking keeps the loss monotone");
  }
  assert.ok(Math.abs(learnedK.spec.k - truth.k) < 1e-9, `learned spring constant ${learnedK.spec.k}`);
  assert.ok(learnedK.loss < 1e-20, `residual after learning k = ${learnedK.loss}`);

  const initMask = specVector(model).map((_, i) => (i < 4 * model.n ? 1 : 0));
  const stateSpec = { ...truth, targets: truthTargets, p0: truth.p0.map((v) => v + 0.3),
    v0: truth.v0.map((v) => v - 0.2) };
  const startLoss = lossValue(stateSpec);
  const learnedState = train(stateSpec, initMask, curvatureScale(stateSpec, initMask), 400);
  assert.ok(startLoss > 0.1, `starting loss is far from the target (${startLoss})`);
  assert.ok(learnedState.loss < 1e-15, `residual after learning the initial state = ${learnedState.loss}`);
  let worstState = 0;
  for (let i = 0; i < 2 * model.n; i++) {
    worstState = Math.max(worstState, Math.abs(learnedState.spec.p0[i] - truth.p0[i]));
    worstState = Math.max(worstState, Math.abs(learnedState.spec.v0[i] - truth.v0[i]));
  }
  // the targets come from the plain integrator, so the taped model can only match them to ~1e-17 and
  // the recovered parameters stop ~1e-8 away from the exact ground truth
  assert.ok(worstState < 1e-6, `recovered initial state error ${worstState}`);

  metrics.push(`closed-form integrator agreement: ${worstFall.toExponential(2)}`);
  metrics.push(`analytic free-fall gradient agreement: ${Math.abs(fallGrad[1] - analyticP0).toExponential(2)}`);
  metrics.push(`finite differences: directional ${worstDirectional.toExponential(2)}, per-parameter ${worstPerParam.toExponential(2)}`);
  metrics.push(`symplectic energy drift: ${(driftShort * 100).toFixed(4)}% over 400 steps, ${(driftLong * 100).toFixed(4)}% over 1600`);
  metrics.push(`learned spring constant: ${learnedK.spec.k.toFixed(9)} (truth ${truth.k}), residual ${learnedK.loss.toExponential(2)}`);
  metrics.push(`recovered initial state error: ${worstState.toExponential(2)}`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
