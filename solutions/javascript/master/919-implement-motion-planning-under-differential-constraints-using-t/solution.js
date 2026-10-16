/** Task 919 - Implement motion planning under differential constraints using trajectory optimization.
 *  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * The differential constraint is a unicycle / differential-drive robot: x' = v cos(theta),
 * y' = v sin(theta), theta' = omega, with box bounds on v and omega, integrated by explicit Euler
 * (dt = 0.2). Planning is direct single-shooting trajectory optimization: the decision variables are
 * the N = 12 controls (v_k, omega_k), the cost is  quadratic goal distance + control effort +
 * a soft obstacle penalty `wObs * max(0, r+margin-d)^2`, and the gradient is obtained by an exact
 * adjoint (backward) pass over the rollout. Descent is projected gradient with backtracking.
 * Simplification: local optimisation only (no homotopy / no global guarantee), circular obstacles,
 * fixed timestep. Verified: agreement with a closed-form optimum and with an exhaustive control-grid
 * search, dynamics feasibility, control bounds, collision freedom, boundary cases and determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() { // xorshift32
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17; x ^= x << 5;
    return (this.state = x >>> 0);
  }
  float() { return this.next() / 4294967296; }
  sym(s) { return (this.float() * 2 - 1) * s; }
}

const DT = 0.2;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

function makeProblem(overrides = {}) {
  return {
    start: { x: 0, y: 0, theta: 0 },
    goal: { x: 2, y: 0 },
    obstacles: [],
    vMax: 1.5, omegaMax: 1.2,
    wGoal: 50, wCtrl: 1, wObs: 20, margin: 0.15,
    ...overrides,
  };
}

function validate(problem, horizon) {
  assert.ok(Number.isInteger(horizon) && horizon > 0, "horizon must be a positive integer");
  assert.ok(problem.vMax >= 0 && problem.omegaMax >= 0, "actuator limits must be non-negative");
  assert.ok(Number.isFinite(problem.goal.x) && Number.isFinite(problem.goal.y), "goal must be finite");
  for (const o of problem.obstacles) assert.ok(o.r >= 0, "obstacle radius must be non-negative");
}

function clampControl(problem, control) {
  return { v: clamp(control.v, 0, problem.vMax), omega: clamp(control.omega, -problem.omegaMax, problem.omegaMax) };
}

/** Explicit-Euler rollout: x' = v cos(theta), y' = v sin(theta), theta' = omega. */
function rollout(start, controls) {
  const states = [{ ...start }];
  for (const c of controls) {
    const s = states[states.length - 1];
    states.push({ x: s.x + DT * c.v * Math.cos(s.theta), y: s.y + DT * c.v * Math.sin(s.theta),
      theta: s.theta + DT * c.omega });
  }
  return states;
}

/** Minimum clearance to any obstacle (negative means penetration) and the soft penalty value. */
function obstacleStats(problem, states) {
  let clearance = Infinity, penalty = 0;
  for (const s of states) {
    for (const o of problem.obstacles) {
      const d = Math.hypot(s.x - o.x, s.y - o.y);
      clearance = Math.min(clearance, d - o.r);
      const slack = o.r + problem.margin - d;
      if (slack > 0) penalty += problem.wObs * slack * slack;
    }
  }
  return { clearance, penalty };
}

function totalCost(problem, states, controls) {
  const last = states[states.length - 1];
  let cost = problem.wGoal * ((last.x - problem.goal.x) ** 2 + (last.y - problem.goal.y) ** 2);
  for (const c of controls) cost += problem.wCtrl * (c.v * c.v + c.omega * c.omega);
  return cost + obstacleStats(problem, states).penalty;
}

/** Exact adjoint pass: returns dJ/d(v_k) and dJ/d(omega_k) for every step and the rollout. */
function costGradient(problem, controls) {
  const states = rollout(problem.start, controls);
  const n = controls.length;
  const grad = controls.map(() => ({ v: 0, omega: 0 }));
  const last = states[n];
  let px = 2 * problem.wGoal * (last.x - problem.goal.x);
  let py = 2 * problem.wGoal * (last.y - problem.goal.y);
  let pTheta = 0;
  for (let k = n - 1; k >= 0; k--) {
    const s = states[k], c = controls[k];
    const cos = Math.cos(s.theta), sin = Math.sin(s.theta);
    grad[k].v = px * DT * cos + py * DT * sin + 2 * problem.wCtrl * c.v;
    grad[k].omega = pTheta * DT + 2 * problem.wCtrl * c.omega;
    // Transposed state Jacobian: (x_k, y_k) pass through, theta_k steers the next position.
    let nextPx = px, nextPy = py;
    let nextPTheta = pTheta - px * DT * c.v * sin + py * DT * c.v * cos;
    for (const o of problem.obstacles) { // gradient of the soft penalty evaluated at state k
      const dx = s.x - o.x, dy = s.y - o.y;
      const d = Math.hypot(dx, dy);
      const slack = o.r + problem.margin - d;
      if (slack > 0 && d > 1e-9) {
        nextPx += -2 * problem.wObs * slack * dx / d;
        nextPy += -2 * problem.wObs * slack * dy / d;
      }
    }
    px = nextPx; py = nextPy; pTheta = nextPTheta;
  }
  return { states, grad };
}

function initialGuess(problem, horizon, prng) {
  const controls = [];
  const bearing = Math.atan2(problem.goal.y - problem.start.y, problem.goal.x - problem.start.x);
  for (let k = 0; k < horizon; k++) {
    const steer = prng ? prng.sym(0.05) : 0;
    controls.push(clampControl(problem, { v: problem.vMax * 0.6 + steer, omega: steeringBias(problem, bearing) }));
  }
  return controls;
}

function steeringBias(problem, bearing) {
  const error = Math.atan2(Math.sin(bearing - problem.start.theta), Math.cos(bearing - problem.start.theta));
  return clamp(error / (DT * 5), -problem.omegaMax, problem.omegaMax);
}

/** Projected gradient descent with backtracking; keeps the best accepted cost trace. */
function optimize(problem, options = {}) {
  const horizon = options.horizon ?? 12;
  validate(problem, horizon);
  let controls = initialGuess(problem, horizon, options.prng);
  let states = rollout(problem.start, controls);
  let cost = totalCost(problem, states, controls);
  const trace = [cost];
  let iterations = 0;
  for (; iterations < (options.maxIter ?? 300); iterations++) {
    const { grad } = costGradient(problem, controls);
    if (grad.every((g) => g.v ** 2 + g.omega ** 2 < 1e-14)) break;
    let step = options.step ?? 0.5, accepted = false;
    for (let back = 0; back < 30; back++) {
      const trial = controls.map((c, k) => clampControl(problem,
        { v: c.v - step * grad[k].v, omega: c.omega - step * grad[k].omega }));
      const trialStates = rollout(problem.start, trial);
      const trialCost = totalCost(problem, trialStates, trial);
      if (trialCost < cost - 1e-12) {
        controls = trial; states = trialStates; cost = trialCost; accepted = true;
        break;
      }
      step *= 0.5;
    }
    trace.push(cost);
    if (!accepted) break; // first-order stationary point reached
  }
  const last = states[states.length - 1];
  return {
    controls, states, cost, iterations, trace,
    goalDistance: Math.hypot(last.x - problem.goal.x, last.y - problem.goal.y),
    ...obstacleStats(problem, states),
  };
}

/** Independent oracle: exhaustive search over a coarse control grid on a short horizon. */
function bruteForcePlan(problem, horizon, vLevels, omegaLevels) {
  const combos = [];
  for (const v of vLevels) for (const omega of omegaLevels) combos.push({ v, omega });
  let bestCost = Infinity, bestControls = null, plans = 0;
  for (let code = 0; code < combos.length ** horizon; code++) {
    const controls = [];
    let rest = code;
    for (let k = 0; k < horizon; k++) {
      controls.push(combos[rest % combos.length]);
      rest = Math.floor(rest / combos.length);
    }
    const cost = totalCost(problem, rollout(problem.start, controls), controls);
    plans++;
    if (cost < bestCost) { bestCost = cost; bestControls = controls; }
  }
  return { cost: bestCost, controls: bestControls, plans };
}

/** Closed-form optimum for an obstacle-free straight-ahead goal with an active effort penalty. */
function straightLineOptimum(problem, horizon) {
  const distance = problem.goal.x - problem.start.x;
  const sumV = problem.wGoal * (2 * DT * distance) / (2 * DT * DT * problem.wGoal + 2 * problem.wCtrl / horizon);
  const v = sumV / horizon;
  const reach = DT * sumV;
  const cost = problem.wGoal * (reach - distance) ** 2 + problem.wCtrl * horizon * v * v;
  return { v, sumV, cost, reach };
}

function main() {
  // 1. Closed-form oracle: obstacle-free straight-ahead goal, effort-weighted optimum.
  const straight = makeProblem();
  const analytic = straightLineOptimum(straight, 12);
  const planned = optimize(straight, { horizon: 12 });
  assert.ok(Math.abs(planned.cost - analytic.cost) < 1e-3,
    `cost ${planned.cost} vs closed form ${analytic.cost}`);
  assert.ok(planned.controls.every((c) => Math.abs(c.v - analytic.v) < 1e-3), "v_k differs from v*");
  assert.ok(planned.controls.every((c) => Math.abs(c.omega) < 1e-6), "steering is not zero");
  assert.ok(planned.states.every((s) => Math.abs(s.y) < 1e-9), "trajectory drifted off the axis");

  // 2. Independent oracle: the continuous optimum cannot be worse than any grid plan.
  const grid = bruteForcePlan(makeProblem(), 4, [0, 0.75, 1.5], [-1.2, 0, 1.2]);
  const shortPlan = optimize(makeProblem(), { horizon: 4 });
  assert.ok(grid.plans === 9 ** 4 && grid.controls, "the grid oracle must enumerate 9^4 plans");
  assert.ok(shortPlan.cost <= grid.cost + 1e-9,
    `optimiser ${shortPlan.cost} worse than grid best ${grid.cost}`);

  // 3. Happy path with an obstacle on the straight line: goal reached, actuator limits respected.
  const cluttered = makeProblem({
    goal: { x: 2, y: 1.2 },
    start: { x: 0, y: 0, theta: 0 },
    obstacles: [{ x: 1.0, y: 0.55, r: 0.25 }],
  });
  const detour = optimize(cluttered, { horizon: 14 });
  assert.ok(detour.goalDistance < 0.2, `goal distance ${detour.goalDistance} too large`);
  assert.ok(detour.controls.every((c) => c.v >= 0 && c.v <= cluttered.vMax + 1e-12
    && Math.abs(c.omega) <= cluttered.omegaMax + 1e-12), "control outside the actuator box");
  assert.ok(detour.clearance > 0, `the plan collides (clearance ${detour.clearance})`);

  // 4. Dynamics feasibility: replaying the returned controls reproduces the returned states.
  const replay = rollout(cluttered.start, detour.controls);
  assert.ok(replay.every((s, k) => Math.abs(s.x - detour.states[k].x) < 1e-12
    && Math.abs(s.theta - detour.states[k].theta) < 1e-12), "rollout is not reproducible");

  // 5. Boundary case: the robot already sits on the goal, so doing nothing is optimal.
  const atGoal = makeProblem({ start: { x: 2, y: 0, theta: 0.3 } });
  const idle = optimize(atGoal, { horizon: 6 });
  assert.ok(idle.cost < 1e-9, `expected near-zero cost, got ${idle.cost}`);
  assert.ok(idle.controls.every((c) => Math.abs(c.v) < 1e-6 && Math.abs(c.omega) < 1e-6),
    "the idle plan should ask for no motion");

  // 6. Monotone descent: every accepted iterate has a smaller cost.
  assert.ok(detour.trace.length > 2, "expected several accepted steps");
  assert.ok(detour.trace.every((v, k) => k === 0 || v < detour.trace[k - 1] + 1e-12),
    "the cost trace increased");

  // 7. Higher obstacle weight buys more clearance (same start, same obstacles, same guess).
  const weak = optimize(cluttered, { horizon: 14, prng: new Prng(7) });
  const strong = optimize(makeProblem({ ...cluttered, wObs: 400 }), { horizon: 14, prng: new Prng(7) });
  assert.ok(strong.clearance >= weak.clearance - 1e-9,
    `strong penalty clearance ${strong.clearance} < weak ${weak.clearance}`);

  // 8. Invalid input is rejected instead of silently mis-solved.
  assert.throws(() => optimize(straight, { horizon: 0 }), /positive integer/);
  assert.throws(() => optimize(makeProblem({ vMax: -1 })), /non-negative/);
  assert.throws(() => optimize(makeProblem({ obstacles: [{ x: 0, y: 0, r: -1 }] })), /non-negative/);

  // 9. Determinism: identical seeds replay identically, and noisy guesses still converge.
  const a = optimize(cluttered, { horizon: 14, prng: new Prng(11) });
  const b = optimize(cluttered, { horizon: 14, prng: new Prng(11) });
  assert.deepStrictEqual(a.controls, b.controls);
  const noisy = optimize(cluttered, { horizon: 14, prng: new Prng(12) });
  assert.ok(noisy.goalDistance < 0.2, "a perturbed initial guess failed to converge");
  console.log(`trajectory: closed-form cost ${analytic.cost.toFixed(6)} vs planned ` +
    `${planned.cost.toFixed(6)}, detour clearance ${detour.clearance.toFixed(4)}, ` +
    `${detour.iterations} accepted iterations`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, makeProblem, rollout, obstacleStats, totalCost, costGradient, optimize,
  bruteForcePlan, straightLineOptimum, clampControl };