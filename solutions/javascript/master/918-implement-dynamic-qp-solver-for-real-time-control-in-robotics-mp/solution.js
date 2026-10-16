/** Task 918 - Implement dynamic QP solver for real-time control in robotics (MPC small demo).
 *  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Plant: double integrator (position/velocity state, acceleration input). Each tick builds the dense
 * box QP  min 0.5 z'Hz + g'z  over the horizon accelerations z; H and g come from the closed-form
 * sensitivity of each predicted state to each control (the posCoef/velCoef tables). Solver: FISTA
 * with monotone restart, step 1/L from the Gershgorin bound; z* is optimal iff the projected fixed
 * point z* = proj_[lo,hi](z* - (1/L)(Hz*+g)) holds, which is the residual asserted here.
 * Simplified vs production MPC: scalar control, LTI plant, box bounds only, no terminal set and no
 * warm start. Verified: KKT optimality vs a 3^n active-set oracle, rollout feasibility, actuator
 * limits, closed-loop convergence, determinism.
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
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const matVec = (M, v) => v.map((_, i) => v.reduce((a, x, j) => a + M[i][j] * x, 0));
const quadratic = (H, g, z) => {
  const Hz = matVec(H, z);
  return z.reduce((a, x, i) => a + g[i] * x + 0.5 * x * Hz[i], 0);
};
/** Gershgorin disc bound on lambda_max(H): the largest absolute row sum. */
const gershgorinBound = (H) =>
  H.reduce((w, row) => Math.max(w, row.reduce((s, v) => s + Math.abs(v), 0)), 0);

/** Gauss-Jordan with partial pivoting; returns null when singular. Row i's pivot sits at row[i]. */
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    if (Math.abs(M[pivot][col]) < 1e-12) return null;
    const swap = M[col]; M[col] = M[pivot]; M[pivot] = swap;
    for (let r = 0; r < n; r++) if (r !== col) {
      const factor = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

/** min 0.5 z'Hz + g'z over lo <= z <= hi; returns z, the KKT residual and the objective trace. */
function solveBoxQp(H, g, lo, hi, options = {}) {
  const n = g.length;
  assert.strictEqual(H.length, n, "H and g must agree in dimension");
  assert.ok(H.every((row) => row.length === n), "H must be square");
  assert.ok(lo.every((v, i) => v <= hi[i]), "invalid box: lo > hi");
  if (n === 0) return { z: [], objective: 0, kkt: 0, iterations: 0, L: 0, trace: [] };
  const L = options.L ?? gershgorinBound(H);
  assert.ok(L > 0, "Q must be positive definite: the Gershgorin bound is zero");
  const step = 1 / L;
  const gradientStep = (point) => {
    const Hp = matVec(H, point);
    return point.map((v, i) => clamp(v - step * (Hp[i] + g[i]), lo[i], hi[i]));
  };
  let z = gradientStep(new Array(n).fill(0));
  let best = quadratic(H, g, z), y = z.slice(), t = 1, iterations = 0;
  const trace = [best];
  for (; iterations < (options.maxIter ?? 4000); iterations++) {
    const candidate = gradientStep(y);
    const candidateObjective = quadratic(H, g, candidate);
    if (candidateObjective > best - 1e-15) { // momentum overshot: restart from a plain step
      const plain = gradientStep(z), plainObjective = quadratic(H, g, plain);
      if (plainObjective >= best - 1e-15) break; // optimal up to machine precision
      z = plain; y = plain.slice(); t = 1; best = plainObjective; trace.push(best);
      continue;
    }
    const tNext = 0.5 * (1 + Math.sqrt(1 + 4 * t * t)), momentum = (t - 1) / tNext;
    y = candidate.map((v, i) => v + momentum * (v - z[i]));
    z = candidate; t = tNext; best = candidateObjective; trace.push(best);
  }
  z = gradientStep(z);
  const kkt = gradientStep(z).reduce((w, v, i) => Math.max(w, Math.abs(z[i] - v)), 0);
  return { z, objective: quadratic(H, g, z), kkt, iterations, L, trace };
}

/**
 * Independent oracle: enumerate all 3^n patterns (coordinate free / pinned at lo / pinned at hi),
 * solve the stationarity system exactly on the free block, drop infeasible candidates, keep the best.
 */
function bruteForceBoxQp(H, g, lo, hi) {
  const n = g.length;
  let bestZ = null, bestObjective = Infinity, candidates = 0;
  for (let code = 0; code < 3 ** n; code++) {
    const z = new Array(n), free = [], pinned = [];
    let rest = code;
    for (let i = 0; i < n; i++) {
      const status = rest % 3;
      rest = (rest - status) / 3;
      if (status === 0) free.push(i);
      else { z[i] = status === 1 ? lo[i] : hi[i]; pinned.push(i); }
    }
    if (free.length > 0) { // stationarity: H_FF z_F = -(g_F + H_FB z_B)
      const A = free.map((i) => free.map((j) => H[i][j]));
      const b = free.map((i) => -(g[i] + pinned.reduce((a, j) => a + H[i][j] * z[j], 0)));
      const stationarity = solveLinear(A, b);
      if (stationarity === null) return null; // H not definite: the oracle does not apply
      free.forEach((i, k) => { z[i] = stationarity[k]; });
    }
    if (z.some((v, i) => v < lo[i] - 1e-9 || v > hi[i] + 1e-9)) continue;
    candidates++;
    const objective = quadratic(H, g, z);
    if (objective < bestObjective) { bestObjective = objective; bestZ = z.slice(); }
  }
  return { z: bestZ, objective: bestObjective, candidates };
}

const DT = 0.2;
const Q_POS = 1.0, Q_VEL = 0.1, Q_TERM_POS = 2.0, Q_TERM_VEL = 0.2, R_CTRL = 0.05;

/** Dense QP for one MPC solve; the decision variables are the horizon accelerations. */
function buildMpcQp(state, setpoint, horizon, uMax) {
  const n = horizon;
  const posOff = new Array(n + 1).fill(0), velOff = new Array(n + 1).fill(0);
  const posCoef = Array.from({ length: n + 1 }, () => new Array(n).fill(0));
  const velCoef = Array.from({ length: n + 1 }, () => new Array(n).fill(0));
  posOff[0] = state.position;
  velOff[0] = state.velocity;
  for (let k = 0; k < n; k++) { // p' = p + dt*v + dt^2*u/2, v' = v + dt*u; u_k acts at step k
    posOff[k + 1] = posOff[k] + DT * velOff[k];
    velOff[k + 1] = velOff[k];
    for (let j = 0; j < n; j++) {
      posCoef[k + 1][j] = posCoef[k][j] + DT * velCoef[k][j] + (j === k ? 0.5 * DT * DT : 0);
      velCoef[k + 1][j] = velCoef[k][j] + (j === k ? DT : 0);
    }
  }
  const H = Array.from({ length: n }, () => new Array(n).fill(0));
  const g = new Array(n).fill(0);
  for (let k = 1; k <= n; k++) { // step 0 depends on no control, so it only shifts g
    const wPos = k === n ? Q_TERM_POS : Q_POS, wVel = k === n ? Q_TERM_VEL : Q_VEL;
    const posErr = posOff[k] - setpoint.position, velErr = velOff[k] - setpoint.velocity;
    for (let i = 0; i < n; i++) {
      g[i] += 2 * (wPos * posErr * posCoef[k][i] + wVel * velErr * velCoef[k][i]);
      for (let j = 0; j < n; j++) {
        H[i][j] += 2 * (wPos * posCoef[k][i] * posCoef[k][j] + wVel * velCoef[k][i] * velCoef[k][j]);
      }
    }
  }
  for (let i = 0; i < n; i++) H[i][i] += 2 * R_CTRL;
  return { H, g, lo: new Array(n).fill(-uMax), hi: new Array(n).fill(uMax) };
}
function rollOut(state, controls) {
  const states = [{ ...state }];
  for (const u of controls) {
    const last = states[states.length - 1];
    states.push({ position: last.position + DT * last.velocity + 0.5 * DT * DT * u,
      velocity: last.velocity + DT * u });
  }
  return states;
}
function mpcSolve(state, setpoint, options = {}) {
  const qp = buildMpcQp(state, setpoint, options.horizon ?? 10, options.uMax ?? 1.5);
  return { ...solveBoxQp(qp.H, qp.g, qp.lo, qp.hi, options), qp };
}

/** Receding-horizon loop against the exact plant; optional seeded measurement noise. */
function runClosedLoop(initial, setpoint, steps, options = {}) {
  const { horizon = 10, uMax = 1.5, noise = null } = options;
  const controls = [];
  const states = [{ ...initial }];
  for (let k = 0; k < steps; k++) {
    const previous = states[states.length - 1];
    const measured = noise ? { position: previous.position + noise.sym(0.002),
      velocity: previous.velocity + noise.sym(0.002) } : previous;
    controls.push(mpcSolve(measured, setpoint, { horizon, uMax }).z[0]);
    states.push(rollOut(previous, [controls[k]])[1]);
  }
  return { controls, states };
}

/** Diagonally dominant, hence positive definite, random QP instance. */
function randomSpdQp(prng, n) {
  const H = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) H[i][j] = H[j][i] = prng.sym(1);
    H[i][i] += n + 1;
  }
  const g = Array.from({ length: n }, () => prng.sym(2));
  const lo = Array.from({ length: n }, () => prng.sym(0.6) - 0.5);
  return { H, g, lo, hi: lo.map((v) => v + 0.2 + prng.float()) };
}

function main() {
  const setpoint = { position: 1.0, velocity: 0.0 };
  let worstKkt = 0, oracleGap = 0;

  // 1. KKT optimality and box feasibility, cross-checked against the active-set oracle.
  const prng = new Prng(20250918);
  for (let trial = 0; trial < 20; trial++) {
    const n = 2 + (trial % 4);
    const { H, g, lo, hi } = randomSpdQp(prng, n);
    const solved = solveBoxQp(H, g, lo, hi);
    worstKkt = Math.max(worstKkt, solved.kkt);
    assert.ok(solved.kkt < 1e-8, `KKT residual ${solved.kkt} too large on trial ${trial}`);
    const oracle = bruteForceBoxQp(H, g, lo, hi);
    assert.ok(oracle && oracle.z && oracle.candidates > 0, "oracle found no feasible candidate");
    assert.ok(solved.z.every((v, i) =>
      v >= lo[i] - 1e-9 && v <= hi[i] + 1e-9 && Math.abs(v - oracle.z[i]) < 1e-6),
      "solution is infeasible or differs from the oracle");
    oracleGap = Math.max(oracleGap, solved.objective - oracle.objective);
    assert.ok(solved.objective <= oracle.objective + 1e-7, "solver worse than the oracle");
  }

  // 2. With inactive bounds the QP degenerates to the linear solve Hz = -g.
  const wide = randomSpdQp(new Prng(90210), 4);
  const free = solveBoxQp(wide.H, wide.g, new Array(4).fill(-1e4), new Array(4).fill(1e4));
  const exact = solveLinear(wide.H, wide.g.map((v) => -v));
  assert.ok(free.z.every((v, i) => Math.abs(v - exact[i]) < 1e-6), "unconstrained solve differs");
  assert.ok(free.z.every((v) => Math.abs(v) < 1e4), "unconstrained solution hit the box");

  // 3. Boundary cases: empty program, zero-width box, already optimal interior point.
  assert.deepStrictEqual(solveBoxQp([], [], [], []).z, []);
  const pinned = solveBoxQp([[4, 1], [1, 3]], [-2, -2], [0.5, -1], [0.5, -1]);
  assert.ok(Math.abs(pinned.z[0] - 0.5) < 1e-12 && Math.abs(pinned.z[1] + 1) < 1e-12,
    "a zero-width box must return the pinned point");
  assert.ok(solveBoxQp([[2, 0], [0, 2]], [0, 0], [-1, -1], [1, 1]).z.every((v) => Math.abs(v) < 1e-9),
    "the interior optimum must be zero");

  // 4. The accepted objective never increases: the monotone restart is doing its job.
  const { H: Hm, g: gm, lo: lom, hi: him } = randomSpdQp(new Prng(31337), 6);
  const { trace } = solveBoxQp(Hm, gm, lom, him, { maxIter: 300 });
  assert.ok(trace.length > 1, "expected several accepted steps");
  assert.ok(trace.every((v, k) => k === 0 || v <= trace[k - 1] + 1e-12), "objective increased");

  // 5. The MPC solve respects actuator limits, rolls out feasibly and beats standing still.
  const robot = { position: -0.4, velocity: 0.3 };
  const plan = mpcSolve(robot, setpoint, { horizon: 8, uMax: 1.2 });
  const predicted = rollOut(robot, plan.z);
  assert.ok(plan.z.every((u) => Math.abs(u) <= 1.2 + 1e-9), "control exceeds the actuator bound");
  assert.strictEqual(predicted.length, plan.z.length + 1);
  assert.ok(predicted[predicted.length - 1].position > robot.position, "prediction moves backwards");
  assert.ok(plan.objective < quadratic(plan.qp.H, plan.qp.g, plan.qp.hi.map(() => 0)) - 1e-6,
    "control did not improve the MPC cost");

  // 6. Closed-loop stability: the setpoint is reached, held and within the actuator limit.
  const closed = runClosedLoop({ position: 0, velocity: 0 }, setpoint, 40, { horizon: 10 });
  const final = closed.states[closed.states.length - 1];
  assert.ok(Math.abs(final.position - setpoint.position) < 0.02, `position ${final.position} missed`);
  assert.ok(Math.abs(final.velocity) < 0.05, `velocity ${final.velocity} did not settle`);
  assert.ok(closed.controls.every((u) => Math.abs(u) <= 1.5 + 1e-9));

  // 7. Invalid input is rejected rather than silently mis-solved.
  assert.throws(() => solveBoxQp([[1, 0], [0, 1]], [0, 0], [1, 0], [0, 1]), /invalid box/);
  assert.throws(() => solveBoxQp([[0, 0], [0, 0]], [1, 1], [-1, -1], [1, 1]), /positive definite/);
  assert.throws(() => solveBoxQp([[1, 0]], [0, 0], [-1], [1]), /dimension/);
  assert.throws(() => solveBoxQp([[1, 0], [0]], [0, 0], [-1, -1], [1, 1]), /square/);

  // 8. Determinism: identical seeds replay identically; seeded noise perturbs but stays bounded.
  const loop = (noise) => runClosedLoop({ position: 0, velocity: 0 }, setpoint, 15,
    { horizon: 6, noise }).controls;
  assert.deepStrictEqual(loop(null), loop(null));
  assert.deepStrictEqual(loop(new Prng(5)), loop(new Prng(5)));
  assert.notDeepStrictEqual(loop(new Prng(5)), loop(null));
  assert.ok(loop(new Prng(5)).every((u) => Math.abs(u) <= 1.5 + 1e-9));
  console.log(`qp: worst KKT residual ${worstKkt.toExponential(2)}, worst gap vs active-set oracle ` +
    `${oracleGap.toExponential(2)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, solveBoxQp, bruteForceBoxQp, buildMpcQp, rollOut, mpcSolve, runClosedLoop,
  gershgorinBound, solveLinear, quadratic, randomSpdQp };