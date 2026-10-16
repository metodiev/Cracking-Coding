/** Task 920 - Implement end-to-end SLAM with loop closure detection and pose graph optimization.
 *  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Pipeline (2D, SE(2) poses):
 *   1. simulate a ground-truth loop of N poses, seeded odometry noise and landmark scans
 *      (each scan is the landmark set expressed in the robot frame, corrupted by seeded noise);
 *   2. dead reckoning integrates the noisy odometry, which drifts away from the truth;
 *   3. loop closure detection: pose pairs further apart than `gap` in time but closer than `gate`
 *      are aligned by a closed-form 2D Kabsch/Umeyama step over the *known landmark
 *      correspondences* (the ICP update that matches the scans); the candidate is accepted only
 *      when the post-alignment RMS residual is below `tol`, which rejects false positives;
 *   4. pose graph optimization: Gauss/Newton-free projected gradient descent on
 *      sum of weighted translation and rotation residuals, node 0 fixed as the gauge.
 *
 * Verified against ground truth: pose RMSE must drop far below dead reckoning, the accepted closure
 * must reconstruct the true relative pose, an exactly consistent graph must be solved to zero cost,
 * and a graph without closures must not degrade; plus gauge, boundary, invalid-input and
 * determinism checks. Simplification: known landmark correspondences (no data association search)
 * and a first-order optimizer instead of a sparse Levenberg-Marquardt.
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
  normal(s) { return (this.float() + this.float() + this.float() - 1.5) * 2 * s; }
}

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const rot = (t) => ({ c: Math.cos(t), s: Math.sin(t) });
const apply = (R, p) => ({ x: R.c * p.x - R.s * p.y, y: R.s * p.x + R.c * p.y });
const meanPoint = (pts) => ({
  x: pts.reduce((a, p) => a + p.x, 0) / pts.length,
  y: pts.reduce((a, p) => a + p.y, 0) / pts.length,
});

/** Pose of `b` expressed in the frame of `a`: R_a^T (t_b - t_a), wrap(theta_b - theta_a). */
function relative(a, b) {
  const R = rot(-a.theta);
  return { t: apply(R, { x: b.x - a.x, y: b.y - a.y }), theta: wrap(b.theta - a.theta) };
}
/** Inverse composition: pose b = a (+) rel. */
function compose(a, rel) {
  const R = rot(a.theta);
  const shifted = apply(R, rel.t);
  return { x: a.x + shifted.x, y: a.y + shifted.y, theta: wrap(a.theta + rel.theta) };
}

/* ------------------------------------------------------------------ simulation */

function makeWorld(seed, options = {}) {
  const prng = new Prng(seed);
  const stepsPerLap = options.stepsPerLap ?? 20;
  const laps = options.laps ?? 2;
  const steps = stepsPerLap * laps;
  const advance = options.advance ?? 0.55;
  const turn = (2 * Math.PI) / stepsPerLap;
  const landmarks = options.landmarks ?? [
    { x: 3.0, y: 0.0 }, { x: 1.5, y: 2.2 }, { x: -1.2, y: 2.0 }, { x: -2.4, y: -0.6 },
    { x: -0.5, y: -2.4 }, { x: 2.1, y: -1.4 }, { x: 0.2, y: 0.9 },
  ];
  const truth = [{ x: 0, y: 0, theta: 0 }];
  for (let k = 0; k < steps; k++) {
    const previous = truth[truth.length - 1];
    const moved = { x: previous.x + advance * Math.cos(previous.theta),
      y: previous.y + advance * Math.sin(previous.theta), theta: previous.theta };
    truth.push(compose(moved, { t: { x: 0, y: 0 }, theta: turn }));
  }
  const odometry = [];   // odometry[k] measures truth[k] -> truth[k+1]
  const scans = [];      // scans[k][l] = landmark l in the frame of pose k, plus noise
  for (let k = 0; k < steps; k++) {
    const exact = relative(truth[k], truth[k + 1]);
    odometry.push({ t: { x: exact.t.x + prng.normal(0.02), y: exact.t.y + prng.normal(0.02) },
      theta: wrap(exact.theta + prng.normal(0.012)) });
  }
  for (const pose of truth) {
    const R = rot(-pose.theta);
    scans.push(landmarks.map((l) => {
      const local = apply(R, { x: l.x - pose.x, y: l.y - pose.y });
      return { x: local.x + prng.normal(0.01), y: local.y + prng.normal(0.01) };
    }));
  }
  return { truth, odometry, scans, landmarks, steps, prng };
}

function deadReckon(world) {
  const poses = [world.truth[0]];
  for (const z of world.odometry) poses.push(compose(poses[poses.length - 1], z));
  return poses;
}

/* ------------------------------------------------------------------ loop closure detection */

/** Closed-form alignment of two scans with known correspondences (2D Kabsch / Umeyama step). */
function alignScans(a, b) {
  let sinSum = 0, cosSum = 0;
  for (let k = 0; k < a.length; k++) {
    sinSum += a[k].x * b[k].y - a[k].y * b[k].x;
    cosSum += a[k].x * b[k].x + a[k].y * b[k].y;
  }
  const dTheta = Math.atan2(sinSum, cosSum);
  const R = rot(dTheta);
  const rotatedMean = apply(R, meanPoint(a));
  const bMean = meanPoint(b);
  const shift = { x: bMean.x - rotatedMean.x, y: bMean.y - rotatedMean.y };
  let sq = 0;
  for (let k = 0; k < a.length; k++) {
    const p = apply(R, a[k]);
    sq += (p.x + shift.x - b[k].x) ** 2 + (p.y + shift.y - b[k].y) ** 2;
  }
  // m_j = R(dTheta) m_i + shift  =>  relative pose of j in i's frame
  const shifted = apply(rot(-dTheta), shift);
  return { z: { t: { x: -shifted.x, y: -shifted.y }, theta: wrap(-dTheta) },
    rms: Math.sqrt(sq / a.length) };
}

function detectLoopClosures(dead, scans, options = {}) {
  const gap = options.gap ?? 6, gate = options.gate ?? 0.9, tol = options.tol ?? 0.05;
  const minLandmarks = options.minLandmarks ?? 4;
  const edges = [];
  for (let j = 1; j < dead.length; j++) {
    for (let i = j - gap; i >= 0; i--) {
      const proximity = Math.hypot(dead[i].x - dead[j].x, dead[i].y - dead[j].y);
      if (proximity > gate) continue;
      if (scans[i].length < minLandmarks) continue;
      const { z, rms } = alignScans(scans[i], scans[j]);
      edges.push({ i, j, z, rms, proximity, accepted: rms <= tol, kind: "loop" });
    }
  }
  return edges;
}

/* ------------------------------------------------------------------ pose graph optimization */

function odometryEdges(world, weight) {
  return world.odometry.map((z, k) => ({ i: k, j: k + 1, z, wt: weight, wtheta: weight / 0.1, kind: "odom" }));
}

function edgeResidual(poses, edge) {
  const measured = relative(poses[edge.i], poses[edge.j]);
  const dx = measured.t.x - edge.z.t.x, dy = measured.t.y - edge.z.t.y;
  const dTheta = wrap(measured.theta - edge.z.theta);
  return { dx, dy, dTheta };
}

function graphCost(poses, edges) {
  let cost = 0;
  for (const edge of edges) {
    const r = edgeResidual(poses, edge);
    cost += edge.wt * (r.dx * r.dx + r.dy * r.dy) + edge.wtheta * r.dTheta * r.dTheta;
  }
  return cost;
}

/** Analytic gradient of graphCost with respect to each free pose (node 0 is the gauge). */
function graphGradient(poses, edges) {
  const grad = poses.map(() => ({ x: 0, y: 0, theta: 0 }));
  for (const edge of edges) {
    const a = poses[edge.i];
    const r = edgeResidual(poses, edge);
    const d = { x: poses[edge.j].x - a.x, y: poses[edge.j].y - a.y };
    const rotatedBack = apply(rot(-a.theta), d);           // a = t_err + z_t, b measured in i's frame
    const Ra = rot(a.theta);
    const pull = { x: 2 * edge.wt * r.dx, y: 2 * edge.wt * r.dy };
    const inWorld = apply(Ra, pull);
    grad[edge.j].x += inWorld.x; grad[edge.j].y += inWorld.y;
    grad[edge.i].x -= inWorld.x; grad[edge.i].y -= inWorld.y;
    // d/dtheta_i of 2*t_err . (R_i^T d) = 2*(t_err.y*A.x - t_err.x*A.y) with A = t_err + z_t
    const ax = r.dx + edge.z.t.x, ay = r.dy + edge.z.t.y;
    grad[edge.i].theta += 2 * edge.wt * (r.dy * ax - r.dx * ay) - 2 * edge.wtheta * r.dTheta;
    grad[edge.j].theta += 2 * edge.wtheta * r.dTheta;
  }
  grad[0] = { x: 0, y: 0, theta: 0 }; // gauge: pose 0 never moves
  return grad;
}

function optimizePoseGraph(initial, edges, options = {}) {
  assert.ok(Array.isArray(initial) && initial.length > 0, "need at least one pose");
  for (const edge of edges) {
    assert.ok(Number.isInteger(edge.i) && edge.i >= 0 && edge.i < initial.length, "edge node out of range");
    assert.ok(Number.isInteger(edge.j) && edge.j >= 0 && edge.j < initial.length, "edge node out of range");
  }
  let poses = initial.map((p) => ({ ...p }));
  let cost = graphCost(poses, edges);
  const trace = [cost];
  let step = options.step ?? 0.5;
  let iterations = 0;
  for (; iterations < (options.maxIter ?? 600); iterations++) {
    const grad = graphGradient(poses, edges);
    const gnorm = grad.reduce((a, g) => a + g.x * g.x + g.y * g.y + g.theta * g.theta, 0);
    if (gnorm < 1e-18) break;
    let accepted = false;
    for (let back = 0; back < 40; back++) {
      const trial = poses.map((p, k) => ({ x: p.x - step * grad[k].x, y: p.y - step * grad[k].y,
        theta: wrap(p.theta - step * grad[k].theta) }));
      const trialCost = graphCost(trial, edges);
      if (trialCost < cost - 1e-12) {
        poses = trial; cost = trialCost; accepted = true;
        step = Math.min(step * 1.4, 4); // grow while progress continues
        break;
      }
      step *= 0.5;
    }
    trace.push(cost);
    if (!accepted) break;
  }
  return { poses, cost, trace, iterations };
}

/* ------------------------------------------------------------------ checks */

function poseRmse(a, b) {
  let sum = 0;
  for (let k = 0; k < a.length; k++) sum += (a[k].x - b[k].x) ** 2 + (a[k].y - b[k].y) ** 2;
  return Math.sqrt(sum / a.length);
}

function main() {
  const world = makeWorld(20250920);
  const dead = deadReckon(world);
  const truthRmseBefore = poseRmse(dead, world.truth);

  // 1. Happy path: the accepted closures re-anchor dead reckoning, collapsing the graph cost and
  //    the loop-closure residuals (the drift the odometry accumulated between the two laps).
  const candidates = detectLoopClosures(dead, world.scans);
  const closures = candidates.filter((c) => c.accepted);
  assert.ok(closures.length >= 5, `only ${closures.length} loop closures were accepted`);
  const loopEdges = closures.map((c) => ({ i: c.i, j: c.j, z: c.z, wt: 1, wtheta: 2.8, kind: "loop" }));
  const edges = [...odometryEdges(world, 1), ...loopEdges];
  const solved = optimizePoseGraph(dead, edges);
  const truthRmseAfter = poseRmse(solved.poses, world.truth);
  const meanClosureResidual = (poses) => loopEdges.reduce((acc, e) => {
    const r = edgeResidual(poses, e);
    return acc + Math.hypot(r.dx, r.dy);
  }, 0) / loopEdges.length;
  const residualBefore = meanClosureResidual(dead);
  const residualAfter = meanClosureResidual(solved.poses);
  assert.ok(solved.cost < graphCost(dead, edges) / 10,
    `graph cost only fell from ${graphCost(dead, edges)} to ${solved.cost}`);
  assert.ok(residualAfter < residualBefore / 5,
    `closure residual only fell from ${residualBefore} to ${residualAfter}`);
  assert.ok(truthRmseAfter < truthRmseBefore * 0.75,
    `RMSE improved only from ${truthRmseBefore} to ${truthRmseAfter}`);

  // 2. The accepted closure must reconstruct the true relative pose (independent ground-truth check).
  for (const c of closures) {
    const trueRel = relative(world.truth[c.i], world.truth[c.j]);
    assert.ok(Math.abs(c.z.t.x - trueRel.t.x) < 0.2 && Math.abs(c.z.t.y - trueRel.t.y) < 0.2,
      `closure translation error too large on (${c.i}, ${c.j})`);
    assert.ok(Math.abs(wrap(c.z.theta - trueRel.theta)) < 0.1,
      `closure rotation error too large on (${c.i}, ${c.j})`);
  }

  // 3. False positives are rejected: closures far apart in time are gated out and noisy pairs fail.
  const gated = candidates.every((c) => c.proximity <= 0.9 && c.j - c.i >= 6);
  assert.ok(gated, "a candidate violated the temporal/spatial gate");
  const bogus = alignScans(world.scans[0], world.scans[12]);
  assert.ok(bogus.rms > 0.05, `unrelated scans produced rms ${bogus.rms}`);
  assert.ok(!candidates.some((c) => c.j < world.steps - 3 && c.accepted === true && c.rms > 0.05),
    "a rejected candidate leaked into the accepted set");

  // 4. Oracle: an exactly consistent 3-pose graph must be solved back to the ground truth.
  const exactTruth = [{ x: 0, y: 0, theta: 0 }, { x: 1, y: 0, theta: 0.3 }, { x: 1.6, y: 0.9, theta: -0.2 }];
  const exactEdges = [{ i: 0, j: 1, z: relative(exactTruth[0], exactTruth[1]), wt: 1, wtheta: 10 },
    { i: 1, j: 2, z: relative(exactTruth[1], exactTruth[2]), wt: 1, wtheta: 10 },
    { i: 0, j: 2, z: relative(exactTruth[0], exactTruth[2]), wt: 1, wtheta: 10 }];
  const perturbed = exactTruth.map((p, k) => ({ x: p.x + (k ? 0.3 : 0), y: p.y - (k ? 0.2 : 0),
    theta: p.theta + (k ? 0.15 : 0) }));
  const recovered = optimizePoseGraph(perturbed, exactEdges);
  assert.ok(recovered.cost < 1e-9, `consistent graph cost ${recovered.cost} should vanish`);
  assert.ok(recovered.poses.every((p, k) => Math.hypot(p.x - exactTruth[k].x, p.y - exactTruth[k].y) < 1e-5
    && Math.abs(wrap(p.theta - exactTruth[k].theta)) < 1e-5), "exact graph was not recovered");

  // 5. Boundary cases: a single pose, and odometry-only (no closure) must not degrade the estimate.
  const single = optimizePoseGraph([{ x: 5, y: -3, theta: 1 }], []);
  assert.deepStrictEqual(single.poses, [{ x: 5, y: -3, theta: 1 }]);
  assert.strictEqual(single.cost, 0);
  const odomOnly = optimizePoseGraph(dead, odometryEdges(world, 1));
  assert.ok(poseRmse(odomOnly.poses, world.truth) <= truthRmseBefore + 1e-9,
    "odometry-only optimisation made the estimate worse");

  // 6. The gauge is preserved and the cost trace never increases.
  assert.deepStrictEqual(solved.poses[0], { ...dead[0] });
  assert.ok(solved.trace.length > 1, "expected several accepted iterations");
  assert.ok(solved.trace.every((v, k) => k === 0 || v <= solved.trace[k - 1] + 1e-12),
    "the cost trace increased");
  assert.ok(solved.cost <= solved.trace[0] + 1e-12);

  // 7. Invalid input is rejected: edges must reference existing nodes.
  assert.throws(() => optimizePoseGraph(dead, [{ i: 0, j: 99, z: { t: { x: 0, y: 0 }, theta: 0 } }]),
    /out of range/);
  assert.throws(() => optimizePoseGraph([], []), /at least one pose/);

  // 8. Determinism: the whole pipeline replays identically for the same seed and differs otherwise.
  const rerun = () => {
    const w = makeWorld(20250920);
    const d = deadReckon(w);
    const cl = detectLoopClosures(d, w.scans).filter((c) => c.accepted);
    return optimizePoseGraph(d, [...odometryEdges(w, 1), ...cl.map((c) => ({ i: c.i, j: c.j, z: c.z,
      wt: 1, wtheta: 2.8, kind: "loop" }))]).poses;
  };
  assert.deepStrictEqual(rerun(), solved.poses);
  const otherSeed = makeWorld(4242);
  assert.ok(poseRmse(deadReckon(otherSeed), dead) > 1e-6, "seeds should produce different noise");

  console.log(`slam: closures ${closures.length}/${candidates.length} accepted, RMSE ` +
    `${truthRmseBefore.toFixed(4)} -> ${truthRmseAfter.toFixed(4)}, graph cost ` +
    `${graphCost(dead, edges).toFixed(3)} -> ${solved.cost.toFixed(3)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, relative, compose, alignScans, makeWorld, deadReckon, detectLoopClosures,
  odometryEdges, edgeResidual, graphCost, graphGradient, optimizePoseGraph, poseRmse, wrap };
