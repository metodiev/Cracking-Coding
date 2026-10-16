/** Task 731 - Implement adaptive mesh refinement driving solver coupling. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * An adaptive solver does not refine the geometry, it refines what the *solution* cannot
 * resolve: solve, estimate where the discretization is worst, refine only those cells, solve
 * again. This file couples the three pieces for the 1D model problem -u'' = f, u(0) = u(1) = 0:
 *
 *   * mesh: a dyadic binary tree over [0, 1]; splitting a cell replaces it by its two halves,
 *     so every cell edge is a power-of-two fraction of [0, 1] and the mesh is always a clean
 *     partition (no gaps, no overlaps).
 *   * solver: linear finite elements with the exact element stiffness (1/h)[[1,-1],[-1,1]],
 *     Gauss-quadrature load vectors and a Thomas sweep for the resulting tridiagonal system.
 *   * estimator/driver: the classical jump indicator eta_e^2 = (1/2) sum_over_faces h * |jump
 *     of u'|^2 -- large where the discrete derivative kinks -- and the refine loop marks the
 *     cells whose indicator is above a threshold, refines them, and repeats.
 *
 * Simplified: one spatial dimension, which is what makes the hanging-node issue disappear (a
 * cell face is shared by at most one cell on either side, so no constrained-node interpolation
 * is needed), a fixed refinement fraction rather than a Dorfler marking strategy, and a
 * hand-rolled Thomas solver instead of a library. What is verified: the mesh invariant after
 * every cycle; the second-order convergence of the finite-element solution on uniform meshes
 * against the exact solution; the FE solution being nodally exact for a constant load; each AMR
 * cycle lowering the true error and the estimator; adaptivity beating a uniform mesh with the
 * same number of cells by a wide margin on a boundary layer, with the refined cells landing
 * where the error actually is; and the degenerate cases (one cell, refined-out mesh, zero
 * load). Everything is deterministic: no randomness, fixed grids and fixed quadrature.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

const LAYER_CENTER = 0.15;          // position of the sharp feature for the adaptive tests
const LAYER_WIDTH = 0.03;

/**
 * Manufactured solution of -u'' = f with homogeneous Dirichlet data: a Gaussian bump, which is
 * numerically zero at both ends (exp(-25) and exp(-800)) and needs a graded mesh to resolve.
 */
const layerSolution = (x) => Math.exp(-(((x - LAYER_CENTER) / LAYER_WIDTH) ** 2));
const layerDerivative = (x) => {
  const s = (x - LAYER_CENTER) / LAYER_WIDTH;
  return (-2 * s * Math.exp(-s * s)) / LAYER_WIDTH;
};
const layerLoad = (x) => {
  const s = (x - LAYER_CENTER) / LAYER_WIDTH;
  return ((2 - 4 * s * s) * Math.exp(-s * s)) / LAYER_WIDTH ** 2;
};
const sineSolution = (x) => Math.sin(Math.PI * x);
const sineLoad = (x) => Math.PI ** 2 * Math.sin(Math.PI * x);

const uniformMesh = (cells) => Array.from({ length: cells + 1 }, (_, i) => i / cells);

/** Refine the listed cells (indices into the node array) into halves; keeps the order. */
function refineMesh(nodes, marked) {
  const targets = new Set(marked);
  const out = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    out.push(nodes[i]);
    if (targets.has(i)) out.push((nodes[i] + nodes[i + 1]) / 2);
  }
  out.push(nodes[nodes.length - 1]);
  return out;
}

/** Dyadic mesh invariant: ordered, endpoints exact, every cell length 2^-k uniform-per-level. */
function validateMesh(nodes) {
  ok(nodes.length >= 2, "a mesh needs at least one cell");
  strictEqual(nodes[0], 0, "the mesh starts at 0");
  strictEqual(nodes[nodes.length - 1], 1, "and ends at 1");
  for (let i = 0; i < nodes.length - 1; i++) {
    const h = nodes[i + 1] - nodes[i];
    ok(h > 0, `cell ${i} must have positive length`);
    const level = Math.round(-Math.log2(h));
    ok(Math.abs(h - 2 ** -level) < 1e-15, `cell ${i} has a non-dyadic length ${h}`);
    ok(nodes[i] * 2 ** level === Math.floor(nodes[i] * 2 ** level + 1e-12),
      `cell ${i} does not start on the dyadic grid`);
  }
  return true;
}

const GAUSS = [
  { x: -Math.sqrt(3 / 5), w: 5 / 9 }, { x: 0, w: 8 / 9 }, { x: Math.sqrt(3 / 5), w: 5 / 9 },
];

/**
 * Assemble and solve the linear finite-element system. Interior node i (1 <= i <= cells-1) gets
 * the usual tridiagonal row; `off[i]` is the (symmetric) coupling between interior nodes i and
 * i+1, accumulated from the elements that touch both.
 */
function solveFem(nodes, load) {
  const cells = nodes.length - 1;
  const size = nodes.length - 2;
  const diagonal = new Array(size).fill(0);
  const off = new Array(Math.max(0, size - 1)).fill(0);
  const rhs = new Array(size).fill(0);
  const interior = (node) => node >= 1 && node <= cells - 1;
  for (let e = 0; e < cells; e++) {
    const a = nodes[e];
    const b = nodes[e + 1];
    const h = b - a;
    const stiffness = 1 / h;
    const left = e;
    const right = e + 1;
    if (interior(left)) diagonal[left - 1] += stiffness;
    if (interior(right)) diagonal[right - 1] += stiffness;
    if (interior(left) && interior(right)) off[left - 1] -= stiffness;
    for (const point of GAUSS) {
      const x = a + ((b - a) * (point.x + 1)) / 2;
      const weight = (point.w * (b - a)) / 2;
      const value = load(x) * weight;
      if (interior(left)) rhs[left - 1] += value * (1 - (x - a) / h);
      if (interior(right)) rhs[right - 1] += value * ((x - a) / h);
    }
  }
  const u = new Array(nodes.length).fill(0);
  if (size > 0) {
    const x = thomas(diagonal, off, rhs);
    for (let i = 0; i < size; i++) u[i + 1] = x[i];
  }
  return { u, diagonal, off, rhs };
}

/** Thomas algorithm for a symmetric tridiagonal system (diagonal, off-diagonal, rhs). */
function thomas(diagonal, off, rhs) {
  const n = diagonal.length;
  const c = new Array(n).fill(0);
  const d = new Array(n).fill(0);
  c[0] = n > 1 ? off[0] / diagonal[0] : 0;
  d[0] = rhs[0] / diagonal[0];
  for (let i = 1; i < n; i++) {
    const denominator = diagonal[i] - off[i - 1] * c[i - 1];
    c[i] = i < n - 1 ? off[i] / denominator : 0;
    d[i] = (rhs[i] - off[i - 1] * d[i - 1]) / denominator;
  }
  const x = new Array(n).fill(0);
  x[n - 1] = d[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
  return x;
}

/** H1 seminorm error (the energy norm), which must decrease monotonically under refinement. */
function energyError(nodes, u, derivative) {
  let total = 0;
  for (let e = 0; e < nodes.length - 1; e++) {
    const a = nodes[e];
    const b = nodes[e + 1];
    const slope = (u[e + 1] - u[e]) / (b - a);
    for (const point of GAUSS) {
      const x = a + ((b - a) * (point.x + 1)) / 2;
      const weight = (point.w * (b - a)) / 2;
      total += weight * (slope - derivative(x)) ** 2;
    }
  }
  return Math.sqrt(total);
}

/** L2 error against an exact solution, integrated with the same Gauss rule per element. */
function l2Error(nodes, u, exact) {
  let total = 0;
  for (let e = 0; e < nodes.length - 1; e++) {
    const a = nodes[e];
    const b = nodes[e + 1];
    for (const point of GAUSS) {
      const x = a + ((b - a) * (point.x + 1)) / 2;
      const weight = (point.w * (b - a)) / 2;
      const interpolated = (u[e] * (b - x) + u[e + 1] * (x - a)) / (b - a);
      total += weight * (interpolated - exact(x)) ** 2;
    }
  }
  return Math.sqrt(total);
}

/** Jump estimator: per-cell indicator from the kinks of the discrete derivative. */
function estimators(nodes, u) {
  const slopes = nodes.slice(0, -1).map((_, e) => (u[e + 1] - u[e]) / (nodes[e + 1] - nodes[e]));
  return nodes.slice(0, -1).map((_, e) => {
    const leftJump = e === 0 ? 0 : slopes[e] - slopes[e - 1];
    const rightJump = e === slopes.length - 1 ? 0 : slopes[e + 1] - slopes[e];
    return Math.sqrt(0.5 * ((nodes[e + 1] - nodes[e]) * leftJump ** 2 + (nodes[e + 1] - nodes[e]) * rightJump ** 2));
  });
}

/** Mark the cells whose indicator is in the top `fraction` (at least the worst one). */
function markCells(indicator, fraction) {
  const ranked = indicator.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value);
  const count = Math.max(1, Math.round(indicator.length * fraction));
  return ranked.slice(0, count).filter((entry) => entry.value > 0).map((entry) => entry.index);
}

/** One solve-estimate-refine cycle; returns the mesh, solution and diagnostics of every pass. */
function adaptiveSolve(startingMesh, load, exact, { cycles, fraction, maxCells, derivative = null }) {
  const history = [];
  let nodes = startingMesh.slice();
  for (let cycle = 0; cycle < cycles; cycle++) {
    const { u } = solveFem(nodes, load);
    const indicator = estimators(nodes, u);
    history.push({
      cycle,
      cells: nodes.length - 1,
      error: l2Error(nodes, u, exact),
      energy: derivative === null ? null : energyError(nodes, u, derivative),
      indicatorTotal: indicator.reduce((a, b) => a + b, 0),
    });
    if (nodes.length - 1 >= maxCells) break;
    const marked = markCells(indicator, fraction);
    if (marked.length === 0) break;
    nodes = refineMesh(nodes, marked);
    validateMesh(nodes);
  }
  return { nodes, history };
}

module.exports = {
  layerSolution, layerDerivative, layerLoad, sineSolution, sineLoad, uniformMesh, refineMesh,
  validateMesh, solveFem, thomas, l2Error, energyError, estimators, markCells, adaptiveSolve,
};

if (require.main === module) {
  // -- the mesh invariant and refinement bookkeeping -------------------------------------
  const small = refineMesh([0, 0.5, 1], [0, 1]);
  deepStrictEqual(small, [0, 0.25, 0.5, 0.75, 1], "refining both halves gives a uniform mesh");
  validateMesh(small);
  deepStrictEqual(refineMesh([0, 0.5, 1], []), [0, 0.5, 1], "refining nothing changes nothing");
  deepStrictEqual(refineMesh([0, 0.5, 1], [1]), [0, 0.5, 0.75, 1], "one sided refinement is allowed");
  validateMesh([0, 0.25, 0.5, 0.75, 1]);
  throws(() => validateMesh([0, 0.3, 1]), /non-dyadic/);
  throws(() => validateMesh([0, 0.5]), /ends at 1/);
  throws(() => validateMesh([0.1, 0.5, 1]), /starts at 0/);

  // -- the finite-element solution is nodally exact for a constant load ------------------
  const constantLoad = () => 2;
  const uniform = uniformMesh(8);
  const constantSolution = solveFem(uniform, constantLoad);
  for (let i = 0; i < uniform.length; i++) {
    const expected = 2 * (uniform[i] * (1 - uniform[i])) / 2;
    ok(Math.abs(constantSolution.u[i] - expected) < 1e-12,
      `linear elements must be nodally exact for -u'' = 2 at node ${i}`);
  }
  const twiceAsFine = solveFem(uniformMesh(16), constantLoad);
  ok(Math.abs(twiceAsFine.u[8] - constantSolution.u[4]) < 1e-12, "and identical at shared nodes");

  // -- second-order convergence against the exact sine solution --------------------------
  const errors = [];
  for (const cells of [8, 16, 32, 64]) {
    const nodes = uniformMesh(cells);
    const { u } = solveFem(nodes, sineLoad);
    errors.push(l2Error(nodes, u, sineSolution));
  }
  for (let i = 1; i < errors.length; i++) {
    const ratio = errors[i - 1] / errors[i];
    ok(ratio > 3.2 && ratio < 4.8,
      `halving h must cut the error by about 4, got ${ratio.toFixed(3)} (${errors[i - 1]} -> ${errors[i]})`);
  }
  console.log(`uniform L2 errors: ${errors.map((e) => e.toExponential(3)).join(" ")}`);

  // -- AMR: the energy error falls monotonically, the L2 error collapses ------------------
  const start = uniformMesh(4);
  const adaptive = adaptiveSolve(start, layerLoad, layerSolution,
    { cycles: 10, fraction: 0.4, maxCells: 512, derivative: layerDerivative });
  ok(adaptive.history.length >= 8, "the adaptive loop must run several cycles");
  for (let i = 1; i < adaptive.history.length; i++) {
    ok(adaptive.history[i].energy < adaptive.history[i - 1].energy,
      `cycle ${i} must lower the energy error: refinement nests the spaces, so it cannot rise ` +
      `(${adaptive.history[i].energy} vs ${adaptive.history[i - 1].energy})`);
  }
  for (let i = 2; i < adaptive.history.length; i++) {
    ok(adaptive.history[i].error <= adaptive.history[i - 1].error,
      `cycle ${i} must lower the L2 error once the feature is resolved (${adaptive.history[i].error} vs ${adaptive.history[i - 1].error})`);
  }
  const first = adaptive.history[0];
  const last = adaptive.history[adaptive.history.length - 1];
  ok(last.error < first.error / 100,
    `ten cycles must cut the L2 error by two orders: ${first.error} -> ${last.error}`);
  ok(last.indicatorTotal < first.indicatorTotal / 2,
    `and the estimator must drop with it: ${first.indicatorTotal} -> ${last.indicatorTotal}`);
  validateMesh(adaptive.nodes);
  console.log(`adaptive: ${adaptive.history.map((h) => `${h.cells}c/${h.error.toExponential(1)}`).join(" ")}`);

  // -- the refined cells land where the feature is ------------------------------------------
  const near = adaptive.nodes.filter((x) => Math.abs(x - LAYER_CENTER) < 5 * LAYER_WIDTH).length;
  ok(near >= adaptive.nodes.length * 0.8,
    `the refinement must concentrate on the feature: ${near} of ${adaptive.nodes.length} nodes`);
  const lengths = adaptive.nodes.slice(0, -1).map((x, i) => adaptive.nodes[i + 1] - x);
  ok(Math.max(...lengths) > 100 * Math.min(...lengths),
    "the mesh must be strongly graded");
  console.log(`grading: finest=${Math.min(...lengths).toExponential(2)}` +
    ` coarsest=${Math.max(...lengths).toExponential(2)} near-feature=${near}/${adaptive.nodes.length}`);

  // -- adaptivity beats a uniform mesh with the same number of cells ----------------------
  const finalCells = adaptive.nodes.length - 1;
  const adaptiveError = l2Error(adaptive.nodes, solveFem(adaptive.nodes, layerLoad).u, layerSolution);
  ok(adaptiveError <= last.error, "the final (unestimated) mesh must be at least as good as the last cycle");
  const uniformPeer = uniformMesh(finalCells);
  const peerError = l2Error(uniformPeer, solveFem(uniformPeer, layerLoad).u, layerSolution);
  ok(adaptiveError * 10 < peerError,
    `with ${finalCells} cells the adaptive mesh (${adaptiveError.toExponential(2)}) must beat` +
    ` uniform (${peerError.toExponential(2)}) by an order of magnitude`);
  const finerUniform = uniformMesh(4 * finalCells);
  const finerError = l2Error(finerUniform, solveFem(finerUniform, layerLoad).u, layerSolution);
  ok(adaptiveError < finerError,
    `even a uniform mesh with 4x the cells must not win (${finerError.toExponential(2)})`);
  console.log(`cells=${finalCells}: adaptive=${adaptiveError.toExponential(3)}` +
    ` uniform=${peerError.toExponential(3)} uniform(4x)=${finerError.toExponential(3)}`);

  // -- estimators localize the error and drive the marking ---------------------------------
  const layerMesh = uniformMesh(16);
  const layerRun = solveFem(layerMesh, layerLoad);
  const indicator = estimators(layerMesh, layerRun.u);
  const worst = indicator.indexOf(Math.max(...indicator));
  const perCellError = layerMesh.slice(0, -1).map((a, i) =>
    l2Error([a, layerMesh[i + 1]], [layerRun.u[i], layerRun.u[i + 1]], layerSolution));
  const trueWorst = perCellError.indexOf(Math.max(...perCellError));
  ok(Math.abs(worst - Math.round(LAYER_CENTER * 16)) <= 3,
    `the jump estimator must point at the feature (cell ${worst}, feature cell ${Math.round(LAYER_CENTER * 16)})`);
  const rankedCells = markCells(indicator, 0.25);
  ok(rankedCells.includes(trueWorst), "and the marking must include the worst cell by true error");
  strictEqual(rankedCells.length, 4, "marking a quarter of 16 cells marks four cells");
  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  const featureBand = indicator.slice(0, 6);
  const farField = indicator.slice(8);
  ok(mean(featureBand) > 1e-3, "the feature band must carry real estimator mass");
  ok(Math.max(...farField) < mean(featureBand) / 100,
    "and the far field must be essentially resolved by comparison");
  deepStrictEqual(markCells(indicator.map(() => 0), 0.5), [],
    "a converged solution marks nothing");
  console.log(`estimator: worst cell ${worst}, true worst ${trueWorst},` +
    ` feature mean=${mean(featureBand).toExponential(2)} far max=${Math.max(...farField).toExponential(2)}`);

  // -- degenerate cases ----------------------------------------------------------------------
  const single = solveFem([0, 1], sineLoad);
  deepStrictEqual(single.u, [0, 0], "one cell has no interior degree of freedom");
  strictEqual(single.diagonal.length, 0);
  const noLoad = solveFem(uniformMesh(8), () => 0);
  ok(noLoad.u.every((value) => value === 0), "a zero load gives a zero solution");
  const zeroIndicator = estimators(uniformMesh(8), noLoad.u);
  ok(zeroIndicator.every((value) => value === 0), "and a zero estimator");
  const capped = adaptiveSolve(uniformMesh(2), layerLoad, layerSolution, { cycles: 6, fraction: 0.5, maxCells: 4 });
  ok(capped.history.length < 6, "the cell cap must cut the loop short");
  ok(capped.nodes.length - 1 <= 6, "and the mesh must stay near the cap (one refinement step past it)");
  const uncapped = adaptiveSolve(uniformMesh(2), layerLoad, layerSolution, { cycles: 6, fraction: 0.5, maxCells: 1024 });
  strictEqual(uncapped.history.length, 6, "without a cap the loop runs every requested cycle");
  ok(uncapped.nodes.length > capped.nodes.length, "and refines further than the capped run");
  const frozen = adaptiveSolve(uniformMesh(4), () => 0, () => 0, { cycles: 4, fraction: 0.5, maxCells: 64 });
  strictEqual(frozen.nodes.length, 5, "nothing is refined when the estimator is zero everywhere");

  // -- determinism ----------------------------------------------------------------------------
  const replay = () => {
    const run = adaptiveSolve(uniformMesh(4), layerLoad, layerSolution, { cycles: 4, fraction: 0.4, maxCells: 256 });
    return `${run.nodes.length}/${run.history.map((h) => h.error.toExponential(6)).join(",")}`;
  };
  deepStrictEqual(replay(), replay(), "identical problems must refine identically");
  console.log("PASS");
}
