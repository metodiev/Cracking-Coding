/** Task 734 - Implement fast multipole method conceptual small demo. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The fast multipole method (Greengard & Rokhlin 1987) evaluates an N-body sum in O(N) by
 * writing each *cluster* of sources as a short multipole series and evaluating that series for
 * every target far enough away, so the cluster is touched once per target instead of once per
 * source. This demo implements the single-level version for the 2D logarithmic kernel, where
 * the idea is easiest to see:
 *
 *     phi(z) = sum_j q_j log|z - z_j|
 *     phi_cell(z) ~ Q log|z - c| + Re sum_{k=1..p} a_k / (z - c)^k,
 *     Q = sum_j q_j,   a_k = -(1/k) sum_j q_j (z_j - c)^k
 *
 * The series is the Taylor expansion of log(1 - (z_j - c)/(z - c)); the truncation error of a
 * cluster of radius r seen from distance d is O((r/d)^(p+1)), so p terms suffice whenever r/d
 * is small -- which is exactly what makes the far field cheap.
 *
 * Simplified: one level of cells (no upward/downward multipole-to-local translation passes), a
 * uniform grid over a square domain, the log kernel instead of 1/r, no adaptive cells and no
 * parallel traversal; "near" means the target's own cell and its adjacent cells, which are
 * summed directly. What is verified: the coefficients are the exact cluster moments (checked
 * against hand complex arithmetic); the truncation error falls geometrically with the order p
 * at the distance/radius rate; a cluster sitting at its own cell centre is represented exactly
 * by its monopole; a single cell (nothing far) reproduces the direct sum exactly; order 8
 * reaches ~1e-9 on a well-separated cloud while the monopole alone is visibly wrong; the work
 * collapses from sources x targets to about p per far cell; and the degenerate inputs (no
 * sources, no targets, zero charges, coincident points, malformed sources) are handled. Fully
 * deterministic: fixed grid, fixed charges, no randomness at all.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

// Complex helpers: the kernel is complex even though the potential is its real part.
const sub = (a, b) => ({ real: a.real - b.real, imag: a.imag - b.imag });
const mul = (a, b) => ({
  real: a.real * b.real - a.imag * b.imag,
  imag: a.real * b.imag + a.imag * b.real,
});
const div = (a, b) => {
  const denominator = b.real * b.real + b.imag * b.imag;
  return {
    real: (a.real * b.real + a.imag * b.imag) / denominator,
    imag: (a.imag * b.real - a.real * b.imag) / denominator,
  };
};
const abs = (a) => Math.hypot(a.real, a.imag);

/** Direct logarithmic potential of every target against every source (the oracle). */
function directPotential(sources, targets) {
  let interactions = 0;
  const values = targets.map((z) => {
    let sum = 0;
    for (const source of sources) {
      sum += source.q * Math.log(abs(sub(z, source.z)));
      interactions++;
    }
    return sum;
  });
  return { values, interactions };
}

/** Multipole coefficients of one cluster: the exact charge and the p Taylor moments. */
function multipole(sources, centre, order) {
  const coefficients = new Array(order + 1).fill(null).map(() => ({ real: 0, imag: 0 }));
  let charge = 0;
  for (const source of sources) {
    ok(source !== null && source !== undefined && typeof source.q === "number"
      && source.z !== null && source.z !== undefined
      && typeof source.z.real === "number" && typeof source.z.imag === "number",
    "malformed source: expected { z: { real, imag }, q }");
    const delta = sub(source.z, centre);
    charge += source.q;
    let power = delta;
    for (let k = 1; k <= order; k++) {
      const term = mul({ real: -source.q / k, imag: 0 }, power);
      coefficients[k] = { real: coefficients[k].real + term.real, imag: coefficients[k].imag + term.imag };
      power = mul(power, delta);
    }
  }
  return { charge, coefficients };
}

/** Evaluate a multipole expansion at a target: the real part is the potential. */
function evaluateMultipole(expansion, centre, target) {
  const offset = sub(target, centre);
  let value = expansion.charge * Math.log(abs(offset));
  let denominator = offset;
  for (let k = 1; k < expansion.coefficients.length; k++) {
    value += div(expansion.coefficients[k], denominator).real;
    denominator = mul(denominator, offset);
  }
  return value;
}

/**
 * Single-level FMM on a `cells x cells` grid: targets are summed directly against their own and
 * the adjacent cells, and against every other cell through its truncated multipole expansion.
 */
function fmmPotential(sources, targets, { cells, order, extent = 1 }) {
  ok(Number.isInteger(cells) && cells >= 1, "the grid needs at least one cell");
  ok(Number.isInteger(order) && order >= 0, "the expansion order must be a non-negative integer");
  const cellKey = (z) => {
    const ix = Math.min(cells - 1, Math.max(0, Math.floor((z.real / extent) * cells)));
    const iy = Math.min(cells - 1, Math.max(0, Math.floor((z.imag / extent) * cells)));
    return `${ix}|${iy}`;
  };
  const members = new Map();
  for (const source of sources) {
    const key = cellKey(source.z);
    if (!members.has(key)) members.set(key, []);
    members.get(key).push(source);
  }
  const clusters = [];
  for (const [key, group] of members) {
    const [ix, iy] = key.split("|").map(Number);
    const centre = { real: ((ix + 0.5) * extent) / cells, imag: ((iy + 0.5) * extent) / cells };
    clusters.push({ ix, iy, centre, group, expansion: multipole(group, centre, order) });
  }
  clusters.sort((a, b) => (a.ix - b.ix) || (a.iy - b.iy));

  let directPairs = 0;
  let farTerms = 0;
  const values = targets.map((target) => {
    const [tx, ty] = cellKey(target).split("|").map(Number);
    let sum = 0;
    for (const cluster of clusters) {
      const near = Math.abs(cluster.ix - tx) <= 1 && Math.abs(cluster.iy - ty) <= 1;
      if (near) {
        for (const source of cluster.group) {
          sum += source.q * Math.log(abs(sub(target, source.z)));
          directPairs++;
        }
      } else {
        sum += evaluateMultipole(cluster.expansion, cluster.centre, target);
        farTerms += order + 1;
      }
    }
    return sum;
  });
  return { values, directPairs, farTerms, interactions: directPairs + farTerms, occupiedCells: clusters.length };
}

/**
 * Deterministic cloud: charges on a small circle inside every second cell. The angle offset and
 * the growing radius break the rotational symmetry, so the cluster moments do not cancel (a
 * symmetric ring would have an exactly vanishing dipole and mask the order-1 term).
 */
function makeCloud(cells, perCell, extent = 1) {
  const sources = [];
  for (let ix = 0; ix < cells; ix++) {
    for (let iy = 0; iy < cells; iy++) {
      if ((ix + iy) % 2 === 1) continue;
      const cx = ((ix + 0.5) * extent) / cells;
      const cy = ((iy + 0.5) * extent) / cells;
      for (let k = 0; k < perCell; k++) {
        const angle = (2 * Math.PI * k) / perCell + 0.37 * ix + 0.21 * iy;
        const radius = 0.008 + 0.004 * k;
        sources.push({
          z: { real: cx + radius * Math.cos(angle), imag: cy + radius * Math.sin(angle) },
          q: 0.5 + ((k * 2 + ix + iy) % 3) * 0.5,
        });
      }
    }
  }
  return sources;
}

/** Largest distance from a cluster centre to its own sources: the r of the r/d error rate. */
function clusterRadius(sources, cells, extent = 1) {
  let worst = 0;
  for (const source of sources) {
    const ix = Math.min(cells - 1, Math.max(0, Math.floor((source.z.real / extent) * cells)));
    const iy = Math.min(cells - 1, Math.max(0, Math.floor((source.z.imag / extent) * cells)));
    const centre = { real: ((ix + 0.5) * extent) / cells, imag: ((iy + 0.5) * extent) / cells };
    worst = Math.max(worst, abs(sub(source.z, centre)));
  }
  return worst;
}

const maxError = (approximate, exact) => {
  let worst = 0;
  for (let i = 0; i < exact.length; i++) worst = Math.max(worst, Math.abs(approximate[i] - exact[i]));
  return worst;
};

module.exports = { directPotential, multipole, evaluateMultipole, fmmPotential, makeCloud, maxError, abs, sub };

if (require.main === module) {
  // -- the coefficients are the exact moments, checked against hand arithmetic -------------
  const cluster = [
    { z: { real: 0.1, imag: 0.05 }, q: 2 },
    { z: { real: -0.05, imag: 0.1 }, q: -1 },
    { z: { real: 0.0, imag: -0.08 }, q: 3 },
  ];
  const centre = { real: 0, imag: 0 };
  const expansion = multipole(cluster, centre, 3);
  strictEqual(expansion.charge, 4, "the monopole is exactly the cluster charge");
  const handOrder1 = cluster.reduce((acc, s) => ({
    real: acc.real - s.q * s.z.real,
    imag: acc.imag - s.q * s.z.imag,
  }), { real: 0, imag: 0 });
  ok(Math.abs(expansion.coefficients[1].real - handOrder1.real) < 1e-15
    && Math.abs(expansion.coefficients[1].imag - handOrder1.imag) < 1e-15,
  "the order-1 coefficient is exactly -sum q delta, computed by hand");
  const handOrder2 = cluster.reduce((acc, s) => ({
    real: acc.real - (s.q * (s.z.real * s.z.real - s.z.imag * s.z.imag)) / 2,
    imag: acc.imag - (s.q * (2 * s.z.real * s.z.imag)) / 2,
  }), { real: 0, imag: 0 });
  ok(Math.abs(expansion.coefficients[2].real - handOrder2.real) < 1e-15
    && Math.abs(expansion.coefficients[2].imag - handOrder2.imag) < 1e-15,
  "and the order-2 coefficient is exactly -sum q delta^2 / 2, computed by hand");
  console.log(`multipole: Q=${expansion.charge}, |a1|=${abs(expansion.coefficients[1]).toFixed(4)}`);

  // -- a cluster sitting at its own centre is exactly its monopole -------------------------
  const central = [{ z: { real: 0.25, imag: 0.25 }, q: 1.5 }, { z: { real: 0.25, imag: 0.25 }, q: -0.5 }];
  const centralCentre = { real: 0.25, imag: 0.25 };
  const centralExpansion = multipole(central, centralCentre, 4);
  ok(centralExpansion.coefficients.every((value) => value.real === 0 && value.imag === 0),
    "a centred cluster has no higher moments");
  const farTarget = { real: 0.9, imag: 0.9 };
  const exactCentral = directPotential(central, [farTarget]).values[0];
  for (const order of [0, 1, 4]) {
    const approximate = evaluateMultipole(multipole(central, centralCentre, order), centralCentre, farTarget);
    ok(Math.abs(approximate - exactCentral) < 1e-14,
      `a centred cluster must be exact at order ${order}`);
  }

  // -- one cell: nothing is far, so the FMM is the direct sum ------------------------------
  const cloud = makeCloud(4, 6);
  const targets = [{ real: 0.07, imag: 0.11 }, { real: 0.53, imag: 0.51 }, { real: 0.93, imag: 0.87 },
    { real: 0.31, imag: 0.77 }];
  const oracle = directPotential(cloud, targets);
  const single = fmmPotential(cloud, targets, { cells: 1, order: 4 });
  deepStrictEqual(single.values, oracle.values, "with a single cell the FMM is the direct sum");
  strictEqual(single.directPairs, cloud.length * targets.length, "and every pair is touched directly");
  strictEqual(single.farTerms, 0, "nothing is far away in a one-cell world");

  // -- truncation error falls geometrically with the order ----------------------------------
  const errors = [];
  for (let order = 0; order <= 8; order++) {
    errors.push(maxError(fmmPotential(cloud, targets, { cells: 4, order }).values, oracle.values));
  }
  for (let order = 1; order <= 8; order++) {
    ok(errors[order] < errors[order - 1],
      `order ${order} must beat order ${order - 1} (${errors[order]} vs ${errors[order - 1]})`);
  }
  const ratios = errors.slice(1).map((value, i) => errors[i] / value);
  ok(ratios.filter((ratio) => ratio > 1.5).length >= 6,
    "the error must fall geometrically over most of the sweep");
  ok(errors[0] > 1e-3, `the monopole alone must be visibly wrong, got ${errors[0]}`);
  ok(errors[8] < 1e-9, `order 8 must reach the 1e-9 range, got ${errors[8]}`);
  const radius = clusterRadius(cloud, 4);
  const nearestFar = 1 / 4;                     // closest cell centre outside the 3x3 neighbourhood
  const predicted = nearestFar / radius;        // each extra term buys a factor d/r
  const observed = ratios[ratios.length - 1];
  ok(observed > predicted / 4 && observed < predicted * 4,
    `the improvement factor ${observed.toFixed(1)} must track the distance/radius ratio ${predicted.toFixed(1)}`);
  console.log(`order sweep: ${errors.map((e) => e.toExponential(1)).join(" ")}`);
  console.log(`improvement factors: ${ratios.map((r) => r.toFixed(2)).join(" ")}` +
    ` (predicted ${predicted.toFixed(1)})`);

  // -- the cost claim: the far field ignores the cluster density, the near field does not --
  const probes = [];
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 10; j++) probes.push({ real: (i + 0.5) / 10, imag: (j + 0.5) / 10 });
  }
  const sparseCloud = makeCloud(4, 2);
  const denseCloud = makeCloud(4, 16);
  const sparseRun = fmmPotential(sparseCloud, probes, { cells: 4, order: 2 });
  const denseRun = fmmPotential(denseCloud, probes, { cells: 4, order: 2 });
  strictEqual(denseRun.occupiedCells, 8, "the cloud occupies every second cell of the 4x4 grid");
  strictEqual(denseRun.farTerms, sparseRun.farTerms,
    "the far-field cost does not care how many charges a cell holds");
  strictEqual(denseRun.directPairs, sparseRun.directPairs * 8,
    "while the near-field work scales exactly with the density");
  const sparseRatio = (sparseCloud.length * probes.length) / sparseRun.interactions;
  const denseRatio = (denseCloud.length * probes.length) / denseRun.interactions;
  ok(sparseRatio < 1, `with tiny clusters the per-cell overhead makes the FMM lose (${sparseRatio.toFixed(2)}x)`);
  ok(denseRatio > 1.5, `with dense clusters it wins handsomely (${denseRatio.toFixed(2)}x)`);
  ok(denseRatio > sparseRatio * 2, "and the advantage grows with the density, as the theory says");
  const accurate = fmmPotential(cloud, targets, { cells: 4, order: 10 });
  ok(maxError(accurate.values, oracle.values) < 1e-11,
    "enough terms make the single-level FMM numerically exact here");
  ok(maxError(fmmPotential(cloud, targets, { cells: 2, order: 10 }).values, oracle.values) < 1e-11,
    "and a coarser grid works just as well at that order");
  console.log(`cost: sparse ${sparseRatio.toFixed(2)}x, dense ${denseRatio.toFixed(2)}x` +
    ` (far terms ${sparseRun.farTerms} either way, near pairs ${sparseRun.directPairs} -> ${denseRun.directPairs})`);

  // -- degenerate inputs ----------------------------------------------------------------------
  const empty = fmmPotential([], targets, { cells: 4, order: 4 });
  deepStrictEqual(empty.values, [0, 0, 0, 0], "no sources means a zero potential");
  strictEqual(empty.interactions, 0);
  deepStrictEqual(fmmPotential(cloud, [], { cells: 4, order: 4 }).values, [], "no targets means no output");
  const zeroCharges = fmmPotential(cloud.map((s) => ({ z: s.z, q: 0 })), targets, { cells: 4, order: 3 });
  ok(zeroCharges.values.every((value) => value === 0), "zero charges give a zero potential");
  const coincident = [{ z: { real: 0.5, imag: 0.5 }, q: 2 }, { z: { real: 0.5, imag: 0.5 }, q: 3 }];
  const coincidentFar = { real: 0.9, imag: 0.9 };
  const coincidentRun = fmmPotential(coincident, [coincidentFar], { cells: 4, order: 0 });
  ok(Math.abs(coincidentRun.values[0] - directPotential(coincident, [coincidentFar]).values[0]) < 1e-14,
    "coincident sources collapse to their total charge: a monopole is exact for a point cluster");
  const singleSource = fmmPotential([{ z: { real: 0.1, imag: 0.1 }, q: 1 }], [{ real: 0.5, imag: 0.5 }],
    { cells: 2, order: 2 });
  ok(Math.abs(singleSource.values[0] - Math.log(Math.hypot(0.4, 0.4))) < 1e-14,
    "a single source in a near cell is summed directly");
  throws(() => multipole([{ z: null, q: 1 }], centre, 1), /malformed source/,
    "a malformed source must be rejected instead of producing NaN");
  throws(() => fmmPotential(cloud, targets, { cells: 0, order: 2 }), /at least one cell/);
  throws(() => fmmPotential(cloud, targets, { cells: 4, order: -1 }), /non-negative integer/);
  ok(Number.isFinite(fmmPotential(cloud, targets, { cells: 3, order: 4 }).values[0]),
    "a grid that does not divide the cloud evenly still produces finite numbers");
  console.log("PASS");
}
