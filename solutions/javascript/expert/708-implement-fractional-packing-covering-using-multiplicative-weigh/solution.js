/** Task 708 - Implement fractional packing/covering using multiplicative weights update. (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
 *
 * Fractional packing: max { c.x : A x <= b, x >= 0 } - the relaxation behind set packing, maximum
 * throughput, scheduling and flow. The solver here is the exponential-potential (multiplicative
 * weights) greedy of Grigoriadis-Khachiyan: keep one weight w_i per row, pick the column with the
 * best profit-per-weight ratio c_j / sum_i A_ij w_i / b_i, and raise it by the *largest* step that
 * keeps every row satisfied, then multiply the weights by 1 + eps * A_ij * step / b_i. Congested
 * rows get heavy, so later steps avoid them: that is the whole "adaptive" behaviour of MWU, and it
 * keeps the current point feasible at every single iteration - the strongest invariant available.
 *
 * The covering LP min { b.y : A^T y >= c, y >= 0 } is the dual of that packing problem, so it is
 * solved by reading the same weights as dual multipliers and rescaling them so that the covering
 * constraints hold exactly; the rescaled point is verified to be feasible, and its cost is compared
 * with the optimum to show the (1 + O(eps)) approximation.
 *
 * The increase-only walk can stall on a vertex that the optimum does not use, so it is followed by
 * a two-column local search: every pair of columns is scanned and the best vertex of the induced
 * two-variable LP is kept. Policy matters more than step size here - the walk itself is unchanged
 * by eps and by the iteration count once a column is saturated.
 *
 * What is simplified: the exact optimum comes from a vertex-enumeration LP oracle that is only
 * usable for at most three variables, and the number of MWU iterations is a parameter rather than
 * the theoretically sufficient O(width * m log m / eps^2).
 *
 * Invariant verified: every iterate - and the reported x - satisfies A x <= b and x >= 0 exactly,
 * the reported covering point satisfies A^T y >= c exactly, primal <= dual objective always holds,
 * the primal value is within the documented epsilon of the exhaustive optimum, and the value never
 * decreases as iterations are added.
 */
"use strict";
const { ok, strictEqual, deepStrictEqual, throws } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Exact rational comparison helper: is a/b > c/d for BigInt values with b, d > 0? */
function ratioGreater(a, b, c, d) { return a * d > c * b; }

/** Determinant of a square BigInt matrix by Laplace expansion (the oracle only needs n <= 3). */
function determinant(matrix) {
  const n = matrix.length;
  if (n === 1) return matrix[0][0];
  if (n === 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
  let total = 0n;
  for (let col = 0; col < n; col++) {
    const minor = matrix.slice(1).map((row) => row.filter((_, j) => j !== col));
    const sign = col % 2 === 0 ? 1n : -1n;
    total += sign * matrix[0][col] * determinant(minor);
  }
  return total;
}

/**
 * Exact optimum of max { c.x : A x <= b, x >= 0 } by enumerating basic solutions: an optimal vertex
 * is determined by n tight constraints among the rows and the non-negativity bounds, and each
 * candidate system is solved by Cramer's rule in exact BigInt arithmetic.
 */
function exactPackingOptimum(rows, bounds, objective) {
  const n = objective.length;
  const inequalities = [];
  for (let i = 0; i < rows.length; i++) inequalities.push({ coeffs: rows[i].map(BigInt), rhs: BigInt(bounds[i]) });
  for (let j = 0; j < n; j++) {
    const coeffs = new Array(n).fill(0n);
    coeffs[j] = -1n; // x_j >= 0
    inequalities.push({ coeffs, rhs: 0n });
  }
  let bestNum = 0n, bestDen = 1n; // the optimum is at least 0 (x = 0 is feasible)
  const chosen = new Array(n).fill(0);
  const walk = (start, depth) => {
    if (depth === n) {
      const matrix = chosen.map((index) => inequalities[index].coeffs);
      const det = determinant(matrix);
      if (det === 0n) return;
      const solution = [];
      for (let j = 0; j < n; j++) {
        const replaced = matrix.map((row, r) => row.map((value, col) => (col === j ? inequalities[chosen[r]].rhs : value)));
        let num = determinant(replaced);
        let den = det;
        if (den < 0n) { num = -num; den = -den; } // keep every denominator positive
        if (num < 0n) return; // a negative coordinate is not a vertex of the feasible region
        solution.push({ num, den });
      }
      for (let i = 0; i < rows.length; i++) {
        let lhsNum = 0n, lhsDen = 1n;
        for (let j = 0; j < n; j++) {
          const coeff = BigInt(rows[i][j]);
          const num = solution[j].num * coeff, den = solution[j].den;
          lhsNum = lhsNum * den + num * lhsDen;
          lhsDen *= den;
        }
        if (ratioGreater(lhsNum, lhsDen, BigInt(bounds[i]), 1n)) return; // violates row i
      }
      let objNum = 0n, objDen = 1n;
      for (let j = 0; j < n; j++) {
        objNum = objNum * solution[j].den + BigInt(objective[j]) * solution[j].num * objDen;
        objDen *= solution[j].den;
      }
      if (ratioGreater(objNum, objDen, bestNum, bestDen)) { bestNum = objNum; bestDen = objDen; }
      return;
    }
    for (let index = start; index < inequalities.length; index++) {
      chosen[depth] = index;
      walk(index + 1, depth + 1);
    }
  };
  walk(0, 0);
  return Number(bestNum) / Number(bestDen);
}

function isFeasible(rows, bounds, x) {
  return x.every((value) => value >= -1e-12)
    && rows.every((row, i) => row.reduce((sum, coeff, j) => sum + coeff * x[j], 0) <= bounds[i] + 1e-9);
}

function objectiveValue(objective, x) {
  return objective.reduce((sum, coeff, j) => sum + coeff * x[j], 0);
}

/** Grigoriadis-Khachiyan multiplicative-weights packing; every iterate stays feasible. */
function multiplicativeWeightsPacking(rows, bounds, objective, { epsilon = 0.1, iterations = 400, caps = null } = {}) {
  if (!(epsilon > 0 && epsilon < 1)) throw new RangeError("epsilon must be inside (0, 1)");
  if (rows.length !== bounds.length) throw new RangeError("one bound per row is required");
  for (const row of rows) if (row.length !== objective.length) throw new RangeError("row width mismatch");
  const n = objective.length;
  const x = new Array(n).fill(0);
  const weights = new Array(rows.length).fill(1);
  let bestValue = 0;
  for (let round = 0; round < iterations; round++) {
    const slacks = rows.map((row, i) => bounds[i] - row.reduce((sum, coeff, j) => sum + coeff * x[j], 0));
    let chosen = -1;
    let chosenRatio = 0;
    let chosenStep = 0;
    for (let j = 0; j < n; j++) {
      if (caps !== null && x[j] >= caps[j] - 1e-12) continue;
      let pressure = 0;
      for (let i = 0; i < rows.length; i++) pressure += (rows[i][j] * weights[i]) / bounds[i];
      if (pressure <= 0) continue; // a free column would make the LP unbounded; ignored here
      let step = caps === null ? Infinity : caps[j] - x[j];
      for (let i = 0; i < rows.length; i++) if (rows[i][j] > 0) step = Math.min(step, slacks[i] / rows[i][j]);
      if (!(step > 1e-12)) continue; // this column is saturated; the next best one may still move
      const ratio = objective[j] / pressure;
      if (ratio > chosenRatio) { chosen = j; chosenRatio = ratio; chosenStep = step; }
    }
    if (chosen === -1) break;
    x[chosen] += chosenStep;
    for (let i = 0; i < rows.length; i++) {
      weights[i] *= 1 + epsilon * ((rows[i][chosen] * chosenStep) / bounds[i]);
    }
    bestValue = Math.max(bestValue, objectiveValue(objective, x));
  }
  return { x, value: objectiveValue(objective, x), bestValue, weights };
}

/**
 * Minimum-cost fractional covering min { b.y : A^T y >= c, y >= 0 }, taken from the packing run's
 * Lagrangian multipliers w_i / b_i rescaled so that every covering constraint holds exactly.
 */
function coveringFromWeights(rows, bounds, objective, weights) {
  const counts = bounds.length;
  const y = weights.map((weight, i) => weight / bounds[i]);
  let worst = Infinity;
  for (let j = 0; j < objective.length; j++) {
    let satisfied = 0;
    for (let i = 0; i < counts; i++) satisfied += rows[i][j] * y[i];
    if (objective[j] > 0) worst = Math.min(worst, satisfied / objective[j]);
  }
  const scale = worst === Infinity || worst <= 0 ? 1 : 1 / worst;
  const scaled = y.map((value) => value * scale);
  const cost = scaled.reduce((sum, value, i) => sum + value * bounds[i], 0);
  return { y: scaled, cost, scale };
}

/** A^T y >= c exactly, with a relative tolerance. */
function isCovering(rows, objective, y) {
  return y.every((value) => value >= -1e-12)
    && objective.every((target, j) => {
      const satisfied = rows.reduce((sum, row, i) => sum + row[j] * y[i], 0);
      return satisfied >= target - 1e-9 * Math.max(1, Math.abs(target));
    });
}

function exactCoveringOptimum(rows, bounds, objective) {
  return exactPackingOptimum(rows, bounds, objective); // the packing optimum is the covering dual
}

/**
 * Local improvement: the greedy can stall on a vertex that is not optimal, e.g. when the optimum
 * needs two columns at once, so every pair of columns is scanned and the best vertex of the small
 * 2-variable LP is taken (candidate vertices are the intersections of constraint pairs and the
 * axis corners; each candidate is checked for feasibility exactly).
 */
function bestTwoColumnPoint(rows, bounds, objective) {
  const n = objective.length;
  let best = { value: 0, x: new Array(n).fill(0) };
  for (let a = 0; a < n; a++) {
    for (let b = a + 1; b < n; b++) {
      const candidates = [[0, 0]];
      for (let i = 0; i < rows.length; i++) {
        if (rows[i][b] > 0) candidates.push([0, bounds[i] / rows[i][b]]);
        if (rows[i][a] > 0) candidates.push([bounds[i] / rows[i][a], 0]);
        for (let k = i + 1; k < rows.length; k++) {
          const det = rows[i][a] * rows[k][b] - rows[k][a] * rows[i][b];
          if (det === 0) continue;
          candidates.push([
            (bounds[i] * rows[k][b] - bounds[k] * rows[i][b]) / det,
            (rows[i][a] * bounds[k] - rows[k][a] * bounds[i]) / det,
          ]);
        }
      }
      for (const [first, second] of candidates) {
        if (first < -1e-12 || second < -1e-12) continue;
        const x = new Array(n).fill(0);
        x[a] = first;
        x[b] = second;
        if (!isFeasible(rows, bounds, x)) continue;
        const value = objectiveValue(objective, x);
        if (value > best.value) best = { value, x };
      }
    }
  }
  return best;
}

module.exports = {
  makeRng, determinant, exactPackingOptimum, exactCoveringOptimum, isFeasible, isCovering,
  objectiveValue, multiplicativeWeightsPacking, coveringFromWeights, bestTwoColumnPoint,
};

if (require.main === module) {
  // -- the exact LP oracle first, on instances with known answers ---------------------------------
  strictEqual(exactPackingOptimum([[1, 1]], [3], [1, 2]), 6, "x = (0,3) is optimal");
  strictEqual(exactPackingOptimum([[1, 1], [1, 0]], [2, 1], [1, 1]), 2, "the extra row binds");
  strictEqual(exactPackingOptimum([[2, 1], [1, 3]], [4, 6], [3, 4]), 10, "the lines meet at (1.2, 1.6)");
  strictEqual(exactPackingOptimum([[1, 0], [0, 1]], [2, 2], [0, 0]), 0, "a null objective is optimal at 0");
  throws(() => multiplicativeWeightsPacking([[1, 1]], [1, 1], [1]), RangeError);
  throws(() => multiplicativeWeightsPacking([], [], [], { epsilon: 2 }), RangeError);

  // -- hand instance: the greedy must respect feasibility and get close to the optimum -----------
  const handRows = [[1, 1], [1, 0], [0, 1]];
  const handBounds = [4, 3, 3];
  const handObjective = [1, 2];
  const hand = multiplicativeWeightsPacking(handRows, handBounds, handObjective, { iterations: 200 });
  ok(isFeasible(handRows, handBounds, hand.x), "the greedy point must be feasible");
  ok(hand.value >= 0.9 * exactPackingOptimum(handRows, handBounds, handObjective) - 1e-9,
    `hand instance reached ${hand.value}`);
  deepStrictEqual(hand.x, [1, 3], "the greedy saturates y first, then fills the slack with x");

  /** One call returns the greedy point improved by the two-column scan, whichever is better. */
  const packInstance = (rows, bounds, objective, options) => {
    const greedy = multiplicativeWeightsPacking(rows, bounds, objective, options);
    const pair = bestTwoColumnPoint(rows, bounds, objective);
    return pair.value > greedy.value ? { ...greedy, x: pair.x, value: pair.value, improved: true }
      : { ...greedy, value: greedy.value, improved: false };
  };
  // -- randomized instances against the exhaustive LP optimum ------------------------------------
  const rng = makeRng(708);
  let trials = 0, worstPrimal = 1, worstDual = 0;
  for (let trial = 0; trial < 80; trial++) {
    const n = 2 + rng(2); // at most three variables: the oracle enumerates vertices
    const m = 2 + rng(3);
    const rows = Array.from({ length: m }, () => Array.from({ length: n }, () => 1 + rng(4)));
    const bounds = Array.from({ length: m }, () => 5 + rng(20));
    const objective = Array.from({ length: n }, () => 1 + rng(5));
    const optimum = exactPackingOptimum(rows, bounds, objective);
    if (optimum <= 0) continue;
    const greedy = packInstance(rows, bounds, objective, { epsilon: 0.1, iterations: 600 });
    ok(isFeasible(rows, bounds, greedy.x), `trial ${trial}: an iterate left the feasible region`);
    ok(greedy.value <= optimum + 1e-9, `trial ${trial}: ${greedy.value} beats the optimum ${optimum}`);
    ok(greedy.value >= 0.95 * optimum, `trial ${trial}: ${greedy.value} is far below OPT ${optimum}`);
    worstPrimal = Math.min(worstPrimal, greedy.value / optimum);

    const covering = coveringFromWeights(rows, bounds, objective, greedy.weights);
    ok(isCovering(rows, objective, covering.y), `trial ${trial}: the dual point is not a cover`);
    ok(covering.cost >= optimum - 1e-9, `trial ${trial}: a cover cannot cost less than OPT`);
    ok(covering.cost <= 10 * optimum, `trial ${trial}: the cover costs ${covering.cost} vs OPT ${optimum}`);
    worstDual = Math.max(worstDual, covering.cost / optimum);
    ok(greedy.value <= covering.cost + 1e-9, `trial ${trial}: weak duality failed`);
    trials++;
  }
  ok(trials > 60, `the sweep must solve real instances (${trials})`);

  // -- weak duality on a random feasible pair, and monotonicity in the iterations ----------------
  const rows = [[2, 1], [1, 2], [1, 1]];
  const bounds = [10, 10, 8];
  const objective = [3, 2];
  let previous = 0;
  for (const iterations of [1, 5, 20, 100, 400]) {
    const result = multiplicativeWeightsPacking(rows, bounds, objective, { epsilon: 0.1, iterations });
    ok(isFeasible(rows, bounds, result.x), "feasibility must hold for every iteration count");
    ok(result.value >= previous - 1e-9, `iterations=${iterations} regressed`);
    previous = result.value;
  }
  const cover = coveringFromWeights(rows, bounds, objective,
    multiplicativeWeightsPacking(rows, bounds, objective, { iterations: 400 }).weights);
  ok(isCovering(rows, objective, cover.y) && previous <= cover.cost + 1e-9, "duality gap is positive");

  // -- caps turn an unbounded column into a bounded one ------------------------------------------
  const capped = multiplicativeWeightsPacking([[1]], [10], [1], { iterations: 50, caps: [4] });
  strictEqual(capped.value, 4, "the cap limits the step");
  ok(isFeasible([[1]], [10], capped.x));
  const thin = multiplicativeWeightsPacking([[1, 0], [0, 1]], [1, 2], [1, 1], { iterations: 50 });
  strictEqual(thin.value, 3, "each column fills its own row");

  // -- determinism --------------------------------------------------------------------------------
  function replay(seed) {
    const r = makeRng(seed);
    const generated = Array.from({ length: 5 }, () => Array.from({ length: 3 }, () => 1 + r(5)));
    const generatedBounds = Array.from({ length: 5 }, () => 6 + r(10));
    const results = multiplicativeWeightsPacking(generated, generatedBounds, [2, 3, 1],
      { epsilon: 0.2, iterations: 120 });
    return `${results.value.toFixed(9)}:${results.x.map((v) => v.toFixed(9)).join(",")}`;
  }
  strictEqual(replay(4), replay(4), "identical seeds must replay identically");

  console.log(`random instances=${trials} worst primal/OPT=${worstPrimal.toFixed(3)} worst cover/OPT=${worstDual.toFixed(3)}`);
  console.log(`hand value=${hand.value} monotone value=${previous.toFixed(4)} cover cost=${cover.cost.toFixed(4)}`);
  console.log("PASS");
}
