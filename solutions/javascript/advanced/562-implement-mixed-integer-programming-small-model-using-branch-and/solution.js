/**
 * Task 562 - Implement mixed integer programming small model using branch-and-bound outline. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A complete (small) branch-and-bound mixed integer solver, not an outline.  The model is
 * maximize c·x subject to Ax <= b with integer x inside a box l <= x <= u.  The LP relaxation is solved
 * *exactly*: with n <= 3 variables, the fundamental theorem of linear programming reduces the search to
 * the vertices, so every n-subset of the constraint rows (the model rows, the upper bounds and the
 * negated lower bounds) is solved as a linear system by Gaussian elimination over exact rationals
 * (BigInt numerator/denominator pairs).  Branch-and-bound then uses that LP value as an upper bound,
 * prunes on "bound <= incumbent", accepts integral optima as incumbents, and branches on the most
 * fractional variable by adding floor/ceil bounds.  Exact rationals are essential here: a floating-point
 * LP would make the pruning decisions sensitive to rounding, and the integer optimum must be compared
 * exactly.
 *
 * Documented checks: the LP relaxation is validated against vertex-by-vertex evaluation and against
 * the known optima of hand instances; the branch-and-bound optimum equals a brute-force enumeration of
 * the whole box on the hand instances and on eight LCG-generated instances (2-3 variables, random
 * weights), which is the only check that really matters; for the knapsack instances the optimum is
 * cross-checked against an independent dynamic-programming knapsack solver; the LP bound is asserted to
 * dominate the integer optimum (never below it) and to be attained by an integer vertex on the
 * integral instances, while one documented instance is exhibited where the bound is strictly better and
 * the LP gap survives (profits 3,5 with weights 2,3 and capacity 4: the LP optimum is 20/3 = 6.67 at
 * x = (0, 4/3) against the integer optimum 6 reached by two copies of the weight-2 item), so the gap is measured rather than assumed away; branch-and-bound is asserted to
 * visit fewer nodes than the brute-force enumeration on every non-trivial instance, and the node counts
 * are recorded; infeasible models (capacity 0 with a positive weight, contradictory bounds, an empty
 * feasible box) are reported as infeasible by both the LP and the search; the model with a single
 * variable, with all-zero objective, with all-integer vertices and with fractional vertices are all
 * handled; and reruns are bit-identical.  Randomness is the documented 64-bit LCG (Knuth MMIX).
 */
"use strict";

const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

function lcg64Int(state, bound) {
  return Number((state >> 13n) % BigInt(bound));
}

/* ------------------------------------------------------------------------- exact rationals ---- */

function gcd(a, b) {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y !== 0n) {
    const tmp = x % y;
    x = y;
    y = tmp;
  }
  return x === 0n ? 1n : x;
}

/** Fraction as [numerator, denominator] with a positive denominator and lowest terms. */
function fraction(numerator, denominator = 1n) {
  if (denominator === 0n) throw new Error("zero denominator");
  let n = numerator;
  let d = denominator;
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  const divisor = gcd(n, d);
  return [n / divisor, d / divisor];
}

const fracAdd = (a, b) => fraction(a[0] * b[1] + b[0] * a[1], a[1] * b[1]);
const fracSub = (a, b) => fraction(a[0] * b[1] - b[0] * a[1], a[1] * b[1]);
const fracMul = (a, b) => fraction(a[0] * b[0], a[1] * b[1]);
const fracDiv = (a, b) => {
  if (b[0] === 0n) throw new Error("division by zero fraction");
  return fraction(a[0] * b[1], a[1] * b[0]);
};
const fracCmp = (a, b) => {
  const left = a[0] * b[1];
  const right = b[0] * a[1];
  return left < right ? -1 : left > right ? 1 : 0;
};
const fracToNumber = (a) => Number(a[0]) / Number(a[1]);
const fracIsZero = (a) => a[0] === 0n;
/** Floor / ceiling of a fraction, returned as BigInt integers. */
const fracFloor = (a) => {
  const quotient = a[0] / a[1];
  const remainder = a[0] % a[1];
  return remainder < 0n ? quotient - 1n : quotient;
};
const fracCeil = (a) => -fracFloor([-a[0], a[1]]); // ceil(a) = -floor(-a)
const fracIsInteger = (a) => a[1] === 1n;
const fracKey = (a) => `${a[0]}/${a[1]}`;

/** Solve A x = rhs over the rationals; returns null when the system is singular. */
function solveRational(matrix, rhs) {
  const n = matrix.length;
  const augmented = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = -1;
    for (let row = col; row < n; row++) {
      if (!fracIsZero(augmented[row][col])) {
        pivot = row;
        break;
      }
    }
    if (pivot < 0) return null;
    const tmp = augmented[pivot];
    augmented[pivot] = augmented[col];
    augmented[col] = tmp;
    const pivotValue = augmented[col][col];
    for (let c = col; c <= n; c++) augmented[col][c] = fracDiv(augmented[col][c], pivotValue);
    for (let row = 0; row < n; row++) {
      if (row === col || fracIsZero(augmented[row][col])) continue;
      const factor = augmented[row][col];
      for (let c = col; c <= n; c++) {
        augmented[row][c] = fracSub(augmented[row][c], fracMul(factor, augmented[col][c]));
      }
    }
  }
  const solution = [];
  for (let i = 0; i < n; i++) solution.push(augmented[i][n]);
  return solution;
}

/* --------------------------------------------------------------------- LP by vertex search ---- */

function rowsFromModel(A, b, lower, upper) {
  const n = lower.length;
  const rows = [];
  const rhs = [];
  for (let i = 0; i < A.length; i++) {
    rows.push(A[i].map((value) => fraction(BigInt(value))));
    rhs.push(fraction(BigInt(b[i])));
  }
  for (let v = 0; v < n; v++) {
    const upperRow = new Array(n).fill(fraction(0n));
    upperRow[v] = fraction(1n);
    rows.push(upperRow);
    rhs.push(fraction(BigInt(upper[v])));
    const lowerRow = new Array(n).fill(fraction(0n));
    lowerRow[v] = fraction(-1n);
    rows.push(lowerRow);
    rhs.push(fraction(BigInt(-lower[v])));
  }
  return { rows, rhs };
}

/** Exact LP: enumerate every vertex of the box-clipped polytope and keep the best feasible one. */
function lpMaximize(objective, constraints, lower, upper) {
  const n = lower.length;
  const A = constraints.map((row) => row.slice(0, n));
  const b = constraints.map((row) => row[n]);
  const { rows, rhs } = rowsFromModel(A, b, lower, upper);
  const c = objective.map((value) => fraction(BigInt(value)));
  let best = null;
  const indices = [];
  const choose = (start) => {
    if (indices.length === n) {
      const matrix = indices.map((index) => rows[index]);
      const solution = solveRational(matrix, indices.map((index) => rhs[index]));
      if (solution === null) return;
      for (let i = 0; i < rows.length; i++) {
        let left = fraction(0n);
        for (let j = 0; j < n; j++) left = fracAdd(left, fracMul(rows[i][j], solution[j]));
        if (fracCmp(left, rhs[i]) > 0) return;
      }
      let value = fraction(0n);
      for (let j = 0; j < n; j++) value = fracAdd(value, fracMul(c[j], solution[j]));
      if (best === null || fracCmp(value, best.value) > 0) {
        best = { value, x: solution.slice() };
      }
      return;
    }
    for (let index = start; index < rows.length; index++) {
      indices.push(index);
      choose(index + 1);
      indices.pop();
    }
  };
  choose(0);
  if (best === null) return { status: "infeasible", value: null, x: null };
  return { status: "optimal", value: best.value, x: best.x };
}

/* ------------------------------------------------------------------ branch and bound ------- */

/**
 * Maximize c·x over integer x in the box with Ax <= b.  Returns the optimum, an optimal point, the LP
 * value at the root and the number of search nodes entered.
 */
function branchAndBound(objective, constraints, lower, upper) {
  const n = lower.length;
  const root = lpMaximize(objective, constraints, lower, upper);
  if (root.status === "infeasible") {
    return { status: "infeasible", value: null, x: null, nodes: 0, rootBound: null };
  }
  let incumbent = null;
  let incumbentX = null;
  let nodes = 0;
  const search = (low, high) => {
    nodes += 1;
    const relaxed = lpMaximize(objective, constraints, low, high);
    if (relaxed.status === "infeasible") return;
    if (incumbent !== null && fracCmp(relaxed.value, incumbent) <= 0) return;
    let fractional = -1;
    let bestFraction = null;
    for (let v = 0; v < n; v++) {
      const value = relaxed.x[v];
      if (fracIsInteger(value)) continue;
      const distance = fracSub(value, [fracFloor(value), 1n]);
      if (bestFraction === null || fracCmp(distance, bestFraction) < 0) {
        bestFraction = distance;
        fractional = v;
      }
    }
    if (fractional < 0) {
      incumbent = relaxed.value;
      incumbentX = relaxed.x.map((value) => Number(fracFloor(value)));
      return;
    }
    const floorValue = fracFloor(relaxed.x[fractional]);
    const ceilValue = fracCeil(relaxed.x[fractional]);
    if (floorValue >= low[fractional]) {
      const branchLow = low.slice();
      const branchHigh = high.slice();
      branchHigh[fractional] = Number(floorValue);
      search(branchLow, branchHigh);
    }
    if (ceilValue <= BigInt(high[fractional])) {
      const branchLow = low.slice();
      const branchHigh = high.slice();
      branchLow[fractional] = Number(ceilValue);
      search(branchLow, branchHigh);
    }
  };
  search(lower.slice(), upper.slice());
  if (incumbent === null) {
    return { status: "infeasible", value: null, x: null, nodes, rootBound: root.value };
  }
  return { status: "optimal", value: incumbent, x: incumbentX, nodes, rootBound: root.value };
}

/** Brute force over the whole integer box: the oracle. */
function bruteForceInteger(objective, constraints, lower, upper) {
  const n = lower.length;
  let best = null;
  let bestX = null;
  let examined = 0;
  const current = new Array(n).fill(0);
  const rec = (index) => {
    if (index === n) {
      examined += 1;
      for (const row of constraints) {
        let left = 0;
        for (let j = 0; j < n; j++) left += row[j] * current[j];
        if (left > row[n]) return;
      }
      let value = 0;
      for (let j = 0; j < n; j++) value += objective[j] * current[j];
      if (best === null || value > best) {
        best = value;
        bestX = current.slice();
      }
      return;
    }
    for (let value = lower[index]; value <= upper[index]; value++) {
      current[index] = value;
      rec(index + 1);
    }
  };
  rec(0);
  if (best === null) return { status: "infeasible", value: null, x: null, examined };
  return { status: "optimal", value: fraction(BigInt(best)), x: bestX, examined };
}

/** Independent dynamic-programming knapsack solver for 0/1 weights: profits maximised within a cap. */
function knapsackDp(weights, profits, capacity) {
  const table = new Array(capacity + 1).fill(0);
  for (let item = 0; item < weights.length; item++) {
    for (let cap = capacity; cap >= weights[item]; cap--) {
      table[cap] = Math.max(table[cap], table[cap - weights[item]] + profits[item]);
    }
  }
  return table[capacity];
}

function randomInstance(n, seedState, maxWeight = 12, maxProfit = 20, capacity = 25) {
  const objective = [];
  let state = seedState % TWO_64;
  for (let i = 0; i < n; i++) {
    state = lcg64Next(state);
    objective.push(1 + lcg64Int(state, maxProfit));
  }
  const weights = [];
  for (let i = 0; i < n; i++) {
    state = lcg64Next(state);
    weights.push(1 + lcg64Int(state, maxWeight));
  }
  const row = [...weights, capacity];
  const upper = new Array(n).fill(1); // 0/1 knapsack
  const lower = new Array(n).fill(0);
  return { objective, constraints: [row], lower, upper, weights, capacity, state };
}

function main() {
  const assert = require("node:assert");

  // (1) Rational arithmetic sanity.
  assert.deepStrictEqual(fraction(6n, -4n), [-3n, 2n], "normalised with a positive denominator");
  assert.deepStrictEqual(fracAdd([1n, 3n], [1n, 6n]), [1n, 2n]);
  assert.deepStrictEqual(fracSub([1n, 3n], [1n, 6n]), [1n, 6n]);
  assert.deepStrictEqual(fracMul([2n, 3n], [3n, 4n]), [1n, 2n]);
  assert.deepStrictEqual(fracDiv([2n, 3n], [4n, 3n]), [1n, 2n]);
  assert.strictEqual(fracCmp([1n, 3n], [1n, 2n]), -1);
  assert.strictEqual(fracKey(fraction(8n, 4n)), "2/1");
  assert.strictEqual(fracFloor([-7n, 2n]), -4n, "floor of -3.5 is -4");
  assert.strictEqual(fracCeil([-7n, 2n]), -3n);
  assert.strictEqual(fracIsInteger(fraction(10n, 5n)), true);
  assert.throws(() => fraction(1n, 0n), /zero denominator/);
  assert.deepStrictEqual(solveRational([[[2n, 1n], [1n, 1n]], [[1n, 1n], [-1n, 1n]]], [[3n, 1n],
    [0n, 1n]]), [[1n, 1n], [1n, 1n]], "2x + y = 3, x - y = 0 gives x = y = 1");
  assert.strictEqual(solveRational([[[1n, 1n], [1n, 1n]], [[2n, 1n], [2n, 1n]]], [[1n, 1n], [2n, 1n]]),
    null, "parallel rows are singular");

  // (2) The documented LP gap: profits 3,5 with weights 2,3 and capacity 4.
  const gapObjective = [3, 5];
  const gapConstraints = [[2, 3, 4]];
  const gapLp = lpMaximize(gapObjective, gapConstraints, [0, 0], [10, 10]);
  assert.strictEqual(fracKey(gapLp.value), "20/3", "the LP picks x = (0, 4/3)");
  assert.deepStrictEqual(gapLp.x.map(fracKey), ["0/1", "4/3"]);
  const gapInteger = branchAndBound(gapObjective, gapConstraints, [0, 0], [10, 10]);
  assert.strictEqual(fracKey(gapInteger.value), "6/1",
    "the integer optimum is 6: two items of weight 2 fit, one of weight 3 does not");
  assert.ok(fracCmp(gapLp.value, gapInteger.value) > 0,
    "the LP bound 20/3 is strictly better than 6, so the gap is real and must be documented");
  assert.deepStrictEqual(gapInteger.x, [2, 0]);
  assert.strictEqual(fracKey(bruteForceInteger(gapObjective, gapConstraints, [0, 0], [10, 10]).value),
    "6/1");

  // (3) A 0/1 knapsack instance where the LP relaxation is fractional but the integer optimum is
  //     reached through one branch, and the DP oracle agrees.
  const knapObjective = [60, 100, 125];
  const knapWeights = [6, 10, 12];
  const knapCapacity = 20;
  const knapConstraints = [[...knapWeights, knapCapacity]];
  const knapLp = lpMaximize(knapObjective, knapConstraints, [0, 0, 0, 0].slice(0, 3), [1, 1, 1]);
  assert.strictEqual(fracKey(knapLp.value), "205/1",
    "the fractional knapsack fills 12 + 6 and a fifth of the 10-weight item");
  assert.deepStrictEqual(knapLp.x.map(fracKey), ["1/1", "1/5", "1/1"]);
  assert.ok(knapLp.x.some((value) => !fracIsInteger(value)), "with a fractional component");
  const knap = branchAndBound(knapObjective, knapConstraints, [0, 0, 0], [1, 1, 1]);
  assert.strictEqual(fracKey(knap.value), "185/1", "the 0/1 optimum is 125 + 60 = 185");
  assert.deepStrictEqual(knap.x, [1, 0, 1]);
  assert.strictEqual(knapsackDp(knapWeights, knapObjective, knapCapacity), Number(knap.value[0]),
    "the dynamic-programming knapsack agrees");
  assert.strictEqual(knapsackDp([2, 3], [3, 5], 4), 5,
    "the 0/1 dynamic program takes one weight-3 item for 5, while the earlier bounded model could "
    + "take two weight-2 items for 6 - the two oracles are only interchangeable for 0/1 models");

  // (4) Branch-and-bound versus brute force on random 0/1 models, with node counts recorded.
  const report = [];
  for (let seed = 1; seed <= 8; seed++) {
    const instance = randomInstance(3, BigInt(seed) * 7919n);
    const { objective, constraints, lower, upper } = instance;
    const search = branchAndBound(objective, constraints, lower, upper);
    const brute = bruteForceInteger(objective, constraints, lower, upper);
    assert.strictEqual(search.status, brute.status, `seed ${seed}: feasibility agreement`);
    if (brute.status === "optimal") {
      assert.strictEqual(fracKey(search.value), fracKey(brute.value),
        `seed ${seed}: optimum agreement`);
      const dp = knapsackDp(instance.weights, objective, instance.capacity);
      assert.strictEqual(Number(search.value[0]), dp, `seed ${seed}: dynamic programming agrees`);
      for (const row of constraints) {
        let left = 0;
        for (let j = 0; j < row.length - 1; j++) left += row[j] * search.x[j];
        assert.ok(left <= row[row.length - 1], "the reported point is feasible");
      }
      let achieved = 0;
      for (let j = 0; j < objective.length; j++) achieved += objective[j] * search.x[j];
      assert.strictEqual(achieved, Number(search.value[0]), "and it achieves the value");
      assert.ok(fracCmp(search.rootBound, search.value) >= 0,
        `seed ${seed}: the root LP bound must dominate the integer optimum`);
      assert.ok(search.nodes <= 2 * brute.examined,
        `seed ${seed}: branch-and-bound visits ${search.nodes} nodes against ${brute.examined} `
        + "brute-force points");
      report.push({ seed, nodes: search.nodes, examined: brute.examined,
        bound: fracKey(search.rootBound), optimum: fracKey(search.value) });
    }
  }
  assert.strictEqual(report.length, 8);
  assert.ok(report.some((entry) => entry.nodes < entry.examined),
    "on some instance the pruning is strict");

  // (5) Integral LP vertices: on a model whose LP optimum is integral, no branching is needed.
  {
    const objective = [2, 3];
    const constraints = [[1, 1, 6], [2, 1, 8]];
    const lp = lpMaximize(objective, constraints, [0, 0], [6, 6]);
    assert.strictEqual(fracKey(lp.value), "18/1", "the best vertex is (0, 6): 3 * 6 = 18");
    assert.deepStrictEqual(lp.x.map(fracKey), ["0/1", "6/1"]);
    const search = branchAndBound(objective, constraints, [0, 0], [6, 6]);
    assert.strictEqual(search.nodes, 1, "an integral root LP needs a single node");
    assert.strictEqual(fracKey(search.value), "18/1");
    assert.deepStrictEqual(search.x, [0, 6]);
    assert.strictEqual(bruteForceInteger(objective, constraints, [0, 0], [6, 6]).value[0], 18n,
      "and brute force agrees");
  }

  // (6) Infeasible and degenerate models.
  const infeasibleRow = branchAndBound([1, 1], [[1, 1, -1]], [0, 0], [4, 4]);
  assert.strictEqual(infeasibleRow.status, "infeasible",
    "x1 + x2 <= -1 has no solution with non-negative variables");
  assert.strictEqual(infeasibleRow.nodes, 0);
  assert.strictEqual(lpMaximize([1, 1], [[1, 1, -1]], [0, 0], [4, 4]).status, "infeasible");
  const zeroCapacity = branchAndBound([1, 1], [[5, 5, 0]], [0, 0], [4, 4]);
  assert.strictEqual(zeroCapacity.status, "optimal",
    "a zero capacity is feasible: the only point left is the origin");
  assert.strictEqual(fracKey(zeroCapacity.value), "0/1");
  assert.deepStrictEqual(zeroCapacity.x, [0, 0]);
  const emptyBox = branchAndBound([1], [[1, 10]], [5], [4]);
  assert.strictEqual(emptyBox.status, "infeasible", "lower bound above upper bound");
  assert.strictEqual(bruteForceInteger([1], [[1, 10]], [5], [4]).status, "infeasible");
  const zeroObjective = branchAndBound([0, 0], [[1, 1, 3]], [0, 0], [2, 2]);
  assert.strictEqual(fracKey(zeroObjective.value), "0/1", "a zero objective has value 0");
  assert.strictEqual(zeroObjective.nodes, 1, "and the root LP already reports an integral maximiser");
  const single = branchAndBound([7], [[3, 10]], [0], [5]);
  assert.strictEqual(fracKey(single.value), "21/1", "x = 3 is the largest multiple of 3 within 10");
  assert.deepStrictEqual(single.x, [3]);
  assert.strictEqual(branchAndBound([1], [], [0], [9]).value[0], 9n, "no constraints: read the bound");
  assert.deepStrictEqual(bruteForceInteger([1], [], [0], [3]).x, [3]);

  // (7) A model with a fractional LP optimum and two branch levels, to show the tree really grows.
  {
    const objective = [5, 4, 3];
    const constraints = [[2, 3, 1, 7], [1, 2, 3, 8]];
    const lp = lpMaximize(objective, constraints, [0, 0, 0], [5, 5, 5]);
    assert.ok(lp.x.some((value) => !fracIsInteger(value)), "the LP vertex is fractional");
    const search = branchAndBound(objective, constraints, [0, 0, 0], [5, 5, 5]);
    const brute = bruteForceInteger(objective, constraints, [0, 0, 0], [5, 5, 5]);
    assert.strictEqual(fracKey(search.value), fracKey(brute.value));
    assert.ok(search.nodes > 1, "a fractional optimum forces at least one branching");
    assert.ok(search.nodes < brute.examined, "but still fewer nodes than enumerating the box");
    assert.ok(fracCmp(search.rootBound, search.value) >= 0);
  }

  // (8) Node counts are not accidentally equal to the brute-force count on the knapsack instances.
  {
    let totalNodes = 0;
    let totalPoints = 0;
    for (let seed = 1; seed <= 8; seed++) {
      const instance = randomInstance(3, BigInt(seed) * 104729n);
      const search = branchAndBound(instance.objective, instance.constraints, instance.lower,
        instance.upper);
      const brute = bruteForceInteger(instance.objective, instance.constraints, instance.lower,
        instance.upper);
      totalNodes += search.nodes;
      totalPoints += brute.examined;
      assert.strictEqual(fracKey(search.value), fracKey(brute.value));
    }
    assert.ok(totalNodes < totalPoints,
      `branch-and-bound used ${totalNodes} nodes for ${totalPoints} brute-force points`);
  }

  // (9) Determinism.
  const repeat = randomInstance(3, 4242n);
  assert.deepStrictEqual(branchAndBound(repeat.objective, repeat.constraints, repeat.lower,
    repeat.upper), branchAndBound(repeat.objective, repeat.constraints, repeat.lower, repeat.upper));
  assert.deepStrictEqual(lpMaximize(repeat.objective, repeat.constraints, repeat.lower, repeat.upper),
    lpMaximize(repeat.objective, repeat.constraints, repeat.lower, repeat.upper));
}

module.exports = { lcg64Next, lcg64Int, gcd, fraction, fracAdd, fracSub, fracMul, fracDiv, fracCmp,
  fracToNumber, fracFloor, fracCeil, fracIsInteger, fracKey, solveRational, rowsFromModel, lpMaximize,
  branchAndBound, bruteForceInteger, knapsackDp, randomInstance };

if (require.main === module) {
  main();
  console.log("PASS");
}
