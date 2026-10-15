/**
 * Task 377 - Simplex algorithm for small LPs (outline). (Tier: Intermediate, CTCI Ch 6)
 *
 * Standard form: maximise c.x subject to A x <= b, x >= 0.  The tableau simplex starts
 * from the slack basis (x = 0) when b >= 0.  Rows with b_i < 0 are negated (their slack
 * coefficient becomes -1) and get an artificial variable; a phase-1 objective (-1 on the
 * artificial columns, i.e. minimising their sum) drives them out, and a positive
 * phase-1 optimum means the LP is infeasible.  Phase 2 then runs the real objective with
 * the artificial columns forbidden from entering.
 *
 * Bland's rule is used for both iterations: the entering variable is the smallest index
 * with a positive reduced cost and the leaving variable breaks ratio ties by the
 * smallest basic-variable index.  This guarantees termination (no cycling); an
 * iteration cap is kept as a failure sentinel and is never hit by the test problems.
 *
 * Documented sentinels (the API returns a status string; x is null and value is NaN
 * unless "optimal"):
 *   * "optimal"    -> x is the optimal vertex,
 *   * "unbounded"  -> some entering column has no positive tableau coefficient,
 *   * "infeasible" -> a positive phase-1 optimum (the constraints contradict).
 *
 * Re-derivation note: the task brief suggests "max 3x + 2y s.t. x+y<=4, x+3y<=6 ->
 * x=3, y=1, value 11".  That point is the intersection of the two constraint lines, but
 * the vertex (4, 0) satisfies both constraints (4+0=4, 4+0<=6) and gives
 * 3*4 + 2*0 = 12 > 11, so the true optimum is 12 at (4, 0) - the code asserts 12.
 *
 * The objective values are small exact integers here; comparisons use the documented
 * tolerance 1e-9 because the tableau arithmetic is in doubles.  A Cramer-rule vertex
 * enumeration is used as an independent oracle for the LCG-generated bounded LPs.
 */
"use strict";

const EPS = 1e-9;
const MAX_ITERATIONS = 10000;

class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/** Primal simplex core (Bland's rule). Mutates the tableau; returns {status, value}. */
function runSimplex(tableau, rhs, basis, cost, canEnter) {
  const rows = tableau.length;
  const columns = cost.length;

  let reduced = cost.slice();
  let value = 0;
  for (let i = 0; i < rows; i++) {
    const basicCost = cost[basis[i]];
    if (basicCost !== 0) {
      value += basicCost * rhs[i];
      for (let j = 0; j < columns; j++) reduced[j] -= basicCost * tableau[i][j];
    }
  }

  let iterations = 0;
  for (;;) {
    iterations++;
    if (iterations > MAX_ITERATIONS) throw new Error("simplex iteration cap exceeded");
    let entering = -1;
    for (let j = 0; j < columns; j++) {                 // Bland: smallest improving index
      if (canEnter[j] && reduced[j] > EPS) {
        entering = j;
        break;
      }
    }
    if (entering === -1) return { status: "optimal", value };

    let leaving = -1;
    let bestRatio = 0;
    for (let i = 0; i < rows; i++) {
      if (tableau[i][entering] > EPS) {
        const ratio = rhs[i] / tableau[i][entering];
        if (leaving === -1 || ratio < bestRatio - 1e-12) {
          leaving = i;
          bestRatio = ratio;
        } else if (Math.abs(ratio - bestRatio) <= 1e-12 && basis[i] < basis[leaving]) {
          leaving = i;                                  // Bland ratio tie-break
          bestRatio = ratio;
        }
      }
    }
    if (leaving === -1) return { status: "unbounded", value: NaN };

    const pivot = tableau[leaving][entering];
    tableau[leaving] = tableau[leaving].map((v) => v / pivot);
    rhs[leaving] /= pivot;
    for (let i = 0; i < rows; i++) {
      if (i !== leaving && tableau[i][entering] !== 0) {
        const factor = tableau[i][entering];
        for (let j = 0; j < columns; j++) tableau[i][j] -= factor * tableau[leaving][j];
        rhs[i] -= factor * rhs[leaving];
      }
    }
    const factor = reduced[entering];
    for (let j = 0; j < columns; j++) reduced[j] -= factor * tableau[leaving][j];
    value += factor * rhs[leaving];
    basis[leaving] = entering;
  }
}

/** max c.x s.t. A x <= b, x >= 0.  Returns {status, x, value} (see sentinel doc). */
function simplex(objective, constraints, bounds) {
  const n = objective.length;
  const m = constraints.length;
  const negativeRows = [];
  for (let i = 0; i < m; i++) if (bounds[i] < 0) negativeRows.push(i);
  const artificialCount = negativeRows.length;
  const columns = n + m + artificialCount;

  const tableau = [];
  const rhs = [];
  const basis = [];
  const artificialOfRow = new Array(m).fill(-1);
  let nextArtificial = n + m;
  for (let i = 0; i < m; i++) {
    const row = new Array(columns).fill(0);
    if (bounds[i] >= 0) {
      for (let j = 0; j < n; j++) row[j] = constraints[i][j];
      row[n + i] = 1;                                   // slack basis
      basis.push(n + i);
      rhs.push(bounds[i]);
    } else {
      for (let j = 0; j < n; j++) row[j] = -constraints[i][j];
      row[n + i] = -1;
      row[nextArtificial] = 1;                          // artificial basis
      artificialOfRow[i] = nextArtificial;
      basis.push(nextArtificial);
      nextArtificial++;
      rhs.push(-bounds[i]);
    }
    tableau.push(row);
  }

  if (artificialCount > 0) {
    const phaseOneCost = new Array(columns).fill(0);
    for (let i = 0; i < m; i++) {
      if (artificialOfRow[i] !== -1) phaseOneCost[artificialOfRow[i]] = -1;
    }
    const outcome = runSimplex(tableau, rhs, basis, phaseOneCost, new Array(columns).fill(true));
    if (outcome.status !== "optimal") throw new Error("phase 1 must be bounded");
    if (outcome.value < -EPS) return { status: "infeasible", x: null, value: NaN };
  }

  const phaseTwoCost = new Array(columns).fill(0);
  for (let j = 0; j < n; j++) phaseTwoCost[j] = objective[j];
  const canEnter = new Array(columns).fill(false);
  for (let j = 0; j < n + m; j++) canEnter[j] = true;   // artificials stay out
  const outcome = runSimplex(tableau, rhs, basis, phaseTwoCost, canEnter);
  if (outcome.status !== "optimal") return { status: outcome.status, x: null, value: NaN };

  const x = new Array(n).fill(0);
  for (let i = 0; i < m; i++) {
    if (basis[i] < n) x[basis[i]] = rhs[i];
  }
  return { status: "optimal", x, value: outcome.value };
}

/** Exact-ish recursive determinant for tiny matrices (n <= 3 in this task). */
function determinant(matrix) {
  const size = matrix.length;
  if (size === 1) return matrix[0][0];
  let total = 0;
  for (let j = 0; j < size; j++) {
    const minor = matrix.slice(1).map((row) => row.filter((_, c) => c !== j));
    total += (j % 2 === 0 ? 1 : -1) * matrix[0][j] * determinant(minor);
  }
  return total;
}

function combinations(items, k) {
  if (k === 0) return [[]];
  if (items.length < k) return [];
  const result = [];
  const [first, ...rest] = items;
  for (const tail of combinations(rest, k - 1)) result.push([first, ...tail]);
  for (const tail of combinations(rest, k)) result.push(tail);
  return result;
}

/** Cramer-rule vertex enumeration for bounded feasible LPs (unique optimum value). */
function bruteForceLp(objective, constraints, bounds) {
  const n = objective.length;
  const m = constraints.length;
  const hyperplanes = [];
  const hyperRhs = [];
  for (let i = 0; i < m; i++) {
    hyperplanes.push(constraints[i].slice());
    hyperRhs.push(bounds[i]);
  }
  for (let j = 0; j < n; j++) {
    const unit = new Array(n).fill(0);
    unit[j] = 1;
    hyperplanes.push(unit);
    hyperRhs.push(0);
  }

  const tolerance = 1e-7;
  let best = null;
  for (const combo of combinations([...hyperplanes.keys()], n)) {
    const matrix = combo.map((i) => hyperplanes[i]);
    const rhs = combo.map((i) => hyperRhs[i]);
    const det = determinant(matrix);
    if (Math.abs(det) < 1e-12) continue;
    const x = [];
    for (let col = 0; col < n; col++) {
      const replaced = matrix.map((row, r) => row.map((v, c) => (c === col ? rhs[r] : v)));
      x.push(determinant(replaced) / det);
    }
    if (x.some((value) => value < -tolerance)) continue;
    let feasible = true;
    for (let i = 0; i < m && feasible; i++) {
      let lhs = 0;
      for (let j = 0; j < n; j++) lhs += constraints[i][j] * x[j];
      if (lhs > bounds[i] + tolerance) feasible = false;
    }
    if (!feasible) continue;
    let value = 0;
    for (let j = 0; j < n; j++) value += objective[j] * x[j];
    if (best === null || value > best) best = value;
  }
  return best;
}

function lcgBoundedLp(lcg) {
  const n = 1 + lcg.next(2);
  const m = 2 + lcg.next(2);
  const constraints = [];
  for (let i = 0; i < m; i++) {
    const row = [];
    for (let j = 0; j < n; j++) row.push(lcg.next(9));
    constraints.push(row);
  }
  for (let j = 0; j < n; j++) {                         // every column positive
    if (constraints.every((row) => row[j] === 0)) {
      constraints[lcg.next(m)][j] = 1 + lcg.next(3);
    }
  }
  const bounds = [];
  for (let i = 0; i < m; i++) bounds.push(1 + lcg.next(20));
  const objective = [];
  for (let j = 0; j < n; j++) objective.push(lcg.next(10));
  return { objective, constraints, bounds };
}

module.exports = { simplex, bruteForceLp };

if (require.main === module) {
  const assert = require("node:assert");

  // the brief's LP, re-derived: optimum (4, 0) with value 12, NOT (3, 1)/11
  let result = simplex([3, 2], [[1, 1], [1, 3]], [4, 6]);
  assert.strictEqual(result.status, "optimal", "brief LP optimal");
  assert.ok(Math.abs(result.value - 12) < EPS, `brief LP value 12, got ${result.value}`);
  assert.ok(Math.abs(result.x[0] - 4) < EPS && Math.abs(result.x[1]) < EPS, "brief LP vertex (4, 0)");
  assert.strictEqual(bruteForceLp([3, 2], [[1, 1], [1, 3]], [4, 6]), 12, "brief LP oracle 12");

  // hand-solved: vertices (0,0)=0, (2.5,0)=5, (0,4)=12, (1,3)=11 -> 12 at (0,4)
  result = simplex([2, 3], [[1, 1], [2, 1]], [4, 5]);
  assert.ok(result.status === "optimal" && Math.abs(result.value - 12) < EPS, "second LP 12");
  assert.ok(Math.abs(result.x[0]) < EPS && Math.abs(result.x[1] - 4) < EPS, "second LP vertex");
  assert.strictEqual(bruteForceLp([2, 3], [[1, 1], [2, 1]], [4, 5]), 12, "second LP oracle");

  // 3 variables: optima (4,1,0), (2,2,1), (0,3,2) all give 14; Bland picks one
  result = simplex([3, 2, 4], [[1, 1, 1], [2, 1, 3]], [5, 9]);
  assert.ok(result.status === "optimal" && Math.abs(result.value - 14) < EPS, "3-var LP 14");
  assert.ok(Math.abs(result.x[0] + result.x[1] + result.x[2] - 5) < EPS, "3-var sum is 5");
  assert.ok(result.x.every((c) => c > -EPS), "3-var non-negative");
  assert.ok(2 * result.x[0] + result.x[1] + 3 * result.x[2] <= 9 + 1e-7, "3-var second constraint");
  assert.strictEqual(bruteForceLp([3, 2, 4], [[1, 1, 1], [2, 1, 3]], [5, 9]), 14, "3-var LP oracle");

  // tie along x + y <= 1 (degenerate objective direction)
  result = simplex([1, 1], [[1, 1]], [1]);
  assert.ok(result.status === "optimal" && Math.abs(result.value - 1) < EPS, "tie LP");
  assert.ok(Math.abs(result.x[0] + result.x[1] - 1) < EPS, "tie LP sum");

  // duplicate constraints (degenerate basis)
  result = simplex([1], [[1], [1]], [3, 3]);
  assert.ok(result.status === "optimal" && Math.abs(result.value - 3) < EPS, "duplicate LP");

  // --- sentinels ---------------------------------------------------------
  result = simplex([1], [], []);
  assert.ok(result.status === "unbounded" && result.x === null, "no constraints -> unbounded");
  result = simplex([1], [[-1]], [-1]);                 // x >= 1
  assert.ok(result.status === "unbounded" && result.x === null, "x >= 1 -> unbounded");
  result = simplex([1], [[1], [-1]], [1, -3]);         // x <= 1 and x >= 3
  assert.ok(result.status === "infeasible" && result.x === null, "x <= 1 and x >= 3 infeasible");

  // feasible LP with a negative bound (phase 1 needed, artificial leaves the basis)
  result = simplex([1], [[1], [-1]], [5, -2]);         // 2 <= x <= 5
  assert.ok(result.status === "optimal" && Math.abs(result.x[0] - 5) < EPS, "negative bound max x");
  result = simplex([-1], [[1], [-1]], [5, -2]);
  assert.ok(result.status === "optimal" && Math.abs(result.value + 2) < EPS, "negative bound min x");

  // --- LCG cross-checks against vertex enumeration -----------------------
  const lcg = new Lcg(37713);
  for (let trial = 0; trial < 40; trial++) {
    const { objective, constraints, bounds } = lcgBoundedLp(lcg);
    result = simplex(objective, constraints, bounds);
    assert.strictEqual(result.status, "optimal", `random LP optimal ${trial}`);
    const oracle = bruteForceLp(objective, constraints, bounds);
    assert.notStrictEqual(oracle, null, `random LP has an oracle vertex ${trial}`);
    assert.ok(Math.abs(result.value - oracle) < 1e-6, `random LP value ${trial}`);
    constraints.forEach((row, i) => {
      let lhs = 0;
      for (let j = 0; j < objective.length; j++) lhs += row[j] * result.x[j];
      assert.ok(lhs <= bounds[i] + 1e-7, `random LP feasibility ${trial}`);
    });
    assert.ok(result.x.every((c) => c > -1e-7), `random LP non-negative ${trial}`);
  }
  console.log("PASS");
}
