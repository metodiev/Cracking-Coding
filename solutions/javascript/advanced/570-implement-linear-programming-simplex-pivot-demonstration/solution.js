/**
 * Task 570 - Implement linear programming simplex pivot demonstration. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * A full tableau simplex for max c^T x subject to A x <= b, x >= 0 with b >= 0 (slack variables
 * give the obvious initial basis).  Two pivot rules are implemented side by side: Dantzig's
 * (most negative reduced cost) and Bland's (smallest-index entering column, smallest-index basic
 * variable among the minimum-ratio ties), which is guaranteed to terminate.  Every pivot mutates
 * the tableau with Gauss-Jordan elimination; the reduced cost of the slack columns is the shadow
 * price vector c_B B^-1, so the solver also emits a genuine optimality certificate.
 *
 * Documented checks: on the shipped 2-variable and 3-variable instances the simplex optimum equals
 * an independent exact vertex enumeration (all pairs/triples of constraint and non-negativity
 * hyperplanes solved by Gaussian elimination and tested for feasibility); the returned x is
 * feasible, has at most n non-zero entries (a basic solution), and the objective row has no
 * negative reduced costs at termination; the dual vector y = -reduced cost of the slacks satisfies
 * b . y = primal optimum (strong duality), A^T y >= c (dual feasibility) and the complementary
 * slackness conditions y_i * slack_i = 0.  Unboundedness is detected on an instance where a
 * negative reduced-cost column has no positive entry, degenerate instances are solved to the same
 * optimum as the enumeration, and pivot counts are deterministic across reruns.  Beale's classical
 * cycling example (max 10 x1 - 57 x2 - 9 x3 - 24 x4 under its three constraints) is shipped to
 * separate the rules: Dantzig's rule revisits a basis after exactly 6 pivots (the cycle is detected
 * by hashing the basis set, cap 200 pivots) while Bland's rule terminates in 7 pivots at the
 * optimal vertex x = (1, 0, 1, 0) with objective 1 (the constraints force x3 = 1 once x1 = 1) -
 * the textbook point being that Dantzig's rule is not finite in general even though it is what a
 * naive implementation uses, and that the cycling run never gets above objective 0.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;
const EPS = 1e-9;
const PIVOT_CAP = 200;

/** Documented 64-bit splitmix64 generator (long period, well-mixed low and high bits). */
class Rng {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    const z = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    this.state = z;
    let x = z;
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return x ^ (x >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

/** Solve the n x n system M x = rhs by Gaussian elimination with partial pivoting. */
function solveLinearSystem(M, rhs) {
  const n = rhs.length;
  const a = M.map((row, i) => row.concat([rhs[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) <= 1e-12) return null; // singular
    const tmp = a[col];
    a[col] = a[pivot];
    a[pivot] = tmp;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r][col] / a[col][col];
      if (factor === 0) continue;
      for (let j = col; j <= n; j++) a[r][j] -= factor * a[col][j];
    }
  }
  return a.map((row, i) => row[n] / row[i]);
}

/**
 * Exact ground truth for small LPs: enumerate every intersection of n hyperplanes taken from the m
 * constraints plus the n non-negativity planes, keep the feasible ones, take the best objective.
 */
function enumerateVertices(A, b, c) {
  const m = A.length;
  const n = c.length;
  const planes = [];
  for (let i = 0; i < m; i++) planes.push({ coef: A[i].slice(), rhs: b[i] });
  for (let j = 0; j < n; j++) {
    const coef = new Array(n).fill(0);
    coef[j] = 1;
    planes.push({ coef, rhs: 0 });
  }
  const vertices = [];
  const combination = new Array(n).fill(0);
  const rec = (start, depth) => {
    if (depth === n) {
      const matrix = combination.map((idx) => planes[idx].coef);
      const rhs = combination.map((idx) => planes[idx].rhs);
      const x = solveLinearSystem(matrix, rhs);
      if (!x) return;
      if (x.some((v) => !Number.isFinite(v))) return;
      for (let i = 0; i < m; i++) {
        let lhs = 0;
        for (let j = 0; j < n; j++) lhs += A[i][j] * x[j];
        if (lhs > b[i] + 1e-7) return;
      }
      if (x.some((v) => v < -1e-7)) return;
      vertices.push({ x: x.map((v) => (Math.abs(v) < 1e-12 ? 0 : v)), planes: combination.slice() });
      return;
    }
    for (let i = start; i < planes.length; i++) {
      combination[depth] = i;
      rec(i + 1, depth + 1);
    }
  };
  rec(0, 0);
  let best = -Infinity;
  let bestVertex = null;
  for (const vertex of vertices) {
    let value = 0;
    for (let j = 0; j < n; j++) value += c[j] * vertex.x[j];
    if (value > best + 1e-12) {
      best = value;
      bestVertex = vertex;
    }
  }
  const evaluated = binomial(planes.length, n);
  return { x: bestVertex ? bestVertex.x : null, objective: best, vertices, evaluated };
}

function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 1; i <= k; i++) result = (result * (n - k + i)) / i;
  return Math.round(result);
}

function dot(u, v) {
  let total = 0;
  for (let i = 0; i < u.length; i++) total += u[i] * v[i];
  return total;
}

/**
 * Tableau simplex.  rule = "dantzig" | "bland".  Returns the optimum, the primal/dual solutions,
 * the pivot count and a status ("optimal", "unbounded", "cycle").
 */
function simplexSolve(A, b, c, rule = "dantzig") {
  const m = A.length;
  const n = c.length;
  if (b.some((value) => value < -EPS)) throw new Error("simplex: b >= 0 required (no phase I)");
  if (rule !== "dantzig" && rule !== "bland") throw new Error("unknown pivot rule");
  const cols = n + m + 1;
  const T = [];
  for (let i = 0; i < m; i++) {
    const row = new Array(cols).fill(0);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n + i] = 1;
    row[n + m] = b[i];
    T.push(row);
  }
  // Objective row: stored negated, so obj[j] = y A_j - c_j (>= 0 at an optimum) and the slack
  // entries obj[n + i] are exactly the shadow prices y = c_B B^-1.
  const obj = new Array(cols).fill(0);
  for (let j = 0; j < n; j++) obj[j] = -c[j];
  const basis = [];
  for (let i = 0; i < m; i++) basis.push(n + i);
  const seen = new Set([basis.slice().sort((x, y) => x - y).join(",")]);
  const status = { rule, pivots: 0, status: "optimal", cycleLength: 0 };
  let ray = null;

  let terminated = false;
  for (let pivot = 0; pivot < PIVOT_CAP; pivot++) {
    terminated = false;
    let entering = -1;
    if (rule === "bland") {
      for (let j = 0; j < cols - 1; j++) {
        if (obj[j] < -EPS) {
          entering = j;
          break;
        }
      }
    } else {
      let best = -EPS;
      for (let j = 0; j < cols - 1; j++) {
        if (obj[j] < best - EPS) {
          best = obj[j];
          entering = j;
        }
      }
    }
    if (entering < 0) {
      status.status = "optimal";
      terminated = true;
      break;
    }
    // Ratio test; ties resolved by the leaving variable's index under Bland's rule.
    let leaving = -1;
    let bestRatio = Infinity;
    for (let i = 0; i < m; i++) {
      if (T[i][entering] <= EPS) continue;
      const ratio = T[i][n + m] / T[i][entering];
      if (ratio < bestRatio - EPS) {
        bestRatio = ratio;
        leaving = i;
      } else if (rule === "bland" && Math.abs(ratio - bestRatio) <= EPS && leaving >= 0
          && basis[i] < basis[leaving]) {
        leaving = i;
      }
    }
    if (leaving < 0) {
      status.status = "unbounded";
      // Feasible direction: raise the entering variable and follow the basic variables.
      ray = new Array(n + m).fill(0);
      ray[entering] = 1;
      for (let i = 0; i < m; i++) ray[basis[i]] = -T[i][entering];
      terminated = true;
      break;
    }
    const pivotValue = T[leaving][entering];
    for (let j = 0; j < cols; j++) T[leaving][j] /= pivotValue;
    for (let i = 0; i < m; i++) {
      if (i === leaving) continue;
      const factor = T[i][entering];
      if (factor === 0) continue;
      for (let j = 0; j < cols; j++) T[i][j] -= factor * T[leaving][j];
    }
    const objFactor = obj[entering];
    for (let j = 0; j < cols; j++) obj[j] -= objFactor * T[leaving][j];
    basis[leaving] = entering;
    status.pivots = pivot + 1;
    const signature = basis.slice().sort((x, y) => x - y).join(",");
    if (seen.has(signature)) {
      status.status = "cycle";
      status.cycleLength = status.pivots;
      terminated = true;
      break;
    }
    seen.add(signature);
  }
  if (!terminated) status.status = "pivot-cap";
  const x = new Array(n).fill(0);
  for (let i = 0; i < m; i++) if (basis[i] < n) x[basis[i]] = T[i][n + m];
  const slacks = new Array(m).fill(0);
  for (let i = 0; i < m; i++) if (basis[i] >= n && basis[i] < n + m) slacks[basis[i] - n] = T[i][n + m];
  for (let i = 0; i < m; i++) {
    if (basis.indexOf(n + i) >= 0) continue;
    slacks[i] = b[i] - dot(A[i], x);
  }
  // The objective-row entry of slack i is y_i = c_B B^-1 e_i, i.e. the shadow price of row i.
  const y = [];
  for (let i = 0; i < m; i++) y.push(obj[n + i]);
  return { x, slacks, y, objective: dot(c, x), basis: basis.slice(), pivots: status.pivots,
    status: status.status, cycleLength: status.cycleLength, rule, ray,
    reducedCosts: obj.slice(0, n + m), tableau: T };
}

/** Independent feasibility/optimality certificate checks for a claimed primal/dual pair. */
function certificate(A, b, c, result) {
  const primalFeasible = result.x.every((v) => v >= -1e-7)
    && A.every((row, i) => dot(row, result.x) <= b[i] + 1e-7);
  const dualFeasible = result.y.every((v) => v >= -1e-7)
    && c.every((cj, j) => result.y.reduce((acc, yi, i) => acc + yi * A[i][j], 0) >= cj - 1e-7);
  const strongDuality = Math.abs(dot(b, result.y) - result.objective) <= 1e-7;
  let complementary = true;
  for (let i = 0; i < A.length; i++) {
    if (Math.abs(result.y[i] * result.slacks[i]) > 1e-6) complementary = false;
  }
  return { primalFeasible, dualFeasible, strongDuality, complementary };
}

/** Beale's classical cycling example (Dantzig's rule cycles; Bland's rule terminates). */
const BEALE_A = [[0.5, -5.5, -2.5, 9.0], [0.5, -1.5, -0.5, 1.0], [1.0, 0.0, 0.0, 0.0]];
const BEALE_B = [0.0, 0.0, 1.0];
const BEALE_C = [10.0, -57.0, -9.0, -24.0];

/**
 * Random *bounded* LP: the constraint matrix is non-negative and every column has a positive entry,
 * so A x <= b with b > 0 bounds the feasible region.  Unboundedness is exercised separately.
 */
function randomLp(rng, constraints, variables, bound) {
  const A = [];
  const b = [];
  for (let i = 0; i < constraints; i++) {
    const row = [];
    for (let j = 0; j < variables; j++) row.push(rng.nextInt(bound + 1));
    A.push(row);
    b.push(rng.nextInt(3 * bound + 1) + 1);
  }
  for (let j = 0; j < variables; j++) A[j % constraints][j] = Math.max(1, A[j % constraints][j]);
  const c = [];
  for (let j = 0; j < variables; j++) c.push(rng.nextInt(2 * bound + 1) - bound);
  return { A, b, c };
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-solved examples.
  {
    // Non-degenerate instance: max 3x + 2y s.t. x + y <= 4, x + 3y <= 6 -> (4,0) with z = 12.
    const A2 = [[1, 1], [1, 3]];
    const b2 = [4, 6];
    const c2 = [3, 2];
    const solved2 = simplexSolve(A2, b2, c2, "bland");
    assert.strictEqual(solved2.status, "optimal");
    assert.ok(Math.abs(solved2.x[0] - 4.0) <= 1e-9);
    assert.ok(Math.abs(solved2.x[1]) <= 1e-9);
    assert.ok(Math.abs(solved2.objective - 12.0) <= 1e-9);
    assert.deepStrictEqual(enumerateVertices(A2, b2, c2).x, [4.0, 0.0]);
    assert.ok(Math.abs(enumerateVertices(A2, b2, c2).objective - 12.0) <= 1e-9);
    // Unique optimal basis here, so the shadow prices are unique: y = (3, 0).
    assert.ok(Math.abs(solved2.y[0] - 3.0) <= 1e-9);
    assert.ok(Math.abs(solved2.y[1]) <= 1e-9);
    assert.ok(Math.abs(dot(b2, solved2.y) - 12.0) <= 1e-9, "strong duality b.y = 12");

    // Degenerate instance: max 3x + 2y s.t. x + y <= 4, x + 3y <= 6, x <= 3 -> (3,1) with z = 11,
    // where all three constraints are tight, so only the certificate (not y itself) is pinned.
    const A = [[1, 1], [1, 3], [1, 0]];
    const b = [4, 6, 3];
    const c = [3, 2];
    const solved = simplexSolve(A, b, c, "bland");
    assert.strictEqual(solved.status, "optimal");
    assert.ok(Math.abs(solved.x[0] - 3.0) <= 1e-9);
    assert.ok(Math.abs(solved.x[1] - 1.0) <= 1e-9);
    assert.ok(Math.abs(solved.objective - 11.0) <= 1e-9);
    const truth = enumerateVertices(A, b, c);
    assert.ok(Math.abs(truth.objective - 11.0) <= 1e-9, "exact enumeration agrees");
    assert.deepStrictEqual(truth.x, [3.0, 1.0]);
    assert.ok(Math.abs(dot(b, solved.y) - 11.0) <= 1e-9, "degenerate instance: b.y = 11");
    assert.ok(solved.y.every((v) => v >= -1e-9));
    const cert = certificate(A, b, c, solved);
    assert.deepStrictEqual(cert, { primalFeasible: true, dualFeasible: true, strongDuality: true,
      complementary: true });
  }

  // (2) Random 2-variable and 3-variable LPs: simplex = exact vertex enumeration.
  {
    const rng = new Rng(5150);
    let checked2 = 0;
    let checked3 = 0;
    for (let trial = 0; trial < 120; trial++) {
      const { A, b, c } = randomLp(rng, 3 + rng.nextInt(4), 2, 5);
      const truth = enumerateVertices(A, b, c);
      if (!truth.x) continue;
      for (const rule of ["bland", "dantzig"]) {
        const solved = simplexSolve(A, b, c, rule);
        assert.strictEqual(solved.status, "optimal", `rule ${rule} must not cycle on ${trial}`);
        assert.ok(Math.abs(solved.objective - truth.objective) <= 1e-7,
          `trial ${trial} ${rule}: ${solved.objective} vs enumeration ${truth.objective}`);
        const cert = certificate(A, b, c, solved);
        assert.ok(cert.primalFeasible && cert.dualFeasible && cert.strongDuality
          && cert.complementary, `trial ${trial} ${rule} certificate`);
        assert.ok(solved.reducedCosts.every((value) => value >= -1e-7),
          "no negative reduced cost at the optimum");
        const nonZero = solved.x.filter((v) => v > 1e-9).length;
        assert.ok(nonZero <= 2, "a basic solution has at most n = 2 non-zero variables");
        assert.ok(Math.abs(dot(c, solved.x) - solved.objective) <= 1e-9);
        assert.ok(solved.tableau.every((row) => row.every((v) => Number.isFinite(v))));
      }
      checked2 += 1;
    }
    assert.ok(checked2 >= 100, `2-variable instances verified: ${checked2}`);

    for (let trial = 0; trial < 40; trial++) {
      const { A, b, c } = randomLp(rng, 4 + rng.nextInt(3), 3, 4);
      const truth = enumerateVertices(A, b, c);
      assert.ok(truth.x, "the origin is feasible so vertices exist");
      const solved = simplexSolve(A, b, c, "bland");
      assert.strictEqual(solved.status, "optimal");
      assert.ok(Math.abs(solved.objective - truth.objective) <= 1e-7,
        `3-variable trial ${trial}: ${solved.objective} vs ${truth.objective}`);
      const cert = certificate(A, b, c, solved);
      assert.ok(cert.primalFeasible && cert.dualFeasible && cert.strongDuality
        && cert.complementary);
      checked3 += 1;
    }
    assert.strictEqual(checked3, 40);
  }

  // (3) Beale's cycling example: Dantzig's rule cycles, Bland's rule terminates at the optimum.
  {
    const bland = simplexSolve(BEALE_A, BEALE_B, BEALE_C, "bland");
    assert.strictEqual(bland.status, "optimal", "Bland's rule always terminates");
    assert.ok(Math.abs(bland.objective - 1.0) <= 1e-9, "the optimum of Beale's example is 1");
    const truthBeale = enumerateVertices(BEALE_A, BEALE_B, BEALE_C);
    assert.ok(Math.abs(truthBeale.objective - 1.0) <= 1e-9);
    assert.deepStrictEqual(truthBeale.x, [1.0, 0.0, 1.0, 0.0],
      "the constraints force x3 = 1 once x1 = 1, so (1,0,1,0) is the optimal vertex");
    assert.deepStrictEqual(bland.x, truthBeale.x, "Bland's rule reaches that vertex");
    assert.strictEqual(bland.pivots, 7, "Bland needs 7 pivots (documented)");
    const cert = certificate(BEALE_A, BEALE_B, BEALE_C, bland);
    assert.ok(cert.primalFeasible && cert.dualFeasible && cert.strongDuality && cert.complementary);

    const dantzig = simplexSolve(BEALE_A, BEALE_B, BEALE_C, "dantzig");
    assert.strictEqual(dantzig.status, "cycle", "Dantzig's rule revisits a basis");
    assert.strictEqual(dantzig.cycleLength, 6, "the documented cycle has period 6 pivots");
    assert.strictEqual(dantzig.pivots, 6);
    assert.ok(Math.abs(dantzig.objective - 0.0) <= 1e-9,
      "so the cycling run is stuck at the degenerate objective value 0, far from the optimum 1");
    assert.ok(dantzig.objective < truthBeale.objective);
  }

  // (4) Unboundedness: reported instead of looping, and certified by a feasible improving ray.
  {
    const cases = [
      { A: [[-1, 1]], b: [1], c: [1, 0] },
      { A: [[1, -1]], b: [0], c: [0, 1] },
      { A: [[-2, 1], [0, 1]], b: [3, 4], c: [5, -1] },
    ];
    for (const { A, b, c } of cases) {
      const solved = simplexSolve(A, b, c, "bland");
      assert.strictEqual(solved.status, "unbounded", "the ray exists, so the LP is unbounded");
      const ray = solved.ray;
      assert.strictEqual(ray.length, c.length + A.length);
      const structural = ray.slice(0, c.length);
      assert.ok(structural.every((v) => v >= -1e-9), "the ray points inside the non-negative cone");
      A.forEach((row) => assert.ok(dot(row, structural) <= 1e-9,
        "the ray never leaves the half-space A x <= b"));
      assert.ok(dot(c, structural) > 1e-9, "and it strictly improves the objective");
    }
    assert.throws(() => simplexSolve([[1, 1]], [-1], [1, 1]), /b >= 0 required/);
    assert.throws(() => simplexSolve([[1, 1]], [1], [1, 1], "random"), /unknown pivot rule/);
  }

  // (5) Degenerate instances still agree with the enumeration, and reruns are deterministic.
  {
    const A = [[1, 1], [1, 0], [0, 1], [2, 2]];
    const b = [2, 1, 1, 4];
    const c = [1, 1];
    const truth = enumerateVertices(A, b, c);
    assert.ok(Math.abs(truth.objective - 2.0) <= 1e-9);
    for (const rule of ["bland", "dantzig"]) {
      const solved = simplexSolve(A, b, c, rule);
      assert.strictEqual(solved.status, "optimal");
      assert.ok(Math.abs(solved.objective - 2.0) <= 1e-9);
      assert.ok(Math.abs(solved.x[0] + solved.x[1] - 2.0) <= 1e-9);
      assert.ok(solved.slacks.every((v) => v >= -1e-9));
      assert.ok(Math.abs(solved.slacks[3]) <= 1e-9, "the redundant 2x + 2y <= 4 row stays tight");
    }
    const first = simplexSolve(A, b, c, "bland");
    const second = simplexSolve(A, b, c, "bland");
    assert.deepStrictEqual(first, second, "reruns are bit-identical");
    assert.strictEqual(simplexSolve([[1, 1]], [1], [1, 1], "bland").pivots, 1);
    assert.strictEqual(simplexSolve([[1, 1]], [1], [-1, -1], "bland").objective, 0,
      "the optimum of a minimisation-shaped objective is 0 at the origin");
  }

  // (6) Trivial instances: all-zero objective, a single tight constraint, binomial values.
  {
    const zeroObjective = simplexSolve([[1, 1], [1, -1]], [1, 1], [0, 0], "bland");
    assert.strictEqual(zeroObjective.objective, 0, "any feasible point is optimal");
    assert.strictEqual(zeroObjective.pivots, 0, "no pivot is needed");
    const tight = simplexSolve([[1, 1]], [2], [1, 1], "dantzig");
    assert.ok(Math.abs(tight.objective - 2.0) <= 1e-9);
    assert.ok(Math.abs(tight.y[0] - 1.0) <= 1e-9, "y = 1 for a tight constraint with c = (1,1)");
    assert.strictEqual(binomial(5, 2), 10);
    assert.strictEqual(binomial(7, 3), 35);
  }
}

module.exports = { MASK64, EPS, PIVOT_CAP, Rng, solveLinearSystem, enumerateVertices, binomial, dot,
  simplexSolve, certificate, BEALE_A, BEALE_B, BEALE_C, randomLp };

if (require.main === module) {
  main();
  console.log("PASS");
}
