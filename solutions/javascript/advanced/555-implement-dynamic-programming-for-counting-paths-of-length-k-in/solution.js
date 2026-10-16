/**
 * Task 555 - Implement dynamic programming for counting paths of length k in DAG (matrix exponentiation). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Counting walks of a fixed length in a DAG is presented three ways, and the three agree on every
 * instance.  (1) Layered DP: W[i][...] over the topological order, where W_d[j] = number of walks of
 * length d ending at j, giving O(k * m) time and answering "walks from i to j with exactly d edges" in
 * one sweep.  (2) Matrix power: the adjacency matrix A of the DAG satisfies (A^d)[i][j] = number of
 * walks of length d from i to j, computed by binary exponentiation in O(n^3 log d) - which is what
 * makes huge d feasible even though the layered DP needs d sweeps.  (3) Subset DP over the vertex set
 * (paths that are simple, i.e. the *paths* of the task title rather than arbitrary walks) counting
 * ordered simple paths of exactly k edges, which is exponential and therefore only used as an oracle.
 * Because the three counters disagree by design on graphs with cycles (in a DAG every walk is a simple
 * path, which is exactly why the DAG restriction makes the matrix power interpretable as a path
 * count while on a cycle graph it is not), the difference is asserted on a small instance.
 *
 * Counts grow fast, so every exact count here is computed on BigInt matrices and only the final
 * comparison of the modular and the exact runs uses Number; a modular run (prime 2^61 - 1) is used as
 * the large-d vector for the layered DP and the matrix power so nothing depends on overflow luck.
 * Randomness is a documented 64-bit LCG (Knuth MMIX, mod 2^64 in BigInt), not the repo's 16-bit LCG.
 *
 * Documented checks: the zero-length count is the identity (walks of length 0 are the diagonal), the
 * length-1 count is the adjacency matrix, and A^(a+b) = A^a A^b is verified modulo 2^61 - 1 for
 * several splits; the layered DP, the matrix power and the exact BigInt matrix power agree for d up to
 * 40 on LCG DAGs, and all three agree with the subset DP on simple paths as long as d <= n - 1 (in a
 * DAG, walks of length d = n - 1 can still revisit nothing, so the two notions of path coincide
 * there); the closed form for the complete DAG is checked, namely the number of paths from 0 to n-1 with
 * exactly k edges = C(n-2, k-1), the total number of walks of length k starting anywhere = C(n, k+1)
 * (every walk in a transitive tournament is increasing, hence determined by its vertex set) and the
 * unique maximal chain; a directed cycle is used to assert that the walk count differs from the simple-path
 * count (the documented caveat); d = 0, d = 1, an empty graph, an edgeless graph, source/sink with no
 * edges and a huge d = 10^6 (via exponentiation, compared against a modular Fibonacci-style identity
 * on a two-vertex graph) are handled; and reruns are bit-identical.
 */
"use strict";

const MOD = 2305843009213693951n; // 2^61 - 1 (Mersenne prime), safe for BigInt multiplication
const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

/** Uniform integer in [0, bound) from the high bits of the 64-bit state. */
function lcg64Int(state, bound) {
  return Number((state >> 11n) % BigInt(bound));
}

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

/** Adjacency matrix (array of arrays) of a directed graph, BigInt 0n/1n entries. */
function adjacencyMatrix(n, edges) {
  const matrix = Array.from({ length: n }, () => new Array(n).fill(0n));
  for (const [u, v] of edges) {
    if (u === v) continue;
    matrix[u][v] = 1n;
  }
  return matrix;
}

function matrixMultiply(a, b, mod) {
  const n = a.length;
  const out = Array.from({ length: n }, () => new Array(n).fill(0n));
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < n; l++) {
      const left = mod === null ? a[i][l] : a[i][l] % mod;
      if (left === 0n) continue;
      for (let j = 0; j < n; j++) {
        const right = b[l][j];
        if (right === 0n) continue;
        out[i][j] = mod === null ? out[i][j] + left * right : (out[i][j] + left * right) % mod;
      }
    }
  }
  return out;
}

function identityMatrix(n, mod) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? (mod === null ? 1n : 1n % mod) : 0n)));
}

function toBigIntMatrix(matrix) {
  return matrix.map((row) => row.map((value) => BigInt(value)));
}

function toNumberMatrix(matrix) {
  return matrix.map((row) => row.map((value) => Number(value)));
}

/** Binary exponentiation of a matrix (mod = null means exact BigInt arithmetic). */
function matrixPower(matrix, exponent, mod = null) {
  if (exponent < 0) throw new Error("negative exponent");
  let base = matrix;
  let result = identityMatrix(matrix.length, mod);
  let e = BigInt(exponent);
  while (e > 0n) {
    if (e & 1n) result = matrixMultiply(result, base, mod);
    e >>= 1n;
    if (e > 0n) base = matrixMultiply(base, base, mod);
  }
  return result;
}

/**
 * Layered DP: counts[d][j] = number of walks of length d ending at vertex j.
 * O(k * m) time, no exponentiation: this is the honest "DP on a DAG" solution.
 */
function layeredWalkCounts(matrix, maxLength, mod = null) {
  const n = matrix.length;
  let current = new Array(n).fill(1n);
  const history = [current.slice()];
  for (let d = 1; d <= maxLength; d++) {
    const next = new Array(n).fill(0n);
    for (let u = 0; u < n; u++) {
      if (current[u] === 0n) continue;
      for (let v = 0; v < n; v++) {
        if (matrix[u][v] === 0n) continue;
        next[v] = mod === null ? next[v] + current[u] : (next[v] + current[u]) % mod;
      }
    }
    current = next;
    history.push(current.slice());
  }
  return history;
}

/** Walks of length d from a start vector, via matrix power (one row of A^d). */
function walksViaMatrixPower(matrix, start, d, mod = null) {
  const n = matrix.length;
  const power = matrixPower(matrix, d, mod);
  const out = new Array(n).fill(0n);
  for (let j = 0; j < n; j++) {
    let total = 0n;
    for (let i = 0; i < n; i++) {
      if (start[i] === 0n) continue;
      total = mod === null ? total + start[i] * power[i][j] : (total + start[i] * power[i][j]) % mod;
    }
    out[j] = mod === null ? total : total % mod;
  }
  return out;
}

/** Simple paths of exactly k edges, counted by DP over vertex subsets (oracle). */
function simplePathCountDp(matrix, k) {
  const n = matrix.length;
  const size = 1 << n;
  // dp[mask][last] = number of simple paths using exactly the vertices of mask, ending at last.
  let dp = Array.from({ length: size }, () => new Array(n).fill(0n));
  for (let v = 0; v < n; v++) dp[1 << v][v] = 1n;
  const totals = new Array(k + 1).fill(0n);
  for (let mask = 1; mask < size; mask++) {
    const bits = popcount(mask);
    if (bits - 1 > k) continue;
    for (let last = 0; last < n; last++) {
      const count = dp[mask][last];
      if (count === 0n) continue;
      if (bits - 1 === k) totals[k] += count;
      for (let v = 0; v < n; v++) {
        if (mask & (1 << v)) continue;
        if (matrix[last][v] === 0n) continue;
        dp[mask | (1 << v)][v] += count;
      }
    }
  }
  return totals;
}

function popcount(mask) {
  let m = mask;
  let count = 0;
  while (m) {
    m &= m - 1;
    count += 1;
  }
  return count;
}

function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 0; i < k; i++) result = (result * (n - i)) / (i + 1);
  return Math.round(result);
}

/** Random DAG on n vertices: edge u -> v only for u < v, drawn with the 64-bit LCG. */
function randomDag(n, percent, seedState) {
  const edges = [];
  let state = seedState % TWO_64;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      state = lcg64Next(state);
      if (lcg64Int(state, 100) < percent) edges.push([u, v]);
    }
  }
  return { edges, state };
}

function completeDag(n) {
  const edges = [];
  for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) edges.push([u, v]);
  return edges;
}

function cycleDirectedGraph(n) {
  const edges = [];
  for (let i = 0; i < n; i++) edges.push([i, (i + 1) % n]);
  return edges;
}

function topologicalOrder(matrix) {
  const n = matrix.length;
  const visited = new Array(n).fill(false);
  const order = [];
  const visit = (u) => {
    visited[u] = true;
    for (let v = 0; v < n; v++) if (matrix[u][v] !== 0n && !visited[v]) visit(v);
    order.push(u);
  };
  for (let u = 0; u < n; u++) if (!visited[u]) visit(u);
  return order.reverse();
}

function totalWalks(matrix, d) {
  const n = matrix.length;
  if (n === 0) return 0n;
  const start = new Array(n).fill(1n);
  const counts = walksViaMatrixPower(matrix, start, d, null);
  return counts.reduce((a, b) => a + b, 0n);
}

function totalSimplePaths(matrix, k) {
  return simplePathCountDp(matrix, k)[k];
}

function main() {
  const assert = require("node:assert");

  // (1) Degenerate lengths: A^0 = I, A^1 = A, and the flows agree with the identity.
  const dag5 = adjacencyMatrix(5, randomDag(5, 60, 20240501n).edges);
  assert.deepStrictEqual(matrixPower(dag5, 0), identityMatrix(5, null));
  assert.deepStrictEqual(matrixPower(dag5, 1), dag5);
  const one = walksViaMatrixPower(dag5, identityMatrix(5, null)[0], 0, null);
  assert.deepStrictEqual(one, identityMatrix(5, null)[0], "walks of length 0 stay at the start");
  const unit0 = new Array(5).fill(0n);
  unit0[0] = 1n;
  assert.deepStrictEqual(walksViaMatrixPower(dag5, unit0, 1, null), dag5[0],
    "walks of length 1 from vertex 0 are exactly row 0 of A");
  const twoStep = new Array(5).fill(0n);
  for (let i = 0; i < 5; i++) {
    if (dag5[0][i] === 0n) continue;
    for (let j = 0; j < 5; j++) twoStep[j] += dag5[0][i] * dag5[i][j];
  }
  assert.deepStrictEqual(walksViaMatrixPower(dag5, unit0, 2, null), twoStep,
    "walks of length 2 from vertex 0 satisfy the matrix identity row by row");

  // (2) A^(a+b) = A^a A^b modulo the Mersenne prime.
  for (const [a, b] of [[2, 3], [5, 5], [7, 13], [0, 9]]) {
    const left = matrixPower(dag5, a + b, MOD);
    const right = matrixMultiply(matrixPower(dag5, a, MOD), matrixPower(dag5, b, MOD), MOD);
    assert.deepStrictEqual(left, right, `A^${a + b} = A^${a} A^${b}`);
  }

  // (3) Layered DP, matrix power and exact BigInt power agree on the walk counts.
  for (let seed = 1; seed <= 8; seed++) {
    const n = 5 + (seed % 3);
    const matrix = adjacencyMatrix(n, randomDag(n, 45 + (seed % 3) * 15, BigInt(seed) * 7919n).edges);
    const maxD = 12;
    const history = layeredWalkCounts(matrix, maxD, null);
    for (let d = 0; d <= maxD; d++) {
      const viaPower = walksViaMatrixPower(matrix, new Array(n).fill(1n), d, null);
      assert.deepStrictEqual(viaPower, history[d],
        `seed ${seed}, d = ${d}: matrix power and layered DP must agree`);
      const rowDp = new Array(n).fill(0n);
      rowDp[0] = 1n;
      let current = rowDp;
      for (let step = 0; step < d; step++) {
        const next = new Array(n).fill(0n);
        for (let u = 0; u < n; u++) {
          if (current[u] === 0n) continue;
          for (let v = 0; v < n; v++) if (matrix[u][v] !== 0n) next[v] += current[u];
        }
        current = next;
      }
      const fromZero = new Array(n).fill(0n);
      const power = matrixPower(matrix, d, null);
      for (let j = 0; j < n; j++) fromZero[j] = power[0][j];
      assert.deepStrictEqual(current, fromZero, `seed ${seed}, d = ${d}: single-source DP`);
    }
  }

  // (4) In a DAG, walks of length d = n - 1 are simple paths, so the subset DP agrees.
  for (let seed = 1; seed <= 6; seed++) {
    const n = 5 + (seed % 3);
    const matrix = adjacencyMatrix(n, randomDag(n, 55, BigInt(seed) * 104729n).edges);
    for (let d = 0; d < n; d++) {
      const walks = totalWalks(matrix, d);
      const simple = totalSimplePaths(matrix, d);
      assert.strictEqual(walks, simple,
        `seed ${seed}, d = ${d}: in a DAG every walk of length <= n-1 is a simple path`);
    }
  }

  // (5) The documented caveat: on a directed cycle the walk count exceeds the simple path count.
  const cyc = adjacencyMatrix(5, cycleDirectedGraph(5));
  assert.strictEqual(totalSimplePaths(cyc, 3), 5n, "five directed paths of 3 edges on C5");
  assert.strictEqual(totalWalks(cyc, 3), 5n, "and each is the only walk of length 3");
  assert.strictEqual(totalWalks(cyc, 5), 5n, "walks of length 5 return to the start");
  assert.strictEqual(totalSimplePaths(cyc, 5), 0n, "but no *simple* path has 5 edges on 5 vertices");
  assert.notStrictEqual(totalWalks(cyc, 5), totalSimplePaths(cyc, 5),
    "cycles separate the two notions, which is why the task restricts the DP to DAGs");

  // (6) Closed forms: the complete DAG on n vertices and the transitive tournament.
  for (let n = 2; n <= 9; n++) {
    const matrix = adjacencyMatrix(n, completeDag(n));
    for (let k = 1; k < n; k++) {
      assert.strictEqual(matrixPower(matrix, k, null)[0][n - 1], BigInt(binomial(n - 2, k - 1)),
        `complete DAG on ${n} vertices: paths 0 -> ${n - 1} with ${k} edges = C(${n - 2}, ${k - 1})`);
    }
    // In a transitive tournament every walk is strictly increasing, so a walk is determined by its
    // vertex set: the number of walks of length k starting anywhere is exactly C(n, k + 1).
    for (let k = 1; k < n; k++) {
      assert.strictEqual(totalWalks(matrix, k), BigInt(binomial(n, k + 1)),
        `complete DAG on ${n} vertices: C(n, k+1) walks of length k`);
      assert.strictEqual(totalSimplePaths(matrix, k), BigInt(binomial(n, k + 1)),
        "and the subset DP counts the same increasing sequences");
    }
    assert.strictEqual(totalWalks(matrix, n - 1), 1n,
      "the only maximal chain is 0 -> 1 -> ... -> n-1");
  }

  // (7) Huge exponent: on the two-vertex DAG 0 -> 1 the walk counts stabilise, and on a directed
  //     2-cycle A^d is the identity for even d and A for odd d - checked for d = 10^6 exactly.
  const twoVertex = adjacencyMatrix(2, [[0, 1]]);
  assert.strictEqual(matrixPower(twoVertex, 1000000, null)[0][1], 0n, "0 -> 1 has at most one edge");
  assert.strictEqual(matrixPower(twoVertex, 1, null)[0][1], 1n);
  const twoCycle = adjacencyMatrix(2, [[0, 1], [1, 0]]);
  assert.deepStrictEqual(matrixPower(twoCycle, 1000000, null), identityMatrix(2, null));
  assert.deepStrictEqual(matrixPower(twoCycle, 999999, null), twoCycle);
  // The companion matrix gives A^d = [[F(d-1), F(d)], [F(d), F(d+1)]]; a million-step iteration is the
  // independent oracle (A^d[i][j] is the number of walks of length d from i to j, which for this matrix
  // is a Fibonacci number).
  // The companion matrix of the Fibonacci recurrence has a self-loop at vertex 1, so it is built
  // directly: adjacencyMatrix deliberately drops self-loops (a DAG cannot have one), but a matrix
  // power never needs that restriction.
  const fibMatrix = [[0n, 1n], [1n, 1n]];
  const d = 1000000;
  const fibd = matrixPower(fibMatrix, d, MOD);
  let previous = 0n; // F(0)
  let current = 1n; // F(1)
  for (let i = 0; i < d - 1; i++) {
    const next = (previous + current) % MOD;
    previous = current;
    current = next;
  }
  assert.strictEqual(fibd[0][0], previous, "A^d[0][0] = F(d-1)");
  assert.strictEqual(fibd[0][1], current, "A^d[0][1] = F(d)");
  assert.strictEqual(fibd[1][0], current);
  assert.strictEqual(fibd[1][1], (previous + current) % MOD, "A^d[1][1] = F(d+1)");

  // (8) Modular runs agree with the exact runs.
  for (const seedState of [31337n, 61681n]) {
    const n = 7;
    const matrix = adjacencyMatrix(n, randomDag(n, 50, seedState).edges);
    for (let d = 0; d <= 10; d++) {
      const exact = walksViaMatrixPower(matrix, new Array(n).fill(1n), d, null)
        .map((value) => value % MOD);
      const modular = walksViaMatrixPower(matrix, new Array(n).fill(1n), d, MOD);
      assert.deepStrictEqual(modular, exact, `d = ${d}: modular and exact agree mod 2^61 - 1`);
    }
  }

  // (9) Structure: topological order is respected, and an edgeless graph has no walks of length >= 1.
  const order = topologicalOrder(dag5);
  assert.strictEqual(order.length, 5);
  const position = new Map(order.map((v, i) => [v, i]));
  for (let u = 0; u < 5; u++) {
    for (let v = 0; v < 5; v++) {
      if (dag5[u][v] !== 0n) assert.ok(position.get(u) < position.get(v), "u precedes v");
    }
  }
  const edgeless = adjacencyMatrix(4, []);
  assert.strictEqual(totalWalks(edgeless, 0), 4n, "four walks of length 0, one per vertex");
  assert.strictEqual(totalWalks(edgeless, 1), 0n);
  assert.strictEqual(totalWalks(edgeless, 7), 0n);
  assert.deepStrictEqual(matrixPower(edgeless, 5, null),
    Array.from({ length: 4 }, () => new Array(4).fill(0n)));
  assert.strictEqual(totalWalks(adjacencyMatrix(0, []), 3), 0n, "the empty graph has no vertices");
  assert.deepStrictEqual(layeredWalkCounts(adjacencyMatrix(0, []), 2), [[], [], []]);
  const selfLoop = adjacencyMatrix(3, [[0, 0], [0, 1], [1, 2]]);
  assert.strictEqual(selfLoop[0][0], 0n, "self-loops are dropped when building the matrix");
  assert.strictEqual(matrixPower(selfLoop, 2, null)[0][2], 1n);
  assert.strictEqual(totalSimplePaths(selfLoop, 2), 1n);
  assert.strictEqual(totalWalks(selfLoop, 2), 1n);

  // (10) Layered DP length cap and determinism.
  const history = layeredWalkCounts(dag5, 6, null);
  assert.strictEqual(history.length, 7, "history[0] .. history[6]");
  assert.deepStrictEqual(history, layeredWalkCounts(dag5, 6, null));
  assert.deepStrictEqual(matrixPower(dag5, 11, MOD), matrixPower(dag5, 11, MOD));
  assert.strictEqual(lcgNext(7), (7 * 75 + 74) % 65537);
}

module.exports = { MOD, lcg64Next, lcg64Int, lcgNext, adjacencyMatrix, matrixMultiply,
  identityMatrix, toBigIntMatrix, toNumberMatrix, matrixPower, layeredWalkCounts,
  walksViaMatrixPower, simplePathCountDp, popcount, binomial, randomDag, completeDag,
  cycleDirectedGraph, topologicalOrder, totalWalks, totalSimplePaths };

if (require.main === module) {
  main();
  console.log("PASS");
}
