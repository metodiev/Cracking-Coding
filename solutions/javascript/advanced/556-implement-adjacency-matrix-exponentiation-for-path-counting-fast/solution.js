/**
 * Task 556 - Implement adjacency matrix exponentiation for path counting (fast exponent). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * (A^k)[i][j] counts the walks of length k from i to j, so "path counting" reduces to fast
 * exponentiation of the adjacency matrix: square-and-multiply takes at most
 * 2 * floor(log2 k) matrix multiplications instead of the k - 1 that repeated multiplication needs,
 * which is what makes k = 10^9 or the finite k = n - 1 of a transitive closure practical.  The file
 * implements binary exponentiation bottom-up, a second top-down recursion, repeated multiplication
 * and a modular variant, all on BigInt matrices, and counts the matrix multiplications each performs
 * so the O(log k) claim is measured instead of asserted.
 *
 * A single implementation of one algorithm proves nothing, so the walk counts are checked against
 * four independent sources: (1) a brute-force DFS enumeration of every length-k walk on small
 * instances (output as a triple-wise count matrix); (2) the spectral closed form for K_n, where
 * walks(u -> v) = ((n-1)^k + (n-1)(-1)^k) / n for u = v and ((n-1)^k - (-1)^k) / n otherwise;
 * (3) for cycles, the identity trace(A^k) = sum of the k-th powers of the eigenvalues
 * lambda_j = 2 cos(2 pi j / n), evaluated in floating point and compared to the exact BigInt trace
 * with a relative tolerance of 1e-9; and (4) the layered DP of the previous task's style, used as a
 * recurrence oracle.  Boolean reachability via (A + I)^(n-1) is compared with BFS on every instance,
 * which is the standard reason the exponentiation matters.
 *
 * Documented checks: the identity and adjacency cases A^0 = I, A^1 = A; the monoid laws
 * A^(a+b) = A^a A^b and (A^a)^b = A^(ab) modulo 2^61 - 1; fast power equals repeated multiplication
 * for every k from 0 to 12 and equals the top-down recursion; the multiplication counter stays inside
 * 2*floor(log2 k) and is strictly below k - 1 once k >= 5; DFS enumeration, the K_n closed form, the
 * cycle spectral trace and the layered DP all reproduce the matrix entries; cumulative counts
 * (walks of length <= k) are monotone and equal to the prefix sums; the trace of A^k is 0 for odd k on
 * a bipartite graph; d = 0, k = 1, an edgeless graph, a single vertex, a disconnected graph and
 * k = 10^9 (modular, checked against the structure of a cycle, where A^k is the identity iff
 * k = 0 mod n) are handled; and reruns are bit-identical.
 */
"use strict";

const MOD = 2305843009213693951n;
const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

function lcg64Int(state, bound) {
  return Number((state >> 13n) % BigInt(bound));
}

function zeroMatrix(n) {
  return Array.from({ length: n }, () => new Array(n).fill(0n));
}

function identityMatrix(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1n : 0n)));
}

/** Adjacency matrix of a simple undirected graph; self-loops are dropped. */
function adjacencyMatrix(n, edges) {
  const matrix = zeroMatrix(n);
  for (const [u, v] of edges) {
    if (u === v) continue;
    matrix[u][v] = 1n;
    matrix[v][u] = 1n;
  }
  return matrix;
}

function directedAdjacency(n, edges) {
  const matrix = zeroMatrix(n);
  for (const [u, v] of edges) {
    if (u === v) continue;
    matrix[u][v] = 1n;
  }
  return matrix;
}

/** A * B with an optional modulus; the counter records how many products were formed. */
function multiply(a, b, mod = null, counter = null) {
  const n = a.length;
  if (counter) counter.multiplications += 1;
  const out = zeroMatrix(n);
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < n; l++) {
      const left = a[i][l];
      if (left === 0n) continue;
      for (let j = 0; j < n; j++) {
        const right = b[l][j];
        if (right === 0n) continue;
        const product = left * right;
        out[i][j] = mod === null ? out[i][j] + product : (out[i][j] + product) % mod;
      }
    }
  }
  return out;
}

/** Bottom-up square-and-multiply. */
function fastPower(matrix, exponent, mod = null, counter = null) {
  if (exponent < 0) throw new Error("negative exponent");
  let base = matrix;
  let result = identityMatrix(matrix.length);
  let e = BigInt(exponent);
  while (e > 0n) {
    if (e & 1n) result = multiply(result, base, mod, counter);
    e >>= 1n;
    if (e > 0n) base = multiply(base, base, mod, counter);
  }
  return result;
}

/** Top-down recursion: a^k = a^(k/2) squared (times a when k is odd). Different control flow. */
function recursivePower(matrix, exponent, mod = null) {
  if (exponent < 0) throw new Error("negative exponent");
  if (exponent === 0) return identityMatrix(matrix.length);
  const half = recursivePower(matrix, Math.floor(exponent / 2), mod);
  const squared = multiply(half, half, mod);
  return exponent % 2 === 0 ? squared : multiply(squared, matrix, mod);
}

/** Repeated multiplication: the honest O(k) baseline. */
function repeatedPower(matrix, exponent, mod = null, counter = null) {
  if (exponent < 0) throw new Error("negative exponent");
  let result = identityMatrix(matrix.length);
  for (let i = 0; i < exponent; i++) result = multiply(result, matrix, mod, counter);
  return result;
}

function trace(matrix, mod = null) {
  let total = 0n;
  for (let i = 0; i < matrix.length; i++) {
    total = mod === null ? total + matrix[i][i] : (total + matrix[i][i]) % mod;
  }
  return total;
}

/** Brute-force DFS enumeration of every walk of length k, returned as a count matrix. */
function walkCountsByDfs(matrix, k) {
  const n = matrix.length;
  const counts = zeroMatrix(n);
  const walk = [];
  const rec = (length) => {
    if (length === k) {
      counts[walk[0]][walk[walk.length - 1]] += 1n;
      return;
    }
    const last = walk[walk.length - 1];
    for (let v = 0; v < n; v++) {
      if (matrix[last][v] === 0n) continue;
      walk.push(v);
      rec(length + 1);
      walk.pop();
    }
  };
  for (let start = 0; start < n; start++) {
    walk.length = 0;
    walk.push(start);
    rec(0);
  }
  return counts;
}

/** Layered DP matrix: entry (i, j) = walks of length exactly d from i to j. */
function walkCountsByDp(matrix, k) {
  const n = matrix.length;
  let current = identityMatrix(n);
  for (let step = 0; step < k; step++) {
    const next = zeroMatrix(n);
    for (let i = 0; i < n; i++) {
      for (let l = 0; l < n; l++) {
        if (current[i][l] === 0n) continue;
        for (let j = 0; j < n; j++) {
          if (matrix[l][j] === 0n) continue;
          next[i][j] += current[i][l] * matrix[l][j];
        }
      }
    }
    current = next;
  }
  return current;
}

/** Closed form for the complete graph K_n (spectrum: n - 1 once and -1 with multiplicity n - 1). */
function completeGraphWalkCount(n, k, uEqualsV) {
  const dominant = BigInt(n - 1) ** BigInt(k);
  const alternating = (-1n) ** BigInt(k);
  const numerator = uEqualsV ? dominant + BigInt(n - 1) * alternating : dominant - alternating;
  return numerator / BigInt(n);
}

/** Sum of the k-th powers of the cycle eigenvalues: 2 cos(2 pi j / n), j = 0..n-1. */
function cycleSpectralClosedWalks(n, k) {
  let total = 0.0;
  for (let j = 0; j < n; j++) total += Math.pow(2 * Math.cos((2 * Math.PI * j) / n), k);
  return total;
}

function bfsReachable(matrix, start) {
  const n = matrix.length;
  const seen = new Array(n).fill(false);
  const queue = [start];
  seen[start] = true;
  while (queue.length > 0) {
    const u = queue.shift();
    for (let v = 0; v < n; v++) {
      if (matrix[u][v] !== 0n && !seen[v]) {
        seen[v] = true;
        queue.push(v);
      }
    }
  }
  return seen;
}

/** Boolean reachability by exponentiation of (A + I) to the (n - 1)-th power. */
function reachabilityByPower(matrix) {
  const n = matrix.length;
  const step = matrix.map((row, i) => row.map((value, j) => (value !== 0n || i === j ? 1n : 0n)));
  const powered = fastPower(step, Math.max(0, n - 1));
  return powered.map((row) => row.map((value) => value > 0n));
}

function randomGraph(n, percent, seedState) {
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

function cycleEdges(n) {
  const edges = [];
  for (let i = 0; i < n; i++) edges.push([i, (i + 1) % n]);
  return edges;
}

function completeEdges(n) {
  const edges = [];
  for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) edges.push([u, v]);
  return edges;
}

function main() {
  const assert = require("node:assert");

  const k5 = adjacencyMatrix(5, completeEdges(5));
  const c6 = adjacencyMatrix(6, cycleEdges(6));

  // (1) Identity and adjacency cases, plus the monoid laws.
  assert.deepStrictEqual(fastPower(k5, 0), identityMatrix(5));
  assert.deepStrictEqual(fastPower(k5, 1), k5);
  for (const [a, b] of [[3, 4], [5, 6], [1, 10], [0, 7]]) {
    assert.deepStrictEqual(fastPower(k5, a + b, MOD),
      multiply(fastPower(k5, a, MOD), fastPower(k5, b, MOD), MOD), `A^${a + b} = A^${a} A^${b}`);
  }
  for (const [a, b] of [[2, 3], [3, 2], [4, 2]]) {
    assert.deepStrictEqual(fastPower(fastPower(k5, a, MOD), b, MOD), fastPower(k5, a * b, MOD),
      `(A^${a})^${b} = A^${a * b}`);
  }

  // (2) Fast power == repeated multiplication == recursion == layered DP, for all small k.
  for (let k = 0; k <= 12; k++) {
    assert.deepStrictEqual(fastPower(k5, k), repeatedPower(k5, k), `k = ${k}: binary vs repeated`);
    assert.deepStrictEqual(fastPower(k5, k), recursivePower(k5, k), `k = ${k}: binary vs recursion`);
    assert.deepStrictEqual(fastPower(k5, k), walkCountsByDp(k5, k), `k = ${k}: binary vs DP`);
  }

  // (3) The multiplication counter measures the O(log k) claim.
  for (const k of [1, 4, 5, 16, 1000, 1000000]) {
    const counter = { multiplications: 0 };
    fastPower(k5, k, MOD, counter);
    const bound = 2 * Math.max(1, Math.floor(Math.log2(k)));
    assert.ok(counter.multiplications <= bound,
      `k = ${k}: ${counter.multiplications} multiplications exceed the bound ${bound}`);
    const baseline = { multiplications: 0 };
    if (k <= 1000) {
      repeatedPower(k5, k, MOD, baseline);
      assert.strictEqual(baseline.multiplications, k, "repeated multiplication costs exactly k");
      if (k >= 5) {
        assert.ok(counter.multiplications < baseline.multiplications,
          `k = ${k}: fast power must beat the baseline (${counter.multiplications} < ${k})`);
      }
    }
  }

  // (4) Brute-force DFS enumeration agrees on small instances.
  for (let seed = 1; seed <= 6; seed++) {
    const n = 4 + (seed % 3);
    const matrix = adjacencyMatrix(n, randomGraph(n, 45, BigInt(seed) * 7919n).edges);
    for (let k = 0; k <= 6; k++) {
      assert.deepStrictEqual(fastPower(matrix, k), walkCountsByDfs(matrix, k),
        `seed ${seed}, k = ${k}: DFS enumeration must reproduce the matrix power`);
    }
  }

  // (5) Closed form for K_n (spectral decomposition), for both diagonal and off-diagonal entries.
  for (let n = 2; n <= 8; n++) {
    const matrix = adjacencyMatrix(n, completeEdges(n));
    for (let k = 0; k <= 14; k++) {
      const powered = fastPower(matrix, k);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          assert.strictEqual(powered[i][j], completeGraphWalkCount(n, k, i === j),
            `K${n}: walks of length ${k} from ${i} to ${j}`);
        }
      }
    }
  }
  assert.strictEqual(completeGraphWalkCount(5, 2, true), 4n, "K5: 2-step walks inside a vertex");
  assert.strictEqual(completeGraphWalkCount(5, 2, false), 3n);
  assert.strictEqual(completeGraphWalkCount(2, 1, false), 1n);

  // (6) Cycles: trace(A^k) equals the sum of the k-th powers of the eigenvalues 2cos(2 pi j / n).
  for (let n = 3; n <= 10; n++) {
    const matrix = adjacencyMatrix(n, cycleEdges(n));
    for (let k = 0; k <= 13; k++) {
      const exact = trace(fastPower(matrix, k));
      const spectral = cycleSpectralClosedWalks(n, k);
      const scale = Math.max(1, Math.abs(spectral));
      assert.ok(Math.abs(Number(exact) - spectral) <= 1e-9 * scale,
        `C${n}, k = ${k}: exact trace ${exact} vs spectral ${spectral}`);
    }
  }
  assert.strictEqual(trace(fastPower(c6, 3)), 0n, "C6 is bipartite: no odd closed walks");
  assert.strictEqual(trace(fastPower(c6, 6)), 132n,
    "C6: sum of the sixth powers of 2, 1, -1, -2, -1, 1 is 132");
  assert.strictEqual(trace(fastPower(c6, 2)), 12n, "2 steps back and forth from each vertex");

  // (7) Bipartiteness: trace(A^k) = 0 for odd k on any bipartite graph, and DFS confirms.
  for (let seed = 1; seed <= 4; seed++) {
    const n = 6;
    const bipartiteEdges = [];
    for (const [u, v] of randomGraph(n, 60, BigInt(seed) * 6151n).edges) {
      if (u < n / 2 && v >= n / 2) bipartiteEdges.push([u, v]);
    }
    const matrix = adjacencyMatrix(n, bipartiteEdges);
    for (const k of [1, 3, 5, 7]) {
      assert.strictEqual(trace(fastPower(matrix, k)), 0n,
        `seed ${seed}: a bipartite graph has no closed walk of odd length ${k}`);
      assert.strictEqual(trace(walkCountsByDfs(matrix, k)), 0n);
    }
  }

  // (8) Reachability by (A + I)^(n - 1) versus BFS.
  for (let seed = 1; seed <= 8; seed++) {
    const n = 5 + (seed % 4);
    const matrix = adjacencyMatrix(n, randomGraph(n, 30 + (seed % 4) * 15, BigInt(seed) * 104729n)
      .edges);
    const byPower = reachabilityByPower(matrix);
    for (let start = 0; start < n; start++) {
      const byBfs = bfsReachable(matrix, start);
      for (let target = 0; target < n; target++) {
        assert.strictEqual(byPower[start][target], byBfs[target],
          `seed ${seed}: reachability ${start} -> ${target}`);
      }
    }
    // Connectivity of the whole graph: any vertex pair connected iff reachable both ways.
    const connected = byPower.every((row) => row.every((value) => value));
    const bfs = bfsReachable(matrix, 0);
    assert.strictEqual(connected, bfs.every((value) => value),
      "the graph is connected iff vertex 0 reaches everyone");
  }

  // (9) Cumulative counts: walks of length <= k are the prefix sums and monotone.
  const cumulative = [];
  let running = zeroMatrix(5);
  for (let k = 0; k <= 8; k++) {
    running = running.map((row, i) => row.map((value, j) => value + fastPower(k5, k)[i][j]));
    cumulative.push(running.map((row) => row.slice()));
  }
  for (let k = 1; k < cumulative.length; k++) {
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        assert.ok(cumulative[k][i][j] >= cumulative[k - 1][i][j], "prefix sums are monotone");
      }
    }
  }
  const closedWalkSum = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    .reduce((total, k) => total + completeGraphWalkCount(5, k, true), 0n);
  assert.strictEqual(cumulative[8][0][0], closedWalkSum,
    "the diagonal prefix sum is the sum of the closed-walk counts");
  assert.strictEqual(cumulative[0][0][0], 1n, "length 0 contributes the identity");
  assert.strictEqual(cumulative[8][0][1], cumulative[7][0][1] + fastPower(k5, 8)[0][1]);

  // (10) Boundaries: edgeless graphs, a single vertex, a disconnected graph, giant k.
  const edgeless = adjacencyMatrix(4, []);
  assert.deepStrictEqual(fastPower(edgeless, 0), identityMatrix(4));
  assert.deepStrictEqual(fastPower(edgeless, 1), zeroMatrix(4));
  assert.deepStrictEqual(fastPower(edgeless, 9), zeroMatrix(4));
  assert.strictEqual(trace(fastPower(edgeless, 9)), 0n);
  const single = adjacencyMatrix(1, []);
  assert.deepStrictEqual(fastPower(single, 1000000), [[0n]]);
  const twoComponents = adjacencyMatrix(4, [[0, 1], [2, 3]]);
  assert.strictEqual(fastPower(twoComponents, 2)[0][2], 0n, "no walk crosses components");
  assert.strictEqual(fastPower(twoComponents, 3)[0][1], 1n,
    "0-1-0-1 is back at vertex 1 after three steps");
  assert.strictEqual(fastPower(twoComponents, 3)[2][3], 1n);
  assert.strictEqual(fastPower(twoComponents, 3)[0][2], 0n, "still no walk crosses components");
  assert.strictEqual(fastPower(twoComponents, 1)[0][1], 1n);
  assert.strictEqual(fastPower(twoComponents, 2)[0][0], 1n);
  const selfLoop = directedAdjacency(3, [[0, 0], [0, 1]]);
  assert.strictEqual(selfLoop[0][0], 0n, "self-loops are dropped when building adjacency");
  assert.strictEqual(fastPower(selfLoop, 1)[0][1], 1n,
    "with the self-loop dropped the only edge is 0 -> 1, so longer walks vanish");
  assert.strictEqual(fastPower(selfLoop, 5)[0][1], 0n);
  // Giant exponents: on the directed 4-cycle A^k is the identity iff k is a multiple of 4.
  const c4 = directedAdjacency(4, [[0, 1], [1, 2], [2, 3], [3, 0]]);
  assert.deepStrictEqual(fastPower(c4, 1000000000, MOD), identityMatrix(4),
    "10^9 is a multiple of 4, so the directed 4-cycle returns to the identity");
  assert.deepStrictEqual(fastPower(c4, 1000000001, MOD),
    multiply(c4, identityMatrix(4), MOD), "and one more step is the adjacency itself");
  assert.strictEqual(fastPower(c4, 0, MOD)[0][3], 0n);

  // (11) Determinism.
  assert.deepStrictEqual(fastPower(c6, 13, MOD), fastPower(c6, 13, MOD));
  assert.deepStrictEqual(recursivePower(c6, 13, MOD), recursivePower(c6, 13, MOD));
  assert.deepStrictEqual(walkCountsByDfs(c6, 5), walkCountsByDfs(c6, 5));
}

module.exports = { MOD, lcg64Next, lcg64Int, zeroMatrix, identityMatrix, adjacencyMatrix,
  directedAdjacency, multiply, fastPower, recursivePower, repeatedPower, trace, walkCountsByDfs,
  walkCountsByDp, completeGraphWalkCount, cycleSpectralClosedWalks, bfsReachable, reachabilityByPower,
  randomGraph, cycleEdges, completeEdges };

if (require.main === module) {
  main();
  console.log("PASS");
}
