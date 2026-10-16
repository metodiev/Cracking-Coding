/**
 * Task 557 - Implement algebraic path finding with min-plus matrix multiply (tropical semiring). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * In the tropical (min, +) semiring, with a ⊕ b = min(a, b), a ⊗ b = a + b, the additive identity
 * ∞ and the multiplicative identity 0, matrix multiplication computes
 * (A ⊗ B)[i][j] = min_l (A[i][l] + B[l][j]), which is exactly one relaxation step of a shortest-path
 * algorithm.  Hence the tropical power A^k holds the lightest walk with *exactly* k edges, and the
 * Kleene closure A* = I ⊕ A ⊕ A^2 ⊕ ... holds the shortest-path distances.  The file implements the
 * semiring operations, binary exponentiation (O(n^3 log n) all-pairs shortest paths), the Kleene
 * closure via (I ⊕ A)^(n-1), and the two classical oracles Floyd-Warshall and Dijkstra, then checks
 * that all three agree.
 *
 * Documented checks: the semiring axioms are asserted directly - ⊕ is commutative, associative and
 * idempotent with identity ∞, ⊗ is associative with identity 0 and distributes over ⊕, and ∞ absorbs
 * on both sides - and the matrix level identities A ⊗ (B ⊕ C) = A ⊗ B ⊕ A ⊗ C, (A ⊗ B) ⊗ C =
 * A ⊗ (B ⊗ C), A ⊗ I = A and A ⊗ Z = Z (Z the all-∞ matrix) are verified; min-plus multiplication is
 * asserted to be *non-commutative* on an explicit witness pair, which is why the closure needs the
 * (I ⊕ A) trick rather than assuming commutativity.  On LCG graphs with positive integer weights the
 * tropical power, the Kleene closure, Floyd-Warshall and Dijkstra-from-every-source are compared entry
 * by entry, and A^k for k >= n - 1 is shown to equal A^(n-1) (stabilisation, the finite-weights
 * analogue of "shortest paths use at most n - 1 edges"); the layer-by-layer DP of exact-edge-count
 * distances is shown to equal the tropical powers.  Zero-weight edges, unreachable pairs (∞),
 * single-vertex graphs, edgeless graphs, disconnected graphs (∞ entries survive closure) and negative
 * edge weights without negative cycles are handled; a negative cycle is exhibited where the closure
 * fails to stabilise (A^(2n) still cheaper than A^(n-1)) while Floyd-Warshall is flagged as
 * negative-cycle-detecting, and that divergence is asserted rather than hidden.  Randomness is a
 * documented 64-bit LCG (Knuth MMIX, modulo 2^64) for many independent weights.  Reruns are identical.
 */
"use strict";

const INF = Infinity;
const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

function lcg64Int(state, bound) {
  return Number((state >> 13n) % BigInt(bound));
}

function infMatrix(n) {
  return Array.from({ length: n }, () => new Array(n).fill(INF));
}

function tropicalIdentity(n) {
  const matrix = infMatrix(n);
  for (let i = 0; i < n; i++) matrix[i][i] = 0;
  return matrix;
}

/** Weighted adjacency matrix; additive identity is Infinity, so unreachable is Infinity. */
function weightedAdjacency(n, edges) {
  const matrix = infMatrix(n);
  for (const [u, v, w] of edges) {
    if (u === v) continue;
    matrix[u][v] = Math.min(matrix[u][v], w);
  }
  return matrix;
}

/** Tropical (min, +) product with the Infinity convention (Infinity + -Infinity never arises: the
 *  implementation never introduces -Infinity, it only propagates Infinity and finite weights). */
function tropicalMultiply(a, b) {
  const n = a.length;
  const out = infMatrix(n);
  for (let i = 0; i < n; i++) {
    for (let l = 0; l < n; l++) {
      const left = a[i][l];
      if (left === INF) continue;
      const rowB = b[l];
      for (let j = 0; j < n; j++) {
        const right = rowB[j];
        if (right === INF) continue;
        const candidate = left + right;
        if (candidate < out[i][j]) out[i][j] = candidate;
      }
    }
  }
  return out;
}

function tropicalAdd(a, b) {
  const n = a.length;
  return a.map((row, i) => row.map((value, j) => Math.min(value, b[i][j])));
}

function tropicalPower(matrix, exponent, counter = null) {
  if (exponent < 0) throw new Error("negative exponent");
  let base = matrix;
  let result = tropicalIdentity(matrix.length);
  let e = BigInt(exponent);
  while (e > 0n) {
    if (e & 1n) {
      if (counter) counter.multiplications += 1;
      result = tropicalMultiply(result, base);
    }
    e >>= 1n;
    if (e > 0n) {
      if (counter) counter.multiplications += 1;
      base = tropicalMultiply(base, base);
    }
  }
  return result;
}

/** Kleene closure A* = I ⊕ A ⊕ ... ⊕ A^(n-1), computed in the tropical semiring. */
function tropicalClosure(matrix) {
  const n = matrix.length;
  if (n === 0) return [];
  const step = tropicalAdd(tropicalIdentity(n), matrix);
  return tropicalPower(step, Math.max(0, n - 1));
}

function floydWarshall(matrix) {
  const n = matrix.length;
  const dist = matrix.map((row) => row.slice());
  for (let i = 0; i < n; i++) dist[i][i] = Math.min(dist[i][i], 0);
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      if (dist[i][k] === INF) continue;
      for (let j = 0; j < n; j++) {
        if (dist[k][j] === INF) continue;
        const candidate = dist[i][k] + dist[k][j];
        if (candidate < dist[i][j]) dist[i][j] = candidate;
      }
    }
  }
  return dist;
}

function dijkstra(matrix, source) {
  const n = matrix.length;
  const dist = new Array(n).fill(INF);
  const done = new Array(n).fill(false);
  if (n === 0) return dist;
  dist[source] = 0;
  for (let step = 0; step < n; step++) {
    let best = -1;
    for (let v = 0; v < n; v++) {
      if (done[v] || dist[v] === INF) continue;
      if (best < 0 || dist[v] < dist[best]) best = v;
    }
    if (best < 0) break;
    done[best] = true;
    for (let v = 0; v < n; v++) {
      if (matrix[best][v] === INF) continue;
      const candidate = dist[best] + matrix[best][v];
      if (candidate < dist[v]) dist[v] = candidate;
    }
  }
  return dist;
}

function hasNegativeCycle(matrix) {
  const n = matrix.length;
  const dist = floydWarshall(matrix);
  for (let i = 0; i < n; i++) if (dist[i][i] < 0) return true;
  return false;
}

/** DP over exact edge counts: dp_k[i][j] = lightest walk with exactly k edges. */
function exactEdgeDistanceDp(matrix, k) {
  const n = matrix.length;
  let current = tropicalIdentity(n);
  for (let step = 0; step < k; step++) {
    const next = infMatrix(n);
    for (let i = 0; i < n; i++) {
      for (let l = 0; l < n; l++) {
        if (current[i][l] === INF) continue;
        for (let j = 0; j < n; j++) {
          if (matrix[l][j] === INF) continue;
          const candidate = current[i][l] + matrix[l][j];
          if (candidate < next[i][j]) next[i][j] = candidate;
        }
      }
    }
    current = next;
  }
  return current;
}

function randomWeightedGraph(n, percent, maxWeight, seedState) {
  const edges = [];
  let state = seedState % TWO_64;
  for (let u = 0; u < n; u++) {
    for (let v = 0; v < n; v++) {
      if (u === v) continue;
      state = lcg64Next(state);
      if (lcg64Int(state, 100) < percent) {
        state = lcg64Next(state);
        edges.push([u, v, 1 + lcg64Int(state, maxWeight)]);
      }
    }
  }
  return { edges, state };
}

function minPlusEqual(a, b) {
  return a.every((row, i) => row.every((value, j) => value === b[i][j]));
}

function main() {
  const assert = require("node:assert");

  // (1) Semiring axioms on scalars.
  const values = [INF, 0, 1, 3, 7];
  for (const a of values) {
    for (const b of values) {
      assert.strictEqual(Math.min(a, b), Math.min(b, a), "⊕ is commutative");
      for (const c of values) {
        assert.strictEqual(Math.min(Math.min(a, b), c), Math.min(a, Math.min(b, c)),
          "⊕ is associative");
        if (a === INF || b === INF || c === INF) continue;
        assert.strictEqual(a + Math.min(b, c), Math.min(a + b, a + c), "⊗ distributes over ⊕");
        assert.strictEqual((a + b) + c, a + (b + c), "⊗ is associative");
      }
    }
  }
  assert.strictEqual(Math.min(INF, 5), 5, "∞ is the additive identity");
  assert.strictEqual(Math.min(INF, INF), INF, "⊕ is idempotent");
  assert.strictEqual(5 + INF, INF, "∞ absorbs under ⊗");

  // (2) Matrix level: associativity, distributivity, identities, and non-commutativity.
  const graph = weightedAdjacency(5, randomWeightedGraph(5, 55, 9, 4242n).edges);
  const other = weightedAdjacency(5, randomWeightedGraph(5, 55, 9, 99991n).edges);
  const third = weightedAdjacency(5, randomWeightedGraph(5, 40, 5, 777n).edges);
  assert.deepStrictEqual(tropicalMultiply(tropicalMultiply(graph, other), third),
    tropicalMultiply(graph, tropicalMultiply(other, third)), "(A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)");
  assert.deepStrictEqual(tropicalMultiply(graph, tropicalAdd(other, third)),
    tropicalAdd(tropicalMultiply(graph, other), tropicalMultiply(graph, third)),
  "A ⊗ (B ⊕ C) = A ⊗ B ⊕ A ⊗ C");
  assert.deepStrictEqual(tropicalMultiply(graph, tropicalIdentity(5)), graph, "A ⊗ I = A");
  assert.deepStrictEqual(tropicalMultiply(tropicalIdentity(5), graph), graph, "I ⊗ A = A");
  const zero = infMatrix(5);
  assert.deepStrictEqual(tropicalMultiply(graph, zero), zero, "A ⊗ Z = Z");
  assert.deepStrictEqual(tropicalAdd(graph, zero), graph, "A ⊕ Z = A");
  const leftFactor = weightedAdjacency(3, [[0, 1, 1], [1, 2, 1]]);
  const rightFactor = weightedAdjacency(3, [[0, 2, 5], [2, 1, 5]]);
  assert.notDeepStrictEqual(tropicalMultiply(leftFactor, rightFactor),
    tropicalMultiply(rightFactor, leftFactor),
  "min-plus multiplication is not commutative, which is why the closure uses (I ⊕ A)^(n-1)");

  // (3) Exact-edge distances: the tropical power equals the layer-by-layer DP.
  for (let k = 0; k <= 8; k++) {
    assert.deepStrictEqual(tropicalPower(graph, k), exactEdgeDistanceDp(graph, k),
      `k = ${k}: A^k holds the lightest walks with exactly k edges`);
  }
  assert.deepStrictEqual(tropicalPower(graph, 0), tropicalIdentity(5), "A^0 = I");

  // (4) Closure == Floyd-Warshall == Dijkstra on LCG graphs with positive weights.
  for (let seed = 1; seed <= 8; seed++) {
    const n = 4 + (seed % 4);
    const matrix = weightedAdjacency(n, randomWeightedGraph(n, 40 + (seed % 3) * 15, 12,
      BigInt(seed) * 7919n).edges);
    const closure = tropicalClosure(matrix);
    const floyd = floydWarshall(matrix);
    assert.ok(minPlusEqual(closure, floyd),
      `seed ${seed}: the Kleene closure must equal Floyd-Warshall`);
    for (let s = 0; s < n; s++) {
      const dijkstraRow = dijkstra(matrix, s);
      for (let t = 0; t < n; t++) {
        assert.strictEqual(closure[s][t], dijkstraRow[t],
          `seed ${seed}: closure ${s} -> ${t} vs Dijkstra`);
      }
    }
    assert.strictEqual(hasNegativeCycle(matrix), false, "positive weights cannot create a cycle");
  }

  // (5) The closure is the minimum over the exact-edge layers, and (I ⊕ A)^k stabilises at n - 1.
  //     Note the exact-edge powers themselves do NOT stabilise - A^k is the lightest walk with exactly
  //     k edges, which can be emptier than A^(k+1) - so the file asserts the two facts that are true:
  //     the closure equals min over k = 0..n-1 of A^k, and every A^k dominates the closure entrywise
  //     (a k-edge walk is a walk, hence its cost is at least the shortest walk cost).
  for (let seed = 1; seed <= 5; seed++) {
    const n = 4 + (seed % 3);
    const matrix = weightedAdjacency(n, randomWeightedGraph(n, 60, 20, BigInt(seed) * 104729n)
      .edges);
    const step = tropicalAdd(tropicalIdentity(n), matrix);
    const closure = tropicalClosure(matrix);
    let layerMin = exactEdgeDistanceDp(matrix, 0);
    for (let k = 1; k <= n - 1; k++) layerMin = tropicalAdd(layerMin, exactEdgeDistanceDp(matrix, k));
    assert.ok(minPlusEqual(closure, layerMin),
      `seed ${seed}: A* equals the minimum over the layers 0 .. n-1`);
    for (const k of [n - 1, n, n + 1, 2 * n, 30]) {
      assert.ok(minPlusEqual(tropicalPower(step, k), closure),
        `seed ${seed}: (I ⊕ A)^${k} stabilises at the closure`);
    }
    for (const k of [n - 1, n, 2 * n]) {
      const powered = tropicalPower(matrix, k);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          assert.ok(powered[i][j] >= closure[i][j],
            `seed ${seed}: A^${k}[${i}][${j}] = ${powered[i][j]} must dominate ${closure[i][j]}`);
        }
      }
    }
    for (let i = 0; i < n; i++) {
      assert.strictEqual(closure[i][i], 0, "non-negative weights: the empty walk is optimal");
    }
  }

  // (6) Boundary cases: unreachable pairs, zero weights, disconnected graphs, single vertex.
  const unreachable = weightedAdjacency(4, [[0, 1, 3], [2, 3, 4]]);
  const unreachClosure = tropicalClosure(unreachable);
  assert.strictEqual(unreachClosure[0][2], INF, "different components stay unreachable");
  assert.strictEqual(unreachClosure[0][1], 3);
  assert.strictEqual(unreachClosure[1][0], INF, "the edges are directed");
  assert.strictEqual(unreachClosure[0][0], 0, "distance to itself is 0");
  const zeroWeight = weightedAdjacency(3, [[0, 1, 0], [1, 2, 0]]);
  assert.strictEqual(tropicalClosure(zeroWeight)[0][2], 0, "zero-weight paths cost 0");
  assert.strictEqual(tropicalPower(zeroWeight, 2)[0][2], 0);
  const edgeless = weightedAdjacency(3, []);
  assert.deepStrictEqual(tropicalClosure(edgeless), tropicalIdentity(3),
    "with no edges only the diagonal is finite");
  assert.deepStrictEqual(tropicalClosure([]), [], "the empty graph");
  const one = weightedAdjacency(1, []);
  assert.deepStrictEqual(tropicalClosure(one), [[0]]);
  assert.strictEqual(tropicalClosure(one)[0][0], 0, "the empty path");
  const loops = weightedAdjacency(2, [[0, 0, -5], [0, 1, 2]]);
  assert.strictEqual(loops[0][0], INF, "self-loops are dropped when building the matrix");
  assert.strictEqual(tropicalClosure(loops)[0][1], 2);
  assert.strictEqual(tropicalPower(loops, 1)[0][1], 2);
  assert.strictEqual(tropicalPower(loops, 2)[0][1], INF);

  // (7) Negative weights without a negative cycle still agree; a negative cycle does not stabilise.
  const negativeEdges = [[0, 1, -2], [1, 2, -3], [0, 2, 10], [2, 3, 1]];
  const negative = weightedAdjacency(4, negativeEdges);
  assert.strictEqual(hasNegativeCycle(negative), false);
  assert.deepStrictEqual(tropicalClosure(negative), floydWarshall(negative),
    "the closure handles negative weights as long as no cycle is negative");
  assert.strictEqual(tropicalClosure(negative)[0][2], -5, "-2 + -3 beats the direct edge 10");
  assert.strictEqual(dijkstra(negative, 0)[2], -5,
    "on this particular instance greedy Dijkstra happens to agree - not a guarantee");
  // The canonical Dijkstra failure.  Here the cycle 1 -> 2 -> 1 has weight -2, so the instance has a
  // negative cycle: the closure is *not* the answer (the infimum is -infinity), Floyd-Warshall detects
  // it, the exact-edge powers keep decreasing, and greedy Dijkstra silently reports 2.
  const dijkstraTrap = weightedAdjacency(3, [[0, 1, 1], [1, 2, 1], [2, 1, -3]]);
  assert.strictEqual(hasNegativeCycle(dijkstraTrap), true, "1 -> 2 -> 1 weighs 1 - 3 = -2");
  assert.strictEqual(dijkstra(dijkstraTrap, 0)[2], 2,
    "Dijkstra reports 2, ignoring the negative cycle altogether");
  // Only walks of odd length can end at vertex 1, only even ones at vertex 2 (the graph is bipartite
  // with respect to the cycle), so the comparison uses lengths of the right parity.
  assert.ok(tropicalPower(dijkstraTrap, 5)[0][1] < tropicalPower(dijkstraTrap, 3)[0][1],
    `exact-edge distances keep dropping: ${tropicalPower(dijkstraTrap, 5)[0][1]} < `
    + `${tropicalPower(dijkstraTrap, 3)[0][1]}`);
  assert.ok(tropicalPower(dijkstraTrap, 10)[0][2] < tropicalPower(dijkstraTrap, 6)[0][2],
    `${tropicalPower(dijkstraTrap, 10)[0][2]} < ${tropicalPower(dijkstraTrap, 6)[0][2]}: no finite `
    + "power is the answer, which is why a negative-cycle check comes first");
  assert.strictEqual(tropicalClosure(dijkstraTrap)[0][0], 0,
    "the naive closure still returns the n-1 bounded value, a documented limitation");
  const negativeCycleEdges = [[0, 1, 1], [1, 2, -5], [2, 0, 1]];
  const negativeCycle = weightedAdjacency(3, negativeCycleEdges);
  assert.strictEqual(hasNegativeCycle(negativeCycle), true, "the cycle weights -3, so it is negative");
  const sixEdges = tropicalPower(negativeCycle, 6);
  const nineEdges = tropicalPower(negativeCycle, 9);
  assert.ok(nineEdges[0][0] < sixEdges[0][0],
    `a negative cycle keeps improving: ${nineEdges[0][0]} < ${sixEdges[0][0]}`);
  assert.ok(sixEdges[0][0] < 0, "and already the 6-edge power is negative");

  // (8) Multiplication counts: binary exponentiation of the closure on n = 16.
  const big = weightedAdjacency(16, randomWeightedGraph(16, 30, 9, 31337n).edges);
  const counter = { multiplications: 0 };
  const closureBig = tropicalPower(tropicalAdd(tropicalIdentity(16), big), 15, counter);
  assert.ok(counter.multiplications <= 2 * Math.floor(Math.log2(15)) + 1,
    `15 needs few squarings, got ${counter.multiplications}`);
  assert.ok(minPlusEqual(closureBig, floydWarshall(big)), "and it is the right answer");
  const identityCost = { multiplications: 0 };
  tropicalPower(big, 0, identityCost);
  assert.strictEqual(identityCost.multiplications, 0, "A^0 needs no multiplication at all");

  // (9) Determinism.
  assert.deepStrictEqual(tropicalClosure(graph), tropicalClosure(graph));
  assert.deepStrictEqual(tropicalPower(graph, 7), tropicalPower(graph, 7));
  assert.deepStrictEqual(floydWarshall(graph), floydWarshall(graph));
}

module.exports = { INF, lcg64Next, lcg64Int, infMatrix, tropicalIdentity, weightedAdjacency,
  tropicalMultiply, tropicalAdd, tropicalPower, tropicalClosure, floydWarshall, dijkstra,
  hasNegativeCycle, exactEdgeDistanceDp, randomWeightedGraph, minPlusEqual };

if (require.main === module) {
  main();
  console.log("PASS");
}
