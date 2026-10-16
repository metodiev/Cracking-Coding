/**
 * Task 538 - Implement Christofides algorithm outline for metric TSP. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Full Christofides implementation on Euclidean point sets: (1) a deterministic Prim minimum
 * spanning tree (ties by smallest index), (2) the odd-degree vertices of that tree, (3) a
 * minimum-weight perfect matching on them - exact subset DP over bitmasks rather than a greedy
 * blossom shortcut, so the 1.5-approximation argument actually applies - (4) the Eulerian circuit
 * of tree + matching by Hierholzer's algorithm over explicit edge ids (the multigraph is
 * connected and every vertex has even degree), and (5) the shortcut walk that removes repeated
 * vertices, which can only shorten the tour by the triangle inequality.
 *
 * Documented checks: the matching DP is cross-checked against brute-force enumeration of every
 * perfect matching on the same odd set (equal weights, <= 1e-9); the tree+matching multigraph has
 * all-even degrees and the Hierholzer circuit uses each multigraph edge exactly once and returns
 * to its start; the final tour is a permutation of all points with cost <= the Eulerian cost, at
 * most 1.5 x the exact Held-Karp optimum, and at least the optimum; the matching alone is at most
 * OPT/2 and the MST at most OPT (the two halves of the classical proof); and the degenerate
 * cases n = 1, 2, 3, duplicated points and a hand-checked unit square (= optimal tour) are
 * asserted, together with determinism.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function lcgPoints(n, seed, limit) {
  const points = [];
  let s = seed % 65537;
  for (let i = 0; i < n; i++) {
    s = lcgNext(s);
    const x = s % limit;
    s = lcgNext(s);
    const y = s % limit;
    points.push([x, y]);
  }
  return points;
}

function distanceMatrix(points) {
  const n = points.length;
  const d = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const dx = points[i][0] - points[j][0];
      const dy = points[i][1] - points[j][1];
      d[i][j] = Math.sqrt(dx * dx + dy * dy);
    }
  }
  return d;
}

function tourCost(d, tour) {
  let total = 0;
  for (let i = 0; i < tour.length; i++) total += d[tour[i]][tour[(i + 1) % tour.length]];
  return total;
}

function tourIsPermutation(n, tour) {
  if (tour.length !== n) return false;
  return tour.slice().sort((a, b) => a - b).every((v, i) => v === i);
}

/** Prim from vertex 0 with smallest-index tie-breaking; returns weighted tree edges. */
function minimumSpanningTree(n, d) {
  const inTree = new Array(n).fill(false);
  const best = new Array(n).fill(Infinity);
  const parent = new Array(n).fill(-1);
  const edges = [];
  let weight = 0;
  best[0] = 0;
  for (let step = 0; step < n; step++) {
    let u = -1;
    for (let i = 0; i < n; i++) {
      if (inTree[i]) continue;
      if (u === -1 || best[i] < best[u]) u = i;
    }
    if (u === -1 || best[u] === Infinity) break;
    inTree[u] = true;
    if (parent[u] !== -1) {
      edges.push([parent[u], u, best[u]]);
      weight += best[u];
    }
    for (let v = 0; v < n; v++) {
      if (inTree[v] || v === u) continue;
      if (d[u][v] < best[v]) {
        best[v] = d[u][v];
        parent[v] = u;
      }
    }
  }
  return { edges, weight, parent, connected: inTree.every(Boolean) };
}

function degrees(n, edges) {
  const deg = new Array(n).fill(0);
  for (const [u, v] of edges) {
    deg[u] += 1;
    deg[v] += 1;
  }
  return deg;
}

/** Exact min-weight perfect matching over a complete graph, by subset DP (k must be even). */
function minWeightPerfectMatching(cost) {
  const k = cost.length;
  if (k % 2 !== 0) throw new Error("perfect matching needs an even number of vertices");
  const size = 1 << k;
  const dp = new Array(size).fill(Infinity);
  const choice = new Array(size).fill(null);
  dp[0] = 0;
  for (let mask = 0; mask < size; mask++) {
    if (dp[mask] === Infinity) continue;
    let i = -1;
    for (let b = 0; b < k; b++) {
      if (!(mask & (1 << b))) {
        i = b;
        break;
      }
    }
    if (i === -1) continue;
    for (let j = i + 1; j < k; j++) {
      if (mask & (1 << j)) continue;
      const next = mask | (1 << i) | (1 << j);
      const candidate = dp[mask] + cost[i][j];
      if (candidate < dp[next]) {
        dp[next] = candidate;
        choice[next] = [mask, i, j];
      }
    }
  }
  const pairs = [];
  let mask = size - 1;
  while (mask !== 0) {
    const [prev, i, j] = choice[mask];
    pairs.push([i, j]);
    mask = prev;
  }
  return { weight: dp[size - 1], pairs };
}

/** Independent ground truth: enumerate every perfect matching. */
function bruteMatching(cost) {
  const k = cost.length;
  const used = new Array(k).fill(false);
  let best = Infinity;
  let bestPairs = null;
  const chosen = [];
  const rec = (weight) => {
    let i = -1;
    for (let b = 0; b < k; b++) {
      if (!used[b]) {
        i = b;
        break;
      }
    }
    if (i === -1) {
      if (weight < best) {
        best = weight;
        bestPairs = chosen.slice();
      }
      return;
    }
    used[i] = true;
    for (let j = i + 1; j < k; j++) {
      if (used[j]) continue;
      used[j] = true;
      chosen.push([i, j]);
      rec(weight + cost[i][j]);
      chosen.pop();
      used[j] = false;
    }
    used[i] = false;
  };
  rec(0);
  return { weight: best, pairs: bestPairs };
}

/** Hierholzer over explicit edge ids; adjacency lists are sorted, so the circuit is unique. */
function eulerianCircuit(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  edges.forEach(([u, v], id) => {
    adjacency[u].push(id);
    adjacency[v].push(id);
  });
  for (const list of adjacency) list.sort((a, b) => a - b);
  const used = new Array(edges.length).fill(false);
  const stack = [0];
  const circuit = [];
  while (stack.length) {
    const u = stack[stack.length - 1];
    let picked = -1;
    while (adjacency[u].length) {
      const id = adjacency[u].shift();
      if (!used[id]) {
        picked = id;
        break;
      }
    }
    if (picked === -1) {
      circuit.push(stack.pop());
    } else {
      used[picked] = true;
      const [a, b] = edges[picked];
      stack.push(a === u ? b : a);
    }
  }
  circuit.reverse();
  return circuit;
}

function shortcutWalk(circuit) {
  const seen = new Set();
  const tour = [];
  for (const v of circuit) {
    if (!seen.has(v)) {
      seen.add(v);
      tour.push(v);
    }
  }
  return tour;
}

function christofides(points) {
  const n = points.length;
  const d = distanceMatrix(points);
  if (n <= 1) {
    return { tour: n === 1 ? [0] : [], cost: 0, mstWeight: 0, matchingWeight: 0, odd: [],
      circuit: n === 1 ? [0] : [], matchingEdges: [], treeEdges: [], d };
  }
  const tree = minimumSpanningTree(n, d);
  const deg = degrees(n, tree.edges);
  const odd = [];
  for (let i = 0; i < n; i++) if (deg[i] % 2 === 1) odd.push(i);
  const k = odd.length;
  const cost = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => d[odd[i]][odd[j]]));
  const matching = minWeightPerfectMatching(cost);
  const matchingEdges = matching.pairs.map(([i, j]) => [odd[i], odd[j]]);
  const multigraph = tree.edges.map(([u, v]) => [u, v]).concat(matchingEdges);
  const circuit = eulerianCircuit(n, multigraph);
  const tour = shortcutWalk(circuit);
  return { tour, cost: tourCost(d, tour), mstWeight: tree.weight,
    matchingWeight: matching.weight, odd, circuit, matchingEdges, treeEdges: tree.edges,
    multigraph, d };
}

/** Exact Held-Karp optimum - the ratio witness. */
function heldKarp(d, start = 0) {
  const n = d.length;
  if (n <= 1) return 0;
  const full = 1 << n;
  const dp = Array.from({ length: n }, () => new Float64Array(full).fill(Infinity));
  dp[start][1 << start] = 0;
  for (let mask = 1; mask < full; mask++) {
    for (let u = 0; u < n; u++) {
      if (!(mask & (1 << u))) continue;
      const current = dp[u][mask];
      if (current === Infinity) continue;
      for (let v = 0; v < n; v++) {
        if (mask & (1 << v)) continue;
        const next = mask | (1 << v);
        const candidate = current + d[u][v];
        if (candidate < dp[v][next]) dp[v][next] = candidate;
      }
    }
  }
  let best = Infinity;
  for (let u = 0; u < n; u++) {
    const candidate = dp[u][full - 1] + d[u][start];
    if (candidate < best) best = candidate;
  }
  return best;
}

function main() {
  const assert = require("node:assert");

  // (1) Matching DP versus brute-force enumeration (independent method) on symmetric costs.
  for (let seed = 1; seed <= 4; seed++) {
    const k = 4 + 2 * (seed % 2);
    const cost = Array.from({ length: k }, (_, i) =>
      Array.from({ length: k }, (_, j) => i === j ? 0 : ((i * 7 + j * 13 + seed * 5) % 17) + 1));
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) cost[i][j] = Math.max(cost[i][j], cost[j][i]);
      cost[i][i] = 0;
    }
    const dp = minWeightPerfectMatching(cost);
    const brute = bruteMatching(cost);
    assert.ok(Math.abs(dp.weight - brute.weight) <= 1e-9,
      `seed ${seed}: matching DP ${dp.weight} vs brute force ${brute.weight}`);
    assert.strictEqual(dp.pairs.length, k / 2);
    for (const [i, j] of dp.pairs) assert.ok(cost[i][j] > 0 && i < j);
    const used = new Set();
    for (const [i, j] of dp.pairs) {
      used.add(i);
      used.add(j);
    }
    assert.strictEqual(used.size, k, "the reported matching covers every odd vertex once");
  }

  // (2) Hand-checked unit square: MST 3 + matching 1 = 4-edge multigraph, tour is optimal.
  const square = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const squareRun = christofides(square);
  assert.deepStrictEqual(squareRun.odd, [2, 3],
    "Prim from vertex 0 ties by smallest index, leaving the far corners odd");
  assert.ok(Math.abs(squareRun.mstWeight - 3) <= 1e-9);
  assert.ok(Math.abs(squareRun.matchingWeight - 1) <= 1e-9);
  assert.deepStrictEqual(squareRun.tour, [0, 1, 2, 3]);
  assert.ok(Math.abs(squareRun.cost - 4) <= 1e-9);
  assert.ok(Math.abs(squareRun.cost - heldKarp(squareRun.d)) <= 1e-9);

  // (3) Random Euclidean instances: structural invariants, ratio bounds, determinism.
  for (let seed = 11; seed <= 18; seed++) {
    const n = 6 + (seed % 6);
    const points = lcgPoints(n, seed * 29, 25);
    const run = christofides(points);
    const optimal = heldKarp(run.d);

    assert.ok(tourIsPermutation(n, run.tour), `seed ${seed}: tour is a permutation`);
    assert.ok(tourIsPermutation(n, shortcutWalk(run.circuit)), `seed ${seed}: shortcut walk is one too`);
    assert.ok(run.odd.length % 2 === 0, `seed ${seed}: an even number of odd-degree vertices`);
    assert.ok(Math.abs(run.matchingWeight - minWeightPerfectMatching(
      run.odd.map((u) => run.odd.map((v) => run.d[u][v]))).weight) <= 1e-9);

    const deg = degrees(n, run.multigraph);
    for (let i = 0; i < n; i++) {
      assert.strictEqual(deg[i] % 2, 0, `seed ${seed}: vertex ${i} has even multigraph degree`);
    }
    assert.strictEqual(run.circuit.length, run.multigraph.length + 1,
      `seed ${seed}: an Eulerian circuit uses each edge once`);
    assert.strictEqual(run.circuit[0], run.circuit[run.circuit.length - 1]);
    const seenEdges = new Set();
    for (let i = 0; i + 1 < run.circuit.length; i++) {
      const u = run.circuit[i];
      const v = run.circuit[i + 1];
      let hit = -1;
      run.multigraph.forEach(([a, b], id) => {
        if (hit === -1 && !seenEdges.has(id) && ((a === u && b === v) || (a === v && b === u))) {
          hit = id;
        }
      });
      assert.ok(hit >= 0, `seed ${seed}: circuit hop ${u}->${v} must be a multigraph edge`);
      seenEdges.add(hit);
    }
    assert.strictEqual(seenEdges.size, run.multigraph.length);
    for (let i = 0; i < n; i++) assert.ok(run.circuit.includes(i), "circuit covers every vertex");

    assert.ok(run.cost <= tourCost(run.d, run.circuit) + 1e-9,
      `seed ${seed}: shortcutting cannot lengthen the walk (triangle inequality)`);
    assert.ok(run.cost >= optimal - 1e-9, `seed ${seed}: cannot beat the optimum`);
    assert.ok(run.cost <= 1.5 * optimal + 1e-9,
      `seed ${seed}: Christofides bound ${run.cost} <= 1.5 * ${optimal}`);
    assert.ok(run.matchingWeight <= 0.5 * optimal + 1e-9,
      `seed ${seed}: matching <= OPT/2`);
    assert.ok(run.mstWeight <= optimal + 1e-9, `seed ${seed}: MST <= OPT`);

    assert.deepStrictEqual(christofides(points).tour, run.tour, `seed ${seed}: deterministic`);
    assert.deepStrictEqual(christofides(points).circuit, run.circuit);
  }

  // (4) Degenerate inputs.
  assert.deepStrictEqual(christofides([[4, 4]]).tour, [0]);
  assert.strictEqual(christofides([[4, 4]]).cost, 0);
  const pair = christofides([[0, 0], [3, 0]]);
  assert.deepStrictEqual(pair.tour, [0, 1]);
  assert.ok(Math.abs(pair.cost - 6) <= 1e-9, "go there and back");
  assert.ok(Math.abs(pair.mstWeight - 3) <= 1e-9);
  assert.deepStrictEqual(pair.odd, [0, 1]);
  const triangle = christofides([[0, 0], [4, 0], [0, 3]]);
  assert.ok(Math.abs(triangle.cost - 12) <= 1e-9, "3 + 4 + 5 perimeter");
  assert.ok(Math.abs(triangle.cost - heldKarp(triangle.d)) <= 1e-9);
  const duplicated = christofides([[1, 1], [1, 1], [5, 1], [5, 1]]);
  assert.ok(tourIsPermutation(4, duplicated.tour), "duplicate points still give a permutation");
  assert.ok(duplicated.cost <= 8 + 1e-9, "two paid hops of 4 with the twins free");
  assert.ok(duplicated.cost >= heldKarp(duplicated.d) - 1e-9);
  assert.ok(Math.abs(heldKarp(duplicated.d) - 8) <= 1e-9);
}

module.exports = { lcgNext, lcgPoints, distanceMatrix, tourCost, tourIsPermutation,
  minimumSpanningTree, degrees, minWeightPerfectMatching, bruteMatching, eulerianCircuit,
  shortcutWalk, christofides, heldKarp };

if (require.main === module) {
  main();
  console.log("PASS");
}
