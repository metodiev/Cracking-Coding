/** Task 448 - Implement parametric search for optimizing monotone predicates. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function parametricMin(candidates, predicate) {
  let lo = 0;
  let hi = candidates.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (predicate(candidates[mid])) hi = mid;
    else lo = mid + 1;
  }
  return lo < candidates.length ? candidates[lo] : null;
}

function linearScan(candidates, predicate) {
  for (const candidate of candidates) {
    if (predicate(candidate)) return candidate;
  }
  return null;
}

function reachableAll(n, edges, d, source) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [u, v, w] of edges) {
    if (w <= d) {
      adjacency[u].push(v);
      adjacency[v].push(u);
    }
  }
  const seen = new Array(n).fill(false);
  seen[source] = true;
  const stack = [source];
  while (stack.length > 0) {
    const x = stack.pop();
    for (const y of adjacency[x]) {
      if (!seen[y]) {
        seen[y] = true;
        stack.push(y);
      }
    }
  }
  return seen.every(Boolean);
}

function componentsLeq(n, edges, d, k) {
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x) {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  let components = n;
  for (const [u, v, w] of edges) {
    if (w <= d) {
      const ru = find(u);
      const rv = find(v);
      if (ru !== rv) {
        parent[ru] = rv;
        components--;
      }
    }
  }
  return components <= k;
}

function lcgGraphs(count, seed, maxN) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    const n = 3 + (seed % (maxN - 2));
    const edges = [];
    for (let node = 1; node < n; node++) {
      seed = (seed * 75 + 74) % 65537;
      const parent = seed % node;
      seed = (seed * 75 + 74) % 65537;
      const weight = 1 + (seed % 30);
      edges.push([parent, node, weight]);
    }
    for (let extra = 0; extra < n; extra++) {
      seed = (seed * 75 + 74) % 65537;
      const u = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const v = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const weight = 1 + (seed % 30);
      if (u !== v) edges.push([u, v, weight]);
    }
    graphs.push([n, edges]);
  }
  return graphs;
}

function assertMonotone(candidates, predicate) {
  const values = candidates.map(predicate);
  for (let i = 1; i < values.length; i++) {
    assertOk(!(values[i - 1] === true && values[i] === false), "predicate not monotone");
  }
}

function assertOk(condition, label) {
  if (!condition) throw new Error("FAILED: " + label);
}

module.exports = { parametricMin, linearScan, reachableAll, componentsLeq, lcgGraphs, assertMonotone };

if (require.main === module) {
  const assert = require("node:assert");

  const edges = [[0, 1, 4], [1, 2, 1], [2, 3, 7]];
  const candidates = [1, 4, 7];
  const reach = (d) => reachableAll(4, edges, d, 0);
  const comps2 = (d) => componentsLeq(4, edges, d, 2);
  const comps1 = (d) => componentsLeq(4, edges, d, 1);
  assert.deepStrictEqual(candidates.map(reach), [false, false, true]);
  assert.deepStrictEqual(candidates.map(comps2), [false, true, true]);
  assert.strictEqual(parametricMin(candidates, reach), 7);
  assert.strictEqual(parametricMin(candidates, comps2), 4);
  assert.strictEqual(parametricMin(candidates, comps1), 7);
  assertMonotone(candidates, reach);
  assertMonotone(candidates, comps2);

  assert.strictEqual(parametricMin([5], (d) => reachableAll(2, [[0, 1, 5]], d, 0)), 5);
  assert.strictEqual(parametricMin([1, 2, 3], () => false), null);
  assert.strictEqual(parametricMin([1, 2, 3], () => true), 1);

  let checked = 0;
  for (const [n, graphEdges] of lcgGraphs(20, 448, 12)) {
    const graphCandidates = [...new Set(graphEdges.map((edge) => edge[2]))].sort((a, b) => a - b);
    for (const source of [0, n - 1]) {
      const predicate = (d) => reachableAll(n, graphEdges, d, source);
      assertMonotone(graphCandidates, predicate);
      assert.strictEqual(parametricMin(graphCandidates, predicate), linearScan(graphCandidates, predicate), `source ${source}`);
      checked++;
    }
    for (const k of [1, 2, 4]) {
      const predicate = (d) => componentsLeq(n, graphEdges, d, k);
      assertMonotone(graphCandidates, predicate);
      assert.strictEqual(parametricMin(graphCandidates, predicate), linearScan(graphCandidates, predicate), `k=${k}`);
      checked++;
    }
  }
  assert.strictEqual(checked, 100);
  console.log("PASS");
}
