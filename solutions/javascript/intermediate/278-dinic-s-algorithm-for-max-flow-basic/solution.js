/** Task 278 - Dinic's algorithm for max flow (basic). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function dinic(n, capacity, source, sink) {
  const residual = capacity.map((row) => row.slice());
  let total = 0;
  for (;;) {
    const level = new Array(n).fill(-1);
    level[source] = 0;
    const queue = [source];
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (let v = 0; v < n; v++) {
        if (level[v] === -1 && residual[u][v] > 0) {
          level[v] = level[u] + 1;
          queue.push(v);
        }
      }
    }
    if (level[sink] === -1) return total;

    const pointer = new Array(n).fill(0);
    const dfs = (u, pushed) => {
      if (u === sink) return pushed;
      while (pointer[u] < n) {
        const v = pointer[u];
        if (residual[u][v] > 0 && level[v] === level[u] + 1) {
          const here = dfs(v, Math.min(pushed, residual[u][v]));
          if (here > 0) {
            residual[u][v] -= here;
            residual[v][u] += here;
            return here;
          }
        }
        pointer[u]++;
      }
      return 0;
    };
    for (;;) {
      const pushed = dfs(source, Infinity);
      if (pushed === 0) break;
      total += pushed;
    }
  }
}

function edmondsKarp(n, capacity, source, sink) {
  const residual = capacity.map((row) => row.slice());
  let total = 0;
  for (;;) {
    const parent = new Array(n).fill(-1);
    parent[source] = source;
    const queue = [source];
    for (let head = 0; head < queue.length && parent[sink] === -1; head++) {
      const u = queue[head];
      for (let v = 0; v < n; v++) {
        if (parent[v] === -1 && residual[u][v] > 0) {
          parent[v] = u;
          queue.push(v);
        }
      }
    }
    if (parent[sink] === -1) return total;
    let bottleneck = Infinity;
    for (let v = sink; v !== source; v = parent[v]) bottleneck = Math.min(bottleneck, residual[parent[v]][v]);
    for (let v = sink; v !== source; v = parent[v]) {
      const u = parent[v];
      residual[u][v] -= bottleneck;
      residual[v][u] += bottleneck;
    }
    total += bottleneck;
  }
}

function matrixFromEdges(n, edges) {
  const capacity = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [u, v, cap] of edges) capacity[u][v] += cap;
  return capacity;
}

function lcgCapacities(n, seed = 97531) {
  const capacity = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let u = 0; u < n; u++) {
    for (let v = 0; v < n; v++) {
      if (u === v) continue;
      seed = (seed * 75 + 74) % 65537;
      if (seed % 4 === 0) capacity[u][v] = seed % 10;
    }
  }
  return capacity;
}

function minCutBruteForce(n, capacity, source, sink) {
  const others = [];
  for (let v = 0; v < n; v++) if (v !== source && v !== sink) others.push(v);
  let best = Infinity;
  for (let mask = 0; mask < 1 << others.length; mask++) {
    const side = new Set([source]);
    for (let i = 0; i < others.length; i++) if ((mask >> i) & 1) side.add(others[i]);
    let cut = 0;
    for (let u = 0; u < n; u++) {
      if (!side.has(u)) continue;
      for (let v = 0; v < n; v++) if (!side.has(v)) cut += capacity[u][v];
    }
    best = Math.min(best, cut);
  }
  return best;
}

module.exports = { dinic, edmondsKarp, matrixFromEdges, lcgCapacities, minCutBruteForce };

if (require.main === module) {
  const assert = require("node:assert");
  const clrs = matrixFromEdges(6, [
    [0, 1, 16], [0, 2, 13], [1, 2, 10], [2, 1, 4], [1, 3, 12],
    [3, 2, 9], [2, 4, 14], [4, 3, 7], [3, 5, 20], [4, 5, 4],
  ]);
  assert.strictEqual(dinic(6, clrs, 0, 5), 23);
  assert.strictEqual(edmondsKarp(6, clrs, 0, 5), 23);

  assert.strictEqual(dinic(4, matrixFromEdges(4, [[0, 1, 3], [0, 2, 2], [1, 3, 2], [2, 3, 3]]), 0, 3), 4);
  const crossing = matrixFromEdges(4, [[0, 1, 1], [0, 2, 1], [1, 2, 1], [1, 3, 1], [2, 3, 1]]);
  assert.strictEqual(dinic(4, crossing, 0, 3), 2);
  assert.strictEqual(dinic(3, matrixFromEdges(3, [[0, 1, 1]]), 0, 2), 0);

  for (let trial = 0; trial < 150; trial++) {
    const capacity = lcgCapacities(6);
    assert.strictEqual(dinic(6, capacity, 0, 5), edmondsKarp(6, capacity, 0, 5), `trial ${trial}`);
  }

  const layered = matrixFromEdges(8, [
    [0, 1, 10], [0, 2, 10], [1, 3, 5], [1, 4, 6], [2, 5, 6],
    [2, 6, 5], [3, 7, 4], [4, 7, 7], [5, 7, 5], [6, 7, 4],
  ]);
  assert.strictEqual(dinic(8, layered, 0, 7), edmondsKarp(8, layered, 0, 7));
  assert.strictEqual(dinic(8, layered, 0, 7), minCutBruteForce(8, layered, 0, 7));
  assert.strictEqual(dinic(6, clrs, 0, 5), minCutBruteForce(6, clrs, 0, 5));
  console.log("PASS");
}
