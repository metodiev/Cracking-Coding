/** Task 279 - Min cut using max flow (s-t cut extraction). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function maxFlow(n, capacity, source, sink) {
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
    if (parent[sink] === -1) return { value: total, residual };
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

function minCut(n, capacity, source, sink) {
  const { value, residual } = maxFlow(n, capacity, source, sink);
  const side = new Array(n).fill(false);
  side[source] = true;
  const queue = [source];
  for (let head = 0; head < queue.length; head++) {
    const u = queue[head];
    for (let v = 0; v < n; v++) {
      if (!side[v] && residual[u][v] > 0) {
        side[v] = true;
        queue.push(v);
      }
    }
  }
  const inS = new Set();
  const inT = new Set();
  for (let v = 0; v < n; v++) (side[v] ? inS : inT).add(v);
  const cutEdges = [];
  for (const u of inS) for (const v of inT) if (capacity[u][v] > 0) cutEdges.push([u, v, capacity[u][v]]);
  return { value, inS, inT, cutEdges };
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

function matrixFromEdges(n, edges) {
  const capacity = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [u, v, cap] of edges) capacity[u][v] += cap;
  return capacity;
}

function lcgCapacities(n, seed = 24680) {
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

module.exports = { maxFlow, minCut, minCutBruteForce, matrixFromEdges, lcgCapacities };

if (require.main === module) {
  const assert = require("node:assert");
  const clrs = matrixFromEdges(6, [
    [0, 1, 16], [0, 2, 13], [1, 2, 10], [2, 1, 4], [1, 3, 12],
    [3, 2, 9], [2, 4, 14], [4, 3, 7], [3, 5, 20], [4, 5, 4],
  ]);
  let result = minCut(6, clrs, 0, 5);
  assert.strictEqual(result.value, 23);
  assert.strictEqual(result.cutEdges.reduce((a, [, , c]) => a + c, 0), 23);
  assert.ok(result.inS.has(0) && result.inT.has(5));
  assert.ok(result.cutEdges.every(([, v]) => result.inT.has(v)));
  assert.strictEqual(minCutBruteForce(6, clrs, 0, 5), 23);

  const diamond = matrixFromEdges(4, [[0, 1, 3], [0, 2, 2], [1, 3, 2], [2, 3, 3]]);
  result = minCut(4, diamond, 0, 3);
  assert.strictEqual(result.value, 4);
  assert.strictEqual(result.cutEdges.reduce((a, [, , c]) => a + c, 0), 4);
  assert.strictEqual(minCutBruteForce(4, diamond, 0, 3), 4);

  const stranded = matrixFromEdges(3, [[0, 1, 7]]);
  result = minCut(3, stranded, 0, 2);
  assert.strictEqual(result.value, 0);
  assert.deepStrictEqual([...result.inS].sort(), [0, 1]);
  assert.deepStrictEqual([...result.inT], [2]);
  assert.deepStrictEqual(result.cutEdges, []);

  for (let trial = 0; trial < 60; trial++) {
    const capacity = lcgCapacities(6);
    const got = minCut(6, capacity, 0, 5);
    assert.strictEqual(got.cutEdges.reduce((a, [, , c]) => a + c, 0), got.value, `cut==flow ${trial}`);
    assert.strictEqual(got.value, minCutBruteForce(6, capacity, 0, 5), `brute ${trial}`);
  }
  console.log("PASS");
}
