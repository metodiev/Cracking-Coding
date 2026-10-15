/** Task 450 - Implement stoer-wagner minimum cut algorithm. (Tier: Advanced, CTCI Ch 4) */
"use strict";

function stoerWagner(n, edges) {
  const weight = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [u, v, w] of edges) {
    if (u === v) continue;
    weight[u][v] += w;
    weight[v][u] += w;
  }
  let vertices = Array.from({ length: n }, (_, i) => i);
  const groups = Array.from({ length: n }, (_, i) => [i]);
  let best = null;
  let bestSide = null;
  while (vertices.length > 1) {
    const remaining = new Set(vertices);
    const connection = new Map(vertices.map((v) => [v, 0]));
    const order = [];
    while (remaining.size > 0) {
      let pick = null;
      for (const v of remaining) {
        if (pick === null || connection.get(v) > connection.get(pick)) pick = v;
      }
      remaining.delete(pick);
      order.push(pick);
      for (const other of remaining) connection.set(other, connection.get(other) + weight[pick][other]);
    }
    const t = order[order.length - 1];
    const s = order[order.length - 2];
    const cut = connection.get(t);
    if (best === null || cut < best) {
      best = cut;
      bestSide = groups[t].slice();
    }
    for (const v of vertices) {
      if (v !== s && v !== t) {
        weight[s][v] += weight[t][v];
        weight[v][s] = weight[s][v];
      }
    }
    for (const v of vertices) {
      weight[t][v] = 0;
      weight[v][t] = 0;
    }
    vertices = vertices.filter((v) => v !== t);
    groups[s] = groups[s].concat(groups[t]);
  }
  bestSide.sort((a, b) => a - b);
  return [best, bestSide];
}

function bruteMinCut(n, edges) {
  let best = null;
  for (let mask = 1; mask < 1 << (n - 1); mask++) {
    let cut = 0;
    for (const [u, v, w] of edges) {
      const bu = u > 0 ? (mask >> (u - 1)) & 1 : 0;
      const bv = v > 0 ? (mask >> (v - 1)) & 1 : 0;
      if (bu !== bv) cut += w;
    }
    if (best === null || cut < best) best = cut;
  }
  return best;
}

function crossingWeight(edges, side) {
  const onSide = new Set(side);
  let total = 0;
  for (const [u, v, w] of edges) {
    if (onSide.has(u) !== onSide.has(v)) total += w;
  }
  return total;
}

function lcgGraphs(count, seed, maxN, connected) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    const n = 2 + (seed % (maxN - 1));
    const edges = [];
    const start = connected ? 1 : 0;
    for (let node = start; node < n; node++) {
      seed = (seed * 75 + 74) % 65537;
      const parent = node > 0 ? seed % node : 0;
      seed = (seed * 75 + 74) % 65537;
      edges.push([node, parent, 1 + (seed % 20)]);
    }
    for (let extra = 0; extra < n + 2; extra++) {
      seed = (seed * 75 + 74) % 65537;
      const u = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const v = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const w = 1 + (seed % 20);
      if (u !== v) edges.push([u, v, w]);
    }
    graphs.push([n, edges]);
  }
  return graphs;
}

module.exports = { stoerWagner, bruteMinCut, crossingWeight, lcgGraphs };

if (require.main === module) {
  const assert = require("node:assert");

  const triangle = [[0, 1, 1], [1, 2, 1], [0, 2, 1]];
  const k4 = [];
  for (let u = 0; u < 4; u++) {
    for (let v = u + 1; v < 4; v++) k4.push([u, v, 1]);
  }
  assert.strictEqual(bruteMinCut(3, triangle), 2);
  assert.strictEqual(bruteMinCut(4, k4), 3);
  let [value, side] = stoerWagner(3, triangle);
  assert.strictEqual(value, 2);
  assert.strictEqual(crossingWeight(triangle, side), 2);
  [value, side] = stoerWagner(4, k4);
  assert.strictEqual(value, 3);
  assert.strictEqual(crossingWeight(k4, side), 3);

  const cycle4 = [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 0, 1]];
  assert.strictEqual(stoerWagner(4, cycle4)[0], 2);
  assert.strictEqual(bruteMinCut(4, cycle4), 2);

  const disjoint = [[0, 1, 5], [1, 2, 5], [0, 2, 5], [3, 4, 7], [4, 5, 7], [3, 5, 7]];
  assert.strictEqual(bruteMinCut(6, disjoint), 0);
  [value, side] = stoerWagner(6, disjoint);
  assert.strictEqual(value, 0);
  assert.strictEqual(crossingWeight(disjoint, side), 0);

  const parallel = [[0, 1, 3], [1, 0, 4], [0, 1, 5]];
  assert.strictEqual(stoerWagner(2, parallel)[0], 12);
  const weighted = [[0, 1, 5], [1, 2, 5], [0, 2, 5]];
  assert.strictEqual(stoerWagner(3, weighted)[0], 10);

  let checked = 0;
  for (const connected of [true, false]) {
    for (const [n, edges] of lcgGraphs(8, connected ? 450 : 451, 8, connected)) {
      const expected = bruteMinCut(n, edges);
      const result = stoerWagner(n, edges);
      assert.strictEqual(crossingWeight(edges, result[1]), result[0], "witness");
      assert.strictEqual(result[0], expected, `n=${n}`);
      checked++;
    }
  }
  for (const extra of [452, 453, 454, 455, 456, 457]) {
    for (const [n, edges] of lcgGraphs(3, extra, 7, true)) {
      assert.strictEqual(stoerWagner(n, edges)[0], bruteMinCut(n, edges));
      checked++;
    }
  }
  assert.strictEqual(checked, 34);
  console.log("PASS");
}
