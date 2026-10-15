/** Task 424 - Implement small-to-large merging on trees for frequency queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function rootTree(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const parent = new Array(n).fill(-1);
  parent[0] = 0;
  const order = [0];
  for (let i = 0; i < order.length; i++) {
    const u = order[i];
    for (const v of adjacency[u]) {
      if (v !== parent[u]) {
        parent[v] = u;
        order.push(v);
      }
    }
  }
  const children = Array.from({ length: n }, () => []);
  for (let i = 1; i < order.length; i++) children[parent[order[i]]].push(order[i]);
  return { children, order };
}

function subtreeFrequencySummary(n, edges, colors) {
  const { children, order } = rootTree(n, edges);
  const counters = new Array(n).fill(null);
  const distinct = new Array(n).fill(0);
  const topFrequency = new Array(n).fill(0);
  for (let index = order.length - 1; index >= 0; index--) {
    const u = order[index];
    let biggest = -1;
    for (const v of children[u]) {
      if (biggest === -1 || counters[v].size > counters[biggest].size) biggest = v;
    }
    let table;
    let best;
    if (biggest === -1) {
      table = new Map([[colors[u], 1]]);
      best = 1;
    } else {
      table = counters[biggest];                 // reuse the largest counter
      best = topFrequency[biggest];
      const count = (table.get(colors[u]) || 0) + 1;
      table.set(colors[u], count);
      if (count > best) best = count;
      for (const v of children[u]) {
        if (v === biggest) continue;
        for (const [color, amount] of counters[v]) {
          const merged = (table.get(color) || 0) + amount;
          table.set(color, merged);
          if (merged > best) best = merged;
        }
      }
    }
    counters[u] = table;
    distinct[u] = table.size;
    topFrequency[u] = best;
  }
  return { distinct, topFrequency };
}

function naiveFrequencySummary(n, edges, colors) {
  const { children } = rootTree(n, edges);
  const distinct = new Array(n).fill(0);
  const topFrequency = new Array(n).fill(0);
  for (let u = 0; u < n; u++) {
    const counts = new Map();
    const stack = [u];
    while (stack.length) {
      const node = stack.pop();
      counts.set(colors[node], (counts.get(colors[node]) || 0) + 1);
      for (const v of children[node]) stack.push(v);
    }
    distinct[u] = counts.size;
    topFrequency[u] = Math.max(...counts.values());
  }
  return { distinct, topFrequency };
}

module.exports = { rootTree, subtreeFrequencySummary, naiveFrequencySummary };

if (require.main === module) {
  const assert = require("node:assert");

  // tree 0-1, 0-2, 1-3 with colours [0, 1, 0, 1]
  let summary = subtreeFrequencySummary(4, [[0, 1], [0, 2], [1, 3]], [0, 1, 0, 1]);
  assert.deepStrictEqual(summary.distinct, [2, 1, 1, 1]);
  assert.deepStrictEqual(summary.topFrequency, [2, 2, 1, 1]);
  let naive = naiveFrequencySummary(4, [[0, 1], [0, 2], [1, 3]], [0, 1, 0, 1]);
  assert.deepStrictEqual(summary.distinct, naive.distinct);
  assert.deepStrictEqual(summary.topFrequency, naive.topFrequency);

  // path 0-1-2-3-4 with colours [1, 2, 2, 3, 3]
  const pathEdges = [[0, 1], [1, 2], [2, 3], [3, 4]];
  summary = subtreeFrequencySummary(5, pathEdges, [1, 2, 2, 3, 3]);
  assert.deepStrictEqual(summary.distinct, [3, 2, 2, 1, 1]);
  assert.deepStrictEqual(summary.topFrequency, [2, 2, 2, 2, 1]);
  naive = naiveFrequencySummary(5, pathEdges, [1, 2, 2, 3, 3]);
  assert.deepStrictEqual(summary.distinct, naive.distinct);
  assert.deepStrictEqual(summary.topFrequency, naive.topFrequency);

  // single node
  summary = subtreeFrequencySummary(1, [], [7]);
  assert.deepStrictEqual(summary.distinct, [1]);
  assert.deepStrictEqual(summary.topFrequency, [1]);

  // star with five nodes, all the same colour
  const star = [[0, 1], [0, 2], [0, 3], [0, 4]];
  summary = subtreeFrequencySummary(5, star, [3, 3, 3, 3, 3]);
  assert.deepStrictEqual(summary.distinct, [1, 1, 1, 1, 1]);
  assert.deepStrictEqual(summary.topFrequency, [5, 1, 1, 1, 1]);

  for (const seed of [424, 2024, 77777]) {
    let local = seed;
    for (let trial = 0; trial < 20; trial++) {
      local = (local * 75 + 74) % 65537;
      const n = 2 + (local % 59);
      const edges = [];
      for (let node = 1; node < n; node++) {
        local = (local * 75 + 74) % 65537;
        edges.push([local % node, node]);
      }
      const palette = [];
      for (let node = 0; node < n; node++) {
        local = (local * 75 + 74) % 65537;
        palette.push(local % (2 + (seed % 5)));
      }
      const got = subtreeFrequencySummary(n, edges, palette);
      const want = naiveFrequencySummary(n, edges, palette);
      assert.deepStrictEqual(got.distinct, want.distinct, `seed=${seed} n=${n}`);
      assert.deepStrictEqual(got.topFrequency, want.topFrequency, `seed=${seed} n=${n}`);
    }
  }

  // a mostly-unique palette stresses distinct counts instead of frequencies
  let local = 909090;
  for (let trial = 0; trial < 10; trial++) {
    local = (local * 75 + 74) % 65537;
    const n = 2 + (local % 59);
    const edges = [];
    for (let node = 1; node < n; node++) {
      local = (local * 75 + 74) % 65537;
      edges.push([local % node, node]);
    }
    const palette = [];
    for (let node = 0; node < n; node++) {
      local = (local * 75 + 74) % 65537;
      palette.push(local % 1000);
    }
    const got = subtreeFrequencySummary(n, edges, palette);
    const want = naiveFrequencySummary(n, edges, palette);
    assert.deepStrictEqual(got.distinct, want.distinct, `wide palette n=${n}`);
    assert.deepStrictEqual(got.topFrequency, want.topFrequency, `wide palette n=${n}`);
  }

  console.log("PASS");
}
