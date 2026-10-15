/** Task 430 - Implement Mo's algorithm on trees (Euler tour + Mo). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function eulerTour(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const children = Array.from({ length: n }, () => []);
  const tin = new Array(n).fill(0);
  const tout = new Array(n).fill(0);
  const euler = new Array(n).fill(0);         // euler[tin[u]] = u
  let timer = 0;

  function dfs(u, parent) {
    tin[u] = timer;
    euler[timer] = u;
    timer++;
    for (const v of adjacency[u]) {
      if (v !== parent) {
        children[u].push(v);
        dfs(v, u);
      }
    }
    tout[u] = timer - 1;
  }

  dfs(0, -1);
  return { tin, tout, euler, children };
}

function subtreeDistinctCountsMo(n, edges, colors, queries) {
  const { tin, tout, euler } = eulerTour(n, edges);
  const block = Math.max(1, Math.floor(Math.sqrt(n)));

  const order = Array.from({ length: queries.length }, (_, i) => i);
  order.sort((a, b) => {
    const slotA = Math.floor(tin[queries[a]] / block);
    const slotB = Math.floor(tin[queries[b]] / block);
    if (slotA !== slotB) return slotA - slotB;
    const rightA = slotA % 2 === 0 ? tout[queries[a]] : -tout[queries[a]];
    const rightB = slotB % 2 === 0 ? tout[queries[b]] : -tout[queries[b]];
    return rightA - rightB;
  });

  const counts = new Map();
  let distinct = 0;
  let currentLeft = 0;
  let currentRight = -1;
  const answers = new Array(queries.length).fill(0);

  function add(position) {
    const color = colors[euler[position]];
    const amount = (counts.get(color) || 0) + 1;
    counts.set(color, amount);
    if (amount === 1) distinct++;
  }

  function remove(position) {
    const color = colors[euler[position]];
    const amount = counts.get(color) - 1;
    counts.set(color, amount);
    if (amount === 0) distinct--;
  }

  for (const index of order) {
    const left = tin[queries[index]];
    const right = tout[queries[index]];
    while (currentLeft > left) {
      currentLeft--;
      add(currentLeft);
    }
    while (currentRight < right) {
      currentRight++;
      add(currentRight);
    }
    while (currentLeft < left) {
      remove(currentLeft);
      currentLeft++;
    }
    while (currentRight > right) {
      remove(currentRight);
      currentRight--;
    }
    answers[index] = distinct;
  }
  return answers;
}

function naiveSubtreeDistinct(n, edges, colors, queries) {
  const { children } = eulerTour(n, edges);
  return queries.map((start) => {
    const seen = new Set();
    const stack = [start];
    while (stack.length) {
      const node = stack.pop();
      seen.add(colors[node]);
      for (const v of children[node]) stack.push(v);
    }
    return seen.size;
  });
}

module.exports = { eulerTour, subtreeDistinctCountsMo, naiveSubtreeDistinct };

if (require.main === module) {
  const assert = require("node:assert");

  const edges = [[0, 1], [0, 2], [1, 3]];
  const colors = [0, 1, 0, 1];
  assert.deepStrictEqual(subtreeDistinctCountsMo(4, edges, colors, [0, 1, 2, 3]), [2, 1, 1, 1]);
  assert.deepStrictEqual(subtreeDistinctCountsMo(4, edges, colors, [3, 2, 1, 0]), [1, 1, 1, 2]);
  assert.deepStrictEqual(subtreeDistinctCountsMo(4, edges, colors, [2, 0, 2, 0]), [1, 2, 1, 2]);
  assert.deepStrictEqual(subtreeDistinctCountsMo(4, edges, [5, 5, 5, 5], [0, 1, 2, 3]), [1, 1, 1, 1]);
  assert.deepStrictEqual(subtreeDistinctCountsMo(1, [], [9], [0]), [1]);

  for (const seed of [430, 24680, 13579]) {
    let local = seed;
    for (let trial = 0; trial < 15; trial++) {
      local = (local * 75 + 74) % 65537;
      const n = 2 + (local % 49);
      const treeEdges = [];
      for (let node = 1; node < n; node++) {
        local = (local * 75 + 74) % 65537;
        treeEdges.push([local % node, node]);
      }
      const palette = [];
      for (let node = 0; node < n; node++) {
        local = (local * 75 + 74) % 65537;
        palette.push(local % (2 + (seed % 6)));
      }
      const queries = [];
      for (let node = 0; node < n; node++) queries.push(node);
      for (let i = 0; i < 12; i++) {
        local = (local * 75 + 74) % 65537;
        queries.push(local % n);
      }
      const got = subtreeDistinctCountsMo(n, treeEdges, palette, queries);
      const want = naiveSubtreeDistinct(n, treeEdges, palette, queries);
      assert.deepStrictEqual(got, want, `seed=${seed} n=${n}`);
    }
  }

  // a mostly-unique palette makes nearly every subtree fully distinct
  let local = 123123;
  for (let trial = 0; trial < 10; trial++) {
    local = (local * 75 + 74) % 65537;
    const n = 2 + (local % 49);
    const treeEdges = [];
    for (let node = 1; node < n; node++) {
      local = (local * 75 + 74) % 65537;
      treeEdges.push([local % node, node]);
    }
    const palette = [];
    for (let node = 0; node < n; node++) {
      local = (local * 75 + 74) % 65537;
      palette.push(local % 1000);
    }
    const queries = Array.from({ length: n }, (_, i) => i);
    const got = subtreeDistinctCountsMo(n, treeEdges, palette, queries);
    const want = naiveSubtreeDistinct(n, treeEdges, palette, queries);
    assert.deepStrictEqual(got, want, `wide palette n=${n}`);
  }

  console.log("PASS");
}
