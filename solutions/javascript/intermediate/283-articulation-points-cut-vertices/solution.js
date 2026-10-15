/** Task 283 - Articulation points (cut vertices). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

/**
 * One DFS with disc/low values: a non-root u is a cut vertex when some child v has
 * low[v] >= disc[u]; the root is a cut vertex only when it has more than one child.
 * Cross-checked with a delete-each-vertex oracle that counts components.
 */
function articulationPoints(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }

  const disc = new Array(n).fill(-1);
  const low = new Array(n).fill(0);
  let timer = 0;
  const points = new Set();

  const dfs = (u, parent) => {
    disc[u] = low[u] = timer++;
    let children = 0;
    for (const v of adjacency[u]) {
      if (disc[v] === -1) {                    // tree edge
        children++;
        dfs(v, u);
        low[u] = Math.min(low[u], low[v]);
        if (parent !== -1 && low[v] >= disc[u]) points.add(u); // v's subtree cannot escape
      } else if (v !== parent) {               // back edge to an ancestor
        low[u] = Math.min(low[u], disc[v]);
      }
    }
    if (parent === -1 && children > 1) points.add(u); // root special case
  };

  for (let v = 0; v < n; v++) if (disc[v] === -1) dfs(v, -1);
  return [...points].sort((a, b) => a - b);
}

function countComponents(n, edges, skipped = -1) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    if (a === skipped || b === skipped) continue;
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const seen = new Array(n).fill(false);
  let total = 0;
  for (let s = 0; s < n; s++) {
    if (s === skipped || seen[s]) continue;
    total++;
    const stack = [s];
    seen[s] = true;
    while (stack.length) {
      const x = stack.pop();
      for (const y of adjacency[x]) {
        if (!seen[y]) { seen[y] = true; stack.push(y); }
      }
    }
  }
  return total;
}

/** Oracle: remove each vertex and count connected components. */
function bruteForceArticulationPoints(n, edges) {
  const base = countComponents(n, edges);
  const result = [];
  for (let v = 0; v < n; v++) if (countComponents(n, edges, v) > base) result.push(v);
  return result;
}

/** Each unordered pair {u, v} appears at most once; edge when seed % divisor == 0. */
function randomSimpleGraphs(n, count, divisor = 3, seed = 13579) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    const edges = [];
    for (let u = 0; u < n; u++) {
      for (let v = u + 1; v < n; v++) {
        seed = (seed * 75 + 74) % 65537;
        if (seed % divisor === 0) edges.push([u, v]);
      }
    }
    graphs.push(edges);
  }
  return graphs;
}

module.exports = { articulationPoints, countComponents, bruteForceArticulationPoints, randomSimpleGraphs };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(articulationPoints(4, [[0, 1], [1, 2], [2, 3]]), [1, 2]);          // path middles
  assert.deepStrictEqual(articulationPoints(4, [[0, 1], [1, 2], [2, 3], [3, 0]]), []);      // cycle
  assert.deepStrictEqual(articulationPoints(5, [[0, 1], [0, 2], [0, 3], [0, 4]]), [0]);     // star center
  assert.deepStrictEqual(articulationPoints(3, [[0, 1], [0, 2]]), [0]);                     // root with 2 children
  assert.deepStrictEqual(articulationPoints(6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]), []);
  assert.deepStrictEqual(articulationPoints(4, [[0, 1], [1, 2], [2, 0], [2, 3]]), [2]);     // triangle plus tail
  assert.deepStrictEqual(articulationPoints(2, [[0, 1]]), []);                              // single edge
  assert.deepStrictEqual(articulationPoints(3, []), []);                                    // isolated vertices
  assert.deepStrictEqual(articulationPoints(3, [[0, 1], [0, 1], [1, 2]]), [1]);             // parallel edges
  assert.deepStrictEqual(
    articulationPoints(7, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [2, 6]]),
    [2]
  );

  for (const divisor of [3, 5]) {
    let seenCut = false, seenNone = false;
    let trial = 0;
    for (const edges of randomSimpleGraphs(6, 120, divisor)) {
      const points = articulationPoints(6, edges);
      assert.deepStrictEqual(points, bruteForceArticulationPoints(6, edges), `divisor ${divisor} trial ${trial}`);
      if (points.length) seenCut = true; else seenNone = true;
      trial++;
    }
    assert.ok(seenCut && seenNone, `coverage ${divisor}`);
  }

  const pathEdges = Array.from({ length: 9 }, (_, i) => [i, i + 1]);
  assert.deepStrictEqual(articulationPoints(10, pathEdges), [1, 2, 3, 4, 5, 6, 7, 8]);

  console.log("PASS");
}
