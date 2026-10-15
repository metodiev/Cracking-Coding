/** Task 456 - Implement matchings in general graph (Edmonds blossom outline). (Tier: Advanced, CTCI Ch 4) */
"use strict";

function buildAdj(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) {
    adj[u].push(v);
    adj[v].push(u);
  }
  return adj;
}

/** Validates the matching and returns its size (asserts validity). */
function checkMatching(assert, n, adj, mate) {
  let size = 0;
  const neighbourSets = adj.map((a) => new Set(a));
  for (let v = 0; v < n; v++) {
    if (mate[v] !== -1) {
      assert.strictEqual(mate[mate[v]], v, "symmetric mates");
      assert.notStrictEqual(mate[v], v, "no self match");
      assert.ok(neighbourSets[v].has(mate[v]), "matched pair must be an edge");
      size++;
    }
  }
  assert.strictEqual(size % 2, 0, "even matched count");
  return size / 2;
}

function greedyMatching(n, adj) {
  const mate = new Array(n).fill(-1);
  for (let u = 0; u < n; u++) {
    if (mate[u] === -1) {
      for (const v of adj[u]) {
        if (v > u && mate[v] === -1) {
          mate[u] = v;
          mate[v] = u;
          break;
        }
      }
    }
  }
  return mate;
}

function findAugmentingPath(n, adj, mate, root) {
  const onPath = new Array(n).fill(false);
  onPath[root] = true;
  const path = [];

  function dfs(u) {
    for (const v of adj[u]) {
      if (mate[u] === v || onPath[v]) continue;
      if (mate[v] === -1) {
        path.push([u, v]);
        return true;
      }
      const w = mate[v];
      if (onPath[w]) continue;
      onPath[v] = true;
      onPath[w] = true;
      path.push([u, v]);
      path.push([v, w]);
      if (dfs(w)) return true;
      path.pop();
      path.pop();
      onPath[v] = false;
      onPath[w] = false;
    }
    return false;
  }

  return dfs(root) ? path : null;
}

function applyAugmenting(assert, mate, path) {
  for (let i = 0; i < path.length; i++) {
    if (i % 2 === 1) {
      const [a, b] = path[i];
      assert.ok(mate[a] === b && mate[b] === a, "path alternates");
      mate[a] = -1;
      mate[b] = -1;
    }
  }
  for (let i = 0; i < path.length; i++) {
    if (i % 2 === 0) {
      const [a, b] = path[i];
      mate[a] = b;
      mate[b] = a;
    }
  }
}

/** Greedy + augmenting paths; returns {mate, augmentations}. */
function maxMatchingOutline(assert, n, adj) {
  const mate = greedyMatching(n, adj);
  let augmentations = 0;
  let changed = true;
  while (changed) {
    changed = false;
    for (let u = 0; u < n; u++) {
      if (mate[u] === -1) {
        const path = findAugmentingPath(n, adj, mate, u);
        if (path !== null) {
          applyAugmenting(assert, mate, path);
          augmentations++;
          changed = true;
        }
      }
    }
  }
  return { mate, augmentations };
}

/** Exact maximum matching by exhaustive backtracking (first-free-vertex branching). */
function bruteMaxMatching(n, adj) {
  const mate = new Array(n).fill(-1);
  function rec(i) {
    while (i < n && mate[i] !== -1) i++;
    if (i === n) return 0;
    let best = rec(i + 1);
    for (const v of adj[i]) {
      if (v > i && mate[v] === -1) {
        mate[i] = v;
        mate[v] = i;
        best = Math.max(best, 1 + rec(i + 1));
        mate[i] = -1;
        mate[v] = -1;
      }
    }
    return best;
  }
  return rec(0);
}

function lcgGeneralGraphs(count, seed) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 2 + (seed % 7);
    const edges = [];
    for (let node = 1; node < n; node++) {
      seed = (seed * 75 + 74) % 65537;
      edges.push([node, seed % node]);
    }
    const hasEdge = (u, v) => edges.some(([a, b]) => (a === u && b === v) || (a === v && b === u));
    for (let k = 0; k < n + 1; k++) {
      seed = (seed * 75 + 74) % 65537;
      const u = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const v = seed % n;
      if (u !== v && !hasEdge(u, v)) edges.push([u, v]);
    }
    graphs.push({ n, edges });
  }
  return graphs;
}

module.exports = { buildAdj, checkMatching, greedyMatching, findAugmentingPath, maxMatchingOutline, bruteMaxMatching, lcgGeneralGraphs };

if (require.main === module) {
  const assert = require("node:assert");
  const handGraphs = [
    ["triangle", 3, [[0, 1], [1, 2], [0, 2]], 1],
    ["C5", 5, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], 2],
    ["K4", 4, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], 2],
    ["star", 5, [[0, 1], [0, 2], [0, 3], [0, 4]], 1],
    ["C6", 6, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], 3],
    ["C7", 7, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]], 3],
    ["two triangles", 6, [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], 2],
    ["K3,3", 6, [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]], 3],
    ["diamond", 4, [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], 2],
    ["P4", 4, [[0, 1], [1, 2], [2, 3]], 2],
    ["empty", 5, [], 0],
    ["single", 2, [[0, 1]], 1],
  ];
  for (const [label, n, edges, want] of handGraphs) {
    const adj = buildAdj(n, edges);
    assert.strictEqual(bruteMaxMatching(n, adj), want, `${label} brute`);
    const result = maxMatchingOutline(assert, n, adj);
    assert.strictEqual(checkMatching(assert, n, adj, result.mate), want, `${label} outline`);
  }

  let improvedGraphs = 0;
  let checked = 0;
  let totalAugmentations = 0;
  for (const seed of [4561, 4562, 4563, 4564]) {
    for (const { n, edges } of lcgGeneralGraphs(6, seed)) {
      const adj = buildAdj(n, edges);
      const greedySize = checkMatching(assert, n, adj, greedyMatching(n, adj));
      const result = maxMatchingOutline(assert, n, adj);
      const size = checkMatching(assert, n, adj, result.mate);
      assert.ok(size >= greedySize, "outline at least greedy");
      if (size > greedySize) improvedGraphs++;
      assert.strictEqual(size, bruteMaxMatching(n, adj), `outline equals brute n=${n}`);
      totalAugmentations += result.augmentations;
      checked++;
    }
  }
  assert.strictEqual(checked, 24);
  assert.ok(improvedGraphs >= 1, "augmenting improved something");
  assert.ok(totalAugmentations >= improvedGraphs, "augmentation count");
  console.log("PASS");
}
