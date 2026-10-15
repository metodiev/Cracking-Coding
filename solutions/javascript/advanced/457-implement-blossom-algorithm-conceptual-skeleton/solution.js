/** Task 457 - Implement blossom algorithm (conceptual/skeleton). (Tier: Advanced, CTCI Ch 4) */
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

/** Edmonds' algorithm with labelled BFS + blossom contraction; returns {mate, contractions}. */
function blossomMatching(n, adj) {
  const mate = new Array(n).fill(-1);
  const base = new Array(n).fill(0);
  const p = new Array(n).fill(-1);
  const used = new Array(n).fill(false);
  const blossom = new Array(n).fill(false);
  let contractions = 0;

  function lca(a, b) {
    const seen = new Array(n).fill(false);
    for (;;) {
      a = base[a];
      seen[a] = true;
      if (mate[a] === -1) break;
      a = p[mate[a]];
    }
    for (;;) {
      b = base[b];
      if (seen[b]) return b;
      b = p[mate[b]];
    }
  }

  function markPath(v, b, child) {
    while (base[v] !== b) {
      blossom[base[v]] = true;
      blossom[base[mate[v]]] = true;
      p[v] = child;
      child = mate[v];
      v = p[mate[v]];
    }
  }

  function findPath(root) {
    for (let i = 0; i < n; i++) {
      used[i] = false;
      p[i] = -1;
      base[i] = i;
    }
    used[root] = true;
    const queue = [root];
    for (let qh = 0; qh < queue.length; qh++) {
      const v = queue[qh];
      for (const to of adj[v]) {
        if (base[v] === base[to] || mate[v] === to) continue;
        if (to === root || (mate[to] !== -1 && p[mate[to]] !== -1)) {
          contractions++;
          const curbase = lca(v, to);
          blossom.fill(false);
          markPath(v, curbase, to);
          markPath(to, curbase, v);
          for (let i = 0; i < n; i++) {
            if (blossom[base[i]]) {
              base[i] = curbase;
              if (!used[i]) {
                used[i] = true;
                queue.push(i);
              }
            }
          }
        } else if (p[to] === -1) {
          p[to] = v;
          if (mate[to] === -1) return to;
          const nxt = mate[to];
          used[nxt] = true;
          queue.push(nxt);
        }
      }
    }
    return -1;
  }

  for (let i = 0; i < n; i++) {
    if (mate[i] === -1) {
      let v = findPath(i);
      while (v !== -1) {
        const pv = p[v];
        const ppv = mate[pv];
        mate[v] = pv;
        mate[pv] = v;
        v = ppv;
      }
    }
  }
  return { mate, contractions };
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

module.exports = { buildAdj, checkMatching, blossomMatching, bruteMaxMatching, lcgGeneralGraphs };

if (require.main === module) {
  const assert = require("node:assert");
  const c5adj = buildAdj(5, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]);
  const c5 = blossomMatching(5, c5adj);
  assert.strictEqual(checkMatching(assert, 5, c5adj, c5.mate), 2, "C5 optimum");
  assert.ok(c5.contractions >= 1, "C5 contractions");
  assert.strictEqual(bruteMaxMatching(5, c5adj), 2, "C5 brute");

  const handGraphs = [
    ["triangle", 3, [[0, 1], [1, 2], [0, 2]], 1],
    ["C5", 5, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], 2],
    ["K4", 4, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], 2],
    ["star", 5, [[0, 1], [0, 2], [0, 3], [0, 4]], 1],
    ["C6", 6, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], 3],
    ["C7", 7, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0]], 3],
    ["K3,3", 6, [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]], 3],
    ["two triangles", 6, [[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], 2],
    ["diamond", 4, [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], 2],
    ["P4", 4, [[0, 1], [1, 2], [2, 3]], 2],
    ["empty", 5, [], 0],
    ["single", 2, [[0, 1]], 1],
  ];
  for (const [label, n, edges, want] of handGraphs) {
    const adj = buildAdj(n, edges);
    assert.strictEqual(bruteMaxMatching(n, adj), want, `${label} brute`);
    const result = blossomMatching(n, adj);
    assert.strictEqual(checkMatching(assert, n, adj, result.mate), want, `${label} optimum`);
  }

  let checked = 0;
  let totalContractions = 0;
  for (const seed of [4571, 4572, 4573, 4574]) {
    for (const { n, edges } of lcgGeneralGraphs(6, seed)) {
      const adj = buildAdj(n, edges);
      const result = blossomMatching(n, adj);
      totalContractions += result.contractions;
      const size = checkMatching(assert, n, adj, result.mate);
      assert.strictEqual(size, bruteMaxMatching(n, adj), `blossom equals brute n=${n}`);
      checked++;
    }
  }
  assert.strictEqual(checked, 24);
  assert.ok(totalContractions >= 1, "contractions exercised");
  console.log("PASS");
}
