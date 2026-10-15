/** Task 275 - Maximum bipartite matching (Hopcroft-Karp). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function hopcroftKarp(nl, adj, nr) {
  const pairU = new Array(nl).fill(-1);
  const pairV = new Array(nr).fill(-1);
  const dist = new Array(nl).fill(-1);

  function bfs() {
    let queue = [];
    for (let u = 0; u < nl; u++) {
      if (pairU[u] === -1) { dist[u] = 0; queue.push(u); }
      else dist[u] = -1;
    }
    let reachable = false;
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (const v of adj[u]) {
        const w = pairV[v];
        if (w === -1) reachable = true;
        else if (dist[w] === -1) { dist[w] = dist[u] + 1; queue.push(w); }
      }
    }
    return reachable;
  }

  function dfs(u) {
    for (const v of adj[u]) {
      const w = pairV[v];
      if (w === -1 || (dist[w] === dist[u] + 1 && dfs(w))) {
        pairU[u] = v;
        pairV[v] = u;
        return true;
      }
    }
    dist[u] = -1;
    return false;
  }

  let matching = 0;
  while (bfs()) {
    for (let u = 0; u < nl; u++) if (pairU[u] === -1 && dfs(u)) matching++;
  }
  const pairs = [];
  for (let u = 0; u < nl; u++) if (pairU[u] !== -1) pairs.push([u, pairU[u]]);
  return { size: matching, pairs };
}

function kuhn(nl, adj, nr) {
  const matchV = new Array(nr).fill(-1);
  function tryKuhn(u, visited) {
    for (const v of adj[u]) {
      if (!visited[v]) {
        visited[v] = true;
        if (matchV[v] === -1 || tryKuhn(matchV[v], visited)) {
          matchV[v] = u;
          return true;
        }
      }
    }
    return false;
  }
  let count = 0;
  for (let u = 0; u < nl; u++) if (tryKuhn(u, new Array(nr).fill(false))) count++;
  return count;
}

function isValidMatching(pairs, adj, nl, nr) {
  const lefts = new Set();
  const rights = new Set();
  for (const [u, v] of pairs) {
    if (!(u >= 0 && u < nl && v >= 0 && v < nr)) return false;
    if (lefts.has(u) || rights.has(v)) return false;
    if (!adj[u].includes(v)) return false;
    lefts.add(u);
    rights.add(v);
  }
  return true;
}

function randomGraphs(nl, nr, count, seed = 12345) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    const adj = Array.from({ length: nl }, () => []);
    for (let u = 0; u < nl; u++) {
      for (let v = 0; v < nr; v++) {
        seed = (seed * 75 + 74) % 65537;
        if (seed % 3 === 0) adj[u].push(v);
      }
    }
    graphs.push(adj);
  }
  return graphs;
}

module.exports = { hopcroftKarp, kuhn, isValidMatching, randomGraphs };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(hopcroftKarp(3, [[0, 1, 2], [0, 1, 2], [0, 1, 2]], 3).size, 3);
  assert.strictEqual(hopcroftKarp(3, [[], [], []], 3).size, 0);
  assert.strictEqual(hopcroftKarp(3, [[0, 1], [0], []], 2).size, 2);
  assert.strictEqual(hopcroftKarp(3, [[0], [0], []], 1).size, 1);
  assert.strictEqual(hopcroftKarp(3, [[0], [0, 1], [1]], 2).size, 2);
  assert.strictEqual(hopcroftKarp(3, [[0, 1], [1], [2]], 3).size, 3);
  assert.strictEqual(hopcroftKarp(2, [[0, 1], [0]], 2).size, 2);
  assert.strictEqual(hopcroftKarp(2, [[0, 1, 2, 3, 4], [0]], 5).size, 2);

  const k33 = [[0, 1, 2], [0, 1, 2], [0, 1, 2]];
  const result = hopcroftKarp(3, k33, 3);
  assert.ok(isValidMatching(result.pairs, k33, 3, 3));
  assert.strictEqual(result.pairs.length, result.size);

  let trial = 0;
  for (const adj of randomGraphs(6, 6, 200)) {
    const { size, pairs } = hopcroftKarp(6, adj, 6);
    assert.ok(isValidMatching(pairs, adj, 6, 6), `valid ${trial}`);
    assert.strictEqual(pairs.length, size);
    assert.strictEqual(size, kuhn(6, adj, 6), `agreement ${trial}`);
    assert.ok(size <= 6);
    trial++;
  }
  console.log("PASS");
}
