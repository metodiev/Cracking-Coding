/** Task 264 - Find longest path in DAG (DP on topo order). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * Longest path in an UNWEIGHTED DAG measured in EDGES (documented; no weighted
 * variant). Kahn's topological order feeds the DP: when u is popped, dist[u] is final
 * because all its predecessors are done; every edge u->v relaxes
 * dist[v] = max(dist[v], dist[u] + 1). Isolated/disconnected vertices stay at 0.
 * A cyclic input throws (documented): Kahn's queue drains before every vertex is
 * processed, and a longest path in a cyclic graph is ill-defined.
 * O(V + E) time and space.
 */
function longestPath(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  const indeg = new Array(n).fill(0);
  for (const [u, v] of edges) {
    adj[u].push(v);
    indeg[v]++;
  }
  const q = [];
  for (let i = 0; i < n; i++) if (indeg[i] === 0) q.push(i);
  const dist = new Array(n).fill(0);
  let processed = 0;
  for (let head = 0; head < q.length; head++) {
    const u = q[head];
    processed++;
    for (const v of adj[u]) {
      if (dist[u] + 1 > dist[v]) dist[v] = dist[u] + 1;
      if (--indeg[v] === 0) q.push(v);
    }
  }
  if (processed !== n) throw new Error("graph contains a cycle");
  return n === 0 ? 0 : Math.max(...dist);
}

/** Memoized DFS reference (valid on acyclic inputs); used only for cross-checks. */
function longestPathDfs(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) adj[u].push(v);
  const memo = new Array(n).fill(-1);
  const depth = (u) => {
    if (memo[u] >= 0) return memo[u];
    let best = 0;
    for (const v of adj[u]) best = Math.max(best, 1 + depth(v));
    memo[u] = best;
    return best;
  };
  let best = 0;
  for (let i = 0; i < n; i++) best = Math.max(best, depth(i));
  return best;
}

module.exports = { longestPath, longestPathDfs };

if (require.main === module) {
  const assert = require("node:assert");
  const dag = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [2, 5]];
  assert.strictEqual(longestPath(6, dag), 3);                  // hand-verified: 0 -> 2 -> 3 -> 4
  assert.strictEqual(longestPath(8, dag), 3);                  // vertices 6, 7 disconnected
  assert.strictEqual(longestPath(5, [[0, 1], [1, 2], [2, 3], [3, 4]]), 4);   // linear chain
  assert.strictEqual(longestPath(1, []), 0);
  assert.strictEqual(longestPath(0, []), 0);
  assert.strictEqual(longestPath(5, [[0, 1], [1, 2], [3, 4]]), 2);           // disconnected chains
  assert.strictEqual(longestPath(4, [[0, 1], [0, 2], [1, 3], [2, 3]]), 2);   // diamond
  assert.strictEqual(longestPath(3, [[0, 2], [2, 1]]), 2);                   // topo order != index order
  assert.strictEqual(longestPath(4, [[3, 2], [2, 1], [1, 0]]), 3);           // reversed chain

  assert.throws(() => longestPath(3, [[0, 1], [1, 2], [2, 0]]), /cycle/);
  assert.throws(() => longestPath(4, [[0, 1], [1, 2], [2, 0], [2, 3]]), /cycle/);

  // deterministic LCG so the cross-check is reproducible
  let seed = 264;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  for (let test = 0; test < 300; test++) {      // cross-check vs the memoized-DFS oracle
    const n = randInt(0, 12);
    const edges = [];
    for (let u = 0; u < n; u++)
      for (let v = u + 1; v < n; v++)
        if (rand() < 0.25) edges.push([u, v]);
    assert.strictEqual(longestPath(n, edges), longestPathDfs(n, edges), `random agreement #${test}`);
  }
  console.log("PASS");
}
