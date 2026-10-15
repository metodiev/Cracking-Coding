/** Task 273 - Find an Eulerian trail in a directed graph (Hierholzer). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function findEulerianTrail(n, edges, start = null) {
  if (edges.length === 0) return [];
  const outDegree = new Array(n).fill(0);
  const inDegree = new Array(n).fill(0);
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    outDegree[a]++;
    inDegree[b]++;
  }
  for (const list of adjacency) list.sort((x, y) => x - y);

  const starts = [];
  const ends = [];
  for (let v = 0; v < n; v++) {
    if (outDegree[v] - inDegree[v] === 1) starts.push(v);
    if (inDegree[v] - outDegree[v] === 1) ends.push(v);
  }
  if (starts.length > 1 || ends.length > 1) return null;
  for (let v = 0; v < n; v++) {
    if (outDegree[v] !== inDegree[v] && !starts.includes(v) && !ends.includes(v)) return null;
  }

  let from;
  if (start === null) {
    from = starts.length ? starts[0] : outDegree.findIndex((d) => d > 0);
  } else {
    from = start;
    if (starts.length && from !== starts[0]) return null;
    const diff = outDegree[from] - inDegree[from];
    if (diff !== 0 && diff !== 1) return null;
  }

  const active = [];
  for (let v = 0; v < n; v++) if (outDegree[v] > 0 || inDegree[v] > 0) active.push(v);
  if (active.length) {
    const undirected = Array.from({ length: n }, () => []);
    for (const [a, b] of edges) {
      undirected[a].push(b);
      undirected[b].push(a);
    }
    const seen = new Set([from]);
    const stack = [from];
    while (stack.length) {
      const u = stack.pop();
      for (const v of undirected[u]) if (!seen.has(v)) { seen.add(v); stack.push(v); }
    }
    if (active.some((v) => !seen.has(v))) return null;
  }

  const pointer = new Array(n).fill(0);
  const trail = [];
  const walk = [from];
  while (walk.length) {
    const u = walk[walk.length - 1];
    if (pointer[u] < adjacency[u].length) walk.push(adjacency[u][pointer[u]++]);
    else trail.push(walk.pop());
  }
  trail.reverse();
  return trail.length === edges.length + 1 ? trail : null;
}

function usesAllEdges(trail, edges) {
  if (trail === null) return false;
  const count = (pairs) => {
    const map = new Map();
    for (const [a, b] of pairs) {
      const k = `${a}->${b}`;
      map.set(k, (map.get(k) || 0) + 1);
    }
    return map;
  };
  const used = count(trail.slice(1).map((b, i) => [trail[i], b]));
  const wanted = count(edges);
  if (used.size !== wanted.size) return false;
  for (const [k, v] of wanted) if (used.get(k) !== v) return false;
  return true;
}

module.exports = { findEulerianTrail, usesAllEdges };

if (require.main === module) {
  const assert = require("node:assert");
  const cycle = [[0, 1], [1, 2], [2, 0]];
  assert.deepStrictEqual(findEulerianTrail(3, cycle), [0, 1, 2, 0]);
  assert.ok(usesAllEdges(findEulerianTrail(3, cycle), cycle));
  assert.deepStrictEqual(findEulerianTrail(3, cycle, 1), [1, 2, 0, 1]);

  assert.deepStrictEqual(findEulerianTrail(3, [[0, 1], [1, 2]]), [0, 1, 2]);
  const mixed = [[0, 1], [1, 2], [2, 0], [0, 3]];
  const trail = findEulerianTrail(4, mixed);
  assert.deepStrictEqual(trail, [0, 1, 2, 0, 3]);
  assert.ok(usesAllEdges(trail, mixed));

  assert.strictEqual(findEulerianTrail(4, [[0, 1], [0, 2], [2, 0], [0, 3]]), null);
  assert.strictEqual(findEulerianTrail(4, [[0, 1], [2, 3]]), null);
  assert.deepStrictEqual(findEulerianTrail(3, []), []);
  assert.deepStrictEqual(findEulerianTrail(3, cycle, 2), [2, 0, 1, 2]);
  assert.deepStrictEqual(findEulerianTrail(3, [[1, 2], [2, 0]], 1), [1, 2, 0]);
  console.log("PASS");
}
