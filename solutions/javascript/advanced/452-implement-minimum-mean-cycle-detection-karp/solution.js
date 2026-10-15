/** Task 452 - Implement minimum mean cycle detection (Karp). (Tier: Advanced, CTCI Ch 4) */
"use strict";

const INF = 1_000_000_000;

function gcd(a, b) {
  a = Math.abs(a);
  while (b !== 0) {
    const t = a % b;
    a = b;
    b = t;
  }
  return a;
}

function reduceFraction(num, den) {
  const g = gcd(num, den);
  return [num / g, den / g];
}

function fracLess(a, b) {
  return a[0] * b[1] < b[0] * a[1];
}

function fracGreater(a, b) {
  return a[0] * b[1] > b[0] * a[1];
}

/** Returns the min mean cycle weight as a reduced [num, den], or null when acyclic. */
function karpMinMeanCycle(n, edges) {
  const d = [];
  for (let k = 0; k <= n; k++) d.push(new Array(n).fill(INF));
  for (let v = 0; v < n; v++) d[0][v] = 0;
  for (let k = 1; k <= n; k++) {
    for (const [u, v, w] of edges) {
      if (d[k - 1][u] < INF && d[k - 1][u] + w < d[k][v]) d[k][v] = d[k - 1][u] + w;
    }
  }
  let best = null;
  for (let v = 0; v < n; v++) {
    if (d[n][v] >= INF) continue;
    let local = null;
    for (let k = 0; k < n; k++) {
      if (d[k][v] >= INF) continue;
      const cand = reduceFraction(d[n][v] - d[k][v], n - k);
      if (local === null || fracGreater(cand, local)) local = cand;
    }
    if (best === null || fracLess(local, best)) best = local;
  }
  return best;
}

/** Minimum mean over all simple cycles by enumeration (each cycle from its min vertex). */
function bruteMinMeanCycle(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v, w] of edges) adj[u].push([v, w]);
  let best = null;
  const path = [];

  function walk(total) {
    const cur = path[path.length - 1];
    const start = path[0];
    for (const [v, w] of adj[cur]) {
      if (v === start) {
        const cand = reduceFraction(total + w, path.length);
        if (best === null || fracLess(cand, best)) best = cand;
      } else if (v > start && !path.includes(v)) {
        path.push(v);
        walk(total + w);
        path.pop();
      }
    }
  }

  for (let s = 0; s < n; s++) {
    path.push(s);
    walk(0);
    path.pop();
  }
  return best;
}

function lcgDigraphs(count, seed, minN, maxN) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    seed = (seed * 75 + 74) % 65537;
    const n = minN + (seed % (maxN - minN + 1));
    const edges = [];
    for (let node = 0; node < n - 1; node++) {
      seed = (seed * 75 + 74) % 65537;
      edges.push([node, node + 1, (seed % 15) - 5]);
    }
    seed = (seed * 75 + 74) % 65537;
    edges.push([n - 1, 0, (seed % 15) - 5]);
    for (let k = 0; k < 2 * n; k++) {
      seed = (seed * 75 + 74) % 65537;
      const u = seed % n;
      if (seed % 100 < 20) {
        seed = (seed * 75 + 74) % 65537;
        edges.push([u, u, (seed % 9) - 4]);
        continue;
      }
      seed = (seed * 75 + 74) % 65537;
      const v = seed % n;
      seed = (seed * 75 + 74) % 65537;
      if (u !== v) edges.push([u, v, (seed % 15) - 5]);
    }
    graphs.push({ n, edges });
  }
  return graphs;
}

module.exports = { karpMinMeanCycle, bruteMinMeanCycle, reduceFraction, lcgDigraphs };

if (require.main === module) {
  const assert = require("node:assert");
  const fourCycle = [[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 0, 1], [2, 0, 5]];
  assert.deepStrictEqual(karpMinMeanCycle(4, fourCycle), [1, 1]);
  assert.deepStrictEqual(bruteMinMeanCycle(4, fourCycle), [1, 1]);

  const negative = [[0, 1, -2], [1, 2, -1], [2, 0, -3]];
  assert.deepStrictEqual(karpMinMeanCycle(3, negative), [-2, 1]);
  assert.deepStrictEqual(bruteMinMeanCycle(3, negative), [-2, 1]);

  const selfLoop = [[0, 0, 7]];
  assert.deepStrictEqual(karpMinMeanCycle(1, selfLoop), [7, 1]);
  assert.deepStrictEqual(bruteMinMeanCycle(1, selfLoop), [7, 1]);

  const twoCycles = [[0, 1, 2], [1, 0, 2], [1, 2, -1], [2, 1, -1]];
  assert.deepStrictEqual(karpMinMeanCycle(3, twoCycles), [-1, 1]);
  assert.deepStrictEqual(bruteMinMeanCycle(3, twoCycles), [-1, 1]);

  const half = [[0, 1, 1], [1, 0, 0]];
  assert.deepStrictEqual(karpMinMeanCycle(2, half), [1, 2]);
  assert.deepStrictEqual(bruteMinMeanCycle(2, half), [1, 2]);

  const dag = [[0, 1, 1], [1, 2, 2]];
  assert.strictEqual(karpMinMeanCycle(3, dag), null);
  assert.strictEqual(bruteMinMeanCycle(3, dag), null);

  let checked = 0;
  for (const seed of [4521, 4522, 4523, 4524]) {
    for (const { n, edges } of lcgDigraphs(5, seed, 3, 6)) {
      const expected = bruteMinMeanCycle(n, edges);
      const got = karpMinMeanCycle(n, edges);
      assert.notStrictEqual(expected, null, "LCG has cycle");
      assert.deepStrictEqual(got, expected, `n=${n} edges=${JSON.stringify(edges)}`);
      checked++;
    }
  }
  assert.strictEqual(checked, 20);
  console.log("PASS");
}
