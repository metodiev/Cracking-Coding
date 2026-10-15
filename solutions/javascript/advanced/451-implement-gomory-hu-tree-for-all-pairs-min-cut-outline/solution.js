/** Task 451 - Implement Gomory-Hu tree for all-pairs min-cut (outline). (Tier: Advanced, CTCI Ch 4) */
"use strict";

const INF = Number.MAX_SAFE_INTEGER;

function lcgGraphs(count, seed, minN, maxN) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    seed = (seed * 75 + 74) % 65537;
    const n = minN + (seed % (maxN - minN + 1));
    const edges = [];
    for (let node = 1; node < n; node++) {
      seed = (seed * 75 + 74) % 65537;
      const parent = seed % node;
      seed = (seed * 75 + 74) % 65537;
      edges.push([node, parent, 1 + (seed % 9)]);
    }
    for (let k = 0; k < n; k++) {
      seed = (seed * 75 + 74) % 65537;
      const u = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const v = seed % n;
      seed = (seed * 75 + 74) % 65537;
      if (u !== v) edges.push([u, v, 1 + (seed % 9)]);
    }
    graphs.push({ n, edges });
  }
  return graphs;
}

function capacityMatrix(n, edges) {
  const cap = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [u, v, w] of edges) {
    if (u !== v) {
      cap[u][v] += w;
      cap[v][u] += w;
    }
  }
  return cap;
}

/** Edmonds-Karp on the directed model of an undirected graph. Returns {value, side[]}. */
function maxFlowUndirected(n, edges, s, t) {
  const cap = capacityMatrix(n, edges);
  let flow = 0;
  for (;;) {
    const parent = new Array(n).fill(-1);
    parent[s] = s;
    const q = [s];
    for (let head = 0; head < q.length; head++) {
      const u = q[head];
      for (let v = 0; v < n; v++) {
        if (parent[v] === -1 && cap[u][v] > 0) {
          parent[v] = u;
          q.push(v);
        }
      }
    }
    const side = parent.map((p) => p !== -1);
    if (parent[t] === -1) return { value: flow, side };
    let aug = INF;
    for (let v = t; v !== s; v = parent[v]) aug = Math.min(aug, cap[parent[v]][v]);
    for (let v = t; v !== s; v = parent[v]) {
      cap[parent[v]][v] -= aug;
      cap[v][parent[v]] += aug;
    }
    flow += aug;
  }
}

/** Gusfield simplification: n-1 max-flow runs. Returns {parent, weight, runs}. */
function gomoryHu(n, edges) {
  const parent = new Array(n).fill(0);
  const weight = new Array(n).fill(0);
  let runs = 0;
  for (let s = 1; s < n; s++) {
    const t = parent[s];
    const result = maxFlowUndirected(n, edges, s, t);
    runs++;
    weight[s] = result.value;
    for (let i = s + 1; i < n; i++) {
      if (parent[i] === t && result.side[i]) parent[i] = s;
    }
  }
  return { parent, weight, runs };
}

/** Minimum edge weight on the tree path u..v. */
function treeMinCut(n, parent, weight, u, v) {
  const tree = Array.from({ length: n }, () => []);
  for (let i = 1; i < n; i++) {
    tree[i].push([parent[i], weight[i]]);
    tree[parent[i]].push([i, weight[i]]);
  }
  const seen = new Array(n).fill(false);
  seen[u] = true;
  const q = [[u, INF]];
  for (let head = 0; head < q.length; head++) {
    const [node, best] = q[head];
    if (node === v) return best;
    for (const [nxt, w] of tree[node]) {
      if (!seen[nxt]) {
        seen[nxt] = true;
        q.push([nxt, Math.min(best, w)]);
      }
    }
  }
  return null;
}

/** Exact min u-v cut: minimum crossing weight over all vertex partitions. */
function brutePairMinCut(n, edges, u, v) {
  let best = INF;
  for (let mask = 0; mask < (1 << n); mask++) {
    if (((mask >> u) & 1) === ((mask >> v) & 1)) continue;
    let cut = 0;
    for (const [a, b, w] of edges) {
      if (((mask >> a) & 1) !== ((mask >> b) & 1)) cut += w;
    }
    if (cut < best) best = cut;
  }
  return best;
}

module.exports = { lcgGraphs, maxFlowUndirected, gomoryHu, treeMinCut, brutePairMinCut };

if (require.main === module) {
  const assert = require("node:assert");
  const triangle = [[0, 1, 1], [1, 2, 1], [0, 2, 1]];
  let gh = gomoryHu(3, triangle);
  assert.strictEqual(gh.runs, 2);
  for (let u = 0; u < 3; u++) {
    for (let v = u + 1; v < 3; v++) {
      assert.strictEqual(treeMinCut(3, gh.parent, gh.weight, u, v), 2);
      assert.strictEqual(treeMinCut(3, gh.parent, gh.weight, u, v), treeMinCut(3, gh.parent, gh.weight, v, u));
      assert.strictEqual(brutePairMinCut(3, triangle, u, v), 2);
    }
  }

  const path = [[0, 1, 5], [1, 2, 3], [2, 3, 4]];
  const expected = [[0, 1, 5], [0, 2, 3], [0, 3, 3], [1, 2, 3], [1, 3, 3], [2, 3, 4]];
  gh = gomoryHu(4, path);
  assert.strictEqual(gh.runs, 3);
  for (const [u, v, w] of expected) {
    assert.strictEqual(treeMinCut(4, gh.parent, gh.weight, u, v), w);
    assert.strictEqual(brutePairMinCut(4, path, u, v), w);
  }

  const k4 = [];
  for (let u = 0; u < 4; u++) for (let v = u + 1; v < 4; v++) k4.push([u, v, 1]);
  gh = gomoryHu(4, k4);
  for (let u = 0; u < 4; u++) {
    for (let v = u + 1; v < 4; v++) {
      assert.strictEqual(treeMinCut(4, gh.parent, gh.weight, u, v), 3);
      assert.strictEqual(brutePairMinCut(4, k4, u, v), 3);
    }
  }

  const parallel = [[0, 1, 3], [1, 0, 4]];
  gh = gomoryHu(2, parallel);
  assert.strictEqual(treeMinCut(2, gh.parent, gh.weight, 0, 1), 7);
  assert.strictEqual(brutePairMinCut(2, parallel, 0, 1), 7);

  let pairChecks = 0;
  let graphCount = 0;
  for (const seed of [4511, 4512, 4513, 4514]) {
    for (const { n, edges } of lcgGraphs(4, seed, 4, 7)) {
      gh = gomoryHu(n, edges);
      assert.strictEqual(gh.runs, n - 1);
      for (let u = 0; u < n; u++) {
        for (let v = u + 1; v < n; v++) {
          const got = treeMinCut(n, gh.parent, gh.weight, u, v);
          const want = brutePairMinCut(n, edges, u, v);
          assert.strictEqual(got, want, `pair ${u},${v} n=${n}`);
          assert.strictEqual(treeMinCut(n, gh.parent, gh.weight, v, u), got);
          pairChecks++;
        }
      }
      const direct = maxFlowUndirected(n, edges, 0, n - 1);
      assert.strictEqual(direct.value, brutePairMinCut(n, edges, 0, n - 1));
      graphCount++;
    }
  }
  assert.strictEqual(graphCount, 16);
  assert.ok(pairChecks >= 16 * 6);
  console.log("PASS");
}
