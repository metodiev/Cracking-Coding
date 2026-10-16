/**
 * Task 536 - Implement constrained shortest path (resource constrained) DP. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * Every edge carries (cost >= 0, resource >= 1) and we want the cheapest s->t walk whose total
 * resource consumption stays within an integer budget B.  The DP is a layered graph over states
 * (node, resource): dist[v][r] is the cheapest way to reach v having burned exactly r resource,
 * and every edge strictly raises the layer by at least 1, so a single ascending sweep over
 * r = 0..B (all nodes inside each layer) is already a topological-order relaxation - no
 * Bellman-Ford loop is needed and min_{r <= B} dist[t][r] is exact.  A predecessor table
 * (v, r) -> (u, r') reconstructs the optimal walk, and the file asserts the reconstruction is a
 * genuine walk: every consecutive pair is a real edge, the resource total fits the budget, the
 * cost total equals the reported optimum, and the layer index equals the walk's resource sum.
 * Because costs are non-negative an optimal walk never repeats a vertex, so the layered DP is
 * cross-checked on small LCG graphs against exhaustive enumeration of all simple s->t paths that
 * fit the budget, and for an effectively unlimited budget it must collapse onto plain Dijkstra.
 * Also checked: cost is non-increasing in the budget, an unaffordable budget yields +Infinity
 * with a null walk, parallel edges are resolved per layer, cycles are harmless (resource only
 * grows), and invalid input (resource 0/ fractional, negative cost) is rejected.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function buildAdj(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v, cost, res] of edges) adj[u].push([v, cost, res]);
  for (const list of adj) list.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]));
  return adj;
}

function validate(edges) {
  for (const [u, v, cost, res] of edges) {
    if (!Number.isInteger(res) || res < 1) {
      throw new Error(`resource consumption must be a positive integer (edge ${u}->${v})`);
    }
    if (cost < 0) throw new Error(`cost must be non-negative (edge ${u}->${v})`);
  }
}

/**
 * Layered DP over (node, resource).  Exact in one ascending sweep because res >= 1 makes the
 * state graph a DAG whose topological order is simply increasing resource.
 */
function constrainedShortestPath(n, edges, source, target, budget) {
  validate(edges);
  if (!Number.isInteger(budget) || budget < 0) {
    throw new Error("budget must be a non-negative integer");
  }
  const adj = buildAdj(n, edges);
  const dist = Array.from({ length: n }, () => new Array(budget + 1).fill(Infinity));
  const parent = Array.from({ length: n }, () => new Array(budget + 1).fill(null));
  const stats = { relaxations: 0 };
  dist[source][0] = 0;
  for (let r = 0; r <= budget; r++) {
    for (let u = 0; u < n; u++) {
      const here = dist[u][r];
      if (here === Infinity) continue;
      for (const [v, cost, res] of adj[u]) {
        const nr = r + res;
        if (nr > budget) continue;
        stats.relaxations += 1;
        const candidate = here + cost;
        if (candidate < dist[v][nr]) {
          dist[v][nr] = candidate;
          parent[v][nr] = [u, r];
        }
      }
    }
  }
  let best = Infinity;
  let bestResource = -1;
  for (let r = 0; r <= budget; r++) {
    if (dist[target][r] < best) {
      best = dist[target][r];
      bestResource = r;
    }
  }
  let walk = null;
  if (best < Infinity) {
    walk = [];
    let cur = [target, bestResource];
    while (cur !== null) {
      walk.push(cur[0]);
      cur = parent[cur[0]][cur[1]];
    }
    walk.reverse();
  }
  return { cost: best, resource: bestResource < 0 ? null : bestResource, walk, dist, stats };
}

/** Exhaustive simple-path enumeration - the independent ground truth for small graphs. */
function bruteConstrained(n, edges, source, target, budget) {
  const adj = Array.from({ length: n }, () => []);
  edges.forEach(([u, v, cost, res], id) => adj[u].push([v, cost, res, id]));
  for (const list of adj) list.sort((a, b) => (a[0] - b[0]) || (a[3] - b[3]));
  let best = Infinity;
  let bestPath = null;
  const visited = new Array(n).fill(false);
  const path = [];
  const dfs = (u, cost, res) => {
    if (u === target) {
      if (cost < best) {
        best = cost;
        bestPath = path.slice();
      }
      return;
    }
    for (const [v, c, r] of adj[u]) {
      if (visited[v] || res + r > budget) continue;
      visited[v] = true;
      path.push(v);
      dfs(v, cost + c, res + r);
      path.pop();
      visited[v] = false;
    }
  };
  visited[source] = true;
  path.push(source);
  dfs(source, 0, 0);
  return { cost: best, path: bestPath };
}

/** Plain Dijkstra on cost alone - the budget-free cross-check. */
function dijkstra(n, edges, source) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v, cost] of edges) adj[u].push([v, cost]);
  const cost = new Array(n).fill(Infinity);
  const done = new Array(n).fill(false);
  cost[source] = 0;
  for (let it = 0; it < n; it++) {
    let u = -1;
    for (let i = 0; i < n; i++) {
      if (done[i]) continue;
      if (u === -1 || cost[i] < cost[u]) u = i;
    }
    if (u === -1 || cost[u] === Infinity) break;
    done[u] = true;
    for (const [v, c] of adj[u]) if (cost[u] + c < cost[v]) cost[v] = cost[u] + c;
  }
  return cost;
}

/** Cheapest single-edge interpretation of every hop, used to audit a reconstructed walk. */
function walkResource(edges, walk) {
  const map = new Map();
  for (const [u, v, cost, res] of edges) {
    const key = `${u}->${v}`;
    if (!map.has(key) || cost < map.get(key)[0]) map.set(key, [cost, res]);
  }
  let cost = 0;
  let res = 0;
  for (let i = 0; i + 1 < walk.length; i++) {
    const hit = map.get(`${walk[i]}->${walk[i + 1]}`);
    if (hit === undefined) throw new Error(`walk uses missing edge ${walk[i]}->${walk[i + 1]}`);
    cost += hit[0];
    res += hit[1];
  }
  return { cost, res };
}

function randomGraph(n, percent, seed) {
  const edges = [];
  let s = seed % 65537;
  for (let u = 0; u < n; u++) {
    for (let v = 0; v < n; v++) {
      if (u === v) continue;
      s = lcgNext(s);
      if (s % 100 < percent) {
        s = lcgNext(s);
        const cost = (s % 9) + 1;
        s = lcgNext(s);
        const res = (s % 3) + 1;
        edges.push([u, v, cost, res]);
      }
    }
  }
  return { edges, seed: s };
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-checked diamond: the cheap-but-hungry route needs 11 resource, the lean route 2.
  const toy = {
    n: 4,
    edges: [[0, 1, 5, 1], [1, 3, 5, 1], [0, 2, 2, 10], [2, 3, 2, 1]],
  };
  const starved = constrainedShortestPath(toy.n, toy.edges, 0, 3, 1);
  assert.strictEqual(starved.cost, Infinity, "no route fits inside 1 unit of resource");
  const small = constrainedShortestPath(toy.n, toy.edges, 0, 3, 2);
  assert.strictEqual(small.cost, 10, "0->1->3 costs 10 and burns exactly 2 resource");
  assert.strictEqual(small.resource, 2);
  assert.deepStrictEqual(small.walk, [0, 1, 3]);
  const justShort = constrainedShortestPath(toy.n, toy.edges, 0, 3, 10);
  assert.strictEqual(justShort.cost, 10, "the hungry route needs 11, so 10 is still not enough");
  const wide = constrainedShortestPath(toy.n, toy.edges, 0, 3, 11);
  assert.strictEqual(wide.cost, 4, "with room to spare the hungry route wins");
  assert.strictEqual(wide.resource, 11);
  assert.deepStrictEqual(wide.walk, [0, 2, 3]);
  const nothing = constrainedShortestPath(toy.n, toy.edges, 0, 3, 0);
  assert.strictEqual(nothing.cost, Infinity, "every edge consumes resource, so budget 0 fails");
  assert.strictEqual(nothing.walk, null);

  // (2) Reconstructed walks are real: edges exist, the resource fits, the cost matches.
  for (let seed = 1; seed <= 6; seed++) {
    const n = 6;
    const { edges } = randomGraph(n, 40, seed);
    for (let budget = 0; budget <= 9; budget++) {
      const res = constrainedShortestPath(n, edges, 0, n - 1, budget);
      const truth = bruteConstrained(n, edges, 0, n - 1, budget);
      assert.strictEqual(res.cost, truth.cost, `seed ${seed} B=${budget}: DP vs enumeration`);
      if (res.cost === Infinity) {
        assert.strictEqual(res.walk, null);
        continue;
      }
      const audit = walkResource(edges, res.walk);
      assert.strictEqual(audit.cost, res.cost, `seed ${seed} B=${budget}: walk cost`);
      assert.ok(audit.res <= budget, `seed ${seed} B=${budget}: walk fits the budget`);
      assert.strictEqual(res.walk[0], 0);
      assert.strictEqual(res.walk[res.walk.length - 1], n - 1);
      assert.strictEqual(audit.res, res.resource, "the reported layer is the walk's resource total");
    }
  }

  // (3) Monotonicity in the budget, and the Dijkstra collapse for an ample budget.
  for (let seed = 7; seed <= 10; seed++) {
    const n = 7;
    const { edges } = randomGraph(n, 35, seed);
    let previous = Infinity;
    for (let budget = 0; budget <= 15; budget++) {
      const value = constrainedShortestPath(n, edges, 0, n - 1, budget).cost;
      assert.ok(value <= previous + 1e-12, `seed ${seed}: cost must not rise with the budget`);
      previous = value;
    }
    const plain = dijkstra(n, edges, 0)[n - 1];
    assert.strictEqual(constrainedShortestPath(n, edges, 0, n - 1, 1000).cost, plain,
      `seed ${seed}: unlimited budget = plain shortest path`);
  }

  // (4) Parallel edges: the DP keeps the cheaper option independently per layer.
  const parallel = { n: 2, edges: [[0, 1, 4, 1], [0, 1, 1, 5], [0, 1, 3, 2]] };
  assert.strictEqual(constrainedShortestPath(2, parallel.edges, 0, 1, 1).cost, 4);
  assert.strictEqual(constrainedShortestPath(2, parallel.edges, 0, 1, 2).cost, 3);
  assert.strictEqual(constrainedShortestPath(2, parallel.edges, 0, 1, 5).cost, 1);
  assert.strictEqual(constrainedShortestPath(2, parallel.edges, 0, 1, 5).resource, 5);

  // (5) Boundary cases.
  const singleton = constrainedShortestPath(1, [], 0, 0, 0);
  assert.strictEqual(singleton.cost, 0, "empty walk to itself");
  assert.deepStrictEqual(singleton.walk, [0]);
  assert.strictEqual(constrainedShortestPath(2, [[0, 1, 7, 3]], 0, 1, 2).cost, Infinity);
  assert.strictEqual(constrainedShortestPath(2, [[0, 1, 7, 3]], 0, 1, 3).cost, 7);
  const twoHop = [[0, 1, 1, 1], [1, 2, 1, 1]];
  assert.strictEqual(constrainedShortestPath(3, twoHop, 0, 2, 2).cost, 2);
  assert.strictEqual(constrainedShortestPath(3, twoHop, 0, 2, 1).cost, Infinity);
  assert.strictEqual(constrainedShortestPath(3, twoHop, 2, 0, 5).cost, Infinity,
    "no route flows backwards");

  // (6) Cycles are harmless (resource strictly increases) and the sweep stays bounded.
  const cycle = { n: 3, edges: [[0, 1, 1, 1], [1, 0, 1, 1], [1, 2, 1, 1], [2, 1, 1, 1]] };
  const cycling = constrainedShortestPath(3, cycle.edges, 0, 2, 6);
  assert.strictEqual(cycling.cost, 2);
  assert.strictEqual(cycling.resource, 2);
  assert.deepStrictEqual(cycling.walk, [0, 1, 2], "non-negative costs: never loop for free");
  assert.ok(cycling.stats.relaxations <= 4 * 7);

  // (7) Invalid input is rejected rather than mis-computed.
  assert.throws(() => constrainedShortestPath(2, [[0, 1, 1, 0]], 0, 1, 5), /positive integer/);
  assert.throws(() => constrainedShortestPath(2, [[0, 1, 1, 1.5]], 0, 1, 5), /positive integer/);
  assert.throws(() => constrainedShortestPath(2, [[0, 1, -1, 1]], 0, 1, 5), /non-negative/);
  assert.throws(() => constrainedShortestPath(2, [[0, 1, 1, 1]], 0, 1, -1), /budget/);

  // (8) Determinism.
  const repeat = randomGraph(6, 45, 4242).edges;
  for (let budget = 0; budget <= 6; budget++) {
    assert.deepStrictEqual(constrainedShortestPath(6, repeat, 0, 5, budget),
      constrainedShortestPath(6, repeat, 0, 5, budget));
  }
}

module.exports = { lcgNext, buildAdj, validate, constrainedShortestPath, bruteConstrained, dijkstra,
  walkResource, randomGraph };

if (require.main === module) {
  main();
  console.log("PASS");
}
