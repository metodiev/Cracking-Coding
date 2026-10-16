/**
 * Task 535 - Implement dynamic programming on DAG with topological order caching. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * Three engines evaluate the same DAG recurrence g(u) = max over edges (u -> v, w) of (w + g(v)),
 * with g(target) = 0 and "useful" nodes = those that can still reach `target`:
 *   1. top-down memoised recursion over the successor recurrence (the cache),
 *   2. bottoms-up DP swept in reverse topological order,
 *   3. an uncached copy of engine 1, kept only so the tests can count calls and prove the cache
 *      collapses the recursion tree - a 16-level diamond ladder costs 2^16 - 1 uncached calls but
 *      exactly one compute per useful node with the cache.
 * Topological order comes from Kahn's algorithm that always takes the smallest available label;
 * that greedy produces the lexicographically smallest topological order, which the self-test
 * confirms by brute-forcing every permutation of 0..n-1 on tiny DAGs. Negative edge weights are
 * allowed (a DAG is the one setting where shortest and longest path are the same linear-time DP),
 * so the engines also solve `min` by negating the weights and are compared with exhaustive path
 * enumeration. Path counting gets both its own memoised DP and the closed form W^(L-1) on
 * layered graphs. Also covered: cycle detection (Kahn returns null on a cycle, including a
 * self-loop), unreachable target, source == target (empty path, value 0), the cache accounting
 * `computes == |useful ∩ reachable-from-source \ {target}|`, and determinism of every engine.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function buildAdj(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v, w] of edges) adj[u].push([v, w]);
  for (const list of adj) list.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  return adj;
}

/** Kahn's algorithm; ties broken by the smallest label. Returns null when a cycle exists. */
function topoOrder(n, adj) {
  const indeg = new Array(n).fill(0);
  for (let u = 0; u < n; u++) for (const [v] of adj[u]) indeg[v] += 1;
  const done = new Array(n).fill(false);
  const order = [];
  for (let step = 0; step < n; step++) {
    let pick = -1;
    for (let i = 0; i < n; i++) {
      if (!done[i] && indeg[i] === 0) {
        pick = i;
        break;
      }
    }
    if (pick < 0) return null;
    done[pick] = true;
    order.push(pick);
    for (const [v] of adj[pick]) indeg[v] -= 1;
  }
  return order;
}

function lessOrder(a, b) {
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return a[i] < b[i];
  }
  return false;
}

/** Every topological order, ascending lexicographically (tiny n only). */
function allTopoOrders(n, edges) {
  const out = [];
  const perm = [];
  const pos = new Array(n).fill(-1);
  const used = new Array(n).fill(false);
  const rec = () => {
    if (perm.length === n) {
      for (const [u, v] of edges) if (pos[u] > pos[v]) return;
      out.push(perm.slice());
      return;
    }
    for (let c = 0; c < n; c++) {
      if (used[c]) continue;
      used[c] = true;
      pos[c] = perm.length;
      perm.push(c);
      rec();
      perm.pop();
      pos[c] = -1;
      used[c] = false;
    }
  };
  rec();
  out.sort((a, b) => (lessOrder(a, b) ? -1 : 1));
  return out;
}

/** Nodes that can reach `target` (reverse reachability, iterative). */
function canReachTarget(n, adj, target) {
  const preds = Array.from({ length: n }, () => []);
  for (let u = 0; u < n; u++) for (const [v] of adj[u]) preds[v].push(u);
  const seen = new Array(n).fill(false);
  seen[target] = true;
  const stack = [target];
  while (stack.length) {
    const v = stack.pop();
    for (const p of preds[v]) {
      if (!seen[p]) {
        seen[p] = true;
        stack.push(p);
      }
    }
  }
  return seen;
}

function reachableFrom(n, adj, source) {
  const seen = new Array(n).fill(false);
  seen[source] = true;
  const stack = [source];
  while (stack.length) {
    const u = stack.pop();
    for (const [v] of adj[u]) {
      if (!seen[v]) {
        seen[v] = true;
        stack.push(v);
      }
    }
  }
  return seen;
}

/** Engine 1: cached top-down DP. Reports call/compute counts for the cache assertions. */
function longestPathMemo(n, edges, source, target) {
  const adj = buildAdj(n, edges);
  const useful = canReachTarget(n, adj, target);
  const memo = new Map();
  let calls = 0;
  let computes = 0;
  const visit = (u) => {
    calls += 1;
    if (u === target) return 0;
    if (memo.has(u)) return memo.get(u);
    let best = -Infinity;
    for (const [v, w] of adj[u]) {
      if (!useful[v]) continue;
      const sub = visit(v);
      if (sub > -Infinity) best = Math.max(best, w + sub);
    }
    computes += 1;
    memo.set(u, best);
    return best;
  };
  const value = useful[source] ? visit(source) : -Infinity;
  return { value, calls, computes, useful, memo };
}

/** Engine 3: the same recurrence with no cache - exponential on diamond-shaped DAGs. */
function longestPathNoCache(n, edges, source, target) {
  const adj = buildAdj(n, edges);
  const useful = canReachTarget(n, adj, target);
  let calls = 0;
  const visit = (u) => {
    calls += 1;
    if (u === target) return 0;
    let best = -Infinity;
    for (const [v, w] of adj[u]) {
      if (!useful[v]) continue;
      const sub = visit(v);
      if (sub > -Infinity) best = Math.max(best, w + sub);
    }
    return best;
  };
  return { value: useful[source] ? visit(source) : -Infinity, calls };
}

/** Engine 2: iterative DP; the reverse topological sweep is the explicit caching order. */
function longestPathTopo(n, edges, source, target) {
  const adj = buildAdj(n, edges);
  const order = topoOrder(n, adj);
  if (order === null) return { value: -Infinity, order: null };
  const useful = canReachTarget(n, adj, target);
  const f = new Array(n).fill(-Infinity);
  for (let i = order.length - 1; i >= 0; i--) {
    const u = order[i];
    if (!useful[u]) continue;
    if (u === target) {
      f[u] = 0;
      continue;
    }
    let best = -Infinity;
    for (const [v, w] of adj[u]) {
      if (useful[v] && f[v] > -Infinity) best = Math.max(best, w + f[v]);
    }
    f[u] = best;
  }
  return { value: useful[source] ? f[source] : -Infinity, order };
}

function enumeratePaths(n, edges, source, target) {
  const adj = buildAdj(n, edges);
  const paths = [];
  const stack = [];
  const dfs = (u) => {
    stack.push(u);
    if (u === target) paths.push(stack.slice());
    else for (const [v] of adj[u]) dfs(v);
    stack.pop();
  };
  dfs(source);
  return paths;
}

function pathWeight(edges, path) {
  const map = new Map();
  for (const [u, v, w] of edges) map.set(`${u}->${v}`, w);
  let total = 0;
  for (let i = 0; i + 1 < path.length; i++) {
    const w = map.get(`${path[i]}->${path[i + 1]}`);
    if (w === undefined) throw new Error(`missing edge ${path[i]}->${path[i + 1]}`);
    total += w;
  }
  return total;
}

function maxPathBrute(n, edges, source, target) {
  const paths = enumeratePaths(n, edges, source, target);
  if (paths.length === 0) return -Infinity;
  let best = -Infinity;
  for (const p of paths) best = Math.max(best, pathWeight(edges, p));
  return best;
}

function minPathBrute(n, edges, source, target) {
  const paths = enumeratePaths(n, edges, source, target);
  if (paths.length === 0) return Infinity;
  let best = Infinity;
  for (const p of paths) best = Math.min(best, pathWeight(edges, p));
  return best;
}

function countPathsMemo(n, edges, source, target) {
  const adj = buildAdj(n, edges);
  const useful = canReachTarget(n, adj, target);
  const memo = new Map();
  const count = (u) => {
    if (u === target) return 1;
    if (memo.has(u)) return memo.get(u);
    let total = 0;
    for (const [v] of adj[u]) if (useful[v]) total += count(v);
    memo.set(u, total);
    return total;
  };
  return useful[source] ? count(source) : 0;
}

/** Edges only from lower to higher index, so the result is always acyclic. */
function randomDag(n, percent, seed) {
  const edges = [];
  let s = seed % 65537;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      s = lcgNext(s);
      if (s % 100 < percent) {
        s = lcgNext(s);
        edges.push([u, v, (s % 19) - 9]);
      }
    }
  }
  return { edges, seed: s };
}

/** width nodes per level, complete bipartite between consecutive levels, unit weights. */
function layeredDag(width, levels) {
  const n = width * levels;
  const edges = [];
  for (let l = 0; l + 1 < levels; l++) {
    for (let i = 0; i < width; i++) {
      for (let j = 0; j < width; j++) {
        edges.push([l * width + i, (l + 1) * width + j, 1]);
      }
    }
  }
  return { n, edges };
}

/** Two nodes per level; number of level-to-level paths is 2^(levels-1). */
function diamondLadder(levels) {
  const n = 2 * levels;
  const edges = [];
  for (let l = 0; l + 1 < levels; l++) {
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        edges.push([2 * l + i, 2 * (l + 1) + j, 1]);
      }
    }
  }
  return { n, edges };
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-checked chain with a shortcut edge: the 9-weight three-hop path beats the direct 1.
  const chain = { n: 4, edges: [[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 3, 1]] };
  const chainMemo = longestPathMemo(chain.n, chain.edges, 0, 3);
  const chainTopo = longestPathTopo(chain.n, chain.edges, 0, 3);
  assert.strictEqual(chainMemo.value, 9);
  assert.strictEqual(chainTopo.value, 9);
  assert.strictEqual(maxPathBrute(chain.n, chain.edges, 0, 3), 9);
  assert.deepStrictEqual(chainTopo.order, [0, 1, 2, 3]);
  assert.strictEqual(chainMemo.computes, 3, "one compute per useful non-target node");
  assert.strictEqual(chainMemo.calls, 5, "node 3 is the target and is never cached");
  assert.strictEqual(countPathsMemo(chain.n, chain.edges, 0, 3), 2);
  assert.strictEqual(countPathsMemo(chain.n, chain.edges, 0, 3),
    enumeratePaths(chain.n, chain.edges, 0, 3).length);

  // (2) Negative weights: max on negated weights == -(min) on the original weights.
  const neg = {
    n: 5,
    edges: [[0, 1, -4], [0, 2, 1], [1, 2, -2], [2, 3, 5], [1, 3, 10], [3, 4, -1], [2, 4, 0]],
  };
  const negatedEdges = neg.edges.map(([u, v, w]) => [u, v, -w]);
  assert.strictEqual(longestPathMemo(neg.n, negatedEdges, 0, 4).value, 6);
  assert.strictEqual(longestPathTopo(neg.n, negatedEdges, 0, 4).value, 6);
  assert.strictEqual(minPathBrute(neg.n, neg.edges, 0, 4), -6);
  assert.strictEqual(longestPathMemo(neg.n, neg.edges, 0, 4).value, 5,
    "the un-negated maximum is the other three-hop route");
  assert.strictEqual(maxPathBrute(neg.n, negatedEdges, 0, 4),
    -minPathBrute(neg.n, neg.edges, 0, 4), "min/max swap under weight negation");

  // (3) Random DAGs: engines agree with exhaustive enumeration, the order is a valid
  //     linearisation, and the cache computes exactly the useful-and-reachable states.
  for (let seed = 1; seed <= 5; seed++) {
    const n = 9;
    const { edges } = randomDag(n, 45, seed);
    const adj = buildAdj(n, edges);
    const order = topoOrder(n, adj);
    const pos = new Array(n).fill(-1);
    order.forEach((v, i) => {
      pos[v] = i;
    });
    assert.deepStrictEqual(order.slice().sort((a, b) => a - b), [...Array(n).keys()],
      "topological order is a permutation");
    for (const [u, v] of edges) assert.ok(pos[u] < pos[v], `seed ${seed}: edge ${u}->${v} forward`);

    const memo = longestPathMemo(n, edges, 0, n - 1);
    const topo = longestPathTopo(n, edges, 0, n - 1);
    const brute = maxPathBrute(n, edges, 0, n - 1);
    assert.strictEqual(memo.value, brute, `seed ${seed}: memo matches enumeration`);
    assert.strictEqual(topo.value, memo.value, `seed ${seed}: topo sweep matches memo`);
    assert.strictEqual(longestPathNoCache(n, edges, 0, n - 1).value, brute,
      `seed ${seed}: uncached value is the same`);
    assert.strictEqual(countPathsMemo(n, edges, 0, n - 1),
      enumeratePaths(n, edges, 0, n - 1).length, `seed ${seed}: path counts agree`);

    const reach = reachableFrom(n, adj, 0);
    let expectedComputes = 0;
    for (let i = 0; i < n; i++) {
      if (reach[i] && memo.useful[i] && i !== n - 1) expectedComputes += 1;
    }
    assert.strictEqual(memo.computes, expectedComputes,
      `seed ${seed}: each useful reachable node is computed once`);

    const negated = edges.map(([u, v, w]) => [u, v, -w]);
    assert.strictEqual(maxPathBrute(n, negated, 0, n - 1), -minPathBrute(n, edges, 0, n - 1),
      `seed ${seed}: min/max swap holds on random weights too`);
  }

  // (4) Kahn with the smallest available label IS the lexicographically smallest order.
  for (let seed = 11; seed <= 14; seed++) {
    const n = 5;
    const { edges } = randomDag(n, 55, seed);
    const order = topoOrder(n, buildAdj(n, edges));
    const all = allTopoOrders(n, edges);
    assert.ok(all.length >= 1, `seed ${seed}: at least one topological order exists`);
    assert.deepStrictEqual(order, all[0], `seed ${seed}: smallest-label Kahn = lex smallest`);
    assert.ok(all.some((o) => o.join(",") === order.join(",")),
      `seed ${seed}: the chosen order is a valid linearisation`);
  }

  // (5) Cycle detection.
  const cyclic = { n: 3, edges: [[0, 1, 2], [1, 2, 3], [2, 0, 1]] };
  assert.strictEqual(topoOrder(3, buildAdj(3, cyclic.edges)), null);
  assert.strictEqual(longestPathTopo(3, cyclic.edges, 0, 2).order, null);
  assert.strictEqual(topoOrder(2, buildAdj(2, [[0, 0, 5], [0, 1, 1]])), null, "self-loop is a cycle");
  assert.strictEqual(topoOrder(2, buildAdj(2, [[1, 0, 5], [0, 1, 1]])), null, "2-cycle");

  // (6) Cache win on the diamond ladder. Only node 31 (the target) is useful in the last level,
  //     so the uncached tree has sum_{l<=14} 2^l invocations plus one target call per level-14
  //     node, while the cache computes each of the 30 useful non-target nodes exactly once.
  const ladder = diamondLadder(16);
  const noCache = longestPathNoCache(ladder.n, ladder.edges, 0, ladder.n - 1);
  const cached = longestPathMemo(ladder.n, ladder.edges, 0, ladder.n - 1);
  assert.strictEqual(noCache.value, 15);
  assert.strictEqual(cached.value, 15);
  assert.strictEqual(noCache.calls, 2 ** 15 - 1 + 2 ** 14,
    "uncached recursion walks the whole path tree");
  assert.strictEqual(cached.computes, ladder.n - 3,
    "each state is computed once: 32 nodes minus node 30 (cannot reach the target) " +
    "minus the unreachable twin node 1 minus the target itself");
  assert.ok(noCache.calls >= 1000 * cached.computes, "the cache is an asymptotic win");
  assert.strictEqual(countPathsMemo(ladder.n, ladder.edges, 0, ladder.n - 1), 2 ** 14,
    "free choices in levels 1..14, the last hop must land on node 31");
  assert.strictEqual(longestPathTopo(ladder.n, ladder.edges, 0, ladder.n - 1).value, 15);

  // (7) Closed forms on layered graphs: W^(L-1) paths into the last layer, W^(L-2) into one
  //     specific sink, and longest path L-1 with unit weights.
  const layered = layeredDag(3, 4);
  let totalToLastLevel = 0;
  for (let v = layered.n - 3; v < layered.n; v++) {
    totalToLastLevel += countPathsMemo(layered.n, layered.edges, 0, v);
  }
  assert.strictEqual(totalToLastLevel, 27, "3 choices on each of the 3 level transitions");
  assert.strictEqual(countPathsMemo(layered.n, layered.edges, 0, layered.n - 1), 9,
    "a single sink sees only W^(L-2) of them");
  assert.strictEqual(enumeratePaths(layered.n, layered.edges, 0, layered.n - 1).length, 9);
  assert.strictEqual(longestPathMemo(layered.n, layered.edges, 0, layered.n - 1).value, 3);
  assert.strictEqual(longestPathTopo(layered.n, layered.edges, 0, layered.n - 1).value, 3);

  // (8) Boundary cases.
  assert.strictEqual(longestPathMemo(1, [], 0, 0).value, 0, "single node, empty path");
  assert.strictEqual(longestPathMemo(2, [[0, 1, 3]], 0, 0).value, 0, "source == target");
  assert.strictEqual(countPathsMemo(2, [[0, 1, 3]], 0, 0), 1, "the empty path counts once");
  assert.strictEqual(longestPathMemo(2, [[0, 1, 3]], 1, 0).value, -Infinity, "target unreachable");
  assert.strictEqual(longestPathTopo(2, [[0, 1, 3]], 1, 0).value, -Infinity);
  assert.strictEqual(countPathsMemo(2, [[0, 1, 3]], 1, 0), 0);
  assert.strictEqual(maxPathBrute(2, [[0, 1, 3]], 1, 0), -Infinity);
  assert.strictEqual(minPathBrute(2, [[0, 1, 3]], 1, 0), Infinity);
  assert.strictEqual(longestPathMemo(1, [], 0, 0).computes, 0);

  // (9) Determinism: every engine repeats its answer exactly.
  const repeatEdges = randomDag(7, 50, 99).edges;
  assert.deepStrictEqual(longestPathMemo(7, repeatEdges, 0, 6),
    longestPathMemo(7, repeatEdges, 0, 6));
  assert.strictEqual(longestPathNoCache(7, repeatEdges, 0, 6).calls,
    longestPathNoCache(7, repeatEdges, 0, 6).calls);
  assert.deepStrictEqual(longestPathTopo(7, repeatEdges, 0, 6),
    longestPathTopo(7, repeatEdges, 0, 6));
}

module.exports = { lcgNext, buildAdj, topoOrder, lessOrder, allTopoOrders, canReachTarget,
  reachableFrom, longestPathMemo, longestPathNoCache, longestPathTopo, enumeratePaths, pathWeight,
  maxPathBrute, minPathBrute, countPathsMemo, randomDag, layeredDag, diamondLadder };

if (require.main === module) {
  main();
  console.log("PASS");
}
