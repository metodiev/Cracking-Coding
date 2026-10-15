/** Task 459 - Implement virtual tree building from set of nodes (for queries). (Tier: Advanced, CTCI Ch 4) */
"use strict";

function buildTreeData(n, parent, edgeWeight) {
  const children = Array.from({ length: n }, () => []);
  for (let v = 1; v < n; v++) children[parent[v]].push(v);
  const tin = new Array(n).fill(0);
  const depth = new Array(n).fill(0);
  const dist = new Array(n).fill(0);
  let order = 0;
  const stack = [0];
  while (stack.length > 0) {
    const u = stack.pop();
    tin[u] = order++;
    for (let i = children[u].length - 1; i >= 0; i--) {
      const child = children[u][i];
      depth[child] = depth[u] + 1;
      dist[child] = dist[u] + edgeWeight[child];
      stack.push(child);
    }
  }
  return { children, tin, depth, dist };
}

function lca(parent, depth, u, v) {
  while (depth[u] > depth[v]) u = parent[u];
  while (depth[v] > depth[u]) v = parent[v];
  while (u !== v) {
    u = parent[u];
    v = parent[v];
  }
  return u;
}

function isAncestor(parent, candidate, node) {
  while (node !== -1) {
    if (node === candidate) return true;
    node = parent[node];
  }
  return false;
}

function naiveDistance(parent, depth, dist, u, v) {
  return dist[u] + dist[v] - 2 * dist[lca(parent, depth, u, v)];
}

/** Returns {nodes, edges} with edges as [vtParent, vtChild, weight]. */
function buildVirtualTree(parent, depth, dist, tin, selected) {
  const order = selected.slice().sort((a, b) => tin[a] - tin[b]);
  if (order.length === 0) return { nodes: [], edges: [] };
  const stack = [order[0]];
  const edges = [];
  for (let idx = 1; idx < order.length; idx++) {
    const v = order[idx];
    const l = lca(parent, depth, v, stack[stack.length - 1]);
    if (l === stack[stack.length - 1]) {
      stack.push(v);
      continue;
    }
    while (stack.length >= 2 && depth[stack[stack.length - 2]] >= depth[l]) {
      const child = stack.pop();
      const par = stack[stack.length - 1];
      edges.push([par, child, dist[child] - dist[par]]);
    }
    if (stack[stack.length - 1] !== l) {
      const child = stack.pop();
      edges.push([l, child, dist[child] - dist[l]]);
      stack.push(l);
    }
    stack.push(v);
  }
  while (stack.length >= 2) {
    const child = stack.pop();
    const par = stack[stack.length - 1];
    edges.push([par, child, dist[child] - dist[par]]);
  }
  const nodeSet = new Set(selected);
  for (const [a, b] of edges) {
    nodeSet.add(a);
    nodeSet.add(b);
  }
  const nodes = Array.from(nodeSet).sort((a, b) => a - b);
  return { nodes, edges };
}

function virtualTreePathWeight(nodes, edges, source, target) {
  const adj = new Map();
  for (const v of nodes) adj.set(v, []);
  for (const [a, b, w] of edges) {
    adj.get(a).push([b, w]);
    adj.get(b).push([a, w]);
  }
  const seen = new Set([source]);
  const stack = [[source, 0]];
  while (stack.length > 0) {
    const [node, total] = stack.pop();
    if (node === target) return total;
    for (const [nxt, w] of adj.get(node)) {
      if (!seen.has(nxt)) {
        seen.add(nxt);
        stack.push([nxt, total + w]);
      }
    }
  }
  return null;
}

function checkVirtualTree(assert, n, parent, selected, data) {
  const { nodes, edges } = buildVirtualTree(parent, data.depth, data.dist, data.tin, selected);
  const nodeSet = new Set(nodes);
  for (const s of selected) assert.ok(nodeSet.has(s), "selected present");
  for (const v of nodes) {
    assert.ok(selected.some((s) => isAncestor(parent, v, s)), `ancestor-or-selected ${v}`);
  }
  assert.strictEqual(edges.length, nodes.length - 1, "tree edge count");
  assert.ok(nodes.length <= 2 * selected.length - 1, "node bound");
  for (const [a, b, w] of edges) {
    assert.ok(w === data.dist[b] - data.dist[a] && w > 0, "edge weight");
  }
  for (let i = 0; i < selected.length; i++) {
    for (let j = i + 1; j < selected.length; j++) {
      const got = virtualTreePathWeight(nodes, edges, selected[i], selected[j]);
      const want = naiveDistance(parent, data.depth, data.dist, selected[i], selected[j]);
      assert.strictEqual(got, want, `pair distance ${selected[i]},${selected[j]}`);
    }
  }
}

function lcgTrees(count, seed) {
  const trees = [];
  for (let g = 0; g < count; g++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 8 + (seed % 5);
    const parent = new Array(n).fill(-1);
    const edgeWeight = new Array(n).fill(0);
    for (let v = 1; v < n; v++) {
      seed = (seed * 75 + 74) % 65537;
      parent[v] = seed % v;
      seed = (seed * 75 + 74) % 65537;
      edgeWeight[v] = 1 + (seed % 10);
    }
    trees.push({ n, parent, edgeWeight });
  }
  return trees;
}

function lcgSelection(n, seed, size) {
  seed = (seed * 75 + 74) % 65537;
  const pool = [];
  for (let i = 0; i < n; i++) pool.push(i);
  const count = Math.min(size, n);
  const selected = [];
  for (let k = 0; k < count; k++) {
    seed = (seed * 75 + 74) % 65537;
    const idx = seed % pool.length;
    selected.push(pool[idx]);
    pool[idx] = pool[pool.length - 1];
    pool.pop();
    seed = (seed * 75 + 74) % 65537;
  }
  return selected;
}

module.exports = { buildTreeData, lca, isAncestor, naiveDistance, buildVirtualTree, virtualTreePathWeight, checkVirtualTree, lcgTrees, lcgSelection };

if (require.main === module) {
  const assert = require("node:assert");
  const chainParent = [-1, 0, 1, 2, 3];
  const chainWeight = [0, 1, 1, 1, 1];
  const chain = buildTreeData(5, chainParent, chainWeight);
  checkVirtualTree(assert, 5, chainParent, [2, 4], chain);
  let built = buildVirtualTree(chainParent, chain.depth, chain.dist, chain.tin, [2, 4]);
  assert.deepStrictEqual(built.nodes, [2, 4]);
  assert.deepStrictEqual(built.edges, [[2, 4, 2]]);
  checkVirtualTree(assert, 5, chainParent, [1, 3], chain);
  checkVirtualTree(assert, 5, chainParent, [0, 3], chain);
  checkVirtualTree(assert, 5, chainParent, [1], chain);
  built = buildVirtualTree(chainParent, chain.depth, chain.dist, chain.tin, [1]);
  assert.deepStrictEqual(built.nodes, [1]);
  assert.deepStrictEqual(built.edges, []);

  const starParent = [-1, 0, 0, 0];
  const starWeight = [0, 1, 1, 1];
  const star = buildTreeData(4, starParent, starWeight);
  checkVirtualTree(assert, 4, starParent, [1, 2], star);
  built = buildVirtualTree(starParent, star.depth, star.dist, star.tin, [1, 2]);
  assert.deepStrictEqual(built.nodes, [0, 1, 2]);
  checkVirtualTree(assert, 4, starParent, [1, 2, 3], star);

  const balParent = [-1, 0, 0, 1, 1, 2, 2];
  const balWeight = [0, 3, 4, 5, 6, 7, 8];
  const balanced = buildTreeData(7, balParent, balWeight);
  checkVirtualTree(assert, 7, balParent, [3, 4, 5, 6], balanced);

  let checked = 0;
  let trees = 0;
  for (const seed of [4591, 4592, 4593, 4594]) {
    for (const { n, parent, edgeWeight } of lcgTrees(4, seed)) {
      const data = buildTreeData(n, parent, edgeWeight);
      const seed2 = seed + n;
      for (let extra = 0; extra < 2; extra++) {
        const size = 2 + (seed2 % 4);
        const selected = lcgSelection(n, seed2 + extra * 3, size);
        checkVirtualTree(assert, n, parent, selected, data);
        checked++;
      }
      trees++;
    }
  }
  assert.strictEqual(trees, 16);
  assert.strictEqual(checked, 32);
  console.log("PASS");
}
