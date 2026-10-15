/** Task 426 - Implement tree DP for independent set in tree. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function rootTree(n, edges) {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adjacency[a].push(b);
    adjacency[b].push(a);
  }
  const parent = new Array(n).fill(-1);
  parent[0] = 0;
  const order = [0];
  for (let i = 0; i < order.length; i++) {
    const u = order[i];
    for (const v of adjacency[u]) {
      if (v !== parent[u]) {
        parent[v] = u;
        order.push(v);
      }
    }
  }
  const children = Array.from({ length: n }, () => []);
  for (let i = 1; i < order.length; i++) children[parent[order[i]]].push(order[i]);
  return { children, order };
}

function maxWeightIndependentSet(n, edges, weights) {
  const { children, order } = rootTree(n, edges);
  const take = new Array(n).fill(0);
  const skip = new Array(n).fill(0);
  for (let index = order.length - 1; index >= 0; index--) {
    const u = order[index];
    let chosen = weights[u];
    let notChosen = 0;
    for (const v of children[u]) {
      chosen += skip[v];
      notChosen += Math.max(take[v], skip[v]);
    }
    take[u] = chosen;
    skip[u] = notChosen;
  }
  const picked = new Array(n).fill(false);
  const stack = [[0, take[0] > skip[0]]];
  while (stack.length) {
    const [u, taking] = stack.pop();
    picked[u] = taking;
    for (const v of children[u]) stack.push([v, taking ? false : take[v] > skip[v]]);
  }
  let total = 0;
  for (let u = 0; u < n; u++) {
    if (picked[u]) total += weights[u];
  }
  return { total, picked };
}

function isValid(n, edges, picked) {
  return edges.every(([a, b]) => !(picked[a] && picked[b]));
}

function weightOf(n, weights, picked) {
  let total = 0;
  for (let u = 0; u < n; u++) {
    if (picked[u]) total += weights[u];
  }
  return total;
}

function bruteForce(n, edges, weights) {
  let best = 0;
  for (let mask = 0; mask < (1 << n); mask++) {
    let weight = 0;
    for (let i = 0; i < n; i++) {
      if ((mask >> i) & 1) weight += weights[i];
    }
    if (weight <= best) continue;
    if (edges.every(([a, b]) => !(((mask >> a) & 1) && ((mask >> b) & 1)))) best = weight;
  }
  return best;
}

module.exports = { rootTree, maxWeightIndependentSet, isValid, weightOf, bruteForce };

if (require.main === module) {
  const assert = require("node:assert");

  let result = maxWeightIndependentSet(3, [[0, 1], [1, 2]], [10, 1, 10]);
  assert.strictEqual(result.total, 20);
  assert.deepStrictEqual(result.picked, [true, false, true]);

  result = maxWeightIndependentSet(3, [[0, 1], [0, 2]], [5, 4, 4]);
  assert.strictEqual(result.total, 8);
  assert.deepStrictEqual(result.picked, [false, true, true]);

  result = maxWeightIndependentSet(2, [[0, 1]], [-1, -2]);
  assert.strictEqual(result.total, 0);
  assert.deepStrictEqual(result.picked, [false, false]);

  result = maxWeightIndependentSet(1, [], [7]);
  assert.strictEqual(result.total, 7);
  assert.deepStrictEqual(result.picked, [true]);
  result = maxWeightIndependentSet(1, [], [-7]);
  assert.strictEqual(result.total, 0);
  assert.deepStrictEqual(result.picked, [false]);

  result = maxWeightIndependentSet(4, [[0, 1], [1, 2], [2, 3]], [3, 10, 3, 10]);
  assert.strictEqual(result.total, 20);
  assert.deepStrictEqual(result.picked, [false, true, false, true]);

  let local = 426426;
  for (let trial = 0; trial < 15; trial++) {
    local = (local * 75 + 74) % 65537;
    const n = 2 + (local % 15);
    const edges = [];
    const weights = [];
    for (let node = 0; node < n; node++) {
      local = (local * 75 + 74) % 65537;
      weights.push((local % 21) - 10);
      if (node > 0) {
        local = (local * 75 + 74) % 65537;
        edges.push([local % node, node]);
      }
    }
    const got = maxWeightIndependentSet(n, edges, weights);
    const expected = bruteForce(n, edges, weights);
    assert.strictEqual(got.total, expected, `small trial ${trial}`);
    assert.ok(isValid(n, edges, got.picked), `validity ${trial}`);
    assert.strictEqual(weightOf(n, weights, got.picked), got.total, `reconstruction ${trial}`);
  }

  local = 99999;
  for (let trial = 0; trial < 10; trial++) {
    local = (local * 75 + 74) % 65537;
    const n = 20 + (local % 41);
    const edges = [];
    const weights = [];
    for (let node = 0; node < n; node++) {
      local = (local * 75 + 74) % 65537;
      weights.push((local % 41) - 20);
      if (node > 0) {
        local = (local * 75 + 74) % 65537;
        edges.push([local % node, node]);
      }
    }
    const got = maxWeightIndependentSet(n, edges, weights);
    assert.ok(isValid(n, edges, got.picked), `large validity ${trial}`);
    assert.strictEqual(weightOf(n, weights, got.picked), got.total, `large reconstruction ${trial}`);
    assert.ok(got.total >= 0, `large non-negative ${trial}`);
  }

  console.log("PASS");
}
