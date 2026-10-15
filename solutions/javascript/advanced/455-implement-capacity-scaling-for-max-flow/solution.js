/** Task 455 - Implement capacity scaling for max flow. (Tier: Advanced, CTCI Ch 4) */
"use strict";

const INF = Number.MAX_SAFE_INTEGER;

/** Arc-list flow network: to/cap arrays, reverse arc index is idx ^ 1. */
class MaxFlow {
  constructor(n) {
    this.n = n;
    this.to = [];
    this.initial = [];
    this.cap = [];
    this.head = Array.from({ length: n }, () => []);
  }

  addEdge(u, v, capacity) {
    const idx = this.to.length;
    this.to.push(v);
    this.initial.push(capacity);
    this.cap.push(capacity);
    this.head[u].push(idx);
    this.to.push(u);
    this.initial.push(0);
    this.cap.push(0);
    this.head[v].push(idx + 1);
    return idx;
  }

  augment(parentEdge, s, t) {
    let aug = INF;
    for (let v = t; v !== s; v = this.to[parentEdge[v] ^ 1]) aug = Math.min(aug, this.cap[parentEdge[v]]);
    for (let v = t; v !== s; v = this.to[parentEdge[v] ^ 1]) {
      const idx = parentEdge[v];
      this.cap[idx] -= aug;
      this.cap[idx ^ 1] += aug;
    }
    return aug;
  }

  findPath(s, t, minCapacity) {
    const parentEdge = new Array(this.n).fill(-1);
    parentEdge[s] = -2;
    const q = [s];
    for (let qh = 0; qh < q.length; qh++) {
      const u = q[qh];
      for (const idx of this.head[u]) {
        const v = this.to[idx];
        if (parentEdge[v] === -1 && this.cap[idx] >= minCapacity) {
          parentEdge[v] = idx;
          q.push(v);
        }
      }
    }
    return parentEdge[t] !== -1 ? parentEdge : null;
  }

  edmondsKarp(s, t) {
    let flow = 0;
    for (;;) {
      const parentEdge = this.findPath(s, t, 1);
      if (parentEdge === null) return flow;
      flow += this.augment(parentEdge, s, t);
    }
  }

  /** Returns {value, deltas, perPhase, amounts}. */
  capacityScaling(s, t) {
    let maxCapacity = 0;
    for (let idx = 0; idx < this.to.length; idx += 2) maxCapacity = Math.max(maxCapacity, this.initial[idx]);
    let delta = 0;
    if (maxCapacity >= 1) {
      delta = 1;
      while (delta * 2 <= maxCapacity) delta *= 2;
    }
    const deltas = [];
    const perPhase = [];
    const amounts = [];
    let flow = 0;
    while (delta >= 1) {
      deltas.push(delta);
      let count = 0;
      for (;;) {
        const parentEdge = this.findPath(s, t, delta);
        if (parentEdge === null) break;
        const amount = this.augment(parentEdge, s, t);
        if (amount < delta) throw new Error("FAILED: augmentation at least delta");
        amounts.push(amount);
        flow += amount;
        count++;
      }
      perPhase.push(count);
      delta >>= 1;
    }
    return { value: flow, deltas, perPhase, amounts };
  }
}

function buildNetwork(n, edges) {
  const network = new MaxFlow(n);
  for (const [u, v, c] of edges) network.addEdge(u, v, c);
  return network;
}

function lcgFlowGraphs(count, seed) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 4 + (seed % 4);
    const edges = [];
    for (let node = 0; node < n - 1; node++) {
      seed = (seed * 75 + 74) % 65537;
      edges.push([node, node + 1, 1 + (seed % 500)]);
    }
    for (let k = 0; k < n + 3; k++) {
      seed = (seed * 75 + 74) % 65537;
      const u = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const v = seed % n;
      seed = (seed * 75 + 74) % 65537;
      if (u !== v) edges.push([u, v, 1 + (seed % 500)]);
    }
    graphs.push({ n, edges });
  }
  return graphs;
}

function phaseCountFor(maxCapacity) {
  let delta = 1;
  let count = 0;
  while (delta <= maxCapacity) {
    count++;
    delta *= 2;
  }
  return count;
}

module.exports = { MaxFlow, buildNetwork, lcgFlowGraphs, phaseCountFor };

if (require.main === module) {
  const assert = require("node:assert");
  let scaling = buildNetwork(2, [[0, 1, 6]]).capacityScaling(0, 1);
  assert.strictEqual(scaling.value, 6);
  assert.deepStrictEqual(scaling.deltas, [4, 2, 1]);
  assert.deepStrictEqual(scaling.perPhase, [1, 0, 0]);
  assert.deepStrictEqual(scaling.amounts, [6]);

  scaling = buildNetwork(2, [[0, 1, 3], [0, 1, 4]]).capacityScaling(0, 1);
  assert.strictEqual(scaling.value, 7);
  assert.deepStrictEqual(scaling.deltas, [4, 2, 1]);
  assert.deepStrictEqual(scaling.perPhase, [1, 1, 0]);
  assert.deepStrictEqual(scaling.amounts, [4, 3]);

  const chainEdges = [[0, 1, 1000], [1, 2, 999]];
  scaling = buildNetwork(3, chainEdges).capacityScaling(0, 2);
  assert.strictEqual(scaling.value, 999);
  assert.strictEqual(scaling.deltas.length, 10);
  assert.strictEqual(scaling.deltas[0], 512);
  assert.strictEqual(scaling.deltas[9], 1);
  assert.strictEqual(scaling.amounts.reduce((a, b) => a + b, 0), scaling.value);
  assert.strictEqual(buildNetwork(3, chainEdges).edmondsKarp(0, 2), 999);

  scaling = buildNetwork(2, [[0, 1, 1]]).capacityScaling(0, 1);
  assert.strictEqual(scaling.value, 1);
  assert.deepStrictEqual(scaling.deltas, [1]);
  assert.deepStrictEqual(scaling.amounts, [1]);

  scaling = buildNetwork(2, []).capacityScaling(0, 1);
  assert.strictEqual(scaling.value, 0);
  assert.strictEqual(scaling.deltas.length, 0);
  assert.strictEqual(scaling.amounts.length, 0);

  const clrs = [
    [0, 1, 16], [0, 2, 13], [1, 2, 10], [2, 1, 4], [1, 3, 12], [3, 2, 9],
    [2, 4, 14], [4, 3, 7], [3, 5, 20], [4, 5, 4],
  ];
  scaling = buildNetwork(6, clrs).capacityScaling(0, 5);
  assert.strictEqual(scaling.value, 23);
  assert.deepStrictEqual(scaling.deltas, [16, 8, 4, 2, 1]);
  assert.strictEqual(scaling.amounts.reduce((a, b) => a + b, 0), scaling.value);
  assert.strictEqual(buildNetwork(6, clrs).edmondsKarp(0, 5), 23);

  let checked = 0;
  let latePhaseWork = 0;
  let totalAugmentations = 0;
  for (const seed of [4551, 4552, 4553]) {
    for (const { n, edges } of lcgFlowGraphs(8, seed)) {
      scaling = buildNetwork(n, edges).capacityScaling(0, n - 1);
      const maxCapacity = Math.max(...edges.map(([, , c]) => c));
      assert.strictEqual(scaling.deltas.length, phaseCountFor(maxCapacity));
      assert.ok(scaling.deltas[0] * 2 > maxCapacity && scaling.deltas[0] <= maxCapacity);
      assert.strictEqual(scaling.amounts.reduce((a, b) => a + b, 0), scaling.value);
      assert.strictEqual(scaling.perPhase.reduce((a, b) => a + b, 0), scaling.amounts.length);
      totalAugmentations += scaling.amounts.length;
      if (scaling.perPhase.slice(1).some((c) => c > 0)) latePhaseWork++;
      const expected = buildNetwork(n, edges).edmondsKarp(0, n - 1);
      assert.strictEqual(scaling.value, expected, `LCG value n=${n}`);
      checked++;
    }
  }
  assert.strictEqual(checked, 24);
  assert.ok(totalAugmentations > 0);
  assert.ok(latePhaseWork >= 1);
  console.log("PASS");
}
