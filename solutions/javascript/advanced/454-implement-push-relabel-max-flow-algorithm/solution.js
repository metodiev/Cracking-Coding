/** Task 454 - Implement push-relabel max-flow algorithm. (Tier: Advanced, CTCI Ch 4) */
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

  flowOn(idx) {
    return this.initial[idx] - this.cap[idx];
  }

  /** FIFO preflow-push; returns [max flow, push count, relabel count]. */
  pushRelabel(s, t) {
    const n = this.n;
    const height = new Array(n).fill(0);
    const excess = new Array(n).fill(0);
    height[s] = n;
    const inQueue = new Array(n).fill(false);
    const queue = [];
    for (const idx of this.head[s]) {
      const amount = this.cap[idx];
      if (amount > 0) {
        const v = this.to[idx];
        this.cap[idx] -= amount;
        this.cap[idx ^ 1] += amount;
        excess[v] += amount;
        excess[s] -= amount;
      }
    }
    let pushes = 0;
    let relabels = 0;
    for (let v = 0; v < n; v++) {
      if (v !== s && v !== t && excess[v] > 0) {
        queue.push(v);
        inQueue[v] = true;
      }
    }
    for (let qh = 0; qh < queue.length; qh++) {
      const u = queue[qh];
      inQueue[u] = false;
      while (excess[u] > 0) {
        let pushed = false;
        for (const idx of this.head[u]) {
          const v = this.to[idx];
          if (this.cap[idx] > 0 && height[u] === height[v] + 1) {
            const amount = Math.min(excess[u], this.cap[idx]);
            this.cap[idx] -= amount;
            this.cap[idx ^ 1] += amount;
            excess[u] -= amount;
            excess[v] += amount;
            pushes++;
            if (v !== s && v !== t && !inQueue[v] && excess[v] > 0) {
              queue.push(v);
              inQueue[v] = true;
            }
            pushed = true;
            break;
          }
        }
        if (!pushed) {
          let best = INF;
          for (const idx of this.head[u]) {
            if (this.cap[idx] > 0 && height[this.to[idx]] < best) best = height[this.to[idx]];
          }
          height[u] = best + 1;
          relabels++;
        }
      }
    }
    return [excess[t], pushes, relabels];
  }

  edmondsKarp(s, t) {
    let flow = 0;
    for (;;) {
      const parentEdge = new Array(this.n).fill(-1);
      parentEdge[s] = -2;
      const q = [s];
      for (let qh = 0; qh < q.length; qh++) {
        const u = q[qh];
        for (const idx of this.head[u]) {
          const v = this.to[idx];
          if (parentEdge[v] === -1 && this.cap[idx] > 0) {
            parentEdge[v] = idx;
            q.push(v);
          }
        }
      }
      if (parentEdge[t] === -1) return flow;
      let aug = INF;
      for (let v = t; v !== s; v = this.to[parentEdge[v] ^ 1]) aug = Math.min(aug, this.cap[parentEdge[v]]);
      for (let v = t; v !== s; v = this.to[parentEdge[v] ^ 1]) {
        const idx = parentEdge[v];
        this.cap[idx] -= aug;
        this.cap[idx ^ 1] += aug;
      }
      flow += aug;
    }
  }
}

function buildNetwork(n, edges) {
  const network = new MaxFlow(n);
  const ids = edges.map(([u, v, c]) => network.addEdge(u, v, c));
  return { network, ids };
}

function checkFlowValid(assert, n, edges, ids, network, s, t, value) {
  const balance = new Array(n).fill(0);
  for (let i = 0; i < edges.length; i++) {
    const [u, v, c] = edges[i];
    const f = network.flowOn(ids[i]);
    assert.ok(f >= 0 && f <= c, "capacity bound");
    balance[u] -= f;
    balance[v] += f;
  }
  assert.strictEqual(balance[t], value, "sink balance");
  assert.strictEqual(balance[s], -value, "source balance");
  for (let node = 0; node < n; node++) {
    if (node !== s && node !== t) assert.strictEqual(balance[node], 0, `conservation at ${node}`);
  }
}

function lcgFlowGraphs(count, seed) {
  const graphs = [];
  for (let g = 0; g < count; g++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 4 + (seed % 5);
    const edges = [];
    for (let node = 0; node < n - 1; node++) {
      seed = (seed * 75 + 74) % 65537;
      edges.push([node, node + 1, 1 + (seed % 20)]);
    }
    for (let k = 0; k < n + 3; k++) {
      seed = (seed * 75 + 74) % 65537;
      const u = seed % n;
      seed = (seed * 75 + 74) % 65537;
      const v = seed % n;
      seed = (seed * 75 + 74) % 65537;
      if (u !== v) edges.push([u, v, 1 + (seed % 20)]);
    }
    graphs.push({ n, edges });
  }
  return graphs;
}

module.exports = { MaxFlow, buildNetwork, checkFlowValid, lcgFlowGraphs };

if (require.main === module) {
  const assert = require("node:assert");
  const clrs = [
    [0, 1, 16], [0, 2, 13], [1, 2, 10], [2, 1, 4], [1, 3, 12], [3, 2, 9],
    [2, 4, 14], [4, 3, 7], [3, 5, 20], [4, 5, 4],
  ];
  let built = buildNetwork(6, clrs);
  let result = built.network.pushRelabel(0, 5);
  assert.strictEqual(result[0], 23, "CLRS 26.1 value");
  assert.ok(result[1] > 0 && result[2] > 0, "CLRS pushes and relabels");
  checkFlowValid(assert, 6, clrs, built.ids, built.network, 0, 5, 23);
  assert.strictEqual(buildNetwork(6, clrs).network.edmondsKarp(0, 5), 23);

  const single = [[0, 1, 5]];
  built = buildNetwork(2, single);
  assert.strictEqual(built.network.pushRelabel(0, 1)[0], 5);
  checkFlowValid(assert, 2, single, built.ids, built.network, 0, 1, 5);

  const parallel = [[0, 1, 3], [0, 1, 4], [1, 2, 2]];
  assert.strictEqual(buildNetwork(3, parallel).network.pushRelabel(0, 2)[0], 2);
  assert.strictEqual(buildNetwork(3, parallel).network.edmondsKarp(0, 2), 2);

  const noPath = [[0, 1, 9], [2, 3, 9]];
  assert.strictEqual(buildNetwork(4, noPath).network.pushRelabel(0, 3)[0], 0);
  const zeroCap = [[0, 1, 0], [1, 2, 5]];
  assert.strictEqual(buildNetwork(3, zeroCap).network.pushRelabel(0, 2)[0], 0);

  let checked = 0;
  let totalPushes = 0;
  let totalRelabels = 0;
  for (const seed of [4541, 4542, 4543, 4544]) {
    for (const { n, edges } of lcgFlowGraphs(8, seed)) {
      built = buildNetwork(n, edges);
      result = built.network.pushRelabel(0, n - 1);
      totalPushes += result[1];
      totalRelabels += result[2];
      checkFlowValid(assert, n, edges, built.ids, built.network, 0, n - 1, result[0]);
      const expected = buildNetwork(n, edges).network.edmondsKarp(0, n - 1);
      assert.strictEqual(result[0], expected, `LCG value n=${n}`);
      checked++;
    }
  }
  assert.strictEqual(checked, 32);
  assert.ok(totalPushes > 0 && totalRelabels > 0);
  console.log("PASS");
}
