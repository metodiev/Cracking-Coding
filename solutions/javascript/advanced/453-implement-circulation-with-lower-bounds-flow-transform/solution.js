/** Task 453 - Implement circulation with lower bounds (flow transform). (Tier: Advanced, CTCI Ch 4) */
"use strict";

const INF = Number.MAX_SAFE_INTEGER;

/** Edmonds-Karp on an explicit arc list; each arc keeps its own reverse residual. */
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

  used(idx) {
    return this.initial[idx] - this.cap[idx];
  }

  maxFlow(s, t) {
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

/** Returns the edge flows for edges = [u, v, lower, upper], or null when infeasible. */
function circulation(n, edges) {
  const ss = n;
  const tt = n + 1;
  const network = new MaxFlow(n + 2);
  const balance = new Array(n).fill(0);
  const arcIds = [];
  for (const [u, v, lower, upper] of edges) {
    balance[v] += lower;
    balance[u] -= lower;
    arcIds.push(network.addEdge(u, v, upper - lower));
  }
  let required = 0;
  for (let v = 0; v < n; v++) {
    if (balance[v] > 0) {
      network.addEdge(ss, v, balance[v]);
      required += balance[v];
    } else if (balance[v] < 0) {
      network.addEdge(v, tt, -balance[v]);
    }
  }
  if (network.maxFlow(ss, tt) !== required) return null;
  return edges.map(([, , lower], i) => lower + network.used(arcIds[i]));
}

/** s-t flow of exactly `requiredValue` via the t->s [F, F] circulation transform. */
function circulationWithRequiredValue(n, edges, s, t, requiredValue) {
  return circulation(n, [...edges, [t, s, requiredValue, requiredValue]]);
}

/** Enumerate every integer flow assignment in [lower, upper] and test conservation. */
function bruteFeasible(n, edges) {
  const m = edges.length;
  const values = new Array(m).fill(0);
  for (;;) {
    const balance = new Array(n).fill(0);
    for (let i = 0; i < m; i++) {
      const f = edges[i][2] + values[i];
      balance[edges[i][0]] -= f;
      balance[edges[i][1]] += f;
    }
    if (balance.every((b) => b === 0)) return true;
    let j = 0;
    while (j < m) {
      if (values[j] + 1 < edges[j][3] - edges[j][2] + 1) {
        values[j]++;
        break;
      }
      values[j] = 0;
      j++;
    }
    if (j === m) return false;
  }
}

function checkFlowValid(assert, n, edges, flows) {
  const balance = new Array(n).fill(0);
  for (let i = 0; i < edges.length; i++) {
    const [u, v, lower, upper] = edges[i];
    assert.ok(lower <= flows[i] && flows[i] <= upper, "flow within bounds");
    balance[u] -= flows[i];
    balance[v] += flows[i];
  }
  for (const b of balance) assert.strictEqual(b, 0, "conservation");
}

function lcgInstances(count, seed) {
  const instances = [];
  for (let g = 0; g < count; g++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 3 + (seed % 2);
    seed = (seed * 75 + 74) % 65537;
    const m = 3 + (seed % 3);
    const pool = [];
    for (let u = 0; u < n; u++) for (let v = 0; v < n; v++) if (u !== v) pool.push([u, v]);
    const edges = [];
    for (let k = 0; k < Math.min(m, pool.length); k++) {
      seed = (seed * 75 + 74) % 65537;
      const idx = seed % pool.length;
      const [u, v] = pool[idx];
      pool[idx] = pool[pool.length - 1];
      pool.pop();
      seed = (seed * 75 + 74) % 65537;
      const lower = seed % 3;
      seed = (seed * 75 + 74) % 65537;
      const upper = lower + (seed % 3);
      edges.push([u, v, lower, upper]);
    }
    instances.push({ n, edges });
  }
  return instances;
}

module.exports = { MaxFlow, circulation, circulationWithRequiredValue, bruteFeasible, lcgInstances };

if (require.main === module) {
  const assert = require("node:assert");
  const triangle = [[0, 1, 1, 3], [1, 2, 1, 3], [2, 0, 1, 3]];
  let flows = circulation(3, triangle);
  assert.deepStrictEqual(flows, [1, 1, 1]);
  checkFlowValid(assert, 3, triangle, flows);
  assert.ok(bruteFeasible(3, triangle));

  const tightOut = [[0, 1, 2, 3], [1, 2, 0, 1], [2, 0, 1, 3]];
  assert.strictEqual(circulation(3, tightOut), null);
  assert.ok(!bruteFeasible(3, tightOut));

  const twoWay = [[0, 1, 3, 5], [1, 0, 0, 5]];
  flows = circulation(2, twoWay);
  assert.deepStrictEqual(flows, [3, 3]);
  checkFlowValid(assert, 2, twoWay, flows);
  assert.ok(bruteFeasible(2, twoWay));

  const tooMuchIn = [[0, 1, 4, 5], [1, 0, 0, 2]];
  assert.strictEqual(circulation(2, tooMuchIn), null);
  assert.ok(!bruteFeasible(2, tooMuchIn));

  const disjoint = [[0, 1, 1, 2], [1, 0, 1, 2], [2, 3, 0, 1], [3, 2, 0, 1]];
  flows = circulation(4, disjoint);
  assert.notStrictEqual(flows, null);
  checkFlowValid(assert, 4, disjoint, flows);
  assert.ok(bruteFeasible(4, disjoint));

  const deadEnd = [[0, 1, 1, 1]];
  assert.strictEqual(circulation(2, deadEnd), null);
  assert.ok(!bruteFeasible(2, deadEnd));

  const withIsolated = [[0, 1, 1, 1], [1, 0, 1, 1]];
  flows = circulation(3, withIsolated);
  assert.deepStrictEqual(flows, [1, 1]);
  checkFlowValid(assert, 3, withIsolated, flows);
  assert.ok(bruteFeasible(3, withIsolated));

  const stEdges = [[0, 1, 0, 2], [1, 3, 0, 2], [0, 2, 0, 1], [2, 3, 0, 1]];
  flows = circulationWithRequiredValue(4, stEdges, 0, 3, 3);
  assert.notStrictEqual(flows, null);
  assert.strictEqual(flows[flows.length - 1], 3);
  checkFlowValid(assert, 4, [...stEdges, [3, 0, 3, 3]], flows);
  assert.strictEqual(circulationWithRequiredValue(4, stEdges, 0, 3, 4), null);

  let feasibleCount = 0;
  let infeasibleCount = 0;
  for (const seed of [4531, 4532, 4533]) {
    for (const { n, edges } of lcgInstances(8, seed)) {
      flows = circulation(n, edges);
      const expected = bruteFeasible(n, edges);
      assert.strictEqual(flows !== null, expected, `n=${n} edges=${JSON.stringify(edges)}`);
      if (flows !== null) {
        checkFlowValid(assert, n, edges, flows);
        feasibleCount++;
      } else {
        infeasibleCount++;
      }
    }
  }
  assert.ok(feasibleCount >= 1 && infeasibleCount >= 1);
  assert.strictEqual(feasibleCount + infeasibleCount, 24);
  console.log("PASS");
}
