/** Task 706 - Implement max flow with lower/upper bounds and circulation feasibility. (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
 *
 * Flows with both a lower and an upper bound per edge cannot be built by pushing augmenting paths
 * from scratch, because the lower bounds already force flow into the vertices. The standard
 * construction removes them: every edge (u, v, lo, hi) contributes lo units to v and -lo to u, the
 * imbalance b(x) = in(x) - out(x) becomes the demand of x, and a circulation is feasible exactly
 * when a maximum flow in an auxiliary network with a super source (into the positive demands) and
 * a super sink (out of the negative demands) saturates every super-source edge. For a *maximum*
 * s-t flow with bounds, an edge t->s with bounds [0, INF] turns the problem into a circulation;
 * after that feasible circulation is found the value is the flow on t->s, and the remaining slack
 * is closed by augmenting s->t in the residual network of the original edges.
 *
 * What is simplified: integer capacities (the construction is exact for them, and the brute-force
 * oracle enumerates integer flows), no cost optimisation, and the oracle only runs on instances
 * small enough to enumerate every assignment of integer flows within the bounds.
 *
 * Invariant verified: every returned flow respects lo <= f <= hi on every edge and conserves flow
 * at every internal vertex, its value equals the flow leaving the source, that value never exceeds
 * sum(hi out of s) - sum(lo into s), it equals the plain max flow whenever every lower bound is 0,
 * and it equals the exhaustive optimum on small random instances; infeasible circulations and
 * invalid input are detected instead of silently producing a flow.
 */
"use strict";
const { strictEqual, ok, throws } = require("node:assert");
/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
/** Dinic's algorithm on integer capacities; keepTrack records the flow of every added edge. */
class Dinic {
  constructor(size) {
    this.size = size;
    this.head = Array.from({ length: size }, () => []);
    this.edges = [];
  }
  /** Adds a directed edge u->v with capacity; returns its index for flow bookkeeping. */
  addEdge(u, v, capacity) {
    if (capacity < 0) throw new RangeError("capacity must be non-negative");
    const index = this.edges.length;
    const forward = { to: v, capacity, flow: 0 };
    const backward = { to: u, capacity: 0, flow: 0 };
    forward.reverse = backward;
    backward.reverse = forward;
    this.head[u].push(forward);
    this.head[v].push(backward);
    this.edges.push(forward);
    return index;
  }
  #levels(source) {
    const level = new Array(this.size).fill(-1);
    level[source] = 0;
    const queue = [source];
    for (let at = 0; at < queue.length; at++) {
      for (const edge of this.head[queue[at]]) {
        if (edge.capacity - edge.flow > 0 && level[edge.to] === -1) {
          level[edge.to] = level[queue[at]] + 1;
          queue.push(edge.to);
        }
      }
    }
    return level;
  }
  #augment(v, sink, level, pushed, next) {
    if (v === sink) return pushed;
    for (; next[v] < this.head[v].length; next[v]++) {
      const edge = this.head[v][next[v]];
      const residual = edge.capacity - edge.flow;
      if (residual <= 0 || level[edge.to] !== level[v] + 1) continue;
      const sent = this.#augment(edge.to, sink, level, Math.min(pushed, residual), next);
      if (sent > 0) {
        edge.flow += sent;
        edge.reverse.flow -= sent;
        return sent;
      }
    }
    return 0;
  }
  maxFlow(source, sink) {
    let total = 0;
    for (;;) {
      const level = this.#levels(source);
      if (level[sink] === -1) return total;
      const next = new Array(this.size).fill(0);
      for (let sent = this.#augment(source, sink, level, Infinity, next); sent > 0;) {
        total += sent;
        sent = this.#augment(source, sink, level, Infinity, next);
      }
    }
  }
}

function validateBounds(size, edges) {
  for (const edge of edges) {
    for (const endpoint of [edge.u, edge.v]) {
      if (!Number.isInteger(endpoint) || endpoint < 0 || endpoint >= size) {
        throw new RangeError(`vertex ${endpoint} is outside 0..${size - 1}`);
      }
    }
    if (!Number.isInteger(edge.lo) || !Number.isInteger(edge.hi) || edge.lo < 0) {
      throw new RangeError("bounds must be non-negative integers");
    }
    if (edge.lo > edge.hi) throw new RangeError(`lo > hi on edge ${edge.u}->${edge.v}`);
  }
}
/**
 * Feasible circulation with lower/upper bounds. Returns {flows, value} where flows[i] is the flow
 * on edges[i], or null when no circulation satisfies the bounds. The super source/sink network
 * must saturate every super-source edge for the demands to be met.
 */
function feasibleCirculation(size, edges) {
  validateBounds(size, edges);
  const superSource = size;
  const superSink = size + 1;
  const network = new Dinic(size + 2);
  const balance = new Array(size).fill(0);
  for (const edge of edges) {
    balance[edge.v] += edge.lo; // lo units must arrive at v
    balance[edge.u] -= edge.lo; // and leave u
  }
  const indices = edges.map((edge) => network.addEdge(edge.u, edge.v, edge.hi - edge.lo));
  let required = 0;
  for (let v = 0; v < size; v++) {
    if (balance[v] > 0) {
      network.addEdge(superSource, v, balance[v]);
      required += balance[v];
    } else if (balance[v] < 0) {
      network.addEdge(v, superSink, -balance[v]);
    }
  }
  const saturated = network.maxFlow(superSource, superSink);
  if (saturated !== required) return null; // the lower bounds cannot all be met
  return { flows: indices.map((index, i) => edges[i].lo + network.edges[index].flow), value: 0 };
}
/**
 * Maximum s-t flow with lower and upper bounds. An edge t->s with bounds [0, INF] converts the
 * problem into a circulation; after a feasible one is found its flow on t->s is the current value,
 * that artificial edge is dropped, and the remaining slack is closed by augmenting s->t in the
 * residual network (forward capacity hi - f, backward capacity f - lo, so flow may be rerouted).
 */
function maxFlowWithBounds(size, source, sink, edges) {
  if (source === sink) throw new RangeError("source and sink must differ");
  validateBounds(size, edges);
  const augmented = [...edges, { u: sink, v: source, lo: 0, hi: Number.MAX_SAFE_INTEGER }];
  const circulation = feasibleCirculation(size, augmented);
  if (circulation === null) return null;
  const start = circulation.flows[edges.length]; // restored flow on the artificial t->s edge
  const residual = new Dinic(size);
  const forward = [];
  const backward = [];
  edges.forEach((edge, i) => {
    const flow = circulation.flows[i];
    forward.push(edge.hi - flow > 0 ? residual.addEdge(edge.u, edge.v, edge.hi - flow) : -1);
    backward.push(flow - edge.lo > 0 ? residual.addEdge(edge.v, edge.u, flow - edge.lo) : -1);
  });
  const extra = residual.maxFlow(source, sink);
  const flows = edges.map((edge, i) =>
    circulation.flows[i]
      + (forward[i] === -1 ? 0 : residual.edges[forward[i]].flow)
      - (backward[i] === -1 ? 0 : residual.edges[backward[i]].flow));
  return { flows, value: start + extra };
}
/** Max flow without lower bounds, used as an independent cross-check for the lo = 0 case. */
function plainMaxFlow(size, source, sink, edges) {
  const network = new Dinic(size);
  for (const edge of edges) network.addEdge(edge.u, edge.v, edge.hi);
  const value = network.maxFlow(source, sink);
  const flows = network.edges.map((edge) => edge.flow);
  return { value, flows };
}
/** Conservation, bound satisfaction and consistency of a reported flow. */
function checkFlow(size, edges, flows, source, sink, value) {
  const net = new Array(size).fill(0);
  for (let i = 0; i < edges.length; i++) {
    if (flows[i] < edges[i].lo || flows[i] > edges[i].hi) return `edge ${i} violates its bounds`;
    net[edges[i].u] -= flows[i];
    net[edges[i].v] += flows[i];
  }
  for (let v = 0; v < size; v++) {
    if (v === source || v === sink) continue;
    if (net[v] !== 0) return `vertex ${v} is imbalanced by ${net[v]}`;
  }
  if (source !== undefined && -net[source] !== value) return `value ${value} != outflow ${-net[source]}`;
  return null;
}
/**
 * Brute-force oracle: enumerate every integer assignment of flows inside the bounds and keep the
 * largest s-t value with perfect conservation. Only usable for tiny ranges, which is intended.
 */
function bruteMaxFlowWithBounds(size, source, sink, edges) {
  const flows = new Array(edges.length).fill(0);
  let best = -1;
  const recurse = (index) => {
    if (index === edges.length) {
      const net = new Array(size).fill(0);
      for (let i = 0; i < edges.length; i++) {
        net[edges[i].u] -= flows[i];
        net[edges[i].v] += flows[i];
      }
      for (let v = 0; v < size; v++) {
        if (v === source || v === sink) continue;
        if (net[v] !== 0) return;
      }
      if (-net[source] < 0) return;
      if (-net[source] !== net[sink]) return;
      if (-net[source] > best) best = -net[source];
      return;
    }
    for (let value = edges[index].lo; value <= edges[index].hi; value++) {
      flows[index] = value;
      recurse(index + 1);
    }
  };
  recurse(0);
  return best < 0 ? -1 : best + 0; // +0 normalises the IEEE negative zero of an empty flow
}

module.exports = {
  makeRng, Dinic, feasibleCirculation, maxFlowWithBounds, plainMaxFlow, checkFlow,
  bruteMaxFlowWithBounds,
};

if (require.main === module) {
  // -- input validation ------------------------------------------------------------------------
  throws(() => new Dinic(2).addEdge(0, 1, -1), RangeError);
  throws(() => feasibleCirculation(2, [{ u: 0, v: 2, lo: 0, hi: 1 }]), RangeError);
  throws(() => feasibleCirculation(2, [{ u: 0, v: 1, lo: 3, hi: 2 }]), RangeError);
  throws(() => feasibleCirculation(2, [{ u: 0, v: 1, lo: -1, hi: 2 }]), RangeError);
  throws(() => maxFlowWithBounds(2, 0, 0, []), RangeError);

  // -- a single edge with a lower bound -------------------------------------------------------
  const oneEdge = maxFlowWithBounds(2, 0, 1, [{ u: 0, v: 1, lo: 2, hi: 5 }]);
  strictEqual(oneEdge.value, 5, "the lower bound does not cap the flow, the upper bound does");
  strictEqual(checkFlow(2, [{ u: 0, v: 1, lo: 2, hi: 5 }], oneEdge.flows, 0, 1, oneEdge.value), null);
  const noPath = maxFlowWithBounds(3, 0, 2, [{ u: 0, v: 1, lo: 1, hi: 3 }]);
  strictEqual(noPath, null, "vertex 2 cannot be reached at all, so no flow exists");

  // -- hand-checked network with a lower bound on the direct edge ------------------------------
  const handEdges = [
    { u: 0, v: 1, lo: 2, hi: 4 },
    { u: 1, v: 2, lo: 1, hi: 3 },
    { u: 0, v: 2, lo: 0, hi: 1 },
  ];
  const hand = maxFlowWithBounds(3, 0, 2, handEdges);
  strictEqual(hand.value, 4, "the middle edge caps the direct edge's contribution");
  strictEqual(checkFlow(3, handEdges, hand.flows, 0, 2, hand.value), null);
  strictEqual(plainMaxFlow(3, 0, 2, handEdges).value, 4, "the bounds do not reduce this instance");
  strictEqual(bruteMaxFlowWithBounds(3, 0, 2, handEdges), 4);

  // -- an infeasible circulation: a lower bound that no flow can satisfy -----------------------
  const infeasible = feasibleCirculation(3, [
    { u: 0, v: 1, lo: 2, hi: 2 },
    { u: 1, v: 2, lo: 2, hi: 2 },
  ]);
  strictEqual(infeasible, null, "vertex 0 has no way to receive its forced outflow");
  const feasible = feasibleCirculation(3, [
    { u: 0, v: 1, lo: 1, hi: 2 },
    { u: 1, v: 2, lo: 1, hi: 2 },
    { u: 2, v: 0, lo: 1, hi: 2 },
  ]);
  ok(feasible !== null && feasible.flows.every((f, i) => f >= 1 && f <= 2), "a 3-cycle circulates");
  const cycleEdges = [
    { u: 0, v: 1, lo: 1, hi: 2 }, { u: 1, v: 2, lo: 1, hi: 2 }, { u: 2, v: 0, lo: 1, hi: 2 },
  ];
  strictEqual(checkFlow(3, cycleEdges, feasible.flows), null, "a circulation must conserve everywhere");

  // -- zero lower bounds must reproduce the plain max flow -------------------------------------
  const rng = makeRng(706);
  let compared = 0;
  for (let trial = 0; trial < 60; trial++) {
    const size = 4 + rng(3);
    const edges = [];
    for (let i = 0; i < 4 + rng(5); i++) {
      const u = rng(size);
      const v = rng(size);
      if (u !== v) edges.push({ u, v, lo: 0, hi: 1 + rng(6) });
    }
    if (edges.length === 0) continue;
    const source = 0;
    const sink = size - 1;
    const withBounds = maxFlowWithBounds(size, source, sink, edges);
    const plain = plainMaxFlow(size, source, sink, edges);
    if (withBounds === null) continue;
    strictEqual(withBounds.value, plain.value, `trial ${trial}: lo=0 must equal Dinic's answer`);
    strictEqual(checkFlow(size, edges, withBounds.flows, source, sink, withBounds.value), null);
    compared++;
  }
  ok(compared > 30, `the lo=0 sweep must compare real instances (${compared})`);

  // -- randomized instances with lower bounds against the exhaustive oracle --------------------
  let bruteCompared = 0;
  for (let trial = 0; trial < 80; trial++) {
    const size = 3 + rng(2);
    const edges = [];
    for (let i = 0; i < 3 + rng(3); i++) {
      const u = rng(size);
      const v = rng(size);
      if (u === v) continue;
      const lo = rng(2);
      edges.push({ u, v, lo, hi: lo + 1 + rng(2) });
    }
    if (edges.length === 0) continue;
    const source = 0;
    const sink = size - 1;
    const optimum = bruteMaxFlowWithBounds(size, source, sink, edges);
    const solved = maxFlowWithBounds(size, source, sink, edges);
    if (optimum < 0) {
      strictEqual(solved, null, `trial ${trial}: the brute force found no feasible flow`);
      continue;
    }
    ok(solved !== null, `trial ${trial}: a feasible flow exists but was not found`);
    strictEqual(solved.value, optimum, `trial ${trial}: value ${solved.value} != optimum ${optimum}`);
    strictEqual(checkFlow(size, edges, solved.flows, source, sink, solved.value), null);
    // value <= sum(hi out of s) - sum(lo into s), the trivial cut bound at the source
    const outOfSource = edges.filter((e) => e.u === source).reduce((sum, e) => sum + e.hi, 0)
      - edges.filter((e) => e.v === source).reduce((sum, e) => sum + e.lo, 0);
    ok(solved.value <= outOfSource, `trial ${trial}: value above the source cut bound`);
    bruteCompared++;
  }
  ok(bruteCompared > 25, `the brute-force sweep must compare real instances (${bruteCompared})`);

  // -- source and sink swapped, and a larger layered instance ----------------------------------
  const layered = [];
  for (let i = 0; i < 8; i++) layered.push({ u: 0, v: 1 + i, lo: 0, hi: 2 });
  for (let i = 0; i < 8; i++) layered.push({ u: 1 + i, v: 9, lo: 1, hi: 3 });
  const layeredFlow = maxFlowWithBounds(10, 0, 9, layered);
  strictEqual(layeredFlow.value, 16, "eight parallel middles of capacity 2");
  strictEqual(checkFlow(10, layered, layeredFlow.flows, 0, 9, layeredFlow.value), null);
  const reversed = maxFlowWithBounds(3, 2, 0, handEdges.map((e) => ({ ...e, u: e.v, v: e.u })));
  strictEqual(reversed.value, 4, "swapping source and sink keeps the value");

  // -- determinism -------------------------------------------------------------------------------
  function replay(seed) {
    const r = makeRng(seed);
    const edges = Array.from({ length: 7 }, () => {
      const lo = r(2);
      return { u: r(4), v: r(4), lo, hi: lo + 1 + r(3) };
    }).filter((e) => e.u !== e.v);
    const solved = maxFlowWithBounds(4, 0, 3, edges);
    return solved === null ? "null" : `${solved.value}:${solved.flows.join(",")}`;
  }
  strictEqual(replay(5), replay(5), "identical seeds must replay identically");

  console.log(`lo=0 comparisons=${compared} brute-force comparisons=${bruteCompared}`);
  console.log(`hand value=${hand.value} layered value=${layeredFlow.value}`);
  console.log("PASS");
}
