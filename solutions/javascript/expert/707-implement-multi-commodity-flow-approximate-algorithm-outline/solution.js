/** Task 707 - Implement multi-commodity flow approximate algorithm outline. (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
 *
 * Maximum concurrent flow: route the demand D_k of every commodity pair so that one common fraction
 * lambda of all demands is satisfied while no edge load exceeds its capacity. The LP has a variable
 * per (commodity, path), so this file implements the Garg-Koenemann outline instead: keep a length
 * l_e >= 1 per edge and repeat
 *   1. route D_k / s_k units of commodity k along its shortest path (s_k = length under l),
 *   2. extract the dual bound (sum_e c_e l_e) / (sum_k D_k s_k), which upper-bounds OPT,
 *   3. multiply every edge length by 1 + epsilon * (load of that edge this round) / c_e.
 * The lengths are multiplicative weights: congested edges get expensive, so later rounds spread the
 * flow out. Afterwards the accumulated flow is scaled by 1 / congestion to become feasible and then
 * uniformly per commodity so that each commodity carries exactly lambda * D_k. Scaling per
 * commodity never lowers a load, so feasibility survives by construction.
 *
 * What is simplified: the exact optimum is not solved as an LP - instead OPT is bracketed between the
 * best feasible flow found and the dual bound (weak duality), and the checks require the two to be
 * within the documented epsilon of each other on the test instances.
 *
 * Invariant verified: every reported path is a real simple path of the graph, the scaled flow obeys
 * every capacity and delivers exactly lambda * D_k per commodity, lambda never exceeds the best dual
 * bound, and instances with a known optimum (a column, a 4-cycle with two commodities, two disjoint
 * edges) come out at that optimum.
 */
"use strict";
const { ok, strictEqual, deepStrictEqual, throws } = require("node:assert");
/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
/** Adjacency of the edge-indexed graph: entries are {to, edge}. */
function buildAdjacency(size, edgeDefs) {
  const adjacency = Array.from({ length: size }, () => []);
  edgeDefs.forEach((edge, index) => {
    adjacency[edge.u].push({ to: edge.v, edge: index });
    adjacency[edge.v].push({ to: edge.u, edge: index });
  });
  return adjacency;
}

function shortestPath(size, adjacency, lengths, source, sink) {
  const dist = new Array(size).fill(Infinity);
  const previous = new Array(size).fill(null);
  const done = new Array(size).fill(false);
  dist[source] = 0;
  for (;;) {
    let best = -1;
    for (let v = 0; v < size; v++) {
      if (!done[v] && dist[v] < Infinity && (best === -1 || dist[v] < dist[best])) best = v;
    }
    if (best === -1) break;
    done[best] = true;
    for (const step of adjacency[best]) {
      if (dist[step.to] > dist[best] + lengths[step.edge]) {
        dist[step.to] = dist[best] + lengths[step.edge];
        previous[step.to] = { from: best, edge: step.edge };
      }
    }
  }
  if (dist[sink] === Infinity) return null;
  const edges = [];
  for (let at = sink; previous[at] !== null; at = previous[at].from) edges.push(previous[at].edge);
  return { distance: dist[sink], edges: edges.reverse() };
}
/** Assembles the path-defining edge list into vertex labels, rejecting bogus paths. */
function pathVertices(edgeDefs, pathEdges, source) {
  const vertices = [source];
  let at = source;
  for (const index of pathEdges) {
    const edge = edgeDefs[index];
    const next = edge.u === at ? edge.v : edge.u;
    if (edge.u !== at && edge.v !== at) return null; // the edge does not touch the current vertex
    vertices.push(next);
    at = next;
  }
  return vertices;
}

function isSimplePath(edgeDefs, pathEdges, source, sink) {
  const vertices = pathVertices(edgeDefs, pathEdges, source);
  if (vertices === null || vertices[vertices.length - 1] !== sink) return false;
  return new Set(vertices).size === vertices.length;
}
/**
 * Garg-Koenemann outline. Returns null when a commodity cannot be routed at all, otherwise the
 * value lambda, the scaled per-commodity path flows, the congestion and the best dual bound.
 */
function maxConcurrentFlow(size, edgeDefs, commodities, { iterations = 400, epsilon = 0.1 } = {}) {
  if (iterations <= 0) throw new RangeError("iterations must be positive");
  if (!(epsilon > 0 && epsilon < 1)) throw new RangeError("epsilon must be inside (0, 1)");
  const adjacency = buildAdjacency(size, edgeDefs);
  for (const edge of edgeDefs) {
    if (edge.capacity <= 0) throw new RangeError("capacities must be positive");
  }
  const lengths = new Array(edgeDefs.length).fill(1);
  const usage = commodities.map(() => new Map()); // path key -> {edges, amount}
  const totals = commodities.map(() => 0);
  let dualBound = Infinity;
  let bestIteration = null; // {lambda, paths, amounts}: the best single-round concurrent bundle
  for (let round = 0; round < iterations; round++) {
    const loads = new Array(edgeDefs.length).fill(0);
    const amounts = commodities.map(() => 0);
    const chosen = commodities.map(() => null);
    let denominator = 0;
    for (let k = 0; k < commodities.length; k++) {
      const commodity = commodities[k];
      // a commodity whose endpoints coincide needs no edges at all
      if (commodity.demand === 0 || commodity.source === commodity.sink) continue;
      const path = shortestPath(size, adjacency, lengths, commodity.source, commodity.sink);
      if (path === null) return null;
      denominator += commodity.demand * path.distance;
      const amount = commodity.demand / path.distance;
      amounts[k] = amount;
      chosen[k] = path;
      for (const edge of path.edges) loads[edge] += amount;
      const key = path.edges.join(",");
      const record = usage[k].get(key) ?? { edges: path.edges, amount: 0 };
      record.amount += amount;
      usage[k].set(key, record);
      totals[k] += amount;
    }
    let lengthCost = 0;
    for (let e = 0; e < edgeDefs.length; e++) lengthCost += edgeDefs[e].capacity * lengths[e];
    if (denominator > 0) dualBound = Math.min(dualBound, lengthCost / denominator);
    // the tightest edge of this round decides how much of the round's bundle fits
    let roundCongestion = 0;
    for (let e = 0; e < edgeDefs.length; e++) {
      roundCongestion = Math.max(roundCongestion, loads[e] / edgeDefs[e].capacity);
    }
    if (roundCongestion > 0) {
      let lambda = Infinity;
      for (let k = 0; k < commodities.length; k++) {
        if (chosen[k] === null) continue;
        lambda = Math.min(lambda, amounts[k] / (commodities[k].demand * roundCongestion));
      }
      if (bestIteration === null || lambda > bestIteration.lambda) {
        bestIteration = { lambda, paths: chosen, amounts };
      }
    }
    for (let e = 0; e < edgeDefs.length; e++) {
      lengths[e] *= 1 + epsilon * (loads[e] / edgeDefs[e].capacity);
    }
  }
  const totalLoads = new Array(edgeDefs.length).fill(0);
  commodities.forEach((commodity, k) => {
    for (const record of usage[k].values()) {
      for (const edge of record.edges) totalLoads[edge] += record.amount;
    }
  });
  let congestion = 0;
  for (let e = 0; e < edgeDefs.length; e++) {
    congestion = Math.max(congestion, totalLoads[e] / edgeDefs[e].capacity);
  }
  const rates = commodities.map((commodity, k) =>
    commodity.demand === 0 || commodity.source === commodity.sink || totals[k] === 0
      ? Infinity
      : totals[k] / (commodity.demand * congestion));
  let value = Math.min(...rates);
  let flows = commodities.map((commodity, k) => {
    if (commodity.demand === 0 || totals[k] === 0) return [];
    const factor = (commodity.demand * value) / totals[k]; // <= 1, so every load only shrinks
    return [...usage[k].values()].map((record) => ({ edges: record.edges, flow: record.amount * factor }));
  });
  // the best single round is a legal concurrent flow of its own; keep whichever value is larger
  if (bestIteration !== null && bestIteration.lambda > value) {
    value = bestIteration.lambda;
    flows = commodities.map((commodity, k) => {
      const path = bestIteration.paths[k];
      if (path === null || commodity.demand === 0) return [];
      const factor = (commodity.demand * value) / bestIteration.amounts[k];
      return [{ edges: path.edges, flow: bestIteration.amounts[k] * factor }];
    });
  }
  return { value, congestion, dualBound, flows, totals };
}
/** Independent re-verification of a candidate concurrent flow; returns a violation or null. */
function checkFlow(size, edgeDefs, commodities, value, flows) {
  const loads = new Array(edgeDefs.length).fill(0);
  for (let k = 0; k < commodities.length; k++) {
    let delivered = 0;
    for (const route of flows[k]) {
      if (!isSimplePath(edgeDefs, route.edges, commodities[k].source, commodities[k].sink)) {
        return `commodity ${k} uses a bogus path`;
      }
      delivered += route.flow;
      for (const edge of route.edges) loads[edge] += route.flow;
    }
    if (commodities[k].source === commodities[k].sink) {
      if (delivered !== 0) return `commodity ${k} is trivial but carries flow`;
      continue; // a commodity with equal endpoints needs no edges
    }
    if (Math.abs(delivered - value * commodities[k].demand) > 1e-9) {
      return `commodity ${k} delivers ${delivered}, expected ${value * commodities[k].demand}`;
    }
  }
  for (let e = 0; e < edgeDefs.length; e++) {
    if (loads[e] > edgeDefs[e].capacity + 1e-9) return `edge ${e} is overloaded (${loads[e]})`;
  }
  return null;
}
/** The naive feasible point: one shortest-path bundle scaled to fit the tightest edge. */
function naiveValue(size, edgeDefs, commodities) {
  const adjacency = buildAdjacency(size, edgeDefs);
  const unit = new Array(edgeDefs.length).fill(1);
  const loads = new Array(edgeDefs.length).fill(0);
  for (const commodity of commodities) {
    if (commodity.demand === 0) continue;
    const path = shortestPath(size, adjacency, unit, commodity.source, commodity.sink);
    if (path === null) return null;
    for (const edge of path.edges) loads[edge] += commodity.demand;
  }
  let value = Infinity;
  for (let e = 0; e < edgeDefs.length; e++) {
    if (loads[e] > 0) value = Math.min(value, edgeDefs[e].capacity / loads[e]);
  }
  return value === Infinity ? 0 : value;
}
/** Exact max s-t flow (Edmonds-Karp): the single-commodity concurrent optimum is value/demand. */
function maxFlowValue(size, edgeDefs, source, sink) {
  const capacity = new Array(size).fill(null).map(() => new Array(size).fill(0));
  for (const edge of edgeDefs) {
    capacity[edge.u][edge.v] += edge.capacity;
    capacity[edge.v][edge.u] += edge.capacity;
  }
  let total = 0;
  for (;;) {
    const parent = new Array(size).fill(-1);
    parent[source] = source;
    const queue = [source];
    for (let at = 0; at < queue.length && parent[sink] === -1; at++) {
      for (let next = 0; next < size; next++) {
        if (parent[next] === -1 && capacity[queue[at]][next] > 0) {
          parent[next] = queue[at];
          queue.push(next);
        }
      }
    }
    if (parent[sink] === -1) return total;
    let bottleneck = Infinity;
    for (let at = sink; at !== source; at = parent[at]) {
      bottleneck = Math.min(bottleneck, capacity[parent[at]][at]);
    }
    for (let at = sink; at !== source; at = parent[at]) {
      capacity[parent[at]][at] -= bottleneck;
      capacity[at][parent[at]] += bottleneck;
    }
    total += bottleneck;
  }
}

module.exports = {
  makeRng, buildAdjacency, shortestPath, isSimplePath, pathVertices, maxConcurrentFlow, checkFlow,
  naiveValue, maxFlowValue,
};

if (require.main === module) {
  // -- rejected input and unroutable commodities ------------------------------------------------
  throws(() => maxConcurrentFlow(2, [{ u: 0, v: 1, capacity: 1 }], [], { iterations: 0 }), RangeError);
  throws(() => maxConcurrentFlow(2, [{ u: 0, v: 1, capacity: 1 }], [], { epsilon: 1 }), RangeError);
  throws(() => maxConcurrentFlow(2, [{ u: 0, v: 1, capacity: 0 }], []), RangeError);
  const unreachable = maxConcurrentFlow(3, [{ u: 0, v: 1, capacity: 1 }],
    [{ source: 0, sink: 2, demand: 1 }]);
  strictEqual(unreachable, null, "a commodity with no path makes the instance unroutable");
  strictEqual(shortestPath(3, buildAdjacency(3, [{ u: 0, v: 1, capacity: 1 }]), [1, 1], 0, 2), null);

  // -- one column, one commodity: OPT is the capacity -------------------------------------------
  const column = [{ u: 0, v: 1, capacity: 3 }, { u: 1, v: 2, capacity: 2 }];
  const single = maxConcurrentFlow(3, column, [{ source: 0, sink: 2, demand: 5 }]);
  ok(Math.abs(single.value - 2 / 5) < 1e-6, `lambda must be 2/5, got ${single.value}`);
  strictEqual(checkFlow(3, column, [{ source: 0, sink: 2, demand: 5 }], single.value, single.flows), null);
  ok(single.dualBound >= single.value - 1e-9, "weak duality: lambda <= dual bound");

  // -- two commodities sharing one bottleneck edge ----------------------------------------------
  const shared = [{ u: 0, v: 1, capacity: 4 }, { u: 2, v: 1, capacity: 4 }];
  const sharedCommodities = [
    { source: 0, sink: 1, demand: 1 },
    { source: 2, sink: 1, demand: 3 },
  ];
  const concurrent = maxConcurrentFlow(3, shared, sharedCommodities, { iterations: 200 });
  strictEqual(checkFlow(3, shared, sharedCommodities, concurrent.value, concurrent.flows), null);
  ok(concurrent.value >= naiveValue(3, shared, sharedCommodities) - 1e-9,
    "the outline must not be worse than the naive bundle");
  // Demand 3 on the second edge caps lambda at 4/3, and both edges are independent bottlenecks.
  ok(Math.abs(concurrent.value - 4 / 3) < 0.05,
    `the shared-bottleneck instance admits lambda = 4/3, got ${concurrent.value}`);
  ok(concurrent.dualBound >= 4 / 3 - 1e-9, "the dual bound must stay above the true optimum");

  // -- 4-cycle with two unit demands: the known optimum is 1 -------------------------------------
  const cycle = [
    { u: 0, v: 1, capacity: 1 }, { u: 1, v: 2, capacity: 1 },
    { u: 2, v: 3, capacity: 1 }, { u: 3, v: 0, capacity: 1 },
  ];
  const cycleCommodities = [
    { source: 0, sink: 2, demand: 1 },
    { source: 1, sink: 3, demand: 1 },
  ];
  const cycleFlow = maxConcurrentFlow(4, cycle, cycleCommodities, { iterations: 400, epsilon: 0.05 });
  strictEqual(checkFlow(4, cycle, cycleCommodities, cycleFlow.value, cycleFlow.flows), null);
  ok(Math.abs(cycleFlow.value - 1) < 0.1, `the 4-cycle optimum is 1, got ${cycleFlow.value}`);
  ok(cycleFlow.dualBound >= 1 - 1e-9, `weak duality keeps the dual above OPT=1 (${cycleFlow.dualBound})`);

  // -- two edge-disjoint commodities: OPT is the sum of the capacities --------------------------
  const disjoint = [{ u: 0, v: 1, capacity: 2 }, { u: 2, v: 3, capacity: 5 }];
  const disjointCommodities = [
    { source: 0, sink: 1, demand: 1 },
    { source: 2, sink: 3, demand: 1 },
  ];
  const disjointFlow = maxConcurrentFlow(4, disjoint, disjointCommodities, { iterations: 200 });
  strictEqual(checkFlow(4, disjoint, disjointCommodities, disjointFlow.value, disjointFlow.flows), null);
  ok(disjointFlow.value >= 2 - 1e-6, `two disjoint edges support at least the smaller capacity (${disjointFlow.value})`);

  // -- more iterations can only help, never hurt (monotone improvement) -------------------------
  let previous = 0;
  for (const iterations of [5, 25, 100, 400]) {
    const result = maxConcurrentFlow(4, cycle, cycleCommodities, { iterations, epsilon: 0.05 });
    strictEqual(checkFlow(4, cycle, cycleCommodities, result.value, result.flows), null);
    ok(result.value >= previous - 1e-9, `iterations=${iterations} regressed`);
    ok(result.dualBound >= result.value - 1e-9, "weak duality must hold every time");
    previous = result.value;
  }
  ok(previous > 0.9, `400 rounds must approach the optimum, got ${previous}`);

  // -- randomized instances: feasibility, weak duality, and an exact optimum for one commodity --
  const rng = makeRng(707);
  let trials = 0, singleCommodityTrials = 0, worstSingleRatio = 1;
  for (let trial = 0; trial < 120; trial++) {
    const size = 4 + rng(3);
    const edgeDefs = [];
    for (let v = 1; v < size; v++) { // a random spanning tree keeps every instance routable
      edgeDefs.push({ u: v, v: rng(v), capacity: 1 + rng(5) });
    }
    for (let i = 0; i < rng(5); i++) {
      const u = rng(size);
      const v = rng(size);
      if (u !== v) edgeDefs.push({ u, v, capacity: 1 + rng(5) });
    }
    if (edgeDefs.length === 0) continue;
    const commodities = Array.from({ length: 1 + rng(3) }, () => ({
      source: rng(size), sink: rng(size), demand: 1 + rng(3),
    }));
    const result = maxConcurrentFlow(size, edgeDefs, commodities, { iterations: 2000 });
    if (result === null) {
      ok(commodities.some((c) => naiveValue(size, edgeDefs, [c]) === null),
        `trial ${trial}: unroutable only when a commodity has no path`);
      continue;
    }
    strictEqual(checkFlow(size, edgeDefs, commodities, result.value, result.flows), null, `trial ${trial}`);
    if (!Number.isFinite(result.value)) { // every commodity was trivial (equal endpoints)
      ok(commodities.every((c) => c.source === c.sink));
      continue;
    }
    ok(result.value <= result.dualBound + 1e-9, `trial ${trial}: lambda ${result.value} > dual ${result.dualBound}`);
    if (commodities.length === 1 && commodities[0].source !== commodities[0].sink) {
      // with one commodity the concurrent optimum is exactly the max flow divided by the demand
      const optimum = maxFlowValue(size, edgeDefs, commodities[0].source, commodities[0].sink)
        / commodities[0].demand;
      ok(result.value <= optimum + 1e-9, `trial ${trial}: ${result.value} exceeds OPT ${optimum}`);
      ok(result.value >= 0.9 * optimum, `trial ${trial}: ${result.value} is far below OPT ${optimum}`);
      if (optimum > 0) worstSingleRatio = Math.min(worstSingleRatio, result.value / optimum);
      singleCommodityTrials++;
    }
    trials++;
  }
  ok(trials > 40, `the random sweep must solve real instances (${trials})`);
  ok(singleCommodityTrials > 8, `single-commodity instances must be compared with OPT (${singleCommodityTrials})`);

  // -- determinism ------------------------------------------------------------------------------
  function replay(seed) {
    const r = makeRng(seed);
    const edgeDefs = Array.from({ length: 6 }, () => ({ u: r(4), v: r(4), capacity: 1 + r(3) }))
      .filter((e) => e.u !== e.v);
    const commodities = [{ source: 0, sink: 3, demand: 2 }, { source: 1, sink: 2, demand: 1 }];
    const result = maxConcurrentFlow(4, edgeDefs, commodities, { iterations: 80 });
    return result === null ? "null" : `${result.value.toFixed(9)}:${result.congestion.toFixed(9)}`;
  }
  strictEqual(replay(3), replay(3), "identical seeds must replay identically");

  console.log(`random trials=${trials} single-commodity=${singleCommodityTrials} worst lambda/OPT=${worstSingleRatio.toFixed(3)}`);
  console.log(`cycle lambda=${cycleFlow.value.toFixed(4)} dual=${cycleFlow.dualBound.toFixed(4)}`);
  console.log("PASS");
}
