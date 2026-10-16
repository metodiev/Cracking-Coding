/** Task 746 - Implement topological sort with cycle debugging and minimal feedback arc set
 *  heuristics.  (Tier: Expert, CTCI Ch 4 - Trees and Graphs)
 *
 * Two orderings are produced. Kahn's algorithm peels sources and, when it stalls, reports the
 * actual blocked subgraph; a DFS with colouring then extracts one concrete cycle from that
 * residue ("cycle debugging": the answer is the edge list that closes the loop, which is what a
 * caller can act on, not just "there is a cycle"). When the graph is cyclic, breaking it needs a
 * *feedback arc set*: the code implements two heuristics, a greedy "remove the vertex with the
 * largest out-degree minus in-degree" rule and the Eades-Lin-Smyth GR rule (repeatedly sink the
 * sinks and source the sources, otherwise remove the vertex with the largest out-degree minus
 * in-degree, and finally place the vertices left to right), and one exact solver - a subset DP
 * that maximises the number of edges kept by some linear order, so its complement is a minimum
 * feedback arc set.
 *
 * What is simplified: the heuristics are not the exact optimum in general (they are compared
 * against the DP, and against each other, on the small instances here), self-loops are reported
 * as their own one-vertex cycle, and the DP is exponential and therefore only used for the small
 * graphs the tests build. What is verified: a returned topological order really orients every
 * edge forward (and the count of such orders matches an independent linear-extension DP and, for
 * tiny graphs, a brute-force permutation count); a reported cycle is a walk of existing edges
 * that returns to its start with no repeated vertices; after removing either heuristic's feedback
 * arc set the remainder is acyclic and the set is a subset of the original arcs; the exact DP
 * cost never exceeds a heuristic's and never exceeds a brute-force optimum over all arc subsets
 * for graphs of at most five vertices; and degenerate inputs (empty graph, single vertex,
 * self-loop, duplicated arcs) behave as documented.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

function makeRng(seed) {
  let state = seed >>> 0 || 0x2545f491;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** A directed graph without duplicate arcs: arcs are index pairs [u, v]. */
class Digraph {
  constructor(vertexCount, arcs = []) {
    if (!Number.isInteger(vertexCount) || vertexCount < 0) throw new RangeError("bad vertex count");
    this.vertexCount = vertexCount;
    this.adjacency = Array.from({ length: vertexCount }, () => []);
    this.arcs = [];
    const seen = new Set();
    for (const [u, v] of arcs) {
      if (!Number.isInteger(u) || !Number.isInteger(v) || u < 0 || v < 0 ||
        u >= vertexCount || v >= vertexCount) {
        throw new RangeError(`arc ${u}->${v} is out of range`);
      }
      const key = `${u},${v}`;
      if (seen.has(key)) continue; // duplicate arcs are ignored, documented
      seen.add(key);
      this.arcs.push([u, v]);
      this.adjacency[u].push(v);
    }
    for (const list of this.adjacency) list.sort((a, b) => a - b);
  }

  inDegrees() {
    const degrees = new Array(this.vertexCount).fill(0);
    for (const [, v] of this.arcs) degrees[v]++;
    return degrees;
  }

  /** Kahn's algorithm; on failure the leftover vertices are the blocked (cyclic) part. */
  kahn() {
    const inDegree = this.inDegrees();
    const ready = [];
    for (let v = 0; v < this.vertexCount; v++) if (inDegree[v] === 0) ready.push(v);
    ready.sort((a, b) => a - b);
    const order = [];
    while (ready.length > 0) {
      const u = ready.shift();
      order.push(u);
      for (const v of this.adjacency[u]) {
        inDegree[v]--;
        if (inDegree[v] === 0) {
          ready.push(v);
          ready.sort((a, b) => a - b);
        }
      }
    }
    const blocked = order.length === this.vertexCount ? [] :
      [...Array(this.vertexCount).keys()].filter((v) => !order.includes(v));
    return { order, blocked };
  }

  /** DFS colouring that returns one concrete cycle inside the blocked part. */
  findCycle(blocked = null) {
    const allowed = blocked === null ? null : new Set(blocked);
    const colour = new Array(this.vertexCount).fill(0); // 0 white, 1 grey, 2 black
    const stack = [];
    let found = null;
    const visit = (u) => {
      colour[u] = 1;
      stack.push(u);
      for (const v of this.adjacency[u]) {
        if (allowed !== null && !allowed.has(u) && !allowed.has(v)) continue;
        if (found !== null) break;
        if (colour[v] === 0) {
          if (visit(v)) return true;
        } else if (colour[v] === 1) {
          found = stack.slice(stack.indexOf(v));
          return true;
        }
      }
      stack.pop();
      colour[u] = 2;
      return false;
    };
    for (let v = 0; v < this.vertexCount && found === null; v++) {
      if (colour[v] === 0) visit(v);
    }
    return found;
  }

  isAcyclic() {
    return this.kahn().order.length === this.vertexCount;
  }

  /** Number of topological orders, by the standard subset DP (independent of Kahn). */
  countTopologicalOrders() {
    const n = this.vertexCount;
    const prereqMask = new Array(n).fill(0);
    for (const [u, v] of this.arcs) prereqMask[v] |= 1 << u;
    const size = 1 << n;
    const table = new Array(size).fill(0);
    table[0] = 1;
    for (let mask = 1; mask < size; mask++) {
      for (let v = 0; v < n; v++) {
        if ((mask & (1 << v)) === 0) continue;
        const rest = mask ^ (1 << v);
        if ((prereqMask[v] & ~rest) !== 0) continue; // a prerequisite is still missing
        table[mask] += table[rest];
      }
    }
    return table[size - 1];
  }
}

/** Removes the given arcs (by index into `arcs`) and returns the trimmed arc list. */
function withoutArcs(graph, arcIndices) {
  const drop = new Set(arcIndices);
  return graph.arcs.filter((_, index) => !drop.has(index));
}

function graphFrom(graph, arcList) {
  return new Digraph(graph.vertexCount, arcList);
}

// ---- feedback arc set heuristics -------------------------------------------------------------
/**
 * Greedy: repeatedly remove the vertex maximising out-degree minus in-degree of the current
 * subgraph, appending the result right to left; every arc that ends up violated joins the set.
 */
function greedyFeedbackArcs(graph) {
  const remaining = [...Array(graph.vertexCount).keys()];
  const placed = [];
  while (remaining.length > 0) {
    let pick = 0;
    let bestScore = -Infinity;
    remaining.forEach((v, index) => {
      let out = 0;
      let into = 0;
      for (const w of graph.adjacency[v]) if (remaining.includes(w)) out++;
      for (const w of remaining) if (graph.adjacency[w].includes(v)) into++;
      const score = out - into;
      if (score > bestScore) {
        bestScore = score;
        pick = index;
      }
    });
    const vertex = remaining.splice(pick, 1)[0];
    placed.unshift(vertex);
  }
  return arcsViolating(graph, placed);
}

/** Eades-Lin-Smyth GR: sink/source peeling around a greedy core, then a left-to-right order. */
function grFeedbackArcs(graph) {
  const remaining = new Set([...Array(graph.vertexCount).keys()]);
  const left = [];
  const right = [];
  while (remaining.size > 0) {
    let peeled = true;
    while (peeled) {
      peeled = false;
      for (const v of [...remaining]) {
        const outs = graph.adjacency[v].filter((w) => remaining.has(w));
        if (outs.length === 0) {
          remaining.delete(v);
          right.unshift(v);
          peeled = true;
          continue;
        }
        const ins = [...remaining].filter((w) => graph.adjacency[w].includes(v));
        if (ins.length === 0) {
          remaining.delete(v);
          left.push(v);
          peeled = true;
        }
      }
    }
    if (remaining.size === 0) break;
    let pick = null;
    let bestScore = -Infinity;
    for (const v of [...remaining].sort((a, b) => a - b)) {
      const outs = graph.adjacency[v].filter((w) => remaining.has(w)).length;
      const ins = [...remaining].filter((w) => graph.adjacency[w].includes(v)).length;
      if (outs - ins > bestScore) {
        bestScore = outs - ins;
        pick = v;
      }
    }
    remaining.delete(pick);
    left.push(pick);
  }
  return arcsViolating(graph, left.concat(right));
}

/** Indices of the arcs that point backwards in the given linear order. */
function arcsViolating(graph, order) {
  const position = new Array(graph.vertexCount).fill(0);
  order.forEach((v, index) => {
    position[v] = index;
  });
  const indices = [];
  graph.arcs.forEach(([u, v], index) => {
    if (position[u] > position[v]) indices.push(index);
  });
  return indices;
}

/**
 * Exact minimum feedback arc set: the subset DP maximises the arcs kept by a best linear order,
 * so the arcs left over are a minimum feedback arc set.
 */
function exactFeedbackArcs(graph) {
  const n = graph.vertexCount;
  const size = 1 << n;
  const incoming = new Array(n).fill(0);
  const outgoing = new Array(n).fill(0);
  for (const [u, v] of graph.arcs) {
    outgoing[u] |= 1 << v;
    incoming[v] |= 1 << u;
  }
  const best = new Array(size).fill(-Infinity);
  const choice = new Array(size).fill(-1);
  best[0] = 0;
  for (let mask = 1; mask < size; mask++) {
    for (let v = 0; v < n; v++) {
      if ((mask & (1 << v)) === 0) continue;
      const rest = mask ^ (1 << v);
      if (best[rest] === -Infinity) continue;
      // v goes last in the order for `mask`; arcs from `rest` into v become forward arcs
      let kept = 0;
      let bits = incoming[v] & rest;
      while (bits !== 0) {
        kept++;
        bits &= bits - 1;
      }
      const candidate = best[rest] + kept;
      if (candidate > best[mask]) {
        best[mask] = candidate;
        choice[mask] = v;
      }
    }
  }
  const full = size - 1;
  if (best[full] === -Infinity) return { indices: graph.arcs.map((_, i) => i), cost: graph.arcs.length };
  const order = [];
  let mask = full;
  while (mask !== 0) {
    const v = choice[mask];
    order.unshift(v);
    mask ^= 1 << v;
  }
  const indices = arcsViolating(graph, order);
  return { indices, cost: indices.length, order, kept: best[full] };
}

/** Brute force optimum: remove the smallest subset of arcs that makes the graph acyclic. */
function bruteFeedbackCost(graph) {
  const m = graph.arcs.length;
  for (let size = 0; size <= m; size++) {
    const chosen = [];
    const search = (start) => {
      if (chosen.length === size) {
        return graphFrom(graph, withoutArcs(graph, chosen)).isAcyclic();
      }
      for (let index = start; index < m; index++) {
        chosen.push(index);
        const okFound = search(index + 1);
        chosen.pop();
        if (okFound) return true;
      }
      return false;
    };
    if (search(0)) return size;
  }
  return m;
}

/** Counts topological orders by enumerating permutations - only usable for tiny graphs. */
function bruteCountOrders(graph) {
  const n = graph.vertexCount;
  let count = 0;
  const used = new Array(n).fill(false);
  const order = [];
  const place = () => {
    if (order.length === n) {
      count++;
      return;
    }
    for (let v = 0; v < n; v++) {
      if (used[v]) continue;
      if (!graph.adjacency.every((list, u) => !list.includes(v) || used[u])) continue;
      used[v] = true;
      order.push(v);
      place();
      order.pop();
      used[v] = false;
    }
  };
  place();
  return count;
}

function cycleIsValid(graph, cycle) {
  if (cycle === null) return false;
  if (cycle.length === 1) return graph.adjacency[cycle[0]].includes(cycle[0]);
  if (new Set(cycle).size !== cycle.length) return false;
  for (let i = 0; i < cycle.length; i++) {
    const u = cycle[i];
    const v = cycle[(i + 1) % cycle.length];
    if (!graph.adjacency[u].includes(v)) return false;
  }
  return true;
}

module.exports = { Digraph, greedyFeedbackArcs, grFeedbackArcs, exactFeedbackArcs,
  bruteFeedbackCost, bruteCountOrders, cycleIsValid, arcsViolating };

if (require.main === module) {
  const rng = makeRng(746746);

  // -- degenerate graphs --------------------------------------------------------------------
  const empty = new Digraph(0);
  deepStrictEqual(empty.kahn(), { order: [], blocked: [] });
  strictEqual(empty.countTopologicalOrders(), 1, "the empty order counts once");
  deepStrictEqual(empty.findCycle(), null);
  const single = new Digraph(1);
  deepStrictEqual(single.kahn().order, [0]);
  const selfLoop = new Digraph(2, [[1, 1]]);
  strictEqual(selfLoop.isAcyclic(), false, "a self-loop is a cycle");
  deepStrictEqual(selfLoop.findCycle(), [1], "the self-loop cycle is reported");
  ok(cycleIsValid(selfLoop, [1]), "the reported cycle must be a real cycle");
  const duplicated = new Digraph(3, [[0, 1], [0, 1], [1, 2]]);
  strictEqual(duplicated.arcs.length, 2, "duplicate arcs are collapsed");
  deepStrictEqual(duplicated.kahn().order, [0, 1, 2]);
  throws_helper: {
    let threw = false;
    try {
      new Digraph(2, [[0, 5]]);
    } catch (error) {
      threw = error instanceof RangeError;
    }
    ok(threw, "an out-of-range arc must be rejected");
  }

  // -- DAGs: the order is valid, and the order count matches two independent methods ----------
  const dag = new Digraph(6, [[5, 2], [5, 0], [4, 0], [4, 1], [2, 3], [3, 1]]);
  const dagOrder = dag.kahn().order;
  strictEqual(dagOrder.length, 6, "a DAG has a full topological order");
  const position = new Map(dagOrder.map((v, i) => [v, i]));
  for (const [u, v] of dag.arcs) {
    ok(position.get(u) < position.get(v), `edge ${u}->${v} must point forward in the order`);
  }
  strictEqual(dag.countTopologicalOrders(), bruteCountOrders(dag),
    "the subset DP and the brute-force permutation count must agree");
  ok(dag.countTopologicalOrders() > 1, "this example admits several orders");
  strictEqual(dag.findCycle(), null, "a DAG has no cycle to debug");

  // -- cyclic graphs: cycle debugging ---------------------------------------------------------
  const cyclic = new Digraph(4, [[0, 1], [1, 2], [2, 3], [3, 1]]);
  const blocked = cyclic.kahn().blocked;
  deepStrictEqual(blocked, [1, 2, 3], "the blocked part is exactly the cyclic vertex set");
  const cycle = cyclic.findCycle(blocked);
  strictEqual(cycle.length, 3, "the triangle 1 -> 2 -> 3 -> 1 is reported");
  ok(cycleIsValid(cyclic, cycle), "the reported cycle must be walkable");
  ok(cycle.includes(1), "the cycle must contain the blocked vertices");
  const twoCycle = new Digraph(2, [[0, 1], [1, 0]]);
  ok(cycleIsValid(twoCycle, twoCycle.findCycle()), "a two cycle is debugged as well");
  const multi = new Digraph(5, [[0, 1], [1, 0], [2, 3], [3, 4], [4, 2]]);
  const multiCycle = multi.findCycle();
  ok(cycleIsValid(multi, multiCycle), "with two cycles the debugger still returns a real one");
  ok(multiCycle.length === 2 || multiCycle.length === 3, "either cycle is acceptable");

  // -- feedback arc sets against the exact optimum ---------------------------------------------
  let greedyTotal = 0;
  let grTotal = 0;
  let exactTotal = 0;
  let comparisonCount = 0;
  for (let trial = 0; trial < 60; trial++) {
    const n = 3 + rng(5);
    const arcs = [];
    const density = rng(50) + 10;
    for (let u = 0; u < n; u++) {
      for (let v = 0; v < n; v++) {
        if (u !== v && rng(100) < density) arcs.push([u, v]);
      }
    }
    const graph = new Digraph(n, arcs);
    const greedy = greedyFeedbackArcs(graph);
    const gr = grFeedbackArcs(graph);
    const exact = exactFeedbackArcs(graph);
    for (const [label, indices] of [["greedy", greedy], ["gr", gr]]) {
      const trimmed = graphFrom(graph, withoutArcs(graph, indices));
      ok(trimmed.isAcyclic(), `${label} must make the graph acyclic (trial ${trial})`);
      for (const index of indices) {
        ok(index >= 0 && index < graph.arcs.length, "the arc set must reference real arcs");
      }
      strictEqual(new Set(indices).size, indices.length, "the arc set must not repeat arcs");
    }
    const trimmedExact = graphFrom(graph, withoutArcs(graph, exact.indices));
    ok(trimmedExact.isAcyclic(), "the exact solver must make the graph acyclic");
    strictEqual(exact.cost, exact.indices.length, "the exact cost must match its arc set");
    strictEqual(exact.cost + exact.kept, graph.arcs.length,
      "the arcs kept plus the arcs removed must be all of them");
    ok(greedy.length >= exact.cost, `greedy ${greedy.length} cannot beat the optimum ${exact.cost}`);
    ok(gr.length >= exact.cost, `GR ${gr.length} cannot beat the optimum ${exact.cost}`);
    ok(gr.length <= exact.cost + 2, `GR ${gr.length} must stay close to ${exact.cost}`);
    greedyTotal += greedy.length;
    grTotal += gr.length;
    exactTotal += exact.cost;
    comparisonCount++;
  }
  strictEqual(comparisonCount, 60, "every trial must be compared");
  // the exact solver must also match a brute-force optimum on the smallest instances
  let bruteVerified = 0;
  for (let trial = 0; trial < 40; trial++) {
    const n = 3 + rng(3);
    const arcs = [];
    for (let u = 0; u < n; u++) {
      for (let v = 0; v < n; v++) {
        if (u !== v && rng(100) < 45) arcs.push([u, v]);
      }
    }
    const graph = new Digraph(n, arcs);
    const exact = exactFeedbackArcs(graph);
    strictEqual(exact.cost, bruteFeedbackCost(graph),
      `trial ${trial}: the DP optimum must equal the brute force optimum`);
    bruteVerified++;
  }
  strictEqual(bruteVerified, 40, "every small instance must be brute-force verified");

  // -- a tournament where feedback arcs must exist ---------------------------------------------
  const tournament = new Digraph(5, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [2, 0], [3, 1]]);
  const tournamentExact = exactFeedbackArcs(tournament);
  ok(tournamentExact.cost >= 1, "a cyclic tournament needs at least one removal");
  ok(graphFrom(tournament, withoutArcs(tournament, tournamentExact.indices)).isAcyclic(),
    "the exact set breaks the tournament cycle");
  ok(cycleIsValid(tournament, tournament.findCycle()), "the tournament cycle is debuggable");

  // -- determinism --------------------------------------------------------------------------------
  deepStrictEqual(greedyFeedbackArcs(tournament), greedyFeedbackArcs(tournament));
  deepStrictEqual(grFeedbackArcs(tournament), grFeedbackArcs(tournament));
  deepStrictEqual(exactFeedbackArcs(tournament).indices, exactFeedbackArcs(tournament).indices);
  deepStrictEqual(new Digraph(6, [[5, 2], [5, 0], [4, 0], [4, 1], [2, 3], [3, 1]]).kahn().order,
    dagOrder, "Kahn is deterministic");

  console.log(`kahn_example_orders=${dag.countTopologicalOrders()} blocked_debug=${blocked.length}`);
  console.log(`random_trials=${comparisonCount} greedy=${greedyTotal} gr=${grTotal} exact=${exactTotal}`);
  console.log(`brute_force_checked=${bruteVerified} tournament_fas=${tournamentExact.cost}`);
  console.log("PASS");
}
