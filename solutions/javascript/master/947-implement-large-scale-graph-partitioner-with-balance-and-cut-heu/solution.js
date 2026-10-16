/** Task 947 - Implement large-scale graph partitioner with balance and cut heuristics.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A multilevel graph partitioner in the METIS tradition. Coarsening repeatedly merges heavy-edge
 * matched pairs so the graph shrinks by roughly half per level while the cut structure survives; the
 * coarsest graph is partitioned greedily and the assignment is projected back level by level, where a
 * Kernighan-Lin style pass swaps boundary vertices whenever the swap lowers the cut while keeping
 * every part inside the balance budget. A repair step then moves vertices out of overfull parts, so
 * the balance invariant holds after every phase. Simplified: the matching is a single random-order
 * greedy pass, the refinement looks only at boundary vertices, and the balance tolerance is a hard
 * constraint rather than part of the objective. Verified: cut size against an edge-by-edge oracle,
 * the balance invariant after every phase, monotone refinement, the multilevel result against a flat
 * greedy baseline and the exact optimum of a small graph (brute force), k=1/k=n and disconnected
 * boundaries, determinism and the partition invariant itself.
 */
"use strict";
const assert = require("node:assert");
/** splitmix64 - the only source of randomness; identical stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}
/** Undirected graph as a list of Sets, built from a clustered or random edge model. */
function buildGraph({ clusters = 6, clusterSize = 25, internal = 5, bridges = 10, seed = 947 }) {
  const rng = splitmix64(seed);
  const n = clusters * clusterSize;
  const adjacency = Array.from({ length: n }, () => new Set());
  const link = (a, b) => { if (a === b) return; adjacency[a].add(b); adjacency[b].add(a); };
  for (let c = 0; c < clusters; c += 1) {
    for (let i = 0; i < clusterSize; i += 1) {
      for (let d = 0; d < internal; d += 1) link(c * clusterSize + i, c * clusterSize + rng.int(clusterSize));
    }
  }
  for (let b = 0; b < bridges; b += 1) link(rng.int(n), rng.int(n));
  return { n, adjacency };
}
const edgesOf = (graph) => graph.adjacency.flatMap((neighbours, v) => [...neighbours].filter((w) => w > v).map((w) => [v, w]));
/** Oracle: count crossing edges by looking at every edge once. */
function cutSize(graph, assignment) {
  let cut = 0;
  for (const [a, b] of edgesOf(graph)) if (assignment[a] !== assignment[b]) cut += 1;
  return cut;
}
function partSizes(assignment, k) {
  const sizes = new Array(k).fill(0);
  for (const part of assignment) sizes[part] += 1;
  return sizes;
}
const maxPartSize = (n, k, balance) => Math.ceil((1 + balance) * (n / k));
const isBalanced = (graph, assignment, k, balance) => Math.max(...partSizes(assignment, k)) <= maxPartSize(graph.n, k, balance);
const isPartition = (assignment, n, k) => assignment.length === n && assignment.every((p) => Number.isInteger(p) && p >= 0 && p < k);
/** Flat baseline: vertices in a random order, each going to the currently smallest part. */
function greedyPartition(graph, k, seed = 1) {
  const rng = splitmix64(seed);
  const order = Array.from({ length: graph.n }, (_, v) => v);
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = rng.int(i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  const sizes = new Array(k).fill(0);
  const assignment = new Array(graph.n);
  for (const v of order) {
    let best = 0;
    for (let p = 1; p < k; p += 1) if (sizes[p] < sizes[best]) best = p;
    assignment[v] = best;
    sizes[best] += 1;
  }
  return assignment;
}
/** One greedy heavy-edge matching pass; each matched pair becomes a coarse vertex. */
function heavyEdgeMatching(adjacency, seed = 1) {
  const rng = splitmix64(seed);
  const partner = new Map();
  const order = adjacency.map((_, v) => v);
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = rng.int(i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  for (const v of order) {
    if (partner.has(v)) continue;
    for (const w of adjacency[v]) if (!partner.has(w)) { partner.set(v, w); partner.set(w, v); break; }
  }
  return partner;
}
/** Contract matched pairs into super-nodes, keeping the edge set of the coarse graph. */
function coarsen(adjacency, partner) {
  const coarseOf = new Map();
  const coarse = [];
  adjacency.forEach((_, v) => {
    if (coarseOf.has(v)) return;
    coarseOf.set(v, coarse.length);
    if (partner.has(v)) coarseOf.set(partner.get(v), coarse.length);
    coarse.push([]);
  });
  const sets = coarse.map(() => new Set());
  for (const [v, neighbours] of adjacency.entries()) {
    for (const w of neighbours) {
      const a = coarseOf.get(v);
      const b = coarseOf.get(w);
      if (a !== b) sets[a].add(b);
    }
  }
  return { coarseOf, adjacency: sets, n: sets.length };
}
/** Kernighan-Lin style refinement: swap or move boundary vertices while the cut improves. */
function refine(graph, assignment, k, balance, rounds = 8) {
  const current = [...assignment];
  const sizes = partSizes(current, k);
  const cap = maxPartSize(graph.n, k, balance);
  const entryBalanced = isBalanced(graph, current, k, balance);
  let cut = cutSize(graph, current);
  for (let round = 0; round < rounds; round += 1) {
    let improved = false;
    for (let v = 0; v < graph.n; v += 1) {
      const from = current[v];
      const counts = new Array(k).fill(0);
      for (const w of graph.adjacency[v]) counts[current[w]] += 1;
      for (let to = 0; to < k; to += 1) {
        if (to === from) continue;
        if (sizes[to] + 1 > cap) continue;
        const gain = counts[to] - counts[from];
        if (gain <= 0) continue;
        current[v] = to;
        sizes[from] -= 1;
        sizes[to] += 1;
        cut -= gain;
        improved = true;
        break;
      }
    }
    assert(cut >= 0 && cutSize(graph, current) === cut, "the incremental cut matches the recomputed cut");
    assert(!entryBalanced || isBalanced(graph, current, k, balance), "refinement keeps a balanced assignment balanced");
    if (!improved) break;
  }
  return { assignment: current, cut };
}
/** Move vertices out of overfull parts until the balance budget is met. */
function repairBalance(graph, assignment, k, balance) {
  const current = [...assignment];
  const sizes = partSizes(current, k);
  const cap = maxPartSize(graph.n, k, balance);
  for (let v = 0; v < graph.n; v += 1) {
    const from = current[v];
    if (sizes[from] <= cap) continue;
    let best = from;
    for (let p = 0; p < k; p += 1) if (p !== from && sizes[p] < sizes[best]) best = p;
    if (best === from) continue;
    current[v] = best;
    sizes[from] -= 1;
    sizes[best] += 1;
  }
  return current;
}
function multilevelPartition(graph, k, { balance = 0.05, maxLevels = 3, seed = 1 } = {}) {
  const levels = [{ adjacency: graph.adjacency, n: graph.n, partner: null }];
  for (let level = 0; level < maxLevels; level += 1) {
    const top = levels[levels.length - 1];
    if (top.n <= Math.max(2 * k, 6)) break;
    const partner = heavyEdgeMatching(top.adjacency, seed + level);
    const coarse = coarsen(top.adjacency, partner);
    levels.push({ adjacency: coarse.adjacency, n: coarse.n, partner, coarseOf: coarse.coarseOf });
  }
  const top = levels[levels.length - 1];
  let assignment = greedyPartition({ n: top.n, adjacency: top.adjacency }, k, seed);
  let refined = refine({ n: top.n, adjacency: top.adjacency }, assignment, k, balance);
  assignment = refined.assignment;
  for (let level = levels.length - 2; level >= 0; level -= 1) {
    const { coarseOf } = levels[level + 1];
    const projected = Array.from({ length: levels[level].n }, (_, v) => assignment[coarseOf.get(v)]);
    const balanced = repairBalance({ n: levels[level].n }, projected, k, balance);
    const local = refine({ n: levels[level].n, adjacency: levels[level].adjacency }, balanced, k, balance);
    assignment = local.assignment;
    assert(isBalanced({ n: levels[level].n }, assignment, k, balance), `level ${level} is balanced after projection`);
  }
  return { assignment, cut: cutSize(graph, assignment), levels: levels.length };
}
/** Exact oracle for tiny graphs: exhaustive search with the balance budget. */
function bruteForceOptimalCut(graph, k, balance = 1) {
  const cap = maxPartSize(graph.n, k, balance);
  const edges = edgesOf(graph);
  let best = Infinity;
  const assignment = new Array(graph.n).fill(0);
  const explore = (v, sizes) => {
    if (v === graph.n) {
      let cut = 0;
      for (const [a, b] of edges) if (assignment[a] !== assignment[b]) cut += 1;
      best = Math.min(best, cut);
      return;
    }
    for (let p = 0; p < k; p += 1) {
      if (sizes[p] + 1 > cap) continue;
      assignment[v] = p;
      sizes[p] += 1;
      explore(v + 1, sizes);
      sizes[p] -= 1;
    }
    assignment[v] = 0;
  };
  explore(0, new Array(k).fill(0));
  return best;
}
function main() {
  const graph = buildGraph({}); const k = 4;
  const result = multilevelPartition(graph, k, { balance: 0.05 });
  // 1. Partition invariant plus cut accounting against the edge-by-edge oracle.
  assert(isPartition(result.assignment, graph.n, k), "every vertex is assigned exactly one part");
  assert(result.cut === cutSize(graph, result.assignment) && result.cut > 0 && result.cut <= edgesOf(graph).length, `cut ${result.cut} matches the oracle`);
  // 2. Balance: the budget holds, and it is a real restriction at this k.
  const cap = maxPartSize(graph.n, k, 0.05);
  assert(Math.max(...partSizes(result.assignment, k)) <= cap && cap < graph.n / k + 4, `parts fit in ${cap} vertices, just above the ideal share`);
  // 3. Refinement is monotone and the multilevel result beats the flat greedy baseline.
  const flat = greedyPartition(graph, k, 3); const refined = refine(graph, flat, k, 0.05);
  const flatCut = refine(graph, repairBalance(graph, greedyPartition(graph, k, 1), k, 0.05), k, 0.05).cut;
  assert(refined.cut <= cutSize(graph, flat) && isBalanced(graph, refined.assignment, k, 0.05), "refinement lowers the cut and keeps the balance");
  assert(result.cut <= flatCut && result.levels >= 2, `multilevel ${result.cut} beats flat ${flatCut} over ${result.levels} levels`);
  // 4. Exact oracle on a small graph: the heuristic stays close to the optimum.
  const ring = { n: 12, adjacency: Array.from({ length: 12 }, (_, v) => new Set([(v + 1) % 12, (v + 11) % 12])) };
  const ringPartition = multilevelPartition(ring, 2, { balance: 0.2, seed: 5 }); const optimal = bruteForceOptimalCut(ring, 2, 0.2);
  assert(optimal === 2 && ringPartition.cut >= optimal && ringPartition.cut <= optimal * 2, `ring bisection ${ringPartition.cut} vs optimal ${optimal}`);
  // 5. Boundary: k = 1 keeps everything together, k = n isolates every vertex.
  const single = multilevelPartition(graph, 1, { seed: 2 });
  const all = multilevelPartition({ n: 8, adjacency: Array.from({ length: 8 }, () => new Set([0, 1, 2, 3, 4, 5, 6, 7])) }, 8, { seed: 2 });
  assert(single.cut === 0 && partSizes(single.assignment, 1)[0] === graph.n, "one part means no cutting");  assert(new Set(all.assignment).size === 8 && partSizes(all.assignment, 8).every((size) => size === 1), "eight parts isolate the vertices");
  // 6. Boundary: two disconnected cliques split without cutting an edge.
  const clique = (base, size) => Array.from({ length: size }, (_, i) => new Set(Array.from({ length: size }, (_, j) => base + j).filter((w) => w !== base + i)));
  const islands = { n: 12, adjacency: [...clique(0, 6), ...clique(6, 6)] }; const split = multilevelPartition(islands, 2, { balance: 0.05, seed: 7 });
  assert(split.cut === 0 && isBalanced(islands, split.assignment, 2, 0.05), `components split without a cut, got ${split.cut}`);
  // 7. Repair: an unbalanced start is repaired without violating the partition invariant.
  const repaired = repairBalance({ n: 30 }, new Array(30).fill(0), 3, 0.1);
  assert(isPartition(repaired, 30, 3) && Math.max(...partSizes(repaired, 3)) <= Math.ceil(11), "repair balances an unbalanced assignment");
  // 8. Every k from 2 to 6 produces a balanced, correctly accounted partition.
  for (let parts = 2; parts <= 6; parts += 1) {
    const run = multilevelPartition(graph, parts, { balance: 0.1, seed: 11 });
    assert(isBalanced(graph, run.assignment, parts, 0.1) && run.cut === cutSize(graph, run.assignment), `k=${parts} is balanced and accounted`);
  }
  // 9. Determinism: identical inputs give the identical partition and cut.
  const rerun = multilevelPartition(graph, k, { balance: 0.05 });
  assert(JSON.stringify(rerun.assignment) === JSON.stringify(result.assignment) && rerun.cut === result.cut, "partitioning is deterministic");
  console.log(`partitioner: n=${graph.n} edges=${edgesOf(graph).length} k=${k} cut=${result.cut} flat=${flatCut} levels=${result.levels} ring=${ringPartition.cut}/${optimal}`);
  console.log("PASS");
}
module.exports = { splitmix64, buildGraph, edgesOf, cutSize, partSizes, maxPartSize, isBalanced, greedyPartition,
  heavyEdgeMatching, coarsen, refine, repairBalance, multilevelPartition, bruteForceOptimalCut };
if (require.main === module) main();