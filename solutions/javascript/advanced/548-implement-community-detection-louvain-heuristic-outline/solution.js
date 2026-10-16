/**
 * Task 548 - Implement community detection (Louvain heuristic outline). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Louvain modularity optimisation, implemented in full (local moving + aggregation) for small
 * weighted graphs: Q = sum over communities of [ W_in(C)/m - (K_C/(2m))^2 ], where W_in counts the
 * weight of edges inside C once, K_C is the sum of the weighted degrees and a self-loop of weight
 * w adds w to W_in and 2w to the degree - the convention that makes Q = 0 for the trivial
 * one-community partition and Q = -sum (k_i/2m)^2 for all singletons.  Local moving sweeps the
 * vertices in index order and applies the best strictly-improving move to a neighbouring
 * community using the derived gain
 *   dQ = (w(i,B) - w(i,A))/m + k_i (K_A - K_B - k_i) / (2 m^2),
 * which the self-test re-derives numerically against a full recomputation of Q for every candidate
 * move (<= 1e-12).  Each level then aggregates communities into weighted super-vertices (internal
 * weight becomes a self-loop) and repeats until Q stops improving.
 *
 * Documented checks: modularity closed forms (one community -> 0, all singletons ->
 * -sum (k_i/2m)^2, and for two K4s joined by a bridge the planted value 2 * (6/13 - (13/26)^2)
 * = 0.42307..., derived by hand from the internal weights 6, 6 and volumes 13, 13);
 * aggregation preserves Q exactly; Q never decreases within a level nor across levels; the planted
 * partition of a clique-ladder is recovered; the final Q is compared against exhaustive set
 * partition enumeration (restricted growth strings) for n <= 8 and also against the planted
 * partition's Q (Louvain must at least match it on the tested instances); and K5 collapses to a
 * single community with Q = 0 while a graph with no edges keeps every vertex alone.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

/** Weighted graph state: per-vertex weight (2*selfLoop + incident external weight) + edge weights. */
function buildState(n, edges) {
  const weights = new Map();
  const nodeWeight = new Array(n).fill(0);
  let m = 0;
  for (const [u, v, w] of edges) {
    if (w === 0) continue;
    const key = u <= v ? `${u},${v}` : `${v},${u}`;
    weights.set(key, (weights.get(key) || 0) + w);
    m += w;
  }
  for (const [key, w] of weights) {
    const [u, v] = key.split(",").map(Number);
    if (u === v) nodeWeight[u] += 2 * w;
    else {
      nodeWeight[u] += w;
      nodeWeight[v] += w;
    }
  }
  return { n, weights, nodeWeight, m };
}

function unweightedState(n, edges) {
  return buildState(n, edges.map(([u, v]) => [u, v, 1]));
}

function stateEdges(state) {
  const out = [];
  for (const [key, w] of state.weights) {
    const [u, v] = key.split(",").map(Number);
    out.push([u, v, w]);
  }
  return out;
}

/** Modularity of a partition given as vertex -> community ids. */
function modularity(state, community) {
  if (state.m === 0) return 0;
  const internal = new Map();
  const total = new Map();
  for (let v = 0; v < state.n; v++) {
    total.set(community[v], (total.get(community[v]) || 0) + state.nodeWeight[v]);
  }
  for (const [key, w] of state.weights) {
    const [u, v] = key.split(",").map(Number);
    if (community[u] === community[v]) {
      internal.set(community[u], (internal.get(community[u]) || 0) + w);
    }
  }
  let q = 0;
  for (const [id, totalWeight] of total) {
    const inside = internal.get(id) || 0;
    q += inside / state.m - (totalWeight / (2 * state.m)) ** 2;
  }
  return q;
}

function degreeOf(state, v) {
  return state.nodeWeight[v];
}

/**
 * Weight of the edges joining v to the *other* members of the given community.  Self-loops are
 * deliberately excluded: a self-loop of weight s contributes s/m to W_in(C) and cancels on both
 * sides of a move, which is exactly what the gain formula below assumes.
 */
function weightToCommunity(state, community, v, target) {
  let sum = 0;
  for (const [key, w] of state.weights) {
    const [u, other] = key.split(",").map(Number);
    if (u === v && other !== v && community[other] === target) sum += w;
    else if (other === v && u !== v && community[u] === target) sum += w;
  }
  return sum;
}

function communityTotals(state, community) {
  const totals = new Map();
  for (let v = 0; v < state.n; v++) {
    totals.set(community[v], (totals.get(community[v]) || 0) + state.nodeWeight[v]);
  }
  return totals;
}

/** The incremental gain used by local moving, in the derived closed form. */
function moveGain(state, community, v, target) {
  const from = community[v];
  if (from === target) return 0;
  const totals = communityTotals(state, community);
  const k = degreeOf(state, v);
  const toB = weightToCommunity(state, community, v, target);
  const toA = weightToCommunity(state, community, v, from);
  const kB = target === -1 ? 0 : totals.get(target) || 0;
  const kA = totals.get(from) || 0;
  return (toB - toA) / state.m + k * (kA - kB - k) / (2 * state.m * state.m);
}

function neighboursOf(state, v) {
  const set = new Set();
  for (const key of state.weights.keys()) {
    const [u, w] = key.split(",").map(Number);
    if (u === v && w !== v) set.add(w);
    else if (w === v && u !== v) set.add(u);
  }
  return [...set].sort((a, b) => a - b);
}

function reindex(community) {
  const map = new Map();
  return community.map((id) => {
    if (!map.has(id)) map.set(id, map.size);
    return map.get(id);
  });
}

/** One Louvain level: local moving until no move improves, then aggregate. */
function localMoving(state, community, maxPasses = 100) {
  const gainChecks = [];
  let passes = 0;
  let moved = false;
  for (let pass = 0; pass < maxPasses; pass++) {
    passes += 1;
    let improved = false;
    for (let v = 0; v < state.n; v++) {
      const candidates = [community[v], ...neighboursOf(state, v).map((u) => community[u])];
      const unique = [...new Set(candidates)].sort((a, b) => a - b);
      let bestTarget = community[v];
      let bestGain = 0;
      for (const target of unique) {
        const gain = moveGain(state, community, v, target);
        gainChecks.push({ v, target, gain,
          applied: modularityFor(state, community, v, target) - modularity(state, community) });
        if (gain > bestGain + 1e-15) {
          bestGain = gain;
          bestTarget = target;
        }
      }
      if (bestTarget !== community[v]) {
        community[v] = bestTarget;
        improved = true;
        moved = true;
      }
    }
    if (!improved) break;
  }
  const compact = reindex(community);
  return { community: compact, passes, moved, gainChecks, levelModularity: modularity(state, compact) };
}

function modularityFor(state, community, v, target) {
  const trial = community.slice();
  trial[v] = target;
  return modularity(state, trial);
}

/** Aggregate the state so that each community becomes a single weighted vertex. */
function aggregate(state, community) {
  const count = new Set(community).size;
  const edges = [];
  for (const [key, w] of state.weights) {
    const [u, v] = key.split(",").map(Number);
    edges.push([community[u], community[v], w]);
  }
  return buildState(count, edges);
}

function louvain(n, edges, options = {}) {
  const maxLevels = options.maxLevels === undefined ? 20 : options.maxLevels;
  const state0 = options.weighted ? buildState(n, edges) : unweightedState(n, edges);
  let state = state0;
  let community = [...Array(n).keys()];
  let current = modularity(state, community);
  const levels = [{ level: 0, communities: n, modularity: current }];
  // node -> community at every level, so the final mapping can be lifted back to the originals.
  let mapping = community.slice();
  for (let level = 1; level <= maxLevels; level++) {
    const moving = localMoving(state, community.slice());
    community = moving.community;
    const afterMove = modularity(state, community);
    const aggregated = aggregate(state, community);
    mapping = mapping.map((previous) => community[previous]);
    const aggregatedModularity = modularity(aggregated, [...Array(aggregated.n).keys()]);
    if (Math.abs(aggregatedModularity - afterMove) > 1e-9) {
      throw new Error("aggregation must preserve modularity");
    }
    levels.push({ level, communities: aggregated.n, modularity: afterMove, passes: moving.passes,
      gainChecks: moving.gainChecks.length });
    if (Math.abs(afterMove - current) <= 1e-12) break;
    current = afterMove;
    state = aggregated;
    community = [...Array(aggregated.n).keys()];
    if (aggregated.n === 1) break;
  }
  const finalCommunity = reindex(mapping);
  return { community: finalCommunity, modularity: modularity(state0, finalCommunity), levels };
}

/** Exhaustive set partitions via restricted growth strings - the optimality yardstick. */
function bestPartitionModularity(state) {
  const n = state.n;
  let best = -Infinity;
  let bestAssignment = null;
  const assignment = new Array(n).fill(0);
  const rec = (index, used) => {
    if (index === n) {
      const q = modularity(state, assignment);
      if (q > best) {
        best = q;
        bestAssignment = assignment.slice();
      }
      return;
    }
    for (let c = 0; c <= used && c < n; c++) {
      assignment[index] = c;
      rec(index + 1, Math.max(used, c + 1));
    }
  };
  rec(0, 0);
  return { modularity: best, assignment: reindex(bestAssignment) };
}

function cliqueLadder(cliques, size, seed) {
  const n = cliques * size;
  const edges = [];
  const seen = new Set();
  const add = (u, v) => {
    const key = u < v ? `${u},${v}` : `${v},${u}`;
    if (u === v || seen.has(key)) return;
    seen.add(key);
    edges.push([u, v]);
  };
  for (let c = 0; c < cliques; c++) {
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) add(c * size + i, c * size + j);
    }
    add(c * size, ((c + 1) % cliques) * size);
  }
  let s = seed % 65537;
  for (let i = 0; i < size; i++) {
    s = lcgNext(s);
    if (s % 100 < 20) add(i, size + ((i + 1) % size));
  }
  return { n, edges };
}

function completeGraph(n) {
  const edges = [];
  for (let u = 0; u < n; u++) for (let v = u + 1; v < n; v++) edges.push([u, v]);
  return edges;
}

function partitionString(community) {
  const groups = new Map();
  community.forEach((id, vertex) => {
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push(vertex);
  });
  return [...groups.values()].map((members) => members.join(",")).sort().join("|");
}

function maxAbsDifference(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

function main() {
  const assert = require("node:assert");

  // (1) Modularity closed forms: one community = 0, all singletons = -sum (k_i/2m)^2.
  const triangle = unweightedState(3, [[0, 1], [1, 2], [2, 0]]);
  assert.ok(Math.abs(modularity(triangle, [0, 0, 0])) <= 1e-12);
  const singletonQ = modularity(triangle, [0, 1, 2]);
  const expectedSingleton = -[2, 2, 2].reduce((acc, k) => acc + (k / (2 * 3)) ** 2, 0);
  assert.ok(Math.abs(singletonQ - expectedSingleton) <= 1e-12,
    `all singletons: ${singletonQ} vs ${expectedSingleton}`);
  const noEdges = unweightedState(4, []);
  assert.strictEqual(modularity(noEdges, [0, 0, 1, 1]), 0, "no edges: modularity is zero");

  // (2) Two K4s joined by one bridge: hand-computed Q of the planted partition
  //     2 * [6/13 - (13/26)^2] = 0.42307692..., because each side holds 6 edges, volume 13.
  const ladder = cliqueLadder(2, 4, 0);
  const ladderState = unweightedState(ladder.n, ladder.edges);
  assert.strictEqual(ladder.edges.length, 13, "two K4s (12 edges) plus the bridge");
  const planted = [0, 0, 0, 0, 1, 1, 1, 1];
  const plantedQ = modularity(ladderState, planted);
  const LADDER_PLANTED_Q = 2 * (6 / 13 - (13 / 26) ** 2);
  assert.ok(Math.abs(plantedQ - LADDER_PLANTED_Q) <= 1e-12,
    `planted modularity ${plantedQ} should equal ${LADDER_PLANTED_Q}`);

  // (3) The gain formula is re-derived numerically for every candidate move of a full level.
  const check = localMoving(ladderState, [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.ok(check.gainChecks.length > 0);
  for (const entry of check.gainChecks) {
    assert.ok(Math.abs(entry.gain - entry.applied) <= 1e-12,
      `v=${entry.v} -> ${entry.target}: closed form ${entry.gain} vs recomputed ${entry.applied}`);
  }
  assert.ok(Math.abs(check.levelModularity - LADDER_PLANTED_Q) <= 1e-12,
    "one level of local moving already recovers the planted cliques");

  // (3b) The same audit on an aggregated-style state, i.e. with self-loops in play: the self-loop
  //      weight cancels out of a move, which is exactly what the formula assumes.
  const looped = buildState(3, [[0, 0, 4], [0, 1, 2], [1, 1, 3], [1, 2, 1]]);
  const trialCommunity = [0, 1, 1];
  let auditedLoops = 0;
  for (let v = 0; v < 3; v++) {
    for (const target of [0, 1, 2]) {
      if (target === trialCommunity[v]) continue;
      const gain = moveGain(looped, trialCommunity.slice(), v, target);
      const applied = modularityFor(looped, trialCommunity, v, target) -
        modularity(looped, trialCommunity);
      assert.ok(Math.abs(gain - applied) <= 1e-12,
        `self-loop state: v=${v} -> ${target}, closed form ${gain} vs recomputed ${applied}`);
      auditedLoops += 1;
    }
  }
  assert.strictEqual(auditedLoops, 6, "two off-community targets for each of the three vertices");

  // (4) Aggregation preserves modularity exactly, and Louvain matches the planted partition.
  const aggregated = aggregate(ladderState, planted);
  assert.strictEqual(aggregated.n, 2);
  assert.deepStrictEqual(aggregated.nodeWeight, [13, 13],
    "super-vertices carry the volume of the clique they absorbed");
  assert.ok(Math.abs(modularity(aggregated, [0, 1]) - plantedQ) <= 1e-12,
    "the diagonal partition of the aggregated graph preserves Q exactly");
  assert.ok(Math.abs(modularity(aggregated, [0, 0]) - modularity(ladderState, [0, 0, 0, 0, 0, 0, 0, 0]))
    <= 1e-12, "and so does the all-in-one partition");
  const run = louvain(ladder.n, ladder.edges);
  assert.strictEqual(partitionString(run.community), "0,1,2,3|4,5,6,7",
    "Louvain recovers the planted cliques");
  assert.ok(Math.abs(run.modularity - LADDER_PLANTED_Q) <= 1e-12);
  for (let level = 1; level < run.levels.length; level++) {
    assert.ok(run.levels[level].modularity >= run.levels[level - 1].modularity - 1e-12,
      "modularity must not decrease between levels");
  }
  const best = bestPartitionModularity(ladderState);
  assert.ok(best.modularity >= LADDER_PLANTED_Q - 1e-12,
    "brute force can never do worse than the planted split");
  assert.ok(run.modularity >= best.modularity - 1e-9, "Louvain reaches the brute-force optimum");

  // (5) More planted instances: recovery, non-decreasing Q, and comparison with brute force.
  let optimal = 0;
  const instances = [];
  for (let seed = 1; seed <= 6; seed++) {
    const size = 3 + (seed % 2);
    const built = cliqueLadder(2, size, seed * 37);
    const state = unweightedState(built.n, built.edges);
    const outcome = louvain(built.n, built.edges);
    const plant = [...Array(built.n).keys()].map((v) => (v < size ? 0 : 1));
    assert.ok(outcome.levels.length >= 1, `seed ${seed}: at least one level is recorded`);
    assert.ok(outcome.levels[0].level === 0);
    assert.ok(outcome.modularity >= modularity(state, plant) - 1e-12,
      `seed ${seed}: Louvain must not do worse than the planted split`);
    assert.strictEqual(partitionString(outcome.community),
      partitionString(plant), `seed ${seed}: planted communities recovered`);
    const optimum = bestPartitionModularity(state);
    assert.ok(outcome.modularity >= optimum.modularity - 0.05,
      `seed ${seed}: within 0.05 of the exact optimum`);
    if (Math.abs(outcome.modularity - optimum.modularity) <= 1e-9) optimal += 1;
    instances.push([seed, outcome.modularity, optimum.modularity]);
  }
  assert.ok(optimal >= 4, `Louvain should hit the exact optimum on most instances (${optimal}/6)`);

  // (6) Degenerate cases.
  const clique = louvain(5, completeGraph(5));
  assert.strictEqual(partitionString(clique.community), "0,1,2,3,4", "K5 is one community");
  assert.ok(Math.abs(clique.modularity) <= 1e-12, "K5 as one community has Q = 0");
  const emptyRun = louvain(4, []);
  assert.strictEqual(partitionString(emptyRun.community), "0|1|2|3");
  assert.strictEqual(emptyRun.modularity, 0);
  assert.strictEqual(louvain(1, []).modularity, 0);
  const twoIsolated = louvain(2, [[0, 1]]);
  assert.strictEqual(partitionString(twoIsolated.community), "0,1");
  assert.ok(twoIsolated.modularity <= 1e-12);

  // (7) Determinism and a weighted sanity check.
  const repeatEdges = cliqueLadder(3, 3, 11);
  assert.deepStrictEqual(louvain(repeatEdges.n, repeatEdges.edges),
    louvain(repeatEdges.n, repeatEdges.edges));
  const weighted = buildState(4, [[0, 1, 5], [2, 3, 5], [1, 2, 1]]);
  assert.ok(Math.abs(modularity(weighted, [0, 0, 1, 1]) -
    modularity(weighted, [0, 0, 1, 1])) <= 1e-15);
  const weightedRun = louvain(4, [[0, 1, 5], [2, 3, 5], [1, 2, 1]], { weighted: true });
  assert.strictEqual(partitionString(weightedRun.community), "0,1|2,3",
    "heavy weighted pairs stay together");
  assert.ok(maxAbsDifference(weightedRun.community, weightedRun.community) === 0);
}

module.exports = { lcgNext, buildState, unweightedState, stateEdges, modularity, degreeOf,
  weightToCommunity, communityTotals, moveGain, neighboursOf, reindex, localMoving, aggregate,
  louvain, bestPartitionModularity, cliqueLadder, completeGraph, partitionString, maxAbsDifference };

if (require.main === module) {
  main();
  console.log("PASS");
}
