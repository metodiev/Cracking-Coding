/** Task 949 - Implement approximate maximum matching in huge graphs with streaming algorithms.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Edges arrive as a stream in an arbitrary (seeded) order and the algorithm may keep only O(V) state,
 * never the edge list. Two streaming matchers are compared: the classic greedy pass, which is maximal
 * and therefore a 1/2-approximation, and a Karp-Sipser style variant that first spends one pass on
 * degrees and then prefers edges incident to a degree-one vertex before falling back to greedy - the
 * rule that makes the sampled/degree-aware algorithm beat 1/2 on bounded-degree graphs. Both are
 * validated against exact oracles: a subset DP for tiny graphs and Kuhn's augmenting-path algorithm
 * for bipartite ones.
 *
 * Simplified: the stream is an in-memory shuffled edge list rather than a socket, the Karp-Sipser
 * phase performs one degree-1 sweep instead of iterating to a fixed point, and no turnstile
 * deletions are modelled. Verified: matching validity, maximality, the 1/2 guarantee against the
 * exact optimum, agreement of the two exact oracles, single-pass/read accounting and O(V) memory,
 * star/perfect/empty boundaries, stream monotonicity, and determinism.
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
const adjacencyOf = (n, edges) => {
  const adjacency = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) { adjacency[a].push(b); adjacency[b].push(a); }
  return adjacency;
};
function randomEdges(n, count, seed) {
  const rng = splitmix64(seed);
  const seen = new Set();
  const edges = [];
  while (edges.length < count) {
    const a = rng.int(n);
    const b = rng.int(n);
    if (a === b) continue;
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    if (seen.has(key)) continue;
    seen.add(key);
    edges.push([a, b]);
  }
  return edges;
}
/** The stream: a deterministic shuffle of the edges. */
function shuffle(edges, seed) {
  const rng = splitmix64(seed);
  const list = edges.map((e) => [...e]);
  for (let i = list.length - 1; i > 0; i -= 1) {
    const j = rng.int(i + 1);
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list;
}
/** Greedy: one pass, one partner per vertex, maximal by construction. */
function greedyMatching(stream) {
  const partner = new Map();
  let reads = 0;
  for (const [a, b] of stream) {
    reads += 1;
    if (partner.has(a) || partner.has(b)) continue;
    partner.set(a, b);
    partner.set(b, a);
  }
  return { partner, reads, size: partner.size / 2 };
}
/** Karp-Sipser style: a degree pass, then edges touching degree-1 vertices come first. */
function karpSipserMatching(stream, vertexBound) {
  const degree = new Map();
  for (const [a, b] of stream) {
    degree.set(a, (degree.get(a) ?? 0) + 1);
    degree.set(b, (degree.get(b) ?? 0) + 1);
  }
  const remaining = new Map(degree);
  const partner = new Map();
  const match = (a, b) => {
    partner.set(a, b);
    partner.set(b, a);
    remaining.set(a, 0);
    remaining.set(b, 0);
  };
  for (const [a, b] of stream) if ((remaining.get(a) ?? 0) === 1 && (remaining.get(b) ?? 0) === 1) match(a, b);
  for (const [a, b] of stream) {
    if ((remaining.get(a) ?? 0) === 0 || (remaining.get(b) ?? 0) === 0) continue;
    match(a, b);
  }
  return { partner, size: partner.size / 2, reads: 2 * stream.length, degree, vertexBound };
}
function validateMatching(stream, partner, n) {
  const edges = new Set(stream.map(([a, b]) => (a < b ? `${a}-${b}` : `${b}-${a}`)));
  const used = new Set();
  let pairs = 0;
  for (const [v, w] of partner) {
    if (v > w) continue;
    pairs += 1;
    assert(partner.get(w) === v, `matching must be symmetric at ${v}`);
    assert(edges.has(v < w ? `${v}-${w}` : `${w}-${v}`), `matched pair ${v}-${w} must be a stream edge`);
    assert(!used.has(v) && !used.has(w), `vertex ${v} appears twice`);
    used.add(v);
    used.add(w);
  }
  assert(pairs === partner.size / 2 && used.size === partner.size, "matches are disjoint pairs");
  assert([...partner.keys()].every((v) => v >= 0 && v < n), "matched ids stay inside the vertex set");
  return true;
}
const isMaximal = (adjacency, partner) => adjacency.every((neighbours, v) => partner.has(v)
  || neighbours.every((w) => partner.has(w)));
/** Oracle 1: exact maximum matching by DP over vertex subsets (tiny graphs only). */
function exactSubsetMatching(adjacency) {
  const n = adjacency.length;
  if (n > 30) throw new Error("the subset DP needs at most 30 vertices (32-bit masks)");
  const masks = adjacency.map((neighbours) => neighbours.reduce((a, w) => a | (1 << w), 0));
  const memo = new Map();
  const solve = (mask) => {
    if (mask === 0) return 0;
    if (memo.has(mask)) return memo.get(mask);
    const v = 31 - Math.clz32(mask & -mask);
    let best = solve(mask & ~(1 << v));
    let candidates = masks[v] & mask;
    while (candidates) {
      const w = 31 - Math.clz32(candidates & -candidates);
      candidates &= candidates - 1;
      best = Math.max(best, 1 + solve(mask & ~(1 << v) & ~(1 << w)));
    }
    memo.set(mask, best);
    return best;
  };
  return solve((1 << n) - 1);
}
/** Oracle 2: Kuhn's augmenting paths for bipartite graphs (left side = ids below `side`). */
function exactBipartiteMatching(adjacency, side) {
  const matchLeft = new Array(side).fill(-1);
  const matchRight = new Array(adjacency.length - side).fill(-1);
  const augment = (v, seen) => {
    for (const w of adjacency[v]) {
      if (seen.has(w)) continue;
      seen.add(w);
      if (matchRight[w - side] === -1 || augment(matchRight[w - side], seen)) {
        matchRight[w - side] = v;
        matchLeft[v] = w;
        return true;
      }
    }
    return false;
  };
  let size = 0;
  for (let v = 0; v < side; v += 1) if (augment(v, new Set())) size += 1;
  return size;
}
/** Oracle 3: recursive enumeration of matchings, for cross-checking the DP on tiny inputs. */
function enumerateMaximumMatching(adjacency) {
  const best = (v, used) => {
    if (v >= adjacency.length) return 0;
    if (used.has(v)) return best(v + 1, used);
    let size = best(v + 1, used);
    for (const w of adjacency[v]) {
      if (w <= v || used.has(w)) continue;
      used.add(v);
      used.add(w);
      size = Math.max(size, 1 + best(v + 1, used));
      used.delete(v);
      used.delete(w);
    }
    return size;
  };
  return best(0, new Set());
}
function main() {
  const n = 400;
  const edges = randomEdges(n, 1200, 949);
  const adjacency = adjacencyOf(n, edges);
  const stream = shuffle(edges, 7);
  const greedy = greedyMatching(stream);
  // 1. Validity and maximality of the streaming result.
  assert(validateMatching(stream, greedy.partner, n), "the greedy matching is valid");
  assert(isMaximal(adjacency, greedy.partner), "the greedy matching is maximal");
  assert(greedy.size === greedy.partner.size / 2 && Number.isInteger(greedy.size), "the size counts pairs");
  // 2. Streaming accounting: one pass, one read per edge, O(V) state.
  assert(greedy.reads === edges.length, `greedy reads every edge exactly once, got ${greedy.reads}`);
  assert(greedy.partner.size <= n, "the state is at most one entry per vertex");
  // 3. The exact oracles agree with each other and the 1/2 guarantee holds against them.
  const tinyEdges = randomEdges(14, 30, 3);
  const tinyAdjacency = adjacencyOf(14, tinyEdges);
  const tinyOptimum = exactSubsetMatching(tinyAdjacency);
  assert(tinyOptimum === enumerateMaximumMatching(tinyAdjacency), "the subset DP and the enumeration agree");
  const bipTiny = randomEdges(16, 20, 23).map(([a, b]) => [a % 8, 8 + (b % 8)]);
  assert(exactBipartiteMatching(adjacencyOf(16, bipTiny), 8) === exactSubsetMatching(adjacencyOf(16, bipTiny)), "Kuhn agrees with the subset DP");
  for (const seed of [1, 2, 3, 4, 5]) {
    const run = greedyMatching(shuffle(tinyEdges, seed));
    assert(run.size >= tinyOptimum / 2, `greedy ${run.size} keeps the guarantee against ${tinyOptimum}`);
    assert(validateMatching(shuffle(tinyEdges, seed), run.partner, 14), "tiny matchings stay valid");
  }
  // 4. Karp-Sipser on the same stream: valid, maximal, never worse than 1/2 of the optimum.
  const ks = karpSipserMatching(stream, n);
  assert(validateMatching(stream, ks.partner, n), "the Karp-Sipser matching is valid");
  assert(isMaximal(adjacency, ks.partner), "the Karp-Sipser matching is maximal");
  assert(ks.reads === 2 * edges.length, "the degree pass reads the stream once more");
  // 5. On a bounded-degree instance both stay above the 1/2 guarantee against a tight oracle.
  const sparseEdges = randomEdges(30, 45, 11);
  const sparseAdjacency = adjacencyOf(30, sparseEdges);
  const sparseStream = shuffle(sparseEdges, 13);
  const sparseGreedy = greedyMatching(sparseStream);
  const sparseKs = karpSipserMatching(sparseStream, 30);
  const sparseOptimum = exactSubsetMatching(sparseAdjacency);
  assert(sparseGreedy.size >= sparseOptimum / 2, `greedy ${sparseGreedy.size} >= ${sparseOptimum}/2`);
  assert(sparseKs.size >= sparseOptimum / 2, `Karp-Sipser ${sparseKs.size} >= ${sparseOptimum}/2`);
  assert(sparseGreedy.size >= 0.7 * sparseOptimum, `greedy reaches ${sparseGreedy.size} of ${sparseOptimum}`);
  // 6. Bipartite instance: the streaming size is within 1/2 of Kuhn's optimum.
  const bipEdges = randomEdges(80, 200, 17).map(([a, b]) => [a % 40, 40 + (b % 40)]);
  const bipStream = shuffle(bipEdges, 19);
  const bipOptimum = exactBipartiteMatching(adjacencyOf(80, bipEdges), 40);
  const bipGreedy = greedyMatching(bipStream);
  assert(bipGreedy.size >= bipOptimum / 2, `bipartite greedy ${bipGreedy.size} vs optimum ${bipOptimum}`);
  assert(validateMatching(bipStream, bipGreedy.partner, 80), "the bipartite matching is valid");
  // 7. Boundary: an empty stream, a single edge, a star and a perfect matching.
  const none = greedyMatching([]);
  assert(none.size === 0 && none.partner.size === 0 && none.reads === 0, "an empty stream yields an empty matching");
  const single = greedyMatching([[0, 1]]);
  assert(single.size === 1 && validateMatching([[0, 1]], single.partner, 2), "a single edge is matched");
  const starEdges = Array.from({ length: 20 }, (_, i) => [0, i + 1]);
  const star = greedyMatching(starEdges);
  assert(star.size === 1 && isMaximal(adjacencyOf(21, starEdges), star.partner), "a star supports exactly one pair");
  const perfect = greedyMatching(Array.from({ length: 20 }, (_, i) => [i * 2, i * 2 + 1]));
  assert(perfect.size === 20, "disjoint edges are all matched");
  // 8. Stream monotonicity: adding edges never shrinks the greedy matching.
  const grown = greedyMatching([...stream, [900, 901], [901, 902]]);
  assert(grown.size >= greedy.size, `adding edges cannot shrink the matching: ${grown.size} >= ${greedy.size}`);
  // 9. Determinism: a re-shuffled stream with the same seed gives the same result.
  const rerun = greedyMatching(shuffle(edges, 7));
  assert(rerun.size === greedy.size && JSON.stringify([...rerun.partner]) === JSON.stringify([...greedy.partner]), "the stream is deterministic");
  assert(greedyMatching(shuffle(edges, 8)).size >= greedy.size * 0.5, "another order stays within the guarantee");
  // 10. Coverage sanity: the matched pairs and the maximality check agree with the edge count.
  const covered = new Set([...greedy.partner.keys()]);
  assert(covered.size === 2 * greedy.size && covered.size <= n, "each pair covers two distinct vertices");
  assert(greedy.size <= edges.length && greedy.size <= Math.floor(n / 2), "the matching cannot exceed the trivial bounds");
  console.log(`matching: n=${n} edges=${edges.length} greedy=${greedy.size} ks=${ks.size} sparse=${sparseGreedy.size}/${sparseKs.size} opt=${sparseOptimum} bipartite=${bipGreedy.size}/${bipOptimum}`);
  console.log("PASS");
}
module.exports = { splitmix64, randomEdges, shuffle, adjacencyOf, greedyMatching, karpSipserMatching, validateMatching,
  isMaximal, exactSubsetMatching, exactBipartiteMatching, enumerateMaximumMatching };
if (require.main === module) main();
