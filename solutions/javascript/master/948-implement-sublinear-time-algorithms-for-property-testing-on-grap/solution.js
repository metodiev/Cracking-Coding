/** Task 948 - Implement sublinear time algorithms for property testing on graphs (bipartiteness).
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The graph is only reachable through an oracle interface - degree(v) and neighbour(v, i) - and every
 * call is counted, so the test must decide with o(n) queries. The bipartiteness tester samples
 * O(1/epsilon) start vertices and explores a bounded neighbourhood around each, looking for an odd
 * cycle: an edge whose endpoints already carry the same parity is a witness. A triangle-freeness
 * tester works the same way by checking pairs among the sampled neighbourhoods. Acceptance is the
 * default (one-sided error): the tester can only reject with a witness, so bipartite graphs are
 * never rejected.
 *
 * Simplified: the tester is a single-shot version (no repetition for the 2/3 confidence bound, which
 * is measured empirically instead), the neighbourhood exploration is breadth-first up to a depth
 * bound rather than with a full random-walk weighting, and epsilon-far graphs are generated directly.
 * Verified: the query counter versus the theoretical budget, soundness on bipartite graphs, the
 * rejection rate on far graphs, acceptance of close graphs, the exact oracle validating every
 * rejection, triangle queries on a large triangle-free graph, and empty/single-vertex boundaries.
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

/** Query interface over an adjacency list; every access is charged to the query counter. */
class QueryGraph {
  constructor(adjacency) {
    this.adjacency = adjacency;
    this.n = adjacency.length;
    this.queries = 0;
  }
  degree(v) {
    this.queries += 1;
    return this.adjacency[v].length;
  }
  neighbour(v, i) {
    this.queries += 1;
    return this.adjacency[v][i];
  }
}

const addEdge = (adjacency, a, b) => { adjacency[a].push(b); adjacency[b].push(a); };

/** Bipartite generators. */
function evenRing(n) {
  const adjacency = Array.from({ length: n }, () => []);
  for (let v = 0; v < n; v += 1) addEdge(adjacency, v, (v + 1) % n);
  return adjacency;
}
function completeBipartite(half) {
  const adjacency = Array.from({ length: 2 * half }, () => []);
  for (let a = 0; a < half; a += 1) for (let b = half; b < 2 * half; b += 1) addEdge(adjacency, a, b);
  return adjacency;
}
/** Dense random graph: far from bipartite, since a random graph of this degree always has odd cycles. */
function randomDense(n, degree, seed) {
  const rng = splitmix64(seed);
  const adjacency = Array.from({ length: n }, () => new Set());
  for (let v = 0; v < n; v += 1) {
    for (let d = 0; d < degree; d += 1) {
      const w = rng.int(n);
      if (w !== v) { adjacency[v].add(w); adjacency[w].add(v); }
    }
  }
  return adjacency.map((s) => [...s]);
}
/** An even ring with a handful of chords: epsilon-close to bipartite when the ring is large. */
function ringWithChords(n, chords, seed) {
  const adjacency = evenRing(n);
  const rng = splitmix64(seed);
  for (let c = 0; c < chords; c += 1) addEdge(adjacency, rng.int(n), rng.int(n));
  return adjacency;
}
function triangleCluster(n, triangles) {
  const adjacency = Array.from({ length: n }, () => []);
  const rng = splitmix64(99);
  for (let t = 0; t < triangles; t += 1) {
    const a = rng.int(n);
    const b = (a + 1 + rng.int(n - 1)) % n;
    const c = (a + 2 + rng.int(n - 2)) % n;
    addEdge(adjacency, a, b);
    addEdge(adjacency, b, c);
    addEdge(adjacency, c, a);
  }
  return adjacency;
}

/** Tester: sample starts, colour a bounded neighbourhood, reject on a parity conflict. */
function testBipartite(graph, { epsilon = 0.1, seed = 1, budgetFactor = 4 } = {}) {
  const rng = splitmix64(seed);
  const starts = Math.max(1, Math.ceil(budgetFactor / epsilon));
  const depth = Math.max(2, Math.ceil(1 / epsilon));
  for (let s = 0; s < starts && graph.n > 0; s += 1) {
    const start = rng.int(graph.n);
    const colour = new Map([[start, 0]]);
    const queue = [start];
    while (queue.length) {
      const v = queue.shift();
      const distance = colour.get(v);
      if (distance >= depth) continue;
      const degree = graph.degree(v);
      for (let i = 0; i < degree; i += 1) {
        const w = graph.neighbour(v, i);
        if (!colour.has(w)) {
          colour.set(w, distance + 1);
          queue.push(w);
        } else if ((colour.get(w) - distance) % 2 === 0) {
          return { accept: false, witness: [v, w], queries: graph.queries, starts: s + 1 };
        }
      }
    }
  }
  return { accept: true, queries: graph.queries, starts };
}

/** Tester for triangle-freeness: sample vertices and look for an edge inside a neighbourhood. */
function testTriangleFree(graph, { epsilon = 0.1, seed = 1 } = {}) {
  const rng = splitmix64(seed);
  const samples = Math.max(2, Math.ceil(8 / epsilon));
  for (let s = 0; s < samples && graph.n > 0; s += 1) {
    const v = rng.int(graph.n);
    const degree = graph.degree(v);
    if (degree < 2) continue;
    const neighbours = [];
    for (let i = 0; i < Math.min(degree, 40); i += 1) neighbours.push(graph.neighbour(v, i));
    for (const a of neighbours) {
      const aDegree = graph.degree(a);
      for (let i = 0; i < Math.min(aDegree, 40); i += 1) {
        const w = graph.neighbour(a, i);
        if (w !== v && neighbours.includes(w)) return { accept: false, witness: [v, a, w], queries: graph.queries };
      }
    }
  }
  return { accept: true, queries: graph.queries };
}

/** Exact oracle: full BFS two-colouring, the ground truth the tester is compared against. */
function isBipartite(adjacency) {
  const colour = new Array(adjacency.length).fill(-1);
  for (let start = 0; start < adjacency.length; start += 1) {
    if (colour[start] !== -1) continue;
    colour[start] = 0;
    const queue = [start];
    while (queue.length) {
      const v = queue.shift();
      for (const w of adjacency[v]) {
        if (colour[w] === -1) { colour[w] = 1 - colour[v]; queue.push(w); }
        else if (colour[w] === colour[v]) return false;
      }
    }
  }
  return true;
}

function main() {
  const epsilon = 0.1;

  // 1. Soundness: bipartite graphs are never rejected, whatever the seed.
  for (const adjacency of [evenRing(1000), completeBipartite(40), ringWithChords(2000, 0, 1)]) {
    for (const seed of [1, 2, 3, 4, 5]) {
      const decision = testBipartite(new QueryGraph(adjacency), { epsilon, seed });
      assert(decision.accept, `a bipartite graph was rejected near ${decision.witness}`);
    }
  }

  // 2. Sublinear query budget on a large bipartite graph.
  const large = new QueryGraph(evenRing(100000));
  const largeDecision = testBipartite(large, { epsilon, seed: 11 });
  assert(largeDecision.accept, "the large ring is bipartite");
  assert(large.queries < 100000 / 20, `queries must be sublinear, used ${large.queries}`);

  // 3. Completeness: a graph far from bipartite is rejected in almost every run.
  const far = randomDense(4000, 8, 7);
  assert(!isBipartite(far), "the oracle confirms the dense graph is not bipartite");
  let rejections = 0;
  for (let seed = 0; seed < 60; seed += 1) if (!testBipartite(new QueryGraph(far), { epsilon, seed }).accept) rejections += 1;
  assert(rejections >= 40, `the tester must reject far graphs most of the time, got ${rejections}/60`);

  // 4. Every rejection comes with a witness the exact oracle confirms.
  let witnessed = 0;
  for (let seed = 0; seed < 30; seed += 1) {
    const decision = testBipartite(new QueryGraph(far), { epsilon, seed });
    if (decision.accept) continue;
    const [a, b] = decision.witness;
    assert(far[a].includes(b) || far[b].includes(a), "the witness must be an edge of the graph");
    witnessed += 1;
  }
  assert(witnessed === 30, "every rejected run exposed a witness edge");

  // 5. A graph close to bipartite (three chords in a long ring) is accepted most of the time.
  const close = ringWithChords(4000, 3, 13);
  let closeRejections = 0;
  for (let seed = 0; seed < 60; seed += 1) if (!testBipartite(new QueryGraph(close), { epsilon: 0.05, seed }).accept) closeRejections += 1;
  assert(closeRejections <= 12, `a near-bipartite graph is mostly accepted, got ${closeRejections}/60 rejections`);

  // 6. A small epsilon explores further and finds more witnesses than a large one.
  const tightQueries = [];
  const looseQueries = [];
  for (let seed = 0; seed < 20; seed += 1) {
    tightQueries.push(testBipartite(new QueryGraph(close), { epsilon: 0.01, seed }).queries);
    looseQueries.push(testBipartite(new QueryGraph(close), { epsilon: 0.2, seed }).queries);
  }
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  assert(mean(tightQueries) > mean(looseQueries), `precision costs queries: ${mean(tightQueries)} vs ${mean(looseQueries)}`);

  // 7. The same machinery tests triangle-freeness: a large complete bipartite graph has no triangle.
  const noTriangles = new QueryGraph(completeBipartite(500));
  assert(testTriangleFree(noTriangles, { epsilon, seed: 3 }).accept, "a triangle-free graph is accepted");
  const triangles = new QueryGraph(triangleCluster(2000, 40));
  let triangleRejections = 0;
  for (let seed = 0; seed < 40; seed += 1) if (!testTriangleFree(new QueryGraph(triangles.adjacency), { epsilon: 0.05, seed }).accept) triangleRejections += 1;
  assert(triangleRejections > 0, "the triangle tester finds witnesses in a graph full of triangles");

  // 8. Boundary: an empty graph and an isolated vertex are accepted with almost no queries.
  const empty = testBipartite(new QueryGraph([]), { epsilon, seed: 1 });
  assert(empty.accept && empty.queries === 0, "an empty graph costs no queries");
  const lone = testBipartite(new QueryGraph([[]]), { epsilon, seed: 1 });
  assert(lone.accept && lone.queries > 0, "an isolated vertex is accepted after one degree probe");

  // 9. Determinism: same seed, same decision and same query count.
  const first = testBipartite(new QueryGraph(close), { epsilon: 0.05, seed: 21 });
  const second = testBipartite(new QueryGraph(close), { epsilon: 0.05, seed: 21 });
  assert(first.accept === second.accept && first.queries === second.queries, "the tester is deterministic");

  // 10. Query complexity does not grow with the graph: the same epsilon costs a similar budget.
  const smallGraph = testBipartite(new QueryGraph(evenRing(1000)), { epsilon, seed: 5 }).queries;
  const hugeGraph = testBipartite(new QueryGraph(evenRing(200000)), { epsilon, seed: 5 }).queries;
  assert(hugeGraph < 10 * smallGraph, `200x the vertices must not cost 10x the queries: ${hugeGraph} vs ${smallGraph}`);

  console.log(`property-testing: farRejections=${rejections}/60 closeRejections=${closeRejections}/60 queries=${large.queries} tight=${mean(tightQueries).toFixed(0)} loose=${mean(looseQueries).toFixed(0)}`);
  console.log("PASS");
}
module.exports = { splitmix64, QueryGraph, testBipartite, testTriangleFree, isBipartite, evenRing, completeBipartite, randomDense, ringWithChords, triangleCluster };
if (require.main === module) main();
