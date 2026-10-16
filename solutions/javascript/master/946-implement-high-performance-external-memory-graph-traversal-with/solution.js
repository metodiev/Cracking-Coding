/** Task 946 - Implement high-performance external-memory graph traversal with partitioning.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The graph lives on a simulated disk: vertices are clustered into fixed-size pages and a vertex's
 * adjacency list can only be inspected by reading its page. Traversal is level-synchronous BFS -
 * read the frontier's pages, expand, build the next frontier - which is what an external-memory BFS
 * does, and the simulated device counts page transfers (cache misses) plus seeks (a read that is not
 * sequential with the previous one). Three levers are compared: how vertices are assigned to pages
 * (random versus clustered by BFS order), whether the frontier's pages are read in sorted order or
 * scattered, and how big the page cache is.
 *
 * Simplified: pages are in-process arrays rather than files, the device charges one unit per transfer
 * plus one per seek, and the graph is small enough to also traverse in memory. Verified: the
 * traversal distances against an in-memory BFS oracle, the I/O advantage of a clustered partition
 * and of sorted reads, cache monotonicity, an I/O bound in pages and vertices, the locality metric,
 * hub/empty/single-vertex boundaries, disconnected components and determinism.
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

/** A graph with cluster structure: dense inside clusters, a few bridges between them. */
function clusteredGraph({ clusters = 8, clusterSize = 60, internal = 6, bridges = 12, seed = 946 }) {
  const rng = splitmix64(seed);
  const graph = new Map();
  const n = clusters * clusterSize;
  for (let v = 0; v < n; v += 1) graph.set(v, new Set());
  const link = (a, b) => {
    if (a === b) return;
    graph.get(a).add(b);
    graph.get(b).add(a);
  };
  for (let c = 0; c < clusters; c += 1) {
    for (let i = 0; i < clusterSize; i += 1) {
      for (let d = 0; d < internal; d += 1) {
        link(c * clusterSize + i, c * clusterSize + rng.int(clusterSize));
      }
    }
  }
  for (let b = 0; b < bridges; b += 1) link(rng.int(n), rng.int(n));
  return { n, graph, clusters, clusterSize };
}

/** Random page assignment: no locality. */
function randomPartition(n, pageSize, seed = 1) {
  const rng = splitmix64(seed);
  return Array.from({ length: n }, () => Math.floor(rng.int(Math.ceil(n / pageSize))));
}

/** Clustered assignment: a depth-first order keeps a cluster's vertices on neighbouring pages. */
function clusteredPartition(graph, pageSize, n) {
  const order = [];
  const seen = new Set();
  for (let start = 0; start < n; start += 1) {
    if (seen.has(start)) continue;
    const stack = [start];
    while (stack.length) {
      const v = stack.pop();
      if (seen.has(v)) continue;
      seen.add(v);
      order.push(v);
      for (const w of graph.get(v)) if (!seen.has(w)) stack.push(w);
    }
  }
  const page = new Array(n);
  order.forEach((v, i) => { page[v] = Math.floor(i / pageSize); });
  return page;
}

const readPage = (state, page) => {
  if (state.cache.has(page)) {
    state.cache.delete(page);
    state.cache.set(page, 1);
    return;
  }
  state.transfers += 1;
  if (state.lastPage !== page) state.seeks += 1;
  state.lastPage = page;
  state.cache.set(page, 1);
  if (state.cache.size > state.cachePages) state.cache.delete(state.cache.keys().next().value);
};

/** Level-synchronous external BFS; `sorted` reads each frontier's pages in ascending order. */
function externalBfs({ graph, n }, pageOf, { cachePages = 4, sorted = true, batch = true, seed = 1, source = 0 } = {}) {
  const rng = splitmix64(seed);
  const dist = new Map();
  let frontier = graph.has(source) ? [source] : [];
  if (frontier.length) dist.set(source, 0);
  const state = { cache: new Map(), transfers: 0, seeks: 0, lastPage: -1, cachePages };
  while (frontier.length) {
    if (batch) {
      let pages = [...new Set(frontier.map((v) => pageOf[v]))];
      pages = sorted ? pages.sort((a, b) => a - b) : pages.map((p) => ({ p, k: rng.int(1 << 20) })).sort((a, b) => a.k - b.k).map((x) => x.p);
      for (const page of pages) readPage(state, page);
    } else {
      for (const v of frontier) readPage(state, pageOf[v]);
    }
    const next = [];
    for (const v of frontier) {
      for (const w of graph.get(v)) {
        if (dist.has(w)) continue;
        dist.set(w, dist.get(v) + 1);
        next.push(w);
      }
    }
    frontier = next;
  }
  return { dist, transfers: state.transfers, seeks: state.seeks, pages: new Set(pageOf).size };
}

/** Oracle: the same traversal with the whole graph in memory. */
function memoryBfs({ graph }, source = 0) {
  const dist = new Map([[source, 0]]);
  const queue = [source];
  while (queue.length) {
    const v = queue.shift();
    for (const w of graph.get(v)) if (!dist.has(w)) { dist.set(w, dist.get(v) + 1); queue.push(w); }
  }
  return dist;
}

const sameDistances = (a, b) => a.size === b.size && [...a].every(([v, d]) => b.get(v) === d);
const internalEdgeFraction = ({ graph }, pageOf) => {
  let inside = 0;
  let total = 0;
  for (const [v, neighbours] of graph) {
    for (const w of neighbours) {
      if (w <= v) continue;
      total += 1;
      if (pageOf[v] === pageOf[w]) inside += 1;
    }
  }
  return inside / total;
};

function main() {
  const small = clusteredGraph({});
  const pageSize = 16;
  const truth = memoryBfs(small);
  const clustered = clusteredPartition(small.graph, pageSize, small.n);
  const random = randomPartition(small.n, pageSize, 7);

  // 1. Correctness: partitioning and cache size never change the traversal result.
  const base = externalBfs(small, clustered, { cachePages: 16 });
  assert(sameDistances(base.dist, truth), "the external traversal matches the in-memory oracle");
  assert(base.dist.size === small.n, `every vertex is reachable in ${base.dist.size} of ${small.n}`);
  assert(sameDistances(externalBfs(small, random, { cachePages: 4 }).dist, truth), "results are partitioning independent");

  // 2. Partitioning matters: clustering keeps neighbours on the same page.
  const insideClustered = internalEdgeFraction(small, clustered);
  const insideRandom = internalEdgeFraction(small, random);
  assert(insideClustered > 4 * insideRandom, `internal edge share ${insideClustered} vs random ${insideRandom}`);
  assert(insideClustered > 0.25, "a depth-first layout keeps a substantial share of edges inside a page");

  // 3. I/O: the clustered layout needs far fewer page transfers than the random one.
  const clusteredRun = externalBfs(small, clustered, { cachePages: 2 });
  const randomRun = externalBfs(small, random, { cachePages: 2 });
  assert(clusteredRun.transfers < randomRun.transfers / 1.5, `clustered ${clusteredRun.transfers} transfers vs random ${randomRun.transfers}`);

  // 4. Batching the frontier beats per-vertex reads, and read order does not change the result.
  const hub = { n: 300, graph: new Map(Array.from({ length: 300 }, (_, i) => [i,
    i === 0 ? new Set(Array.from({ length: 299 }, (_, k) => k + 1)) : new Set([0])])) };
  const hubPages = randomPartition(300, 16, 5);
  const batched = externalBfs(hub, hubPages, { cachePages: 1, batch: true });
  const perVertex = externalBfs(hub, hubPages, { cachePages: 1, batch: false });
  assert(batched.transfers === 20 && perVertex.transfers > 10 * batched.transfers, `batched ${batched.transfers} vs per-vertex ${perVertex.transfers}`);
  assert(perVertex.seeks > batched.seeks, `per-vertex I/O seeks more: ${perVertex.seeks} vs ${batched.seeks}`);
  const hubScattered = externalBfs(hub, hubPages, { cachePages: 1, sorted: false, seed: 9 });
  assert(hubScattered.transfers === batched.transfers && sameDistances(hubScattered.dist, batched.dist) && batched.dist.size === 300, "read order does not change the result");

  // 5. Cache monotonicity: more pages can only save transfers, and the ceiling is the page count.
  const tiny = externalBfs(small, clustered, { cachePages: 1 }).transfers;
  const medium = externalBfs(small, clustered, { cachePages: 8 }).transfers;
  const big = externalBfs(small, clustered, { cachePages: small.n }).transfers;
  assert(tiny >= medium && medium >= big, `cache saves transfers: ${tiny} >= ${medium} >= ${big}`);
  assert(big === new Set(clustered).size, `a cache holding every page costs exactly one read per page, got ${big}`);

  // 6. I/O bound: transfers never exceed vertices plus pages plus edges over the page size.
  const edges = [...small.graph.values()].reduce((a, s) => a + s.size, 0) / 2;
  assert(clusteredRun.transfers <= small.n + base.pages, `transfers ${clusteredRun.transfers} bounded by n + pages`);

  // 7. Boundary: a single vertex, a hub and an empty graph.
  const single = { n: 1, graph: new Map([[0, new Set()]]) };
  const singleRun = externalBfs(single, [0], {});
  assert(singleRun.dist.get(0) === 0 && singleRun.transfers === 1 && singleRun.seeks === 1, "a single vertex costs one read");
  const star = { n: 40, graph: new Map(Array.from({ length: 40 }, (_, i) => [i, new Set(i === 0 ? Array.from({ length: 39 }, (_, k) => k + 1) : [0])])) };
  const starRun = externalBfs(star, randomPartition(40, 4, 3), { cachePages: 2 });
  assert(starRun.dist.size === 40 && starRun.dist.get(39) === 1, "a hub reaches every leaf in one hop");
  const empty = { n: 0, graph: new Map() };
  const emptyRun = externalBfs(empty, [], {});
  assert(emptyRun.dist.size === 0 && emptyRun.transfers === 0, "an empty graph is a no-op");

  // 8. Disconnected components: only the source's component is visited.
  const splitGraph = new Map(Array.from({ length: 10 }, (_, i) => [i, new Set()]));
  for (let i = 0; i < 4; i += 1) { splitGraph.get(i).add(i + 1); splitGraph.get(i + 1).add(i); }
  for (let i = 5; i < 9; i += 1) { splitGraph.get(i).add(i + 1); splitGraph.get(i + 1).add(i); }
  const splitRun = externalBfs({ n: 10, graph: splitGraph }, randomPartition(10, 2, 1), {});
  assert(splitRun.dist.size === 5 && !splitRun.dist.has(7), "components are traversed independently");

  // 9. Determinism: identical inputs, identical I/O counts.
  const repeat = externalBfs(small, clustered, { cachePages: 2 });
  assert(repeat.transfers === clusteredRun.transfers && repeat.seeks === clusteredRun.seeks, "the traversal is deterministic");
  assert(JSON.stringify([...clusteredPartition(small.graph, pageSize, small.n)]) === JSON.stringify([...clustered]), "the clustering is deterministic");

  // 10. Edge accounting sanity: the adjacency is symmetric and counted once.
  assert([...small.graph].every(([v, s]) => [...s].every((w) => small.graph.get(w).has(v))), "adjacency lists are symmetric");
  assert(Number.isInteger(edges) && edges > small.n, `the clustered graph has ${edges} edges`);

  console.log(`external-bfs: n=${small.n} edges=${edges} clusteredTransfers=${clusteredRun.transfers} randomTransfers=${randomRun.transfers} batched=${batched.transfers} perVertex=${perVertex.transfers}`);
  console.log("PASS");
}
module.exports = { splitmix64, clusteredGraph, randomPartition, clusteredPartition, externalBfs, memoryBfs, internalEdgeFraction };
if (require.main === module) main();
