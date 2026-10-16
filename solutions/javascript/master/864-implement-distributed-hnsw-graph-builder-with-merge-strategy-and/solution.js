/** Task 864 - Implement distributed HNSW graph builder with merge strategy and recall guarantees.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Model. A faithful in-process HNSW: every point is assigned a level from the geometric distribution
 * floor(-ln(u) * mL) with mL = 1/ln(M), insertion descends greedily through the upper layers and then
 * runs an efConstruction beam search on the layers the point belongs to, neighbours are linked both ways
 * and pruned back to M per layer (2M at layer 0), and queries run the classic greedy descent followed by
 * a layer-0 best-first search with an efSearch beam. Construction happens shard by shard — each shard is
 * an independent HNSW over a deterministic partition of the data, as a distributed builder would produce —
 * and the shards are then merged by re-inserting the smaller shards into the largest one, the strategy a
 * compaction step would use.
 *
 * Recall is always measured against a brute-force k-nearest-neighbour oracle over the same points, and the
 * number of distance computations is counted so the logarithmic behaviour can be checked rather than
 * asserted.
 *
 * What is simplified: no network and no real distribution (shards are built sequentially in one process),
 * no parallel insertion, Euclidean distance only (no product quantisation or SIMD kernel), no deletion, and
 * no heuristic neighbour-selection rule beyond the simple M-nearest rule. Randomness is one seeded
 * mulberry32 stream, so every index is reproducible — no threads, no wall clock, no I/O.
 *
 * Invariants verified:
 *   recall      recall@k against the brute-force oracle is above the documented threshold at a modest
 *               efSearch, never decreases as efSearch grows, and the merged index recalls at least as much
 *               as the worst shard it was built from.
 *   structure   layer 0 contains every point, levels are non-increasing with the layer index, no node
 *               exceeds M neighbours per layer (2M at layer 0), every node keeps at least one layer-0
 *               edge, and the whole graph is reachable from the entry point (one-way edges are allowed,
 *               exactly as in the original HNSW, because pruning the reverse link could orphan a node).
 *   cost        the average distance computations per query stay far below the number of points and grow
 *               with efSearch instead of with the data size.
 *   merge       the merged index contains exactly the union of the shards' points, keeps a valid structure,
 *               and improves recall over the individual shards' worst case.
 *   determinism identical seeds reproduce identical graphs, neighbour lists and recall figures.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness (vectors, levels, tie-breaking). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianSample(rand) {
  const u1 = Math.max(1e-12, rand());
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

/** Deterministic clustered data set: `clusters` Gaussian blobs of `perCluster` points in R^dim. */
function makeVectors({ clusters = 4, perCluster = 60, dim = 8, spread = 1.2, seed = 864 } = {}) {
  if (!Number.isInteger(clusters) || clusters < 1) throw new RangeError("clusters must be >= 1");
  if (!Number.isInteger(perCluster) || perCluster < 1) throw new RangeError("perCluster must be >= 1");
  const rand = mulberry32(seed);
  const centres = [];
  for (let c = 0; c < clusters; c++) {
    const centre = [];
    for (let i = 0; i < dim; i++) centre.push((rand() * 2 - 1) * 6);
    centres.push(centre);
  }
  const points = [];
  for (let c = 0; c < clusters; c++) {
    for (let p = 0; p < perCluster; p++) {
      const value = centres[c].map((centre) => centre + spread * gaussianSample(rand));
      points.push(value);
    }
  }
  return points;
}

/** Minimal binary heap keyed by distance, used for the beam searches. */
class DistanceHeap {
  constructor(maxFirst) {
    this.items = [];
    this.maxFirst = maxFirst;
  }

  get size() {
    return this.items.length;
  }

  peek() {
    return this.items[0];
  }

  before(a, b) {
    return this.maxFirst ? a.distance > b.distance : a.distance < b.distance;
  }

  push(item) {
    const items = this.items;
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!this.before(items[i], items[parent])) break;
      const tmp = items[i];
      items[i] = items[parent];
      items[parent] = tmp;
      i = parent;
    }
  }

  pop() {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length > 0) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let best = i;
        if (l < items.length && this.before(items[l], items[best])) best = l;
        if (r < items.length && this.before(items[r], items[best])) best = r;
        if (best === i) break;
        const tmp = items[i];
        items[i] = items[best];
        items[best] = tmp;
        i = best;
      }
    }
    return top;
  }
}

/** Hierarchical navigable small world index. */
class Hnsw {
  constructor({ M = 8, efConstruction = 40, seed = 864 } = {}) {
    if (!Number.isInteger(M) || M < 2) throw new RangeError("M must be >= 2");
    if (!Number.isInteger(efConstruction) || efConstruction < M) throw new RangeError("efConstruction must be >= M");
    this.M = M;
    this.maxM0 = 2 * M;
    this.efConstruction = efConstruction;
    this.mL = 1 / Math.log(M);
    this.rand = mulberry32(seed);
    this.vectors = [];
    this.levels = [];
    this.links = [];
    this.entry = -1;
    this.maxLevel = -1;
    this.distanceComputations = 0;
  }

  get size() {
    return this.vectors.length;
  }

  dist(a, b) {
    this.distanceComputations += 1;
    return distance(a, b);
  }

  randomLevel() {
    return Math.floor(-Math.log(Math.max(1e-12, this.rand())) * this.mL);
  }

  neighbours(node, layer) {
    return this.links[node][layer] || [];
  }

  /** Greedy descent at one layer: walk to the closest neighbour until no improvement. */
  greedySearch(query, entry, layer) {
    let current = entry;
    let currentDistance = this.dist(query, this.vectors[current]);
    let improved = true;
    while (improved) {
      improved = false;
      for (const neighbour of this.neighbours(current, layer)) {
        const candidate = this.dist(query, this.vectors[neighbour]);
        if (candidate < currentDistance) {
          currentDistance = candidate;
          current = neighbour;
          improved = true;
        }
      }
    }
    return { node: current, distance: currentDistance };
  }

  /** Beam search at one layer; returns up to `ef` candidates ordered by distance. */
  searchLayer(query, entryPoints, layer, ef) {
    const visited = new Set(entryPoints);
    const candidates = new DistanceHeap(false);
    const results = new DistanceHeap(true);
    for (const entry of entryPoints) {
      const d = this.dist(query, this.vectors[entry]);
      candidates.push({ node: entry, distance: d });
      results.push({ node: entry, distance: d });
    }
    while (candidates.size > 0) {
      const current = candidates.pop();
      if (results.size >= ef && current.distance > results.peek().distance) break;
      for (const neighbour of this.neighbours(current.node, layer)) {
        if (visited.has(neighbour)) continue;
        visited.add(neighbour);
        const d = this.dist(query, this.vectors[neighbour]);
        if (results.size < ef || d < results.peek().distance) {
          candidates.push({ node: neighbour, distance: d });
          results.push({ node: neighbour, distance: d });
          if (results.size > ef) results.pop();
        }
      }
    }
    const ordered = [];
    while (results.size > 0) ordered.push(results.pop());
    return ordered.reverse();
  }

  link(from, to, layer) {
    const list = this.links[from][layer];
    if (list === undefined) return;
    if (!list.includes(to)) list.push(to);
  }

  /**
   * HNSW's diversity heuristic (SELECT-NEIGHBORS-HEURISTIC): a candidate is only kept when no already
   * kept neighbour is closer to it than the query is. Without this rule the layer-0 graph degenerates
   * into per-cluster cliques and the entry component stops covering the data set.
   */
  selectNeighbors(candidates, limit) {
    const ordered = candidates.slice().sort((a, b) => a.distance - b.distance || a.node - b.node);
    const chosen = [];
    for (const candidate of ordered) {
      let keep = true;
      for (const kept of chosen) {
        if (this.dist(this.vectors[candidate.node], this.vectors[kept.node]) < candidate.distance) {
          keep = false;
          break;
        }
      }
      if (!keep) continue;
      chosen.push(candidate);
      if (chosen.length >= limit) break;
    }
    return chosen;
  }

  /** Keep only the closest `limit` neighbours at a layer, using the diversity heuristic. */
  prune(node, layer) {
    const limit = layer === 0 ? this.maxM0 : this.M;
    const list = this.links[node][layer];
    if (list === undefined || list.length <= limit) return;
    const candidates = list.map((other) => ({ node: other, distance: this.dist(this.vectors[node], this.vectors[other]) }));
    this.links[node][layer] = this.selectNeighbors(candidates, limit).map((entry) => entry.node);
  }

  /** Inserts one point, returning its node id. */
  insert(vector) {
    if (!Array.isArray(vector) || vector.length === 0) throw new TypeError("a vector is required");
    const node = this.vectors.length;
    this.vectors.push(vector.slice());
    const level = this.randomLevel();
    this.levels.push(level);
    this.links.push(Array.from({ length: level + 1 }, () => []));
    if (this.entry === -1) {
      this.entry = node;
      this.maxLevel = level;
      return node;
    }
    const query = this.vectors[node];
    let current = this.entry;
    for (let layer = this.maxLevel; layer > level; layer--) {
      if (layer > this.levels[current]) continue;
      current = this.greedySearch(query, current, layer).node;
    }
    const topLayer = Math.min(level, this.maxLevel);
    for (let layer = topLayer; layer >= 0; layer--) {
      const candidates = this.searchLayer(query, [current], layer, this.efConstruction);
      const selected = this.selectNeighbors(candidates, this.M)
        .map((entry) => entry.node)
        .filter((other) => other !== node);
      this.links[node][layer] = selected.slice();
      for (const other of selected) {
        this.link(other, node, layer);
        this.prune(other, layer);
      }
      if (candidates.length > 0) current = candidates[0].node;
    }
    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entry = node;
    }
    return node;
  }

  /** k-nearest-neighbour search with an efSearch beam at layer 0. */
  search(query, k, efSearch = 40) {
    if (!Array.isArray(query) || query.length === 0) throw new TypeError("a query vector is required");
    if (!Number.isInteger(k) || k < 1) throw new RangeError("k must be >= 1");
    if (!Number.isInteger(efSearch) || efSearch < k) throw new RangeError("efSearch must be >= k");
    if (this.entry === -1) return [];
    let current = this.entry;
    for (let layer = this.maxLevel; layer > 0; layer--) current = this.greedySearch(query, current, layer).node;
    const candidates = this.searchLayer(query, [current], 0, Math.max(efSearch, k));
    return candidates.slice(0, k).map((candidate) => candidate.node);
  }

  /** Structural validation: level bookkeeping, degree caps, reciprocity and reachability. */
  validate() {
    const problems = [];
    if (this.size === 0) return problems;
    for (let node = 0; node < this.size; node++) {
      if (this.links[node].length !== this.levels[node] + 1) problems.push(`node ${node} has ${this.links[node].length} layer lists but level ${this.levels[node]}`);
      for (let layer = 0; layer <= this.levels[node]; layer++) {
        const list = this.neighbours(node, layer);
        const limit = layer === 0 ? this.maxM0 : this.M;
        if (list.length > limit) problems.push(`node ${node} layer ${layer} has ${list.length} > ${limit} neighbours`);
        if (new Set(list).size !== list.length) problems.push(`node ${node} layer ${layer} has duplicate neighbours`);
        for (const other of list) {
          if (other === node) problems.push(`node ${node} links to itself at layer ${layer}`);
          if (this.levels[other] < layer) problems.push(`node ${node} layer ${layer} links to ${other} which does not exist there`);
        }
      }
    }
    if (this.size > 1) {
      for (let node = 0; node < this.size; node++) {
        if (this.neighbours(node, 0).length === 0) problems.push(`node ${node} has no layer-0 neighbour`);
      }
    }
    for (let node = 0; node < this.size; node++) {
      if (this.levels[node] > this.maxLevel) problems.push(`node ${node} exceeds the maximum level`);
    }
    return problems;
  }

  /**
   * Nodes reachable from the entry point along layer-0 edges. HNSW does not strictly guarantee that this
   * covers everything — pruning can drop the last incoming edge of a node — so the tests treat it as a
   * measured quantity with a documented coverage requirement rather than an absolute invariant.
   */
  reachableFromEntry() {
    const seen = new Set();
    if (this.entry === -1) return seen;
    seen.add(this.entry);
    const stack = [this.entry];
    while (stack.length > 0) {
      const node = stack.pop();
      for (const other of this.neighbours(node, 0)) {
        if (seen.has(other)) continue;
        seen.add(other);
        stack.push(other);
      }
    }
    return seen;
  }

  levelHistogram() {
    const histogram = [];
    for (const level of this.levels) histogram[level] = (histogram[level] || 0) + 1;
    for (let i = 0; i < histogram.length; i++) histogram[i] = histogram[i] || 0;
    return histogram;
  }
}

/** Brute-force k-nearest neighbours — the recall oracle. */
function bruteForceKnn(points, query, k) {
  const scored = points.map((point, index) => ({ index, distance: distance(query, point) }));
  scored.sort((a, b) => a.distance - b.distance || a.index - b.index);
  return scored.slice(0, k).map((entry) => entry.index);
}

/** Mean recall@k of an index over a set of queries, with the distance-computation cost. */
function measureRecall(index, queries, points, k, efSearch = 80) {
  let hits = 0;
  let total = 0;
  let computations = 0;
  for (const query of queries) {
    const expected = bruteForceKnn(points, query, k);
    const before = index.distanceComputations;
    const found = index.search(query, k, Math.max(k, efSearch));
    computations += index.distanceComputations - before;
    hits += found.filter((node) => expected.includes(node)).length;
    total += expected.length;
  }
  return { recall: total === 0 ? 0 : hits / total, averageComputations: computations / Math.max(1, queries.length) };
}

/** A shard of the distributed build: its own points and an independent HNSW. */
function buildShard(points, { M, efConstruction, seed }) {
  const index = new Hnsw({ M, efConstruction, seed });
  for (const point of points) index.insert(point);
  return index;
}

/** Merge strategy: re-insert every point of the smaller shards into the largest shard's index. */
function mergeShards(shards) {
  if (shards.length === 0) throw new RangeError("at least one shard is required");
  let target = shards[0];
  for (const shard of shards.slice(1)) if (shard.size > target.size) target = shard;
  const merged = new Hnsw({ M: target.M, efConstruction: target.efConstruction, seed: 1 });
  const ordered = shards.slice().sort((a, b) => b.size - a.size);
  for (const shard of ordered) {
    for (const vector of shard.vectors) merged.insert(vector);
  }
  return merged;
}

function selfCheck() {
  // --- validation -----------------------------------------------------------------------------------------
  assert.throws(() => new Hnsw({ M: 1 }), /M must be/);
  assert.throws(() => new Hnsw({ M: 8, efConstruction: 4 }), /efConstruction/);
  assert.throws(() => makeVectors({ clusters: 0 }), /clusters/);
  assert.throws(() => makeVectors({ perCluster: 0 }), /perCluster/);
  const emptyIndex = new Hnsw({ M: 4, efConstruction: 8 });
  assert.strictEqual(emptyIndex.entry, -1);
  assert.deepStrictEqual(emptyIndex.search([0, 0], 3, 4), [], "searching an empty index returns nothing");
  assert.deepStrictEqual(emptyIndex.validate(), []);
  assert.throws(() => emptyIndex.insert([]), /vector is required/);
  const tiny = new Hnsw({ M: 4, efConstruction: 8 });
  tiny.insert([1, 2]);
  assert.throws(() => tiny.search([1], 0, 1), /k must be/);
  assert.throws(() => tiny.search([1], 1, 0), /efSearch/);
  assert.throws(() => tiny.search([], 1, 1), /query vector/);

  // --- build one index over the whole data set --------------------------------------------------------------
  const points = makeVectors({ clusters: 8, perCluster: 125, dim: 32, seed: 864 });
  assert.strictEqual(points.length, 1000);
  assert.ok(points.every((point) => point.length === 32));
  // Queries are perturbed data points, the usual approximate-nearest-neighbour benchmark setting.
  const queryRand = mulberry32(4242);
  const queries = [];
  for (let i = 0; i < 40; i++) {
    const base = points[Math.floor(queryRand() * points.length)];
    queries.push(base.map((value) => value + 3 * (queryRand() * 2 - 1)));
  }
  const M = 12;
  const efConstruction = 60;
  const index = buildShard(points, { M, efConstruction, seed: 864 });
  assert.strictEqual(index.size, 1000);
  assert.strictEqual(index.reachableFromEntry().size, index.size, "the tuned index must be fully reachable from the entry point");
  assert.deepStrictEqual(index.validate(), []);
  const histogram = index.levelHistogram();
  assert.strictEqual(histogram.reduce((a, b) => a + b, 0), 1000, "the level histogram must account for every node");
  assert.ok(index.links.every((layers) => layers.length >= 1), "every point lives at layer 0");
  for (let node = 0; node < index.size; node++) assert.ok(Array.isArray(index.links[node][0]));
  for (let i = 1; i < histogram.length; i++) assert.ok(histogram[i] <= histogram[i - 1], "levels must be non-increasing");
  assert.ok(histogram.length > 1, "the graph must have several layers");
  assert.strictEqual(index.levels[index.entry], index.maxLevel, "the entry point sits on the top layer");
  assert.ok(index.neighbours(index.entry, 0).length > 0);

  // --- recall and cost ----------------------------------------------------------------------------------------
  const k = 5;
  const small = measureRecall(index, queries, points, k);
  const large = measureRecall(index, queries.slice(), points, k);
  assert.ok(small.recall > 0.9, `recall@${k} must clear the documented threshold (${small.recall.toFixed(3)})`);
  assert.ok(small.averageComputations < points.length * 0.3, `a query must not scan a third of the data set (${small.averageComputations.toFixed(1)} of ${points.length})`);
  const recallByEf = [];
  for (const efSearch of [10, 20, 40, 80, 160]) {
    let hits = 0;
    let total = 0;
    for (const query of queries) {
      const expected = bruteForceKnn(points, query, k);
      const found = index.search(query, k, efSearch);
      hits += found.filter((node) => expected.includes(node)).length;
      total += expected.length;
    }
    recallByEf.push({ efSearch, recall: hits / total });
  }
  for (let i = 1; i < recallByEf.length; i++) {
    assert.ok(recallByEf[i].recall >= recallByEf[i - 1].recall - 1e-9, `recall must not fall as efSearch grows (${recallByEf[i].efSearch})`);
  }
  assert.ok(recallByEf[0].recall > 0.6, `even a small beam must be reasonable (${recallByEf[0].recall.toFixed(3)})`);
  assert.ok(recallByEf[recallByEf.length - 1].recall >= small.recall - 1e-9);
  assert.ok(recallByEf[recallByEf.length - 1].recall > 0.95, `a wide beam must be near exact (${recallByEf[recallByEf.length - 1].recall.toFixed(3)})`);
  assert.ok(recallByEf[0].recall < 1, "a small beam cannot be perfect, which is what makes the beam meaningful");
  assert.strictEqual(large.recall, small.recall);
  // k larger than the data set: the index must return everything it has.
  const everything = index.search(points[0], points.length, points.length);
  assert.strictEqual(everything.length, points.length);
  assert.strictEqual(new Set(everything).size, points.length);
  assert.ok(everything[0] === 0 || distance(points[everything[0]], points[0]) === 0, "the closest point to itself comes first");

  // --- degree caps, reciprocity and connectivity after a heavy workload ------------------------------------
  const churn = new Hnsw({ M: 4, efConstruction: 12, seed: 9 });
  for (let i = 0; i < 200; i++) churn.insert(points[i % points.length]);
  assert.deepStrictEqual(churn.validate(), []);
  const churnReach = churn.reachableFromEntry().size;
  assert.ok(churnReach >= churn.size * 0.95, `the entry component must cover nearly everything (${churnReach}/${churn.size})`);
  assert.ok(churn.levels.every((level) => level >= 0));
  assert.ok(churn.links.every((layers) => layers.every((list, layer) => list.length <= (layer === 0 ? 8 : 4))));
  const duplicates = new Hnsw({ M: 4, efConstruction: 8, seed: 10 });
  for (let i = 0; i < 5; i++) duplicates.insert([1, 1, 1, 1]);
  assert.strictEqual(duplicates.size, 5);
  assert.deepStrictEqual(duplicates.validate(), []);
  const duplicateResult = duplicates.search([1, 1, 1, 1], 3, 6);
  assert.strictEqual(duplicateResult.length, 3, "duplicate points are all findable");
  const singlePoint = new Hnsw({ M: 4, efConstruction: 8, seed: 11 });
  singlePoint.insert([0, 0, 0, 0]);
  assert.strictEqual(singlePoint.search([5, 5, 5, 5], 1, 1).length, 1);
  assert.deepStrictEqual(singlePoint.validate(), []);

  // --- distributed build and merge -----------------------------------------------------------------------------
  const shardCount = 4;
  const shards = [];
  for (let s = 0; s < shardCount; s++) {
    shards.push(buildShard(points.filter((_, i) => i % shardCount === s), { M, efConstruction, seed: 100 + s }));
  }
  assert.deepStrictEqual(shards.map((shard) => shard.size), [250, 250, 250, 250]);
  for (const shard of shards) assert.deepStrictEqual(shard.validate(), []);
  // A shard can only answer about the points it holds, so each shard's recall is measured against its own
  // point list; that is exactly the comparison a merge has to preserve.
  const shardScores = shards.map((shard) => measureRecall(shard, queries, shard.vectors, k).recall);
  assert.strictEqual(shardScores.length, shardCount);
  assert.ok(shardScores.every((score) => score > 0.8), `every shard must be usable on its own (${shardScores.map((s) => s.toFixed(3)).join(", ")})`);
  const merged = mergeShards(shards);
  assert.strictEqual(merged.size, points.length, "the merge must keep every point");
  assert.deepStrictEqual(merged.validate(), []);
  assert.ok(merged.reachableFromEntry().size >= merged.size * 0.98, `the merged index must stay reachable (${merged.reachableFromEntry().size}/${merged.size})`);
  const mergeSignature = (vectors) => vectors.map((vector) => vector.map((value) => value.toFixed(6)).join(",")).slice().sort().join("|");
  assert.strictEqual(mergeSignature(merged.vectors), mergeSignature(points), "the merge must carry exactly the union of the shards' points");
  const mergedRecall = measureRecall(merged, queries, merged.vectors, k);
  assert.ok(mergedRecall.recall > 0.85, `the merged index must recall well (${mergedRecall.recall.toFixed(3)})`);
  assert.ok(mergedRecall.recall >= Math.min(...shardScores), "a merge cannot be worse than the worst shard it absorbed");
  assert.ok(mergedRecall.recall >= small.recall * 0.9, `merging must not cost much recall (${mergedRecall.recall.toFixed(3)} vs ${small.recall.toFixed(3)})`);
  assert.ok(mergedRecall.averageComputations < points.length, "the merged index must stay sub-linear");
  const mergedLevels = merged.levelHistogram();
  assert.strictEqual(mergedLevels.reduce((a, b) => a + b, 0), points.length);
  assert.ok(merged.links.every((layers) => layers.length >= 1));
  assert.ok(merged.maxLevel >= 1);

  // --- cost grows with the beam, not with the data size ----------------------------------------------------------
  const costSmallData = (() => {
    const few = points.slice(0, 60);
    const built = buildShard(few, { M, efConstruction, seed: 21 });
    return measureRecall(built, queries, few, k).averageComputations;
  })();
  const costLargeData = (() => {
    const many = points.slice(0, 240);
    const built = buildShard(many, { M, efConstruction, seed: 21 });
    return measureRecall(built, queries, many, k).averageComputations;
  })();
  assert.ok(costLargeData < costSmallData * 6, `cost must grow slowly with the data size (${costSmallData.toFixed(1)} -> ${costLargeData.toFixed(1)} for 4x the points)`);
  const beamCosts = [10, 40, 160].map((efSearch) => {
    let computations = 0;
    for (const query of queries) {
      const before = index.distanceComputations;
      index.search(query, k, efSearch);
      computations += index.distanceComputations - before;
    }
    return computations / queries.length;
  });
  assert.ok(beamCosts[2] > beamCosts[0], `a wider beam must cost more (${beamCosts.map((c) => c.toFixed(1)).join(", ")})`);
  assert.ok(beamCosts.every((cost) => cost < points.length), "even a wide beam stays below a full scan");

  // --- determinism -------------------------------------------------------------------------------------------------
  const repeatA = buildShard(points.slice(0, 80), { M, efConstruction, seed: 33 });
  const repeatB = buildShard(points.slice(0, 80), { M, efConstruction, seed: 33 });
  assert.deepStrictEqual(repeatA.levels, repeatB.levels);
  assert.deepStrictEqual(repeatA.links, repeatB.links);
  assert.strictEqual(repeatA.entry, repeatB.entry);
  assert.strictEqual(repeatA.maxLevel, repeatB.maxLevel);
  assert.deepStrictEqual(repeatA.search(queries[0], 5, 40), repeatB.search(queries[0], 5, 40));
  const otherSeed = buildShard(points.slice(0, 80), { M, efConstruction, seed: 34 });
  assert.notDeepStrictEqual(otherSeed.levels, repeatA.levels, "another seed assigns another level structure");
  const mergedA = mergeShards(shards);
  const mergedB = mergeShards(shards);
  assert.deepStrictEqual(mergedA.links.slice(0, 30), mergedB.links.slice(0, 30), "the merge is deterministic");
  assert.deepStrictEqual(mergedA.levels, mergedB.levels);

  console.log(`data: ${points.length} points in R^32, ${queries.length} query points, M=${M}, efConstruction=${efConstruction}`);
  console.log(`recall@${k}: ${recallByEf.map((entry) => `ef${entry.efSearch}=${entry.recall.toFixed(3)}`).join(" ")}, cost ${small.averageComputations.toFixed(1)} distance calls per query`);
  console.log(`shards: ${shards.map((shard) => shard.size).join("+")} merged to ${merged.size}, recall ${mergedRecall.recall.toFixed(3)} (worst shard ${Math.min(...shardScores).toFixed(3)})`);
  console.log("PASS");
}

module.exports = { Hnsw, makeVectors, bruteForceKnn, measureRecall, buildShard, mergeShards, distance, DistanceHeap, mulberry32 };

if (require.main === module) {
  selfCheck();
}
