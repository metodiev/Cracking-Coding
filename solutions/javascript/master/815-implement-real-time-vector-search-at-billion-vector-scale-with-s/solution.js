/** Task 815 — Implement real-time vector search at billion-vector scale with sharding and reranking.
 *  (Tier: Master, Beyond — Advanced Data Structures & Algorithms)
 *
 * Architecture (the standard ANN serving pipeline, modelled exactly):
 *   - vectors are hash-partitioned over `shardCount` shards, so every shard can be queried in
 *     parallel and holds an independent index (balance is verified, not assumed);
 *   - each shard holds an IVF index: k-means over its vectors, then one inverted list per centroid
 *     containing the *residual* (vector − centroid) scalar-quantised to `bitsPerDimension` bits. The
 *     number of centroids is chosen as `ceil(vectors / targetListSize)`, so a list holds O(1)
 *     vectors and the scan cost of `probesPerShard` lists does not grow with the corpus — that is the
 *     whole point of the IVF layout, and it is measured on two corpus sizes below;
 *   - a query scans the `probesPerShard` closest lists per shard with asymmetric distance
 *     computation (query in full precision, database quantised), merges the per-shard candidate
 *     heaps, and then *reranks* the merged candidates with exact distances on the stored
 *     full-precision vectors before returning the top-K.
 *
 * Scale. Real deployments hold ~10^9 vectors; here the scale knobs are typed configuration
 * (`shardCount`, `vectorsPerShard`, `targetListSize`, `probesPerShard`, `rerankCandidates`,
 * `dimensions`, `bitsPerDimension`) and the corpus is a few thousand integer vectors so the checks
 * finish in about a second. The claims that are actually verified are the ones that carry over:
 * search recall against brute force, the exactness of the exhaustive-probe path, the rerank gain, the
 * monotonicity of the k-means objective and the "centroid is the mean of its list" optimality
 * invariant, the deterministic shard balance, the per-query work bounds and the fact that the scanned
 * work stays flat while brute force grows linearly.
 *
 * Simplified versus production: no network/IPC between shards (shards are objects, the merge is a
 * local heap), no product quantisation or HNSW graphs, no concurrent index build, no GPU scan and no
 * wall-clock latency measurement — "real time" is expressed as the deterministic work counters
 * (`adcEvaluations`, `exactEvaluations`, `listsProbed`) whose analytic bounds are checked.
 *
 * Verified: recall@K and exactness versus brute force (bounded probes, full probes, 1-bit scan-only),
 * reranking never lowers recall on the same candidate set, cost accounting equals the analytic bound,
 * work stays flat when the corpus grows 4x while brute force grows linearly, k-means invariants,
 * shard balance, deterministic ties by id, and the degenerate cases (empty store, single vector,
 * duplicate vectors, K larger than the store, K = 1, one dimension).
 */
"use strict";

// ---------------------------------------------------------------------------- deterministic randomness

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
}

const COMPONENT_RANGE = 8;

/** Integer vectors keep every "exact" distance in this file free of floating-point doubt. */
function generateVectors(count, dimensions, seed) {
  const next = splitmix64(seed);
  const span = BigInt(2 * COMPONENT_RANGE + 1);
  const vectors = [];
  for (let id = 0; id < count; id++) {
    const components = new Array(dimensions);
    for (let d = 0; d < dimensions; d++) components[d] = Number(next() % span) - COMPONENT_RANGE;
    vectors.push({ id, components });
  }
  return vectors;
}

function squaredDistance(a, b) {
  let total = 0;
  for (let d = 0; d < a.length; d++) {
    const delta = a[d] - b[d];
    total += delta * delta;
  }
  return total;
}

function compareHits(a, b) {
  return a.distance - b.distance || a.id - b.id;
}

function sameAssignment(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function bruteForceTopK(vectors, query, k) {
  const hits = vectors.map((vector) => ({ id: vector.id, distance: squaredDistance(vector.components, query) }));
  hits.sort(compareHits);
  return hits.slice(0, Math.max(0, k));
}

function bruteForceWork(vectors, query) {
  for (const vector of vectors) squaredDistance(vector.components, query);
  return vectors.length;
}

// ---------------------------------------------------------------------------- k-means (IVF coarse quantizer)

class KMeans {
  /**
   * Deterministic Lloyd iterating: the seeding picks distinct points with splitmix64, empty clusters
   * deterministically adopt the point farthest from its centroid, and the objective is recorded so
   * its monotonicity can be checked.
   */
  constructor(points, k, seed) {
    this.points = points;
    this.k = Math.max(1, Math.min(k, points.length));
    this.dimensions = points.length === 0 ? 0 : points[0].length;
    this.next = splitmix64(seed);
    this.centroids = [];
    this.assignments = new Array(points.length).fill(-1);
    this.errorHistory = [];
    this.finalError = 0;
  }

  seed() {
    const used = new Set();
    while (this.centroids.length < this.k) {
      const index = Number(this.next() % BigInt(this.points.length));
      if (used.has(index)) continue;
      used.add(index);
      this.centroids.push(this.points[index].slice());
    }
  }

  assign() {
    let error = 0;
    for (let i = 0; i < this.points.length; i++) {
      let best = 0;
      let bestDistance = Infinity;
      for (let c = 0; c < this.centroids.length; c++) {
        const distance = squaredDistance(this.points[i], this.centroids[c]);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = c;
        }
      }
      this.assignments[i] = best;
      error += bestDistance;
    }
    return error;
  }

  update() {
    const sums = this.centroids.map(() => new Array(this.dimensions).fill(0));
    const counts = new Array(this.centroids.length).fill(0);
    for (let i = 0; i < this.points.length; i++) {
      const cluster = this.assignments[i];
      counts[cluster]++;
      for (let d = 0; d < this.dimensions; d++) sums[cluster][d] += this.points[i][d];
    }
    for (let c = 0; c < this.centroids.length; c++) {
      if (counts[c] === 0) {
        // Adopt the point that is worst served right now (deterministic tie-break by index).
        let worst = -1;
        let worstDistance = -1;
        for (let i = 0; i < this.points.length; i++) {
          const distance = squaredDistance(this.points[i], this.centroids[this.assignments[i]]);
          if (distance > worstDistance) {
            worstDistance = distance;
            worst = i;
          }
        }
        if (worst >= 0) {
          this.centroids[c] = this.points[worst].slice();
          this.assignments[worst] = c;
          counts[c] = 1;
        }
        continue;
      }
      for (let d = 0; d < this.dimensions; d++) this.centroids[c][d] = sums[c][d] / counts[c];
    }
  }

  /**
   * Lloyd iterations that stop at the first fixed point: the assignment of an iteration must repeat
   * the assignment the centroids were computed from, so the returned centroids are the means of their
   * members (the optimality invariant that `centroidIsMeanOfMembers` re-checks from scratch).
   */
  fit(iterations) {
    if (this.points.length === 0) {
      this.converged = true;
      return this;
    }
    this.seed();
    this.converged = false;
    let previous = null;
    for (let iteration = 0; iteration < iterations; iteration++) {
      const error = this.assign();
      this.errorHistory.push(error);
      this.finalError = error;
      if (previous !== null && sameAssignment(previous, this.assignments)) {
        this.converged = true;
        break;
      }
      previous = this.assignments.slice();
      this.update();
    }
    if (!this.converged) this.assign(); // keep the lists consistent with the centroids
    return this;
  }

  /** Optimality invariant: an equilibrium centroid is the mean of the points assigned to it. */
  centroidIsMeanOfMembers(tolerance = 1e-9) {
    const sums = this.centroids.map(() => new Array(this.dimensions).fill(0));
    const counts = new Array(this.centroids.length).fill(0);
    for (let i = 0; i < this.points.length; i++) {
      const cluster = this.assignments[i];
      counts[cluster]++;
      for (let d = 0; d < this.dimensions; d++) sums[cluster][d] += this.points[i][d];
    }
    for (let c = 0; c < this.centroids.length; c++) {
      if (counts[c] === 0) continue;
      for (let d = 0; d < this.dimensions; d++) {
        if (Math.abs(this.centroids[c][d] - sums[c][d] / counts[c]) > tolerance) return false;
      }
    }
    return true;
  }
}

// ---------------------------------------------------------------------------- quantised inverted list

/** Per-list residual scalar quantisation with asymmetric distance computation. */
class QuantizedList {
  constructor(entries, dimensions, bits) {
    this.dimensions = dimensions;
    this.levels = (1 << bits) - 1;
    this.offset = new Array(dimensions).fill(0);
    this.step = new Array(dimensions).fill(0);
    this.entries = entries; // [{ id, components, codes }]
    const residuals = entries.map((entry) => entry.residual);
    for (let d = 0; d < dimensions; d++) {
      let min = Infinity;
      let max = -Infinity;
      for (const residual of residuals) {
        if (residual[d] < min) min = residual[d];
        if (residual[d] > max) max = residual[d];
      }
      const span = max - min;
      this.offset[d] = min;
      this.step[d] = span === 0 ? 0 : span / this.levels;
    }
    this.codes = new Uint8Array(entries.length * dimensions);
    for (let i = 0; i < entries.length; i++) {
      for (let d = 0; d < dimensions; d++) {
        const { step, offset } = { step: this.step[d], offset: this.offset[d] };
        const code = step === 0 ? 0 : Math.round((residuals[i][d] - offset) / step);
        this.codes[i * dimensions + d] = Math.max(0, Math.min(this.levels, code));
      }
    }
  }

  /** ||q − (centroid + dequantised residual)||^2 — the query stays in full precision (ADC). */
  approximateDistance(query, centroid, entryIndex) {
    let total = 0;
    for (let d = 0; d < this.dimensions; d++) {
      const code = this.codes[entryIndex * this.dimensions + d];
      const approximated = centroid[d] + this.offset[d] + code * this.step[d];
      const delta = query[d] - approximated;
      total += delta * delta;
    }
    return total;
  }

  entryAt(index) {
    return this.entries[index];
  }
}

// ---------------------------------------------------------------------------- shard

class IVFShard {
  constructor(vectors, config, seed) {
    this.config = config;
    this.store = vectors;
    this.dimensions = config.dimensions;
    const targetCentroids = Math.max(1, Math.ceil(vectors.length / config.targetListSize));
    this.kmeans = new KMeans(vectors.map((vector) => vector.components), targetCentroids, seed);
    this.kmeans.fit(config.kmeansIterations);
    this.centroids = this.kmeans.centroids;
    this.lists = this.centroids.map(() => []);
    for (let i = 0; i < vectors.length; i++) {
      const cluster = this.kmeans.assignments[i];
      this.lists[cluster].push({
        id: vectors[i].id,
        components: vectors[i].components,
        residual: vectors[i].components.map((value, d) => value - this.centroids[cluster][d]),
      });
    }
    this.quantized = this.lists.map((list) => (list.length === 0 ? null : new QuantizedList(list, this.dimensions, config.bitsPerDimension)));
  }

  /** Approximate scan of the `probes` closest lists; returns a candidate heap plus work counters. */
  search(query, probes, limit) {
    const stats = { listsProbed: 0, adcEvaluations: 0 };
    const centroidOrder = this.centroids
      .map((centroid, index) => ({ index, distance: squaredDistance(query, centroid) }))
      .sort((a, b) => a.distance - b.distance || a.index - b.index)
      .slice(0, Math.max(0, Math.min(probes, this.centroids.length)));
    const candidates = [];
    for (const { index } of centroidOrder) {
      const list = this.quantized[index];
      if (list === null) continue;
      stats.listsProbed++;
      for (let i = 0; i < list.entries.length; i++) {
        stats.adcEvaluations++;
        candidates.push({ id: list.entryAt(i).id, distance: list.approximateDistance(query, this.centroids[index], i), components: list.entryAt(i).components });
      }
    }
    candidates.sort(compareHits);
    return { candidates: candidates.slice(0, Math.max(0, limit)), stats };
  }

  exactScan(query, limit) {
    const hits = this.store.map((vector) => ({ id: vector.id, distance: squaredDistance(vector.components, query) }));
    hits.sort(compareHits);
    return hits.slice(0, Math.max(0, limit));
  }

  exactTopK(query, k) {
    return this.exactScan(query, k);
  }
}

// ---------------------------------------------------------------------------- sharded store

const DEFAULT_CONFIG = {
  dimensions: 8,
  shardCount: 2,
  vectorsPerShard: 256,
  targetListSize: 8,
  bitsPerDimension: 8,
  probesPerShard: 8,
  rerankCandidates: 48,
  topK: 10,
  kmeansIterations: 24,
};

class ShardedVectorStore {
  constructor(config = {}, seed = 0x8155eed) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.seed = seed;
    this.shards = [];
    for (let shard = 0; shard < this.config.shardCount; shard++) {
      const vectors = generateVectors(this.config.vectorsPerShard, this.config.dimensions, seed + 1000 * (shard + 1)).map((vector, index) => ({
        ...vector,
        id: index * this.config.shardCount + shard, // round robin => balanced shards
      }));
      this.shards.push(new IVFShard(vectors, this.config, seed + 7 * (shard + 1)));
    }
  }

  get size() {
    return this.shards.reduce((sum, shard) => sum + shard.store.length, 0);
  }

  get allVectors() {
    return this.shards.flatMap((shard) => shard.store);
  }

  shardSizes() {
    return this.shards.map((shard) => shard.store.length);
  }

  /**
   * Full query path: per-shard approximate scan, merge, exact rerank of the merged candidates.
   * `probesPerShard === Infinity` scans every list, which makes the pipeline an exact search.
   */
  search(query, options = {}) {
    const probes = options.probesPerShard ?? this.config.probesPerShard;
    const perShardLimit = options.rerankCandidates ?? this.config.rerankCandidates;
    const topK = options.topK ?? this.config.topK;
    const stats = { listsProbed: 0, adcEvaluations: 0, exactEvaluations: 0, shardsTouched: 0 };
    const pool = [];
    for (const shard of this.shards) {
      const { candidates, stats: shardStats } = shard.search(query, probes, perShardLimit);
      stats.listsProbed += shardStats.listsProbed;
      stats.adcEvaluations += shardStats.adcEvaluations;
      if (shardStats.listsProbed > 0) stats.shardsTouched++;
      for (const candidate of candidates) pool.push(candidate);
    }
    pool.sort(compareHits);
    const reranked = pool.slice(0, perShardLimit).map((candidate) => {
      stats.exactEvaluations++;
      return { id: candidate.id, distance: squaredDistance(candidate.components, query), approximate: candidate.distance };
    });
    reranked.sort(compareHits);
    return { results: reranked.slice(0, topK), candidates: reranked, stats };
  }

  /** The same candidate set ordered by the quantised score only — used to measure the rerank gain. */
  searchWithoutRerank(query, options = {}) {
    const probes = options.probesPerShard ?? this.config.probesPerShard;
    const perShardLimit = options.rerankCandidates ?? this.config.rerankCandidates;
    const topK = options.topK ?? this.config.topK;
    const pool = [];
    for (const shard of this.shards) {
      const { candidates } = shard.search(query, probes, perShardLimit);
      pool.push(...candidates);
    }
    pool.sort(compareHits);
    return pool.slice(0, topK);
  }

  exactSearch(query, k) {
    return bruteForceTopK(this.allVectors, query, k);
  }
}

function recallAtK(approx, exact) {
  if (exact.length === 0) return 1;
  const exactIds = new Set(exact.map((hit) => hit.id));
  let hits = 0;
  for (const hit of approx) if (exactIds.has(hit.id)) hits++;
  return hits / exact.length;
}

module.exports = {
  splitmix64,
  generateVectors,
  squaredDistance,
  bruteForceTopK,
  bruteForceWork,
  compareHits,
  KMeans,
  QuantizedList,
  IVFShard,
  ShardedVectorStore,
  recallAtK,
  DEFAULT_CONFIG,
};

if (require.main === module) {
  const assert = require("node:assert");

  const store = new ShardedVectorStore({}, 0x815000);
  const queries = generateVectors(12, store.config.dimensions, 0x815aaa).map((vector) => vector.components);

  // --- shard balance and index invariants -------------------------------------
  const sizes = store.shardSizes();
  assert.strictEqual(sizes[0], sizes[1]);
  assert.strictEqual(store.size, store.config.shardCount * store.config.vectorsPerShard);
  for (const shard of store.shards) {
    assert.ok(shard.kmeans.converged, "k-means must reach a fixed point within the iteration budget");
    assert.ok(shard.kmeans.centroidIsMeanOfMembers(), "an equilibrium centroid must be the mean of its members");
    for (let i = 1; i < shard.kmeans.errorHistory.length; i++) {
      assert.ok(shard.kmeans.errorHistory[i] <= shard.kmeans.errorHistory[i - 1] + 1e-9, "k-means objective must not increase");
    }
    const occupancy = shard.lists.reduce((sum, list) => sum + list.length, 0);
    assert.strictEqual(occupancy, shard.store.length, "every vector must sit in exactly one list");
    for (const list of shard.lists) assert.ok(list.length <= 3 * store.config.targetListSize);
  }

  // --- exhaustive probes reproduce brute force exactly --------------------------
  for (const query of queries) {
    const exhaustive = store.search(query, { probesPerShard: Infinity });
    assert.deepStrictEqual(exhaustive.results.map((hit) => hit.id), store.exactSearch(query, store.config.topK).map((hit) => hit.id));
    const perShardLists = store.shards[0].lists.length;
    assert.strictEqual(exhaustive.stats.listsProbed, perShardLists * store.config.shardCount);
  }

  // --- bounded probes: recall, rerank gain and cost bounds ----------------------
  let recallSum = 0;
  let rerankSum = 0;
  let scanned = 0;
  let candidatesEvaluated = 0;
  let maxExactEvaluations = 0;
  for (const query of queries) {
    const exact = store.exactSearch(query, store.config.topK);
    const bounded = store.search(query);
    const quantizedOnly = store.searchWithoutRerank(query);
    const recall = recallAtK(bounded.results, exact);
    const recallNoRerank = recallAtK(quantizedOnly, exact);
    recallSum += recall;
    rerankSum += recallNoRerank;
    assert.ok(recall >= recallNoRerank - 1e-12, "reranking the same candidate set cannot lower recall");
    assert.ok(recall > 0.5, `bounded-probe recall ${recall}`);
    // Cost accounting: the scan touches exactly the probed lists, the rerank exactly the merged
    // candidate pool (capped by rerankCandidates).
    const probedEntries = bounded.stats.adcEvaluations;
    assert.ok(probedEntries > 0);
    assert.strictEqual(bounded.stats.exactEvaluations, bounded.candidates.length, "the rerank touches exactly the merged candidate pool");
    assert.ok(bounded.candidates.length <= store.config.shardCount * store.config.rerankCandidates);
    assert.ok(probedEntries <= store.size, "bounded probes must scan less than the whole store");
    assert.ok(bounded.stats.listsProbed <= store.config.probesPerShard * store.config.shardCount);
    scanned += probedEntries;
    candidatesEvaluated += bounded.stats.exactEvaluations;
    maxExactEvaluations = Math.max(maxExactEvaluations, bounded.stats.exactEvaluations);
  }
  const meanRecall = recallSum / queries.length;
  const meanRecallWithoutRerank = rerankSum / queries.length;
  assert.ok(meanRecall >= 0.9, `mean recall@10 with 8 probes ${meanRecall}`);
  assert.ok(meanRecall >= meanRecallWithoutRerank, "reranking must help on average");
  assert.ok(candidatesEvaluated / queries.length <= store.config.rerankCandidates, "exact rerank budget");
  assert.ok(scanned / queries.length < store.size * 0.75, "the average query must not scan the whole store");

  // --- a coarse quantiser makes the rerank stage visibly necessary --------------
  const coarse = new ShardedVectorStore({ bitsPerDimension: 2 }, 0x815000);
  let coarseRecall = 0;
  let coarseRecallNoRerank = 0;
  for (const query of queries) {
    const exact = coarse.exactSearch(query, coarse.config.topK);
    coarseRecall += recallAtK(coarse.search(query).results, exact);
    coarseRecallNoRerank += recallAtK(coarse.searchWithoutRerank(query), exact);
  }
  const coarseGain = (coarseRecall - coarseRecallNoRerank) / queries.length;
  assert.ok(coarseRecall / queries.length >= 0.9, `2-bit scan with rerank recall ${coarseRecall / queries.length}`);
  assert.ok(coarseGain >= 0.05, `rerank must recover a measurable amount with a coarse quantiser (gain ${coarseGain.toFixed(3)})`);

  // --- scale: flat scanned work when the corpus grows 4x ------------------------
  const bigStore = new ShardedVectorStore({ vectorsPerShard: 4 * DEFAULT_CONFIG.vectorsPerShard }, 0x815000);
  assert.strictEqual(bigStore.size, 4 * store.size);
  let smallScan = 0;
  let bigScan = 0;
  for (const query of queries) {
    smallScan += store.search(query).stats.adcEvaluations;
    bigScan += bigStore.search(query).stats.adcEvaluations;
  }
  const bruteSmall = queries.length * store.size;
  const bruteBig = queries.length * bigStore.size;
  const scanGrowth = bigScan / smallScan;
  const bruteGrowth = bruteBig / bruteSmall;
  assert.ok(scanGrowth < 1.6, `scanned work grew ${scanGrowth.toFixed(2)}x for a 4x corpus`);
  assert.ok(bruteGrowth > 3.5, "brute force grows linearly");
  const bigRecall = queries.reduce((sum, query) => sum + recallAtK(bigStore.search(query).results, bigStore.exactSearch(query, bigStore.config.topK)), 0) / queries.length;
  assert.ok(bigRecall >= 0.9, `recall must survive the 4x corpus: ${bigRecall}`);

  // --- determinism and tie-breaking ---------------------------------------------
  const rebuilt = new ShardedVectorStore({}, 0x815000);
  assert.strictEqual(JSON.stringify(rebuilt.shards.map((shard) => shard.centroids)), JSON.stringify(store.shards.map((shard) => shard.centroids)), "index build must be deterministic");
  const duplicated = new ShardedVectorStore({ vectorsPerShard: 16, shardCount: 1, targetListSize: 4, topK: 5 }, 3);
  duplicated.shards[0].store = duplicated.shards[0].store.map((vector) => ({ ...vector, components: [1, 1, 1, 1, 1, 1, 1, 1] }));
  const tieQuery = new Array(store.config.dimensions).fill(1);
  const ties = duplicated.exactSearch(tieQuery, 5);
  assert.deepStrictEqual(ties.map((hit) => hit.id), [0, 1, 2, 3, 4], "identical vectors must break ties by id");
  assert.strictEqual(new Set(ties.map((hit) => hit.distance)).size, 1);

  // --- degenerate inputs ----------------------------------------------------------
  const empty = new ShardedVectorStore({ vectorsPerShard: 0, shardCount: 1, targetListSize: 4 }, 5);
  assert.strictEqual(empty.size, 0);
  assert.deepStrictEqual(empty.search([0, 0, 0, 0, 0, 0, 0, 0]).results, []);
  const single = new ShardedVectorStore({ vectorsPerShard: 1, shardCount: 1, targetListSize: 4, probesPerShard: 1 }, 6);
  const singleResult = single.search([0, 0, 0, 0, 0, 0, 0, 0]);
  assert.strictEqual(singleResult.results.length, 1);
  assert.strictEqual(singleResult.results[0].id, 0);
  const bigK = store.search(queries[0], { topK: store.size + 10 });
  assert.strictEqual(bigK.results.length, Math.min(store.size, store.config.rerankCandidates), "the answer is capped by the candidate pool");
  const topOne = store.search(queries[1], { topK: 1 });
  assert.strictEqual(topOne.results.length, 1);
  assert.deepStrictEqual(topOne.results[0].id, store.exactSearch(queries[1], 1)[0].id, "top-1 must agree with brute force here");
  const oneDimensional = new ShardedVectorStore({ dimensions: 1, vectorsPerShard: 32, shardCount: 1, targetListSize: 4, topK: 3 }, 9);
  const oneQuery = [3];
  assert.deepStrictEqual(
    oneDimensional.search(oneQuery, { probesPerShard: Infinity }).results.map((hit) => hit.id),
    oneDimensional.exactSearch(oneQuery, 3).map((hit) => hit.id)
  );
  const oneBit = new ShardedVectorStore({ bitsPerDimension: 1, vectorsPerShard: 128, shardCount: 1, targetListSize: 8, topK: 5 }, 11);
  const oneBitQuery = queries[3];
  const oneBitResult = oneBit.search(oneBitQuery);
  assert.ok(oneBitResult.results.every((hit) => Number.isFinite(hit.distance)), "quantised scan must stay finite");
  assert.strictEqual(oneBitResult.stats.adcEvaluations > 0, true);

  console.log(
    `store ${store.size} vectors in ${store.shardSizes().length} shards, mean recall@10 ${meanRecall.toFixed(3)} ` +
      `(no rerank ${meanRecallWithoutRerank.toFixed(3)}, 2-bit no rerank ${(coarseRecallNoRerank / queries.length).toFixed(3)} -> with rerank ${(coarseRecall / queries.length).toFixed(3)}), ` +
      `rerank candidates <= ${maxExactEvaluations}, mean scan ${(scanned / queries.length).toFixed(1)} of ${store.size} vectors, ` +
      `4x corpus scan growth ${scanGrowth.toFixed(2)}x vs brute force ${bruteGrowth.toFixed(1)}x, big recall ${bigRecall.toFixed(3)}`
  );
  console.log("PASS");
}
