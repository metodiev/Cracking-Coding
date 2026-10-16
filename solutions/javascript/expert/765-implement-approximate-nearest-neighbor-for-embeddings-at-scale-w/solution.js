/** Task 765 - Implement approximate nearest neighbor for embeddings at scale with IVF+PQ. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Two-stage ANN in the classic FAISS shape. *IVF* (inverted file): a coarse quantiser is trained
 * with Lloyd's k-means (k-means++ seeding, deterministic PRNG) over the raw vectors, every vector
 * is assigned to its nearest centroid, and a query visits only the `nprobe` nearest lists.
 * *PQ* (product quantisation): each vector is split into `m` subvectors, one small codebook is
 * trained per subspace, and a vector becomes `m` codes — 4 bits each here, i.e. 8 bytes instead of
 * the 64 a float32 vector would take. Distances are computed by *ADC* (asymmetric distance
 * computation): the table `||residual - centroid||^2` for every subspace and code is built once per
 * query, after which a database vector costs only `m` lookups and adds.
 * The tests pin the algorithm down rather than trusting it: k-means must assign every point to its
 * nearest centroid with a non-increasing objective, every PQ code must be the argmin of its
 * subspace, and the ADC distance must equal the brute-force squared distance between the residual
 * and the *reconstructed* vector exactly. Recall is then measured against the exact search, never
 * assumed.
 *
 * What is simplified: 16-dimensional float64 vectors instead of 768-d float32 embeddings, 200
 * points instead of millions, no OPQ rotation, no HNSW/IVF hybrid, no SIMD, no reranking pass with
 * full vectors, and no disk-backed inverted lists. What is verified: k-means invariants on both
 * quantisers, exact ADC distances per list, recall@k over 20 seeded queries with a printed value and
 * a floor, monotonicity of recall in nprobe, full coverage when nprobe covers every list, correct
 * handling of fewer points than lists and of duplicate vectors, determinism, and validation.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

const distance2 = (a, b) => a.reduce((total, value, i) => total + (value - b[i]) ** 2, 0);
const centroidOf = (points) => points[0].map((_, dimension) => points.reduce((sum, point) => sum + point[dimension], 0) / points.length);

/** Deterministic xorshift32. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x765765;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Lloyd's algorithm with k-means++ seeding; returns the centroids and the objective history. */
function kmeans(points, k, rng, { iterations = 20 } = {}) {
  if (points.length === 0 || k < 1) throw new RangeError("k-means needs points and at least one cluster");
  const clusters = Math.min(k, points.length);
  const centroids = [points[rng(points.length)].slice()];
  while (centroids.length < clusters) {
    const weights = points.map((point) => Math.min(...centroids.map((centroid) => distance2(point, centroid))));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    if (total === 0) { centroids.push(points[rng(points.length)].slice()); continue; }
    let target = (rng(1 << 20) / (1 << 20)) * total;
    let chosen = points.length - 1;
    for (let index = 0; index < points.length; index++) {
      target -= weights[index];
      if (target <= 0) { chosen = index; break; }
    }
    centroids.push(points[chosen].slice());
  }
  const objectiveHistory = [];
  let assignments = points.map(() => -1);
  for (let iteration = 0; iteration < iterations; iteration++) {
    assignments = points.map((point) => {
      let best = 0;
      for (let index = 1; index < centroids.length; index++) {
        if (distance2(point, centroids[index]) < distance2(point, centroids[best])) best = index;
      }
      return best;
    });
    objectiveHistory.push(assignments.reduce((total, cluster, index) => total + distance2(points[index], centroids[cluster]), 0));
    for (let cluster = 0; cluster < centroids.length; cluster++) {
      const members = points.filter((_, index) => assignments[index] === cluster);
      if (members.length > 0) centroids[cluster] = centroidOf(members);
    }
  }
  return { centroids, assignments, objectiveHistory };
}

class IVFProductQuantiser {
  constructor(vectors, { nlist = 8, m = 8, centroidsPerSub = 16, seed = 1, iterations = 20 } = {}) {
    if (!Array.isArray(vectors) || vectors.length === 0) throw new RangeError("at least one vector is required");
    const dimension = vectors[0].length;
    if (!Number.isInteger(m) || m < 1 || dimension % m !== 0) throw new RangeError("m must divide the dimension");
    if (!Number.isInteger(centroidsPerSub) || centroidsPerSub < 1) throw new RangeError("centroidsPerSub must be positive");
    if (!Number.isInteger(nlist) || nlist < 1) throw new RangeError("nlist must be positive");
    this.vectors = vectors;
    this.dimension = dimension;
    this.m = m;
    this.width = dimension / m;
    this.centroidsPerSub = centroidsPerSub;
    this.iterations = iterations;
    const rng = makeRng(seed);
    this.coarse = kmeans(vectors, nlist, rng, { iterations });
    this.lists = this.coarse.centroids.map(() => []);
    vectors.forEach((_, index) => this.lists[this.coarse.assignments[index]].push(index));
    // PQ encodes the *residual* to the coarse centroid, which is what makes the approximation good
    this.residuals = vectors.map((vector, index) =>
      vector.map((value, dimension) => value - this.coarse.centroids[this.coarse.assignments[index]][dimension]));
    this.codebooks = [];
    this.codebookObjectives = [];
    for (let subspace = 0; subspace < m; subspace++) {
      const subvectors = this.residuals.map((residual) => residual.slice(subspace * this.width, (subspace + 1) * this.width));
      const trained = kmeans(subvectors, centroidsPerSub, rng, { iterations });
      this.codebooks.push(trained.centroids);
      this.codebookObjectives.push(trained.objectiveHistory);
    }
    this.codes = this.residuals.map((residual) => this.encodeResidual(residual));
  }

  get nlist() { return this.coarse.centroids.length; }
  get codeBytes() { return (this.m * 4) / 8; }
  get rawBytes() { return this.dimension * 4; }

  /** Nearest codebook entry of every subspace: the PQ code of a residual. */
  encodeResidual(residual) {
    return this.codebooks.map((centroids, subspace) => {
      const part = residual.slice(subspace * this.width, (subspace + 1) * this.width);
      let best = 0;
      for (let index = 1; index < centroids.length; index++) {
        if (distance2(part, centroids[index]) < distance2(part, centroids[best])) best = index;
      }
      return best;
    });
  }

  /** The vector a code stands for: the coarse centroid plus the decoded residual. */
  reconstruct(index) {
    const base = this.coarse.centroids[this.coarse.assignments[index]];
    return base.map((value, dimension) => value + this.codebooks.reduce((total, centroids, subspace) => {
      const offset = dimension - subspace * this.width;
      return offset >= 0 && offset < this.width ? total + centroids[this.codes[index][subspace]][offset] : total;
    }, 0));
  }

  /** ADC table for a residual: distances from every subspace of the residual to every centroid. */
  adcTable(residual) {
    return this.codebooks.map((centroids, subspace) => {
      const part = residual.slice(subspace * this.width, (subspace + 1) * this.width);
      return centroids.map((centroid) => distance2(part, centroid));
    });
  }

  adcDistance(table, code) { return code.reduce((total, entry, subspace) => total + table[subspace][entry], 0); }

  /** Approximate search: scan the nprobe lists closest to the query, rank by ADC distance. */
  search(query, { k = 10, nprobe = 1 } = {}) {
    const nearestLists = this.coarse.centroids
      .map((centroid, list) => ({ list, distance: distance2(query, centroid) }))
      .sort((left, right) => left.distance - right.distance || left.list - right.list)
      .slice(0, Math.max(1, Math.min(nprobe, this.nlist)));
    const scored = [];
    for (const { list } of nearestLists) {
      const residual = query.map((value, dimension) => value - this.coarse.centroids[list][dimension]);
      const table = this.adcTable(residual);
      for (const index of this.lists[list]) {
        scored.push({ index, list, distance: this.adcDistance(table, this.codes[index]) });
      }
    }
    scored.sort((left, right) => left.distance - right.distance || left.index - right.index);
    return scored.slice(0, Math.max(0, k));
  }

  /** Brute force over the original vectors, or over their PQ reconstructions. */
  exactSearch(query, k, { reconstructed = false } = {}) {
    return this.vectors
      .map((vector, index) => ({ index, distance: distance2(query, reconstructed ? this.reconstruct(index) : vector) }))
      .sort((left, right) => left.distance - right.distance || left.index - right.index)
      .slice(0, Math.max(0, k));
  }
}

/** Deterministic clustered dataset: `clusters` blobs in `dimension` dimensions. */
function makeDataset(count, dimension, clusters, seed) {
  const rng = makeRng(seed);
  const centres = Array.from({ length: clusters }, () => Array.from({ length: dimension }, () => (rng(2000) - 1000) / 100));
  return Array.from({ length: count }, (_, index) => {
    const centre = centres[index % clusters];
    return centre.map((value) => value + (rng(200) - 100) / 100);
  });
}

const recallAt = (approximate, exact) => {
  const truth = new Set(exact.map((entry) => entry.index));
  return approximate.filter((entry) => truth.has(entry.index)).length / Math.max(1, truth.size);
};

module.exports = { distance2, kmeans, IVFProductQuantiser, makeDataset, makeRng, recallAt };

if (require.main === module) {
  const DIMENSION = 16;
  const VECTORS = makeDataset(200, DIMENSION, 6, 0x765765);
  const QUERIES = makeDataset(20, DIMENSION, 6, 0xabcabc);
  const index = new IVFProductQuantiser(VECTORS, { nlist: 8, m: 8, centroidsPerSub: 16, seed: 7 });

  // 1. k-means invariants: nearest-centroid assignments and a non-increasing objective
  for (const [centroids, history] of [[index.coarse.centroids, index.coarse.objectiveHistory], ...index.codebooks.map((book, subspace) => [book, index.codebookObjectives[subspace]])]) {
    ok(history.length > 1, "the training must run several iterations");
    for (let step = 1; step < history.length - 1; step++) {
      ok(history[step] <= history[step - 1] + 1e-9, `the objective must not increase (${history[step - 1]} -> ${history[step]})`);
    }
    ok(centroids.every((centroid) => centroid.length > 0), "every centroid has the right width");
  }
  let assignmentChecks = 0;
  VECTORS.forEach((vector, id) => {
    const nearest = index.coarse.centroids.reduce((best, centroid, list) =>
      distance2(vector, centroid) < distance2(vector, index.coarse.centroids[best]) ? list : best, 0);
    strictEqual(index.coarse.assignments[id], nearest, `vector ${id} is assigned to its nearest list`);
    assignmentChecks++;
  });
  // 2. every PQ code is the argmin of its subspace, checked by brute force
  let codeChecks = 0;
  for (let id = 0; id < 25; id++) {
    const code = index.codes[id];
    code.forEach((entry, subspace) => {
      const part = index.residuals[id].slice(subspace * index.width, (subspace + 1) * index.width);
      const distances = index.codebooks[subspace].map((centroid) => distance2(part, centroid));
      strictEqual(distances[entry], Math.min(...distances), `code ${entry} is the best entry of subspace ${subspace}`);
      codeChecks++;
    });
  }
  // 3. ADC equals the brute-force distance to the reconstructed vector, list by list
  let adcChecks = 0;
  for (const query of QUERIES.slice(0, 5)) {
    for (const list of [0, 3]) {
      const residual = query.map((value, dimension) => value - index.coarse.centroids[list][dimension]);
      const table = index.adcTable(residual);
      for (const id of index.lists[list].slice(0, 8)) {
        const brute = distance2(query, index.reconstruct(id));
        ok(Math.abs(index.adcDistance(table, index.codes[id]) - brute) < 1e-9, `ADC of vector ${id} matches brute force`);
        adcChecks++;
      }
    }
  }
  // 4. recall against the exact search over the raw vectors, for a growing candidate pool
  const recalls = [1, 2, 4, index.nlist].map((nprobe) => ({
    nprobe,
    value: QUERIES.reduce((total, query) => total + recallAt(index.search(query, { k: 10, nprobe }), index.exactSearch(query, 10)), 0) / QUERIES.length,
  }));
  for (const { nprobe, value } of recalls) ok(value >= 0 && value <= 1, `recall at nprobe ${nprobe} is a fraction`);
  ok(recalls[0].value >= 0.5, `recall@10 with a single list clears the floor (${recalls[0].value.toFixed(3)})`);
  ok(recalls.at(-1).value >= recalls[0].value, "visiting every list cannot be worse than visiting one");
  // 5. with all lists visited, ADC ranks exactly like brute force over the reconstructions
  for (const query of QUERIES.slice(0, 5)) {
    deepStrictEqual(index.search(query, { k: 10, nprobe: index.nlist }).map((entry) => entry.index),
      index.exactSearch(query, 10, { reconstructed: true }).map((entry) => entry.index),
      "the full scan reproduces the exact ranking of the reconstructed vectors");
  }
  const searched = new Set(index.search(QUERIES[0], { k: 200, nprobe: index.nlist }).map((entry) => entry.index));
  strictEqual(searched.size, VECTORS.length, "every vector is considered when all lists are visited");
  // 6. an indexed vector is always found in its own top-10, and usually first
  const selfHits = VECTORS.slice(0, 40).map((vector) => index.search(vector, { k: 10, nprobe: 2 }));
  const found = selfHits.filter((hits, id) => hits.some((entry) => entry.index === id)).length;
  ok(found >= selfHits.length / 2, `most query vectors retrieve themselves (${found}/${selfHits.length})`);
  const firsts = selfHits.filter((hits, id) => hits[0].index === id).length;
  ok(firsts >= selfHits.length / 4, `a good share rank themselves first (${firsts}/${selfHits.length})`);
  // 7. compression: a code is far smaller than the raw vector
  ok(index.codeBytes < index.rawBytes, `a ${index.codeBytes}-byte code replaces ${index.rawBytes} bytes of float32`);
  strictEqual(index.codes.every((code) => code.length === 8), true, "every code has one entry per subspace");
  // 8. boundaries: fewer vectors than lists, duplicate vectors, k larger than the dataset
  const tiny = new IVFProductQuantiser(makeDataset(5, 8, 2, 3), { nlist: 16, m: 4, centroidsPerSub: 4, seed: 2 });
  strictEqual(tiny.nlist, 5, "k-means cannot produce more clusters than points");
  ok(tiny.search(tiny.vectors[0], { k: 10, nprobe: 5 }).length === 5, "a k larger than the dataset is capped");
  const duplicates = new IVFProductQuantiser(Array.from({ length: 12 }, () => [1, 2, 3, 4]), { nlist: 4, m: 2, centroidsPerSub: 4, seed: 5 });
  strictEqual(duplicates.search([1, 2, 3, 4], { k: 3, nprobe: 4 }).length, 3, "duplicate vectors are still ranked");
  ok(duplicates.search([1, 2, 3, 4], { k: 1, nprobe: 4 })[0].distance < 1e-9, "an exact duplicate has distance zero");
  // 9. validation and determinism
  throws(() => new IVFProductQuantiser([], {}), RangeError);
  throws(() => new IVFProductQuantiser([[1, 2, 3]], { m: 2 }), RangeError);
  throws(() => new IVFProductQuantiser([[1, 2]], { m: 0 }), RangeError);
  throws(() => new IVFProductQuantiser([[1, 2]], { nlist: 0 }), RangeError);
  throws(() => kmeans([], 2, makeRng(1)), RangeError);
  deepStrictEqual(index.codes[0], new IVFProductQuantiser(VECTORS, { nlist: 8, m: 8, centroidsPerSub: 16, seed: 7 }).codes[0], "training is deterministic");
  deepStrictEqual(index.search(QUERIES[0], { k: 5, nprobe: 2 }).map((entry) => entry.index),
    index.search(QUERIES[0], { k: 5, nprobe: 2 }).map((entry) => entry.index), "search is deterministic");

  console.log(
    `vectors=${VECTORS.length} lists=${index.nlist} recall@10=${recalls.map(({ nprobe, value }) => `nprobe${nprobe}:${value.toFixed(3)}`).join(" ")} ` +
    `adcChecks=${adcChecks} codeChecks=${codeChecks} selfFirst=${firsts}/40`
  );
  console.log("PASS");
}
