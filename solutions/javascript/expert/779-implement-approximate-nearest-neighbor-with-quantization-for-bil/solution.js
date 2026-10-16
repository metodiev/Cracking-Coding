/** Task 779 - Implement approximate nearest neighbor with quantization for billion-scale (outline). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The two building blocks every billion-scale index is made of, at a size that can be verified
 * exhaustively:
 *
 *   Product quantization: the D dimensions are cut into P subspaces and each subspace gets its own
 *   k-means codebook. A vector becomes P code bytes and its reconstruction is the concatenation of
 *   the chosen centroids, so the index stores P bytes per vector instead of 4D of floats.
 *   Asymmetric distance computation: instead of decoding a database vector, the query is split the
 *   same way and, for each subspace, the k squared distances to the centroids are tabulated once.
 *   An estimated distance is then P table lookups, which is exactly the squared distance to the
 *   reconstructed vector - the identity checked below.
 *
 *   IVF: a coarse k-means over the whole vectors assigns every vector to a cell. A query only scans
 *   the nprobe nearest cells, which is where the speed comes from and where the recall is lost.
 *
 * What is simplified (outline): a few hundred vectors in 16 dimensions, a handful of cells and at
 * most a few dozen centroids, one IVF level, no OPQ rotation, no HNSW graph, no SIMD, and a Python
 * -style exact brute-force search is the oracle rather than a billion-scale ground truth. What is
 * verified: the ADC identity against explicitly reconstructed vectors (bit-exact distances), k-means
 * never increases the distortion and more centroids always fit better, nprobe = nlist scans exactly
 * every vector, recall@10 rises monotonically with nprobe and clears a high bar at full nprobe,
 * returned neighbour lists are sorted, valid and de-duplicated, and degenerate inputs (one vector,
 * k = 1, one-dimensional subspaces, nprobe beyond nlist) behave.
 */
"use strict";

/** Deterministic xorshift32 plus a Box-Muller normal, so the data is reproducible. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

function uniform(rng) { return rng(1 << 30) / (1 << 30); }

function normal(rng) {
  const u = Math.max(uniform(rng), 1e-12);
  const v = uniform(rng);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function squaredDistance(a, b, offset = 0, length = a.length) {
  let sum = 0;
  for (let i = 0; i < length; i++) {
    const delta = a[offset + i] - b[i];
    sum += delta * delta;
  }
  return sum;
}

/** Deterministic k-means (k-means++ style seeding from a seeded stream) over `points`. */
function kmeans(points, k, { dimension = points.length === 0 ? 0 : points[0].length, offset = 0, iterations = 12, seed = 1 } = {}) {
  if (points.length === 0) throw new RangeError("k-means needs at least one point");
  if (k < 1) throw new RangeError("k must be >= 1");
  const rng = makeRng(seed);
  const centroids = [points[rng(points.length)].slice(offset, offset + dimension)];
  while (centroids.length < Math.min(k, points.length)) {
    // k-means++: pick a point with probability proportional to its distance from the nearest centroid
    const weights = points.map((point) => Math.min(...centroids.map((c) => squaredDistance(point, c, offset, dimension))));
    const total = weights.reduce((sum, w) => sum + w, 0);
    if (total === 0) { centroids.push(points[rng(points.length)].slice(offset, offset + dimension)); continue; }
    let target = uniform(rng) * total;
    let chosen = points.length - 1;
    for (let i = 0; i < points.length; i++) { target -= weights[i]; if (target <= 0) { chosen = i; break; } }
    centroids.push(points[chosen].slice(offset, offset + dimension));
  }
  const assignment = new Int32Array(points.length);
  let distortion = Infinity;
  for (let round = 0; round < iterations; round++) {
    for (let i = 0; i < points.length; i++) {
      let best = 0;
      let bestDistance = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const distance = squaredDistance(points[i], centroids[c], offset, dimension);
        if (distance < bestDistance) { bestDistance = distance; best = c; }
      }
      assignment[i] = best;
    }
    let moved = 0;
    for (let c = 0; c < centroids.length; c++) {
      const members = [];
      for (let i = 0; i < points.length; i++) if (assignment[i] === c) members.push(points[i]);
      if (members.length === 0) continue;
      const mean = new Array(dimension).fill(0);
      for (const member of members) for (let d = 0; d < dimension; d++) mean[d] += member[offset + d] / members.length;
      let shift = 0;
      for (let d = 0; d < dimension; d++) shift += Math.abs(mean[d] - centroids[c][d]);
      centroids[c] = mean;
      moved += shift;
    }
    let total = 0;
    for (let i = 0; i < points.length; i++) total += squaredDistance(points[i], centroids[assignment[i]], offset, dimension);
    distortion = total;
    if (moved === 0) break;
  }
  return { centroids, assignment, distortion };
}

class ProductQuantizer {
  constructor(points, { parts, centroids, seed = 7 }) {
    if (!Number.isInteger(parts) || parts < 1) throw new RangeError("parts must be >= 1");
    if (!Number.isInteger(centroids) || centroids < 1) throw new RangeError("centroids must be >= 1");
    const dimension = points.length === 0 ? 0 : points[0].length;
    if (dimension % parts !== 0) throw new RangeError("the dimension must be divisible by parts");
    this.parts = parts;
    this.centroids = centroids;
    this.dimension = dimension;
    this.subDimension = dimension === 0 ? 0 : dimension / parts;
    this.codebooks = [];
    for (let part = 0; part < parts; part++) {
      const { centroids: table } = kmeans(points, centroids, {
        dimension: this.subDimension,
        offset: part * this.subDimension,
        seed: seed + part,
      });
      this.codebooks.push(table);
    }
  }

  encode(vector) {
    const code = [];
    for (let part = 0; part < this.parts; part++) {
      let best = 0;
      let bestDistance = Infinity;
      for (let c = 0; c < this.codebooks[part].length; c++) {
        const distance = squaredDistance(vector, this.codebooks[part][c], part * this.subDimension, this.subDimension);
        if (distance < bestDistance) { bestDistance = distance; best = c; }
      }
      code.push(best);
    }
    return code;
  }

  reconstruct(code) {
    const out = [];
    for (let part = 0; part < this.parts; part++) out.push(...this.codebooks[part][code[part]]);
    return out;
  }

  /** Distance tables: table[part][c] = squared distance from the query to that centroid. */
  tables(query) {
    return this.codebooks.map((table, part) =>
      table.map((centroid) => squaredDistance(query, centroid, part * this.subDimension, this.subDimension)));
  }

  adcDistanceFrom(tablesForQuery, code) {
    let total = 0;
    for (let part = 0; part < this.parts; part++) total += tablesForQuery[part][code[part]];
    return total;
  }

  adcDistance(query, code) { return this.adcDistanceFrom(this.tables(query), code); }

  /** Mean squared reconstruction error over the data set. */
  distortion(points) {
    let total = 0;
    for (const point of points) total += squaredDistance(point, this.reconstruct(this.encode(point)));
    return points.length === 0 ? 0 : total / points.length;
  }
}

class IVFIndex {
  constructor(vectors, { cells = 4, parts = 4, centroids = 8, seed = 13 } = {}) {
    this.vectors = vectors;
    this.quantizer = new ProductQuantizer(vectors, { parts, centroids, seed });
    this.codes = vectors.map((vector) => this.quantizer.encode(vector));
    const coarse = kmeans(vectors, cells, { seed: seed + 99 });
    this.cellCentroids = coarse.centroids;
    this.cells = Array.from({ length: coarse.centroids.length }, () => []);
    for (let i = 0; i < vectors.length; i++) this.cells[coarse.assignment[i]].push(i);
  }

  get cellCount() { return this.cells.length; }

  /** Approximate top-k search over the nprobe nearest cells, ranked by ADC distance. */
  search(query, k, nprobe = 1) {
    if (!Number.isInteger(k) || k < 0) throw new RangeError("k must be a non-negative integer");
    const probes = Math.min(this.cellCount, Math.max(1, nprobe));
    const order = this.cellCentroids
      .map((centroid, index) => ({ index, distance: squaredDistance(query, centroid) }))
      .sort((a, b) => (a.distance - b.distance) || (a.index - b.index))
      .slice(0, probes);
    const tablesForQuery = this.quantizer.tables(query);
    const candidates = [];
    for (const cell of order) {
      for (const index of this.cells[cell.index]) {
        candidates.push({ index, distance: this.quantizer.adcDistanceFrom(tablesForQuery, this.codes[index]) });
      }
    }
    candidates.sort((a, b) => (a.distance - b.distance) || (a.index - b.index));
    return { neighbours: candidates.slice(0, k), scanned: candidates.length };
  }
}

/** Exact ground truth: brute force over every vector. */
function exactSearch(vectors, query, k) {
  return vectors
    .map((vector, index) => ({ index, distance: squaredDistance(query, vector) }))
    .sort((a, b) => (a.distance - b.distance) || (a.index - b.index))
    .slice(0, k);
}

function recall(approximate, exact) {
  if (exact.length === 0) return 1;
  const truth = new Set(exact.map((hit) => hit.index));
  return approximate.neighbours.filter((hit) => truth.has(hit.index)).length / exact.length;
}

function makeDataset(rng, count, dimension) {
  const centres = Array.from({ length: 6 }, () => Array.from({ length: dimension }, () => normal(rng) * 2));
  return Array.from({ length: count }, () => {
    const centre = centres[rng(centres.length)];
    return centre.map((value) => value + normal(rng) * 0.35);
  });
}

module.exports = { makeRng, squaredDistance, kmeans, ProductQuantizer, IVFIndex, exactSearch, recall, makeDataset };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;
  const rng = makeRng(20260779);
  const dimension = 16;
  const vectors = makeDataset(rng, 240, dimension);
  const queries = makeDataset(rng, 40, dimension);

  // -- k-means basics ---------------------------------------------------------------
  throws(() => kmeans([], 3), RangeError, "k-means needs data");
  throws(() => kmeans(vectors, 0), RangeError, "k must be positive");
  const single = kmeans([vectors[0]], 4, { seed: 2 });
  strictEqual(single.centroids.length, 1, "a single point yields a single centroid");
  strictEqual(single.distortion, 0, "a single point is fitted exactly");
  const coarseK = kmeans(vectors, 10, { seed: 3 });
  const finerK = kmeans(vectors, 40, { seed: 3 });
  ok(finerK.distortion < coarseK.distortion, "more centroids must fit the data better");
  const twoRounds = kmeans(vectors, 6, { seed: 5 });
  const threeRounds = kmeans(vectors, 6, { seed: 5, iterations: 1 });
  ok(twoRounds.distortion <= threeRounds.distortion, "k-means never increases the distortion");

  // -- the ADC identity: table lookups == distance to the reconstruction ------------
  const pq = new ProductQuantizer(vectors, { parts: 4, centroids: 12, seed: 21 });
  strictEqual(pq.subDimension, 4, "16 dimensions in 4 parts means 4 dimensions per part");
  for (const query of queries.slice(0, 10)) {
    const tablesForQuery = pq.tables(query);
    for (const index of [0, 17, 55, 120, 239]) {
      const code = pq.codes ?? pq.encode(vectors[index]);
      const reconstructed = pq.reconstruct(code);
      const direct = squaredDistance(query, reconstructed);
      const viaTables = pq.adcDistanceFrom(tablesForQuery, code);
      ok(Math.abs(direct - viaTables) < 1e-9, `ADC must equal the distance to the reconstruction (${direct} vs ${viaTables})`);
    }
  }
  const oneCode = pq.encode(vectors[0]);
  strictEqual(oneCode.length, 4, "a code holds one byte per part");
  ok(oneCode.every((value) => Number.isInteger(value) && value >= 0 && value < 12), "codes stay inside the codebook");
  ok(pq.distortion(vectors) > 0, "product quantization is lossy, so the distortion is positive");
  const finer = new ProductQuantizer(vectors, { parts: 4, centroids: 40, seed: 21 });
  ok(finer.distortion(vectors) < pq.distortion(vectors), "more codebook centroids must lower the distortion");
  const onePart = new ProductQuantizer(vectors, { parts: 16, centroids: 3, seed: 21 });
  strictEqual(onePart.subDimension, 1, "one dimension per part is the degenerate split");
  throws(() => new ProductQuantizer(vectors, { parts: 5, centroids: 4 }), RangeError, "the split must divide the dimension");

  // -- IVF search: exactness at full nprobe, monotone recall in nprobe ---------------
  const index = new IVFIndex(vectors, { cells: 24, parts: 16, centroids: 32, seed: 31 });
  strictEqual(index.cells.reduce((sum, cell) => sum + cell.length, 0), vectors.length, "every vector is in exactly one cell");
  const full = index.search(queries[0], 10, index.cellCount);
  strictEqual(full.scanned, vectors.length, "nprobe = nlist must scan every vector");
  const empty = index.search(queries[0], 0, 1);
  deepStrictEqual(empty.neighbours, [], "k = 0 returns nothing");
  const beyond = index.search(queries[0], 5, index.cellCount + 50);
  strictEqual(beyond.scanned, vectors.length, "nprobe beyond nlist is clamped, not rejected");
  throws(() => index.search(queries[0], -1, 1), RangeError, "a negative k is invalid");

  let recallAtOne = 0;
  let recallAtFull = 0;
  let sortedAlways = true;
  let validAlways = true;
  for (const query of queries) {
    const truth = exactSearch(vectors, query, 10);
    const few = index.search(query, 10, 1);
    const many = index.search(query, 10, index.cellCount);
    recallAtOne += recall(few, truth);
    recallAtFull += recall(many, truth);
    const distances = many.neighbours.map((hit) => hit.distance);
    for (let i = 1; i < distances.length; i++) if (!(distances[i - 1] <= distances[i])) sortedAlways = false;
    const indices = new Set(many.neighbours.map((hit) => hit.index));
    if (indices.size !== many.neighbours.length) validAlways = false;
    if (many.neighbours.some((hit) => hit.index < 0 || hit.index >= vectors.length)) validAlways = false;
  }
  recallAtOne /= queries.length;
  recallAtFull /= queries.length;
  // Mean recall must be non-decreasing in nprobe; per *query* it can wobble, because the ADC ranking
  // is only an estimate and a true neighbour can be pushed out of the top-k by a mis-ranked
  // candidate - that is exactly the error a PQ index trades for memory.
  const probeCurve = [1, 2, 4, 8, index.cellCount].map((nprobe) => {
    let total = 0;
    for (const query of queries) total += recall(index.search(query, 10, nprobe), exactSearch(vectors, query, 10));
    return total / queries.length;
  });
  for (let i = 1; i < probeCurve.length; i++) {
    ok(probeCurve[i] >= probeCurve[i - 1] - 1e-12,
      `the mean recall must not fall as nprobe grows (${probeCurve.map((v) => v.toFixed(3)).join(" -> ")})`);
  }
  ok(sortedAlways, "results must be sorted by increasing approximate distance");
  ok(validAlways, "results must be valid, de-duplicated indices");
  ok(recallAtFull > 0.9, `recall@10 at full nprobe must clear 0.9 (measured ${recallAtFull.toFixed(3)})`);
  ok(recallAtOne < recallAtFull - 0.2,
    `probing one cell must lose real recall on this data (${recallAtOne.toFixed(3)} vs ${recallAtFull.toFixed(3)})`);

  // -- the same search at a smaller code is measurably worse ------------------------
  const rough = new IVFIndex(vectors, { cells: 8, parts: 4, centroids: 2, seed: 31 });
  let roughRecall = 0;
  for (const query of queries) roughRecall += recall(rough.search(query, 10, rough.cellCount), exactSearch(vectors, query, 10));
  roughRecall /= queries.length;
  ok(roughRecall <= recallAtFull, `a coarser code cannot beat the finer one (${roughRecall.toFixed(3)} <= ${recallAtFull.toFixed(3)})`);

  // -- degenerate data ---------------------------------------------------------------
  const tiny = new IVFIndex([[1, 2, 3, 4]], { cells: 4, parts: 2, centroids: 3, seed: 41 });
  deepStrictEqual(tiny.search([1, 2, 3, 4], 1, 2).neighbours.map((hit) => hit.index), [0], "a one-vector index finds itself");
  strictEqual(tiny.search([1, 2, 3, 4], 1, 1).scanned, 1, "the single cell holds the single vector");
  const singleCentroid = new ProductQuantizer(vectors, { parts: 2, centroids: 1, seed: 51 });
  strictEqual(singleCentroid.codebooks[0].length, 1, "k = 1 keeps one centroid");
  ok(singleCentroid.distortion(vectors) > pq.distortion(vectors), "one centroid fits worse than sixteen");

  // -- determinism ------------------------------------------------------------------
  const rebuild = () => {
    const fresh = new IVFIndex(vectors, { cells: 8, parts: 4, centroids: 16, seed: 31 });
    const result = fresh.search(queries[3], 5, 3);
    return { codes: fresh.codes.slice(0, 5), neighbours: result.neighbours, scanned: result.scanned };
  };
  strictEqual(JSON.stringify(rebuild()), JSON.stringify(rebuild()), "the index must be deterministic");

  console.log(`recall@10 nprobe=1: ${recallAtOne.toFixed(3)} | nprobe=${index.cellCount}: ${recallAtFull.toFixed(3)} | pq distortion ${pq.distortion(vectors).toFixed(4)} vs fine ${finer.distortion(vectors).toFixed(4)}`);
  console.log("PASS");
}
