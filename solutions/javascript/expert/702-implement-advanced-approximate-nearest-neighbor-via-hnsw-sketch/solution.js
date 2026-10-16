/** Task 702 - Implement advanced approximate nearest neighbor via HNSW sketch. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Hierarchical Navigable Small World index (Malkov & Yashunin). Every point draws a top level
 * from the geometric law P(level >= l) = m^-l, so the upper layers thin out exponentially; layer
 * l of a point with level >= l keeps at most M_l links (M_0 = 2M at the bottom). Insertion
 * descends greedily through the layers above the point's own level, then runs an efConstruction
 * beam search per layer, linking to the M diversified neighbours - a candidate survives only
 * when it is closer to the query than to every already chosen neighbour - and prunes the
 * neighbour lists with the same rule. Search descends greedily to layer 0 and returns the best k
 * of an ef-wide beam, so a query costs O(log n) + O(ef) distance evaluations instead of a scan.
 *
 * What is simplified: a seeded PRNG makes construction and the synthetic data deterministic;
 * there are no deletions, no parallel construction, and pruning keeps the link set symmetric
 * (the paper tolerates asymmetric links - symmetry is the stronger invariant).
 * Invariant verified: links are symmetric and inside the degree bounds; a point of level L is
 * present in every layer 0..L; layer 0 is reachable from the entry point; results are distinct,
 * distance-sorted and inside the index; recall@k over held-out queries against the exhaustive
 * oracle is high and non-decreasing in ef; identical seeds rebuild the identical graph.
 */
"use strict";
const { ok, strictEqual, deepStrictEqual, throws } = require("node:assert");
/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
function l2(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; sum += d * d; }
  return sum;
}
/** Binary heap over (score, value) pairs; `maxFirst` makes it a worst-element heap. */
class Heap {
  constructor(maxFirst = false) {
    this.items = [];
    this.maxFirst = maxFirst;
    this.sequence = 0;
  }
  get size() { return this.items.length; }
  peek() { return this.items[0]; }
  /** Insertion-order tie break keeps every run of the program identical. */
  before(a, b) {
    if (a.score !== b.score) return this.maxFirst ? a.score > b.score : a.score < b.score;
    return a.sequence < b.sequence;
  }
  push(score, value) {
    this.items.push({ score, value, sequence: this.sequence++ });
    for (let i = this.items.length - 1; i > 0;) {
      const parent = (i - 1) >> 1;
      if (this.before(this.items[parent], this.items[i])) break;
      [this.items[parent], this.items[i]] = [this.items[i], this.items[parent]];
      i = parent;
    }
  }
  pop() {
    const top = this.items[0];
    const last = this.items.pop();
    if (this.items.length === 0) return top;
    this.items[0] = last;
    for (let i = 0;;) {
      const left = 2 * i + 1;
      const right = left + 1;
      let best = i;
      if (left < this.items.length && this.before(this.items[left], this.items[best])) best = left;
      if (right < this.items.length && this.before(this.items[right], this.items[best])) best = right;
      if (best === i) break;
      [this.items[best], this.items[i]] = [this.items[i], this.items[best]];
      i = best;
    }
    return top;
  }
}
class HnswIndex {
  constructor({ dim, m = 8, efConstruction = 32, seed = 1 }) {
    if (!Number.isInteger(dim) || dim <= 0) throw new RangeError("dim must be a positive integer");
    if (!Number.isInteger(m) || m < 2) throw new RangeError("m must be an integer >= 2");
    if (!Number.isInteger(efConstruction) || efConstruction < m) throw new RangeError("efConstruction < m");
    this.dim = dim;
    this.m = m;
    this.m0 = 2 * m;
    this.efConstruction = efConstruction;
    this.levelNorm = 1 / Math.log(m);
    this.rng = makeRng(seed);
    this.points = [];
    this.levels = [];
    this.links = [];
    this.entry = -1;
    this.topLevel = -1;
    this.distanceCalls = 0;
  }

  get size() { return this.points.length; }
  /** P(level >= l) = m^-l, sampled by inverting the geometric distribution. */
  randomLevel() {
    const u = (this.rng(1000000) + 0.5) / 1000000; // uniform in (0, 1)
    return Math.min(31, Math.floor(-Math.log(u) * this.levelNorm));
  }
  #distance(id, vector) {
    this.distanceCalls++;
    return l2(this.points[id], vector);
  }
  /** Greedy descent inside one layer: keep moving to the closest neighbour. */
  #greedy(vector, entry, level) {
    let current = entry, currentDist = this.#distance(current, vector);
    for (let improved = true; improved;) {
      improved = false;
      for (const nb of this.links[current][level]) {
        const d = this.#distance(nb, vector);
        if (d < currentDist) { current = nb; currentDist = d; improved = true; }
      }
    }
    return current;
  }
  /** Beam search inside one layer; returns the ef closest points sorted by distance. */
  searchLayer(vector, entryPoints, ef, level) {
    const visited = new Set();
    const frontier = new Heap(false);
    const results = new Heap(true);
    for (const ep of entryPoints) {
      if (visited.has(ep)) continue;
      visited.add(ep);
      const d = this.#distance(ep, vector);
      frontier.push(d, ep); results.push(d, ep);
    }
    while (frontier.size > 0) {
      const { score, value: id } = frontier.pop();
      if (results.size >= ef && score > results.peek().score) break; // nothing closer can remain
      for (const nb of this.links[id][level]) {
        if (visited.has(nb)) continue;
        visited.add(nb);
        const d = this.#distance(nb, vector);
        if (results.size < ef || d < results.peek().score) {
          frontier.push(d, nb);
          results.push(d, nb);
          if (results.size > ef) results.pop();
        }
      }
    }
    return results.items
      .map(({ score, value }) => ({ id: value, dist: score }))
      .sort((a, b) => a.dist - b.dist || a.id - b.id);
  }
  /** Diversified selection: a candidate survives when it is not closer to a chosen one. */
  selectNeighbours(vector, candidates, limit) {
    const chosen = [];
    const rejected = [];
    for (const candidate of candidates) {
      if (chosen.length >= limit) break;
      const keep = chosen.every((s) => l2(this.points[candidate.id], this.points[s.id]) >= candidate.dist);
      (keep ? chosen : rejected).push(candidate);
    }
    for (const candidate of rejected) { // keepPrunedConnections fallback
      if (chosen.length >= limit) break;
      chosen.push(candidate);
    }
    return chosen;
  }
  /** Re-select an overflowing neighbour list and drop the reverse half of discarded links. */
  #prune(id, level, limit) {
    const list = this.links[id][level];
    if (list.length <= limit) return;
    const vector = this.points[id];
    const candidates = list
      .map((nb) => ({ id: nb, dist: l2(vector, this.points[nb]) }))
      .sort((a, b) => a.dist - b.dist || a.id - b.id);
    const kept = this.selectNeighbours(vector, candidates, limit).map((c) => c.id);
    const survivors = new Set(kept);
    for (const nb of list) {
      if (!survivors.has(nb)) this.links[nb][level] = this.links[nb][level].filter((x) => x !== id);
    }
    this.links[id][level] = kept;
  }
  insert(vector) {
    if (!Array.isArray(vector) || vector.length !== this.dim) {
      throw new RangeError(`point must be an array of ${this.dim} numbers`);
    }
    const id = this.points.length;
    this.points.push(vector.slice());
    const level = this.randomLevel();
    this.levels.push(level);
    this.links.push(Array.from({ length: level + 1 }, () => []));
    if (this.entry === -1) {
      this.entry = id;
      this.topLevel = level;
      return id;
    }
    let ep = this.entry;
    for (let lc = this.topLevel; lc > level; lc--) ep = this.#greedy(vector, ep, lc);
    for (let lc = Math.min(level, this.topLevel); lc >= 0; lc--) {
      const candidates = this.searchLayer(vector, [ep], this.efConstruction, lc);
      const limit = lc === 0 ? this.m0 : this.m;
      const chosen = this.selectNeighbours(vector, candidates, limit);
      this.links[id][lc] = chosen.map((c) => c.id);
      for (const c of chosen) {
        this.links[c.id][lc].push(id);
        this.#prune(c.id, lc, limit);
        if (!this.links[c.id][lc].includes(id)) {
          this.links[id][lc] = this.links[id][lc].filter((x) => x !== c.id); // stay symmetric
        }
      }
      if (candidates.length > 0) ep = candidates[0].id;
    }
    if (level > this.topLevel) {
      this.topLevel = level;
      this.entry = id;
    }
    return id;
  }
  search(vector, k, ef = 50) {
    if (!Number.isInteger(k) || k <= 0) throw new RangeError("k must be a positive integer");
    if (!Array.isArray(vector) || vector.length !== this.dim) throw new RangeError("bad query");
    if (this.entry === -1) return [];
    let ep = this.entry;
    for (let lc = this.topLevel; lc > 0; lc--) ep = this.#greedy(vector, ep, lc);
    return this.searchLayer(vector, [ep], Math.max(ef, k), 0).slice(0, k);
  }
  /** Structural invariants; returns a description of the first violation, or null. */
  checkStructure() {
    for (let id = 0; id < this.size; id++) {
      for (let lc = 0; lc < this.links[id].length; lc++) {
        const limit = lc === 0 ? this.m0 : this.m;
        const list = this.links[id][lc];
        if (list.length > limit) return `degree ${list.length} exceeds ${limit}`;
        if (new Set(list).size !== list.length) return "duplicate link";
        for (const nb of list) {
          if (nb === id) return "self link";
          if (this.levels[nb] < lc) return `link to ${nb} which has no layer ${lc}`;
          if (!this.links[nb][lc].includes(id)) return `link ${id}->${nb} at layer ${lc} is not mirrored`;
        }
      }
    }
    if (this.entry === -1) return this.size === 0 ? null : "no entry point";
    const seen = new Set([this.entry]);
    const queue = [this.entry];
    while (queue.length > 0) {
      for (const nb of this.links[queue.pop()][0]) {
        if (!seen.has(nb)) { seen.add(nb); queue.push(nb); }
      }
    }
    return seen.size === this.size ? null : `layer 0 reaches ${seen.size} of ${this.size} points`;
  }
  snapshot() {
    return this.links.map((layers) => layers.map((list) => list.join(",")).join(";")).join("|");
  }
}
/** Exhaustive k-NN oracle: the ground truth for every recall measurement. */
function bruteKnn(points, vector, k) {
  const ranked = points.map((p, id) => ({ id, dist: l2(p, vector) }));
  return ranked.sort((a, b) => a.dist - b.dist || a.id - b.id).slice(0, k);
}
/** Cluster of tight blobs around random centres; `spread` scales the per-blob radius. */
function clusteredPoints(rng, count, dim, blobs, spread) {
  const centres = Array.from({ length: blobs }, () =>
    Array.from({ length: dim }, () => rng(2000) / 100 - 10));
  const points = [];
  for (let i = 0; i < count; i++) {
    const centre = centres[i % blobs];
    points.push(centre.map((c) => c + (rng(2001) - 1000) / (100 * spread)));
  }
  return points;
}
/** Build an index over `points` with the given options; returns the completed index. */
function buildIndex(points, options) {
  const index = new HnswIndex({ dim: points[0].length, ...options });
  for (const point of points) index.insert(point);
  return index;
}
/** Recall@k over held-out `queries`, with `points` giving the exhaustive ground truth. */
function measureRecall(index, points, queries, k, ef) {
  let found = 0;
  for (const query of queries) {
    const truth = new Set(bruteKnn(points, query, k).map((r) => r.id));
    for (const hit of index.search(query, k, ef)) if (truth.has(hit.id)) found++;
  }
  return found / (queries.length * k);
}
module.exports = { makeRng, l2, Heap, HnswIndex, bruteKnn, clusteredPoints, buildIndex, measureRecall };

if (require.main === module) {
  // -- degenerate configurations and rejected input ------------------------------------------
  const empty = new HnswIndex({ dim: 3 });
  deepStrictEqual([empty.search([0, 0, 0], 5), empty.checkStructure()], [[], null], "empty index");
  throws(() => empty.search([0, 0, 0], 0), RangeError);
  for (const bad of [() => new HnswIndex({ dim: 0 }), () => new HnswIndex({ dim: 3, m: 1 }),
    () => new HnswIndex({ dim: 3, m: 4, efConstruction: 2 }), () => new HnswIndex({ dim: 3 }).insert([1, 2]),
    () => new HnswIndex({ dim: 2 }).search([1, 2], 1.5)]) {
    throws(bad, RangeError);
  }

  // -- single point, hand-checked plane, duplicates, extreme coordinates ---------------------
  const single = new HnswIndex({ dim: 2, m: 4, efConstruction: 8, seed: 3 });
  strictEqual(single.insert([1, 2]), 0);
  deepStrictEqual(single.search([1, 2], 3), [{ id: 0, dist: 0 }], "one point, one hit");
  strictEqual(single.checkStructure(), null);
  const plane = [[0, 0], [1, 0], [10, 0], [10, 10], [9, 10], [0, 10]];
  const tiny = new HnswIndex({ dim: 2, m: 2, efConstruction: 4, seed: 11 });
  for (const point of plane) tiny.insert(point);
  strictEqual(tiny.checkStructure(), null);
  for (const query of [[0.2, 0.1], [9.5, 9], [-4, 3]]) {
    const hits = tiny.search(query, 3, 10);
    deepStrictEqual(hits.map((h) => h.id), bruteKnn(plane, query, 3).map((h) => h.id), `query ${query}`);
    for (let i = 1; i < hits.length; i++) ok(hits[i - 1].dist <= hits[i].dist, "sorted by distance");
  }
  strictEqual(tiny.search([0, 0], 1)[0].id, 0, "an indexed point is its own nearest neighbour");
  strictEqual(tiny.search([0, 0], 99).length, plane.length, "k beyond the index returns everything");
  const dup = new HnswIndex({ dim: 2, m: 4, efConstruction: 8, seed: 5 });
  for (let i = 0; i < 12; i++) dup.insert([2, 2]);
  const dupHits = dup.search([2, 2], 5);
  ok(dupHits.length === 5 && dupHits.every((h) => h.dist === 0), "duplicates sit at distance zero");
  ok(dupHits.every((h, i) => i === 0 || h.id > dupHits[i - 1].id), "ties break by id");
  const extremes = new HnswIndex({ dim: 2, m: 4, efConstruction: 8, seed: 6 });
  extremes.insert([-1e6, 1e6]); extremes.insert([1e6, -1e6]); extremes.insert([0, 0]);
  deepStrictEqual(extremes.search([1e6, -1e6], 2).map((h) => h.id), [1, 2], "self, then origin");

  // -- recall over held-out queries against the exhaustive oracle, growth with ef -------------
  const points = clusteredPoints(makeRng(202), 1000, 8, 20, 0.6); // 20 overlapping blobs
  const queries = clusteredPoints(makeRng(999), 100, 8, 20, 0.6);
  const index = buildIndex(points, { m: 8, efConstruction: 48, seed: 7 });
  strictEqual(index.checkStructure(), null, "invariants must hold after 1000 insertions");
  const recalls = [5, 20, 60].map((ef) => measureRecall(index, points, queries, 10, ef));
  ok(recalls[0] >= 0.8, `recall@10 at ef=5 should be high, got ${recalls[0]}`);
  ok(recalls[1] >= recalls[0] && recalls[2] >= recalls[1], "recall must not fall as ef grows");
  ok(recalls[2] >= 0.99, `recall@10 at ef=60 should be near perfect, got ${recalls[2]}`);
  const recallOne = measureRecall(index, points, queries, 1, 40);
  ok(recallOne >= 0.95, `the exact nearest neighbour must almost always be found, got ${recallOne}`);

  // -- a descent evaluates a small fraction of the distances of a full scan -------------------
  index.distanceCalls = 0;
  index.search(queries[0], 10, 10);
  const perQuery = index.distanceCalls;
  ok(perQuery > 0 && perQuery < points.length / 4, `a descent used ${perQuery} of ${points.length}`);

  // -- determinism, and the geometric level distribution -------------------------------------
  const snapshot = index.snapshot();
  const again = buildIndex(points, { m: 8, efConstruction: 48, seed: 7 });
  strictEqual(again.snapshot(), snapshot, "identical seeds must give identical graphs");
  deepStrictEqual(again.search(queries[0], 10, 40), index.search(queries[0], 10, 40));
  strictEqual(buildIndex(points, { m: 8, efConstruction: 48, seed: 8 }).checkStructure(), null);
  const inLayerOne = index.levels.filter((l) => l >= 1).length;
  ok(index.topLevel >= 1, "1000 points must produce more than one layer");
  ok(inLayerOne > 0 && inLayerOne < 3 * (points.length / 8), `layer 1 holds ${inLayerOne}`);

  console.log(`points=${points.length} queries=${queries.length} layers=${index.topLevel + 1} layer1=${inLayerOne} distances/query=${perQuery}`);
  console.log(`recall@10 ef=5/20/60 = ${recalls.map((r) => r.toFixed(3)).join("/")}; recall@1 = ${recallOne.toFixed(3)}`);
  console.log("PASS");
}
