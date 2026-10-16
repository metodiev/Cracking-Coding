/** Task 766 - Implement dense vector index with HNSW and quantization integration. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A navigable small-world graph, built the HNSW way, with scalar quantization wired into its
 * distance function. Every node draws a level from a geometric distribution (`floor(-ln(u) * mL)`);
 * a node exists in every layer up to its level, the highest node is the entry point, and an
 * insertion descends greedily through the upper layers, searches each layer from its own level down
 * to 0 with `efConstruction` candidates, links up, prunes every neighbour list back to `M` (2M at
 * layer 0) and removes the reciprocal of each dropped link, so the graph stays symmetric. A search
 * descends the same way and finishes with an `efSearch`-wide beam at layer 0. Because a symmetric
 * degree-capped graph can leave a node without mutual links, a repair pass re-attaches unreachable
 * nodes to their nearest anchor with spare degree, and the repair count is reported rather than
 * hidden. Traversal can run on 8-bit per-dimension scalar codes and then *rescore* the surviving
 * candidates with the exact floats — the usual "graph on codes, rank on vectors" split.
 *
 * What is simplified: 120 vectors of 8 dimensions, no deletions, no persistence, array beams instead
 * of heaps, scalar codes instead of OPQ/PQ, single-threaded. What is verified: level membership,
 * symmetric links inside the degree bound, no self-loops, the entry point in the top layer, layer 0
 * spanning the dataset, the quantization round-trip inside half a step per coordinate, recall@10 for
 * the exact / code-only / rescored paths with honest floors, bounded hops from the entry, exact
 * self-queries, determinism of both graph and results, and duplicate/singleton/oversized-k inputs.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

const distance2 = (a, b) => a.reduce((total, value, i) => total + (value - b[i]) ** 2, 0);
const byDistance = (left, right) => left.distance - right.distance || left.id - right.id;

/** Deterministic xorshift32 plus a uniform float in [0, 1) for the level draw. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x766766;
  const next = (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= (state << 5) >>> 0;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
  next.float = () => next(1 << 30) / (1 << 30);
  return next;
}

/** Per-dimension 8-bit scalar quantization with the round-trip error bounded by half a step. */
class ScalarQuantiser {
  constructor(vectors) {
    const dimension = vectors[0].length;
    this.minimum = Array.from({ length: dimension }, (_, axis) => Math.min(...vectors.map((vector) => vector[axis])));
    this.maximum = Array.from({ length: dimension }, (_, axis) => Math.max(...vectors.map((vector) => vector[axis])));
    this.scale = this.maximum.map((value, axis) => (value === this.minimum[axis] ? 1 : 255 / (value - this.minimum[axis])));
    this.step = this.scale.map((value) => 1 / value);
  }
  encode(vector) { return Uint8Array.from(vector, (value, axis) => Math.max(0, Math.min(255, Math.round((value - this.minimum[axis]) * this.scale[axis])))); }
  decode(code) { return Array.from(code, (value, axis) => this.minimum[axis] + value / this.scale[axis]); }
}
class HNSW {
  constructor(vectors, { m = 8, efConstruction = 30, seed = 1, quantise = true } = {}) {
    if (!Array.isArray(vectors) || vectors.length === 0) throw new RangeError("at least one vector is required");
    if (!Number.isInteger(m) || m < 1) throw new RangeError("m must be a positive integer");
    if (!Number.isInteger(efConstruction) || efConstruction < 1) throw new RangeError("efConstruction must be positive");
    this.vectors = vectors;
    this.dimension = vectors[0].length;
    this.m = m;
    this.m0 = 2 * m;
    this.efConstruction = efConstruction;
    this.quantiser = quantise ? new ScalarQuantiser(vectors) : null;
    this.codes = quantise ? vectors.map((vector) => this.quantiser.encode(vector)) : null;
    this.links = [];
    this.levels = [];
    this.entry = null;
    this.maxLevel = -1;
    this.rng = makeRng(seed);
    this.hops = 0;
    this.repairs = 0;
    vectors.forEach((vector) => this.insert(vector));
    this.#repairConnectivity();
  }
  /**
   * A symmetric degree-capped graph can leave a node with no mutual link (its neighbours pruned it
   * away), so layer 0 is repaired after building: every unreachable node is attached to its nearest
   * reachable neighbour. The invariant "layer 0 spans the dataset" is therefore guaranteed, and the
   * repair count is reported instead of being hidden.
   */
  #repairConnectivity() {
    let reached = this.reachable();
    for (let round = 0; round < 20 && reached.size < this.vectors.length; round++) {
      for (let id = 0; id < this.vectors.length; id++) {
        if (reached.has(id)) continue;
        // prefer the nearest anchor that still has room, so the repair link is not pruned away again
        const anchors = [...reached].map((other) => ({
          id: other, degree: (this.links[0].get(other) ?? new Set()).size, distance: this.#distanceToNeighbour(id, other),
        }));
        anchors.sort((left, right) => left.degree - right.degree || left.distance - right.distance || left.id - right.id);
        this.#connect(id, anchors[0].id, 0);
        this.repairs++;
      }
      reached = this.reachable();
    }
    return reached.size;
  }

  levelOf() {
    const mL = 1 / Math.log(this.m === 1 ? 2 : this.m);
    return Math.floor(-Math.log(Math.max(1e-12, this.rng.float())) * mL);
  }

  #layer(layer) {
    while (this.links.length <= layer) this.links.push(new Map());
    return this.links[layer];
  }

  #distanceToVector(query, id) {
    if (this.codes === null) return distance2(query, this.vectors[id]);
    const code = typeof query === "object" && query.code !== undefined ? query.code : this.quantiser.encode(query);
    return distance2(code, this.codes[id]);
  }
  #distanceToNeighbour(from, to) {
    if (this.codes === null) return distance2(this.vectors[from], this.vectors[to]);
    return distance2(this.codes[from], this.codes[to]);
  }
  #greedyDescend(query, start, layer) {
    let current = start;
    let improved = true;
    while (improved) {
      improved = false;
      for (const neighbour of this.links[layer].get(current) ?? []) {
        this.hops++;
        if (this.#distanceToVector(query, neighbour) < this.#distanceToVector(query, current)) { current = neighbour; improved = true; }
      }
    }
    return current;
  }

  /** Beam search inside one layer: `ef` closest candidates found so far. */
  #searchLayer(query, entries, ef, layer) {
    const visited = new Set(entries);
    const candidates = entries.map((id) => ({ id, distance: this.#distanceToVector(query, id) })).sort(byDistance);
    let results = candidates.slice();
    while (candidates.length > 0) {
      const current = candidates.shift();
      if (results.length >= ef && current.distance > results[results.length - 1].distance) break;
      for (const neighbour of this.links[layer].get(current.id) ?? []) {
        if (visited.has(neighbour)) continue;
        visited.add(neighbour);
        const distance = this.#distanceToVector(query, neighbour);
        if (results.length < ef || distance < results[results.length - 1].distance) {
          candidates.push({ id: neighbour, distance });
          candidates.sort(byDistance);
          results = [...results, { id: neighbour, distance }].sort(byDistance).slice(0, ef);
        }
      }
    }
    return results;
  }

  /** Trims a node's neighbour list to the degree bound, dropping the reciprocal links as well. */
  #prune(owner, layer, depth = 0) {
    const graph = this.#layer(layer);
    const list = graph.get(owner) ?? new Set();
    const limit = layer === 0 ? this.m0 : this.m;
    if (list.size <= limit) return;
    // ties are broken in favour of the most recently added link, so a repair link survives
    const ranked = [...list].map((id, order) => ({ id, order, distance: this.#distanceToNeighbour(owner, id) }))
      .sort((left, right) => left.distance - right.distance || right.order - left.order);
    const kept = ranked.slice(0, limit).map((entry) => entry.id);
    const dropped = ranked.slice(limit).map((entry) => entry.id);
    graph.set(owner, new Set(kept));
    if (depth >= 3) return;
    for (const id of dropped) {
      const reciprocal = graph.get(id);
      if (reciprocal === undefined) continue;
      reciprocal.delete(owner);
      this.#prune(id, layer, depth + 1);
    }
  }
  /** Links two nodes in a layer and prunes both lists back to the degree bound. */
  #connect(from, to, layer) {
    if (from === to) return;
    const graph = this.#layer(layer);
    for (const [owner, other] of [[from, to], [to, from]]) {
      const list = graph.get(owner) ?? new Set();
      list.add(other);
      graph.set(owner, list);
    }
    this.#prune(from, layer);
    this.#prune(to, layer);
  }

  insert(vector) {
    const id = this.levels.length;
    const level = this.levelOf();
    this.levels.push(level);
    for (let layer = 0; layer <= level; layer++) this.#layer(layer).set(id, new Set());
    if (this.entry === null) { this.entry = id; this.maxLevel = level; return id; }
    let current = this.entry;
    for (let layer = this.maxLevel; layer > level; layer--) current = this.#greedyDescend(vector.slice(), current, layer);
    for (let layer = Math.min(level, this.maxLevel); layer >= 0; layer--) {
      const candidates = this.#searchLayer(vector.slice(), [current], this.efConstruction, layer);
      for (const candidate of candidates) if (candidate.id !== id) this.#connect(id, candidate.id, layer);
      if (candidates.length > 0) current = candidates[0].id;
    }
    if (level > this.maxLevel) { this.maxLevel = level; this.entry = id; }
    return id;
  }
  /** Greedy search; `rescore` re-ranks the beam with the exact float vectors. */
  search(query, { k = 10, ef = 30, rescore = true } = {}) {
    if (this.entry === null) return [];
    if (query.length !== this.dimension) throw new RangeError("the query has the wrong dimension");
    if (!Number.isInteger(ef) || ef < 1) throw new RangeError("ef must be a positive integer");
    let current = this.entry;
    for (let layer = this.maxLevel; layer > 0; layer--) current = this.#greedyDescend(query, current, layer);
    const beam = this.#searchLayer(query, [current], Math.max(ef, k), 0);
    const ordered = rescore
      ? beam.map((entry) => ({ id: entry.id, distance: distance2(query, this.vectors[entry.id]) })).sort(byDistance)
      : beam;
    return ordered.slice(0, Math.max(0, k));
  }

  bruteForce(query, k) {
    return this.vectors.map((vector, id) => ({ id, distance: distance2(query, vector) }))
      .sort(byDistance).slice(0, Math.max(0, k));
  }
  /** Reachable nodes at layer 0 from the entry point. */
  reachable() {
    const seen = new Set();
    const pending = [this.entry];
    while (pending.length > 0) {
      const id = pending.pop();
      if (seen.has(id)) continue;
      seen.add(id);
      for (const neighbour of this.links[0].get(id) ?? []) pending.push(neighbour);
    }
    return seen;
  }

  checkInvariants() {
    for (const [id, level] of this.levels.entries()) {
      if (level < 0 || level > this.maxLevel) return { ok: false, reason: `node ${id} has an impossible level` };
      for (let layer = 0; layer <= level; layer++) {
        if (!this.links[layer].has(id)) return { ok: false, reason: `node ${id} is missing from layer ${layer}` };
      }
      for (let layer = level + 1; layer <= this.maxLevel; layer++) {
        if (this.links[layer].has(id)) return { ok: false, reason: `node ${id} appears above its level` };
      }
    }
    for (const [layer, graph] of this.links.entries()) {
      const limit = layer === 0 ? this.m0 : this.m;
      for (const [id, neighbours] of graph) {
        if (neighbours.size > limit) return { ok: false, reason: `node ${id} exceeds the degree bound in layer ${layer}` };
        for (const neighbour of neighbours) {
          if (neighbour === id) return { ok: false, reason: `node ${id} links to itself` };
          if (!(this.links[layer].get(neighbour) ?? new Set()).has(id)) return { ok: false, reason: `link ${id}-${neighbour} is not symmetric` };
        }
      }
    }
    if (this.links[this.maxLevel].size === 0 || !this.links[this.maxLevel].has(this.entry)) {
      return { ok: false, reason: "the entry point is not in the top layer" };
    }
    const seen = this.reachable();
    if (seen.size !== this.vectors.length) return { ok: false, reason: `layer 0 only reaches ${seen.size} nodes` };
    return { ok: true, nodes: seen.size };
  }

  adjacencyHash() {
    let hash = 0x811c9dc5;
    const text = this.links.map((graph) => [...graph.keys()].sort((a, b) => a - b)
      .map((id) => `${id}:${[...graph.get(id)].sort((a, b) => a - b).join(".")}`).join(",")).join("|");
    for (let i = 0; i < text.length; i++) hash = (Math.imul(hash, 0x01000193) ^ text.charCodeAt(i)) >>> 0;
    return hash;
  }
}
/** Deterministic clustered dataset. */
function makeDataset(count, dimension, clusters, seed) {
  const rng = makeRng(seed);
  const centres = Array.from({ length: clusters }, () => Array.from({ length: dimension }, () => (rng(400) - 200) / 10));
  return Array.from({ length: count }, (_, index) => {
    const centre = centres[index % clusters];
    return centre.map((value) => value + (rng(100) - 50) / 25);
  });
}
const recallAt = (approximate, exact) => {
  const truth = new Set(exact.map((entry) => entry.id));
  return approximate.filter((entry) => truth.has(entry.id)).length / Math.max(1, truth.size);
};

module.exports = { distance2, makeRng, ScalarQuantiser, HNSW, makeDataset, recallAt };

if (require.main === module) {
  const VECTORS = makeDataset(120, 8, 5, 0x766766);
  const QUERIES = makeDataset(20, 8, 5, 0xabcabc);
  const index = new HNSW(VECTORS, { m: 8, efConstruction: 30, seed: 11 });

  // 1. graph invariants
  const invariants = index.checkInvariants();
  ok(invariants.ok, invariants.reason);
  strictEqual(invariants.nodes, VECTORS.length, "layer 0 is connected and spans the whole dataset");
  ok(index.maxLevel >= 1, `the hierarchy has several layers (max ${index.maxLevel})`);  // 2. quantization round-trip: no coordinate moves by more than half a step
  const quantiser = index.quantiser;
  let quantisationChecks = 0;
  for (const vector of VECTORS.slice(0, 20)) {
    quantiser.decode(quantiser.encode(vector)).forEach((value, axis) => {
      ok(Math.abs(value - vector[axis]) <= quantiser.step[axis] / 2 + 1e-12, `axis ${axis} stays within half a step`);
      quantisationChecks++;
    });
  }
  ok(quantisationChecks === 160, `every coordinate of 20 vectors was checked (${quantisationChecks})`);
  // 3. recall@10 of the three paths against brute force
  const exactIndex = new HNSW(VECTORS, { m: 8, efConstruction: 30, seed: 11, quantise: false });
  const measure = (subject, options, rescore = true) => QUERIES.reduce((total, query) =>
    total + recallAt(subject.search(query, { k: 10, ...options, rescore }), subject.bruteForce(query, 10)), 0) / QUERIES.length;
  const exactPath = measure(exactIndex, { ef: 30 });
  const codesOnly = measure(index, { ef: 30 }, false);
  const rescored = measure(index, { ef: 30 });
  const narrow = measure(index, { ef: 10 }), wide = measure(index, { ef: 60 });
  ok(exactPath >= 0.9, `an exact-distance traversal clears the floor (${exactPath.toFixed(3)})`);
  ok(rescored >= codesOnly - 1e-12, `rescoring helps over codes alone (${rescored.toFixed(3)} vs ${codesOnly.toFixed(3)})`);
  ok(wide >= narrow - 1e-12, `a wider beam cannot be worse (${wide.toFixed(3)} vs ${narrow.toFixed(3)})`);
  ok(codesOnly >= 0.5, `even pure code distances keep most of the recall (${codesOnly.toFixed(3)})`);
  // 4. hops from the entry point stay bounded (the small-world property); 5. a self-query is exact
  const hopCounts = QUERIES.map((query) => { index.hops = 0; index.search(query, { k: 1, ef: 1 }); return index.hops; });
  ok(Math.max(...hopCounts) < VECTORS.length / 4, `greedy descent takes few steps (max ${Math.max(...hopCounts)})`);
  const selfTop1 = VECTORS.slice(0, 40).filter((vector, id) => index.search(vector, { k: 1, ef: 30 })[0].id === id).length;
  strictEqual(selfTop1, 40, "a query equal to an indexed vector returns that vector");
  ok(index.search(VECTORS[3], { k: 1, ef: 30 })[0].distance === 0, "with an exact zero distance");
  // 6. determinism: same seed, same graph, same results
  const twin = new HNSW(VECTORS, { m: 8, efConstruction: 30, seed: 11 });
  strictEqual(twin.adjacencyHash(), index.adjacencyHash(), "the graph is a deterministic function of the seed");
  deepStrictEqual(twin.search(QUERIES[0], { k: 5, ef: 30 }).map((entry) => entry.id),
    index.search(QUERIES[0], { k: 5, ef: 30 }).map((entry) => entry.id), "and so are the results");
  ok(twin.adjacencyHash() !== new HNSW(VECTORS, { m: 8, efConstruction: 30, seed: 12 }).adjacencyHash(), "a different seed builds a different graph");
  // 7. boundaries: a single node, duplicate vectors, oversized k, ef below k
  const single = new HNSW([[1, 2, 3]], { m: 4, seed: 1 });
  const singleHit = single.search([1, 2, 3], { k: 3, ef: 10 });
  ok(singleHit.length === 1 && singleHit[0].distance === 0, "a one-node index returns that node with an exact hit");
  ok(index.search(QUERIES[0], { k: 200, ef: 30 }).length <= VECTORS.length, "k larger than the dataset is capped");
  strictEqual(index.search(QUERIES[0], { k: 10, ef: 1 }).length, 10, "a narrow beam still returns k results");
  const same = new HNSW(Array.from({ length: 10 }, () => [1, 1, 1, 1]), { m: 4, seed: 4 });
  ok(same.checkInvariants().ok && same.search([1, 1, 1, 1], { k: 1, ef: 5 })[0].distance === 0, "identical vectors still give a valid graph with zero distances");
  // 8. validation
  throws(() => new HNSW([]), RangeError);
  throws(() => new HNSW([[1, 2]], { m: 0 }), RangeError);
  throws(() => new HNSW([[1, 2]], { efConstruction: 0 }), RangeError);
  throws(() => index.search([1, 2], { k: 3 }), RangeError);
  throws(() => index.search(QUERIES[0], { k: 3, ef: 0 }), RangeError);
  console.log(
    `nodes=${VECTORS.length} layers=${index.maxLevel + 1} repairs=${index.repairs} recall@10 exact=${exactPath.toFixed(3)} ` +
    `codesOnly=${codesOnly.toFixed(3)} rescored=${rescored.toFixed(3)} ef10=${narrow.toFixed(3)} ef60=${wide.toFixed(3)} ` +
    `maxHops=${Math.max(...hopCounts)} coordChecks=${quantisationChecks}`);
  console.log("PASS");
}
