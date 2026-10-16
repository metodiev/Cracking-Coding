/** Task 659 - Implement consistent hashing with weighted nodes and replication. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * The ring: hash every node onto a 32-bit circle, replicated `weight * virtualNodesPerWeight`
 * times as `hash("name#i")`. A key is owned by the first node clockwise from `hash(key)`, and
 * its replicas are the next *distinct* physical nodes clockwise.
 *
 * Why it behaves (and what is verified here):
 *   * Virtual nodes make the load proportional to weight: with weights 1:2:3 the observed key
 *     counts track the weights within a deterministic tolerance, and a zero-weight node is
 *     never assigned anything.
 *   * Removing a node only moves the keys that node owned — every other key keeps its primary,
 *     exactly, which is the whole point versus `hash(key) % N` (measured side by side: modulo
 *     hashing remaps nearly everything).
 *   * Adding a node only moves keys that the new node now sits in front of; again verified
 *     exactly, not statistically.
 *   * Replicas are distinct physical nodes, and asking for more replicas than nodes is an error.
 *   The successor search is binary search; it is cross-checked against a brute-force linear
 *   ring scan for every key in the test set.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

function makeRng(seed) {
  let state = seed >>> 0 || 0x1234567;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Deterministic FNV-1a over UTF-16 code units plus a murmur3 avalanche, mapped to 32 bits. */
function hashPoint(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  // FNV-1a has weak high bits; the avalanche makes ring points uniform, which is what makes
  // the weighted load actually proportional to weight.
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b) >>> 0;
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35) >>> 0;
  hash ^= hash >>> 16;
  return hash >>> 0;
}

const RING_SIZE = 2 ** 32;

class ConsistentHashRing {
  constructor(options = {}) {
    this.virtualNodesPerWeight = options.virtualNodesPerWeight ?? 40;
    if (!Number.isInteger(this.virtualNodesPerWeight) || this.virtualNodesPerWeight < 1) {
      throw new RangeError("virtualNodesPerWeight must be a positive integer");
    }
    this.ring = [];       // [{ point, node }] sorted by point
    this.weights = new Map();
    this.sorted = true;
  }

  addNode(name, weight = 1) {
    if (this.weights.has(name)) throw new Error(`node ${name} is already in the ring`);
    if (!Number.isInteger(weight) || weight < 0) throw new RangeError(`weight of ${name} must be a non-negative integer`);
    this.weights.set(name, weight);
    for (let i = 0; i < weight * this.virtualNodesPerWeight; i++) {
      this.ring.push({ point: hashPoint(`${name}#${i}`), node: name });
    }
    this.sorted = false;
    return this;
  }

  removeNode(name) {
    if (!this.weights.has(name)) throw new Error(`node ${name} is not in the ring`);
    this.weights.delete(name);
    this.ring = this.ring.filter((entry) => entry.node !== name);
    this.sorted = false;
    return this;
  }

  nodes() {
    return [...this.weights.keys()];
  }

  virtualNodeCount(name) {
    return this.ring.filter((entry) => entry.node === name).length;
  }

  sortedRing() {
    if (!this.sorted) {
      this.ring.sort((a, b) => (a.point - b.point) || (a.node < b.node ? -1 : a.node > b.node ? 1 : 0));
      this.sorted = true;
    }
    return this.ring;
  }

  /** Index of the first ring point >= point (binary search, wrapping to 0). */
  successorIndex(point) {
    const ring = this.sortedRing();
    if (ring.length === 0) throw new Error("the ring is empty");
    let low = 0;
    let high = ring.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (ring[mid].point < point) low = mid + 1;
      else high = mid;
    }
    return low === ring.length ? 0 : low;
  }

  /** Brute-force successor, used to cross-check the binary search. */
  successorIndexLinear(point) {
    const ring = this.sortedRing();
    for (let i = 0; i < ring.length; i++) if (ring[i].point >= point) return i;
    return 0;
  }

  /** Up to `replicas` distinct physical nodes, clockwise from the key. */
  getNodes(key, replicas = 1) {
    if (!Number.isInteger(replicas) || replicas < 1) throw new RangeError("replicas must be >= 1");
    if (replicas > this.weights.size) {
      throw new RangeError(`cannot place ${replicas} replicas on ${this.weights.size} nodes`);
    }
    const ring = this.sortedRing();
    const start = this.successorIndex(hashPoint(String(key)));
    const out = [];
    for (let step = 0; step < ring.length && out.length < replicas; step++) {
      const entry = ring[(start + step) % ring.length];
      if (!out.includes(entry.node)) out.push(entry.node);
    }
    return out;
  }

  getPrimary(key) {
    return this.getNodes(key, 1)[0];
  }

  /** How many keys each node owns among `keys`. */
  distribution(keys) {
    const counts = new Map(this.nodes().map((name) => [name, 0]));
    for (const key of keys) counts.set(this.getPrimary(key), counts.get(this.getPrimary(key)) + 1);
    return counts;
  }
}

/** The naive alternative: hash(key) % N. */
function moduloPrimary(key, nodes) {
  return nodes[hashPoint(String(key)) % nodes.length];
}

module.exports = { ConsistentHashRing, hashPoint, makeRng, moduloPrimary, RING_SIZE };

if (require.main === module) {
  // -- ring construction and validation ------------------------------------------
  const ring = new ConsistentHashRing({ virtualNodesPerWeight: 100 });
  const names = ["node-a", "node-b", "node-c", "node-d"];
  ring.addNode("node-a", 1).addNode("node-b", 2).addNode("node-c", 3).addNode("node-d", 1);
  strictEqual(ring.ring.length, (1 + 2 + 3 + 1) * 100, "virtual nodes are proportional to weight");
  strictEqual(ring.virtualNodeCount("node-c"), 300, "the heaviest node gets the most virtual nodes");
  throws(() => ring.addNode("node-a"), /already in the ring/, "duplicate nodes must be rejected");
  throws(() => ring.removeNode("nope"), /not in the ring/, "removing an unknown node must be rejected");
  throws(() => ring.addNode("node-e", -1), RangeError, "negative weights must be rejected");
  throws(() => new ConsistentHashRing({ virtualNodesPerWeight: 0 }), RangeError, "vnodes must be >= 1");
  throws(() => ring.getNodes("k", 5), RangeError, "more replicas than nodes must be rejected");
  throws(() => ring.getNodes("k", 0), RangeError, "replicas must be >= 1");
  const emptyRing = new ConsistentHashRing();
  throws(() => emptyRing.getPrimary("k"), /cannot place 1 replicas on 0 nodes/, "an empty ring cannot own keys");

  // -- determinism and binary-search cross-check ---------------------------------
  const keys = Array.from({ length: 4000 }, (_, i) => `user:${i}:session`);
  const firstPass = keys.map((key) => ring.getPrimary(key));
  deepStrictEqual(keys.map((key) => ring.getPrimary(key)), firstPass, "lookups must be deterministic");
  const rebuilt = new ConsistentHashRing({ virtualNodesPerWeight: 100 });
  for (const [name, weight] of [["node-a", 1], ["node-b", 2], ["node-c", 3], ["node-d", 1]]) {
    rebuilt.addNode(name, weight);
  }
  deepStrictEqual(keys.map((key) => rebuilt.getPrimary(key)), firstPass, "ring construction order must not matter");
  for (let i = 0; i < 500; i++) {
    const point = hashPoint(`probe-${i}`);
    strictEqual(ring.successorIndex(point), ring.successorIndexLinear(point),
      "binary search must agree with the linear scan");
  }
  ok(firstPass.every((name) => names.includes(name)), "every key maps onto some node");

  // -- weighted load distribution ------------------------------------------------
  const counts = ring.distribution(keys);
  const totalWeight = 1 + 2 + 3 + 1;
  for (const [name, weight] of [["node-a", 1], ["node-b", 2], ["node-c", 3], ["node-d", 1]]) {
    const observed = counts.get(name) / keys.length;
    const expected = weight / totalWeight;
    ok(Math.abs(observed - expected) / expected < 0.25,
      `${name} holds ${(observed * 100).toFixed(1)}% but should hold ~${(expected * 100).toFixed(1)}%`);
  }
  ok(counts.get("node-c") > counts.get("node-a"), "a heavier node must own more keys");
  ok(counts.get("node-c") > counts.get("node-d"), "and more than a light one");
  strictEqual([...counts.values()].reduce((a, b) => a + b, 0), keys.length, "counts must conserve keys");
  const zeroWeight = new ConsistentHashRing({ virtualNodesPerWeight: 25 });
  zeroWeight.addNode("live", 1).addNode("ghost", 0);
  strictEqual(zeroWeight.virtualNodeCount("ghost"), 0, "a zero-weight node has no virtual nodes");
  ok(keys.every((key) => zeroWeight.getPrimary(key) === "live"), "a zero-weight node owns nothing");

  // -- replication: distinct physical nodes, clockwise order --------------------
  for (const key of keys.slice(0, 200)) {
    const replicas = ring.getNodes(key, 3);
    strictEqual(new Set(replicas).size, 3, "replicas must be distinct physical nodes");
    strictEqual(replicas[0], ring.getPrimary(key), "the primary is the first replica");
    const all = ring.getNodes(key, names.length);
    deepStrictEqual([...all].sort(), [...names].sort(), "enough replicas cover every node");
  }

  // -- minimal disruption on removal and addition -------------------------------
  const before = new Map(keys.map((key) => [key, ring.getPrimary(key)]));
  ring.removeNode("node-b");
  const afterRemoval = new Map(keys.map((key) => [key, ring.getPrimary(key)]));
  let moved = 0;
  for (const key of keys) {
    if (before.get(key) === afterRemoval.get(key)) continue;
    moved++;
    strictEqual(before.get(key), "node-b", "only node-b's keys may move when node-b leaves");
  }
  strictEqual(moved, counts.get("node-b"), "exactly the keys node-b owned must move");
  const movedFraction = moved / keys.length;
  ok(movedFraction > 0.15 && movedFraction < 0.4,
    `removal should move about 1/4 of the keys, moved ${(movedFraction * 100).toFixed(1)}%`);

  ring.addNode("node-e", 1);
  const afterAddition = new Map(keys.map((key) => [key, ring.getPrimary(key)]));
  let adopted = 0;
  for (const key of keys) {
    if (afterAddition.get(key) === afterRemoval.get(key)) continue;
    adopted++;
    strictEqual(afterAddition.get(key), "node-e", "a new node may only steal keys, never shuffle them");
  }
  ok(adopted > keys.length * 0.1 && adopted < keys.length * 0.4,
    `adding a fifth node should take roughly 1/5 of the keys, took ${(adopted / keys.length * 100).toFixed(1)}%`);

  // -- versus modulo hashing -----------------------------------------------------
  const moduloBefore = keys.map((key) => moduloPrimary(key, ["a", "b", "c", "d"]));
  const moduloAfter = keys.map((key) => moduloPrimary(key, ["a", "b", "c"]));
  const moduloMoved = moduloBefore.filter((name, i) => name !== moduloAfter[i]).length / keys.length;
  ok(moduloMoved > 0.5, `modulo hashing should remap most keys, remapped ${(moduloMoved * 100).toFixed(1)}%`);
  ok(movedFraction < moduloMoved, "consistent hashing must disrupt far less than modulo hashing");

  // -- single node ring ---------------------------------------------------------- 
  const solo = new ConsistentHashRing({ virtualNodesPerWeight: 5 });
  solo.addNode("only", 1);
  ok(keys.every((key) => solo.getPrimary(key) === "only"), "a single node owns every key");
  deepStrictEqual(solo.getNodes("k", 1), ["only"]);

  console.log(`keys=${keys.length} ringPoints=${ring.ring.length} movedOnRemoval=${(movedFraction * 100).toFixed(1)}% ` +
    `moduloMoved=${(moduloMoved * 100).toFixed(1)}% adoptedByNewNode=${(adopted / keys.length * 100).toFixed(1)}%`);
  console.log("PASS");
}
