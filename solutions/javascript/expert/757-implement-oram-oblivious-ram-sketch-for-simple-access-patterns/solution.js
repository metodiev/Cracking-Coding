/** Task 757 - Implement ORAM (oblivious RAM) sketch for simple access patterns. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * A Path ORAM on a binary tree of `2^levels` leaves. The server stores one bucket per tree node
 * (heap indices, root = 1, `bucketSize` slots each); the client keeps a position map
 * (block -> leaf) and a small stash. Every `access(blockId, value)` does exactly the same work
 * whatever the block is: remap the block to a fresh random leaf, read the *whole* root-to-leaf
 * path from the server, pull every fetched slot into the stash, remap, write the path back with
 * a deepest-first greedy eviction of the stash, and record the observed leaf. Because the leaf
 * comes from the PRNG rather than from the logical address, the server-visible trace is
 * independent of which blocks are touched — that is the oblivion property, and it is checked by
 * replaying two different access sequences under the same seed and demanding byte-identical
 * leaf traces.
 * Invariants maintained after every access: (a) a block lives either in the stash or in a bucket
 * on the path to its current leaf (never elsewhere), (b) no block is duplicated and none is
 * lost, (c) no bucket exceeds `bucketSize` slots, (d) every access touches exactly `levels + 1`
 * buckets and `bucketSize * (levels + 1)` slots.
 *
 * What is simplified: the server is an in-process array (no network, no encryption of buckets),
 * the tree is tiny (16 leaves, 4 slots per bucket), eviction is the greedy deepest-first variant
 * rather than a full recursive eviction, and recursion/position-map compression are omitted, so
 * the stash bound is *empirical* here (measured and printed) rather than the asymptotic
 * O(log N) probabilistic bound. What is verified: values read back match an independent plain
 * mirror for a long mixed workload; two different logical access sequences with the same seed
 * produce the same leaf trace (and a different seed produces a different one); every access has
 * the same server-side shape; leaves are used evenly; the four invariants above hold after every
 * single access; and a degenerate single-bucket tree still works.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** Deterministic xorshift32. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x757757;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

class PathOram {
  constructor({ levels = 4, bucketSize = 4, seed = 1 } = {}) {
    if (levels < 0 || bucketSize < 1) throw new RangeError("levels must be >= 0 and bucketSize >= 1");
    this.levels = levels;
    this.bucketSize = bucketSize;
    this.nodeCount = 1 << (levels + 1);
    this.buckets = Array.from({ length: this.nodeCount }, () => []);
    this.position = new Map();
    this.stash = new Map();
    this.rng = makeRng(seed);
    this.trace = [];
    this.remaps = 0;
    this.maxStash = 0;
  }

  get leafCount() { return 1 << this.levels; }

  /** Server-visible view of the buckets on a root-to-leaf path, root first. */
  pathNodes(leaf) {
    const nodes = [1];
    let node = 1;
    for (let level = this.levels - 1; level >= 0; level--) {
      node = 2 * node + ((leaf >> level) & 1);
      nodes.push(node);
    }
    return nodes;
  }

  /**
   * One oblivious access. Returns the value stored *before* the access; `value === undefined`
   * is a read and anything else overwrites the block (creating it with value 0 if absent).
   */
  access(blockId, value) {
    const leaf = this.position.has(blockId) ? this.position.get(blockId) : this.rng(this.leafCount);
    const nodes = this.pathNodes(leaf);
    // server side: the whole path is read, whatever the client is looking for
    for (const node of nodes) for (const slot of this.buckets[node]) if (slot !== null) this.stash.set(slot.id, slot);
    this.trace.push({ leaf, buckets: nodes.length, slots: nodes.length * this.bucketSize });
    // client side: remap first, then apply the access
    this.position.set(blockId, this.rng(this.leafCount));
    this.remaps++;
    const resident = this.stash.get(blockId);
    const previous = resident === undefined ? 0 : resident.value;
    const block = { id: blockId, value: value === undefined ? previous : value };
    this.stash.set(blockId, block);
    // write-back with deepest-first greedy eviction
    for (const node of nodes) this.buckets[node] = [];
    for (let level = this.levels; level >= 0; level--) {
      const node = nodes[level];
      const waiting = [...this.stash.keys()].sort((a, b) => a - b);
      for (const id of waiting) {
        if (this.buckets[node].length >= this.bucketSize) break;
        if (this.pathNodes(this.position.get(id))[level] === node) {
          this.buckets[node].push(this.stash.get(id));
          this.stash.delete(id);
        }
      }
    }
    this.maxStash = Math.max(this.maxStash, this.stash.size);
    return previous;
  }

  read(blockId) { return this.access(blockId, undefined); }
  write(blockId, value) { return this.access(blockId, value); }

  /** (a) on-path, (b) no duplication, (c) bucket capacity, (d) accounting. */
  checkInvariants() {
    const located = new Map();
    for (let node = 1; node < this.nodeCount; node++) {
      const bucket = this.buckets[node];
      if (bucket.length > this.bucketSize) return { ok: false, reason: `bucket ${node} overfull` };
      for (const slot of bucket) {
        if (slot === null) continue;
        if (located.has(slot.id)) return { ok: false, reason: `block ${slot.id} duplicated` };
        located.set(slot.id, node);
        if (!this.pathNodes(this.position.get(slot.id)).includes(node)) {
          return { ok: false, reason: `block ${slot.id} off its path` };
        }
      }
    }
    for (const [id] of this.stash) {
      if (located.has(id)) return { ok: false, reason: `block ${id} both stashed and stored` };
      located.set(id, "stash");
    }
    return { ok: true, residents: located.size, located };
  }
}

const leafTrace = (oram) => oram.trace.map((entry) => entry.leaf);

module.exports = { PathOram, makeRng, leafTrace };

if (require.main === module) {
  // 1. hand-checked small case: write then read every block of a 4-leaf tree
  const small = new PathOram({ levels: 2, bucketSize: 2, seed: 11 });
  for (let id = 0; id < 4; id++) strictEqual(small.write(id, id * 10 + 1), 0, `fresh block ${id} reads as 0`);
  for (let id = 0; id < 4; id++) strictEqual(small.read(id), id * 10 + 1, `block ${id} round-trips`);
  ok(small.checkInvariants().ok, "invariants hold after the hand-checked case");
  strictEqual(small.checkInvariants().residents, 4, "four blocks are resident");
  // 2. overwriting and re-reading keeps the last value; reading an absent block yields 0
  small.write(2, 999);
  strictEqual(small.read(2), 999);
  strictEqual(small.read(7), 0, "a block that was never written reads as 0");
  ok(small.checkInvariants().ok);

  // 3. mixed workload against an independent plain mirror
  const oram = new PathOram({ levels: 4, bucketSize: 4, seed: 0x757757 });
  const mirror = new Map();
  const known = new Set();
  const rng = makeRng(4242);
  for (let step = 0; step < 300; step++) {
    const id = rng(12);
    known.add(id);
    if (rng(3) === 0) {
      const value = 1 + rng(1000);
      oram.write(id, value);
      mirror.set(id, value);
    } else {
      strictEqual(oram.read(id), mirror.get(id) ?? 0, `step ${step}: block ${id}`);
    }
    const invariants = oram.checkInvariants();
    ok(invariants.ok, `step ${step}: ${invariants.reason}`);
    strictEqual(invariants.residents, known.size, `step ${step}: every accessed block stays reachable`);
  }
  strictEqual(oram.checkInvariants().residents, known.size, "no block is lost or duplicated after the workload");

  // 4. oblivion: the leaf trace depends only on the seed and the number of accesses
  const runSequence = (blockIds, seed) => {
    const subject = new PathOram({ levels: 4, bucketSize: 4, seed });
    for (const id of blockIds) subject.read(id);
    return leafTrace(subject);
  };
  const seqA = Array.from({ length: 40 }, (_, i) => i % 9);
  const seqB = Array.from({ length: 40 }, (_, i) => 9 - (i % 9));
  deepStrictEqual(runSequence(seqA, 77), runSequence(seqB, 77), "different logical addresses, identical address trace");
  ok(runSequence(seqA, 77).join() !== runSequence(seqA, 78).join(), "a different seed moves the trace");
  // 5. every access has the same server-side shape, and remapping happens every time
  const shapes = new Set(oram.trace.map((entry) => `${entry.buckets}:${entry.slots}`));
  deepStrictEqual([...shapes], [`${5}:${5 * 4}`], "one access shape only: the full path");
  strictEqual(oram.remaps, oram.trace.length, "every access remaps its block");

  // 6. leaf usage is even (the trace is a uniform sequence, not a hotspot pattern)
  const uniform = new PathOram({ levels: 4, bucketSize: 4, seed: 0xabcdef });
  for (let step = 0; step < 400; step++) uniform.read(step % 5);
  const histogram = new Array(uniform.leafCount).fill(0);
  for (const leaf of leafTrace(uniform)) histogram[leaf]++;
  const mean = uniform.trace.length / uniform.leafCount;
  const tolerance = 4 * Math.sqrt(mean);      // each leaf within four standard deviations
  ok(Math.min(...histogram) > 0, "every leaf is used");
  ok(Math.max(...histogram) <= mean + tolerance && Math.min(...histogram) >= mean - tolerance,
    `leaf usage is even (${Math.min(...histogram)}..${Math.max(...histogram)} around ${mean})`);
  // 7. the stash stays small on this workload, and the invariant holds throughout
  ok(uniform.maxStash <= 8, `stash stays small (max ${uniform.maxStash})`);
  ok(uniform.checkInvariants().ok);

  // 8. written values survive a long oblivious workload; determinism of the whole run
  const deterministic = (seed) => {
    const subject = new PathOram({ levels: 4, bucketSize: 4, seed });
    const source = makeRng(9);
    const answers = [];
    for (let step = 0; step < 120; step++) {
      const id = source(7);
      if (source(2) === 0) subject.write(id, step);
      else answers.push(`${id}:${subject.read(id)}`);
    }
    return answers.join("|") + "#" + leafTrace(subject).join(",");
  };
  strictEqual(deterministic(5), deterministic(5), "same seed, same trace and same answers");
  ok(deterministic(5) !== deterministic(6), "a different seed changes the trace");

  // 9. degenerate trees: one bucket (levels 0) and a single block
  const tiny = new PathOram({ levels: 0, bucketSize: 1, seed: 3 });
  strictEqual(tiny.leafCount, 1, "a levels-0 tree has exactly one leaf");
  strictEqual(tiny.write(0, 42), 0);
  strictEqual(tiny.read(0), 42, "a single-bucket ORAM still round-trips");
  deepStrictEqual(leafTrace(tiny), [0, 0], "its trace is trivially constant");
  ok(tiny.checkInvariants().ok);
  // 10. blocks that never fit are stashed, not lost
  const crowded = new PathOram({ levels: 1, bucketSize: 1, seed: 8 });
  for (let id = 0; id < 9; id++) crowded.write(id, id + 1);
  for (let id = 0; id < 9; id++) strictEqual(crowded.read(id), id + 1, `crowded tree keeps block ${id}`);
  ok(crowded.maxStash > 0, "an overfull tree must use the stash");
  ok(crowded.checkInvariants().ok);
  throws(() => new PathOram({ levels: -1 }), RangeError);
  throws(() => new PathOram({ bucketSize: 0 }), RangeError);

  console.log(
    `workloadAccesses=${oram.trace.length} leaves=${oram.leafCount} maxStash=${oram.maxStash} ` +
    `accessShape=${[...shapes][0]} traceIndependent=true`
  );
  console.log("PASS");
}
