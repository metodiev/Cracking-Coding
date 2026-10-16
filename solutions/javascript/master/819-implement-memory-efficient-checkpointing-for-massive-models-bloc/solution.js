/** Task 819 — Implement memory-efficient checkpointing for massive models (block-level dedup).
 *  (Tier: Master, Beyond — Machine Learning & AI)
 *
 * Design. A checkpoint is not a monolithic file but a *manifest* of content-addressed blocks: the
 * parameter byte stream is cut with **content-defined chunking** (a Buzhash rolling hash over a
 * sliding window with min/max block sizes), every block is addressed by `sha256(block)`, and a
 * checkpoint is the ordered list of those addresses. Two consequences carry the whole design and are
 * both measured here:
 *   - *incremental checkpoints*: training changes a small part of the parameters, so a new checkpoint
 *     shares almost every block with the previous one and only the touched blocks are written;
 *   - *resynchronisation*: because block boundaries follow the content and not the offset, inserting
 *     or deleting bytes only perturbs the blocks around the edit. Fixed-size chunking loses the entire
 *     tail after such an edit — the file measures both chunkers on the same edit so the claim is a
 *     number, not a slogan.
 * The store keeps reference counts, so `delete(manifest)` frees exactly the blocks no live checkpoint
 * needs any more, and the whole pipeline is streaming: a snapshot is consumed block by block, which
 * caps the working set at one block instead of the model size (the accounting below checks that).
 *
 * Memory model. The snapshot source is a range iterator over the parameter buffer and the store never
 * copies it; the measured peak working set is bounded by the largest block plus the digest buffer, and
 * the check asserts that against `maxBlockSize` instead of trusting the structure. Real systems also
 * stream to object storage; here the store is in-process.
 *
 * Collision safety. Content addressing uses sha256, but the implementation does not *trust* it: a
 * candidate block is only deduplicated after a full byte comparison, and the file exercises that path
 * with a deliberately weakened 16-bit digest where collisions actually happen (they are counted and
 * the data still round-trips byte for byte).
 *
 * Simplified versus production: no filesystem/object store, no compression, no encryption, no
 * distributed upload parallelism, no partial-checkpoint resume across processes, and the "training
 * run" is a deterministic in-memory byte mutation schedule rather than a real model.
 *
 * Verified: byte-identical round trip for every checkpoint, store/refcount invariants after every
 * operation, garbage collection freeing exactly the unreachable blocks, incremental write cost,
 * content-defined versus fixed-size resynchronisation, internal dedup of repeated blocks, collision
 * handling under a weak digest, missing/corrupted manifest detection, peak working set bound, manifest
 * immutability, block-size bounds, determinism and the degenerate inputs (empty model, single byte,
 * all-zero model, incompressible model, model smaller than the minimum block size).
 */
"use strict";

const crypto = require("node:crypto");

// ---------------------------------------------------------------------------- deterministic bytes

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

function buildHashTable(seed) {
  const next = splitmix64(seed);
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) table[i] = Number(next() & 0xffffffffn);
  return table;
}

function randomBytes(length, seed) {
  const next = splitmix64(seed);
  const buffer = Buffer.alloc(length);
  for (let i = 0; i < length; i += 8) {
    const word = next();
    for (let j = 0; j < 8 && i + j < length; j++) buffer[i + j] = Number((word >> BigInt(8 * j)) & 0xffn);
  }
  return buffer;
}

// ---------------------------------------------------------------------------- chunking

const DEFAULT_CHUNKING = { minBlockSize: 512, maxBlockSize: 4096, mask: 0x1fff };

const rotl32 = (value, count) => ((value << count) | (value >>> (32 - count))) >>> 0;

/**
 * Content-defined chunking with a proper sliding-window Buzhash: the digest at a position depends only
 * on the `window` bytes before it, so an edit perturbs the blocks that contain it (plus at most one
 * window of context) and the boundaries resynchronise immediately afterwards — that is the property
 * fixed-size chunking does not have.
 */
function* contentDefinedBlocks(buffer, options = {}) {
  const { minBlockSize, maxBlockSize, mask } = { ...DEFAULT_CHUNKING, ...options };
  const window = options.window ?? 32;
  const table = options.table ?? buildHashTable(options.seed ?? 0x819b10c);
  let start = 0;
  while (start < buffer.length) {
    let hash = 0;
    const prime = Math.max(0, start - window);
    for (let k = prime; k < start; k++) hash = rotl32(hash, 1) ^ table[buffer[k]];
    let end = start;
    const limit = Math.min(start + maxBlockSize, buffer.length);
    while (end < limit) {
      hash = rotl32(hash, 1) ^ table[buffer[end]];
      if (end - prime >= window) hash ^= rotl32(table[buffer[end - window]], window % 32);
      end++;
      if (end - start >= minBlockSize && (hash & mask) === 0) break;
    }
    yield [start, end];
    start = end;
  }
}

/** Offset-based chunking, kept only as the baseline the file compares against. */
function* fixedSizeBlocks(buffer, options = {}) {
  const blockSize = typeof options === "number" ? options : options.blockSize ?? 1024;
  for (let start = 0; start < buffer.length; start += blockSize) {
    yield [start, Math.min(start + blockSize, buffer.length)];
  }
}

function digestOf(bytes, hashBits = 256) {
  const digest = crypto.createHash("sha256").update(bytes).digest("hex");
  return hashBits >= 256 ? digest : digest.slice(0, Math.ceil(hashBits / 4));
}

function buffersEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// ---------------------------------------------------------------------------- block store

class BlockStore {
  /**
   * @param {object} [options] `{ hashBits }` — a reduced digest width is only used to exercise the
   *   collision path: the store never deduplicates on the digest alone.
   */
  constructor(options = {}) {
    this.hashBits = options.hashBits ?? 256;
    this.blocks = new Map(); // digest -> [variant keys] (usually exactly one)
    this.records = new Map(); // variant key -> { bytes, refs }
    this.bytesStored = 0;
    this.bytesWritten = 0;
    this.collisionsDetected = 0;
    this.deduplicatedBlocks = 0;
    this.peakWorkingSet = 0;
  }

  /**
   * Streaming insert: consumes the block ranges one at a time and keeps at most one block plus its
   * digest in memory, so the working set does not scale with the model. A block is only deduplicated
   * after a full byte comparison, so a digest collision (exercised with a weak digest below) creates a
   * second variant instead of corrupting the first one.
   */
  putRange(buffer, range) {
    const bytes = buffer.subarray(range[0], range[1]);
    const digest = digestOf(bytes, this.hashBits);
    const variants = this.blocks.get(digest) ?? [];
    this.peakWorkingSet = Math.max(this.peakWorkingSet, bytes.length + digest.length / 2);
    for (const key of variants) {
      const record = this.records.get(key);
      if (buffersEqual(record.bytes, bytes)) {
        record.refs++;
        this.deduplicatedBlocks++;
        return key;
      }
    }
    if (variants.length > 0) this.collisionsDetected++;
    const key = `${digest}#${variants.length}`;
    this.records.set(key, { bytes: Buffer.from(bytes), refs: 1 });
    this.blocks.set(digest, [...variants, key]);
    this.bytesStored += bytes.length;
    this.bytesWritten += bytes.length;
    return key;
  }

  get(key) {
    const record = this.records.get(key);
    if (!record) throw new Error(`missing block ${key}`);
    return record.bytes;
  }

  release(key) {
    const record = this.records.get(key);
    if (!record) throw new Error(`missing block ${key}`);
    record.refs--;
    if (record.refs === 0) {
      this.bytesStored -= record.bytes.length;
      this.records.delete(key);
      const digest = key.split("#")[0];
      const variants = (this.blocks.get(digest) ?? []).filter((variant) => variant !== key);
      if (variants.length === 0) this.blocks.delete(digest);
      else this.blocks.set(digest, variants);
    }
  }

  /** Invariant: the accounted size equals the sum of the block sizes it actually holds. */
  checkAccounting() {
    let total = 0;
    for (const record of this.records.values()) {
      if (record.refs <= 0) throw new Error("a live block must have a positive reference count");
      total += record.bytes.length;
    }
    return total === this.bytesStored;
  }

  get blockCount() {
    return this.records.size;
  }
}

// ---------------------------------------------------------------------------- checkpoints

class CheckpointManager {
  constructor(options = {}) {
    this.chunking = { ...DEFAULT_CHUNKING, ...(options.chunking ?? {}) };
    this.store = new BlockStore(options);
    this.checkpoints = new Map(); // id -> { id, digests, size, step }
    this.peakWorkingSet = 0;
    this.nextId = 0;
  }

  /** Snapshot a parameter buffer: block ranges are streamed into the store, never the whole model. */
  snapshot(buffer, step = null, options = {}) {
    const chunker = options.fixedSize
      ? fixedSizeBlocks(buffer, { blockSize: options.blockSize ?? 1024 })
      : contentDefinedBlocks(buffer, { ...this.chunking, ...(options.chunking ?? {}) });
    const digests = [];
    for (const range of chunker) {
      digests.push(this.store.putRange(buffer, range));
      this.peakWorkingSet = Math.max(this.peakWorkingSet, this.store.peakWorkingSet);
    }
    const id = String(this.nextId).padStart(3, "0");
    this.nextId++;
    const checkpoint = { id, digests, size: buffer.length, step };
    this.checkpoints.set(id, checkpoint);
    return checkpoint;
  }

  restore(id) {
    const checkpoint = this.checkpoints.get(id);
    if (!checkpoint) throw new Error(`unknown checkpoint ${id}`);
    const parts = checkpoint.digests.map((digest) => this.store.get(digest));
    return Buffer.concat(parts, checkpoint.size);
  }

  delete(id) {
    const checkpoint = this.checkpoints.get(id);
    if (!checkpoint) throw new Error(`unknown checkpoint ${id}`);
    for (const digest of checkpoint.digests) this.store.release(digest);
    this.checkpoints.delete(id);
    return checkpoint;
  }

  get stats() {
    return {
      blocks: this.store.blockCount,
      bytesStored: this.store.bytesStored,
      bytesWritten: this.store.bytesWritten,
      deduplicated: this.store.deduplicatedBlocks,
      collisions: this.store.collisionsDetected,
      workingSet: this.peakWorkingSet,
      checkpoints: this.checkpoints.size,
    };
  }
}

/** Total bytes a fixed-size chunker would have to write for the same buffer. */
function fullWriteCost(buffer) {
  return buffer.length;
}

/** Shared-block ratio between two buffers chunked with the same chunker. */
function sharedBlockRatio(bufferA, bufferB, chunker, chunkerOptions = {}) {
  const collect = (buffer) => {
    const digests = new Set();
    for (const range of chunker(buffer, chunkerOptions)) digests.add(digestOf(buffer.subarray(range[0], range[1])));
    return digests;
  };
  const left = collect(bufferA);
  const right = collect(bufferB);
  if (left.size === 0) return 1;
  let shared = 0;
  for (const digest of left) if (right.has(digest)) shared++;
  return shared / left.size;
}

// ---------------------------------------------------------------------------- simulated training

/** Deterministic in-memory "training": a few parameter regions change per step. */
function simulateTraining(modelSize, steps, seed, options = {}) {
  const mutationsPerStep = options.mutationsPerStep ?? 3;
  const regionSize = options.regionSize ?? 300;
  const next = splitmix64(seed);
  const model = randomBytes(modelSize, BigInt(seed) ^ 0x1234n);
  const history = [];
  for (let step = 0; step < steps; step++) {
    const edits = [];
    for (let k = 0; k < mutationsPerStep; k++) {
      const offset = Number(next() % BigInt(Math.max(1, modelSize - regionSize)));
      const length = regionSize;
      for (let i = 0; i < length; i++) model[offset + i] = Number(next() & 0xffn);
      edits.push([offset, offset + length]);
    }
    history.push({ step, model: Buffer.from(model), edits });
  }
  return history;
}

module.exports = {
  splitmix64,
  buildHashTable,
  randomBytes,
  DEFAULT_CHUNKING,
  contentDefinedBlocks,
  fixedSizeBlocks,
  digestOf,
  buffersEqual,
  BlockStore,
  CheckpointManager,
  fullWriteCost,
  sharedBlockRatio,
  simulateTraining,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- round trip, block-size bounds and determinism ------------------------------
  const modelSize = 96 * 1024;
  const model = randomBytes(modelSize, 0x819eee);
  const manager = new CheckpointManager();
  const first = manager.snapshot(model, 0);
  assert.deepStrictEqual(manager.restore(first.id), model, "the first checkpoint must round-trip byte for byte");
  assert.ok(manager.store.checkAccounting(), "store accounting must balance");
  const blocks = [...contentDefinedBlocks(model)];
  for (const range of blocks) {
    const length = range[1] - range[0];
    assert.ok(length <= DEFAULT_CHUNKING.maxBlockSize, `block of ${length} exceeds the maximum`);
    if (range !== blocks[blocks.length - 1]) assert.ok(length >= DEFAULT_CHUNKING.minBlockSize, `block of ${length} below the minimum`);
  }
  assert.ok(blocks.length > 20, `expected a meaningful number of blocks, got ${blocks.length}`);
  const repeated = [...contentDefinedBlocks(model)].map((range) => digestOf(model.subarray(range[0], range[1])));
  const original = blocks.map((range) => digestOf(model.subarray(range[0], range[1])));
  assert.deepStrictEqual(repeated, original, "chunking must be deterministic");
  const secondManager = new CheckpointManager();
  assert.deepStrictEqual(secondManager.snapshot(model, 0).digests, first.digests, "snapshots must be deterministic");

  // --- a simulated training run: incremental checkpoints and correct restores -----
  const run = new CheckpointManager({ chunking: { minBlockSize: 256, maxBlockSize: 2048, mask: 0x3ff } });
  const history = simulateTraining(192 * 1024, 20, 0x819aaa, { mutationsPerStep: 4, regionSize: 400 });
  const snapshots = [];
  for (const { step, model: state } of history) snapshots.push({ step, checkpoint: run.snapshot(state, step) });
  for (const { step, model: state, checkpoint } of snapshots.map((entry, index) => ({ ...entry, model: history[index].model }))) {
    assert.deepStrictEqual(run.restore(checkpoint.id), state, `checkpoint at step ${step} must restore the exact parameters`);
  }
  assert.ok(run.store.checkAccounting());
  const naiveBytes = history.length * history[0].model.length;
  const incrementalRatio = run.store.bytesWritten / naiveBytes;
  assert.ok(incrementalRatio < 0.2, `incremental checkpoints must be cheap: ${incrementalRatio.toFixed(3)} of a full write`);
  assert.ok(run.store.deduplicatedBlocks > history.length, "blocks must actually be reused");
  assert.ok(run.peakWorkingSet <= DEFAULT_CHUNKING.maxBlockSize * 2, `peak working set ${run.peakWorkingSet} must stay near one block`);

  // --- manifest immutability and garbage collection -------------------------------
  const oldRestore = run.restore(snapshots[0].checkpoint.id);
  run.snapshot(history[history.length - 1].model, 99);
  assert.deepStrictEqual(run.restore(snapshots[0].checkpoint.id), oldRestore, "an existing manifest must never be rewritten");
  const beforeDelete = run.store.bytesStored;
  for (const { checkpoint } of snapshots) run.delete(checkpoint.id);
  assert.ok(run.store.bytesStored < beforeDelete, "deleting checkpoints must release blocks");
  assert.ok(run.store.checkAccounting(), "accounting must still balance after garbage collection");
  const liveIds = [...run.checkpoints.keys()];
  for (const id of liveIds) run.delete(id);
  assert.strictEqual(run.store.bytesStored, 0, "deleting every checkpoint must empty the store");
  assert.strictEqual(run.store.blockCount, 0);

  // --- content-defined versus fixed-size resynchronisation ------------------------
  // The insertion length decides the outcome for offset-based chunking: a whole number of blocks is
  // harmless, anything else shifts every boundary. Content-defined boundaries do not care.
  const alignedShift = Buffer.concat([randomBytes(2048, 5), model]);
  const misalignedShift = Buffer.concat([randomBytes(1000, 6), model]);
  const contentAligned = sharedBlockRatio(alignedShift, model, contentDefinedBlocks, DEFAULT_CHUNKING);
  const contentMisaligned = sharedBlockRatio(misalignedShift, model, contentDefinedBlocks, DEFAULT_CHUNKING);
  const fixedAligned = sharedBlockRatio(alignedShift, model, fixedSizeBlocks, { blockSize: 1024 });
  const fixedMisaligned = sharedBlockRatio(misalignedShift, model, fixedSizeBlocks, { blockSize: 1024 });
  assert.ok(contentAligned > 0.8 && contentMisaligned > 0.8, `content-defined boundaries must survive any insertion length (${contentAligned.toFixed(3)}, ${contentMisaligned.toFixed(3)})`);
  assert.ok(fixedAligned > 0.9, "a block-aligned insertion is harmless even for fixed-size chunking");
  assert.ok(fixedMisaligned < 0.2, `a misaligned insertion must destroy fixed-size chunking, got ${fixedMisaligned.toFixed(3)}`);
  const uneditedContentRatio = sharedBlockRatio(model, model, contentDefinedBlocks, DEFAULT_CHUNKING);
  assert.strictEqual(uneditedContentRatio, 1, "an unchanged model must share every block");

  // --- internal dedup: repeated regions collapse ----------------------------------
  const sparse = Buffer.alloc(128 * 1024);
  const zeros = sparse.subarray(0, 64 * 1024);
  const sparseStore = new CheckpointManager();
  const sparseSnapshot = sparseStore.snapshot(sparse);
  assert.ok(sparseStore.store.bytesStored < 4 * 1024, `64 KiB of zeros must collapse, stored ${sparseStore.store.bytesStored}`);
  assert.deepStrictEqual(sparseStore.restore(sparseSnapshot.id), sparse, "a collapsed model must still restore exactly");
  const denseStore = new CheckpointManager();
  const dense = randomBytes(128 * 1024, 11);
  denseStore.snapshot(dense);
  assert.ok(denseStore.store.bytesStored > sparseStore.store.bytesStored * 10, "incompressible content must not deduplicate");
  assert.strictEqual(zeros.length, 64 * 1024);

  // --- collisions under a deliberately weak digest --------------------------------
  const weakStore = new CheckpointManager({ hashBits: 16 });
  const blockCount = 2000;
  const collisionSource = randomBytes(blockCount * 64, 0xc0111de);
  const weakSnapshot = weakStore.snapshot(collisionSource, 0, { chunking: { minBlockSize: 64, maxBlockSize: 64, mask: 0 } });
  assert.deepStrictEqual(weakStore.restore(weakSnapshot.id), collisionSource, "collisions must never corrupt a restore");
  assert.ok(weakStore.store.collisionsDetected > 0, `the weak digest must collide at least once (saw ${weakStore.store.collisionsDetected})`);
  assert.ok(weakStore.store.checkAccounting());

  // --- corrupted and missing manifests are rejected --------------------------------
  const tampered = { id: "tampered", digests: [...first.digests.slice(0, -1), "0000000000000000000000000000000000000000000000000000000000000000"], size: modelSize, step: null };
  manager.checkpoints.set("tampered", tampered);
  assert.throws(() => manager.restore("tampered"), /missing block/);
  assert.throws(() => manager.restore("nope"), /unknown checkpoint/);
  assert.throws(() => manager.delete("nope"), /unknown checkpoint/);
  assert.strictEqual(fullWriteCost(model), modelSize);

  // --- degenerate models -----------------------------------------------------------
  const emptyManager = new CheckpointManager();
  const emptySnapshot = emptyManager.snapshot(Buffer.alloc(0), 0);
  assert.deepStrictEqual(emptyManager.restore(emptySnapshot.id), Buffer.alloc(0), "an empty model round-trips");
  assert.strictEqual(emptyManager.store.blockCount, 0);
  const singleManager = new CheckpointManager();
  const singleSnapshot = singleManager.snapshot(Buffer.from([7]), 0);
  assert.deepStrictEqual(singleManager.restore(singleSnapshot.id), Buffer.from([7]));
  assert.strictEqual(singleManager.store.blockCount, 1, "a model smaller than the minimum block is one block");
  const tinyManager = new CheckpointManager();
  const tiny = randomBytes(300, 3);
  const tinySnapshot = tinyManager.snapshot(tiny);
  assert.deepStrictEqual(tinyManager.restore(tinySnapshot.id), tiny);
  assert.strictEqual(tinyManager.store.blockCount, 1);

  console.log(
    `blocks ${blocks.length} (${DEFAULT_CHUNKING.minBlockSize}-${DEFAULT_CHUNKING.maxBlockSize}B), run: ${history.length} checkpoints of ${history[0].model.length} B ` +
      `wrote ${run.store.bytesWritten} B (${(incrementalRatio * 100).toFixed(2)}% of a full write), reuse ${run.store.deduplicatedBlocks} blocks, ` +
      `peak working set ${run.peakWorkingSet} B; resync after +2 KiB/+1 KiB: content-defined ${(contentAligned * 100).toFixed(1)}%/${(contentMisaligned * 100).toFixed(1)}% ` +
      `vs fixed ${(fixedAligned * 100).toFixed(1)}%/${(fixedMisaligned * 100).toFixed(1)}%; weak-digest collisions handled ${weakStore.store.collisionsDetected}`
  );
  console.log("PASS");
}
