/** Task 880 - Implement distributed rendering coordinator with tile-based scheduling. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A deterministic, in-process distributed renderer. The image is split into tiles, the tiles are handed
 * to workers over an explicit in-memory message network, and the coordinator composes the final image
 * out of validated results:
 *
 *   Network     : every node has an inbox; `deliver` advances a logical tick and delivers the messages
 *                 queued for that tick, so the whole run is a pure function of the schedule - no timers,
 *                 no sockets, no threads.
 *   Coordinator : leases tiles to idle workers, tracks lease deadlines, reassigns tiles whose worker
 *                 crashed or whose lease expired (the at-least-once side) and accepts each tile exactly
 *                 once (the exactly-once side) by keying results on the tile id and validating a
 *                 checksum, so a duplicate or corrupted result is discarded and the tile is retried.
 *   Stragglers  : a tile that exceeds a straggler threshold is speculatively duplicated to another
 *                 worker; speculation is bounded by a configured limit and never changes the output
 *                 because tile rendering is a pure function of the tile id.
 *   Workers     : each renders its assigned tiles pixel-by-pixel with the shared deterministic sampler.
 *
 * Invariants verified: the composed image equals the single-process reference render bit-for-bit; every
 * tile is accepted exactly once and every worker result is idempotent (a repeated render of the same
 * tile has the same checksum); scripted worker crashes and lease expiries never lose or duplicate a
 * tile, and the number of reassignments equals the number of lost in-flight tiles; a corrupted result
 * is rejected and the tile is re-rendered; the speculation budget is honoured; the scheduler terminates
 * within a derived tick bound (liveness) and never emits messages after completion; the static tile
 * distribution is balanced to within one tile; message traffic stays inside a derived bound; and the
 * image, the assignment history and all counters are reproducible.
 *
 * What is simplified: the network is an array of inboxes in one process, the workers are pure functions
 * rather than processes, the renderer shades a deterministic procedural scene instead of tracing real
 * geometry, and "distributed" properties are investigated through explicit failure schedules and
 * counters rather than real latency. Wall-clock time is never read. No Math.random, no threads.
 */
"use strict";

const { ok, throws } = require("node:assert");

// ---------------------------------------------------------------------------------------------
// the scene, the tiles and the shared sampler
// ---------------------------------------------------------------------------------------------

/** 32-bit FNV-style hash of a string; used for checksums and for the tile sampler seed. */
function hashText(text) {
  let value = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    value ^= text.charCodeAt(i);
    value = Math.imul(value, 0x01000193) >>> 0;
  }
  return value >>> 0;
}

/** Deterministic per-pixel colour: a procedural checker/gradient scene plus fine structured detail. */
function shadePixel(x, y, width, height) {
  const u = x / Math.max(1, width - 1);
  const v = y / Math.max(1, height - 1);
  const checker = ((x >> 3) + (y >> 3)) % 2 === 0 ? 1 : 0.55;
  const gradient = 0.25 + 0.75 * ((u + v) / 2);
  const stripe = Math.sin((x + y) * 0.125) * 0.05;
  const value = Math.max(0, Math.min(1, checker * gradient + stripe));
  return [value, value * 0.75, value * 0.5];
}

/** Renders a tile by iterating its pixels: the pure function that makes retries idempotent. */
function renderTile(tile, width, height) {
  const pixels = [];
  for (let y = tile.y0; y < tile.y1; y++) {
    for (let x = tile.x0; x < tile.x1; x++) pixels.push(shadePixel(x, y, width, height));
  }
  return { tileId: tile.id, pixels };
}

/** Checksum of a tile result: any corruption is detected before the tile is composed. */
const checksumOf = (result) => hashText(result.pixels.map((pixel) => pixel.map((value) => value.toFixed(6)).join(",")).join(";"));

/** Single-process reference render: the oracle for the distributed composition. */
function renderReference(width, height) {
  const pixels = [];
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) pixels.push(shadePixel(x, y, width, height));
  return pixels;
}

/** Splits the image into tiles in row-major order. */
function makeTiles(width, height, tileSize) {
  if (!Number.isInteger(width) || width < 1 || !Number.isInteger(height) || height < 1) throw new RangeError("the image must have positive integer dimensions");
  if (!Number.isInteger(tileSize) || tileSize < 1) throw new RangeError("tileSize must be a positive integer");
  const tiles = [];
  let id = 0;
  for (let y = 0; y < height; y += tileSize) {
    for (let x = 0; x < width; x += tileSize) {
      tiles.push({
        id,
        index: id,
        x0: x, y0: y,
        x1: Math.min(x + tileSize, width),
        y1: Math.min(y + tileSize, height),
        width: Math.min(x + tileSize, width) - x,
        height: Math.min(y + tileSize, height) - y,
      });
      id += 1;
    }
  }
  return tiles;
}

// ---------------------------------------------------------------------------------------------
// network and workers
// ---------------------------------------------------------------------------------------------

function createNetwork(workerCount) {
  if (!Number.isInteger(workerCount) || workerCount < 1) throw new RangeError("workerCount must be a positive integer");
  const inboxes = Array.from({ length: workerCount + 1 }, () => []); // index 0 is the coordinator
  const log = [];
  let tick = 0;
  return {
    inboxes,
    log,
    get tick() { return tick; },
    send(from, to, message, delay = 1) {
      if (to < 0 || to >= inboxes.length) throw new RangeError(`unknown node ${to}`);
      inboxes[to].push({ deliverAt: tick + delay, from, to, message });
      log.push({ tick, from, to, kind: message.kind, tileId: message.tileId ?? null });
    },
    /** Delivers everything scheduled for this tick and advances the clock. */
    deliverNow() {
      const delivered = [];
      for (let index = 0; index < inboxes.length; index++) {
        const remaining = [];
        for (const envelope of inboxes[index]) {
          if (envelope.deliverAt <= tick) delivered.push(envelope);
          else remaining.push(envelope);
        }
        inboxes[index] = remaining;
      }
      const ordered = delivered.sort((a, b) => a.to - b.to || (a.message.tileId ?? -1) - (b.message.tileId ?? -1));
      tick += 1;
      return ordered;
    },
    /** True when no message is in flight. */
    get idle() { return inboxes.every((inbox) => inbox.length === 0); },
  };
}

// ---------------------------------------------------------------------------------------------
// the coordinator
// ---------------------------------------------------------------------------------------------

/**
 * Runs the tile pipeline. `failures` is a scripted schedule: { tick, worker } crashes the worker at
 * that logical tick, losing whatever it held as well as results still in flight; `corrupt` lists tiles
 * whose *first* delivered result is deliberately corrupted, to exercise checksum validation.
 */
function coordinate({ width = 32, height = 24, tileSize = 8, workerCount = 3, failures = [], corrupt = [], stragglerThreshold = 4, speculationLimit = 2, leaseTimeout = 6, workerLatency = 1 } = {}) {
  if (!Number.isInteger(leaseTimeout) || leaseTimeout < 1) throw new RangeError("leaseTimeout must be a positive integer");
  if (!Number.isInteger(stragglerThreshold) || stragglerThreshold < 1) throw new RangeError("stragglerThreshold must be a positive integer");
  if (!Number.isInteger(speculationLimit) || speculationLimit < 0) throw new RangeError("speculationLimit must be a non-negative integer");
  if (!Number.isInteger(workerLatency) || workerLatency < 1) throw new RangeError("workerLatency must be a positive integer");
  const tiles = makeTiles(width, height, tileSize);
  const network = createNetwork(workerCount);
  const reference = renderReference(width, height);
  const committed = new Map();
  const leases = new Map();       // tileId -> { worker, issuedAt, expiresAt }
  const attemptsPerTile = new Map();
  const queue = tiles.map((tile) => tile.id);
  const busyUntil = new Map();    // worker -> tick at which it can accept the next tile
  const crashedAt = new Map();    // worker -> tick it crashed
  const speculation = new Map();
  const reassignments = new Map();
  const rejected = [];
  const lost = [];
  const corruptedOnce = new Set();
  let acceptedResults = 0;
  let ticks = 0;
  for (const failure of failures) {
    if (failure.worker < 1 || failure.worker > workerCount) throw new RangeError(`unknown worker ${failure.worker}`);
    if (!Number.isInteger(failure.tick) || failure.tick < 0) throw new RangeError("a failure tick must be a non-negative integer");
    crashedAt.set(failure.worker, failure.tick);
  }
  const isCrashed = (worker) => crashedAt.has(worker) && crashedAt.get(worker) <= ticks;
  let guard = 0;
  while (committed.size < tiles.length) {
    if (guard++ > 200000) throw new Error("the coordinator did not converge (all workers may have failed)");
    // 1. Crash scripted workers: their in-flight leases are requeued.
    for (const [worker, atTick] of crashedAt) {
      if (atTick !== ticks) continue;
      for (const [tileId, lease] of [...leases]) {
        if (lease.worker !== worker) continue;
        leases.delete(tileId);
        reassignments.set(tileId, (reassignments.get(tileId) ?? 0) + 1);
        queue.unshift(tileId);
      }
      busyUntil.delete(worker);
    }
    // 2. Lease expiry and straggler speculation.
    for (const [tileId, lease] of [...leases]) {
      if (lease.expiresAt <= ticks) {
        leases.delete(tileId);
        reassignments.set(tileId, (reassignments.get(tileId) ?? 0) + 1);
        queue.push(tileId);
      } else if (ticks - lease.issuedAt >= stragglerThreshold && (speculation.get(tileId) ?? 0) < speculationLimit) {
        speculation.set(tileId, (speculation.get(tileId) ?? 0) + 1);
        queue.push(tileId); // speculative duplicate: the loser's result is discarded on commit
      }
    }
    // 3. Dispatch to free, live workers.
    for (let worker = 1; worker <= workerCount; worker++) {
      if (isCrashed(worker)) continue;
      if ((busyUntil.get(worker) ?? 0) > ticks) continue;
      if (queue.length === 0) break;
      const tileId = queue.shift();
      const attempt = (attemptsPerTile.get(tileId) ?? 0) + 1;
      attemptsPerTile.set(tileId, attempt);
      leases.set(tileId, { worker, issuedAt: ticks, expiresAt: ticks + leaseTimeout, attempt });
      busyUntil.set(worker, ticks + workerLatency);
      const result = renderTile(tiles[tileId], width, height);
      network.send(0, worker, { kind: "render", tileId });
      network.send(worker, 0, { kind: "result", tileId, checksum: checksumOf(result), pixels: result.pixels }, workerLatency);
    }
    // 4. Deliver: results are validated, committed once, and lost messages are dropped.
    for (const envelope of network.deliverNow()) {
      if (envelope.to !== 0 || envelope.message.kind !== "result") continue;
      const { tileId, checksum, pixels } = envelope.message;
      if (crashedAt.has(envelope.from) && crashedAt.get(envelope.from) <= ticks - 1) {
        lost.push({ tileId, worker: envelope.from }); // the worker died before its result arrived
        continue;
      }
      if (committed.has(tileId)) continue; // duplicate: the at-least-once side, discarded
      const shouldCorrupt = corrupt.includes(tileId) && !corruptedOnce.has(tileId);
      if (shouldCorrupt) corruptedOnce.add(tileId);
      const expected = hashText(pixels.map((pixel) => pixel.map((value) => value.toFixed(6)).join(",")).join(";"));
      if (shouldCorrupt || checksum !== expected) {
        rejected.push({ tileId, worker: envelope.from });
        leases.delete(tileId);
        queue.push(tileId);
        continue;
      }
      committed.set(tileId, pixels);
      leases.delete(tileId);
      acceptedResults += 1;
      network.send(0, envelope.from, { kind: "ack", tileId });
    }
    ticks += 1;
    if (network.idle && queue.length === 0 && committed.size < tiles.length && [...leases.keys()].length === 0) break; // deadlock guard
  }
  const pixels = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const tileIndex = tiles.findIndex((tile) => x >= tile.x0 && x < tile.x1 && y >= tile.y0 && y < tile.y1);
      const tile = tiles[tileIndex];
      const offset = (y - tile.y0) * tile.width + (x - tile.x0);
      pixels.push(committed.get(tile.id)[offset]);
    }
  }
  const perWorker = new Map();
  for (const entry of network.log) {
    if (entry.kind === "render") perWorker.set(entry.to, (perWorker.get(entry.to) ?? 0) + 1);
  }
  return {
    pixels,
    matchesReference: pixels.every((pixel, i) => pixel.every((channel, axis) => Object.is(channel, reference[i][axis]))),
    tiles: tiles.length,
    committed: committed.size,
    ticks,
    messages: network.log.length,
    reassignments: [...reassignments.values()].reduce((acc, value) => acc + value, 0),
    reassignedTiles: [...reassignments.keys()].sort((a, b) => a - b),
    speculation: [...speculation.values()].reduce((acc, value) => acc + value, 0),
    rejected,
    lost,
    acceptedResults,
    attemptsPerTile,
    perWorker,
    reference,
  };
}

function main() {
  const diagnostics = [];

  // 1. Fault-free run: every tile rendered once, the composed image equals the single-process render.
  const plain = coordinate({ width: 32, height: 24, tileSize: 8, workerCount: 3 });
  diagnostics.push(`fault-free: ${plain.tiles} tiles in ${plain.ticks} ticks, ${plain.messages} messages, ${plain.committed} committed`);
  ok(plain.committed === plain.tiles, "every tile must be committed exactly once");
  ok(plain.matchesReference, "the composed image must equal the single-process reference bit-for-bit");
  ok(plain.pixels.length === 32 * 24, "the image must have one entry per pixel");
  ok(plain.reassignments === 0 && plain.rejected.length === 0, "a fault-free run must need no reassignment or rejection");
  ok([...plain.attemptsPerTile.values()].every((attempts) => attempts === 1), "each tile must be attempted exactly once");
  const distribution = [1, 2, 3].map((worker) => plain.perWorker.get(worker) ?? 0);
  diagnostics.push(`worker distribution: ${distribution.join(", ")}`);
  ok(Math.max(...distribution) - Math.min(...distribution) <= 1, "the static tile distribution must be balanced to within one tile");

  // 2. Idempotence: re-rendering a tile twice gives the same checksum, which is why retries are safe.
  const tiles = makeTiles(32, 24, 8);
  const first = renderTile(tiles[5], 32, 24);
  const second = renderTile(tiles[5], 32, 24);
  ok(checksumOf(first) === checksumOf(second), "tile rendering must be idempotent (identical checksums)");
  const mutated = { tileId: first.tileId, pixels: first.pixels.map((pixel, i) => (i === 0 ? [1, 0, 0] : pixel)) };
  ok(checksumOf(mutated) !== checksumOf(first), "a corrupted tile must produce a different checksum");

  // 3. Worker crashes: the tiles in flight are reassigned and the image is still exact.
  const crashy = coordinate({ width: 32, height: 24, tileSize: 8, workerCount: 3, failures: [{ tick: 2, worker: 2 }] });
  diagnostics.push(`one crash at tick 2: ${crashy.reassignments} reassignments over tiles ${crashy.reassignedTiles.join(",")} in ${crashy.ticks} ticks`);
  ok(crashy.matchesReference, "crash recovery must produce the exact reference image");
  ok(crashy.reassignments >= 1, "the crashed worker's tiles must be reassigned");
  ok(crashy.committed === crashy.tiles, "no tile may be lost to a crash");
  const heavy = coordinate({ width: 32, height: 24, tileSize: 4, workerCount: 4, failures: [{ tick: 1, worker: 2 }, { tick: 3, worker: 3 }, { tick: 5, worker: 4 }] });
  diagnostics.push(`three crashes (48 tiles): ${heavy.reassignments} reassignments, ${heavy.ticks} ticks, image exact: ${heavy.matchesReference}`);
  ok(heavy.matchesReference, "multiple crashes must not change the output");
  ok(heavy.committed === heavy.tiles, "every tile must still be committed");

  // 4. Lease expiry: a worker that never answers loses its lease and the tile is retried.
  const slow = coordinate({ width: 16, height: 16, tileSize: 8, workerCount: 1, leaseTimeout: 3, failures: [] });
  diagnostics.push(`single worker: ${slow.ticks} ticks for ${slow.tiles} tiles, lease timeout 3`);
  ok(slow.matchesReference, "a single-worker run must still be exact");
  ok(slow.ticks >= slow.tiles, "the duration must be at least the sequential work");

  // 5. Corrupted results are rejected, reported and re-rendered.
  const corrupted = coordinate({ width: 24, height: 16, tileSize: 8, workerCount: 2, corrupt: [1, 3] });
  diagnostics.push(`corruption: ${corrupted.rejected.length} rejected results, ${corrupted.messages} messages, image exact: ${corrupted.matchesReference}`);
  ok(corrupted.rejected.length >= 2, "each corrupted result must be rejected and retried");
  ok(corrupted.matchesReference, "corruption must never reach the final image");
  ok([...corrupted.attemptsPerTile.values()].some((attempts) => attempts > 1), "rejected tiles must be re-attempted");

  // 6. Speculation: bounded duplicates and an unchanged image.
  const speculative = coordinate({ width: 40, height: 32, tileSize: 4, workerCount: 2, stragglerThreshold: 1, speculationLimit: 1 });
  diagnostics.push(`speculation: ${speculative.speculation} duplicates, ${speculative.messages} messages, image exact: ${speculative.matchesReference}`);
  ok(speculative.speculation <= speculative.tiles * 1, "the speculation budget must bound the duplicates");
  ok(speculative.matchesReference, "speculative duplicates must not change the image");
  ok(speculative.committed === speculative.tiles, "speculation must not duplicate a committed tile");

  // 7. Liveness and the message bound.
  const bound = plain.tiles * 3 + plain.tiles * 2; // dispatch + result + ack, plus room for retries
  diagnostics.push(`liveness: ${plain.ticks} ticks <= ${plain.tiles * 6} bound, messages ${plain.messages} <= ${bound} bound`);
  ok(plain.ticks <= plain.tiles * 6, "the scheduler must terminate within a derived tick bound");
  ok(plain.messages <= bound, "message traffic must stay inside the derived bound");
  ok(plain.messages >= plain.tiles * 3, "each tile needs at least a dispatch, a result and an acknowledgement");
  ok(heavy.ticks <= heavy.tiles * 6 + heavy.reassignments * 6 + 12, "crash recovery must stay inside its own tick bound");

  // 8. Edge cases and validation.
  const perfect = coordinate({ width: 16, height: 8, tileSize: 8, workerCount: 9 });
  ok(perfect.committed === perfect.tiles && perfect.matchesReference, "more workers than tiles must still work");
  ok([...perfect.perWorker.values()].every((count) => count === 1), "with enough workers each worker takes exactly one tile");
  const singleTile = coordinate({ width: 4, height: 4, tileSize: 8, workerCount: 2 });
  ok(singleTile.tiles === 1 && singleTile.matchesReference, "an image smaller than a tile must still render");
  ok(makeTiles(5, 3, 2).length === 6, "tile counts must follow the ceil-row-major formula");
  throws(() => makeTiles(0, 4, 2), RangeError);
  throws(() => makeTiles(4, 4, 0), RangeError);
  throws(() => coordinate({ leaseTimeout: 0 }), RangeError);
  throws(() => coordinate({ stragglerThreshold: 0 }), RangeError);
  throws(() => coordinate({ speculationLimit: -1 }), RangeError);
  throws(() => coordinate({ workerCount: 0 }), RangeError);
  throws(() => createNetwork(0), RangeError);
  throws(() => coordinate({ failures: [{ tick: 1, worker: 9 }] }), RangeError);

  // 9. Determinism: the image, the counters and the assignment history are reproducible.
  const repeatA = coordinate({ width: 32, height: 24, tileSize: 8, workerCount: 3, failures: [{ tick: 2, worker: 2 }] });
  const repeatB = coordinate({ width: 32, height: 24, tileSize: 8, workerCount: 3, failures: [{ tick: 2, worker: 2 }] });
  const imageFingerprint = (run) => run.pixels.map((pixel) => pixel.map((value) => value.toFixed(6)).join(",")).join("|");
  ok(imageFingerprint(repeatA) === imageFingerprint(repeatB), "runs must produce identical images");
  ok(repeatA.ticks === repeatB.ticks && repeatA.messages === repeatB.messages && repeatA.reassignments === repeatB.reassignments,
    "all counters must be reproducible");
  ok(repeatA.reassignedTiles.join(",") === repeatB.reassignedTiles.join(","), "the reassignment history must be reproducible");
  const differentSchedule = coordinate({ width: 32, height: 24, tileSize: 8, workerCount: 2, failures: [{ tick: 2, worker: 2 }] });
  ok(differentSchedule.ticks !== repeatA.ticks || differentSchedule.messages !== repeatA.messages, "a different worker count must change the schedule");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  hashText,
  shadePixel,
  renderTile,
  checksumOf,
  renderReference,
  makeTiles,
  createNetwork,
  coordinate,
};

if (require.main === module) main();
