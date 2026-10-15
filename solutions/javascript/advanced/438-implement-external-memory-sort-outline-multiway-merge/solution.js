/** Task 438 - Implement external memory sort outline (multiway merge). (Tier: Advanced, CTCI Ch 10)
 *
 * Simulated external sort with documented block parameters: BLOCK_SIZE = 4 integers per disk block,
 * MEMORY_BLOCKS = 4 blocks of RAM (16 integers per initial run) and FAN_IN = 4 (each merge pass merges
 * groups of up to 4 sorted runs). Phase 1 loads up to one memory worth of integers, sorts them in RAM
 * and writes each initial run back to the simulated disk. Phase 2 keeps one block of every run in
 * memory, picks the next output element with a linear min-scan over the at most FAN_IN = 4 buffers (a
 * heap would take over for larger fan-in) and streams a single output block; every block read and
 * every block written is counted by SimDisk. After the last pass the final run is read back. The
 * counted block transfers are asserted against the documented bound 2 * ceil(n/BLOCK_SIZE) *
 * (1 + passes) - one initial write pass, two transfers per block per merge pass, one final read pass -
 * and the result is checked for equality against a naive in-memory sort (correctness only). The disk
 * also keeps every intermediate run, so the accounting assert totalItems == n * (passes + 1) verifies
 * that no element was dropped or duplicated.
 */
"use strict";

const BLOCK_SIZE = 4;
const MEMORY_BLOCKS = 4;
const FAN_IN = 4;

class SimDisk {
  constructor() {
    this.runs = new Map();
    this.nextId = 0;
    this.reads = 0;
    this.writes = 0;
  }

  createRun() {
    const runId = this.nextId++;
    this.runs.set(runId, []);
    return runId;
  }

  readBlock(runId, index) {
    this.reads++;
    return this.runs.get(runId)[index].slice();
  }

  appendBlock(runId, block) {
    this.writes++;
    this.runs.get(runId).push(block.slice());
  }

  blockCount(runId) {
    return this.runs.get(runId).length;
  }

  ioOps() {
    return this.reads + this.writes;
  }

  totalItems() {
    let total = 0;
    for (const run of this.runs.values()) {
      for (const block of run) total += block.length;
    }
    return total;
  }

  blocksSizedOk() {
    for (const run of this.runs.values()) {
      for (const block of run) {
        if (block.length < 1 || block.length > BLOCK_SIZE) return false;
      }
    }
    return true;
  }
}

function ioBound(itemCount, passes) {
  const blocks = Math.ceil(itemCount / BLOCK_SIZE);
  return 2 * blocks * (1 + passes);
}

function mergeGroup(disk, group) {
  const outId = disk.createRun();
  const blocks = group.map((runId) => disk.readBlock(runId, 0));
  const positions = new Array(group.length).fill(0);
  const indices = new Array(group.length).fill(0);
  let outBlock = [];
  for (;;) {
    let best = -1;
    for (let slot = 0; slot < group.length; slot++) {
      if (indices[slot] < blocks[slot].length) {
        if (best === -1 || blocks[slot][indices[slot]] < blocks[best][indices[best]]) best = slot;
      }
    }
    if (best === -1) break;
    outBlock.push(blocks[best][indices[best]]);
    indices[best]++;
    if (outBlock.length === BLOCK_SIZE) {
      disk.appendBlock(outId, outBlock);
      outBlock = [];
    }
    if (indices[best] === blocks[best].length) {
      positions[best]++;
      if (positions[best] < disk.blockCount(group[best])) {
        blocks[best] = disk.readBlock(group[best], positions[best]);
        indices[best] = 0;
      }
    }
  }
  if (outBlock.length > 0) disk.appendBlock(outId, outBlock);
  return outId;
}

function externalSort(data) {
  const disk = new SimDisk();
  const capacity = MEMORY_BLOCKS * BLOCK_SIZE;
  let runs = [];
  for (let start = 0; start < data.length; start += capacity) {
    const chunk = data.slice(start, start + capacity).sort((a, b) => a - b);
    const runId = disk.createRun();
    for (let index = 0; index < chunk.length; index += BLOCK_SIZE) {
      disk.appendBlock(runId, chunk.slice(index, index + BLOCK_SIZE));
    }
    runs.push(runId);
  }
  let passes = 0;
  while (runs.length > 1) {
    passes++;
    const merged = [];
    for (let start = 0; start < runs.length; start += FAN_IN) {
      merged.push(mergeGroup(disk, runs.slice(start, start + FAN_IN)));
    }
    runs = merged;
  }
  const result = [];
  if (runs.length > 0) {
    const runId = runs[0];
    for (let index = 0; index < disk.blockCount(runId); index++) {
      result.push(...disk.readBlock(runId, index));
    }
  }
  return { data: result, disk, passes };
}

module.exports = { BLOCK_SIZE, MEMORY_BLOCKS, FAN_IN, SimDisk, ioBound, externalSort, mergeGroup };

if (require.main === module) {
  const assert = require("node:assert");
  let result = externalSort([]);
  assert.deepStrictEqual(result.data, []);
  assert.strictEqual(result.passes, 0);
  assert.strictEqual(result.disk.ioOps(), 0);

  result = externalSort([7]);
  assert.deepStrictEqual(result.data, [7]);
  assert.strictEqual(result.passes, 0);
  assert.strictEqual(result.disk.reads, 1);
  assert.strictEqual(result.disk.writes, 1);

  const data = [5, 1, 4, 2, 8, 3, 6, 7, 0, 9];
  result = externalSort(data);
  assert.deepStrictEqual(result.data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.strictEqual(result.passes, 0);
  assert.ok(result.disk.ioOps() <= ioBound(data.length, result.passes));
  assert.strictEqual(result.disk.totalItems(), data.length * (result.passes + 1));
  assert.ok(result.disk.blocksSizedOk());

  const duplicates = [-3, 5, 5, 1, 0, -7, 3, 3, 3, -7, 9, 0, -1, 2, 2, 8, -5];
  result = externalSort(duplicates);
  assert.deepStrictEqual(result.data, duplicates.slice().sort((a, b) => a - b));
  assert.strictEqual(result.passes, 1);
  assert.ok(result.disk.ioOps() <= ioBound(duplicates.length, result.passes));
  assert.strictEqual(result.disk.totalItems(), duplicates.length * (result.passes + 1));
  assert.ok(result.disk.blocksSizedOk());

  const big = [];
  let seed = 4380;
  for (let i = 0; i < 2000; i++) {
    seed = (seed * 75 + 74) % 65537;
    big.push(seed % 1000);
  }
  result = externalSort(big);
  const naive = big.slice().sort((a, b) => a - b);
  assert.deepStrictEqual(result.data, naive);
  assert.strictEqual(result.passes, 4, `big passes ${result.passes}`);
  assert.ok(result.disk.ioOps() <= ioBound(big.length, result.passes), "big io bound");
  assert.strictEqual(result.disk.totalItems(), big.length * (result.passes + 1));
  assert.ok(result.disk.blocksSizedOk());
  let maxBlock = 0;
  for (const run of result.disk.runs.values()) {
    for (const block of run) maxBlock = Math.max(maxBlock, block.length);
  }
  assert.strictEqual(maxBlock, BLOCK_SIZE);

  const reversed = [];
  for (let i = 0; i < 200; i++) reversed.push(200 - i);
  result = externalSort(reversed);
  assert.deepStrictEqual(result.data, reversed.slice().sort((a, b) => a - b));
  assert.ok(result.disk.ioOps() <= ioBound(reversed.length, result.passes));
  assert.ok(result.passes > 0);
  console.log("PASS");
}
