/**
 * Task 681 - Implement numeric GPU kernel mapping to CPU threads simulation.
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A scaled-down model of a GPU launch: the grid is `ceil(n / 32)` blocks of one warp each, a warp is 32
 * lanes executing in lockstep, and the simulation reports the numbers an occupancy/coalescing tool would
 * report:
 *
 *   lanes        - the last warp is partially active; the active mask is n mod 32 (or 32 when n divides);
 *   coalescing   - each warp's accesses are mapped to 128-byte memory lines, so the transaction count per
 *                  warp is the number of distinct lines it touches (consecutive lanes cost one line,
 *                  a stride of 32 elements touches 32 lines);
 *   divergence   - a `if (i % 2)`-style kernel executes two paths per warp, so the warp occupies two
 *                  execution slots and the lanes not on the active path are masked out;
 *   occupancy    - blocks per SM limited by registers, shared memory, the block limit and the warp limit,
 *                  reported together with which limit binds.
 *
 * Verified: grid/lane decomposition for n = 0, 1, 32, 33, 100; the tail warp masks; transaction counts for
 * unit stride (one line per warp) and stride 32 (thirty-two lines per warp); a divergent warp really takes
 * two slots while still computing the same values as the sequential reference; the occupancy numbers for
 * hand-chosen limits (32 blocks limited by the warp limit, halved when registers per thread double, zero
 * when a single block cannot fit); and determinism.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed681n;
const WARP = 32;
const LINE_BYTES = 128;
const ELEMENT_BYTES = 4;

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }
  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }
  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

const sequential = (n, expr) => Array.from({ length: n }, (_, i) => expr(i));
const divergentKernel = (i) => (i % 2 === 0 ? i + 1 : -(i + 1));
const linearKernel = (i) => i + 1;

/** Distinct 128-byte lines a warp touches when lane `l` accesses element `base + l * stride`. */
function linesTouched(indices, stride, elementBytes = ELEMENT_BYTES) {
  const lines = new Set();
  for (const index of indices) lines.add(Math.floor((index * stride * elementBytes) / LINE_BYTES));
  return lines.size;
}

/**
 * One launch: warps of 32 lanes, coalescing and divergence counters, and the resulting buffer. The
 * `divergent` flag selects a kernel whose lanes split into two paths, which costs the warp two slots.
 */
function launch(options) {
  const { n, stride = 1, divergent = false, expr = linearKernel, exprElse = expr, elementBytes = ELEMENT_BYTES } = options;
  const warps = Math.ceil(n / WARP);
  const output = new Array(n).fill(0);
  const perWarp = [];
  let transactions = 0;
  let laneSlots = 0;
  let activeLanes = 0;
  for (let w = 0; w < warps; w++) {
    const lanes = [];
    for (let l = 0; l < WARP; l++) {
      const index = w * WARP + l;
      if (index < n) lanes.push(index);
    }
    const lines = linesTouched(lanes, stride, elementBytes);
    transactions += lines;
    activeLanes += lanes.length;
    let paths = 1;
    if (divergent) {
      const even = lanes.filter((i) => i % 2 === 0);
      const odd = lanes.filter((i) => i % 2 === 1);
      paths = (even.length > 0 ? 1 : 0) + (odd.length > 0 ? 1 : 0);
      for (const i of even) output[i] = expr(i);
      for (const i of odd) output[i] = exprElse(i);
    } else {
      for (const i of lanes) output[i] = expr(i);
    }
    laneSlots += paths * WARP;
    perWarp.push({ warp: w, active: lanes.length, lines, paths, mask: lanes.length === WARP ? 0xffffffff : (2 ** lanes.length) - 1 });
  }
  return { warps, transactions, laneSlots, activeLanes, perWarp, output,
    efficiency: activeLanes / Math.max(1, laneSlots) };
}

/** Occupancy: the binding limit among registers, shared memory, block count and warp count. */
function occupancy(options) {
  const { threadsPerBlock = WARP, registersPerThread, sharedPerBlock = 0, limits } = options;
  const warpsPerBlock = Math.ceil(threadsPerBlock / WARP);
  const byRegisters = Math.floor(limits.registersPerSM / (threadsPerBlock * registersPerThread));
  const byShared = sharedPerBlock === 0 ? limits.blocksPerSM : Math.floor(limits.sharedPerSM / sharedPerBlock);
  const byBlocks = limits.blocksPerSM;
  const byWarps = Math.floor(limits.maxWarpsPerSM / warpsPerBlock);
  const blocksPerSM = Math.max(0, Math.min(byRegisters, byShared, byBlocks, byWarps));
  return { blocksPerSM, warpsPerSM: blocksPerSM * warpsPerBlock, limits: { byRegisters, byShared, byBlocks, byWarps },
    occupancy: (blocksPerSM * warpsPerBlock) / limits.maxWarpsPerSM };
}

const TEST_LIMITS = { registersPerSM: 65536, sharedPerSM: 49152, blocksPerSM: 32, maxWarpsPerSM: 48 };

function main() {
  // (1) Grid and lane decomposition for a range of sizes.
  const sizes = [[0, 0, 0], [1, 1, 1], [32, 1, 32], [33, 2, 1], [100, 4, 4], [96, 3, 32]];
  for (const [n, warps, tail] of sizes) {
    const run = launch({ n });
    assert.strictEqual(run.warps, warps, `n = ${n} needs ${warps} warp(s)`);
    assert.strictEqual(run.activeLanes, n, `exactly ${n} lanes are active`);
    if (n > 0) assert.strictEqual(run.perWarp[warps - 1].active, tail, `the tail warp of n = ${n} has ${tail} lanes`);
  }
  assert.deepStrictEqual(launch({ n: 0 }).output, [], "an empty launch produces nothing");
  assert.deepStrictEqual(launch({ n: 0 }).perWarp, [], "and has no warps at all");

  // (2) Tail warp masks and lane slots: a full warp has all 32 bits set, a partial one only its lanes.
  assert.deepStrictEqual(launch({ n: 33 }).perWarp.map((w) => w.mask), [0xffffffff, 1], "the second warp has one lane");
  assert.deepStrictEqual([launch({ n: 96 }).laneSlots, launch({ n: 96 }).activeLanes], [96, 96],
    "a multiple of 32 wastes no lane slot");
  const ragged = launch({ n: 33 });
  assert.deepStrictEqual([ragged.laneSlots, ragged.efficiency], [64, 33 / 64],
    "the ragged launch occupies 64 slots for 33 useful lanes");

  // (3) Coalescing: unit stride costs one line per warp, stride 32 costs 32 lines.
  const unit = launch({ n: 64, stride: 1 });
  assert.deepStrictEqual(unit.perWarp.map((w) => w.lines), [1, 1], "consecutive lanes share one 128-byte line");
  assert.strictEqual(unit.transactions, 2, "so the whole launch needs two transactions");
  const strided = launch({ n: 64, stride: 32 });
  assert.deepStrictEqual(strided.perWarp.map((w) => w.lines), [32, 32], "a stride of 32 elements touches 32 lines per warp");
  assert.strictEqual(strided.transactions, 64, "sixty-four transactions for 64 lanes: the classic uncoalesced access");
  assert.deepStrictEqual([linesTouched([0, 1, 2, 3], 1), linesTouched([0, 32, 64, 96], 1)], [1, 4],
    "four consecutive elements share a line, four lanes 128 bytes apart do not");
  const halves = launch({ n: 64, stride: 1, elementBytes: 8 });
  assert.deepStrictEqual(halves.perWarp.map((w) => w.lines), [2, 2], "8-byte elements halve the lanes per line");

  // (4) Divergence: two paths per warp, doubling the slots but not changing the result.
  const divergent = launch({ n: 32, divergent: true, expr: linearKernel, exprElse: (i) => -(i + 1) });
  assert.deepStrictEqual(divergent.perWarp.map((w) => w.paths), [2], "the warp splits into two paths");
  assert.deepStrictEqual([divergent.laneSlots, divergent.efficiency], [64, 0.5],
    "so it occupies two slots and reaches 50% lane efficiency");
  assert.deepStrictEqual(divergent.output, sequential(32, divergentKernel),
    "documented: divergence costs throughput but never changes the values");
  assert.strictEqual([...new Set(divergent.output)].length, 32, "every lane wrote a distinct value");
  const coherent = launch({ n: 32, expr: divergentKernel });
  assert.deepStrictEqual(coherent.output, divergent.output, "the same kernel without a branch is cheaper and identical");
  assert.deepStrictEqual([coherent.laneSlots, divergent.laneSlots], [32, 64], "and needs half the slots");

  // (5) Kernels agree with the sequential reference for several sizes.
  let compared = 0;
  for (const n of [1, 7, 32, 33, 200]) {
    for (const [expr, divergentFlag] of [[linearKernel, false], [divergentKernel, true]]) {
      assert.deepStrictEqual(launch({ n, expr, divergent: divergentFlag }).output, sequential(n, expr),
        `n = ${n} with divergence = ${divergentFlag} must match the sequential reference`);
      compared += 1;
    }
  }
  assert.strictEqual(compared, 10, "ten kernel shapes were compared against the sequential reference");

  // (6) Occupancy: hand-computed limits and which one binds.
  const base = occupancy({ threadsPerBlock: 32, registersPerThread: 64, sharedPerBlock: 4096, limits: TEST_LIMITS });
  assert.deepStrictEqual([base.blocksPerSM, base.warpsPerSM, base.limits.byWarps], [12, 12, 48],
    "with 4 KiB of shared memory per block, 49152/4096 = 12 blocks fit and the warp limit is not binding");
  assert.deepStrictEqual([base.limits.byRegisters, base.limits.byShared, base.limits.byBlocks], [32, 12, 32],
    "registers would allow 32 blocks and the block limit is 32 as well");
  const noShared = occupancy({ threadsPerBlock: 32, registersPerThread: 64, sharedPerBlock: 0, limits: TEST_LIMITS });
  assert.deepStrictEqual([noShared.blocksPerSM, noShared.limits.byShared], [32, 32], "without shared memory the block limit binds");
  const heavy = occupancy({ threadsPerBlock: 32, registersPerThread: 128, sharedPerBlock: 0, limits: TEST_LIMITS });
  assert.deepStrictEqual([heavy.blocksPerSM, heavy.limits.byRegisters], [16, 16], "doubling the registers per thread halves the blocks");
  const impossible = occupancy({ threadsPerBlock: 32, registersPerThread: 4096, sharedPerBlock: 0, limits: TEST_LIMITS });
  assert.deepStrictEqual([impossible.blocksPerSM, impossible.occupancy], [0, 0], "a block that cannot fit yields zero occupancy");
  assert.ok(heavy.blocksPerSM <= noShared.blocksPerSM, "more registers can never allow more blocks");
  const wide = occupancy({ threadsPerBlock: 1024, registersPerThread: 64, sharedPerBlock: 0, limits: TEST_LIMITS });
  assert.deepStrictEqual([wide.limits.byWarps, wide.blocksPerSM], [1, 1], "1024 threads means 32 warps, so only one block fits");

  // (7) Boundary and determinism.
  assert.deepStrictEqual([launch({ n: 1 }).transactions, launch({ n: 1 }).perWarp[0].mask], [1, 1],
    "a single lane is one partial warp");
  const rng = new SplitMix64(DEFAULT_SEED);
  const sizesToTry = Array.from({ length: 6 }, () => 1 + rng.nextInt(300));
  for (const n of sizesToTry) {
    const again = launch({ n, stride: 1 });
    assert.deepStrictEqual([again.warps, again.transactions, again.output], [Math.ceil(n / 32), launch({ n }).transactions, launch({ n }).output],
      `n = ${n} is reproducible`);
  }
  assert.deepStrictEqual(occupancy({ threadsPerBlock: 32, registersPerThread: 64, sharedPerBlock: 0, limits: TEST_LIMITS }),
    occupancy({ threadsPerBlock: 32, registersPerThread: 64, sharedPerBlock: 0, limits: TEST_LIMITS }), "occupancy is deterministic");
}

module.exports = { SplitMix64, DEFAULT_SEED, WARP, LINE_BYTES, ELEMENT_BYTES, TEST_LIMITS, sequential,
  divergentKernel, linearKernel, linesTouched, launch, occupancy };

if (require.main === module) {
  main();
  console.log("PASS");
}
