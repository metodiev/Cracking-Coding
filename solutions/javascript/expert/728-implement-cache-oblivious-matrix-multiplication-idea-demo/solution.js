/** Task 728 - Implement cache-oblivious matrix multiplication idea demo. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * A cache-oblivious algorithm recurses until the subproblem fits *some* cache level and then
 * stops by itself; because the recursion is oblivious to Z and L it is simultaneously optimal
 * for the whole memory hierarchy (Frigo, Leiserson, Prokop, Ramachandran). The textbook
 * example is matrix multiplication: split both operands into quadrants and recurse into the 8
 * sub-products -- no tuning knob, and each level touches the operands in contiguous blocks.
 *
 * This file implements the recursive kernel next to the cache-aware ikj triple loop and a
 * simulated fully associative LRU cache (line size L, capacity Z) that counts every miss, so
 * the "cache-oblivious" claim can be measured instead of believed: for a small cache the
 * recursive order should miss far less often than the row-major loop, and both must compute
 * exactly the same product.
 *
 * Verified here: the cache model reproduces a hand-computed miss sequence; both kernels give
 * bit-identical integer products (the payload is small so the sums are exact); the arithmetic
 * performed is identical (n^3 multiplies, n^3 adds) while the miss counts differ hugely; a base
 * case at least as large as the matrix degenerates to the plain loop with *equal* counters;
 * the best base case is interior rather than degenerate; and the degenerate sizes (0, 1) behave.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

/** Deterministic splitmix64 -> [0, 1). */
function makeRng(seed) {
  let state = (BigInt(seed >>> 0) + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
  return function next() {
    state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    z ^= z >> 31n;
    return Number(z >> 11n) / 9007199254740992;
  };
}

/**
 * Fully associative LRU cache over cache lines. `access` returns 1 for a miss and 0 for a hit;
 * the Map's insertion order is used as the LRU order (delete + set moves a line to the front).
 */
class Cache {
  constructor(lineSize, lineCount) {
    ok(Number.isInteger(lineSize) && lineSize > 0, "line size must be positive");
    ok(Number.isInteger(lineCount) && lineCount > 0, "line count must be positive");
    this.lineSize = lineSize;
    this.lineCount = lineCount;
    this.lines = new Map();
    this.hits = 0;
    this.misses = 0;
  }

  access(address) {
    const line = Math.floor(address / this.lineSize);
    if (this.lines.has(line)) {
      this.lines.delete(line);
      this.lines.set(line, true);
      this.hits++;
      return 0;
    }
    this.misses++;
    if (this.lines.size === this.lineCount) this.lines.delete(this.lines.keys().next().value);
    this.lines.set(line, true);
    return 1;
  }

  get volume() {
    return this.lineSize * this.lineCount;
  }
}

/** Machine = the cache plus the arithmetic counters shared by every kernel below. */
class Machine {
  constructor(lineSize, lineCount) {
    this.cache = new Cache(lineSize, lineCount);
    this.multiplies = 0;
    this.adds = 0;
  }

  touch(address) {
    return this.cache.access(address);
  }
}

/** Plain row-major ikj triple loop with the block origin and leading dimension given. */
function blockMultiply(a, b, c, size, machine, origin) {
  const { ao, bo, co, stride, bases } = origin;
  for (let i = 0; i < size; i++) {
    for (let k = 0; k < size; k++) {
      const av = a[ao + i * stride + k];
      machine.touch(bases.a + ao + i * stride + k);
      for (let j = 0; j < size; j++) {
        machine.touch(bases.b + bo + k * stride + j);
        machine.touch(bases.c + co + i * stride + j);
        c[co + i * stride + j] += av * b[bo + k * stride + j];
        machine.multiplies++;
        machine.adds++;
      }
    }
  }
}

/** Cache-aware baseline: the whole matrix in ikj order. */
function rowMajorMultiply(a, b, c, size, machine, bases) {
  blockMultiply(a, b, c, size, machine, { ao: 0, bo: 0, co: 0, stride: size, bases });
}

/**
 * Cache-oblivious kernel: split into quadrants, recurse into the 8 sub-products, stop as soon
 * as the block is small enough for the caller's base case. All eight calls accumulate into C.
 */
function recursiveMultiply(a, b, c, size, machine, origin, baseCase) {
  const { ao, bo, co, stride, bases } = origin;
  if (size <= baseCase) {
    blockMultiply(a, b, c, size, machine, origin);
    return;
  }
  const half = size >> 1;
  const ar = ao + half;                 // A12
  const ac = ao + half * stride;        // A21
  const acc = ac + half;                // A22
  const br = bo + half;                 // B12
  const bc = bo + half * stride;        // B21
  const bcc = bc + half;                // B22
  const cr = co + half;                 // C12
  const cc = co + half * stride;        // C21
  const ccc = cc + half;                // C22
  const sub = (ao2, bo2, co2) => recursiveMultiply(a, b, c, half, machine,
    { ao: ao2, bo: bo2, co: co2, stride, bases }, baseCase);
  sub(ao, bo, co);                      // C11 += A11 B11
  sub(ao, br, cr);                      // C12 += A11 B12
  sub(ar, bc, co);                      // C11 += A12 B21
  sub(ar, bcc, cr);                     // C12 += A12 B22
  sub(ac, bo, cc);                      // C21 += A21 B11
  sub(ac, br, ccc);                     // C22 += A21 B12
  sub(acc, bc, cc);                     // C21 += A22 B21
  sub(acc, bcc, ccc);                   // C22 += A22 B22
}

const MARKS = { a: 0, b: 1e6, c: 2e6 };

/** Zero-filled matrices and a run harness that returns the product plus every counter. */
function multiplyBoth(size, baseCase, { lineSize, lineCount }) {
  const rng = makeRng(20260728 + size * 31 + baseCase);
  const a = Array.from({ length: size * size }, () => Math.floor(rng() * 9) - 4);
  const b = Array.from({ length: size * size }, () => Math.floor(rng() * 7) - 3);
  const left = new Array(size * size).fill(0);
  const right = new Array(size * size).fill(0);
  const loopMachine = new Machine(lineSize, lineCount);
  rowMajorMultiply(a, b, left, size, loopMachine, MARKS);
  const recursiveMachine = new Machine(lineSize, lineCount);
  if (size > 0) {
    recursiveMultiply(a, b, right, size, recursiveMachine,
      { ao: 0, bo: 0, co: 0, stride: size, bases: MARKS }, Math.max(1, baseCase));
  }
  return { left, right, loopMachine, recursiveMachine, a, b };
}

const grid = (matrix, size) => Array.from({ length: size }, (_, r) => matrix.slice(r * size, (r + 1) * size));

module.exports = { makeRng, Cache, Machine, blockMultiply, rowMajorMultiply, recursiveMultiply, multiplyBoth, grid };

if (require.main === module) {
  // -- the cache model itself against hand-computed sequences -----------------------------
  const probe = new Cache(4, 2);            // 2 lines of 4 elements
  const touches = [0, 1, 2, 3, 4, 0, 5, 4, 8, 0];
  const handCounted = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1];
  deepStrictEqual(touches.map((address) => probe.access(address)), handCounted,
    "an LRU with two lines must reproduce the hand-counted miss sequence");
  strictEqual(probe.misses, handCounted.filter((value) => value === 1).length, "misses must be counted");
  strictEqual(probe.hits + probe.misses, touches.length, "every access is a hit or a miss");
  deepStrictEqual([...new Cache(4, 2).lines.keys()], [], "a fresh cache is empty");
  throws(() => new Cache(0, 4), /line size must be positive/);
  throws(() => new Cache(4, 0), /line count must be positive/);
  console.log(`cache model: ${probe.misses} misses / ${probe.hits} hits on the probe sequence`);

  // -- both kernels compute exactly the same product -------------------------------------
  let compared = 0;
  for (const size of [1, 2, 4, 8, 16, 32]) {
    for (const baseCase of [1, 2, 4, 8, 64]) {
      const run = multiplyBoth(size, baseCase, { lineSize: 8, lineCount: 16 });
      deepStrictEqual(run.right, run.left, `${size}x${size} base ${baseCase} must match the ikj loop`);
      strictEqual(run.recursiveMachine.multiplies, size ** 3, "both kernels do n^3 multiplies");
      strictEqual(run.recursiveMachine.adds, run.loopMachine.adds, "and the same number of additions");
      compared++;
    }
  }
  console.log(`recursive vs row-major: ${compared} size/base-case combinations agree exactly`);

  // -- base case at least the size degenerates to the plain loop --------------------------
  const degenerate = multiplyBoth(32, 64, { lineSize: 8, lineCount: 16 });
  strictEqual(degenerate.recursiveMachine.cache.misses, degenerate.loopMachine.cache.misses,
    "a base case covering the whole matrix is byte-for-byte the row-major loop");
  ok(degenerate.loopMachine.cache.misses > 0, "the plain loop misses at least once");

  // -- the cache-oblivious claim: far fewer misses on a small cache ------------------------
  const size = 64;
  const small = { lineSize: 8, lineCount: 32 };        // 256 elements of a 4096-element matrix
  const baseCases = [4, 8, 16, 32, 64];
  const scan = baseCases.map((baseCase) => {
    const run = multiplyBoth(size, baseCase, small);
    return { baseCase, misses: run.recursiveMachine.cache.misses, loopMisses: run.loopMachine.cache.misses };
  });
  const naive = scan[scan.length - 1];
  const tuned = scan.reduce((left, right) => (right.misses < left.misses ? right : left));
  for (const entry of scan) {
    strictEqual(entry.loopMisses, naive.loopMisses, "the row-major loop ignores the base case");
  }
  ok(tuned.baseCase < size, "the best base case must be a real recursion (not the degenerate one)");
  ok(tuned.misses * 2 < naive.misses,
    `the recursive order must halve the misses at least: ${tuned.misses} vs ${naive.misses}`);
  ok(tuned.misses < tuned.loopMisses,
    `and it must beat the row-major loop: ${tuned.misses} vs ${tuned.loopMisses}`);
  console.log(`64x64 with a ${small.lineSize * small.lineCount}-element cache:` +
    ` recursive ${scan.map((s) => `b${s.baseCase}:${s.misses}`).join(" ")} vs row-major ${naive.loopMisses}`);

  // -- a large cache removes the advantage, as the theory predicts ------------------------
  const large = { lineSize: 8, lineCount: 1024 };      // 8192 elements: the whole problem fits
  const bigRun = multiplyBoth(size, 8, large);
  ok(bigRun.loopMachine.cache.misses < naive.loopMisses / 10,
    "a bigger cache misses far less than the small one");
  ok(bigRun.recursiveMachine.cache.misses <= bigRun.loopMachine.cache.misses * 2,
    "once the problem nearly fits in cache the two orders become comparable");
  const fits = multiplyBoth(size, 8, { lineSize: 8, lineCount: 2048 });
  strictEqual(fits.recursiveMachine.cache.misses, fits.loopMachine.cache.misses,
    "with the whole working set resident both orders hit the compulsory-miss floor");
  ok(fits.recursiveMachine.cache.misses > 0, "and that floor is not zero: every line is read once");
  const capacityRun = multiplyBoth(size, 8, { lineSize: 8, lineCount: 512 });
  ok(capacityRun.recursiveMachine.cache.misses < capacityRun.loopMachine.cache.misses,
    "the recursive order stays ahead across cache sizes");
  console.log(`cache sweep (line 8): Z=256 recursive=${tuned.misses} loop=${tuned.loopMisses}` +
    ` | Z=4096 recursive=${capacityRun.recursiveMachine.cache.misses} loop=${capacityRun.loopMachine.cache.misses}` +
    ` | Z=8192 recursive=${bigRun.recursiveMachine.cache.misses} loop=${bigRun.loopMachine.cache.misses}`);

  // -- invariants of the product and of the counters ---------------------------------------
  const check = multiplyBoth(8, 2, { lineSize: 4, lineCount: 8 });
  const identity = Array.from({ length: 8 }, (_, r) => Array.from({ length: 8 }, (_, c) => (r === c ? 1 : 0)));
  const identityRun = { a: identity.flat(), b: check.a, c: new Array(64).fill(0) };
  const identityMachine = new Machine(4, 8);
  rowMajorMultiply(identityRun.a, identityRun.b, identityRun.c, 8, identityMachine, MARKS);
  deepStrictEqual(identityRun.c, check.a, "A * I = A holds in the flat layout");
  const rowSums = (matrix) => matrix.map((row) => row.reduce((acc, value) => acc + value, 0));
  deepStrictEqual(rowSums(grid(check.left, 8)), rowSums(grid(check.right, 8)),
    "row sums of both products must agree");
  const multiplyTotal = check.left.every((value, i) => value === check.right[i]);
  ok(multiplyTotal, "every element of C must match element-wise");

  // -- degenerate sizes ---------------------------------------------------------------------
  const empty = multiplyBoth(0, 4, { lineSize: 4, lineCount: 4 });
  strictEqual(empty.loopMachine.cache.misses, 0, "a 0x0 product touches no memory");
  strictEqual(empty.recursiveMachine.cache.misses, 0);
  deepStrictEqual(empty.left, []);
  const single = multiplyBoth(1, 1, { lineSize: 4, lineCount: 1 });
  strictEqual(single.loopMachine.cache.misses, 3, "1x1 reads A, reads B and writes C: three accesses");
  strictEqual(single.recursiveMachine.cache.misses, 3, "and the recursion stops immediately");
  deepStrictEqual(single.left, single.right);
  const tinyBase = multiplyBoth(16, 0, { lineSize: 4, lineCount: 8 });
  deepStrictEqual(tinyBase.right, tinyBase.left, "a base case of 0 is clamped to 1 and stays correct");

  // -- determinism ---------------------------------------------------------------------------
  const replay = () => {
    const run = multiplyBoth(16, 4, { lineSize: 4, lineCount: 8 });
    return `${run.recursiveMachine.cache.misses}/${run.loopMachine.cache.misses}/${run.right.join(",")}`;
  };
  deepStrictEqual(replay(), replay(), "identical inputs must give identical misses and products");
  console.log("PASS");
}
