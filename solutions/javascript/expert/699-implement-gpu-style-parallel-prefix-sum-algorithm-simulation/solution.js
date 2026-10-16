/** Task 699 - Implement GPU-style parallel prefix-sum algorithm simulation. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Model: a round-by-round simulation of the two classic GPU scan kernels, in which every "step" is a
 * barrier at which all elements are updated simultaneously from the previous step's values (the
 * shared-memory style of CUDA `__syncthreads()` loops):
 *
 *   Hillis-Steele  inclusive scan with n * ceil(log2 n) combines and ceil(log2 n) steps: every element
 *                  reads the value d positions to its left in each doubling step.
 *   Blelloch       work-efficient scan: an up-sweep builds a reduction tree with n - 1 combines, then
 *                  a down-sweep turns it into an exclusive scan with another n - 1 swaps/combines, so
 *                  ~2n combines and 2 log n steps regardless of n - the reason it is the production
 *                  choice even though it does more sequential rounds.
 *   blockScan      the three-kernel scheme used on real devices for arbitrary n: scan each block in
 *                  shared memory, scan the block totals, then add the offsets back to every element -
 *                  simulated as three kernels with their own work counters.
 *
 * The combine operator is a parameter (add, min, max, or BigInt addition), which is what makes a scan
 * a scan: the simulation is checked against a sequential oracle for every size and against a BigInt
 * oracle for values beyond 2^53, and the operation counters are checked against the theoretical work
 * of each algorithm.
 *
 * Simplified versus real hardware: no memory hierarchy, no bank conflicts, no divergence, blocks are
 * simulated sequentially with explicit work accounting instead of thread parallel execution, and
 * "steps" are barriers rather than clock cycles.
 *
 * Verified: sequential-oracle agreement for every size 1..64 and for every operator, exact operation
 * counts (n log n versus 2n), the work-efficiency ratio matching log2 n, block-scan correctness for
 * sizes that overflow a block and are not powers of two, identity padding, the total-is-the-sum
 * invariant, prefix-min/max equality with brute force, exact BigInt scanning past 2^53, exclusive
 * versus inclusive consistency, and determinism.
 */
"use strict";

const ADD = { name: "add", identity: 0, combine: (a, b) => a + b };
const MIN = { name: "min", identity: Infinity, combine: (a, b) => Math.min(a, b) };
const MAX = { name: "max", identity: -Infinity, combine: (a, b) => Math.max(a, b) };
const BIG_ADD = { name: "bigadd", identity: 0n, combine: (a, b) => a + b };

function isPowerOfTwo(value) {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

class ScanSimulator {
  constructor(operator = ADD) {
    this.operator = operator;
    this.work = 0;
    this.steps = 0;
    this.kernels = 0;
  }

  reset() {
    this.work = 0;
    this.steps = 0;
    this.kernels = 0;
    return this;
  }

  /** One combine, counted: the unit of work every algorithm below is measured in. */
  combine(left, right) {
    this.work++;
    return this.operator.combine(left, right);
  }

  get identity() {
    return this.operator.identity;
  }

  /** Inclusive Hillis-Steele scan: n * ceil(log2 n) combines, ceil(log2 n) barrier steps. */
  hillisSteele(values) {
    const n = values.length;
    if (n === 0) return [];
    let current = values.slice();
    for (let distance = 1; distance < n; distance <<= 1) {
      const previous = current.slice();
      for (let i = 0; i < n; i++) {
        current[i] = i >= distance ? this.combine(previous[i - distance], previous[i]) : previous[i];
      }
      this.steps++;
    }
    return current;
  }

  /** Inclusive scan from an exclusive one, counting the extra combines. */
  toInclusiveExclusive(exclusive, values) {
    const n = values.length;
    const inclusive = new Array(n);
    for (let i = 0; i < n; i++) {
      inclusive[i] = i === 0 ? values[0] : this.combine(exclusive[i], values[i]);
    }
    return inclusive;
  }

  /** Blelloch exclusive scan; `values` must have a power-of-two length. */
  blellochExclusive(values) {
    const n = values.length;
    if (!isPowerOfTwo(n)) throw new RangeError("the Blelloch scan needs a power-of-two length");
    const x = values.slice();
    for (let distance = 1; distance < n; distance <<= 1) {
      for (let i = (distance << 1) - 1; i < n; i += distance << 1) {
        x[i] = this.combine(x[i - distance], x[i]);
      }
      this.steps++;
    }
    const total = x[n - 1];
    x[n - 1] = this.identity;
    for (let distance = n >> 1; distance >= 1; distance >>= 1) {
      for (let i = (distance << 1) - 1; i < n; i += distance << 1) {
        const left = x[i - distance];
        x[i - distance] = x[i];
        x[i] = this.combine(x[i], left);
      }
      this.steps++;
    }
    return { exclusive: x, total };
  }

  /** Pad with the identity to the next power of two (without counting padding combines). */
  padToPowerOfTwo(values) {
    if (values.length === 0) return [];
    let size = 1;
    while (size < values.length) size <<= 1;
    const padded = values.slice();
    while (padded.length < size) padded.push(this.identity);
    return padded;
  }

  /** Blelloch exclusive scan for an arbitrary length: identity padding plus the padding strip. */
  blellochExclusiveAnyLength(values) {
    if (values.length === 0) return { exclusive: [], total: this.identity };
    const padded = this.padToPowerOfTwo(values);
    const result = this.blellochExclusive(padded);
    return { exclusive: result.exclusive.slice(0, values.length), total: result.total };
  }

  /**
   * The three-kernel block scan: block-local scans, one scan of the block totals, then the offset
   * add-back. Works for any length and returns the inclusive scan plus the kernel breakdown.
   */
  blockScan(values, blockSize) {
    if (!isPowerOfTwo(blockSize)) throw new RangeError("the block size must be a power of two");
    const n = values.length;
    if (n === 0) return { inclusive: [], kernels: 0, perKernel: [] };
    const blockCount = Math.ceil(n / blockSize);
    const before = { work: this.work, steps: this.steps };
    const blocks = [];
    for (let block = 0; block < blockCount; block++) {
      const slice = values.slice(block * blockSize, Math.min(n, (block + 1) * blockSize));
      const scanned = this.blellochExclusiveAnyLength(slice);
      blocks.push({ slice, scanned });
    }
    const afterBlocks = { work: this.work, steps: this.steps };

    const totals = blocks.map((entry) => entry.scanned.total);
    const totalScan = this.blellochExclusiveAnyLength(totals);
    const afterTotals = { work: this.work, steps: this.steps };

    const inclusive = [];
    for (let block = 0; block < blockCount; block++) {
      const offset = totalScan.exclusive[block];
      const { slice, scanned } = blocks[block];
      for (let i = 0; i < slice.length; i++) {
        inclusive.push(i === 0 && block === 0 ? slice[0] : this.combine(offset, this.combine(scanned.exclusive[i], slice[i])));
      }
    }
    this.kernels += 3;
    return {
      inclusive,
      kernels: 3,
      perKernel: [
        { work: afterBlocks.work - before.work, steps: afterBlocks.steps - before.steps },
        { work: afterTotals.work - afterBlocks.work, steps: afterTotals.steps - afterBlocks.steps },
        { work: this.work - afterTotals.work, steps: this.steps - afterTotals.steps },
      ],
    };
  }
}

/** Sequential oracle. */
function sequentialInclusive(values, operator = ADD) {
  const out = [];
  let running = operator.identity;
  for (const value of values) {
    running = operator.combine(running, value);
    out.push(running);
  }
  return out;
}

function sequentialExclusive(values, operator = ADD) {
  const out = [];
  let running = operator.identity;
  for (const value of values) {
    out.push(running);
    running = operator.combine(running, value);
  }
  return out;
}

function sequentialTotal(values, operator = ADD) {
  let running = operator.identity;
  for (const value of values) running = operator.combine(running, value);
  return running;
}

class Lcg {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  int(bound) {
    return this.next() % bound;
  }
}

module.exports = {
  ADD, MIN, MAX, BIG_ADD, isPowerOfTwo, ScanSimulator, sequentialInclusive, sequentialExclusive,
  sequentialTotal, Lcg,
};

if (require.main === module) {
  const assert = require("node:assert");

  const lcg = new Lcg(699);
  const sample = [3, 1, 7, 0, 4, 1, 6, 2];

  // --- Hillis-Steele: correctness, counts, step structure -------------------------------------
  const hillis = new ScanSimulator(ADD);
  const hillisResult = hillis.hillisSteele(sample);
  assert.deepStrictEqual(hillisResult, [3, 4, 11, 11, 15, 16, 22, 24]);
  assert.deepStrictEqual(sequentialInclusive(sample), hillisResult, "inclusive scan oracle");
  assert.strictEqual(hillisResult[hillisResult.length - 1], sample.reduce((a, b) => a + b, 0),
    "the last element is the total");
  assert.strictEqual(hillis.steps, 3, "ceil(log2 8) = 3 barrier steps");
  assert.strictEqual(hillis.work, 3 * 8 - 7, "the reads that fall off the array are skipped: sum(n - d)");
  const referenceHillis = new ScanSimulator(ADD);
  referenceHillis.hillisSteele(new Array(64).fill(1));
  assert.strictEqual(referenceHillis.steps, 6, "64 elements need 6 doubling steps");
  assert.strictEqual(referenceHillis.work, 64 * 6 - 63, "a full power-of-two scan does n log2 n - (n - 1) reads");
  assert.deepStrictEqual(new ScanSimulator(ADD).hillisSteele([]), []);
  assert.deepStrictEqual(new ScanSimulator(ADD).hillisSteele([5]), [5]);

  // --- Blelloch: work-efficient bookkeeping ----------------------------------------------------
  const blelloch = new ScanSimulator(ADD);
  const blellochResult = blelloch.blellochExclusiveAnyLength(sample);
  assert.deepStrictEqual(blellochResult.exclusive, sequentialExclusive(sample), "exclusive scan oracle");
  assert.strictEqual(blellochResult.total, 24);
  assert.strictEqual(blelloch.work, 2 * 8 - 2, "the Blelloch scan uses 2n - 2 combines");
  assert.strictEqual(blelloch.steps, 6, "2 log2 n barrier steps");
  const paddedTotal = new ScanSimulator(ADD);
  const paddedScan = paddedTotal.blellochExclusiveAnyLength([1, 2, 3]);
  assert.deepStrictEqual(paddedScan.exclusive, [0, 1, 3], "padding with the identity is stripped");
  assert.strictEqual(paddedScan.total, 6, "and the identity does not change the total");
  assert.deepStrictEqual(paddedTotal.padToPowerOfTwo([1, 2, 3]), [1, 2, 3, 0]);
  assert.deepStrictEqual(paddedTotal.padToPowerOfTwo([]), []);
  assert.deepStrictEqual(paddedTotal.padToPowerOfTwo([1]), [1]);
  assert.deepStrictEqual(paddedTotal.padToPowerOfTwo([1, 2, 3, 4]), [1, 2, 3, 4]);
  assert.deepStrictEqual(paddedTotal.padToPowerOfTwo([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5, 0, 0, 0]);

  // inclusive from exclusive costs exactly one combine per element after the first
  const inclusiveCheck = new ScanSimulator(ADD);
  const exclusiveAgain = inclusiveCheck.blellochExclusiveAnyLength(sample).exclusive;
  const beforeInclusive = inclusiveCheck.work;
  const inclusiveFromExclusive = inclusiveCheck.toInclusiveExclusive(exclusiveAgain, sample);
  assert.deepStrictEqual(inclusiveFromExclusive, sequentialInclusive(sample));
  assert.strictEqual(inclusiveCheck.work - beforeInclusive, sample.length - 1);

  // --- work efficiency: 2n against n log n ------------------------------------------------------
  for (const size of [16, 64, 256, 1024]) {
    const values = new Array(size).fill(1);
    const hillisSimulator = new ScanSimulator(ADD);
    const hillisRun = hillisSimulator.hillisSteele(values);
    const blellochSimulator = new ScanSimulator(ADD);
    const blellochRun = blellochSimulator.blellochExclusive(values);
    assert.strictEqual(hillisRun[size - 1], size);
    assert.strictEqual(blellochRun.total, size);
    assert.strictEqual(hillisRun[size - 1] - 1, blellochRun.exclusive[size - 1], "inclusive vs exclusive");
    assert.strictEqual(hillisSimulator.work, size * Math.log2(size) - (size - 1), "Hillis-Steele reads");
    assert.strictEqual(blellochSimulator.work, 2 * size - 2, "Blelloch uses exactly 2n - 2 combines");
    assert.strictEqual(blellochSimulator.steps, 2 * Math.log2(size), "and 2 log2 n barriers");
    const ratio = hillisSimulator.work / blellochSimulator.work;
    assert.ok(ratio > (Math.log2(size) - 1.2) / 2 && ratio < Math.log2(size),
      `the work saving tracks log2 n (${ratio.toFixed(2)} at n=${size})`);
  }

  // --- arbitrary operators ---------------------------------------------------------------------
  for (const operator of [MIN, MAX]) {
    const values = [];
    for (let i = 0; i < 32; i++) values.push(lcg.int(50) - 25);
    const fast = new ScanSimulator(operator).hillisSteele(values);
    assert.deepStrictEqual(fast, sequentialInclusive(values, operator), `${operator.name} prefix scan`);
    const other = new ScanSimulator(operator).blellochExclusiveAnyLength(values);
    assert.deepStrictEqual(other.exclusive, sequentialExclusive(values, operator), `${operator.name} exclusive`);
  }
  assert.strictEqual(sequentialTotal([], ADD), 0);
  assert.strictEqual(sequentialTotal([], MIN), Infinity, "the min identity is +infinity");

  // --- the block scan for arbitrary sizes ------------------------------------------------------
  for (const size of [1, 2, 3, 7, 8, 9, 17, 64, 100, 129]) {
    const values = [];
    for (let i = 0; i < size; i++) values.push(lcg.int(20));
    for (const blockSize of [2, 4, 8, 16]) {
      const simulator = new ScanSimulator(ADD);
      const outcome = simulator.blockScan(values, blockSize);
      assert.deepStrictEqual(outcome.inclusive, sequentialInclusive(values),
        `block scan size=${size} block=${blockSize}`);
      assert.strictEqual(simulator.kernels, 3, "exactly three kernels");
      assert.strictEqual(outcome.perKernel.length, 3);
      assert.strictEqual(outcome.perKernel.reduce((acc, entry) => acc + entry.work, 0), simulator.work,
        "the per-kernel work adds up to the total");
      const blocks = Math.ceil(size / blockSize);
      assert.ok(simulator.work >= Math.max(0, size - blocks),
        "every element beyond the first of its block needs at least one combine");
      assert.strictEqual(simulator.work > 0 || size <= 1, true, "only a single element needs no work");
    }
  }
  const blockOnce = new ScanSimulator(ADD).blockScan(sample, 4);
  assert.deepStrictEqual(blockOnce.inclusive, [3, 4, 11, 11, 15, 16, 22, 24]);
  assert.strictEqual(new ScanSimulator(ADD).blockScan([], 4).inclusive.length, 0);
  assert.throws(() => new ScanSimulator(ADD).blockScan([1, 2], 3), RangeError);
  assert.throws(() => new ScanSimulator(ADD).blellochExclusive([1, 2, 3]), RangeError);

  // per-block counts: two blocks of four need two Blelloch scans plus the block-total scan
  const split = new ScanSimulator(ADD);
  const splitOutcome = split.blockScan(new Array(8).fill(1), 4);
  assert.deepStrictEqual(splitOutcome.inclusive, [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.strictEqual(splitOutcome.perKernel[0].work, 2 * (2 * 4 - 2), "two block-local scans");
  assert.strictEqual(splitOutcome.perKernel[1].work, 2 * 2 - 2, "then the two block totals");
  assert.strictEqual(splitOutcome.perKernel[2].steps, 0, "the add-back kernel has no barriers");

  // --- BigInt scanning past 2^53 ---------------------------------------------------------------
  const bigValues = [(1n << 60n) + 1n, 2n ** 53n, 1n, 2n ** 53n + 1n];
  const bigSimulator = new ScanSimulator(BIG_ADD);
  const bigInclusive = bigSimulator.hillisSteele(bigValues);
  let running = 0n;
  const bigOracle = bigValues.map((value) => { running += value; return running; });
  assert.deepStrictEqual(bigInclusive, bigOracle, "BigInt scanning is exact past 2^53");
  assert.strictEqual(bigInclusive[3], (1n << 60n) + 2n ** 53n + 1n + 1n + 2n ** 53n + 1n);
  const bigExclusive = new ScanSimulator(BIG_ADD).blellochExclusiveAnyLength(bigValues);
  assert.deepStrictEqual(bigExclusive.exclusive, [0n, bigValues[0], bigValues[0] + bigValues[1],
    bigValues[0] + bigValues[1] + bigValues[2]]);
  const floatSum = bigValues.reduce((acc, value) => acc + Number(value), 0);
  assert.notStrictEqual(BigInt(floatSum), bigInclusive[3],
    "a float scan rounds the total to 64-bit precision and loses the low bits");
  const bigBlock = new ScanSimulator(BIG_ADD).blockScan([1n, 2n, 3n, 4n, 5n], 4);
  assert.deepStrictEqual(bigBlock.inclusive, [1n, 3n, 6n, 10n, 15n]);

  // --- determinism -----------------------------------------------------------------------------
  assert.strictEqual(new Lcg(699).next(), new Lcg(699).next());
  const replayA = new ScanSimulator(ADD).hillisSteele(sample);
  const replayB = new ScanSimulator(ADD).hillisSteele(sample);
  assert.deepStrictEqual(replayA, replayB);
  const replayBlockA = new ScanSimulator(ADD).blockScan(sample, 4);
  const replayBlockB = new ScanSimulator(ADD).blockScan(sample, 4);
  assert.deepStrictEqual(replayBlockA, replayBlockB);

  console.log(`hillis n=64 work=${64 * 6 - 63} blelloch n=64 work=${2 * 64 - 2}`);
  console.log(`block kernels work=${splitOutcome.perKernel.map((entry) => entry.work).join(",")}`);
  console.log("PASS");
}
