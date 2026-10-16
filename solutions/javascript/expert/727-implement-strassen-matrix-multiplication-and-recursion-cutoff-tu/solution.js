/** Task 727 - Implement Strassen matrix multiplication and recursion cutoff tuning. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Strassen splits each operand into quadrants and computes the product with 7 recursive
 * multiplications instead of 8, at the cost of 18 quadrant additions per level. The recursion
 * is only worth it above a *cutoff*: below it the naive triple loop wins on constant factors,
 * so a real implementation multiplies by the classic algorithm until the blocks are small
 * enough (Eigen, MKL and NumPy all tune exactly this knob).
 *
 * This file implements naive and Strassen products over integer payloads (integer arithmetic
 * is exact in float64 for the magnitudes used here, so every comparison below is an equality
 * rather than a tolerance), pads any rectangular shape up to power-of-two dimensions so the
 * quadrant split is always exact, and counts the scalar multiplications and additions each
 * strategy performs.
 *
 * Verified here: Strassen equals the naive product element-for-element for power-of-two,
 * odd, rectangular and degenerate shapes and for every cutoff; the padding adds only zeros;
 * the measured multiply count matches the closed form 7^(log2(n/c)) * c^3 and the 7-way
 * branching claim (7^k scalar multiplications at cutoff 1); the cutoff scan picks a value
 * that beats both extremes; a large product really does fewer scalar multiplications than the
 * naive loop; associativity, identity and transposition identities hold; invalid shapes are
 * rejected; and the whole thing is deterministic.
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

const zeros = (rows, cols) => Array.from({ length: rows }, () => new Array(cols).fill(0));

const nextPowerOfTwo = (value) => {
  let size = 1;
  while (size < value) size *= 2;
  return size;
};

/** Copy a matrix into the top-left corner of a larger zero matrix. */
function pad(matrix, rows, cols) {
  const out = zeros(rows, cols);
  for (let r = 0; r < matrix.length; r++) {
    for (let c = 0; c < matrix[r].length; c++) out[r][c] = matrix[r][c];
  }
  return out;
}

const shape = (matrix) => (matrix.length === 0 ? [0, 0] : [matrix.length, matrix[0].length]);

function assertShapes(a, b) {
  const [aRows, aCols] = shape(a);
  const [bRows, bCols] = shape(b);
  for (const row of a) strictEqual(row.length, aCols, "A must be rectangular");
  for (const row of b) strictEqual(row.length, bCols, "B must be rectangular");
  if (aCols !== bRows) throw new Error(`shape mismatch: ${aRows}x${aCols} times ${bRows}x${bCols}`);
  return { aRows, aCols, bRows, bCols };
}

/** Naive triple loop; `counter` records the scalar work for the tuning checks. */
function naiveMultiply(a, b, counter = null) {
  const { aRows, aCols, bCols } = assertShapes(a, b);
  const out = zeros(aRows, bCols);
  for (let r = 0; r < aRows; r++) {
    for (let k = 0; k < aCols; k++) {
      const value = a[r][k];
      for (let c = 0; c < bCols; c++) {
        out[r][c] += value * b[k][c];
        if (counter) {
          counter.multiplies++;
          counter.adds++;
        }
      }
    }
  }
  return out;
}

/**
 * The seven products and the 18 quadrant combinations of one Strassen level, in the standard
 * M1..M7 labelling: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4,
 * C22 = M1 - M2 + M3 + M6.
 */
function strassenBlocks(a, b, counter) {
  const half = a.length / 2;
  const quad = (matrix, r0, c0) => {
    const block = zeros(half, half);
    for (let r = 0; r < half; r++) for (let c = 0; c < half; c++) block[r][c] = matrix[r0 + r][c0 + c];
    return block;
  };
  const sub = (left, right) => left.map((row, r) => row.map((value, c) => {
    counter.adds++;
    return value - right[r][c];
  }));
  const add = (left, right) => left.map((row, r) => row.map((value, c) => {
    counter.adds++;
    return value + right[r][c];
  }));
  const a11 = quad(a, 0, 0);
  const a12 = quad(a, 0, half);
  const a21 = quad(a, half, 0);
  const a22 = quad(a, half, half);
  const b11 = quad(b, 0, 0);
  const b12 = quad(b, 0, half);
  const b21 = quad(b, half, 0);
  const b22 = quad(b, half, half);
  const m1 = strassen(add(a11, a22), add(b11, b22), counter);
  const m2 = strassen(add(a21, a22), b11, counter);
  const m3 = strassen(a11, sub(b12, b22), counter);
  const m4 = strassen(a22, sub(b21, b11), counter);
  const m5 = strassen(add(a11, a12), b22, counter);
  const m6 = strassen(sub(a21, a11), add(b11, b12), counter);
  const m7 = strassen(sub(a12, a22), add(b21, b22), counter);
  const c11 = add(sub(add(m1, m4), m5), m7);
  const c12 = add(m3, m5);
  const c21 = add(m2, m4);
  const c22 = add(sub(add(m1, m3), m2), m6);
  const blocksBack = [[c11, c12], [c21, c22]];
  const out = zeros(a.length, a.length);
  for (let br = 0; br < 2; br++) {
    for (let bc = 0; bc < 2; bc++) {
      for (let r = 0; r < half; r++) {
        for (let c = 0; c < half; c++) out[br * half + r][bc * half + c] = blocksBack[br][bc][r][c];
      }
    }
  }
  return out;
}

/** Recursive Strassen product of square, power-of-two, equally sized operands. */
function strassen(a, b, counter) {
  if (a.length === 0) return [];
  if (a.length <= counter.cutoff) {
    counter.baseCalls++;
    return naiveMultiply(a, b, counter);
  }
  counter.levels++;
  return strassenBlocks(a, b, counter);
}

/** Public entry point: validates, pads to power-of-two dimensions, multiplies, crops. */
function strassenMultiply(a, b, cutoff = 1) {
  const { aRows, aCols, bCols } = assertShapes(a, b);
  const size = nextPowerOfTwo(Math.max(aRows, aCols, bCols, 1));
  const counter = { multiplies: 0, adds: 0, cutoff: Math.max(1, cutoff), levels: 0, baseCalls: 0 };
  if (size === 1) {
    const result = naiveMultiply(a, b, counter);
    return { result, counter };
  }
  const paddedA = pad(a, size, size);
  const paddedB = pad(b, size, size);
  const product = strassen(paddedA, paddedB, counter);
  const result = zeros(aRows, bCols);
  for (let r = 0; r < aRows; r++) for (let c = 0; c < bCols; c++) result[r][c] = product[r][c];
  return { result, counter };
}

/** Closed form for the multiply count of a power-of-two size with a power-of-two cutoff. */
function predictedMultiplies(size, cutoff) {
  const levels = Math.log2(size) - Math.log2(cutoff);
  return 7 ** levels * cutoff ** 3;
}

const transpose = (matrix) => (matrix.length === 0 ? [] : matrix[0].map((_, c) => matrix.map((row) => row[c])));

function randomMatrix(rng, rows, cols, span = 5) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.floor(rng() * (2 * span + 1)) - span));
}

module.exports = {
  makeRng, naiveMultiply, strassenMultiply, strassen, predictedMultiplies, pad, zeros,
  nextPowerOfTwo, transpose, randomMatrix, shape,
};

if (require.main === module) {
  const rng = makeRng(20260727);

  // -- the shapes Strassen has to get right ------------------------------------------------
  const shapes = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [8, 8, 8], [16, 16, 16],
    [3, 5, 7], [7, 3, 2], [1, 9, 1], [6, 6, 6], [13, 13, 13]];
  let compared = 0;
  for (const [rows, inner, cols] of shapes) {
    const a = randomMatrix(rng, rows, inner);
    const b = randomMatrix(rng, inner, cols);
    const expected = naiveMultiply(a, b);
    for (const cutoff of [1, 2, 4, 8, 64]) {
      const { result } = strassenMultiply(a, b, cutoff);
      deepStrictEqual(result, expected,
        `${rows}x${inner}x${cols} with cutoff ${cutoff} must match the naive product`);
      compared++;
    }
  }
  console.log(`strassen vs naive: ${compared} shape/cutoff combinations agree exactly`);

  // -- padding adds zeros only -------------------------------------------------------------
  const odd = randomMatrix(rng, 5, 3);
  const oddB = randomMatrix(rng, 3, 5);
  const padded = pad(odd, 8, 8);
  deepStrictEqual(padded.slice(0, 5).map((row) => row.slice(0, 3)), odd, "the payload lands top-left");
  ok(padded.slice(5).every((row) => row.every((value) => value === 0)), "the extra rows are zeros");
  ok(padded.every((row) => row.slice(3).every((value) => value === 0)), "the extra columns are zeros");
  strictEqual(nextPowerOfTwo(1), 1);
  strictEqual(nextPowerOfTwo(5), 8);
  strictEqual(nextPowerOfTwo(8), 8);
  strictEqual(nextPowerOfTwo(9), 16);
  deepStrictEqual(strassenMultiply(odd, oddB, 2).result, naiveMultiply(odd, oddB),
    "odd shapes must still multiply correctly after padding");

  // -- the multiply count obeys the closed form --------------------------------------------
  for (const size of [2, 4, 8]) {
    for (const cutoff of [1, 2, 4]) {
      if (cutoff > size) continue;
      const { counter } = strassenMultiply(randomMatrix(rng, size, size), randomMatrix(rng, size, size), cutoff);
      strictEqual(counter.multiplies, predictedMultiplies(size, cutoff),
        `${size}x${size} with cutoff ${cutoff} must take 7^(log2(n/c)) * c^3 multiplies`);
    }
  }
  const sevenWay = strassenMultiply(randomMatrix(rng, 8, 8), randomMatrix(rng, 8, 8), 1).counter;
  strictEqual(sevenWay.multiplies, 7 ** 3, "at cutoff 1 an 8x8 product takes 7^3 = 343 multiplies");
  ok(sevenWay.multiplies < 8 ** 3, "which is fewer than the naive 512");
  const truncating = strassenMultiply(randomMatrix(rng, 8, 8), randomMatrix(rng, 8, 8), 8).counter;
  strictEqual(truncating.multiplies, 512, "a cutoff at the full size degenerates to the naive loop");
  strictEqual(truncating.levels, 0, "and never opens a Strassen level");
  strictEqual(truncating.baseCalls, 1, "there is a single base-case call");
  console.log(`8x8: cutoff 1 -> ${sevenWay.multiplies} multiplies, cutoff 8 -> ${truncating.multiplies}`);

  // -- cutoff tuning: deeper recursion cuts multiplies, the add overhead decides the total --
  for (const size of [16, 32, 64]) {
    const a = randomMatrix(rng, size, size, 2);
    const b = randomMatrix(rng, size, size, 2);
    const scan = [];
    for (let cutoff = 1; cutoff <= size; cutoff *= 2) {
      const { counter } = strassenMultiply(a, b, cutoff);
      scan.push({ cutoff, ops: counter.multiplies + counter.adds, ...counter });
    }
    const best = scan.reduce((left, right) => (right.ops < left.ops ? right : left));
    const naive = scan[scan.length - 1];
    ok(scan[0].multiplies < naive.multiplies || size === 1,
      `${size}x${size}: recursing all the way down must cut scalar multiplies`);
    strictEqual(scan[0].multiplies, 7 ** Math.log2(size),
      `${size}x${size} at cutoff 1 takes exactly 7^log2(n) multiplies`);
    if (size >= 32) {
      ok(best.ops < naive.ops, `${size}x${size}: the tuned cutoff must beat the naive loop (${best.ops} < ${naive.ops})`);
      ok(best.cutoff > 1 && best.cutoff < size,
        `${size}x${size}: the optimum must be interior, got ${best.cutoff}`);
    } else {
      strictEqual(best.cutoff, size,
        "at 16x16 the 18 additions per level still outweigh the 7/8 saving, so naive wins on totals");
    }
    console.log(`${size}x${size} ops by cutoff: ${scan.map((s) => `c${s.cutoff}:${s.ops}`).join(" ")}` +
      ` best=c${best.cutoff}`);
  }

  // -- algebraic identities that a wrong quadrant formula would break ----------------------
  const p = randomMatrix(rng, 6, 6, 4);
  const q = randomMatrix(rng, 6, 6, 4);
  const r = randomMatrix(rng, 6, 6, 4);
  const identity = zeros(6, 6);
  for (let i = 0; i < 6; i++) identity[i][i] = 1;
  deepStrictEqual(strassenMultiply(p, identity, 2).result, p, "A * I = A");
  deepStrictEqual(strassenMultiply(identity, p, 2).result, p, "I * A = A");
  const leftAssoc = naiveMultiply(strassenMultiply(p, q, 2).result, r);
  const rightAssoc = naiveMultiply(p, strassenMultiply(q, r, 2).result);
  deepStrictEqual(leftAssoc, rightAssoc, "associativity must hold for the exact integer payload");
  const transposed = naiveMultiply(transpose(q), transpose(p));
  deepStrictEqual(transposed, transpose(naiveMultiply(p, q)), "(AB)^T = B^T A^T");
  deepStrictEqual(strassenMultiply(transpose(q), transpose(p), 4).result, transposed,
    "and Strassen agrees with it");
  const twoByTwo = strassenMultiply([[1, 2], [3, 4]], [[5, 6], [7, 8]], 1).result;
  deepStrictEqual(twoByTwo, [[19, 22], [43, 50]], "a hand-checked 2x2 product");
  const larger = strassenMultiply([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[9, 8, 7], [6, 5, 4], [3, 2, 1]], 1).result;
  deepStrictEqual(larger, [[30, 24, 18], [84, 69, 54], [138, 114, 90]], "a hand-checked 3x3 product");

  // -- degenerate operands and rejecting a bad shape ---------------------------------------
  deepStrictEqual(strassenMultiply([], [], 1).result, [], "an empty product is empty");
  deepStrictEqual(strassenMultiply([[5, 6]], [[1], [2]], 1).result, [[17]], "a 1x2 by 2x1 stay 1x1");
  deepStrictEqual(strassenMultiply([[3]], [[4]], 1).result, [[12]], "1x1 products never open a level");
  strictEqual(strassenMultiply([[3]], [[4]], 1).counter.multiplies, 1);
  throws(() => strassenMultiply([[1, 2]], [[1, 2]], 1), /shape mismatch/);
  throws(() => strassenMultiply([[1, 2], [3]], [[1], [2]], 1), /rectangular/);
  const zeroCounter = strassenMultiply(randomMatrix(rng, 4, 4), randomMatrix(rng, 4, 4), 2).counter;
  ok(zeroCounter.adds > 0, "Strassen must spend additions on the quadrant combinations");

  // -- a larger product beats the naive multiply count -------------------------------------
  const large = strassenMultiply(randomMatrix(rng, 32, 32, 3), randomMatrix(rng, 32, 32, 3), 8);
  const naiveLarge = { multiplies: 0, adds: 0, cutoff: 32 };
  naiveMultiply(randomMatrix(rng, 32, 32, 1), randomMatrix(rng, 32, 32, 1), naiveLarge);
  ok(large.counter.multiplies < naiveLarge.multiplies,
    `32x32: Strassen ${large.counter.multiplies} < naive ${naiveLarge.multiplies} multiplies`);
  ok(predictedMultiplies(32, 8) === 7 ** 2 * 8 ** 3, "closed form sanity: 7^2 * 8^3");

  // -- determinism --------------------------------------------------------------------------
  const replay = () => {
    const local = makeRng(999);
    const a = randomMatrix(local, 16, 16, 3);
    const b = randomMatrix(local, 16, 16, 3);
    const run = strassenMultiply(a, b, 4);
    return `${run.counter.multiplies}/${run.counter.adds}/${JSON.stringify(run.result[0])}`;
  };
  deepStrictEqual(replay(), replay(), "identical inputs must give identical products and counters");
  console.log("PASS");
}
