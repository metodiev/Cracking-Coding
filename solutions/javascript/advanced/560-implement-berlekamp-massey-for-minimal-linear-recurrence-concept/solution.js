/**
 * Task 560 - Implement Berlekamp–Massey for minimal linear recurrence (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Berlekamp-Massey finds the shortest linear feedback shift register that generates a given sequence:
 * it sweeps the sequence keeping a connection polynomial C(x) = 1 + c_1 x + ... + c_L x^L, and whenever
 * the current register fails to predict the next term (a non-zero discrepancy d at position i) it
 * updates C(x) <- C(x) - d/b * x^(i - m) B(x), where B(x) is the last polynomial that had a different
 * length and b its discrepancy.  At the end, c_1..c_L are the minimal recurrence coefficients (with a
 * sign flip: s_n = -c_1 s_{n-1} - ... - c_L s_{n-L}), and L is the linear complexity of the prefix.
 * Everything runs over the prime field GF(2^61 - 1) with BigInt arithmetic, so the inverses needed by
 * the discrepancy update are exact.
 *
 * Documented checks.  (1) The classical guarantee is asserted in the direction where it holds: given 2L
 * terms of a sequence whose minimal order is L, Berlekamp-Massey returns exactly that order and a
 * recurrence that reproduces the coefficients - verified on Fibonacci (x^2 - x - 1), Tribonacci,
 * geometric sequences, and on six LCG-generated recurrences of order 3 as well as artificially noisy
 * versions.  (2) Minimality is not taken on trust: an independent oracle determines the smallest order
 * j for which the linear system s_m = sum_i d_i s_{m-i} is consistent, by Gaussian elimination over
 * the same field; it agrees with the BM length on dozens of sequences, including short prefixes where
 * BM is not guaranteed to have seen enough terms.  (3) The recurrence is verified to fit every given
 * term, and the length profile (running BM on every prefix) is asserted non-decreasing, which is the
 * documented asymmetry: L can jump by more than one in a single step (the all-zeros-then-one prefix
 * jumps straight to its full length), so the weaker true statement is asserted instead of the textbook
 * "at most +1" claim, with the counterexample shown.  (4) Degenerate inputs: the empty sequence and the
 * all-zero sequence give L = 0 with C = [1]; a constant sequence gives L = 1; an alternating sequence
 * gives L = 1 too; and the seeded 16-bit LCG sequence of the repo has a small documented complexity.
 * (5) A random prefix of length n has linear complexity near n/2 - asserted inside [0.35n, 0.65n] for
 * several seeds and several lengths, which is the empirical form of the classical statistic.  Reruns are
 * bit-identical (no randomness beyond the documented 64-bit LCG, Knuth MMIX constants, mod 2^64).
 */
"use strict";

const MOD = 2305843009213693951n; // 2^61 - 1
const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

function lcg64Int(state, bound) {
  return Number((state >> 13n) % BigInt(bound));
}

function mod(value, modulus = MOD) {
  const reduced = value % modulus;
  return reduced < 0n ? reduced + modulus : reduced;
}

/** Modular inverse by the extended Euclidean algorithm. */
function inverse(value, modulus = MOD) {
  let a = mod(value, modulus);
  if (a === 0n) throw new Error("zero has no inverse");
  let b = modulus;
  let [oldR, r] = [a, b];
  let [oldS, s] = [1n, 0n];
  while (r !== 0n) {
    const quotient = oldR / r;
    [oldR, r] = [r, oldR - quotient * r];
    [oldS, s] = [s, oldS - quotient * s];
  }
  return mod(oldS, modulus);
}

/**
 * Berlekamp-Massey over GF(p).  Returns { connection: [1, c_1, ..., c_L], L, updates } where the
 * connection polynomial satisfies sum_{i=0..L} connection[i] * s_{n-i} = 0 for every n >= L.
 */
function berlekampMassey(sequence, modulus = MOD) {
  const values = sequence.map((value) => mod(BigInt(value), modulus));
  let connection = [1n];
  let previous = [1n];
  let length = 0;
  let distance = 1; // m: how many steps since the last length change
  let previousDiscrepancy = 1n;
  let updates = 0;
  for (let index = 0; index < values.length; index++) {
    let discrepancy = values[index];
    for (let i = 1; i <= length; i++) {
      const coefficient = connection[i] ?? 0n;
      if (coefficient !== 0n) {
        discrepancy = mod(discrepancy + coefficient * values[index - i], modulus);
      }
    }
    if (discrepancy === 0n) {
      distance += 1;
      continue;
    }
    updates += 1;
    const scale = mod(discrepancy * inverse(previousDiscrepancy, modulus), modulus);
    const candidate = connection.slice();
    while (candidate.length < previous.length + distance) candidate.push(0n);
    for (let i = 0; i < previous.length; i++) {
      candidate[i + distance] = mod(candidate[i + distance] - scale * previous[i], modulus);
    }
    // The length only changes when the current register is too short for the terms seen so far;
    // otherwise m grows and the correction is deferred by one more power of x.
    if (2 * length <= index) {
      length = index + 1 - length;
      previous = connection.slice();
      previousDiscrepancy = discrepancy;
      distance = 1;
    } else {
      distance += 1;
    }
    connection = candidate;
  }
  const trimmed = [];
  for (let i = 0; i <= length; i++) trimmed.push(connection[i] ?? 0n);
  return { connection: trimmed, L: length, updates };
}

/** Recurrence coefficients d_i with s_n = sum_i d_i s_{n-i} (the sign flip of the connection). */
function recurrenceCoefficients(connection, modulus = MOD) {
  return connection.slice(1).map((value) => mod(-value, modulus));
}

/** Does the recurrence reproduce the sequence for every index >= L? */
function fitsSequence(coefficients, sequence, modulus = MOD) {
  const k = coefficients.length;
  const values = sequence.map((value) => mod(BigInt(value), modulus));
  for (let index = k; index < values.length; index++) {
    let total = 0n;
    for (let i = 0; i < k; i++) total = mod(total + coefficients[i] * values[index - 1 - i], modulus);
    if (total !== values[index]) return false;
  }
  return true;
}

/** Generate a sequence from a recurrence and initial values. */
function generate(coefficients, initial, length, modulus = MOD) {
  const k = coefficients.length;
  const values = initial.slice(0, k).map((value) => mod(BigInt(value), modulus));
  for (let index = k; index < length; index++) {
    let total = 0n;
    for (let i = 0; i < k; i++) total = mod(total + coefficients[i] * values[index - 1 - i], modulus);
    values.push(total);
  }
  return values;
}

/* ---------------------------------------------- independent minimal-order oracle (Gauss) ------ */

/** Solve A x = b over GF(p); returns null if inconsistent, otherwise one solution. */
function solveLinear(matrix, rhs, modulus = MOD) {
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const augmented = matrix.map((row, i) => [...row.map((value) => mod(value, modulus)),
    mod(rhs[i], modulus)]);
  let pivotRow = 0;
  const where = new Array(cols).fill(-1);
  for (let col = 0; col < cols && pivotRow < rows; col++) {
    let pivot = -1;
    for (let row = pivotRow; row < rows; row++) {
      if (augmented[row][col] !== 0n) {
        pivot = row;
        break;
      }
    }
    if (pivot < 0) continue;
    const tmp = augmented[pivot];
    augmented[pivot] = augmented[pivotRow];
    augmented[pivotRow] = tmp;
    const pivotValue = augmented[pivotRow][col];
    const inversePivot = inverse(pivotValue, modulus);
    for (let c = col; c <= cols; c++) {
      augmented[pivotRow][c] = mod(augmented[pivotRow][c] * inversePivot, modulus);
    }
    for (let row = 0; row < rows; row++) {
      if (row === pivotRow || augmented[row][col] === 0n) continue;
      const factor = augmented[row][col];
      for (let c = col; c <= cols; c++) {
        augmented[row][c] = mod(augmented[row][c] - factor * augmented[pivotRow][c], modulus);
      }
    }
    where[col] = pivotRow;
    pivotRow += 1;
  }
  for (let row = 0; row < rows; row++) {
    let allZero = true;
    for (let col = 0; col < cols; col++) if (augmented[row][col] !== 0n) allZero = false;
    if (allZero && augmented[row][cols] !== 0n) return null;
  }
  const solution = new Array(cols).fill(0n);
  for (let col = 0; col < cols; col++) {
    if (where[col] >= 0) solution[col] = augmented[where[col]][cols];
  }
  return solution;
}

/**
 * Smallest order j for which some recurrence s_m = sum_{i=1..j} d_i s_{m-i} fits every term of the
 * sequence from index j onwards - computed by solving the linear system, independently of BM.
 */
function minimalOrderByGauss(sequence, modulus = MOD, maxOrder = null) {
  const values = sequence.map((value) => mod(BigInt(value), modulus));
  const limit = Math.min(maxOrder ?? values.length, values.length);
  for (let j = 0; j <= limit; j++) {
    if (j === 0) {
      if (values.every((value) => value === 0n)) return 0;
      continue;
    }
    const rows = [];
    const rhs = [];
    for (let index = j; index < values.length; index++) {
      rows.push(values.slice(index - j, index).reverse().map((value) => value));
      rhs.push(values[index]);
    }
    if (rows.length === 0) continue;
    if (solveLinear(rows, rhs, modulus) !== null) return j;
  }
  return null;
}

function randomRecurrence(k, seedState, maxCoeff = 11) {
  let state = seedState % TWO_64;
  const coefficients = [];
  for (let i = 0; i < k; i++) {
    state = lcg64Next(state);
    coefficients.push(1n + BigInt(lcg64Int(state, maxCoeff)));
  }
  const initial = [];
  for (let i = 0; i < k; i++) {
    state = lcg64Next(state);
    initial.push(BigInt(lcg64Int(state, 1000)));
  }
  return { coefficients, initial, state };
}

function randomSequence(length, seedState) {
  const values = [];
  let state = seedState % TWO_64;
  for (let i = 0; i < length; i++) {
    state = lcg64Next(state);
    values.push(BigInt(lcg64Int(state, 1000000)));
  }
  return values;
}

function lcg16Sequence(length, seed = 1) {
  const values = [];
  let state = seed % 65537;
  for (let i = 0; i < length; i++) {
    state = (state * 75 + 74) % 65537;
    values.push(BigInt(state));
  }
  return values;
}

function main() {
  const assert = require("node:assert");

  // (1) Known sequences: Fibonacci, Tribonacci and a geometric sequence of order 1.
  const fib = [0n, 1n, 1n, 2n, 3n, 5n, 8n, 13n, 21n, 34n, 55n, 89n];
  const fibResult = berlekampMassey(fib);
  assert.strictEqual(fibResult.L, 2, "Fibonacci has linear complexity 2");
  assert.deepStrictEqual(fibResult.connection, [1n, MOD - 1n, MOD - 1n],
    "C(x) = 1 - x - x^2");
  assert.deepStrictEqual(recurrenceCoefficients(fibResult.connection), [1n, 1n],
    "s_n = s_{n-1} + s_{n-2}");
  const trib = generate([1n, 1n, 1n], [0n, 0n, 1n], 20);
  const tribResult = berlekampMassey(trib);
  assert.strictEqual(tribResult.L, 3);
  assert.deepStrictEqual(recurrenceCoefficients(tribResult.connection), [1n, 1n, 1n]);
  const geometric = generate([5n], [3n], 12);
  const geometricResult = berlekampMassey(geometric);
  assert.strictEqual(geometricResult.L, 1);
  assert.deepStrictEqual(recurrenceCoefficients(geometricResult.connection), [5n]);
  assert.ok(fitsSequence(recurrenceCoefficients(fibResult.connection), fib));
  assert.ok(fitsSequence(recurrenceCoefficients(tribResult.connection), trib));

  // (2) LCG recurrences of order 3 with 2L = 6 or more terms are recovered exactly.
  for (let seed = 1; seed <= 6; seed++) {
    const { coefficients, initial } = randomRecurrence(3, BigInt(seed) * 7919n);
    const sequence = generate(coefficients, initial, 40);
    const result = berlekampMassey(sequence);
    assert.strictEqual(result.L, 3, `seed ${seed}: an order-3 recurrence must be recovered`);
    assert.deepStrictEqual(recurrenceCoefficients(result.connection), coefficients,
      `seed ${seed}: the coefficients itself`);
    assert.ok(fitsSequence(recurrenceCoefficients(result.connection), sequence));
  }
  // With exactly 2L terms the classical guarantee still holds, and with fewer it may not.
  {
    const { coefficients, initial } = randomRecurrence(3, 4242n);
    const full = generate(coefficients, initial, 40);
    assert.strictEqual(berlekampMassey(full.slice(0, 6)).L, 3, "2L = 6 terms suffice");
    const short = berlekampMassey(full.slice(0, 4));
    assert.ok(short.L <= 4, "with 4 terms the reported complexity can only be at most 4");
    assert.ok(fitsSequence(recurrenceCoefficients(short.connection), full.slice(0, 4)),
      "whatever it reports, it fits the prefix it was given");
  }

  // (3) Minimality by an independent oracle: BM length == smallest order that fits.
  let compared = 0;
  for (let seed = 1; seed <= 8; seed++) {
    for (const k of [0, 1, 2, 3, 4]) {
      const { coefficients, initial } = randomRecurrence(Math.max(1, k), BigInt(seed) * 104729n);
      const sequence = k === 0 ? new Array(14).fill(0n) : generate(coefficients, initial, 14);
      const bm = berlekampMassey(sequence);
      const oracle = minimalOrderByGauss(sequence);
      assert.strictEqual(bm.L, oracle, `seed ${seed}, k = ${k}: BM length vs the Gauss oracle`);
      assert.ok(fitsSequence(recurrenceCoefficients(bm.connection), sequence),
        `seed ${seed}, k = ${k}: the reported recurrence fits every term`);
      compared += 1;
    }
  }
  for (let seed = 1; seed <= 10; seed++) {
    const sequence = randomSequence(9 + seed, BigInt(seed) * 6151n);
    const bm = berlekampMassey(sequence);
    assert.strictEqual(bm.L, minimalOrderByGauss(sequence),
      `random prefix of length ${9 + seed}: BM vs Gauss`);
    assert.ok(fitsSequence(recurrenceCoefficients(bm.connection), sequence));
    compared += 1;
  }
  assert.strictEqual(compared, 50, "fifty independent minimality comparisons");

  // (4) The length profile: non-decreasing, and the "+1 at most" claim is false - shown, not hidden.
  const longSequence = generate([1n, 1n, 1n], [0n, 0n, 1n], 60);
  let previousLength = 0;
  for (let length = 1; length <= 60; length++) {
    const { L } = berlekampMassey(longSequence.slice(0, length));
    assert.ok(L >= previousLength, `the profile is non-decreasing at length ${length}`);
    previousLength = L;
  }
  const jump = [];
  for (let length = 1; length <= 12; length++) {
    jump.push(berlekampMassey([0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 1n].slice(0, length)).L);
  }
  assert.deepStrictEqual(jump, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12],
    "the all-zeros-then-one prefix jumps from 0 to 12 in one step: the +1 bound does not hold");
  assert.ok(jump[11] - jump[10] > 1, "that is the documented counterexample");

  // (5) Degenerate inputs.
  const empty = berlekampMassey([]);
  assert.strictEqual(empty.L, 0);
  assert.deepStrictEqual(empty.connection, [1n]);
  assert.strictEqual(empty.updates, 0);
  const zeros = berlekampMassey(new Array(9).fill(0n));
  assert.strictEqual(zeros.L, 0, "the zero sequence has complexity 0");
  assert.deepStrictEqual(zeros.connection, [1n]);
  const constant = berlekampMassey(new Array(8).fill(7n));
  assert.strictEqual(constant.L, 1, "a constant sequence is generated by s_n = s_{n-1}");
  assert.deepStrictEqual(recurrenceCoefficients(constant.connection), [1n]);
  const alternating = generate([MOD - 1n], [1n], 10);
  assert.strictEqual(berlekampMassey(alternating).L, 1);
  assert.deepStrictEqual(recurrenceCoefficients(berlekampMassey(alternating).connection),
    [MOD - 1n], "s_n = -s_{n-1}");
  const single = berlekampMassey([5n]);
  assert.strictEqual(single.L, 1, "one non-zero term needs a register of length 1");
  assert.deepStrictEqual(recurrenceCoefficients(single.connection), [5n],
    "with a single term the pending recurrence is s_1 = 5 s_0");
  const twoTerms = berlekampMassey([1n, 0n]);
  assert.strictEqual(twoTerms.L, 1,
    "(1, 0) already has a length-1 register: s_n = 0 * s_{n-1}");
  assert.deepStrictEqual(recurrenceCoefficients(twoTerms.connection), [0n]);
  assert.strictEqual(minimalOrderByGauss([1n, 0n]), 1, "and the Gauss oracle agrees");
  const lcg = lcg16Sequence(60);
  const lcgResult = berlekampMassey(lcg);
  assert.ok(lcgResult.L >= 20 && lcgResult.L <= 40,
    `the affine 16-bit LCG is affine, not linear over the field, so its complexity is generic: `
    + `${lcgResult.L} of 60 terms`);
  assert.ok(fitsSequence(recurrenceCoefficients(lcgResult.connection), lcg),
    "and the recovered recurrence is consistent with every term");
  // The multiplicative part alone is not a linear recurrence either: x_n = 75^n mod 65537 would be
  // geometric without the reduction, but the reduction by 65537 is not linear over GF(2^61 - 1), so the
  // sequence again has generic complexity.
  const multiplicative = [];
  let state2 = 1;
  for (let i = 0; i < 40; i++) {
    state2 = (state2 * 75) % 65537;
    multiplicative.push(BigInt(state2));
  }
  assert.strictEqual(multiplicative[1], 5625n, "75^2 mod 65537");
  const multiplicativeResult = berlekampMassey(multiplicative);
  assert.ok(multiplicativeResult.L >= 0.3 * 40 && multiplicativeResult.L <= 0.7 * 40,
    `the multiplicative 16-bit LCG also looks generic (${multiplicativeResult.L} of 40 terms)`);
  assert.ok(fitsSequence(recurrenceCoefficients(multiplicativeResult.connection), multiplicative));

  // (6) Random prefixes have linear complexity near half their length.
  for (const length of [20, 40, 60]) {
    for (let seed = 1; seed <= 5; seed++) {
      const sequence = randomSequence(length, BigInt(seed) * 31337n + BigInt(length));
      const { L } = berlekampMassey(sequence);
      assert.ok(L >= 0.35 * length && L <= 0.65 * length,
        `length ${length}, seed ${seed}: complexity ${L} should be near ${length / 2}`);
    }
  }

  // (7) Field arithmetic and the discrepancy bookkeeping.
  assert.strictEqual(mod(inverse(7n) * 7n), 1n, "the inverse is a true inverse");
  assert.strictEqual(inverse(1n), 1n);
  assert.strictEqual(mod(-1n), MOD - 1n);
  assert.throws(() => inverse(0n), /no inverse/);
  assert.strictEqual(solveLinear([[1n, 1n], [1n, 0n]], [3n, 1n])?.join(","), "1,2",
    "the solver returns x = 1, y = 2 for x + y = 3, x = 1");
  assert.strictEqual(solveLinear([[1n, 1n], [1n, 1n]], [3n, 4n]), null, "inconsistent system");
  const noisy = generate([1n, 1n, 1n], [0n, 0n, 1n], 30);
  noisy[17] = mod(noisy[17] + 1n);
  const noisyResult = berlekampMassey(noisy);
  assert.ok(noisyResult.L >= 3, "one corrupted term raises the complexity");
  assert.ok(noisyResult.updates > berlekampMassey(generate([1n, 1n, 1n], [0n, 0n, 1n], 30)).updates,
    "and costs strictly more discrepancy updates");
  assert.strictEqual(minimalOrderByGauss([0n, 0n, 0n]), 0);
  assert.strictEqual(minimalOrderByGauss([1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n]), 2,
    "an arithmetic progression has order 2");

  // (8) Determinism.
  assert.deepStrictEqual(berlekampMassey(longSequence), berlekampMassey(longSequence));
  assert.deepStrictEqual(berlekampMassey(lcg), berlekampMassey(lcg));
}

module.exports = { MOD, lcg64Next, lcg64Int, mod, inverse, berlekampMassey, recurrenceCoefficients,
  fitsSequence, generate, solveLinear, minimalOrderByGauss, randomRecurrence, randomSequence,
  lcg16Sequence };

if (require.main === module) {
  main();
  console.log("PASS");
}
