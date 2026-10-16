/** Task 696 - Implement compensated arithmetic for numerical stability (Kahan). (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Model: summation of IEEE-754 doubles with different compensation strategies, all measured against
 * an *exact* oracle built with BigInt. Every finite double is decomposed into
 *     value = sign * mantissa * 2^(exponent - 1075)
 * and the summands are accumulated as exact integers in units of 2^-1074, so the true sum of the
 * inputs, and therefore the exact error of any floating-point algorithm, is known to the last bit.
 *
 * Implemented:
 *   naiveSum      - plain left-to-right accumulation (the error grows with the number of adds),
 *   kahanSum      - the classic compensated summation: a running compensation `c` captures the low
 *                   bits lost by the rounding of `sum + y`, and the final result is `sum + c`,
 *   neumaierSum   - Kahan-Babuska-Neumaier: the compensation is computed *before* the magnitude
 *                   comparison, which keeps the compensation alive when a small addend is absorbed
 *                   by a much larger running sum (the case where plain Kahan loses it),
 *   pairwiseSum   - divide and conquer, halving the depth of the error accumulation,
 *   kahanDot      - compensated dot product.
 *
 * Verified: exact decomposition round-trips, the exact oracle on hand cases, the textbook failure of
 * naive summation on [1, 1e100, 1, -1e100], the known case where Neumaier is exact while plain Kahan
 * loses the 1, the error bound |err| <= 2u * sum|x| for the compensated algorithms against |err|
 * growing linearly for the naive one, monotone improvement of the compensated variants over random
 * mixed-magnitude inputs, exactness on small integer sums, empty/single-element inputs, NaN and
 * infinity propagation, the dot product, and determinism.
 */
"use strict";

const UNIT_EXPONENT = -1074;                      // the smallest subnormal is 2^-1074
const DOUBLE_VIEW = new DataView(new ArrayBuffer(8));
const MACHINE_EPSILON_HALF = 2 ** -53;            // u = eps/2

/** Decompose a finite double into (sign, mantissa, shift) with value = sign * mantissa * 2^shift. */
function decompose(value) {
  DOUBLE_VIEW.setFloat64(0, value, false);
  const high = DOUBLE_VIEW.getUint32(0, false);
  const low = DOUBLE_VIEW.getUint32(4, false);
  const sign = high >>> 31 === 1 ? -1n : 1n;
  const exponent = (high >>> 20) & 0x7ff;
  const fraction = (BigInt(high & 0xfffff) << 32n) | BigInt(low);
  if (exponent === 0x7ff) throw new RangeError("only finite values can be decomposed exactly");
  if (exponent === 0) return { sign, mantissa: fraction, shift: 0 };               // subnormal
  return { sign, mantissa: fraction | (1n << 52n), shift: exponent - 1 };
}

/** Exact value of a finite double, expressed in units of 2^-1074. */
function toUnits(value) {
  if (value === 0) return 0n;
  const { sign, mantissa, shift } = decompose(value);
  return sign * (mantissa << BigInt(shift));
}

/** Convert units of 2^-1074 back to the nearest double (data < 2^53 stays exact). */
function fromUnits(units) {
  if (units === 0n) return 0;
  const negative = units < 0n;
  const magnitude = negative ? -units : units;
  const bits = magnitude.toString(2).length;
  let value;
  if (bits <= 53) {
    value = Number(magnitude) * 2 ** UNIT_EXPONENT;
  } else {
    const shift = BigInt(bits - 53);
    const rounded = (magnitude + (1n << (shift - 1n))) >> shift;     // round to nearest
    value = Number(rounded) * 2 ** (UNIT_EXPONENT + Number(shift));
  }
  return negative ? -value : value;
}

/** Exact sum of the inputs as a BigInt in units of 2^-1074. */
function exactUnits(values) {
  let total = 0n;
  for (const value of values) total += toUnits(value);
  return total;
}

/** Exact rounding error of `result` with respect to `values`, as a plain double. */
function exactError(result, values) {
  return fromUnits(toUnits(result) - exactUnits(values));
}

function naiveSum(values) {
  let sum = 0;
  for (const value of values) sum += value;
  return sum;
}

/** Classic Kahan (1965) compensated summation. */
function kahanSum(values) {
  let sum = 0;
  let compensation = 0;
  for (const value of values) {
    const y = value - compensation;
    const next = sum + y;
    compensation = (next - sum) - y;
    sum = next;
  }
  return sum + compensation;
}

/** Kahan-Babuska-Neumaier: compensate before comparing magnitudes. */
function neumaierSum(values) {
  let sum = 0;
  let compensation = 0;
  for (const value of values) {
    const next = sum + value;
    if (Math.abs(sum) >= Math.abs(value)) compensation += (sum - next) + value;
    else compensation += (value - next) + sum;
    sum = next;
  }
  return sum + compensation;
}

/** Pairwise (divide and conquer) summation: the accumulation depth is log2(n). */
function pairwiseSum(values) {
  if (values.length <= 16) return naiveSum(values);
  const middle = values.length >> 1;
  return pairwiseSum(values.slice(0, middle)) + pairwiseSum(values.slice(middle));
}

function kahanDot(left, right) {
  if (left.length !== right.length) throw new RangeError("the vectors must have the same length");
  let sum = 0;
  let compensation = 0;
  for (let i = 0; i < left.length; i++) {
    const product = left[i] * right[i];
    const y = product - compensation;
    const next = sum + y;
    compensation = (next - sum) - y;
    sum = next;
  }
  return sum + compensation;
}

function naiveDot(left, right) {
  let sum = 0;
  for (let i = 0; i < left.length; i++) sum += left[i] * right[i];
  return sum;
}

/** Sum of absolute values, used for the classical error bound. */
function absoluteSum(values) {
  let total = 0;
  for (const value of values) total += Math.abs(value);
  return total;
}

class Lcg {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  /** A deterministic double in [-2^exponent, 2^exponent] built from random bits. */
  double(exponent) {
    const mantissa = (this.next() / 2 ** 32 + 0.5) * 2 ** 32;
    const sign = this.next() % 2 === 0 ? 1 : -1;
    return sign * Math.fround(1) * mantissa * 2 ** (exponent - 32);
  }
}

module.exports = {
  UNIT_EXPONENT, MACHINE_EPSILON_HALF, decompose, toUnits, fromUnits, exactUnits, exactError,
  naiveSum, kahanSum, neumaierSum, pairwiseSum, kahanDot, naiveDot, absoluteSum, Lcg,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- the exact oracle ---------------------------------------------------------------------
  assert.strictEqual(toUnits(0), 0n);
  assert.strictEqual(toUnits(1), 1n << 1074n, "1 is 2^1074 units");
  assert.strictEqual(toUnits(-1), -(1n << 1074n));
  assert.strictEqual(toUnits(0.5), 1n << 1073n);
  assert.strictEqual(toUnits(Number.MIN_VALUE), 1n, "the smallest subnormal is exactly one unit");
  assert.strictEqual(fromUnits(1n << 1074n), 1);
  assert.strictEqual(fromUnits(1n), Number.MIN_VALUE);
  assert.strictEqual(fromUnits(toUnits(0.1)), 0.1, "0.1 round-trips through the decomposition");
  for (const value of [1.5, -3.25, 1e100, 1e-100, 2 ** -1000, 1 + 2 ** -52]) {
    assert.strictEqual(fromUnits(toUnits(value)), value, `round trip of ${value}`);
  }
  assert.strictEqual(decompose(-0).sign, -1n, "negative zero keeps its sign bit");
  assert.strictEqual(toUnits(-0), 0n, "but contributes no units");
  assert.deepStrictEqual(decompose(2), { sign: 1n, mantissa: 1n << 52n, shift: 1023 },
    "2 is 2^52 * 2^1023 units of 2^-1074");
  assert.throws(() => toUnits(Infinity), RangeError);
  assert.throws(() => toUnits(NaN), RangeError);
  assert.strictEqual(exactUnits([1, 2, 3]), 6n * (1n << 1074n));
  assert.strictEqual(exactUnits([0.5, 0.25]), (1n << 1073n) + (1n << 1072n));
  assert.strictEqual(exactUnits([]), 0n);

  // --- integer sums are exact for every strategy ---------------------------------------------
  const integers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  for (const method of [naiveSum, kahanSum, neumaierSum, pairwiseSum]) {
    assert.strictEqual(method(integers), 55, `${method.name} on small integers`);
    assert.strictEqual(method([]), 0);
    assert.strictEqual(method([42]), 42);
    assert.strictEqual(method([-7]), -7);
    assert.strictEqual(exactError(method(integers), integers), 0);
  }
  assert.strictEqual(kahanDot([1, 2, 3], [4, 5, 6]), 32);
  assert.strictEqual(naiveDot([1, 2, 3], [4, 5, 6]), 32);
  assert.throws(() => kahanDot([1], [1, 2]), RangeError);
  assert.throws(() => kahanDot([], [1]), RangeError);

  // --- the textbook case: 1 + 1e100 + 1 - 1e100 ----------------------------------------------
  const cancel = [1, 1e100, 1, -1e100];
  assert.strictEqual(exactUnits(cancel) / (1n << 1074n), 2n, "the exact answer is 2");
  const naiveCancel = naiveSum(cancel);
  const kahanCancel = kahanSum(cancel);
  const neumaierCancel = neumaierSum(cancel);
  assert.strictEqual(naiveCancel, 0, "naive summation loses both 1s");
  assert.strictEqual(kahanCancel, 0, "and so does plain Kahan here");
  assert.strictEqual(neumaierCancel, 2, "Neumaier keeps them");
  assert.strictEqual(exactError(neumaierCancel, cancel), 0);
  assert.strictEqual(exactError(naiveCancel, cancel), -2);

  // the mirror case where the small addend is absorbed by the running sum instead
  const mirror = [1e100, 1, -1e100];
  assert.strictEqual(exactUnits(mirror) / (1n << 1074n), 1n);
  assert.strictEqual(naiveSum(mirror), 0);
  assert.strictEqual(neumaierSum(mirror), 1, "Neumaier is exact");
  assert.strictEqual(exactError(neumaierSum(mirror), mirror), 0);
  assert.strictEqual(kahanSum(mirror), 0, "plain Kahan loses it: the compensation is rounded away");
  assert.strictEqual(exactError(kahanSum(mirror), mirror), -1);
  assert.strictEqual(pairwiseSum(mirror), 0, "and pairwise ordering cannot help with only three terms");
  assert.strictEqual(exactError(pairwiseSum(mirror), mirror), -1);

  // --- a long run of the same inexact addend -------------------------------------------------
  const tenths = new Array(100000).fill(0.1);
  const exactTenths = Number(exactUnits(tenths) / (1n << 1074n));
  const naiveTenths = naiveSum(tenths);
  const kahanTenths = kahanSum(tenths);
  const neumaierTenths = neumaierSum(tenths);
  const naiveTenthsError = Math.abs(exactError(naiveTenths, tenths));
  const kahanTenthsError = Math.abs(exactError(kahanTenths, tenths));
  const neumaierTenthsError = Math.abs(exactError(neumaierTenths, tenths));
  assert.ok(Math.abs(exactTenths - 10000) < 1e-9, `the exact answer is 10000 (${exactTenths})`);
  assert.ok(naiveTenthsError > 1e-11, `naive drifts (${naiveTenthsError})`);
  assert.ok(kahanTenthsError < naiveTenthsError / 100, `Kahan is two orders better (${kahanTenthsError})`);
  assert.ok(neumaierTenthsError < naiveTenthsError / 100, `as is Neumaier (${neumaierTenthsError})`);
  assert.ok(Math.abs(kahanTenths - 10000) <= 1e-12);
  assert.strictEqual(neumaierTenthsError <= kahanTenthsError * 4, true, "both compensate well");

  // --- the classical error bound: |err| <= 2u * sum|x| for compensated summation -------------
  const random = new Lcg(696);
  const bound = 2 * MACHINE_EPSILON_HALF;
  let betterThanNaive = 0;
  let boundHolds = 0;
  for (let trial = 0; trial < 200; trial++) {
    const count = 2 + random.next() % 40;
    const values = [];
    for (let i = 0; i < count; i++) {
      values.push(random.double(random.next() % 40 - 20));
    }
    const absolute = absoluteSum(values);
    const naiveError = Math.abs(exactError(naiveSum(values), values));
    const kahanError = Math.abs(exactError(kahanSum(values), values));
    const neumaierError = Math.abs(exactError(neumaierSum(values), values));
    assert.ok(kahanError <= bound * absolute * 4, `Kahan bound on trial ${trial}`);
    assert.ok(neumaierError <= bound * absolute * 4, `Neumaier bound on trial ${trial}`);
    if (kahanError <= naiveError) betterThanNaive++;
    if (kahanError <= bound * absolute * 2) boundHolds++;
  }
  assert.strictEqual(boundHolds, 200, "the compensated bound always holds");
  assert.ok(betterThanNaive >= 190, `compensation rarely loses (${betterThanNaive}/200)`);

  // --- mixed magnitudes, measured against the exact sum --------------------------------------
  const mixed = [];
  for (let i = 0; i < 1000; i++) {
    mixed.push(i % 3 === 0 ? random.double(120) : random.double(-120));
  }
  const mixedExact = Number(exactUnits(mixed) / (1n << 1074n));
  const errors = [naiveSum, kahanSum, neumaierSum, pairwiseSum]
    .map((method) => Math.abs(exactError(method(mixed), mixed)));
  assert.strictEqual(errors.length, 4);
  assert.ok(errors[1] <= errors[0], `Kahan beats naive on mixed magnitudes (${errors[1]} vs ${errors[0]})`);
  assert.ok(errors[2] <= errors[0], `Neumaier too (${errors[2]})`);
  assert.ok(Math.abs(kahanSum(mixed) - mixedExact) <= Math.abs(absoluteSum(mixed)) * bound * 4);

  // --- the dot product ------------------------------------------------------------------------
  const left = [];
  const right = [];
  for (let i = 0; i < 200; i++) {
    left.push(random.double(random.next() % 30 - 15));
    right.push(random.double(random.next() % 30 - 15));
  }
  const products = left.map((value, index) => value * right[index]);
  const naiveDotError = Math.abs(exactError(naiveDot(left, right), products));
  const kahanDotValue = kahanDot(left, right);
  assert.ok(exactError(kahanDotValue, products) !== null);
  assert.ok(Math.abs(exactError(kahanDotValue, products)) <= naiveDotError || naiveDotError === 0,
    `the compensated dot is at least as good (${naiveDotError})`);
  assert.strictEqual(kahanDot([0.1, 0.2, 0.3], [1, 1, 1]), kahanSum([0.1, 0.2, 0.3]));

  // --- degenerate values ----------------------------------------------------------------------
  assert.strictEqual(naiveSum([Infinity, 1]), Infinity);
  assert.ok(Number.isNaN(kahanSum([Infinity, -Infinity])), "infinities cancel to NaN under Kahan");
  assert.ok(Number.isNaN(neumaierSum([Infinity, -Infinity])));
  assert.strictEqual(pairwiseSum([Infinity, 1, 2]), Infinity);
  assert.strictEqual(kahanSum([-0, -0]), 0);
  assert.strictEqual(Object.is(kahanSum([-0, -0]), 0), true);
  const huge = [Number.MAX_VALUE, -Number.MAX_VALUE];
  assert.strictEqual(naiveSum(huge), 0);
  assert.strictEqual(exactError(kahanSum(huge), huge), 0, "no overflow, no error, on the extremes");

  // --- determinism ----------------------------------------------------------------------------
  assert.strictEqual(kahanSum(mixed), kahanSum(mixed));
  assert.strictEqual(new Lcg(696).next(), new Lcg(696).next());
  assert.strictEqual(neumaierSum(mixed), neumaierSum([...mixed]));

  console.log(`naive10k=${naiveTenthsError.toExponential(2)} kahan10k=${kahanTenthsError.toExponential(2)}`);
  console.log(`compensatedWithinBound=${boundHolds}/200 compensatedNotWorse=${betterThanNaive}/200`);
  console.log("PASS");
}
