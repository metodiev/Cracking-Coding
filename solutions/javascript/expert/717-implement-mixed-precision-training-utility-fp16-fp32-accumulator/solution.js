/** Task 717 - Implement mixed-precision training utility (FP16/FP32 accumulator). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Mixed-precision training keeps the model and the updates in FP32 while running the multiplications
 * in FP16. That works only because of two tricks, both implemented and measured here:
 *   - an FP32 accumulator: FP16 has 11 bits of significand, so once a sum reaches 1.0 an increment of
 *     2^-11 disappears. Accumulating in FP32 (or in an FP32 master copy) keeps the low bits;
 *   - loss scaling: gradients are multiplied by 2^s before being cast to FP16, so values smaller
 *     than the smallest subnormal 2^-24 survive the cast, and the update is divided by 2^s again in
 *     FP32. An overflow (the scaled gradient becomes inf) aborts the step and halves the scale.
 *
 * The file implements FP16 by hand: the encoder rounds the 52-bit significand of a double to the
 * 10-bit significand of a half with round-half-to-even, handles subnormals, overflow to infinity,
 * NaN and signed zero, and decodes bit patterns back to doubles. Everything is then checked against
 * the properties an FP16 unit has to satisfy: known encodings, the half-ulp round-trip bound,
 * exact ties-to-even decisions, and the failure modes that loss scaling exists to prevent.
 *
 * What is simplified: only the cast, the scaler and an accumulation loop are simulated - there are no
 * tensor cores, no master-weight optimiser state beyond plain SGD, and no per-op statistics.
 *
 * Invariant verified: the encoder/decoder pair reproduces the documented bit patterns, every normal
 * round trip stays inside half an ulp, ties resolve to the even significand, values above 65519
 * become infinity and values below 2^-25 flush to zero, FP16 accumulation is shown to lose small
 * increments while the FP32 accumulator keeps them, and loss scaling rescues a gradient the plain
 * FP16 path flushes to zero.
 */
"use strict";
const { ok, strictEqual, deepStrictEqual, throws } = require("node:assert");

const HALF_INF = 0x7c00;
const HALF_MANTISSA = 0x03ff;

/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Rounds a double to the nearest half, ties to even, and returns the 16-bit pattern. */
function toFloat16Bits(value) {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value);
  const bits = view.getBigUint64(0);
  const sign = Number(bits >> 63n) & 1;
  const signed = (payload) => (sign << 15) | payload;
  const exponent = Number((bits >> 52n) & 0x7ffn);
  const mantissa = bits & ((1n << 52n) - 1n);
  if (exponent === 0x7ff) return signed(HALF_INF | (mantissa === 0n ? 0 : 0x200)); // inf / quiet NaN
  const unbiased = exponent - 1023;
  if (exponent === 0 && mantissa === 0n) return signed(0); // signed zero
  if (unbiased > 15) return signed(HALF_INF); // overflow
  if (unbiased < -60) return signed(0); // far below the smallest subnormal
  if (unbiased < -14) {
    // subnormal: the significand shrinks and the exponent field stays zero
    const shift = BigInt(28 - unbiased);
    const full = (1n << 52n) | mantissa;
    const scaled = full >> shift;
    const remainder = full & ((1n << shift) - 1n);
    const halfway = 1n << (shift - 1n);
    let rounded = scaled;
    if (remainder > halfway || (remainder === halfway && (scaled & 1n) === 1n)) rounded += 1n;
    return signed(Number(rounded));
  }
  const kept = mantissa >> 42n;
  const remainder = mantissa & ((1n << 42n) - 1n);
  const halfway = 1n << 41n;
  let significand = kept;
  if (remainder > halfway || (remainder === halfway && (kept & 1n) === 1n)) significand += 1n;
  let exponentBits = BigInt(unbiased + 15);
  if (significand === 1024n) { significand = 0n; exponentBits += 1n; } // carry into the exponent
  if (exponentBits >= 31n) return signed(HALF_INF);
  return signed((Number(exponentBits) << 10) | Number(significand));
}

/** Decodes a half pattern back to a double (exact, every half is a double). */
function fromFloat16Bits(bits) {
  if (!Number.isInteger(bits) || bits < 0 || bits > 0xffff) throw new RangeError("not a 16-bit pattern");
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & HALF_MANTISSA;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  if (exponent === 0) return sign * mantissa * 2 ** -24; // subnormal (or zero)
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** The value an FP16 pipeline actually keeps after casting. */
function toFloat16(value) { return fromFloat16Bits(toFloat16Bits(value)); }

/** A dot product with the products accumulated in FP16 (the naive, lossy variant). */
function accumulateFloat16(values) {
  let total = 0;
  for (const value of values) total = toFloat16(toFloat16(total) + toFloat16(value));
  return total;
}

/** The same dot product with an FP32 accumulator. */
function accumulateFloat32(values) {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

/** Result of a scaled FP16 cast: either the cast value or an overflow marker. */
function scaledCast(gradient, scale) {
  const scaled = toFloat16(gradient * scale);
  if (!Number.isFinite(scaled)) return { overflow: true, value: null, scaled };
  return { overflow: false, value: scaled / scale, scaled };
}

/**
 * One mixed-precision SGD step. `weights` is the FP32 master copy, `gradients` the FP32 gradients;
 * the casts happen in FP16 and the subtraction stays in FP32.
 */
function masterWeightStep(weights, gradients, { learningRate, scale }) {
  const scaled = gradients.map((gradient) => scaledCast(gradient, scale));
  if (scaled.some((entry) => entry.overflow)) {
    return { weights: weights.slice(), scale: scale / 2, skipped: true, gradients: gradients.map(() => 0) };
  }
  const applied = scaled.map((entry) => entry.value);
  return {
    weights: weights.map((weight, i) => weight - learningRate * applied[i]),
    scale,
    skipped: false,
    gradients: applied,
  };
}

/** Trains with plain FP16 weights and gradients: the variant mixed precision exists to avoid. */
function float16Step(weights, gradients, learningRate) {
  return weights.map((weight, i) => toFloat16(toFloat16(weight) - toFloat16(toFloat16(learningRate) * toFloat16(gradients[i]))));
}

module.exports = {
  makeRng, toFloat16Bits, fromFloat16Bits, toFloat16, accumulateFloat16, accumulateFloat32,
  scaledCast, masterWeightStep, float16Step, HALF_INF,
};

if (require.main === module) {
  // -- known encodings ---------------------------------------------------------------------------
  strictEqual(toFloat16Bits(1), 0x3c00);
  strictEqual(toFloat16Bits(2), 0x4000);
  strictEqual(toFloat16Bits(-2), 0xc000);
  strictEqual(toFloat16Bits(0.5), 0x3800);
  strictEqual(toFloat16Bits(0), 0x0000);
  strictEqual(toFloat16Bits(-0), 0x8000);
  strictEqual(toFloat16Bits(65504), 0x7bff, "the largest finite half");
  strictEqual(toFloat16Bits(Infinity), 0x7c00);
  strictEqual(toFloat16Bits(-Infinity), 0xfc00);
  strictEqual(toFloat16Bits(NaN), 0x7e00, "NaN is emitted as a quiet NaN");
  strictEqual(toFloat16Bits(2 ** -14), 0x0400, "the smallest normal half");
  strictEqual(toFloat16Bits(2 ** -24), 0x0001, "the smallest subnormal half");
  strictEqual(toFloat16Bits(1 + 2 ** -10), 0x3c01);
  strictEqual(fromFloat16Bits(0x3c00), 1);
  strictEqual(fromFloat16Bits(0x0001), 2 ** -24);
  ok(Number.isNaN(fromFloat16Bits(0x7e00)));
  ok(fromFloat16Bits(0x7c00) === Infinity && fromFloat16Bits(0xfc00) === -Infinity);
  strictEqual(fromFloat16Bits(0x8000), -0);
  throws(() => fromFloat16Bits(0x10000), RangeError);

  // -- round trip: at most half an ulp for every normal value ------------------------------------
  const rng = makeRng(717);
  let worstRelative = 0;
  for (let trial = 0; trial < 4000; trial++) {
    const exponent = rng(31) - 15;
    const value = (1 + rng(1000) / 1000) * 2 ** exponent;
    const restored = toFloat16(value);
    if (exponent >= -14) {
      // normals: the error is at most half an ulp in relative terms
      const relative = Math.abs(restored - value) / value;
      worstRelative = Math.max(worstRelative, relative);
      ok(relative <= 2 ** -11 + 1e-15, `half-ulp bound violated at ${value}: ${relative}`);
    } else {
      // subnormals have a fixed absolute step of 2^-24, so the absolute bound is the right one
      ok(Math.abs(restored - value) <= 2 ** -25 + 1e-30, `subnormal bound violated at ${value}`);
    }
  }
  ok(worstRelative > 0, "the cast really does round something");
  ok(toFloat16(1) === 1 && toFloat16(-1) === -1, "exactly representable values are untouched");
  for (const value of [0, 1, 2, 1024, 0.5, 0.25, 2 ** -14, -3.5]) {
    strictEqual(toFloat16(value), value, `${value} is exactly representable`);
  }

  // -- ties round to the even significand ---------------------------------------------------------
  strictEqual(toFloat16Bits(1 + 2 ** -11), 0x3c00, "the tie drops to 1.0 because 0 is even");
  strictEqual(toFloat16Bits(1 + 3 * 2 ** -11), 0x3c02, "the tie rises to 1 + 2^-9 because 2 is even");
  strictEqual(toFloat16Bits(1 + 2 ** -11 + 2 ** -30), 0x3c01, "just above the tie rounds up");
  strictEqual(toFloat16Bits(1 + 2 ** -11 - 2 ** -30), 0x3c00, "just below the tie rounds down");
  strictEqual(toFloat16Bits(2 ** -25), 0, "half of the smallest subnormal ties down to zero");
  strictEqual(toFloat16Bits(2 ** -25 + 2 ** -40), 0x0001, "just above it rounds up");

  // -- overflow and underflow ---------------------------------------------------------------------
  strictEqual(toFloat16Bits(65519), 0x7bff, "just below the halfway point stays finite");
  strictEqual(toFloat16Bits(65520), HALF_INF, "the halfway point ties to the even value, infinity");
  strictEqual(toFloat16Bits(1e6), HALF_INF);
  strictEqual(toFloat16(1e-8), 0, "1e-8 is below the smallest subnormal");
  strictEqual(toFloat16(-1e-8), -0, "and it keeps its sign");
  ok(Math.abs(toFloat16(1e-5) - 1e-5) <= 2 ** -25, "1e-5 is a subnormal and survives to half a step");

  // -- why the accumulator has to be FP32 ----------------------------------------------------------
  const tiny = 2 ** -11;
  const increments = new Array(2048).fill(tiny);
  const fp16Sum = accumulateFloat16([1, ...increments]);
  const fp32Sum = accumulateFloat32([1, ...increments]);
  const exact = 1 + 2048 * tiny;
  ok(Math.abs(fp32Sum - exact) < 1e-9, "the FP32 accumulator is essentially exact here");
  ok(Math.abs(fp16Sum - exact) > 0.5, `the FP16 accumulator loses the increments (${fp16Sum} vs ${exact})`);
  ok(Math.abs(fp16Sum - toFloat16(1)) < 1e-3, "it gets stuck at the first representable step");
  strictEqual(accumulateFloat16([0.5, 0.25, 0.125]), 0.875, "small sums are exact in FP16 as well");

  // -- loss scaling rescues gradients that FP16 alone flushes to zero ------------------------------
  const gradient = 1e-8;
  strictEqual(toFloat16(gradient), 0, "an unscaled 1e-8 gradient disappears");
  const unscaled = scaledCast(gradient, 1);
  strictEqual(unscaled.value, 0);
  const scale = 2 ** 20;
  const scaled = scaledCast(gradient, scale);
  ok(!scaled.overflow && Math.abs(scaled.value - gradient) / gradient < 0.001,
    `loss scaling keeps the gradient (${scaled.value})`);
  const weights = [1];
  const plain = float16Step(weights, [gradient], 0.5);
  strictEqual(plain[0], 1, "the plain FP16 step does nothing at all");
  const step = masterWeightStep(weights, [gradient], { learningRate: 0.5, scale });
  const idealStep = 0.5 * gradient;
  const actualStep = 1 - step.weights[0];
  ok(actualStep > 0 && Math.abs(actualStep - idealStep) / idealStep < 1e-3,
    `the scaled step is within a tenth of a percent of the ideal (${actualStep} vs ${idealStep})`);
  ok(!step.skipped && step.scale === scale);

  // -- an overflow aborts the step and halves the scale --------------------------------------------
  const huge = masterWeightStep([1], [1e30], { learningRate: 0.5, scale: 2 ** 20 });
  ok(huge.skipped, "a gradient that overflows to infinity must abort the step");
  strictEqual(huge.weights[0], 1, "and leave the weights alone");
  strictEqual(huge.scale, 2 ** 19, "the scale is halved for the next attempt");
  deepStrictEqual(huge.gradients, [0]);
  ok(scaledCast(1e30, 2 ** 20).overflow, "a huge gradient with a large scale overflows to infinity");
  ok(!scaledCast(1e-3, 2 ** 10).overflow, "an ordinary gradient with an ordinary scale does not");
  ok(scaledCast(65520, 1).overflow, "values above the FP16 halfway point overflow even unscaled");

  // -- FP32 master weights keep the accumulated progress -------------------------------------------
  const steps = 4000;
  const gradientVector = [1e-3, -2e-4, 5e-5];
  let master = [1, 1, 1];
  let halfPrecision = [1, 1, 1];
  const learningRate = 1e-2;
  const busyScale = 2 ** 12;
  for (let round = 0; round < steps; round++) {
    const result = masterWeightStep(master, gradientVector, { learningRate, scale: busyScale });
    master = result.weights;
    halfPrecision = float16Step(halfPrecision, gradientVector, learningRate);
  }
  for (let i = 0; i < master.length; i++) {
    const exact = 1 - steps * learningRate * gradientVector[i];
    const movement = steps * learningRate * gradientVector[i];
    ok(Math.abs(master[i] - exact) < 0.01 * Math.abs(movement),
      `the master weight ${i} tracks the exact trajectory to within 1% of the travelled distance`);
    ok(Math.abs(master[i] - exact) <= Math.abs(halfPrecision[i] - exact) + 1e-12,
      `the FP32 master is at least as accurate as FP16 weights on coordinate ${i}`);
  }
  deepStrictEqual(halfPrecision, [1, 1, 1],
    "FP16 weights cannot represent any of these steps: each is far below the 2^-10 ulp at 1.0");
  ok(Math.abs(master[0] - 1) > 1e-6, "the FP32 master does move, by the full gradient");
  ok(Math.abs(master[2] - 1) > 1e-6, "even for the smallest gradient");

  // -- determinism ---------------------------------------------------------------------------------
  function replay(seed) {
    const local = makeRng(seed);
    const values = Array.from({ length: 12 }, () => (local(2000) - 1000) / 100);
    const bits = values.map(toFloat16Bits);
    const accumulated = accumulateFloat16(values).toFixed(12);
    return `${bits.join(",")}|${accumulated}`;
  }
  strictEqual(replay(5), replay(5), "identical seeds must replay identically");

  console.log(`worst relative round-trip error=${worstRelative.toExponential(3)} (half ulp = ${(2 ** -11).toExponential(3)})`);
  console.log(`FP16 accumulate=${fp16Sum.toFixed(6)} vs FP32 ${fp32Sum.toFixed(6)} vs exact ${exact}`);
  console.log("PASS");
}
