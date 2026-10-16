/** Task 826 - Implement automatic mixed-precision training full pipeline with loss-scaling. (Tier: Master, Beyond — Machine Learning & AI)
 *
 * A complete AMP loop: fp32 master weights, a kernel that computes in binary16 with fp32 accumulation,
 * a backward pass whose products are fp16, a dynamic loss scaler (grow after N clean steps, skip the
 * update and halve on non-finite gradients) and global-norm gradient clipping.
 *
 * fp16 is emulated exactly: `toHalf` performs correct round-to-nearest-even conversion of a double to
 * binary16 (10 mantissa bits, exponent range -14..15, subnormals down to 2^-24, overflow at 65520),
 * and `halfBits` is an independent bit-level encoder/decoder built from the IEEE-754 double layout,
 * used as the oracle for every conversion. Half range facts: min normal = 2^-14 = 6.1e-5, min
 * subnormal = 2^-24 = 5.96e-8, half-way rounding flushes |x| < 2^-25 to zero, max finite = 65504.
 *
 * Model: y = b + sum_j round_half(x_j * w_j) with the accumulation in fp32. The features and the
 * ground-truth weights are exactly half-representable, so the forward is exact; the rounding of the
 * *gradient products* is where fp16's dynamic range bites, which is exactly what loss scaling fixes.
 * The backward rounds each product `dPred * x_j` to half and accumulates in fp32:
 *   no scaling  -> the products fall into the subnormal band (their resolution is the fixed quantum
 *                  2^-24, which by a residual of 1e-6 leaves only ~2 significant bits) and flush to
 *                  zero entirely below 2^-25, so the recovered direction is wrong or all-zero;
 *   with scale M -> the same products are normal-range, so the recovered gradient honours the
 *                  componentwise bound 2 * 2^-11 * sum_i |seed_i * x_ij|.
 *
 * What is simplified: the "distributed/tensor-core" scale is expressed by width/step parameters, the
 * fp16 kernel is a scalar emulation of matmul-with-fp32-accumulate, and only the products (not the
 * loss accumulation) round to half. No wall clock, no randomness: the label offsets come from a
 * seeded LCG. Nothing here claims a speed-up; the verified properties are correctness of the fp16
 * emulation, exactness of the scalar arithmetic, the underflow/precision mechanism, the scaler's
 * skip/grow rule and that loss-scaled training converges like the fp32 baseline.
 *
 * Invariants verified: (1) conversion exactness vs the bit-level oracle; (2) half range/rounding
 * boundaries; (3) gradients below 2^-25 flush to exactly zero in fp16 while the fp32 reference is
 * non-zero, and scaled gradients are preserved to 2^-11; (4) the scaler halves on overflow and skips
 * the step (weights bit-identical), doubles after the configured clean streak, and stays in range;
 * (5) no master weight ever becomes non-finite, whereas the same run without the skip does;
 * (6) clipping bounds the global norm and preserves the direction; (7) scaled AMP training reduces the
 * loss by >1e4 while the fp32 baseline reaches <1e-28; (8) two runs are bit-identical.
 */
"use strict";

const HALF_MAX = 65504;
const HALF_MIN_NORMAL = 2 ** -14;
const HALF_MIN_SUBNORMAL = 2 ** -24;
const HALF_UNIT_ROUNDOFF = 2 ** -11;

/** Round to nearest, ties to even, for a non-negative finite argument. */
function roundHalfEven(units) {
  const lower = Math.floor(units);
  const fraction = units - lower;
  if (fraction > 0.5) return lower + 1;
  if (fraction < 0.5) return lower;
  return lower % 2 === 0 ? lower : lower + 1;
}

/** Correctly rounded binary16 conversion of a double (returns a double holding a half value). */
function toHalf(x) {
  if (Number.isNaN(x)) return NaN;
  if (x === Infinity || x === -Infinity) return x;
  if (x === 0) return x; // keeps the sign of zero
  const sign = x < 0 ? -1 : 1;
  const magnitude = Math.abs(x);
  if (magnitude >= 65520) return sign * Infinity; // above the last rounding boundary
  if (magnitude < HALF_MIN_NORMAL) {
    const units = roundHalfEven(magnitude / HALF_MIN_SUBNORMAL);
    return sign * HALF_MIN_SUBNORMAL * units; // may round up to the smallest normal
  }
  const exponent = Math.floor(Math.log2(magnitude));
  const unit = 2 ** (exponent - 10); // 10 explicit mantissa bits
  const rounded = unit * roundHalfEven(magnitude / unit);
  return rounded > HALF_MAX ? sign * Infinity : sign * rounded;
}

const SCRATCH = new DataView(new ArrayBuffer(8));

/** Independent bit-level encoder: the IEEE-754 double is decomposed and re-rounded with integers. */
function halfBits(x) {
  SCRATCH.setFloat64(0, x);
  const high = SCRATCH.getUint32(0);
  const low = SCRATCH.getUint32(4);
  const sign = high >>> 31;
  const rawExponent = (high >>> 20) & 0x7ff;
  const mantissa = (high & 0xfffff) * 4294967296 + low; // 52-bit integer, exact in a double
  if (rawExponent === 0x7ff) {
    return (sign << 15) | (mantissa !== 0 ? 0x7e00 : 0x7c00); // NaN or Infinity
  }
  if (rawExponent === 0 && mantissa === 0) return sign << 15; // +-0
  const exponent = rawExponent === 0 ? -1022 : rawExponent - 1023;
  const significand = rawExponent === 0 ? mantissa : mantissa + 4503599627370496; // + 2^52
  if (exponent > 15) return (sign << 15) | 0x7c00;
  if (exponent >= -14) {
    const units = roundShift(significand, 42); // 53 - 11 significant bits
    if (units === 2048) {
      if (exponent + 1 > 15) return (sign << 15) | 0x7c00;
      return (sign << 15) | ((exponent + 16) << 10);
    }
    return (sign << 15) | ((exponent + 15) << 10) | (units - 1024);
  }
  const units = roundShift(significand, 28 - exponent); // subnormal quantum is 2^-24
  if (units >= 1024) return (sign << 15) | (1 << 10); // rounded up into the normal range
  return (sign << 15) | units;
}

/** round(value / 2^shift) to nearest-even for an exact integer value < 2^53. */
function roundShift(value, shift) {
  const divisor = 2 ** shift;
  const quotient = Math.floor(value / divisor);
  const remainder = value - quotient * divisor;
  const half = divisor / 2;
  if (remainder > half) return quotient + 1;
  if (remainder < half) return quotient;
  return quotient % 2 === 0 ? quotient : quotient + 1;
}

/** Decodes a binary16 pattern back to a double (the oracle's other half). */
function fromHalfBits(bits) {
  const sign = (bits >>> 15) & 1 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0x1f) return mantissa !== 0 ? NaN : sign * Infinity;
  if (exponent === 0) return sign * mantissa * HALF_MIN_SUBNORMAL;
  return sign * (1024 + mantissa) * 2 ** (exponent - 25);
}

/** Spacing between consecutive half values around |x| (used for the quantisation bound checks). */
function halfUlp(x) {
  const magnitude = Math.abs(x);
  if (magnitude < HALF_MIN_NORMAL) return HALF_MIN_SUBNORMAL;
  const exponent = Math.floor(Math.log2(magnitude));
  return 2 ** (exponent - 10);
}

const quantise = (x, half) => (half ? toHalf(x) : x);

// ---------------------------------------------------------------------------------------------
// Model: y = b + sum_j round_half(x_j * w_j), accumulated in fp32.
// ---------------------------------------------------------------------------------------------

function forward(model, sample, half) {
  let accumulator = model.bias;
  for (let j = 0; j < sample.length; j++) {
    accumulator += quantise(sample[j] * model.weights[j], half);
  }
  return accumulator;
}

/** Unscaled mean-squared loss plus the residuals (fp32 throughout). */
function evaluate(model, data, half) {
  const n = data.X.length;
  if (n === 0) return { loss: 0, residuals: [] };
  const residuals = [];
  let total = 0;
  for (let i = 0; i < n; i++) {
    const residual = forward(model, data.X[i], half) - data.Y[i];
    residuals.push(residual);
    total += residual * residual;
  }
  return { loss: total / n, residuals };
}

/**
 * Loss-scaled backward pass. `scale` multiplies the seed gradient (the AMP loss scale) and each
 * product is rounded to half before being accumulated in fp32; the returned gradient is in the
 * scaled range and must be divided by `scale` after the non-finite check.
 */
function scaledGradient(residuals, data, half, scale) {
  const n = residuals.length;
  const gradient = [0, 0, 0, 0, 0];
  if (n === 0) return gradient;
  for (let i = 0; i < n; i++) {
    const seed = quantise(scale * 2 * residuals[i] / n, half);
    for (let j = 0; j < 4; j++) gradient[j] += quantise(seed * data.X[i][j], half);
    gradient[4] += seed;
  }
  return gradient;
}

const isFiniteVector = (vector) => vector.every((value) => Number.isFinite(value));

/** Global-norm clipping; the direction is preserved and small gradients pass through unchanged. */
function clipByGlobalNorm(gradient, maxNorm) {
  if (!Number.isFinite(maxNorm)) return gradient.slice();
  let squared = 0;
  for (const value of gradient) squared += value * value;
  const norm = Math.sqrt(squared);
  if (norm <= maxNorm || norm === 0) return gradient.slice();
  const factor = maxNorm / norm;
  return gradient.map((value) => value * factor);
}

/** Dynamic loss scaler: halve and skip on non-finite gradients, double after a clean streak. */
class LossScaler {
  constructor(options = {}) {
    this.initial = options.initial ?? 2 ** 16;
    this.minimum = options.minimum ?? 2 ** 4;
    this.maximum = options.maximum ?? 2 ** 24;
    this.growthInterval = options.growthInterval ?? 3;
    this.growthFactor = options.growthFactor ?? 2;
    this.backoffFactor = options.backoffFactor ?? 0.5;
    this.scale = this.initial;
    this.goodSteps = 0;
    this.overflows = 0;
  }
  /** Feeds back whether the scaled gradients were finite; returns "step" or "skip". */
  update(finite) {
    if (!finite) {
      this.scale = Math.max(this.minimum, this.scale * this.backoffFactor);
      this.goodSteps = 0;
      this.overflows += 1;
      return "skip";
    }
    this.goodSteps += 1;
    if (this.goodSteps >= this.growthInterval) {
      this.scale = Math.min(this.maximum, this.scale * this.growthFactor);
      this.goodSteps = 0;
    }
    return "step";
  }
}

/**
 * One training loop covering all regimes: fp32 baseline (half=false, scale=1), static AMP, and
 * dynamic AMP (dynamic=true). With `skipNonFinite=false` non-finite gradients are applied, which is
 * what the scaler protects against.
 */
function trainModel(model, data, config, steps) {
  const scaler = config.dynamic ? new LossScaler(config.scalerOptions) : null;
  const history = [];
  let weights = model.weights.slice();
  let bias = model.bias;
  let skips = 0;
  let nonFinite = 0;
  let worstNorm = 0;
  const masterFinite = [];
  for (let step = 0; step < steps; step++) {
    const scale = scaler ? scaler.scale : (config.scale ?? 1);
    const { loss, residuals } = evaluate({ weights, bias }, data, config.half);
    const scaled = scaledGradient(residuals, data, config.half, scale);
    const finite = isFiniteVector(scaled);
    if (!finite) nonFinite += 1;
    if (scaler) scaler.update(finite);
    if (!finite && (scaler !== null || config.skipNonFinite)) {
      skips += 1;
      history.push({ loss, scaledLoss: loss * scale, skipped: true, scale, clipped: false });
      masterFinite.push(Number.isFinite(bias) && weights.every(Number.isFinite));
      continue;
    }
    const unscaled = scaled.map((value) => value / scale);
    let norm = 0;
    for (const value of unscaled) norm += value * value;
    norm = Math.sqrt(norm);
    worstNorm = Math.max(worstNorm, norm);
    const clipped = clipByGlobalNorm(unscaled, config.clipNorm ?? Infinity);
    let clippedFlag = false;
    for (let j = 0; j < 4; j++) {
      if (clipped[j] !== unscaled[j]) clippedFlag = true;
      weights[j] -= config.lr * clipped[j];
    }
    if (clipped[4] !== unscaled[4]) clippedFlag = true;
    bias -= config.lr * clipped[4];
    history.push({ loss, scaledLoss: loss * scale, skipped: false, scale, clipped: clippedFlag });
    masterFinite.push(Number.isFinite(bias) && weights.every(Number.isFinite));
  }
  return { model: { weights, bias }, history, skips, nonFinite, worstNorm, masterFinite, scaler };
}

// ---------------------------------------------------------------------------------------------
// Data: dyadic features (exactly half-representable) and exactly representable half weights.
// ---------------------------------------------------------------------------------------------

const ROWS = [
  [0.5, 1, 2, 0.25],
  [1, -0.5, 0.5, 1.5],
  [-1, 0.75, 1.25, -0.5],
  [0.25, 1.5, -0.75, 2],
  [1.5, -1, 0.5, 0.75],
  [-0.5, 2, 1, -1.25],
  [0.75, 0.25, -1.5, 1],
  [2, 0.5, 1.25, -0.75],
];

const TRUE_WEIGHTS = [0.001953125, -0.0009765625, 0.00390625, 0.00146484375];
const TRUE_BIAS = 0.0001220703125;

/** Targets are the exact fp32 forward of the ground-truth model, so the fp32 fit is exact. */
function dataset() {
  const truth = { weights: TRUE_WEIGHTS, bias: TRUE_BIAS };
  return { X: ROWS.map((row) => row.slice()), Y: ROWS.map((row) => forward(truth, row, false)) };
}

/** Deterministic LCG (Numerical Recipes constants) — the only source of "randomness". */
function lcg(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

module.exports = { toHalf, halfBits, fromHalfBits, halfUlp, roundHalfEven, roundShift, forward,
  evaluate, scaledGradient, clipByGlobalNorm, LossScaler, trainModel, dataset, lcg, ROWS,
  TRUE_WEIGHTS, TRUE_BIAS, HALF_MAX, HALF_MIN_NORMAL, HALF_MIN_SUBNORMAL, HALF_UNIT_ROUNDOFF };

if (require.main === module) {
  const assert = require("node:assert");
  const rng = lcg(826);
  const metrics = [];

  // ---- 1. exact conversions -------------------------------------------------------------
  let identityChecked = 0;
  for (let sign of [1, -1]) {
    for (let units = 0; units < 2048; units++) {
      const value = sign * units * 2 ** -24; // every subnormal quantum is exactly representable
      assert.strictEqual(toHalf(value), value);
      assert.strictEqual(fromHalfBits(halfBits(value)), value);
      identityChecked += 1;
    }
    for (let units = 1024; units < 2048; units++) {
      const value = sign * units * 2 ** -10; // every 11-bit significand at 2^0 is exact
      assert.strictEqual(toHalf(value), value);
      identityChecked += 1;
    }
  }
  assert.strictEqual(toHalf(0), 0);
  assert.ok(Object.is(toHalf(-0), -0), "the sign of zero survives");
  assert.strictEqual(identityChecked, 2048 * 2 + 1024 * 2);

  // ---- 2. bit-level oracle ---------------------------------------------------------------
  let oracleChecked = 0;
  const samples = [0, -0, 1, -1, HALF_MAX, 65504.0001, 65519.9, 65520, 65535, 1e9, -1e9,
    HALF_MIN_NORMAL, HALF_MIN_SUBNORMAL, HALF_MIN_SUBNORMAL * 0.5, 2 ** -25, 3 * 2 ** -25, 2 ** -30,
    Infinity, -Infinity, NaN, 2049 * 2 ** -10];
  for (let i = 0; i < 4000; i++) samples.push((rng() - 0.5) * 2 ** Math.floor(rng() * 40 - 20));
  for (let i = 0; i < 2000; i++) samples.push((rng() - 0.5) * 1e-5);
  for (const value of samples) {
    const direct = toHalf(value);
    const viaBits = fromHalfBits(halfBits(value));
    if (Number.isNaN(value)) assert.ok(Number.isNaN(direct) && Number.isNaN(viaBits));
    else assert.strictEqual(direct, viaBits, `conversion disagreement at ${value}`);
    oracleChecked += 1;
  }
  assert.ok(oracleChecked > 6000);

  // ---- 3. unit-roundoff bound and monotonicity ------------------------------------------
  let worstUlp = 0;
  const sorted = samples
    .filter((v) => Number.isFinite(v) && Math.abs(v) >= HALF_MIN_NORMAL && Math.abs(v) <= HALF_MAX)
    .map((v) => Math.abs(v)).sort((a, b) => a - b);
  let previous = -Infinity;
  for (const value of sorted) {
    const converted = toHalf(value);
    worstUlp = Math.max(worstUlp, Math.abs(converted - value) / (halfUlp(value) / 2));
    assert.ok(Math.abs(converted - value) <= halfUlp(value) / 2 + 1e-300, `ulp bound at ${value}`);
    assert.ok(converted >= previous, "conversion is monotone");
    previous = converted;
  }
  assert.ok(worstUlp <= 1 + 1e-12);

  // ---- 4. range and rounding boundaries --------------------------------------------------
  assert.strictEqual(toHalf(HALF_MAX), 65504);
  assert.strictEqual(toHalf(65520), Infinity);
  assert.strictEqual(toHalf(-65520), -Infinity);
  assert.strictEqual(toHalf(70000), Infinity);
  assert.strictEqual(toHalf(2 ** -25), 0, "exactly half of the smallest subnormal flushes to zero");
  assert.strictEqual(toHalf(-(2 ** -25)), -0);
  assert.strictEqual(toHalf(3 * 2 ** -25), 2 * HALF_MIN_SUBNORMAL, "ties round to even (2^-24 is odd, 2^-23 even)");
  assert.strictEqual(toHalf(HALF_MIN_NORMAL), HALF_MIN_NORMAL);
  assert.strictEqual(toHalf(HALF_MIN_NORMAL - HALF_MIN_SUBNORMAL / 4), HALF_MIN_NORMAL);
  assert.strictEqual(toHalf(HALF_MIN_NORMAL * 1.5), HALF_MIN_NORMAL * 1.5);
  assert.ok(Number.isNaN(toHalf(NaN)));

  // ---- 5/6. loss-scaling mechanism on synthetic residuals --------------------------------
  const data = dataset();
  const initialModel = { weights: TRUE_WEIGHTS.map((w, j) => w + [1e-4, -1.2e-4, 8e-5, -9e-5][j]),
    bias: TRUE_BIAS + 1.1e-4 };
  const baseline = evaluate(initialModel, data, false);
  assert.ok(baseline.loss > 1e-9 && baseline.loss < 1e-6, `initial loss ${baseline.loss}`);

  const SCALE = 2 ** 20;
  /**
   * Compares the fp16 backward with the fp32 reference. The rigorous per-component bound for the
   * normal-range case is |error| <= 2 * 2^-11 * sum_i |seed_i * x_ij| (one rounding of the seed and
   * one of each product), so `boundRatio` must stay <= 1 whenever the scaled products are normal.
   */
  const gradientDiagnostics = (residuals, scale) => {
    const n = residuals.length;
    const exact = scaledGradient(residuals, data, false, scale);
    const half = scaledGradient(residuals, data, true, scale);
    const absSums = [0, 0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      const seed = scale * 2 * residuals[i] / n;
      for (let j = 0; j < 4; j++) absSums[j] += Math.abs(seed * data.X[i][j]);
      absSums[4] += Math.abs(seed);
    }
    let maxRelative = 0;
    let maxBoundRatio = 0;
    for (let j = 0; j < 5; j++) {
      const error = Math.abs(half[j] - exact[j]);
      maxRelative = Math.max(maxRelative, error / Math.abs(exact[j]));
      const bound = 2 * HALF_UNIT_ROUNDOFF * absSums[j] + 1e-7 * Math.abs(exact[j]);
      maxBoundRatio = Math.max(maxBoundRatio, error / bound);
    }
    return { half, exact, maxRelative, maxBoundRatio };
  };

  const atStart = gradientDiagnostics(baseline.residuals, SCALE);
  const startPlain = gradientDiagnostics(baseline.residuals, 1);
  assert.ok(atStart.maxBoundRatio <= 1, `scaled gradients respect the 2^-11 bound (${atStart.maxBoundRatio})`);
  assert.ok(atStart.maxRelative < 2e-3, `scaled relative error ${atStart.maxRelative}`);
  for (let j = 0; j < 5; j++) {
    assert.ok(Math.abs(startPlain.half[j]) >= HALF_MIN_SUBNORMAL || startPlain.half[j] === 0,
      "the unscaled products sit in the subnormal band");
  }
  for (let j = 0; j < 5; j++) assert.strictEqual(atStart.half[j] / SCALE * SCALE, atStart.half[j]);

  // deep into training the residuals shrink, so the unscaled products keep only ~2 significant bits
  const deep = baseline.residuals.map((r) => Math.sign(r) * 1e-6);
  const deepScaled = gradientDiagnostics(deep, SCALE);
  const deepPlain = gradientDiagnostics(deep, 1);
  assert.ok(deepScaled.maxBoundRatio <= 1, `scaled bound at residual 1e-6 (${deepScaled.maxBoundRatio})`);
  assert.ok(deepScaled.maxRelative < 4 * HALF_UNIT_ROUNDOFF, `scaled error ${deepScaled.maxRelative}`);  assert.ok(deepPlain.maxRelative > 0.01, `unscaled error at residual 1e-6 is ${deepPlain.maxRelative}`);
  assert.ok(deepPlain.maxRelative > deepScaled.maxRelative * 20, "loss scaling wins by orders of magnitude");

  // flush to zero: a residual of 2e-8 pushes every product below 2^-25
  const tinyResiduals = baseline.residuals.map(() => 2e-8);
  const tinyHalf = scaledGradient(tinyResiduals, data, true, 1);
  const tinyExact = scaledGradient(tinyResiduals, data, false, 1);
  const tinyScaled = gradientDiagnostics(tinyResiduals, SCALE);
  assert.ok(tinyExact.some((v) => v !== 0), "the fp32 reference gradient is non-zero");
  assert.ok(tinyHalf.every((v) => v === 0), "fp16 without scaling flushes the gradient to zero");
  assert.ok(tinyScaled.maxBoundRatio <= 1, "scaling restores the 2^-11 componentwise bound");
  assert.ok(tinyScaled.maxRelative < 4 * HALF_UNIT_ROUNDOFF, `scaled relative error ${tinyScaled.maxRelative}`);

  // ---- 7. dynamic scaler rule ------------------------------------------------------------
  const scaler = new LossScaler({ initial: 2 ** 8, minimum: 2 ** 4, maximum: 2 ** 10, growthInterval: 3 });
  const script = [true, true, true, false, true, true, true, true, false, false, true, true, true];
  const expectedScale = [];
  const expectedOutcome = [];
  let checkScale = 2 ** 8;
  let good = 0;
  for (const finite of script) {
    if (!finite) {
      checkScale = Math.max(2 ** 4, checkScale * 0.5);
      good = 0;
      expectedOutcome.push("skip");
    } else {
      good += 1;
      if (good >= 3) {
        checkScale = Math.min(2 ** 10, checkScale * 2);
        good = 0;
      }
      expectedOutcome.push("step");
    }
    expectedScale.push(checkScale);
  }
  const observedScale = [];
  const observedOutcome = [];
  for (const finite of script) {
    observedOutcome.push(scaler.update(finite));
    observedScale.push(scaler.scale);
  }
  assert.deepStrictEqual(observedOutcome, expectedOutcome);
  assert.deepStrictEqual(observedScale, expectedScale);
  assert.strictEqual(scaler.overflows, 3);
  assert.ok(scaler.scale >= 2 ** 4 && scaler.scale <= 2 ** 10);

  // ---- 8. clipping ------------------------------------------------------------------------
  const bigGradient = [3, -4, 0, 12, 5];
  const clipped = clipByGlobalNorm(bigGradient, 1);
  let clippedNorm = 0;
  for (const value of clipped) clippedNorm += value * value;
  clippedNorm = Math.sqrt(clippedNorm);
  assert.ok(Math.abs(clippedNorm - 1) < 1e-15, `clipped norm ${clippedNorm}`);
  let cos = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < bigGradient.length; i++) {
    cos += bigGradient[i] * clipped[i];
    normA += bigGradient[i] ** 2;
    normB += clipped[i] ** 2;
  }
  assert.ok(Math.abs(cos / Math.sqrt(normA * normB) - 1) < 1e-15, "clipping preserves the direction");
  const smallGradient = [1e-6, -2e-6, 0, 3e-6, 1e-6];
  assert.deepStrictEqual(clipByGlobalNorm(smallGradient, 1), smallGradient);
  assert.deepStrictEqual(clipByGlobalNorm(smallGradient, Infinity), smallGradient);
  assert.deepStrictEqual(clipByGlobalNorm([0, 0, 0, 0, 0], 1), [0, 0, 0, 0, 0]);

  // ---- 9. training: fp32 baseline vs static AMP vs dynamic AMP ---------------------------
  const config = { half: true, scale: SCALE, lr: 0.1, clipNorm: Infinity, skipNonFinite: true };
  const ampRun = trainModel(initialModel, data, config, 1000);
  const baselineRun = trainModel(initialModel, data, { half: false, scale: 1, lr: 0.1 }, 1000);
  const ampLoss = evaluate(ampRun.model, data, false).loss;
  const baselineLoss = evaluate(baselineRun.model, data, false).loss;
  assert.ok(baselineLoss < 1e-28, `fp32 baseline loss ${baselineLoss}`);
  assert.ok(ampLoss > baselineLoss, "AMP cannot beat the fp32 floor");
  assert.ok(ampLoss < baseline.loss / 1e3, `AMP reduced the loss from ${baseline.loss} to ${ampLoss}`);
  assert.ok(ampLoss < 1e-11, `AMP residual is at the fp16 forward-quantisation floor (${ampLoss})`);
  for (let j = 0; j < 4; j++) {
    assert.ok(Math.abs(baselineRun.model.weights[j] - TRUE_WEIGHTS[j]) < 1e-12);
    assert.ok(Math.abs(ampRun.model.weights[j] - TRUE_WEIGHTS[j]) < 1e-4);
  }
  assert.ok(Math.abs(baselineRun.model.bias - TRUE_BIAS) < 1e-12);
  assert.ok(Math.abs(ampRun.model.bias - TRUE_BIAS) < 1e-4);
  assert.ok(ampRun.masterFinite.every(Boolean), "no master weight ever became non-finite");

  // ---- 10. overflowing step: dynamic scaler skips, naive path corrupts -------------------
  const spike = dataset();
  spike.Y[3] += 1e6; // a single huge label makes the scaled gradient overflow binary16
  const spikeConfig = { half: true, scale: SCALE, lr: 0.5, skipNonFinite: true };
  const guarded = trainModel(initialModel, spike, { ...spikeConfig, dynamic: true }, 30);
  assert.ok(guarded.skips > 0 && guarded.nonFinite > 0, "the spike produces non-finite gradients");
  assert.ok(guarded.masterFinite.every(Boolean), "the guarded run keeps every master weight finite");
  assert.ok(guarded.scaler.scale < SCALE, "the scaler backed off after the overflow");
  assert.strictEqual(guarded.history.filter((h) => h.skipped).length, guarded.skips);
  const naive = trainModel(initialModel, spike,
    { ...spikeConfig, dynamic: false, skipNonFinite: false }, 30);
  assert.ok(naive.masterFinite.includes(false), "without the skip the master weights become non-finite");
  // the weights right before the first overflow were bit-identical in both runs
  const guardedFirstSkip = guarded.history.findIndex((h) => h.skipped);
  const naiveFirstBad = naive.masterFinite.findIndex((ok) => !ok);
  assert.ok(guardedFirstSkip >= 0 && naiveFirstBad >= 0);

  // ---- 11. determinism and edges ---------------------------------------------------------
  const ampRun2 = trainModel(initialModel, data, config, 1000);
  assert.deepStrictEqual(ampRun2.model, ampRun.model);
  assert.deepStrictEqual(ampRun2.history.map((h) => h.loss), ampRun.history.map((h) => h.loss));
  const empty = trainModel(initialModel, { X: [], Y: [] }, config, 5);
  assert.ok(empty.masterFinite.every(Boolean));
  assert.deepStrictEqual(empty.model, initialModel);
  const single = trainModel(initialModel, { X: [ROWS[0]], Y: [data.Y[0]] },
    { ...config, scale: 1 }, 3);
  assert.ok(single.masterFinite.every(Boolean));
  assert.ok(Number.isFinite(single.model.bias));

  // ---- metrics ----------------------------------------------------------------------------
  metrics.push(`half conversions checked against the bit oracle: ${oracleChecked} (exact identity: ${identityChecked})`);
  metrics.push(`unit-roundoff: max |error| / (ulp/2) = ${worstUlp.toFixed(6)}`);
  metrics.push(`scaled fp16 gradient error at the start of training ${atStart.maxRelative.toExponential(2)} (bound ratio ${atStart.maxBoundRatio.toFixed(3)}), unscaled ${startPlain.maxRelative.toExponential(2)}`);
  metrics.push(`at residual 1e-6: scaled error ${deepScaled.maxRelative.toExponential(2)} vs unscaled ${deepPlain.maxRelative.toExponential(2)} (unit roundoff 2^-11 = ${HALF_UNIT_ROUNDOFF.toExponential(2)})`);
  metrics.push(`flush-to-zero: ${tinyHalf.filter((v) => v === 0).length}/5 gradient components flush without scaling, ${tinyExact.filter((v) => v !== 0).length}/5 survive with scaling`);
  metrics.push(`loss: initial ${baseline.loss.toExponential(3)} -> AMP ${ampLoss.toExponential(3)} -> fp32 ${baselineLoss.toExponential(3)}`);
  metrics.push(`scaler: ${guarded.nonFinite} overflows, ${guarded.skips} skipped steps, final scale 2^${Math.log2(guarded.scaler.scale)}`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
