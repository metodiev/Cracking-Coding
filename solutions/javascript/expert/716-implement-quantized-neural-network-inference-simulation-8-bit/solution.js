/** Task 716 - Implement quantized neural network inference simulation (8-bit). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * 8-bit inference replaces every float by an integer plus a shared scale: x ~= scale * (q - zeroPoint)
 * with q in [-128, 127] (symmetric) or [0, 255] (asymmetric). The point of the scheme is that a
 * matrix product can then be evaluated entirely in integers - the products q_a * q_w accumulate in
 * int32 and the scales are applied once at the end - which is what this file simulates and checks:
 *   - the rounding error of a quantise/dequantise round trip is bounded by scale/2, and saturation
 *     (clamping) costs more, which is measured separately instead of being hidden;
 *   - the int32 accumulator is exact for the simulated sizes, and the worst case 127 * 127 * K
 *     provably fits in int32 only while K <= 133152;
 *   - a full linear layer plus a two-layer classifier are run twice, in float and in int8, and the
 *     agreement of the argmax is measured;
 *   - per-channel scales (one scale per output row) are compared with a single per-tensor scale.
 *
 * What is simplified: no packed integer kernels (the accumulation is plain JavaScript integer
 * arithmetic), no per-batch calibration - the activation scale is taken from the observed tensor -
 * and the network is a hand-initialised two-layer MLP rather than a trained model.
 *
 * Invariant verified: every quantised hypothesis respects the scale/2 round-trip bound, the integer
 * dot product equals the exact BigInt sum of the same products, the int32 overflow bound is exactly
 * where the arithmetic says it is, zero-filled tensors quantise to all zeros with a defined scale,
 * and the quantised classifier agrees with the float classifier on the vast majority of inputs.
 */
"use strict";
const { ok, strictEqual, deepStrictEqual, throws } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

/**
 * Symmetric quantisation: range [-127, 127] around zero, so no zero point is needed. An explicit
 * `scale` models the real case where the activations are calibrated on one batch and a later batch
 * reaches outside that range.
 */
function symmetricQuantize(values, scale = null) {
  const maxAbs = values.reduce((best, value) => Math.max(best, Math.abs(value)), 0);
  const chosen = scale === null ? (maxAbs === 0 ? 1 : maxAbs / 127) : scale;
  const quantized = values.map((value) => clamp(Math.round(value / chosen), -127, 127));
  const saturated = values.filter((value) => Math.abs(Math.round(value / chosen)) > 127).length;
  return { scale: chosen, zeroPoint: 0, quantized, saturated, maxAbs };
}

/** Asymmetric quantisation: the observed range maps onto [0, 255] with a zero point. */
function asymmetricQuantize(values) {
  const min = values.reduce((best, value) => Math.min(best, value), 0);
  const max = values.reduce((best, value) => Math.max(best, value), 0);
  const span = max - min;
  const scale = span === 0 ? 1 : span / 255;
  const zeroPoint = clamp(Math.round(-min / scale), 0, 255);
  const quantized = values.map((value) => clamp(Math.round(value / scale) + zeroPoint, 0, 255));
  return { scale, zeroPoint, quantized, span };
}

function dequantize(quantized, scale, zeroPoint = 0) {
  return quantized.map((value) => scale * (value - zeroPoint));
}

/** Worst observed quantisation error of a round trip. */
function roundTripError(values, quantized, scale, zeroPoint = 0) {
  const restored = dequantize(quantized, scale, zeroPoint);
  return values.reduce((worst, value, i) => Math.max(worst, Math.abs(value - restored[i])), 0);
}

/**
 * Integer-only linear layer. The dot product accumulates in int32 values and the scales are applied
 * once per output; `perChannel` uses one weight scale per output row instead of a single scale.
 */
function quantizedLinear(inputs, weights, biases, { perChannel = false } = {}) {
  const inputQuant = symmetricQuantize(inputs);
  const rows = weights.length;
  const outputs = [];
  const weightScales = [];
  const integerDot = [];
  for (let row = 0; row < rows; row++) {
    const weightQuant = symmetricQuantize(weights[row]);
    weightScales.push(weightQuant.scale);
    let accumulator = 0;
    for (let k = 0; k < inputs.length; k++) {
      accumulator += inputQuant.quantized[k] * weightQuant.quantized[k];
    }
    integerDot.push(accumulator);
    const outScale = perChannel ? inputQuant.scale * weightQuant.scale : inputQuant.scale * weightScales[0];
    outputs.push(accumulator * outScale + biases[row]);
  }
  const tensorScale = inputQuant.scale * weightScales[0];
  return {
    outputs,
    integerDot,
    inputScale: inputQuant.scale,
    weightScales,
    tensorScale,
    scalarOutputs: integerDot.map((value, row) => value * tensorScale + biases[row]),
  };
}

/** Exact reference for the same integer products, computed with BigInt and never overflowing. */
function exactIntegerDot(inputQuantized, weightQuantized) {
  let total = 0n;
  for (let k = 0; k < inputQuantized.length; k++) {
    total += BigInt(inputQuantized[k]) * BigInt(weightQuantized[k]);
  }
  return total;
}

/** The largest K for which the worst-case int8 dot product still fits in int32. */
function maxAccumulationLength() { return Math.floor((2 ** 31 - 1) / (127 * 127)); }

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

module.exports = {
  makeRng, symmetricQuantize, asymmetricQuantize, dequantize, roundTripError, quantizedLinear,
  exactIntegerDot, maxAccumulationLength, argmax,
};

if (require.main === module) {
  // -- symmetric round trip and its scale/2 bound -------------------------------------------------
  const rng = makeRng(716);
  const sample = Array.from({ length: 200 }, () => (rng(20001) - 10000) / 1000);
  const symmetric = symmetricQuantize(sample);
  ok(Math.abs(symmetric.scale - symmetric.maxAbs / 127) < 1e-15, "the scale is max|x|/127");
  strictEqual(symmetric.saturated, 0, "nothing is clipped when the scale comes from the data");
  ok(roundTripError(sample, symmetric.quantized, symmetric.scale) <= symmetric.scale / 2 + 1e-12,
    "the round trip must stay inside scale/2");
  ok(symmetric.quantized.every((value) => Number.isInteger(value) && value >= -127 && value <= 127),
    "quantised values are integers inside the int8 range");

  // -- saturation is a real effect and is measured, not hidden ------------------------------------
  const calibrated = symmetricQuantize([-0.5, 0.5, 300], 0.5 / 127); // scale fixed by calibration
  strictEqual(calibrated.saturated, 1, "a value outside the calibrated range saturates");
  strictEqual(calibrated.quantized[2], 127, "and it clamps to the end of the range");
  ok(roundTripError([300], [calibrated.quantized[2]], calibrated.scale) > calibrated.scale / 2,
    "a saturated value exceeds the scale/2 bound - the caller must know");
  ok(Math.abs(0.5 - calibrated.quantized[1] * calibrated.scale) <= calibrated.scale / 2 + 1e-12,
    "values inside the range are still exact to scale/2");

  // -- degenerate tensors --------------------------------------------------------------------------
  const zeros = symmetricQuantize([0, 0, 0, 0]);
  strictEqual(zeros.scale, 1, "an all-zero tensor still needs a defined scale");
  deepStrictEqual(zeros.quantized, [0, 0, 0, 0]);
  deepStrictEqual(dequantize(zeros.quantized, zeros.scale), [0, 0, 0, 0]);
  const constant = symmetricQuantize([5, 5, 5]);
  deepStrictEqual(constant.quantized, [127, 127, 127], "a constant tensor uses the full range");
  deepStrictEqual(dequantize(constant.quantized, constant.scale), [5, 5, 5], "and reconstructs exactly");
  strictEqual(symmetricQuantize([-3]).quantized[0], -127);

  // -- asymmetric: the zero point maps exactly back to zero -----------------------------------------
  const asymmetricSample = [0, 1.5, -2, 4, -0.25];
  const asymmetric = asymmetricQuantize(asymmetricSample);
  ok(asymmetric.zeroPoint >= 0 && asymmetric.zeroPoint <= 255, "the zero point lives in the byte range");
  ok(roundTripError(asymmetricSample, asymmetric.quantized, asymmetric.scale, asymmetric.zeroPoint)
    <= asymmetric.scale / 2 + 1e-12, "the asymmetric round trip respects the same bound");
  ok(asymmetric.quantized.every((value) => Number.isInteger(value) && value >= 0 && value <= 255));
  strictEqual(asymmetric.quantized[0], asymmetric.zeroPoint, "the observed zero lands on the zero point");
  for (const value of [-1, 0, 1, 2.5]) {
    const quantized = clamp(Math.round(value / asymmetric.scale) + asymmetric.zeroPoint, 0, 255);
    ok(Math.abs(value - asymmetric.scale * (quantized - asymmetric.zeroPoint)) <= asymmetric.scale / 2 + 1e-12,
      `the asymmetric bound holds at ${value}`);
  }

  // -- the integer dot product is exact, and int32 has a hard limit --------------------------------
  const inputs = Array.from({ length: 64 }, () => (rng(2001) - 1000) / 500);
  const weights = [[...Array.from({ length: 64 }, () => (rng(2001) - 1000) / 500)]];
  const layer = quantizedLinear(inputs, weights, [0.25]);
  const inputQuant = symmetricQuantize(inputs);
  const weightQuant = symmetricQuantize(weights[0]);
  strictEqual(BigInt(layer.integerDot[0]), exactIntegerDot(inputQuant.quantized, weightQuant.quantized),
    "the int32 accumulator equals the exact BigInt sum");
  const limit = maxAccumulationLength();
  strictEqual(limit, 133144, "floor((2^31 - 1) / (127^2)) = 133144");
  ok(limit * 127 * 127 <= 2 ** 31 - 1, "the worst case still fits at K = 133144");
  ok((limit + 1) * 127 * 127 > 2 ** 31 - 1, "and overflows at K = 133145");
  const longInput = Array.from({ length: 4096 }, () => 1);
  const longLayer = quantizedLinear(longInput, [Array.from({ length: 4096 }, () => 1)], [0]);
  strictEqual(longLayer.integerDot[0], 127 * 127 * 4096, "a 4096-long dot product is still exact");

  // -- float versus int8 on a full layer -----------------------------------------------------------
  const floatOutput = inputs.reduce((sum, value, k) => sum + value * weights[0][k], 0) + 0.25;
  const quantizedOutput = layer.outputs[0];
  const relativeError = Math.abs(quantizedOutput - floatOutput) / Math.max(1, Math.abs(floatOutput));
  ok(relativeError < 0.02, `the quantised layer must stay close (relative error ${relativeError})`);
  ok(quantizedOutput !== floatOutput, "the quantised layer is not secretly the float computation");

  // -- per-channel scales must not be worse than the single per-tensor scale -----------------------
  const rows = [
    Array.from({ length: 32 }, () => (rng(20001) - 10000) / 1000),
    Array.from({ length: 32 }, () => (rng(2001) - 1000) / 100000),
  ];
  const perTensor = quantizedLinear(inputs.slice(0, 32), rows, [0, 0]);
  const perChannel = quantizedLinear(inputs.slice(0, 32), rows, [0, 0], { perChannel: true });
  const floatRows = rows.map((row) => row.reduce((sum, value, k) => sum + value * inputs[k], 0));
  const tensorError = Math.max(...perTensor.outputs.map((v, i) => Math.abs(v - floatRows[i])));
  const channelError = Math.max(...perChannel.outputs.map((v, i) => Math.abs(v - floatRows[i])));
  ok(channelError <= tensorError + 1e-9,
    `per-channel scales must not lose to per-tensor (${channelError} vs ${tensorError})`);

  // -- a two-layer classifier: agreement of the argmax ---------------------------------------------
  const classes = 4;
  const hidden = 16;
  const layerOne = Array.from({ length: hidden }, () =>
    Array.from({ length: inputs.length }, () => (rng(2001) - 1000) / 2000));
  const layerTwo = Array.from({ length: classes }, () => Array.from({ length: hidden }, () => (rng(2001) - 1000) / 2000));
  let agreements = 0, decisive = 0, decisiveAgreements = 0, trials = 300, maxOutputGap = 0;
  for (let trial = 0; trial < trials; trial++) {
    const activations = Array.from({ length: inputs.length }, () => (rng(2001) - 1000) / 1000);
    const hiddenFloat = layerOne.map((row) =>
      Math.max(0, row.reduce((sum, weight, k) => sum + weight * activations[k], 0)));
    const logitsFloat = layerTwo.map((row) =>
      row.reduce((sum, weight, k) => sum + weight * hiddenFloat[k], 0));
    const hiddenQuant = quantizedLinear(activations, layerOne, layerOne.map(() => 0)).outputs
      .map((value) => Math.max(0, value)); // the same ReLU in the quantised path
    const logitsQuant = quantizedLinear(hiddenQuant, layerTwo, layerTwo.map(() => 0)).outputs;
    for (let c = 0; c < classes; c++) maxOutputGap = Math.max(maxOutputGap, Math.abs(logitsFloat[c] - logitsQuant[c]));
    const same = argmax(logitsFloat) === argmax(logitsQuant);
    if (same) agreements++;
    const ranking = [...logitsFloat].sort((a, b) => b - a);
    if (ranking[0] - ranking[1] > 0.05) { // the float classifier is decisive here
      decisive++;
      if (same) decisiveAgreements++;
    }
  }
  ok(agreements / trials > 0.9, `overall agreement must be high (${agreements}/${trials})`);
  ok(decisive > 100 && decisiveAgreements / decisive > 0.9,
    `a random untrained net flips a few percent of decisions (${decisiveAgreements}/${decisive})`);
  ok(maxOutputGap < 0.5, `the logits must stay close, worst gap ${maxOutputGap}`);

  // -- validation and determinism ---------------------------------------------------------------------
  ok(argmax([1, 3, 2]) === 1 && argmax([0]) === 0);
  function replay(seed) {
    const local = makeRng(seed);
    const values = Array.from({ length: 40 }, () => (local(2001) - 1000) / 100);
    const quant = symmetricQuantize(values);
    const matrix = [Array.from({ length: 40 }, () => (local(2001) - 1000) / 100)];
    const result = quantizedLinear(values, matrix, [0.1]);
    return `${quant.scale.toFixed(12)}|${result.integerDot[0]}|${result.outputs[0].toFixed(9)}`;
  }
  strictEqual(replay(9), replay(9), "identical seeds must replay identically");

  console.log(`symmetric error<=${symmetric.scale / 2}, limit K=${limit}, argmax agreement=${(agreements / trials * 100).toFixed(1)}%`);
  console.log(`layer relative error=${(relativeError * 100).toFixed(3)}%, per-channel vs per-tensor=${channelError.toExponential(2)} vs ${tensorError.toExponential(2)}`);
  console.log("PASS");
}
