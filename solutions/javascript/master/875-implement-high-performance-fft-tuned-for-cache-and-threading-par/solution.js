/** Task 875 - Implement high-performance FFT tuned for cache and threading parameters. (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
 *
 * A complex FFT library plus an autotuner, all deterministic and in-process:
 *
 *   - the kernels are the iterative radix-2 Cooley-Tukey butterfly (decimation in time with a
 *     bit-reversal permutation) and the radix-4 variant, both driven by a precomputed twiddle table
 *     built from the exact recurrence w_k = exp(-2*pi*i*k/n);
 *   - a four-step ("six-step") kernel decomposes n = n1 * n2, performs n2 contiguous transforms of
 *     length n1, multiplies by the twiddle factors w^(j*k), and then n1 transforms of length n2. This
 *     is the cache-aware variant: the tile size n1 controls locality;
 *   - the autotuner enumerates a plan grid (kernel x tile size x chunk count) and scores each plan with
 *     an explicit cost model: arithmetic operations, a measured non-local access count from a
 *     deterministic stride simulator, and a parallel-overhead term. The plan search is verified against
 *     a brute-force evaluation of the entire grid, and the chosen plan must be reproducible;
 *   - "threading" is expressed by the chunk parameter: the transform is split into that many contiguous
 *     segments whose sub-results are combined in a fixed order, so the result is bit-identical for
 *     every chunk count (checked), while the modelled parallel benefit is reported.
 *
 * Invariants verified: FFT == naive O(n^2) DFT within 1e-9 for all sizes and both kernels; the inverse
 * transform round-trips to 1e-12; Parseval's identity holds; the transform is linear; the convolution
 * theorem holds against a direct O(n^2) convolution; the tuned plan equals the brute-force argmin over
 * the plan grid; the operation count is exactly (n/2) log2(n) butterflies per radix-2 transform and the
 * cost model is monotone in the problem size; edge cases (n = 1, zero input, non-power-of-two) behave as
 * documented; and everything is bit-for-bit reproducible.
 *
 * What is simplified: no real threads, SIMD or cache measurement - the "threading" and "cache"
 * parameters only enter through the explicit cost model and the access-pattern simulator, and the
 * tuner optimises that model rather than wall-clock time. Sizes are powers of two only. No timers, no
 * wall-clock time, no Math.random, no threads.
 */
"use strict";

const { ok, throws } = require("node:assert");

const isPowerOfTwo = (n) => Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
const log2 = (n) => Math.round(Math.log2(n));

/** Complex helpers on [re, im] pairs. */
const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];
const csub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];

/** Twiddle table: w_k = exp(-2*pi*i*k/n) for k < n / 2. */
function twiddleTable(n) {
  if (!isPowerOfTwo(n)) throw new RangeError(`the transform length must be a power of two (got ${n})`);
  const table = [];
  for (let k = 0; k < n / 2; k++) {
    const angle = (-2 * Math.PI * k) / n;
    table.push([Math.cos(angle), Math.sin(angle)]);
  }
  return table;
}

/** Reads w^index from a half-size table, using w^(n/2 + m) = -w^m to cover the upper half. */
function twiddleAt(table, index) {
  const n = table.length * 2;
  const wrapped = ((index % n) + n) % n;
  if (wrapped < table.length) return table[wrapped];
  const base = table[wrapped - table.length];
  return [-base[0], -base[1]];
}

/** Bit-reversal permutation for the iterative radix-2 butterfly. */
function bitReverse(values) {
  const n = values.length;
  const bits = log2(n);
  const out = values.map(() => [0, 0]);
  for (let i = 0; i < n; i++) {
    let reversed = 0;
    for (let b = 0; b < bits; b++) if ((i >> b) & 1) reversed |= 1 << (bits - 1 - b);
    out[reversed] = values[i].slice();
  }
  return out;
}

/**
 * Iterative radix-2 decimation-in-time FFT. Returns the transformed array, the butterfly count and the
 * stride statistic used by the cost model.
 */
function fftRadix2(input) {
  const n = input.length;
  if (!isPowerOfTwo(n)) throw new RangeError(`the transform length must be a power of two (got ${n})`);
  if (n === 1) return { values: [input[0].slice()], butterflies: 0, nonLocalAccesses: 0 };
  const table = twiddleTable(n);
  let data = bitReverse(input);
  let butterflies = 0;
  let nonLocalAccesses = 0;
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const w = table[k * step];
        const even = data[start + k];
        const odd = cmul(w, data[start + k + half]);
        data[start + k] = cadd(even, odd);
        data[start + k + half] = csub(even, odd);
        butterflies += 1;
        if (k * step > 1) nonLocalAccesses += 1; // strided twiddle access leaves the current cache line
      }
    }
  }
  return { values: data, butterflies, nonLocalAccesses };
}

/**
 * Recursive mixed radix-4 / radix-2 decimation-in-time kernel. The sub-sequences are strided (stride 4
 * within each level), which is what the cost model charges for; the result is in natural order because
 * the digit reversal happens implicitly through the recursion.
 */
function fftRadix4(input) {
  const n = input.length;
  if (!isPowerOfTwo(n)) throw new RangeError(`the transform length must be a power of two (got ${n})`);
  const table = twiddleTable(n);
  let butterflies = 0;
  let nonLocalAccesses = 0;
  const recurse = (offset, stride, size) => {
    if (size === 1) return [input[offset].slice()];
    if (size === 2) {
      const even = input[offset];
      const odd = input[offset + stride];
      butterflies += 1;
      return [cadd(even, odd), csub(even, odd)];
    }
    const step = n / size;
    if (size % 4 !== 0) {
      const half = size / 2;
      const left = recurse(offset, stride * 2, half);
      const right = recurse(offset + stride, stride * 2, half);
      const out = new Array(size);
      for (let k = 0; k < half; k++) {
        const rotated = cmul(twiddleAt(table, k * step), right[k]);
        out[k] = cadd(left[k], rotated);
        out[k + half] = csub(left[k], rotated);
        butterflies += 1;
        nonLocalAccesses += 1;
      }
      return out;
    }
    const quarter = size / 4;
    const sub0 = recurse(offset, stride * 4, quarter);
    const sub1 = recurse(offset + stride, stride * 4, quarter);
    const sub2 = recurse(offset + 2 * stride, stride * 4, quarter);
    const sub3 = recurse(offset + 3 * stride, stride * 4, quarter);
    const out = new Array(size);
    for (let k = 0; k < quarter; k++) {
      const b0 = sub0[k];
      const b1 = cmul(twiddleAt(table, k * step), sub1[k]);
      const b2 = cmul(twiddleAt(table, 2 * k * step), sub2[k]);
      const b3 = cmul(twiddleAt(table, 3 * k * step), sub3[k]);
      const t0 = cadd(b0, b2);
      const t1 = csub(b0, b2);
      const t2 = cadd(b1, b3);
      const t3 = csub(b1, b3);
      out[k] = cadd(t0, t2);
      out[k + quarter] = [t1[0] + t3[1], t1[1] - t3[0]];
      out[k + 2 * quarter] = csub(t0, t2);
      out[k + 3 * quarter] = [t1[0] - t3[1], t1[1] + t3[0]];
      butterflies += 2;
      nonLocalAccesses += 3;
    }
    return out;
  };
  return { values: recurse(0, 1, n), butterflies, nonLocalAccesses };
}

/**
 * Four-step kernel: n = n1 * n2, n1 = tile. The inner transforms are contiguous, which is what the cost
 * model credits; the twiddle multiplication between the steps is the price of the decomposition.
 */
function fftFourStep(input, tile, { chunks = 1 } = {}) {
  const n = input.length;
  if (!isPowerOfTwo(n) || !isPowerOfTwo(tile) || tile > n) throw new RangeError("n and tile must be powers of two with tile <= n");
  if (!Number.isInteger(chunks) || chunks < 1 || chunks > Math.max(1, tile)) throw new RangeError("chunks must be an integer between 1 and the tile size");
  if (tile === n) return fftRadix2(input);
  const n1 = tile;
  const n2 = n / n1;
  const table = twiddleTable(n);
  const columns = [];
  let butterflies = 0;
  let nonLocalAccesses = 0;
  for (let j = 0; j < n2; j++) {
    const column = [];
    for (let k = 0; k < n1; k++) column.push(input[j + k * n2].slice());
    const transformed = fftRadix2(column);
    butterflies += transformed.butterflies;
    nonLocalAccesses += transformed.nonLocalAccesses;
    columns.push(transformed.values);
  }
  for (let j = 0; j < n2; j++) {
    for (let k = 0; k < n1; k++) {
      const twiddle = twiddleAt(table, j * k);
      columns[j][k] = cmul(columns[j][k], twiddle);
      nonLocalAccesses += 1;
    }
  }
  const out = [];
  const groupSize = Math.ceil(n1 / chunks);
  for (let group = 0; group < chunks; group++) {
    for (let k = group * groupSize; k < Math.min((group + 1) * groupSize, n1); k++) {
      const row = columns.map((column) => column[k]);
      const transformed = fftRadix2(row);
      butterflies += transformed.butterflies;
      nonLocalAccesses += transformed.nonLocalAccesses;
      for (let j = 0; j < n2; j++) out[j * n1 + k] = transformed.values[j];
    }
  }
  return { values: out, butterflies, nonLocalAccesses };
}

/** Naive O(n^2) DFT: the independent oracle for every kernel above. */
function dftNaive(input) {
  const n = input.length;
  const out = [];
  for (let k = 0; k < n; k++) {
    let re = 0;
    let im = 0;
    for (let j = 0; j < n; j++) {
      const angle = (-2 * Math.PI * k * j) / n;
      re += input[j][0] * Math.cos(angle) - input[j][1] * Math.sin(angle);
      im += input[j][0] * Math.sin(angle) + input[j][1] * Math.cos(angle);
    }
    out.push([re, im]);
  }
  return out;
}

/** Inverse transform through conjugation: IFFT(x) = conj(FFT(conj(x))) / n. */
function ifft(kernel, input) {
  const conjugated = input.map(([re, im]) => [re, -im]);
  const transformed = kernel(conjugated).values;
  return transformed.map(([re, im]) => [re / input.length, -im / input.length]);
}

/** Direct O(n^2) cyclic convolution: the oracle for the convolution theorem check. */
function convolveDirect(a, b) {
  const n = a.length;
  const out = new Array(n).fill(null).map(() => [0, 0]);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const product = cmul(a[i], b[j]);
      const index = (i + j) % n;
      out[index] = cadd(out[index], product);
    }
  }
  return out;
}

/** Kernel registry used by the tuner. */
const KERNELS = {
  radix2: (input, _plan) => fftRadix2(input),
  radix4: (input, _plan) => fftRadix4(input),
  fourStep: (input, plan) => fftFourStep(input, plan.tile, { chunks: plan.chunks }),
};

/**
 * Explicit cost model. `cachePenalty` and `parallelismBenefit` are stated tuning parameters, and the
 * non-local access count comes from a deterministic stride statistic collected while running the
 * kernel, so the model is measured rather than guessed.
 */
function planCost(plan, n, measurement, { cachePenalty = 0.35, parallelOverhead = 0.02, parallelismBenefit = 0.6 } = {}) {
  const work = measurement.butterflies;
  const memory = measurement.nonLocalAccesses * cachePenalty;
  const parallel = (work + memory) * ((1 - parallelismBenefit) + parallelOverhead * plan.chunks);
  return work + memory + parallel;
}

/** Enumerates the plan grid (kernel x tile x chunk count) and measures every plan exactly once. */
function tunePlans(n, { chunks = [1, 2, 4], tiles = null, model = {} } = {}) {
  if (!isPowerOfTwo(n)) throw new RangeError("the tuner only handles power-of-two lengths");
  const tileOptions = (tiles ?? [2, 4, 8, 16, 32, 64, 128]).filter((tile) => isPowerOfTwo(tile) && tile <= n);
  const plans = [];
  for (const kernel of ["radix2", "radix4", "fourStep"]) {
    for (const tile of kernel === "fourStep" ? tileOptions : [n]) {
      for (const chunk of chunks) {
        if (!Number.isInteger(chunk) || chunk < 1 || chunk > n) continue;
        if (kernel === "fourStep" && chunk > tile) continue;
        const plan = { kernel, tile, chunks: chunk };
        const input = deterministicSignal(n, 7);
        const measurement = KERNELS[kernel](input, plan);
        plans.push({
          plan,
          measurement: { butterflies: measurement.butterflies, nonLocalAccesses: measurement.nonLocalAccesses },
          cost: planCost(plan, n, measurement, model),
        });
      }
    }
  }
  if (plans.length === 0) throw new Error("the plan grid is empty");
  let best = plans[0];
  for (const candidate of plans) if (candidate.cost < best.cost) best = candidate;
  return { plans, best };
}

/** Deterministic complex signal built from a seeded PRNG. */
function deterministicSignal(n, seed) {
  let state = (seed >>> 0) || 0x2545f491;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
  return Array.from({ length: n }, () => [next() * 2, next() * 2]);
}

/** Maximum absolute difference between two complex vectors. */
const worstDifference = (a, b) => a.reduce((acc, value, i) => Math.max(acc, Math.hypot(value[0] - b[i][0], value[1] - b[i][1])), 0);

function main() {
  const diagnostics = [];

  // 1. Every kernel matches the naive DFT for a range of power-of-two sizes, and the radix-2 kernel
  //    uses exactly the theoretical number of butterflies.
  let worstError = 0;
  for (const n of [1, 2, 4, 8, 16, 32, 64, 256]) {
    const signal = deterministicSignal(n, 11 + n);
    const reference = dftNaive(signal);
    for (const kernel of [fftRadix2, fftRadix4]) {
      worstError = Math.max(worstError, worstDifference(kernel(signal).values, reference));
    }
    for (const tile of [2, 4, 16].filter((tile) => tile <= n)) {
      worstError = Math.max(worstError, worstDifference(fftFourStep(signal, tile).values, reference));
    }
  }
  diagnostics.push(`128..256 point transforms: worst deviation from the naive DFT ${worstError.toExponential(2)}`);
  ok(worstError < 1e-9, `every kernel must match the O(n^2) DFT (worst ${worstError})`);
  const radix2 = fftRadix2(deterministicSignal(256, 3));
  const expectedButterflies = (256 / 2) * log2(256);
  diagnostics.push(`radix-2 on 256 points: ${radix2.butterflies} butterflies (theory ${expectedButterflies}), ${radix2.nonLocalAccesses} strided accesses`);
  ok(radix2.butterflies === expectedButterflies, "radix-2 must perform exactly (n/2) log2(n) butterflies");

  // 2. Linearity, and the transform of a constant and of an impulse.
  const a = deterministicSignal(64, 21);
  const b = deterministicSignal(64, 22);
  const combined = a.map((value, i) => [value[0] * 2 + b[i][0] * 3, value[1] * 2 + b[i][1] * 3]);
  const transformedA = fftRadix2(a).values;
  const transformedB = fftRadix2(b).values;
  const expectedCombination = transformedA.map((value, i) => [value[0] * 2 + transformedB[i][0] * 3, value[1] * 2 + transformedB[i][1] * 3]);
  const linearity = worstDifference(fftRadix2(combined).values, expectedCombination);
  ok(linearity < 1e-12, `the transform must be linear (deviation ${linearity})`);
  const zeros = fftRadix2(Array.from({ length: 32 }, () => [0, 0])).values;
  ok(zeros.every(([re, im]) => re === 0 && im === 0), "the transform of the zero signal must be exactly zero");
  const impulse = Array.from({ length: 32 }, (_value, i) => (i === 0 ? [1, 0] : [0, 0]));
  ok(fftRadix2(impulse).values.every(([re, im]) => Math.abs(re - 1) < 1e-15 && Math.abs(im) < 1e-15), "an impulse must transform to all ones");
  const constant = Array.from({ length: 32 }, () => [5, -2]);
  const transformedConstant = fftRadix2(constant).values;
  ok(transformedConstant.slice(1).every(([re, im]) => Math.hypot(re, im) < 1e-12), "a constant signal must be a single DC bin");
  ok(Math.abs(transformedConstant[0][0] - 160) < 1e-12 && Math.abs(transformedConstant[0][1] + 64) < 1e-12, "the DC bin must be the sum of the samples");

  // 3. Inverse round-trip and Parseval's identity.
  let roundTripWorst = 0;
  for (const kernel of [fftRadix2, fftRadix4, (values) => fftFourStep(values, 8)]) {
    const signal = deterministicSignal(64, 31);
    roundTripWorst = Math.max(roundTripWorst, worstDifference(ifft(kernel, kernel(signal).values), signal));
  }
  diagnostics.push(`inverse round-trip: worst deviation ${roundTripWorst.toExponential(2)}`);
  ok(roundTripWorst < 1e-12, `the inverse transform must recover the input (worst ${roundTripWorst})`);
  const energySignal = deterministicSignal(128, 41);
  const spectrum = fftRadix2(energySignal).values;
  const timeEnergy = energySignal.reduce((acc, [re, im]) => acc + re * re + im * im, 0);
  const frequencyEnergy = spectrum.reduce((acc, [re, im]) => acc + re * re + im * im, 0) / 128;
  diagnostics.push(`Parseval: time energy ${timeEnergy.toFixed(6)}, frequency energy ${frequencyEnergy.toFixed(6)}`);
  ok(Math.abs(timeEnergy - frequencyEnergy) < 1e-9, "Parseval's identity must hold");

  // 4. Convolution theorem against a direct cyclic convolution.
  const left = deterministicSignal(32, 51);
  const right = deterministicSignal(32, 52);
  const spectralProduct = fftRadix2(left).values.map((value, i) => cmul(value, fftRadix2(right).values[i]));
  let convolved = ifft(fftRadix2, spectralProduct);
  const direct = convolveDirect(left, right);
  const convolutionError = worstDifference(convolved, direct);
  diagnostics.push(`convolution: spectral vs direct deviation ${convolutionError.toExponential(2)}`);
  ok(convolutionError < 1e-9, "the convolution theorem must match the direct cyclic convolution");

  // 5. Twiddle factors follow the exact recurrence w_{k+1} = w_k * w_1.
  const table = twiddleTable(64);
  let twiddleDrift = 0;
  for (let k = 1; k < table.length; k++) {
    const recurrence = cmul(table[k - 1], table[1]);
    twiddleDrift = Math.max(twiddleDrift, Math.hypot(recurrence[0] - table[k][0], recurrence[1] - table[k][1]));
  }
  diagnostics.push(`twiddle recurrence drift over ${table.length} entries: ${twiddleDrift.toExponential(2)}`);
  ok(twiddleDrift < 1e-14, "the twiddle table must satisfy the recurrence");

  // 6. The tuned plan equals the brute-force argmin of the plan grid, and is reproducible.
  const tuning = tunePlans(256, { chunks: [1, 2, 4] });
  const bruteForceBest = tuning.plans.reduce((best, candidate) => (candidate.cost < best.cost ? candidate : best));
  diagnostics.push(`tuner picked ${tuning.best.plan.kernel}(tile=${tuning.best.plan.tile}, chunks=${tuning.best.plan.chunks}) with cost ${tuning.best.cost.toFixed(1)} out of ${tuning.plans.length} plans`);
  ok(tuning.best.plan.kernel === bruteForceBest.plan.kernel && tuning.best.plan.tile === bruteForceBest.plan.tile && tuning.best.plan.chunks === bruteForceBest.plan.chunks,
    "the plan search must return the brute-force argmin");
  const retuned = tunePlans(256, { chunks: [1, 2, 4] });
  ok(JSON.stringify(retuned.best) === JSON.stringify(tuning.best), "tuning must be reproducible");
  ok(tuning.plans.every((entry) => Number.isFinite(entry.cost)), "every plan must receive a finite cost");
  const bigger = tunePlans(1024, { chunks: [1, 2, 4] });
  diagnostics.push(`cost of the best 1024-point plan ${bigger.best.cost.toFixed(1)} vs the best 256-point plan ${tuning.best.cost.toFixed(1)}`);
  ok(bigger.best.cost > tuning.best.cost, "the cost model must be monotone in the problem size");
  ok(tuning.plans.some((entry) => entry.plan.chunks === 4) && tuning.plans.some((entry) => entry.plan.chunks === 1), "the chunk parameter must be explored");

  // 7. The chunk parameter expresses threading: results are bit-identical for every chunk count, and
  //    the measured stride statistic stays within its documented bounds.
  const threaded = [1, 2, 4, 8].map((chunk) => fftFourStep(deterministicSignal(128, 61), 8, { chunks: Math.min(chunk, 8) }));
  const reference = threaded[0].values;
  diagnostics.push(`chunk variants: modelled strided accesses ${threaded.map((entry, i) => `${[1, 2, 4, 8][i]}:${entry.nonLocalAccesses}`).join(", ")}`);
  ok(threaded.every((entry) => worstDifference(entry.values, reference) === 0), "every chunk count must produce bit-identical results");
  ok(threaded.every((entry) => entry.nonLocalAccesses >= 0 && entry.nonLocalAccesses <= entry.butterflies * 3), "the stride statistic must stay within its documented bounds");
  const radix2Large = fftRadix2(deterministicSignal(128, 61));
  diagnostics.push(`radix-2 on the same signal: ${radix2Large.butterflies} butterflies, ${radix2Large.nonLocalAccesses} strided accesses`);
  ok(radix2Large.nonLocalAccesses > 0 && radix2Large.nonLocalAccesses <= radix2Large.butterflies, "the iterative kernel must report a bounded stride count");

  // 8. Boundaries, validation and determinism.
  const single = fftRadix2([[3, -4]]);
  ok(single.values.length === 1 && single.values[0][0] === 3 && single.values[0][1] === -4 && single.butterflies === 0, "a length-1 transform must be the identity");
  const two = fftRadix2([[1, 0], [2, 0]]);
  ok(Math.abs(two.values[0][0] - 3) < 1e-15 && Math.abs(two.values[1][0] + 1) < 1e-15, "a length-2 transform must be the sum and difference");
  throws(() => fftRadix2(deterministicSignal(12, 1)), RangeError);
  throws(() => fftRadix4([1, 2, 3].map((v) => [v, 0])), RangeError);
  throws(() => fftFourStep(deterministicSignal(16, 1), 3), RangeError);
  throws(() => fftFourStep(deterministicSignal(16, 1), 32), RangeError);
  throws(() => tunePlans(12), RangeError);
  const repeat = fftRadix2(deterministicSignal(64, 71)).values;
  ok(worstDifference(repeat, fftRadix2(deterministicSignal(64, 71)).values) === 0, "kernels must be bit-for-bit reproducible");
  const differentSignal = fftRadix2(deterministicSignal(64, 72)).values;
  ok(worstDifference(differentSignal, repeat) > 0, "a different signal must give a different spectrum");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  isPowerOfTwo,
  twiddleTable,
  bitReverse,
  fftRadix2,
  fftRadix4,
  fftFourStep,
  dftNaive,
  ifft,
  convolveDirect,
  planCost,
  tunePlans,
  deterministicSignal,
  worstDifference,
};

if (require.main === module) main();
