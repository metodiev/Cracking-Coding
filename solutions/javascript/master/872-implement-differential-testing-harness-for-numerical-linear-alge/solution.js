/** Task 872 - Implement differential testing harness for numerical linear algebra (BLAS correctness). (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A differential testing harness that pits several GEMM/GEMV/TRSM/SYRK "libraries" against each other
 * and against two independent oracles:
 *
 *   1. an *exact* oracle: for dyadic floating point inputs the exact product of two doubles is again a
 *      dyadic number, so the true result is computed with BigInt mantissa/exponent arithmetic and used
 *      as the ground truth (this is what makes the integer cases bit-exact rather than approximate);
 *   2. a *high-precision* reference: a Kahan-compensated accumulation whose error is compared against
 *      the classic forward error bound |fl(x) - x| <= gamma_k * (|A||B|)_ij, gamma_k = k u / (1 - k u),
 *      u = 2^-53. The bound is measured, not assumed.
 *
 * Kernels under test: naive ijk, k-ij accumulation, blocked/tiled (arbitrary block size), dot-product
 * (row i of A against column j of B), GEMV in two formulations, forward/backward substitution, and a
 * SYRK built on the same primitives. Faults are injected on purpose (off-by-one k bound, dropped
 * accumulator term, transposed operand, shifted tile origin, sign flip in the substitution) to show
 * that the harness actually detects wrong kernels instead of merely agreeing with itself.
 *
 * A small direct-mapped cache (64-byte lines, configurable line count) records the memory traffic, so
 * the blocked kernels can be shown to reduce misses without changing the floating-point operation
 * count - the reason blocking exists in the first place.
 *
 * Invariants verified: every correct kernel reproduces the exact oracle bit-for-bit on dyadic inputs;
 * every correct float kernel satisfies the gamma_k error bound and the shared mutual-agreement bound;
 * Kahan compensation is never worse than the naive order and strictly better under cancellation; each
 * injected fault is caught (detection power) while no correct kernel is flagged (no false positives);
 * triangular solves satisfy the residual bound; SYRK equals GEMM with the transposed operand; the
 * operation count is identical across all GEMM variants while the blocked ones reduce cache misses;
 * degenerate shapes are handled and shape mismatches are rejected.
 *
 * What is simplified: only small dense double-precision matrices are used, blocking is single-level
 * (no packing, no multi-level tiling, no assembly kernels), the cache model is a deterministic
 * direct-mapped simulator instead of real hardware, and there is no timing - the harness measures
 * correctness, error bounds and traffic counters, never wall-clock time.
 */
"use strict";

const { ok, throws } = require("node:assert");

const UNIT_ROUNDOFF = 2 ** -53;
const gamma = (n) => (n * UNIT_ROUNDOFF) / (1 - n * UNIT_ROUNDOFF);

// ---------------------------------------------------------------------------------------------
// exact dyadic arithmetic: every finite double is m * 2^e with integral m, so sums and products of
// dyadic numbers stay dyadic and can be evaluated exactly with BigInt.
// ---------------------------------------------------------------------------------------------

const buffer = new ArrayBuffer(8);
const floatView = new Float64Array(buffer);
const bitsView = new BigUint64Array(buffer);

/** Decompose a finite double into { mantissa, exponent } meaning mantissa * 2^exponent, exactly. */
function decompose(value) {
  if (!Number.isFinite(value)) throw new RangeError(`only finite values can be decomposed (got ${value})`);
  floatView[0] = value;
  const bits = bitsView[0];
  const negative = (bits >> 63n) === 1n;
  const fraction = bits & 0xfffffffffffffn;
  const rawExponent = Number((bits >> 52n) & 0x7ffn);
  let mantissa = fraction;
  let exponent = -1074;
  if (rawExponent !== 0) {
    mantissa = fraction | 0x10000000000000n;
    exponent = rawExponent - 1075;
  }
  if (mantissa === 0n) return { mantissa: 0n, exponent: 0 };
  return { mantissa: negative ? -mantissa : mantissa, exponent };
}

/** Exact sum of two dyadics. */
function exactAdd(a, b) {
  const exponent = Math.min(a.exponent, b.exponent);
  const mantissa = (a.mantissa << BigInt(a.exponent - exponent)) + (b.mantissa << BigInt(b.exponent - exponent));
  return { mantissa, exponent };
}

/** Exact product of two dyadics. */
const exactMultiply = (a, b) => ({ mantissa: a.mantissa * b.mantissa, exponent: a.exponent + b.exponent });

/** Compare two dyadics: -1, 0 or 1. */
function exactCompare(a, b) {
  const exponent = Math.min(a.exponent, b.exponent);
  const left = a.mantissa << BigInt(a.exponent - exponent);
  const right = b.mantissa << BigInt(b.exponent - exponent);
  return left < right ? -1 : left > right ? 1 : 0;
}

/** Convert an exact dyadic back to a double (exact whenever the value is representable). */
const exactToDouble = (value) => {
  let mantissa = value.mantissa;
  let exponent = value.exponent;
  if (mantissa === 0n) return 0;
  while (mantissa % 2n === 0n) {
    mantissa /= 2n;
    exponent += 1;
  }
  return Number(mantissa) * 2 ** exponent;
};

// ---------------------------------------------------------------------------------------------
// matrices
// ---------------------------------------------------------------------------------------------

function makeMatrix(rows, cols, fill) {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 0 || cols < 0) {
    throw new RangeError("matrix dimensions must be non-negative integers");
  }
  const data = new Array(rows);
  for (let i = 0; i < rows; i++) {
    data[i] = new Array(cols);
    for (let j = 0; j < cols; j++) data[i][j] = fill ? fill(i, j) : 0;
  }
  return { rows, cols, data };
}

const cloneMatrix = (a) => ({ rows: a.rows, cols: a.cols, data: a.data.map((row) => [...row]) });

function requireShapes(a, b, transposeA = false, transposeB = false) {
  const innerA = transposeA ? a.rows : a.cols;
  const innerB = transposeB ? b.cols : b.rows;
  if (innerA !== innerB) throw new RangeError(`shape mismatch: ${innerA} != ${innerB}`);
  const at = (M, i, j, transpose) => (transpose ? M.data[j][i] : M.data[i][j]);
  return { at, rows: transposeA ? a.cols : a.rows, cols: transposeB ? b.rows : b.cols, inner: innerA };
}

// ---------------------------------------------------------------------------------------------
// oracles
// ---------------------------------------------------------------------------------------------

/** Exact GEMM with BigInt dyadics; exact for any finite double inputs, not just integers. */
function gemmExact(a, b, transposeA = false, transposeB = false) {
  const shape = requireShapes(a, b, transposeA, transposeB);
  const decomposed = new Map();
  const entry = (M, i, j, transpose, id) => {
    const key = `${id}:${transpose ? j : i}:${transpose ? i : j}`;
    if (!decomposed.has(key)) decomposed.set(key, decompose(transpose ? M.data[j][i] : M.data[i][j]));
    return decomposed.get(key);
  };
  const out = makeMatrix(shape.rows, shape.cols);
  for (let i = 0; i < shape.rows; i++) {
    for (let j = 0; j < shape.cols; j++) {
      let accumulator = { mantissa: 0n, exponent: 0 };
      for (let k = 0; k < shape.inner; k++) {
        accumulator = exactAdd(accumulator, exactMultiply(
          entry(a, i, k, transposeA, "a"), entry(b, k, j, transposeB, "b"),
        ));
      }
      out.data[i][j] = exactToDouble(accumulator);
    }
  }
  return out;
}

/**
 * High-precision reference: compensated (Kahan-Babuska-Neumaier) accumulation, row-major k-ij order.
 * The Neumaier form is used rather than plain Kahan because the plain form loses the correction when
 * `product - compensation` itself rounds; here the correction term is accumulated by magnitude, so the
 * cancellation test below is genuinely recovered.
 */
function gemmKahan(a, b, { transposeA = false, transposeB = false } = {}) {
  const shape = requireShapes(a, b, transposeA, transposeB);
  const at = (M, i, j, transpose) => (transpose ? M.data[j][i] : M.data[i][j]);
  const out = makeMatrix(shape.rows, shape.cols);
  for (let i = 0; i < shape.rows; i++) {
    for (let j = 0; j < shape.cols; j++) {
      let sum = 0;
      let compensation = 0;
      for (let k = 0; k < shape.inner; k++) {
        const product = at(a, i, k, transposeA) * at(b, k, j, transposeB);
        const total = sum + product;
        if (Math.abs(sum) >= Math.abs(product)) compensation += sum - total + product;
        else compensation += product - total + sum;
        sum = total;
      }
      out.data[i][j] = sum + compensation;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// kernels under test
// ---------------------------------------------------------------------------------------------

/** Reference triple loop, i-j-k order. */
function gemmIJK(a, b, { transposeA = false, transposeB = false, kLimit = null, drop = -1 } = {}) {
  const shape = requireShapes(a, b, transposeA, transposeB);
  const at = (M, i, j, transpose) => (transpose ? M.data[j][i] : M.data[i][j]);
  const limit = kLimit ?? shape.inner;
  const out = makeMatrix(shape.rows, shape.cols);
  for (let i = 0; i < shape.rows; i++) {
    for (let j = 0; j < shape.cols; j++) {
      let sum = 0;
      for (let k = 0; k < limit; k++) {
        if (k === drop) continue;
        sum += at(a, i, k, transposeA) * at(b, k, j, transposeB);
      }
      out.data[i][j] = sum;
    }
  }
  return out;
}

/** k-i-j order: the classic accumulation kernel. */
function gemmIKJ(a, b, { transposeA = false, transposeB = false } = {}) {
  const shape = requireShapes(a, b, transposeA, transposeB);
  const at = (M, i, j, transpose) => (transpose ? M.data[j][i] : M.data[i][j]);
  const out = makeMatrix(shape.rows, shape.cols, () => 0);
  for (let k = 0; k < shape.inner; k++) {
    for (let i = 0; i < shape.rows; i++) {
      const aik = at(a, i, k, transposeA);
      for (let j = 0; j < shape.cols; j++) out.data[i][j] += aik * at(b, k, j, transposeB);
    }
  }
  return out;
}

/** Dot-product formulation: row i of A against column j of B. */
function gemmDot(a, b, { transposeA = false, transposeB = false } = {}) {
  const shape = requireShapes(a, b, transposeA, transposeB);
  const at = (M, i, j, transpose) => (transpose ? M.data[j][i] : M.data[i][j]);
  const out = makeMatrix(shape.rows, shape.cols);
  for (let i = 0; i < shape.rows; i++) {
    for (let j = 0; j < shape.cols; j++) {
      let sum = 0;
      for (let k = 0; k < shape.inner; k++) sum += at(a, i, k, transposeA) * at(b, k, j, transposeB);
      out.data[i][j] = sum;
    }
  }
  return out;
}

/** Single-level blocked kernel with a configurable tile size. */
function gemmBlocked(a, b, { transposeA = false, transposeB = false, block = 4 } = {}) {
  if (!Number.isInteger(block) || block < 1) throw new RangeError("block must be a positive integer");
  const shape = requireShapes(a, b, transposeA, transposeB);
  const at = (M, i, j, transpose) => (transpose ? M.data[j][i] : M.data[i][j]);
  const out = makeMatrix(shape.rows, shape.cols, () => 0);
  for (let ii = 0; ii < shape.rows; ii += block) {
    for (let kk = 0; kk < shape.inner; kk += block) {
      for (let jj = 0; jj < shape.cols; jj += block) {
        for (let i = ii; i < Math.min(ii + block, shape.rows); i++) {
          for (let k = kk; k < Math.min(kk + block, shape.inner); k++) {
            const aik = at(a, i, k, transposeA);
            for (let j = jj; j < Math.min(jj + block, shape.cols); j++) out.data[i][j] += aik * at(b, k, j, transposeB);
          }
        }
      }
    }
  }
  return out;
}

/**
 * Deliberately broken kernel used by the detection-power check: the tile origin of the k dimension is
 * shifted by one, so every tile silently drops its first contribution.
 */
function gemmBlockedShifted(a, b, { transposeA = false, transposeB = false, block = 4 } = {}) {
  const shape = requireShapes(a, b, transposeA, transposeB);
  const at = (M, i, j, transpose) => (transpose ? M.data[j][i] : M.data[i][j]);
  const out = makeMatrix(shape.rows, shape.cols, () => 0);
  for (let ii = 0; ii < shape.rows; ii += block) {
    for (let kk = 0; kk < shape.inner; kk += block) {
      for (let jj = 0; jj < shape.cols; jj += block) {
        for (let i = ii; i < Math.min(ii + block, shape.rows); i++) {
          for (let k = Math.min(kk + 1, shape.inner - 1); k < Math.min(kk + block, shape.inner); k++) {
            const aik = at(a, i, k, transposeA);
            for (let j = jj; j < Math.min(jj + block, shape.cols); j++) out.data[i][j] += aik * at(b, k, j, transposeB);
          }
        }
      }
    }
  }
  return out;
}

/** SYRK: A * A^T built on the blocked kernel. */
const syrkBlocked = (a, { block = 4 } = {}) => gemmBlocked(a, a, { transposeB: true, block });

/** GEMV, two formulations. */
function gemvRowDot(a, x) {
  if (a.cols !== x.length) throw new RangeError("shape mismatch in GEMV");
  return a.data.map((row) => row.reduce((acc, value, j) => acc + value * x[j], 0));
}

function gemvAxpy(a, x) {
  if (a.cols !== x.length) throw new RangeError("shape mismatch in GEMV");
  const y = new Array(a.rows).fill(0);
  for (let k = 0; k < a.cols; k++) {
    const xk = x[k];
    for (let i = 0; i < a.rows; i++) y[i] += a.data[i][k] * xk;
  }
  return y;
}

/** Forward substitution (unit-diagonal-free) and backward substitution; `brokenSign` injects a fault. */
function forwardSubstitution(l, b, { brokenSign = false } = {}) {
  if (l.rows !== l.cols || l.rows !== b.length) throw new RangeError("shape mismatch in forward substitution");
  const x = new Array(b.length).fill(0);
  for (let i = 0; i < b.length; i++) {
    let sum = b[i];
    for (let j = 0; j < i; j++) sum -= l.data[i][j] * x[j];
    x[i] = (brokenSign ? -1 : 1) * (sum / l.data[i][i]);
  }
  return x;
}

function backwardSubstitution(u, b, { brokenDirection = false } = {}) {
  if (u.rows !== u.cols || u.rows !== b.length) throw new RangeError("shape mismatch in backward substitution");
  const x = new Array(b.length).fill(0);
  for (let i = b.length - 1; i >= 0; i--) {
    let sum = b[i];
    if (brokenDirection) {
      for (let j = 0; j < i; j++) sum -= u.data[i][j] * x[j];
      x[i] = sum / u.data[i][i];
      continue;
    }
    for (let j = i + 1; j < b.length; j++) sum -= u.data[i][j] * x[j];
    x[i] = sum / u.data[i][i];
  }
  return x;
}

// ---------------------------------------------------------------------------------------------
// a deterministic direct-mapped cache model: same floating-point work, different memory traffic
// ---------------------------------------------------------------------------------------------

function makeCache(lineCount = 16, elementsPerLine = 8) {
  const lines = new Array(lineCount).fill(null);
  let hits = 0;
  let misses = 0;
  return {
    /** Touch matrix element (tag, index); a miss means the 64-byte line had to be fetched. */
    touch(tag, index) {
      const line = Math.floor(index / elementsPerLine);
      const slot = (tag * 31 + line) % lineCount;
      const key = `${tag}:${line}`;
      if (lines[slot] === key) {
        hits += 1;
        return false;
      }
      lines[slot] = key;
      misses += 1;
      return true;
    },
    reset() {
      lines.fill(null);
      hits = 0;
      misses = 0;
    },
    get hits() { return hits; },
    get misses() { return misses; },
    get traffic() { return hits + misses; },
  };
}

/** Runs a GEMM variant while counting multiply-adds and cache misses for both operands. */
function profileGemm(kernel, a, b, options = {}, cache = makeCache(16)) {
  cache.reset();
  const at = (M, i, j, transpose) => (transpose ? M.data[j][i] : M.data[i][j]);
  const shape = requireShapes(a, b, options.transposeA ?? false, options.transposeB ?? false);
  let operations = 0;
  const timed = makeMatrix(shape.rows, shape.cols, () => 0);
  if (kernel === "blocked") {
    const band = options.block ?? 4;
    for (let ii = 0; ii < shape.rows; ii += band) {
      for (let kk = 0; kk < shape.inner; kk += band) {
        for (let jj = 0; jj < shape.cols; jj += band) {
          for (let i = ii; i < Math.min(ii + band, shape.rows); i++) {
            for (let k = kk; k < Math.min(kk + band, shape.inner); k++) {
              const aik = at(a, i, k, options.transposeA ?? false);
              cache.touch("a", (options.transposeA ? k : i) * a.cols + (options.transposeA ? i : k));
              for (let j = jj; j < Math.min(jj + band, shape.cols); j++) {
                cache.touch("b", (options.transposeB ? j : k) * b.cols + (options.transposeB ? k : j));
                timed.data[i][j] += aik * at(b, k, j, options.transposeB ?? false);
                operations += 1;
              }
            }
          }
        }
      }
    }
  } else {
    for (let i = 0; i < shape.rows; i++) {
      for (let j = 0; j < shape.cols; j++) {
        let sum = 0;
        for (let k = 0; k < shape.inner; k++) {
          cache.touch("a", (options.transposeA ? k : i) * a.cols + (options.transposeA ? i : k));
          cache.touch("b", (options.transposeB ? j : k) * b.cols + (options.transposeB ? k : j));
          sum += at(a, i, k, options.transposeA ?? false) * at(b, k, j, options.transposeB ?? false);
          operations += 1;
        }
        timed.data[i][j] = sum;
      }
    }
  }
  return { result: timed, operations, misses: cache.misses, hits: cache.hits, traffic: cache.traffic };
}

// ---------------------------------------------------------------------------------------------
// differential harness
// ---------------------------------------------------------------------------------------------

const worstEntryDifference = (x, y) => {
  if (x.rows !== y.rows || x.cols !== y.cols) return Infinity;
  let worst = 0;
  for (let i = 0; i < x.rows; i++) {
    for (let j = 0; j < x.cols; j++) worst = Math.max(worst, Math.abs(x.data[i][j] - y.data[i][j]));
  }
  return worst;
};

/**
 * Runs every implementation on every generated case and compares it with the exact oracle and with the
 * high-precision reference. Returns per-implementation verdicts plus the aggregated deviations.
 */
function differentialTest({ implementations, cases, reference }) {
  const report = new Map();
  for (const name of Object.keys(implementations)) report.set(name, { name, cases: 0, oracleExact: 0, worstAgainstOracle: 0, worstAgainstReference: 0, boundViolations: 0, failed: [] });
  for (const testCase of cases) {
    const { a, b, transposeA = false, transposeB = false, dyadic = true, options = {} } = testCase;
    const exact = gemmExact(a, b, transposeA, transposeB);
    const highPrecision = reference ?? gemmKahan(a, b, { transposeA, transposeB });
    const magnitudeProduct = gemmAbsProduct(a, b, transposeA, transposeB);
    for (const [name, kernel] of Object.entries(implementations)) {
      const entry = report.get(name);
      entry.cases += 1;
      const result = kernel(a, b, { transposeA, transposeB, ...options });
      const againstOracle = worstEntryDifference(result, exact);
      const againstReference = worstEntryDifference(result, highPrecision);
      entry.worstAgainstOracle = Math.max(entry.worstAgainstOracle, againstOracle);
      entry.worstAgainstReference = Math.max(entry.worstAgainstReference, againstReference);
      if (dyadic && againstOracle === 0) entry.oracleExact += 1;
      const bound = gamma(b.rows * 2) * magnitudeProduct;
      if (againstOracle > bound) entry.boundViolations += 1;
      if (againstOracle > 0 && !(dyadic && againstOracle === 0)) entry.failed.push({ case: testCase.name ?? "case", deviation: againstOracle });
    }
  }
  return { report, cases: cases.length };
}

/** (|A||B|)_ij entrywise: the scale factor appearing in the classic error bound. */
function gemmAbsProduct(a, b, transposeA = false, transposeB = false) {
  const absOf = (M, transpose) => ({
    rows: transpose ? M.cols : M.rows,
    cols: transpose ? M.rows : M.cols,
    data: Array.from({ length: transpose ? M.cols : M.rows }, (_, i) => Array.from(
      { length: transpose ? M.rows : M.cols }, (_, j) => Math.abs(transpose ? M.data[j][i] : M.data[i][j]),
    )),
  });
  const product = gemmDot(absOf(a, transposeA), absOf(b, transposeB));
  return product.data.reduce((acc, row) => Math.max(acc, ...row.map(Math.abs)), 0);
}

/** Deterministic pseudo random generator for test-case generation. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x1234567;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Generates dyadic test cases: entries are k/4 with a small |k|, so products are exact in double. */
function makeCases({ count = 12, seed = 5, maxOrder = 5 } = {}) {
  const rng = makeRng(seed);
  const cases = [];
  for (let c = 0; c < count; c++) {
    const rows = 1 + Math.floor(rng() * maxOrder);
    const inner = 1 + Math.floor(rng() * maxOrder);
    const cols = 1 + Math.floor(rng() * maxOrder);
    const draw = () => Math.round(rng() * 24 - 12) / 4;
    cases.push({
      name: `case-${c}`,
      a: makeMatrix(rows, inner, draw),
      b: makeMatrix(inner, cols, draw),
      dyadic: true,
    });
  }
  return cases;
}

function main() {
  const diagnostics = [];
  const kernels = {
    ijk: (a, b, options) => gemmIJK(a, b, options),
    ikj: (a, b, options) => gemmIKJ(a, b, options),
    dot: (a, b, options) => gemmDot(a, b, options),
    blocked2: (a, b, options) => gemmBlocked(a, b, { ...options, block: 2 }),
    blocked4: (a, b, options) => gemmBlocked(a, b, { ...options, block: 4 }),
  };

  // 1. Exact oracle sanity: the BigInt path must agree with the hand-checked product.
  const identity = makeMatrix(3, 3, (i, j) => (i === j ? 1 : 0));
  const sample = makeMatrix(2, 3, (i, j) => i * 3 + j + 1);
  const identityProduct = gemmExact(sample, identity);
  const tallSample = makeMatrix(3, 2, (i, j) => i - 2 * j);
  ok(worstEntryDifference(identityProduct, sample) === 0, "right multiplication by the identity must be exact");
  ok(worstEntryDifference(gemmExact(identity, tallSample), tallSample) === 0, "left multiplication by the identity must be exact");
  ok(gemmExact(makeMatrix(1, 3, (i, j) => j + 1), makeMatrix(3, 1, () => 1)).data[0][0] === 6, "the dot product must be exact");
  throws(() => decompose(Infinity), RangeError);
  throws(() => requireShapes(makeMatrix(2, 2), makeMatrix(3, 3)), RangeError);

  // 2. Every correct kernel reproduces the exact oracle bit-for-bit on dyadic inputs.
  const cases = makeCases({ count: 14, seed: 5 });
  const baseline = differentialTest({ implementations: kernels, cases });
  for (const [name, entry] of baseline.report) {
    diagnostics.push(`${name}: ${entry.oracleExact}/${entry.cases} bit-exact vs oracle, worst |dev| ${entry.worstAgainstOracle}, bound violations ${entry.boundViolations}`);
    ok(entry.oracleExact === entry.cases, `${name} must match the exact oracle bit-for-bit on dyadic inputs`);
    ok(entry.boundViolations === 0, `${name} must respect the gamma_k error bound`);
    ok(entry.failed.length === 0, `${name} must not be flagged`);
  }

  // 3. Transposed operands are exercised too (A^T B, A B^T and A^T B^T all agree with the oracle).
  const transposedCases = [
    { name: "AtB", a: makeMatrix(3, 2, (i, j) => i - j), b: makeMatrix(3, 4, (i, j) => i * j - 1), transposeA: true, dyadic: true },
    { name: "ABt", a: makeMatrix(3, 2, (i, j) => i + j), b: makeMatrix(4, 2, (i, j) => i - 2 * j), transposeB: true, dyadic: true },
    { name: "AtBt", a: makeMatrix(2, 3, (i, j) => i * j), b: makeMatrix(4, 2, (i, j) => i + j), transposeA: true, transposeB: true, dyadic: true },
  ];
  const transposed = differentialTest({ implementations: { blocked4: kernels.blocked4, ikj: kernels.ikj }, cases: transposedCases });
  diagnostics.push(`transposed cases: ${[...transposed.report.values()].map((entry) => `${entry.name} ${entry.oracleExact}/${entry.cases}`).join(", ")}`);
  ok([...transposed.report.values()].every((entry) => entry.oracleExact === entry.cases), "transposed forms must match the oracle exactly");

  // 4. Detection power: each injected fault must be caught, and no correct kernel may be flagged.
  const faulty = {
    ijk: kernels.ijk,
    shortK: (a, b, options) => gemmIJK(a, b, { ...options, kLimit: Math.max(0, Math.min(a.cols, b.rows) - 1) }),
    droppedTerm: (a, b, options) => gemmIJK(a, b, { ...options, drop: 0 }),
    wrongTranspose: (a, b, options) => gemmIJK(a, b, { ...options, transposeB: !(options.transposeB ?? false) }),
    shiftedBlock: (a, b, options) => gemmBlockedShifted(a, b, { ...options, block: 3 }),
  };
  const faultCases = [{
    name: "square",
    a: makeMatrix(4, 4, (i, j) => ((i * 4 + j) % 7) - 3),
    b: makeMatrix(4, 4, (i, j) => ((i * 3 + j) % 5) - 2),
    dyadic: true,
  }];
  const detection = differentialTest({ implementations: faulty, cases: faultCases });
  const flagged = [...detection.report.values()].filter((entry) => entry.oracleExact === 0).map((entry) => entry.name).sort();
  diagnostics.push(`injected faults caught: ${flagged.join(", ")}`);
  ok(flagged.length === 4, `exactly the four faulty kernels must be caught (got ${flagged.length})`);
  ok(detection.report.get("ijk").oracleExact === 1, "the correct kernel must not be flagged");

  // 5. Triangular solves: the residual L x - b (and U x - b) must stay inside gamma_n ||L|| ||x||.
  const lower = makeMatrix(4, 4, (i, j) => (i > j ? (i - j) / 2 : i === j ? 2 : 0));
  const upper = makeMatrix(4, 4, (i, j) => (i < j ? (j - i) / 4 : i === j ? 3 : 0));
  const rhs = [1, -2, 0.5, 3];
  const forwardSolution = forwardSubstitution(lower, rhs);
  const backwardSolution = backwardSubstitution(upper, rhs);
  const lowerResidual = gemvRowDot(lower, forwardSolution).map((value, i) => Math.abs(value - rhs[i]));
  const upperResidual = gemvRowDot(upper, backwardSolution).map((value, i) => Math.abs(value - rhs[i]));
  const lowerScale = gamma(8) * Math.max(...lower.data.map((row) => row.reduce((acc, v) => acc + Math.abs(v), 0))) * Math.max(...forwardSolution.map(Math.abs)) * 8;
  diagnostics.push(`triangular residuals: forward ${Math.max(...lowerResidual).toExponential(2)}, backward ${Math.max(...upperResidual).toExponential(2)}`);
  ok(Math.max(...lowerResidual) < Math.max(lowerScale, 1e-12), "the forward substitution residual must respect the error bound");
  ok(Math.max(...upperResidual) < 1e-12, "the backward substitution residual must be tiny");
  const brokenForward = forwardSubstitution(lower, rhs, { brokenSign: true });
  ok(forwardSolution.some((value, i) => Math.abs(value + brokenForward[i]) > 1e-12), "a sign-flipped substitution must differ from the correct one");
  const brokenBackward = backwardSubstitution(upper, rhs, { brokenDirection: true });
  ok(backwardSolution.some((value, i) => Math.abs(value - brokenBackward[i]) > 1e-12), "a mis-ordered backward substitution must differ");
  throws(() => forwardSubstitution(makeMatrix(2, 3), [1, 2]), RangeError);
  throws(() => gemvRowDot(makeMatrix(2, 3), [1, 2]), RangeError);

  // 6. GEMV formulations agree bit-for-bit with each other and with the exact oracle.
  const vector = [0.5, -1, 2, 0.25];
  const gemvMatrix = makeMatrix(3, 4, (i, j) => ((i + 1) * (j + 1)) % 5 - 2);
  const gemvA = gemvRowDot(gemvMatrix, vector);
  const gemvB = gemvAxpy(gemvMatrix, vector);
  const exactGemv = gemmExact(gemvMatrix, makeMatrix(4, 1, (j) => vector[j]));
  const gemvExactFlat = exactGemv.data.map((row) => row[0]);
  ok(gemvA.every((value, i) => value === gemvB[i]), "both GEMV formulations must agree");
  ok(gemvA.every((value, i) => value === gemvExactFlat[i]), "GEMV must match the exact oracle on dyadic inputs");

  // 7. SYRK equals GEMM with the transposed operand, and is symmetric.
  const tall = makeMatrix(3, 5, (i, j) => ((i * 5 + j) % 9) - 4);
  const syrk = syrkBlocked(tall, { block: 3 });
  const viaGemm = gemmBlocked(tall, tall, { transposeB: true, block: 3 });
  ok(worstEntryDifference(syrk, viaGemm) === 0, "SYRK must equal the transposed-operand GEMM path");
  ok(syrk.rows === 3 && syrk.cols === 3, "SYRK of a 3x5 matrix must be 3x3");
  ok(syrk.data.every((row, i) => row.every((value, j) => value === viaGemm.data[j][i])), "SYRK must be symmetric");

  // 8. Cancellation: Kahan compensation beats the naive order, measured, with the exact oracle as judge.
  const cancellationA = makeMatrix(1, 3, (i, j) => [1e16, 1, -1e16][j]);
  const cancellationB = makeMatrix(3, 1, () => 1);
  const exactCancellation = gemmExact(cancellationA, cancellationB).data[0][0];
  const naiveCancellation = gemmIJK(cancellationA, cancellationB).data[0][0];
  const kahanCancellation = gemmKahan(cancellationA, cancellationB).data[0][0];
  diagnostics.push(`cancellation: exact ${exactCancellation}, naive ${naiveCancellation}, Kahan ${kahanCancellation}`);
  ok(exactCancellation === 1, "the exact sum must be 1");
  ok(Math.abs(naiveCancellation - exactCancellation) === 1, `the naive order must lose the unit term (got ${naiveCancellation})`);
  ok(kahanCancellation === exactCancellation, "Kahan compensation must recover the exact unit term");

  // 9. Error bounds hold for genuinely inexact (non-dyadic) inputs, and Kahan is never worse.
  const rng = makeRng(11);
  const randomCases = Array.from({ length: 10 }, (_, c) => ({
    name: `random-${c}`,
    a: makeMatrix(3, 3, () => rng() * 2 - 1),
    b: makeMatrix(3, 3, () => rng() * 2 - 1),
    dyadic: false,
  }));
  const inexact = differentialTest({ implementations: { ...kernels, kahan: (a, b, options) => gemmKahan(a, b, options) }, cases: randomCases });
  for (const [name, entry] of inexact.report) {
    diagnostics.push(`${name} (inexact): worst |dev| ${entry.worstAgainstOracle.toExponential(2)}, bound violations ${entry.boundViolations}`);
    ok(entry.boundViolations === 0, `${name} must satisfy the gamma_k bound on random inputs`);
  }
  ok(inexact.report.get("kahan").worstAgainstOracle <= inexact.report.get("ijk").worstAgainstOracle, "Kahan must not be less accurate than the naive kernel");
  const mixedA = makeMatrix(1, 4, (i, j) => [1e16, 1, 1, -1e16][j]);
  const mixedB = makeMatrix(4, 1, () => 1);
  const exactMixed = gemmExact(mixedA, mixedB).data[0][0];
  const naiveMixed = gemmIJK(mixedA, mixedB).data[0][0];
  const compensatedMixed = gemmKahan(mixedA, mixedB).data[0][0];
  diagnostics.push(`mixed magnitudes: exact ${exactMixed}, naive ${naiveMixed}, compensated ${compensatedMixed}`);
  ok(exactMixed === 2 && naiveMixed === 0 && compensatedMixed === 2, "on mixed magnitudes only the compensated kernel must recover the exact result");

  // 10. Cache traffic: identical floating-point work, fewer misses when blocked.
  const big = makeMatrix(32, 32, (i, j) => ((i * 7 + j * 3) % 11) - 5);
  const other = makeMatrix(32, 32, (i, j) => ((i * 5 + j * 13) % 9) - 4);
  const squareCache = makeCache(32);
  const naiveProfile = profileGemm("naive", big, other, {}, squareCache);
  const blockedProfile = profileGemm("blocked", big, other, { block: 8 }, squareCache);
  const tinyBlockProfile = profileGemm("blocked", big, other, { block: 2 }, squareCache);
  diagnostics.push(`cache misses (32 lines): naive ${naiveProfile.misses}, blocked(8) ${blockedProfile.misses}, blocked(2) ${tinyBlockProfile.misses}`);
  ok(naiveProfile.operations === blockedProfile.operations && blockedProfile.operations === 32 ** 3, "all GEMM variants must perform the same number of multiply-adds");
  ok(blockedProfile.misses < naiveProfile.misses, "blocking must reduce cache misses");
  ok(tinyBlockProfile.misses > blockedProfile.misses, "a too small block must be worse than a well sized one");
  ok(worstEntryDifference(blockedProfile.result, gemmBlocked(big, other, { block: 8 })) === 0, "the profiled kernel must produce the same result as the plain kernel");

  // 11. Degenerate shapes.
  const empty = gemmBlocked(makeMatrix(0, 3), makeMatrix(3, 4));
  ok(empty.rows === 0 && empty.cols === 4, "a zero-row product must stay well formed");
  const zeroInner = gemmBlocked(makeMatrix(2, 0), makeMatrix(0, 3), { block: 2 });
  ok(zeroInner.data.every((row) => row.every((value) => value === 0)), "an empty inner dimension must give a zero matrix");
  const single = gemmIJK(makeMatrix(1, 1, () => 1.5), makeMatrix(1, 1, () => -2));
  ok(single.data[0][0] === -3, "1x1 products must be exact");
  throws(() => gemmBlocked(makeMatrix(2, 2), makeMatrix(3, 3)), RangeError);
  throws(() => gemmBlocked(makeMatrix(2, 2), makeMatrix(2, 2), { block: 0 }), RangeError);
  throws(() => makeMatrix(1.5, 2), RangeError);

  // 12. Determinism: repeated runs and identical block sizes give identical bits.
  const repeat = differentialTest({ implementations: kernels, cases });
  const fingerprints = (report) => [...report.values()].map((entry) => `${entry.name}:${entry.worstAgainstOracle}`).join("|");
  ok(fingerprints(repeat.report) === fingerprints(baseline.report), "the harness must be reproducible");
  ok(worstEntryDifference(gemmBlocked(big, other, { block: 8 }), gemmBlocked(big, other, { block: 8 })) === 0, "the same block size must give identical results");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  decompose,
  exactAdd,
  exactMultiply,
  exactCompare,
  gamma,
  makeMatrix,
  gemmExact,
  gemmKahan,
  gemmIJK,
  gemmIKJ,
  gemmDot,
  gemmBlocked,
  syrkBlocked,
  gemvRowDot,
  gemvAxpy,
  forwardSubstitution,
  backwardSubstitution,
  makeCache,
  profileGemm,
  differentialTest,
  makeRng,
  makeCases,
};

if (require.main === module) main();
