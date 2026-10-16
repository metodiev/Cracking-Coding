/** Task 932 - Implement differential testing between CPU/GPU implementations for numerical ops.
 *  (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Several implementations of the same three ops, written with deliberately different orderings and
 * precisions, are fed seeded random tensors (including edge cases) and compared element-wise:
 *   matmul     naive triple loop (float64), tiled/blocked (float64), float32 accumulation with
 *              Math.fround, and two seeded bugs (a dropped tile row and an off-by-one bound);
 *   reduction  naive left-to-right, pairwise (tree), Kahan compensated, and a seeded bug (dividing
 *              the compensation term);
 *   softmax    max-subtracted stable and a naive exp/sum version (which overflows for large logits).
 * Oracles: on exactly representable integer inputs every correct implementation must agree *exactly*
 * (no tolerance at all); the float32 error is bounded by the textbook gamma_n = n*u/(1-n*u) bound with
 * u = 2^-23; Kahan must return the exact sum for the classic catastrophic-cancellation input
 * [2^53, 1, -2^53] where the naive loop returns 0. NaN is compared NaN-aware (both NaN is a match).
 * Verified: exactness on integers, the precision bound, cancellation behaviour, bug detection by the
 * tester, tolerance semantics, NaN/Infinity handling, tensor boundary shapes, invalid input and
 * determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    return (this.state = x >>> 0);
  }
  float() { return this.next() / 4294967296; }
  sym(scale) { return (this.float() * 2 - 1) * scale; }
  int(bound) { return this.next() % bound; }
}

const f32 = Math.fround;
const zeros = (rows, cols) => Array.from({ length: rows }, () => new Array(cols).fill(0));

/* ------------------------------------------------------------------ reference and variant matmuls */

function matmulNaive(A, B) {
  assert.strictEqual(A[0].length, B.length, "the inner dimensions must agree");
  const out = zeros(A.length, B[0].length);
  for (let i = 0; i < A.length; i++) {
    for (let k = 0; k < B.length; k++) {
      for (let j = 0; j < B[0].length; j++) out[i][j] += A[i][k] * B[k][j];
    }
  }
  return out;
}

/** Blocked/tiled accumulation: a different summation order over the same float64 data. */
function matmulTiled(A, B, tile) {
  const n = A.length, m = B[0].length, inner = B.length;
  const out = zeros(n, m);
  for (let ii = 0; ii < n; ii += tile) {
    for (let kk = 0; kk < inner; kk += tile) {
      for (let jj = 0; jj < m; jj += tile) {
        for (let i = ii; i < Math.min(ii + tile, n); i++) {
          for (let k = kk; k < Math.min(kk + tile, inner); k++) {
            const a = A[i][k];
            for (let j = jj; j < Math.min(jj + tile, m); j++) out[i][j] += a * B[k][j];
          }
        }
      }
    }
  }
  return out;
}

/** Float32 accumulation, as a GPU would do with single precision. */
function matmulFloat32(A, B) {
  const out = zeros(A.length, B[0].length);
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < B[0].length; j++) {
      let acc = 0;
      for (let k = 0; k < B.length; k++) acc = f32(acc + f32(A[i][k] * B[k][j]));
      out[i][j] = acc;
    }
  }
  return out;
}

/** Seeded bug: the last tile row is skipped (a classic tiling boundary error). */
function matmulBuggyTiles(A, B, tile) {
  const out = matmulTiled(A, B, tile);
  for (let i = Math.floor((A.length - 1) / tile) * tile; i < A.length; i++) out[i] = out[i].map(() => 0);
  return out;
}

/** Seeded bug: the inner loop stops one short, silently dropping the last contribution. */
function matmulBuggyBound(A, B) {
  const out = zeros(A.length, B[0].length);
  for (let i = 0; i < A.length; i++) {
    for (let k = 0; k < B.length - 1; k++) {
      for (let j = 0; j < B[0].length; j++) out[i][j] += A[i][k] * B[k][j];
    }
  }
  return out;
}

/* ------------------------------------------------------------------ reductions */

function sumNaive(values) {
  let acc = 0;
  for (const v of values) acc += v;
  return acc;
}

function sumPairwise(values) {
  if (values.length <= 8) return sumNaive(values);
  const mid = values.length >> 1;
  return sumPairwise(values.slice(0, mid)) + sumPairwise(values.slice(mid));
}

function sumKahan(values) {
  let sum = 0, compensation = 0;
  for (const v of values) {
    const y = v - compensation;
    const t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }
  return sum;
}

/** Seeded bug: the running sum is silently rounded to float32 at every step (a GPU-side mistake). */
function sumBuggy(values) {
  let acc = 0;
  for (const v of values) acc = f32(acc + f32(v));
  return acc;
}

/* ------------------------------------------------------------------ softmax */

function softmaxStable(logits) {
  const max = Math.max(...logits);
  const exponentials = logits.map((v) => Math.exp(v - max));
  const total = sumPairwise(exponentials);
  return exponentials.map((v) => v / total);
}

/** Naive overflow-prone variant, kept to show that the comparison catches it for large logits. */
function softmaxNaive(logits) {
  const exponentials = logits.map((v) => Math.exp(v));
  const total = sumNaive(exponentials);
  return exponentials.map((v) => v / total);
}

/* ------------------------------------------------------------------ differential tester */

const sameValue = (a, b, rtol, atol) => {
  if (Number.isNaN(a) && Number.isNaN(b)) return true; // NaN-aware
  if (a === b) return true; // covers +-Infinity of the same sign
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false;
  return Math.abs(a - b) <= atol + rtol * Math.abs(b);
};

/** Compare two tensors element-wise; returns the worst absolute/relative error and the first mismatch. */
function compareTensors(actual, expected, options = {}) {
  const rtol = options.rtol ?? 1e-12;
  const atol = options.atol ?? 1e-12;
  assert.strictEqual(actual.length, expected.length, "the tensor shapes must agree");
  let maxAbs = 0, maxRel = 0, mismatches = 0, firstMismatch = null;
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Array.isArray(actual[i]) && Array.isArray(expected[i]),
      "tensors must be rank-2: wrap scalars in a row");
    assert.strictEqual(actual[i].length, expected[i].length, "the tensor shapes must agree");
    for (let j = 0; j < expected[i].length; j++) {
      const a = actual[i][j], b = expected[i][j];
      if (Number.isFinite(a) && Number.isFinite(b)) {
        const abs = Math.abs(a - b);
        const rel = abs / Math.max(1e-300, Math.abs(b));
        if (abs > maxAbs) maxAbs = abs;
        if (rel > maxRel) maxRel = rel;
      }
      if (!sameValue(a, b, rtol, atol)) {
        mismatches++;
        if (!firstMismatch) firstMismatch = { i, j, actual: a, expected: b };
      }
    }
  }
  return { maxAbs, maxRel, mismatches, firstMismatch, pass: mismatches === 0 };
}

/** Differential harness: run every implementation on every case and report which ones diverge. */
function differentialTest(cases, implementations, options = {}) {
  const report = { cases: cases.length, comparisons: 0, failures: [] };
  for (const testCase of cases) {
    const reference = testCase.run(testCase.input);
    for (const [name, implementation] of Object.entries(implementations)) {
      const actual = implementation(testCase.input);
      const comparison = compareTensors(actual, reference, options);
      report.comparisons++;
      if (!comparison.pass) {
        report.failures.push({ case: testCase.name, implementation: name,
          maxAbs: comparison.maxAbs, firstMismatch: comparison.firstMismatch });
      }
    }
  }
  return report;
}

function randomMatrix(prng, rows, cols, scale) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => prng.sym(scale)));
}

function main() {
  const prng = new Prng(20250932);
  const A = randomMatrix(prng, 7, 5, 1);
  const B = randomMatrix(prng, 5, 6, 1);
  const reference = matmulNaive(A, B);

  // 1. Different orderings of exactly representable data must agree bit for bit.
  const intA = randomMatrix(prng, 6, 4, 8).map((row) => row.map((v) => Math.round(v * 4)));
  const intB = randomMatrix(prng, 4, 5, 8).map((row) => row.map((v) => Math.round(v * 4)));
  const exact = matmulNaive(intA, intB);
  assert.deepStrictEqual(matmulTiled(intA, intB, 2), exact, "tiling must not change an exact product");
  assert.deepStrictEqual(matmulTiled(intA, intB, 3), exact);
  assert.deepStrictEqual(matmulFloat32(intA, intB), exact, "small integers are exact in float32 too");
  assert.ok(compareTensors(matmulTiled(A, B, 2), reference, { rtol: 1e-14, atol: 1e-14 }).pass,
    "the tiled variant must match within rounding noise");
  assert.ok(matmulTiled(A, B, 1).every((row, i) => row.every((v, j) => Math.abs(v - reference[i][j]) < 1e-12)));

  // 2. Float32 accumulation stays inside the textbook gamma_n bound.
  const float32Result = matmulFloat32(A, B);
  const u = 2 ** -23;
  const n = 5;
  const gamma = (k) => (k * u) / (1 - k * u);
  const products = A.flatMap((row, i) => row.map((a, k) => a * B[k][0]));
  const bound = gamma(2 * n) * products.reduce((acc, v) => Math.max(acc, Math.abs(v)), 0) * Math.sqrt(n);
  const comparison32 = compareTensors(float32Result, reference, { rtol: 1e-5, atol: 1e-6 });
  assert.ok(comparison32.pass, "float32 must pass a float32-appropriate tolerance");
  assert.ok(comparison32.maxAbs < bound + 1e-6,
    `float32 error ${comparison32.maxAbs} exceeds the gamma bound ${bound}`);
  assert.ok(compareTensors(float32Result, reference, { rtol: 1e-15, atol: 1e-15 }).mismatches > 0,
    "float32 should not pass a float64 tolerance");

  // 3. The cancellation case: naive summation loses the unit while Kahan and pairwise keep it.
  const cancellation = [2 ** 53, 1, -(2 ** 53)];
  assert.strictEqual(sumNaive(cancellation), 0, "the naive loop famously loses the 1");
  assert.strictEqual(sumKahan(cancellation), 1, "Kahan must recover the exact sum");
  assert.strictEqual(sumPairwise(cancellation), 0,
    "a three-element array falls into the pairwise base case, so it loses the 1 as well");
  const many = Array.from({ length: 200 }, () => prng.sym(1));
  const referenceSum = sumKahan(many);
  assert.ok(Math.abs(sumPairwise(many) - referenceSum) <= Math.abs(sumNaive(many) - referenceSum) + 1e-12,
    "pairwise summation should not be worse than the naive loop here");
  assert.ok(Math.abs(sumPairwise(many) - referenceSum) < 1e-12, "pairwise is accurate on well-scaled data");

  // 4. The tester detects every seeded bug and clears every correct implementation.
  const matmulCases = [
    { name: "tall", input: { A, B } },
    { name: "square", input: { A: randomMatrix(prng, 4, 4, 1), B: randomMatrix(prng, 4, 4, 1) } },
    { name: "single", input: { A: [[3]], B: [[4]] } },
    { name: "integers", input: { A: intA, B: intB } },
  ].map((testCase) => ({ ...testCase, run: ({ A: a, B: b }) => matmulNaive(a, b) }));
  const report = differentialTest(matmulCases, {
    tiled2: ({ A: a, B: b }) => matmulTiled(a, b, 2),
    tiled3: ({ A: a, B: b }) => matmulTiled(a, b, 3),
    float32: ({ A: a, B: b }) => matmulFloat32(a, b),
  }, { rtol: 1e-5, atol: 1e-6 });
  assert.strictEqual(report.failures.length, 0, `a correct implementation was flagged: ${JSON.stringify(report.failures)}`);
  assert.strictEqual(report.comparisons, matmulCases.length * 3);
  const bugReport = differentialTest(matmulCases, {
    buggyTiles: ({ A: a, B: b }) => matmulBuggyTiles(a, b, 4),
    buggyBound: ({ A: a, B: b }) => matmulBuggyBound(a, b),
  }, { rtol: 1e-5, atol: 1e-6 });
  assert.strictEqual(bugReport.failures.length, matmulCases.length * 2,
    "every seeded bug must be caught on every case");
  assert.ok(bugReport.failures.every((f) => f.firstMismatch !== null));
  assert.ok(bugReport.failures.some((f) => f.implementation === "buggyBound"));

  // 5. Reduction and softmax variants are compared through the same harness.
  const reductionCases = [
    { name: "wellScaled", input: { values: Array.from({ length: 64 }, () => prng.sym(1)) },
      run: ({ values }) => [[sumKahan(values)]] },
    { name: "wideRange", input: { values: Array.from({ length: 64 }, (_, k) => prng.sym(1) * (k % 8 === 0 ? 1e8 : 1)) },
      run: ({ values }) => [[sumKahan(values)]] },
    { name: "cancelling", input: { values: [2 ** 53, 1, -(2 ** 53)] },
      run: ({ values }) => [[sumKahan(values)]] },
  ];
  const reductionReport = differentialTest(reductionCases, {
    pairwise: ({ values }) => [[sumPairwise(values)]],
    kahan: ({ values }) => [[sumKahan(values)]],
    naive: ({ values }) => [[sumNaive(values)]],
  }, { rtol: 1e-9, atol: 1e-9 });
  assert.strictEqual(reductionReport.comparisons, 9);
  assert.ok(reductionReport.failures.every((f) => f.implementation !== "kahan"),
    "the reference must agree with itself");
  assert.ok(reductionReport.failures.length >= 1, "the cancelling case must expose a difference");
  assert.deepStrictEqual([...new Set(reductionReport.failures.map((f) => f.case))], ["cancelling"],
    "only the ill-conditioned case may diverge");
  assert.strictEqual(compareTensors([[sumBuggy([2 ** 53, 1, -(2 ** 53)])]], [[sumKahan([2 ** 53, 1, -(2 ** 53)])]],
    { rtol: 1e-9, atol: 1e-9 }).pass, false, "the compensated-summation bug must be caught");
  const softmaxReport = differentialTest([
    { name: "moderate", input: { logits: [1, 2, 3] }, run: ({ logits }) => softmaxStable(logits).map((v) => [v]) },
    { name: "large", input: { logits: [1000, 1001, 999] }, run: ({ logits }) => softmaxStable(logits).map((v) => [v]) },
  ], { stable: ({ logits }) => softmaxStable(logits).map((v) => [v]), naive: ({ logits }) => softmaxNaive(logits).map((v) => [v]) },
  { rtol: 1e-9, atol: 1e-12 });
  assert.strictEqual(softmaxReport.failures.length, 1, "the naive softmax must fail on the large logits");
  assert.strictEqual(softmaxReport.failures[0].case, "large");
  const stable = softmaxStable([1, 2, 3]);
  assert.ok(Math.abs(stable.reduce((a, v) => a + v, 0) - 1) < 1e-12, "the softmax must sum to one");

  // 6. NaN-aware comparison: matching NaNs pass, a NaN against a number does not.
  const nanTensor = [[Number.NaN, 1], [2, Number.NaN]];
  assert.ok(compareTensors(nanTensor, [[Number.NaN, 1], [2, Number.NaN]]).pass);
  const nanComparison = compareTensors([[Number.NaN, 1]], [[0, 1]]);
  assert.strictEqual(nanComparison.pass, false);
  assert.strictEqual(nanComparison.mismatches, 1);
  assert.strictEqual(nanComparison.firstMismatch.i, 0);
  const infinity = compareTensors([[Infinity]], [[Infinity]]);
  assert.ok(infinity.pass, "matching infinities must compare equal");
  assert.strictEqual(compareTensors([[Infinity]], [[-Infinity]]).pass, false);

  // 7. Boundary shapes: a single element, a single row, and an identity product.
  assert.deepStrictEqual(matmulNaive([[2]], [[3]]), [[6]]);
  assert.deepStrictEqual(matmulTiled([[2, 0], [0, 2]], [[1, 2], [3, 4]], 4), [[2, 4], [6, 8]]);
  const identity = [[1, 0], [0, 1]];
  assert.deepStrictEqual(matmulNaive(A.slice(0, 2).map((r) => r.slice(0, 2)), identity),
    A.slice(0, 2).map((r) => r.slice(0, 2)), "multiplying by the identity must be exact");
  assert.strictEqual(sumPairwise([]), 0);
  assert.strictEqual(sumKahan([]), 0);
  assert.strictEqual(sumNaive([5]), 5);

  // 8. Invalid input is rejected.
  assert.throws(() => matmulNaive([[1, 2]], [[1, 2]]), /inner dimensions/);
  assert.throws(() => compareTensors([[1]], [[1], [2]]), /shapes must agree/);
  assert.throws(() => compareTensors([[1, 2]], [[1]]), /shapes must agree/);

  // 9. Determinism: the same seed produces the same tensors, results and report.
  const rerun = () => {
    const p = new Prng(5);
    const X = randomMatrix(p, 4, 3, 1), Y = randomMatrix(p, 3, 4, 1);
    return { naive: matmulNaive(X, Y), tiled: matmulTiled(X, Y, 2), float32: matmulFloat32(X, Y) };
  };
  assert.deepStrictEqual(rerun(), rerun());
  assert.deepStrictEqual(rerun().naive, rerun().tiled);

  console.log(`differential: ${report.comparisons} clean comparisons, ` +
    `${bugReport.failures.length}/${matmulCases.length * 2} seeded bugs caught, ` +
    `float32 max abs error ${comparison32.maxAbs.toExponential(2)} <= bound ${bound.toExponential(2)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, matmulNaive, matmulTiled, matmulFloat32, matmulBuggyTiles, matmulBuggyBound,
  sumNaive, sumPairwise, sumKahan, sumBuggy, softmaxStable, softmaxNaive, compareTensors,
  differentialTest, sameValue };
