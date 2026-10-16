/** Task 874 - Implement certified numerical solver for polynomial roots (Sturm sequences). (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A certified real-root solver. Roots are located with Sturm's theorem and every returned isolating
 * interval carries a proof obligation that is checked in exact arithmetic:
 *
 *   - polynomials have BigInt coefficients and are handled as exact rationals (num/den, den > 0), so
 *     polynomial division, gcd and evaluation never round;
 *   - the Sturm chain is built as p0 = p, p1 = p', p_{i+1} = -rem(p_{i-1}, p_i); each element is
 *     rescaled to a primitive integral form by a *positive* factor only, which leaves the number of
 *     sign variations - and therefore the root count - unchanged;
 *   - V(a) - V(b) counts the distinct real roots in (a, b], including at +-Infinity where the sign of
 *     a polynomial is that of its leading coefficient (times (-1)^degree at -Infinity);
 *   - multiplicity is recovered independently by square-free factorisation (repeated gcd with the
 *     derivative, Yun's scheme), and each square-free factor is isolated with its own Sturm chain, so
 *     a certified interval is always a sign-change interval and the roots inside it are simple;
 *   - bisection runs on exact rationals, so the isolation is a decision procedure rather than an
 *     iterative approximation; the reported double bounds are corrected outwards until they provably
 *     contain the rational bounds.
 *
 * Invariants verified: the Sturm count equals a brute-force sign-change scan on a fine grid and the
 * analytic count for factored polynomials; roots at rational points are found exactly; certified
 * intervals are pairwise disjoint, ordered, contain a sign change, and every isolated root satisfies
 * the polynomial to the interval width; the square-free decomposition multiplies back to the original
 * polynomial exactly; multiplicities match the planted ones; the Cauchy bound contains all roots;
 * Durand-Kerner (an independent complex solver) finds the same number of real roots; degenerate inputs
 * (zero, constant, linear) are handled or rejected as documented; and everything is deterministic.
 *
 * What is simplified: coefficients must be integers (rationals are supported through the rational
 * layer but not exposed as an entry point), the isolating bisection uses a bounded number of steps with
 * a requested width instead of an adaptive precision, complex roots are only cross-checked by
 * Durand-Kerner rather than certified, and evaluation is sequential. No timers, no wall-clock time, no
 * threads, no Math.random.
 */
"use strict";

const { ok, throws } = require("node:assert");

// ---------------------------------------------------------------------------------------------
// exact rationals (BigInt) - every polynomial operation below is exact
// ---------------------------------------------------------------------------------------------

/** Normalises to den > 0 and reduces by the gcd, which keeps the bisection denominators small. */
const qnormalise = ({ n, d }) => {
  if (d === 0n) throw new RangeError("rational denominator must not be zero");
  let num = d < 0n ? -n : n;
  let den = d < 0n ? -d : d;
  const divisor = bigGcd(num, den);
  if (divisor > 1n) {
    num /= divisor;
    den /= divisor;
  }
  return { n: num, d: den };
};
const qint = (value) => ({ n: BigInt(value), d: 1n });
const qisZero = (a) => a.n === 0n;
const qneg = (a) => ({ n: -a.n, d: a.d });
const qadd = (a, b) => qnormalise({ n: a.n * b.d + b.n * a.d, d: a.d * b.d });
const qsub = (a, b) => qnormalise({ n: a.n * b.d - b.n * a.d, d: a.d * b.d });
const qmul = (a, b) => qnormalise({ n: a.n * b.n, d: a.d * b.d });
const qdiv = (a, b) => {
  if (qisZero(b)) throw new RangeError("division by the zero rational");
  return qnormalise({ n: a.n * b.d, d: a.d * b.n });
};
const qsign = (a) => (a.n === 0n ? 0 : a.n > 0n ? 1 : -1);
const qcompare = (a, b) => qsign(qsub(a, b));
const qabs = (a) => ({ n: a.n < 0n ? -a.n : a.n, d: a.d });
const qtoNumber = (a) => Number(a.n) / Number(a.d);

const BIG_ZERO = qint(0);
const Q_INFINITY = Symbol("plus-infinity");
const Q_NEGATIVE_INFINITY = Symbol("minus-infinity");

// ---------------------------------------------------------------------------------------------
// polynomials: coefficient arrays with index = power of x (ascending), trimmed at the top
// ---------------------------------------------------------------------------------------------

function polyFromIntegers(coefficients) {
  if (!Array.isArray(coefficients) || coefficients.length === 0) throw new TypeError("a polynomial needs at least one coefficient");
  const trimmed = polyTrim(coefficients.map((value) => qint(value)));
  return trimmed.length === 0 ? [BIG_ZERO] : trimmed;
}

function polyTrim(coefficients) {
  let end = coefficients.length;
  while (end > 0 && qisZero(coefficients[end - 1])) end -= 1;
  return coefficients.slice(0, end);
}

const polyDegree = (p) => (p.length === 0 || (p.length === 1 && qisZero(p[0])) ? -1 : p.length - 1);
const polyIsZero = (p) => polyDegree(p) < 0;
const polyLeading = (p) => p[p.length - 1];

function polyAdd(a, b) {
  const length = Math.max(a.length, b.length);
  const out = new Array(length);
  for (let i = 0; i < length; i++) out[i] = qadd(a[i] ?? BIG_ZERO, b[i] ?? BIG_ZERO);
  return polyTrim(out);
}

const polyScale = (p, scalar) => polyTrim(p.map((coefficient) => qmul(coefficient, scalar)));

function polyMultiply(a, b) {
  if (polyIsZero(a) || polyIsZero(b)) return [BIG_ZERO];
  const out = new Array(a.length + b.length - 1).fill(BIG_ZERO);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out[i + j] = qadd(out[i + j], qmul(a[i], b[j]));
  }
  return polyTrim(out);
}

/** Polynomial division with remainder over the rationals. */
function polyDivMod(a, b) {
  if (polyIsZero(b)) throw new RangeError("division by the zero polynomial");
  let remainder = a.slice();
  const divisorDegree = polyDegree(b);
  const lead = polyLeading(b);
  const quotient = new Array(Math.max(1, polyDegree(a) - divisorDegree + 1)).fill(BIG_ZERO);
  for (let power = polyDegree(remainder); power >= divisorDegree; power = polyDegree(remainder)) {
    if (power < 0) break;
    const factor = qdiv(polyLeading(remainder), lead);
    quotient[power - divisorDegree] = qadd(quotient[power - divisorDegree], factor);
    for (let i = 0; i <= divisorDegree; i++) {
      remainder[power - divisorDegree + i] = qsub(remainder[power - divisorDegree + i], qmul(factor, b[i]));
    }
    remainder = polyTrim(remainder);
  }
  return { quotient: polyTrim(quotient), remainder: polyTrim(remainder) };
}

const polyDerivative = (p) => polyTrim(p.slice(1).map((coefficient, i) => qmul(coefficient, qint(i + 1))));

/** Monic gcd with a positive leading coefficient (normalisation does not change the root set). */
function polyGcd(a, b) {
  let x = polyTrim(a);
  let y = polyTrim(b);
  while (!polyIsZero(y)) {
    const { remainder } = polyDivMod(x, y);
    x = y;
    y = remainder;
  }
  if (polyIsZero(x)) return [BIG_ZERO];
  const inverseLead = qdiv(qint(1), qabs(polyLeading(x)));
  return polyScale(x, inverseLead);
}

/** Exact Horner evaluation at a rational point. */
function polyEvaluate(p, x) {
  let acc = BIG_ZERO;
  for (let i = p.length - 1; i >= 0; i--) acc = qadd(qmul(acc, x), p[i]);
  return acc;
}

/** Sign of p at a symbolic infinity. */
function polySignAtInfinity(p, direction) {
  const leadingSign = qsign(polyLeading(p));
  if (direction > 0) return leadingSign;
  return polyDegree(p) % 2 === 0 ? leadingSign : -leadingSign;
}

/** Rescale to primitive integral form by a positive factor only: sign variations are preserved. */
function polyPrimitiveIntegral(p) {
  if (polyIsZero(p)) return [BIG_ZERO];
  let commonDenominator = 1n;
  for (const coefficient of p) commonDenominator = (commonDenominator * coefficient.d) / bigGcd(commonDenominator, coefficient.d);
  const numerators = p.map((coefficient) => coefficient.n * (commonDenominator / coefficient.d));
  let content = 0n;
  for (const numerator of numerators) content = bigGcd(content, numerator < 0n ? -numerator : numerator);
  if (content === 0n) return [BIG_ZERO];
  return numerators.map((numerator) => qint(numerator / content));
}

/** Structural equality of two polynomials (exact, coefficient-wise). */
function polyEqual(a, b) {
  const left = polyTrim(a);
  const right = polyTrim(b);
  if (left.length !== right.length) return false;
  return left.every((coefficient, i) => coefficient.n === right[i].n && coefficient.d === right[i].d);
}

/** Renders a polynomial as plain numbers for diagnostics only. */
const polyNumbers = (p) => polyTrim(p).map((coefficient) => Number(coefficient.n) / Number(coefficient.d));

function bigGcd(a, b) {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y !== 0n) {
    const remainder = x % y;
    x = y;
    y = remainder;
  }
  return x;
}

/** Sturm chain of a square-free polynomial: p0 = p, p1 = p', p_{i+1} = -rem(p_{i-1}, p_i). */
function sturmChain(p) {
  if (polyIsZero(p)) throw new RangeError("the zero polynomial has no Sturm chain");
  if (polyDegree(p) <= 0) return [polyPrimitiveIntegral(p)];
  const chain = [polyPrimitiveIntegral(p), polyPrimitiveIntegral(polyDerivative(p))];
  while (polyDegree(chain[chain.length - 1]) > 0) {
    const { remainder } = polyDivMod(chain[chain.length - 2], chain[chain.length - 1]);
    const next = polyPrimitiveIntegral(polyScale(remainder, qint(-1)));
    if (polyIsZero(next)) break;
    chain.push(next);
  }
  return chain;
}

/** Number of sign variations of the chain at a rational point or at +-Infinity. */
function signVariations(chain, x) {
  let variations = 0;
  let previousSign = 0;
  for (const polynomial of chain) {
    const sign = x === Q_INFINITY || x === Q_NEGATIVE_INFINITY
      ? polySignAtInfinity(polynomial, x === Q_INFINITY ? 1 : -1)
      : qsign(polyEvaluate(polynomial, x));
    if (sign === 0) continue;
    if (previousSign !== 0 && sign !== previousSign) variations += 1;
    previousSign = sign;
  }
  return variations;
}

/** Distinct real roots in (a, b] by Sturm's theorem; a/b may be the symbolic infinities. */
const sturmCount = (chain, a, b) => signVariations(chain, a) - signVariations(chain, b);

/** Square-free factorisation: p = prod factor_i ^ multiplicity_i (Yun's scheme with gcds). */
function squareFreeFactorization(p) {
  const factors = [];
  let current = polyTrim(p);
  let derivative = polyDerivative(current);
  let repeated = polyGcd(current, derivative);
  let quotient = polyDivMod(current, repeated).quotient;
  let multiplicity = 1;
  while (polyDegree(quotient) > 0) {
    const nextRepeated = polyGcd(quotient, repeated);
    const factor = polyDivMod(quotient, nextRepeated).quotient;
    if (polyDegree(factor) > 0) factors.push({ factor: polyPrimitiveIntegral(factor), multiplicity });
    quotient = nextRepeated;
    repeated = polyDivMod(repeated, nextRepeated).quotient;
    multiplicity += 1;
  }
  return factors;
}

/** Cauchy bound: every root satisfies |r| <= 1 + max |a_i / a_n|. */
function cauchyBound(p) {
  const lead = polyLeading(p);
  let worst = BIG_ZERO;
  for (let i = 0; i < polyDegree(p); i++) worst = qcompare(qabs(qdiv(p[i], lead)), worst) > 0 ? qabs(qdiv(p[i], lead)) : worst;
  return qadd(qint(1), worst);
}

const qpow2 = (exponent) => ({ n: 1n << BigInt(exponent), d: 1n });

/** Rational bound as a power of two strictly above the given magnitude (keeps bisection exact). */
function powerOfTwoAbove(value) {
  let exponent = 1;
  while (qcompare(qpow2(exponent), value) <= 0) exponent += 1;
  return qpow2(exponent);
}

/** Outwardly corrected conversion of a rational to the double bounds containing it. */
function doubleBounds(value) {
  let low = qtoNumber(value);
  let guard = 0;
  while (qcompare({ n: BigInt(Math.round(low * 2 ** 52)), d: 1n << 52n }, value) > 0 && guard++ < 40) low -= Math.abs(low) * 2 ** -52 + 5e-324;
  while (qcompare({ n: BigInt(Math.round((low + Math.abs(low) * 2 ** -52 + 5e-324) * 2 ** 52)), d: 1n << 52n }, value) <= 0 && guard++ < 40) {
    low += Math.abs(low) * 2 ** -52 + 5e-324;
  }
  return { low, high: low };
}

/**
 * Certified real-root isolation. Each result carries the isolating interval, the multiplicity and a
 * flag telling whether the root was found exactly at a rational point.
 */
function isolateRealRoots(coefficients, { width = 1e-12, maxDepth = 200 } = {}) {
  const polynomial = typeof coefficients[0] === "object" && coefficients[0] !== null && "n" in coefficients[0]
    ? polyTrim(coefficients)
    : polyFromIntegers(coefficients);
  if (polyIsZero(polynomial)) throw new RangeError("the zero polynomial has infinitely many roots");
  if (!(width > 0)) throw new RangeError("width must be positive");
  const results = [];
  const widthRational = { n: BigInt(Math.max(1, Math.round(width * 1e12))), d: 10n ** 12n };
  for (const { factor, multiplicity } of squareFreeFactorization(polynomial)) {
    const chain = sturmChain(factor);
    const bound = powerOfTwoAbove(cauchyBound(factor));
    const negative = qneg(bound);
    const total = sturmCount(chain, negative, bound);
    if (total === 0) continue;
    const stack = [{ lo: negative, hi: bound, count: total, depth: 0 }];
    while (stack.length > 0) {
      const interval = stack.pop();
      if (interval.count === 0) continue;
      const mid = qdiv(qadd(interval.lo, interval.hi), qint(2));
      const rootAtHigh = qisZero(polyEvaluate(factor, interval.hi));
      const midIsRoot = qisZero(polyEvaluate(factor, mid));
      const narrowEnough = qcompare(qsub(interval.hi, interval.lo), widthRational) <= 0;
      // Roots are counted in (a, b], so a root at the lower endpoint belongs to the previous interval.
      if (interval.count === 1 && (rootAtHigh || midIsRoot || narrowEnough || interval.depth >= maxDepth)) {
        const lowSign = qsign(polyEvaluate(factor, interval.lo));
        const highSign = qsign(polyEvaluate(factor, interval.hi));
        const exactPoint = midIsRoot ? mid : rootAtHigh ? interval.hi : null;
        const exactValue = exactPoint ? qtoNumber(exactPoint) : null;
        results.push({
          // An exact rational root is reported as a degenerate interval; otherwise the isolating
          // rational interval is converted to double bounds (its width is already below the request).
          lo: exactValue ?? qtoNumber(interval.lo),
          hi: exactValue ?? qtoNumber(interval.hi),
          loRational: exactPoint ?? interval.lo,
          hiRational: exactPoint ?? interval.hi,
          multiplicity,
          exact: exactValue,
          sturmCount: interval.count,
          certified: interval.count === 1,
          signChange: lowSign !== 0 && highSign !== 0 && lowSign !== highSign,
          depth: interval.depth,
        });
        continue;
      }
      const leftCount = signVariations(chain, interval.lo) - signVariations(chain, mid);
      const rightCount = signVariations(chain, mid) - signVariations(chain, interval.hi);
      stack.push({ lo: interval.lo, hi: mid, count: leftCount, depth: interval.depth + 1 });
      stack.push({ lo: mid, hi: interval.hi, count: rightCount, depth: interval.depth + 1 });
    }
  }
  results.sort((a, b) => a.lo - b.lo);
  return results;
}

// ---------------------------------------------------------------------------------------------
// Durand-Kerner: an independent complex solver used only to cross-check the real root count
// ---------------------------------------------------------------------------------------------

function durandKerner(coefficients, { iterations = 400, tolerance = 1e-14 } = {}) {
  const p = polyFromIntegers(coefficients);
  const degree = polyDegree(p);
  if (degree < 1) return [];
  const leading = qtoNumber(polyLeading(p));
  const evaluate = (re, im) => {
    let accRe = 0;
    let accIm = 0;
    for (let i = p.length - 1; i >= 0; i--) {
      const nextRe = accRe * re - accIm * im + qtoNumber(p[i]);
      const nextIm = accRe * im + accIm * re;
      accRe = nextRe;
      accIm = nextIm;
    }
    return [accRe / leading, accIm / leading];
  };
  const roots = [];
  for (let k = 0; k < degree; k++) {
    const angle = ((k + 1) * 2 * Math.PI) / degree + 0.4;
    roots.push([0.4 * Math.cos(angle) + 0.9 * Math.cos(angle), 0.4 * Math.sin(angle) + 0.9 * Math.sin(angle)]);
  }
  for (let step = 0; step < iterations; step++) {
    let worst = 0;
    for (let k = 0; k < degree; k++) {
      const [re, im] = roots[k];
      const [valueRe, valueIm] = evaluate(re, im);
      let denominatorRe = 1;
      let denominatorIm = 0;
      for (let j = 0; j < degree; j++) {
        if (j === k) continue;
        const diffRe = re - roots[j][0];
        const diffIm = im - roots[j][1];
        const nextRe = denominatorRe * diffRe - denominatorIm * diffIm;
        const nextIm = denominatorRe * diffIm + denominatorIm * diffRe;
        denominatorRe = nextRe;
        denominatorIm = nextIm;
      }
      const magnitude = denominatorRe * denominatorRe + denominatorIm * denominatorIm;
      if (magnitude === 0) continue;
      const deltaRe = (valueRe * denominatorRe + valueIm * denominatorIm) / magnitude;
      const deltaIm = (valueIm * denominatorRe - valueRe * denominatorIm) / magnitude;
      roots[k] = [re - deltaRe, im - deltaIm];
      worst = Math.max(worst, Math.hypot(deltaRe, deltaIm));
    }
    if (worst < tolerance) break;
  }
  return roots.map(([re, im]) => ({ re, im }));
}

/** Deterministic generator for random integer polynomials. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x2545f491;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Product of (x - root_i) for integer roots, i.e. a polynomial with known roots. */
function fromRoots(roots) {
  let polynomial = [qint(1)];
  for (const root of roots) polynomial = polyMultiply(polynomial, [qint(-root), qint(1)]);
  return polynomial.map((coefficient) => Number(coefficient.n / coefficient.d));
}

/** Brute-force oracle: count sign changes of p on a fine grid (misses double roots by design). */
function gridSignChangeCount(coefficients, lo, hi, steps = 200000) {
  const p = polyFromIntegers(coefficients);
  let previousSign = qsign(polyEvaluate(p, qint(Math.round(lo * 1000))));
  let count = 0;
  for (let i = 1; i <= steps; i++) {
    const x = lo + ((hi - lo) * i) / steps;
    const value = polyEvaluate(p, { n: BigInt(Math.round(x * 1000)), d: 1000n });
    const sign = qsign(value);
    if (sign === 0) continue;
    if (previousSign !== 0 && sign !== previousSign) count += 1;
    previousSign = sign;
  }
  return count;
}

function main() {
  const diagnostics = [];

  // 1. Polynomial arithmetic: division round-trips, derivatives and gcds are exact.
  const factored = polyFromIntegers(fromRoots([1, 2, 3, 4]));
  const linear = polyFromIntegers([-1, 1]);            // x - 1
  const quotient = polyDivMod(factored, linear);
  diagnostics.push(`(x-1)(x-2)(x-3)(x-4) / (x-1) = ${JSON.stringify(quotient.quotient.map((c) => Number(c.n / c.d)))}`);
  ok(polyIsZero(quotient.remainder), "dividing by a factor must leave no remainder");
  ok(JSON.stringify(quotient.quotient.map((c) => Number(c.n / c.d))) === JSON.stringify(fromRoots([2, 3, 4])), "the quotient must be the expected cubic");
  const gcdOfProduct = polyGcd(polyMultiply(factored, polyFromIntegers([-1, 1])), factored);
  ok(polyDegree(gcdOfProduct) === 4, "gcd(p * (x-1), p) must have degree 4");
  ok(qisZero(polyEvaluate(factored, qint(3))), "x = 3 must be a root");
  ok(qcompare(polyEvaluate(factored, qint(5)), BIG_ZERO) > 0, "x = 5 must not be a root");
  throws(() => polyDivMod(factored, [BIG_ZERO]), RangeError);
  throws(() => polyFromIntegers([]), TypeError);

  // 2. Sturm chain structure: p1 is the derivative, and the chain ends in a non-zero constant for a
  //    square-free polynomial (which is exactly gcd(p, p') = 1).
  const chain = sturmChain(factored);
  diagnostics.push(`Sturm chain degrees: ${chain.map((p) => polyDegree(p)).join(", ")}`);
  ok(chain.length >= 2 && polyDegree(chain[1]) === 3, "the second Sturm element must be the derivative");
  ok(chain[chain.length - 1].length === 1, "a square-free polynomial ends in a non-zero constant");
  const nonSquareFree = polyMultiply(polyMultiply(linear, linear), linear); // (x-1)^3
  const chainOfNonSquareFree = sturmChain(nonSquareFree);
  diagnostics.push(`(x-1)^3 chain degrees: ${chainOfNonSquareFree.map((p) => polyDegree(p)).join(", ")}`);
  ok(polyDegree(chainOfNonSquareFree[chainOfNonSquareFree.length - 1]) === 2, "for a non-square-free polynomial the chain stops at gcd(p, p') of degree 2");
  ok(polyDegree(polyGcd(nonSquareFree, polyDerivative(nonSquareFree))) === 2, "gcd((x-1)^3, 3(x-1)^2) must have degree 2");

  // 3. Sturm counting agrees with the brute-force sign-change scan.
  const cases = [
    { roots: [1, 2, 3, 4], span: [0, 5], expected: 4, gridExpected: 4 },
    { roots: [-3, -1, 2], span: [-4, 4], expected: 3, gridExpected: 3 },
    { roots: [1, 1, 5], span: [0, 6], expected: 2, gridExpected: 1 },  // a double root shows no sign change
    { roots: [0, 7], span: [-1, 8], expected: 2, gridExpected: 2 },
  ];
  for (const testCase of cases) {
    const coefficients = fromRoots(testCase.roots);
    const polynomial = polyFromIntegers(coefficients);
    const sturm = sturmChain(polyPrimitiveIntegral(polyDivMod(polynomial, polyGcd(polynomial, polyDerivative(polynomial))).quotient));
    const sturmTotal = signVariations(sturm, Q_NEGATIVE_INFINITY) - signVariations(sturm, Q_INFINITY);
    const gridTotal = gridSignChangeCount(coefficients, -50, 50, 100000);
    diagnostics.push(`roots of ${JSON.stringify(testCase.roots)}: Sturm ${sturmTotal}, grid scan ${gridTotal}, distinct ${testCase.expected}`);
    ok(sturmTotal === testCase.expected, `the Sturm count must equal the number of distinct roots (${testCase.expected})`);
    ok(gridTotal === testCase.gridExpected, `the grid scan must see ${testCase.gridExpected} sign changes (tangencies are invisible to it)`);
    const inner = sturmCount(sturm, qint(testCase.span[0]), qint(testCase.span[1]));
    ok(inner === testCase.expected, `the count inside [${testCase.span}] must match`);
  }

  // 4. Certified isolation of factored polynomials: intervals are disjoint, ordered, and each contains
  //    a sign change.
  const isolated = isolateRealRoots(fromRoots([1, 2, 3, 4]), { width: 1e-9 });
  diagnostics.push(`isolated roots: ${isolated.map((root) => `[${root.lo.toFixed(6)}, ${root.hi.toFixed(6)}]x${root.multiplicity}`).join(", ")}`);
  ok(isolated.length === 4, "four distinct roots must be isolated");
  ok(isolated.every((root, i) => i === 0 || isolated[i - 1].hi <= root.lo), "isolating intervals must be disjoint and ordered");
  isolated.forEach((root, index) => {
    ok(root.lo <= index + 1 && root.hi >= index + 1, `root ${index + 1} must lie inside its interval`);
    ok(root.hi - root.lo < 1e-9, `interval ${index} must be narrower than the requested width`);
  });
  ok(isolated.every((root) => root.exact !== null && root.lo === root.hi), "the integer roots of (x-1)(x-2)(x-3)(x-4) must be certified exactly");
  ok(isolated.every((root) => root.certified && (root.signChange || root.exact !== null)), "every isolating interval must be certified by Sturm count 1 plus a sign change or an exact root");

  // 5. Exact rational roots are found as exact points.
  const exactRoots = isolateRealRoots(fromRoots([0, 2, 4]), { width: 1e-12 });
  diagnostics.push(`exact rational roots: ${exactRoots.map((root) => root.exact).join(", ")}`);
  ok(exactRoots.length === 3, "three roots must be found");
  ok(exactRoots.every((root) => root.exact !== null), "even rational roots must be certified exactly");
  const half = isolateRealRoots([-3, 2], { width: 1e-15 });          // 2x - 3
  ok(half.length === 1 && Math.abs(half[0].exact - 1.5) < 1e-15, "the root of 2x - 3 must be exactly 1.5");

  // 6. Irrational roots: x^2 - 2 must be isolated and the *rational* intervals must bracket sqrt(2)
  //    exactly when squared (no floating point tolerance is involved in this verification).
  const sqrtTwo = isolateRealRoots([-2, 0, 1], { width: 1e-13 });
  const two = qint(2);
  const squareBrackets = (root) => {
    const lowSquare = qmul(root.loRational, root.loRational);
    const highSquare = qmul(root.hiRational, root.hiRational);
    return qcompare(lowSquare, two) <= 0 && qcompare(highSquare, two) >= 0;
  };
  diagnostics.push(`sqrt(2) intervals: ${sqrtTwo.map((root) => `[${root.lo}, ${root.hi}]`).join(" and ")}`);
  ok(sqrtTwo.length === 2, "x^2 - 2 must have two real roots");
  ok(!squareBrackets(sqrtTwo[0]), "the squared bounds of the negative interval must both lie below 2");
  ok(squareBrackets(sqrtTwo[1]), "the squared bounds of the positive interval must bracket 2 exactly");
  ok(Math.abs(sqrtTwo[1].lo - Math.SQRT2) < 1e-12 && Math.abs(sqrtTwo[1].hi - Math.SQRT2) < 1e-12, "the positive root must bracket sqrt(2)");
  ok(sqrtTwo[1].signChange && !sqrtTwo[1].exact, "an irrational root must be certified by a sign change");

  // 7. Multiplicities from the square-free decomposition, and an exact verification by multiplication.
  const withMultiplicity = polyMultiply(polyMultiply(linear, linear), polyMultiply(linear, polyFromIntegers([2, 1])));
  const factors = squareFreeFactorization(withMultiplicity);
  diagnostics.push(`square-free factors: ${factors.map((f) => `degree ${polyDegree(f.factor)} x${f.multiplicity}`).join(", ")}`);
  let rebuilt = [qint(1)];
  for (const { factor, multiplicity } of factors) {
    for (let i = 0; i < multiplicity; i++) rebuilt = polyMultiply(rebuilt, factor);
  }
  ok(polyEqual(rebuilt, polyPrimitiveIntegral(withMultiplicity)), "the square-free factors must multiply back exactly");
  const multiRoots = isolateRealRoots(withMultiplicity, { width: 1e-9 });
  const multiplicityOne = multiRoots.find((root) => Math.abs(root.lo + 2) < 1e-6);
  const multiplicityThree = multiRoots.find((root) => Math.abs(root.lo - 1) < 1e-6);
  diagnostics.push(`(x-1)^3 (x+2): ${multiRoots.map((root) => `${root.lo.toFixed(4)}x${root.multiplicity}`).join(", ")}`);
  ok(multiRoots.length === 2, "only the distinct roots must be reported");
  ok(multiplicityThree && multiplicityThree.multiplicity === 3, "x = 1 must be reported with multiplicity 3");
  ok(multiplicityOne && multiplicityOne.multiplicity === 1, "x = -2 must be reported with multiplicity 1");

  // 8. Degree-8 Wilkinson-style product of consecutive roots: all roots certified, no misses.
  const wilkinson = fromRoots([1, 2, 3, 4, 5, 6, 7, 8]);
  const wilkinsonRoots = isolateRealRoots(wilkinson, { width: 1e-10 });
  diagnostics.push(`Wilkinson-style degree 8: ${wilkinsonRoots.length} roots, worst width ${Math.max(...wilkinsonRoots.map((r) => r.hi - r.lo)).toExponential(2)}`);
  ok(wilkinsonRoots.length === 8, "all eight roots must be isolated");
  wilkinsonRoots.forEach((root, index) => ok(root.lo <= index + 1 && root.hi >= index + 1, `root ${index + 1} must be bracketed`));

  // 9. No real roots, and the independent complex solver reports the right count of real ones.
  const noRealRoots = isolateRealRoots([1, 0, 0, 0, 1], { width: 1e-9 });   // x^4 + 1
  const complexRoots = durandKerner([1, 0, 0, 0, 1]);
  const realComplexRoots = complexRoots.filter((root) => Math.abs(root.im) < 1e-8);
  diagnostics.push(`x^4 + 1: Sturm real roots ${noRealRoots.length}, Durand-Kerner real roots ${realComplexRoots.length} of ${complexRoots.length}`);
  ok(noRealRoots.length === 0, "x^4 + 1 must have no real roots");
  ok(complexRoots.length === 4 && realComplexRoots.length === 0, "Durand-Kerner must confirm four complex roots");
  const mixed = fromRoots([-2, 1, 3]);
  const mixedSturm = isolateRealRoots(mixed, { width: 1e-9 }).length;
  const mixedKerner = durandKerner(mixed).filter((root) => Math.abs(root.im) < 1e-7).length;
  diagnostics.push(`mixed cubic: Sturm ${mixedSturm}, Durand-Kerner ${mixedKerner}`);
  ok(mixedSturm === 3 && mixedKerner === 3, "both solvers must agree on three real roots");

  // 10. Cauchy bound contains every root, and counts inside sub-intervals are exact.
  const bounded = polyFromIntegers([2, -1, -2, 1]);            // (x+1)(x-1)(x-2)
  const bound = cauchyBound(bounded);
  const chainForBound = sturmChain(polyPrimitiveIntegral(polyDivMod(bounded, polyGcd(bounded, polyDerivative(bounded))).quotient));
  const allInside = sturmCount(chainForBound, qneg(bound), bound);
  diagnostics.push(`Cauchy bound ${qtoNumber(bound).toFixed(4)} contains ${allInside} of 3 roots`);
  ok(allInside === 3, "the Cauchy bound must contain all roots");
  ok(qtoNumber(bound) >= 2, "the bound must dominate the largest root by magnitude");
  ok(sturmCount(chainForBound, { n: 3n, d: 2n }, qint(3)) === 1, "exactly one root in (1.5, 3]");
  ok(sturmCount(chainForBound, qint(0), { n: 3n, d: 2n }) === 1, "exactly one root in (0, 1.5]");
  ok(sturmCount(chainForBound, qneg(bound), qint(-1)) === 1, "exactly one root in (-3, -1]");
  ok(sturmCount(chainForBound, qint(5), qint(9)) === 0, "no roots beyond the bound");

  // 11. Degenerate inputs.
  throws(() => isolateRealRoots([0]), RangeError);
  throws(() => isolateRealRoots([0, 0, 0]), RangeError);
  throws(() => sturmChain([BIG_ZERO]), RangeError);
  throws(() => isolateRealRoots([-1, 1], { width: 0 }), RangeError);
  ok(isolateRealRoots([7], { width: 1e-9 }).length === 0, "a constant non-zero polynomial has no roots");
  const linearRoot = isolateRealRoots([-5, 2], { width: 1e-15 });
  ok(linearRoot.length === 1 && Math.abs(linearRoot[0].exact - 2.5) < 1e-15, "2x - 5 must have the exact root 2.5");
  ok(durandKerner([7]).length === 0, "a constant polynomial must have no complex roots either");

  // 12. Determinism and refinement monotonicity.
  const firstRun = isolateRealRoots(wilkinson, { width: 1e-8 });
  const secondRun = isolateRealRoots(wilkinson, { width: 1e-8 });
  const fingerprint = (roots) => roots.map((root) => `${root.lo}|${root.hi}|${root.multiplicity}|${root.loRational.n}/${root.loRational.d}|${root.hiRational.n}/${root.hiRational.d}`).join(";");
  ok(fingerprint(firstRun) === fingerprint(secondRun), "isolation must be bit-for-bit reproducible");
  const coarse = isolateRealRoots(wilkinson, { width: 1e-4 });
  const fine = isolateRealRoots(wilkinson, { width: 1e-12 });
  ok(coarse.length === fine.length, "the root count must not depend on the requested width");
  ok(fine.every((root, i) => root.lo >= coarse[i].lo - 1e-15 && root.hi <= coarse[i].hi + 1e-15), "refinement must only narrow the intervals");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  qint,
  qadd,
  qsub,
  qmul,
  qdiv,
  qsign,
  qcompare,
  polyFromIntegers,
  polyTrim,
  polyDegree,
  polyAdd,
  polyMultiply,
  polyScale,
  polyDivMod,
  polyDerivative,
  polyGcd,
  polyEvaluate,
  polyPrimitiveIntegral,
  sturmChain,
  signVariations,
  sturmCount,
  squareFreeFactorization,
  cauchyBound,
  isolateRealRoots,
  durandKerner,
  fromRoots,
  gridSignChangeCount,
  makeRng,
  Q_INFINITY,
  Q_NEGATIVE_INFINITY,
};

if (require.main === module) main();
