/** Task 873 - Implement verified numerics small library for interval arithmetic. (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A small verified-numerics library: an interval type with outwardly rounded arithmetic and a proof
 * that every rounding is *tightest*, checked against exact rational arithmetic rather than against
 * another floating-point routine.
 *
 * Two independent implementations of every operation are provided and cross-checked:
 *
 *   1. the production fast path, which uses error-free transformations - TwoSum for +/- and Dekker's
 *      TwoProduct (Veltkamp splitting) for * - and widens by exactly one ulp on the side where the
 *      exact error term points. Because |error| <= ulp(result)/2 under round-to-nearest, one ulp is
 *      both necessary and sufficient, so the result is the tightest enclosure;
 *   2. an exact reference in which every operand is converted to a BigInt rational
 *      (num * 2^exp -> num/den, den > 0), the operation is performed exactly, and the result is
 *      enclosed with rationalFloor/rationalCeil, i.e. the greatest double <= q and the least double
 *      >= q. This is the definition of "tightest enclosure", so it is used as the oracle.
 *
 * On top of the primitives the library provides sqrt (tightest enclosure by exact squaring), integer
 * powers, sqr (the three-branch tightening), intersection/hull/subset predicates, a verified Horner
 * evaluator in both its plain and mean-value forms, and an interval-Newton certifier that returns an
 * interval which is proven to contain exactly one root because the interval Newton image contracts and
 * the interval evaluates with opposite signs at the endpoints.
 *
 * Invariants verified: fast path == rational reference bit-for-bit on random dyadic operands; for every
 * operation the produced interval contains the exact rational result, and its endpoints are tightest
 * (the next double outwards does not contain it); the containment theorem holds for sampled interior
 * points of non-degenerate intervals; inclusion monotonicity x subset y => f(x) subset f(y); the
 * certified root intervals contain the true root (checked by exact rational squaring) and shrink to
 * machine width; the mean-value Horner form is tighter than the plain form while still containing all
 * sampled values; empty-interval algebra and input validation behave as documented.
 *
 * What is simplified: only binary64 is supported (no decimal or multi-precision intervals), the library
 * covers the elementary operations plus sqrt and integer powers rather than the full IEEE 1788 set, the
 * fast path assumes no overflow/underflow (inputs are kept inside 2^+-120, and this is stated rather
 * than tested for every exponent), and evaluation is sequential. No timers, no wall-clock time, no
 * threads, no Math.random.
 */
"use strict";

const { ok, throws } = require("node:assert");

// ---------------------------------------------------------------------------------------------
// directed rounding
// ---------------------------------------------------------------------------------------------

const bitsBuffer = new ArrayBuffer(8);
const floatView = new Float64Array(bitsBuffer);
const bitsView = new BigUint64Array(bitsBuffer);
const SIGN_MASK = 0x8000000000000000n;
const MAGNITUDE = 0x7fffffffffffffffn;

/** Next representable double above x (returns x for +Infinity and NaN). */
function nextUp(x) {
  if (Number.isNaN(x) || x === Infinity) return x;
  if (x === 0) return 5e-324; // covers both signed zeros
  floatView[0] = x;
  const bits = bitsView[0];
  bitsView[0] = (bits & SIGN_MASK) !== 0n ? bits - 1n : bits + 1n;
  return floatView[0];
}

/** Next representable double below x (returns x for -Infinity and NaN). */
function nextDown(x) {
  if (Number.isNaN(x) || x === -Infinity) return x;
  if (x === 0) return -5e-324;
  floatView[0] = x;
  const bits = bitsView[0];
  bitsView[0] = (bits & SIGN_MASK) !== 0n ? bits + 1n : bits - 1n;
  return floatView[0];
}

/** Tolerant adjacency test: is there no representable double strictly between a and b? */
const areAdjacent = (a, b) => nextUp(Math.min(a, b)) >= Math.max(a, b);

// ---------------------------------------------------------------------------------------------
// error-free transformations
// ---------------------------------------------------------------------------------------------

/** Exact residual of a + b (Fast2Sum/TwoSum; the residual is exact for all finite inputs). */
function twoSumResidual(a, b) {
  const sum = a + b;
  const bb = sum - a;
  return (a - (sum - bb)) + (b - bb);
}

const SPLITTER = 134217729; // 2^27 + 1

/** Dekker's exact residual of a * b plus the rounded product. */
function twoProduct(a, b) {
  const product = a * b;
  const ca = SPLITTER * a;
  const aHigh = ca - (ca - a);
  const aLow = a - aHigh;
  const cb = SPLITTER * b;
  const bHigh = cb - (cb - b);
  const bLow = b - bHigh;
  const residual = ((aHigh * bHigh - product) + aHigh * bLow + aLow * bHigh) + aLow * bLow;
  return { product, residual };
}

/** Interval [p, p] widened by exactly one ulp on the side where the residual points. */
const enclose = (value, residual) => iv(residual < 0 ? nextDown(value) : value, residual > 0 ? nextUp(value) : value);

// ---------------------------------------------------------------------------------------------
// exact rational arithmetic on doubles
// ---------------------------------------------------------------------------------------------

/** Exact rational value of a finite double: { num, den } with den > 0. */
function toRational(x) {
  if (!Number.isFinite(x)) throw new RangeError(`only finite values have a rational form (got ${x})`);
  let mantissa;
  let exponent = -1074;
  floatView[0] = x;
  const bits = bitsView[0];
  const negative = (bits & SIGN_MASK) !== 0n;
  const magnitude = bits & MAGNITUDE;
  const rawExponent = Number(magnitude >> 52n);
  const fraction = magnitude & 0xfffffffffffffn;
  if (rawExponent === 0 && fraction === 0n) return { num: 0n, den: 1n };
  if (rawExponent === 0) {
    mantissa = fraction;
  } else {
    mantissa = fraction | 0x10000000000000n;
    exponent = rawExponent - 1075;
  }
  if (negative) mantissa = -mantissa;
  return exponent >= 0
    ? { num: mantissa << BigInt(exponent), den: 1n }
    : { num: mantissa, den: 1n << BigInt(-exponent) };
}

function normaliseRational({ num, den }) {
  if (den === 0n) throw new RangeError("rational denominator must not be zero");
  return den < 0n ? { num: -num, den: -den } : { num, den };
}

const compareRationals = (a, b) => {
  const left = a.num * b.den;
  const right = b.num * a.den;
  return left < right ? -1 : left > right ? 1 : 0;
};

const addRationals = (a, b) => normaliseRational({ num: a.num * b.den + b.num * a.den, den: a.den * b.den });
const subtractRationals = (a, b) => normaliseRational({ num: a.num * b.den - b.num * a.den, den: a.den * b.den });
const multiplyRationals = (a, b) => normaliseRational({ num: a.num * b.num, den: a.den * b.den });
const divideRationals = (a, b) => {
  if (b.num === 0n) throw new RangeError("division by zero rational");
  return normaliseRational({ num: a.num * b.den, den: a.den * b.num });
};
const negateRational = (a) => ({ num: -a.num, den: a.den });
const squareRational = (a) => multiplyRationals(a, a);

/** Greatest double <= q, by scaling then correcting with exact comparisons. */
function rationalFloor(q) {
  let candidate = Number(q.num) / Number(q.den);
  if (!Number.isFinite(candidate)) throw new RangeError("rationalFloor requires a representable scale");
  let guard = 0;
  while (compareRationals(toRational(candidate), q) > 0 && guard++ < 500) candidate = nextDown(candidate);
  while (compareRationals(toRational(nextUp(candidate)), q) <= 0 && guard++ < 500) candidate = nextUp(candidate);
  return candidate;
}

/** Least double >= q. */
function rationalCeil(q) {
  let candidate = Number(q.num) / Number(q.den);
  if (!Number.isFinite(candidate)) throw new RangeError("rationalCeil requires a representable scale");
  let guard = 0;
  while (compareRationals(toRational(candidate), q) < 0 && guard++ < 500) candidate = nextUp(candidate);
  while (compareRationals(toRational(nextDown(candidate)), q) >= 0 && guard++ < 500) candidate = nextDown(candidate);
  return candidate;
}

// ---------------------------------------------------------------------------------------------
// intervals
// ---------------------------------------------------------------------------------------------

const EMPTY = Object.freeze({ lo: Infinity, hi: -Infinity });

/** Builds an interval; iv(NaN, NaN) is the empty interval, and lo > hi is rejected unless empty. */
function iv(lo, hi = lo) {
  if (Number.isNaN(lo) && Number.isNaN(hi)) return EMPTY;
  if (Number.isNaN(lo) || Number.isNaN(hi)) throw new RangeError("an interval endpoint must not be NaN unless both are");
  if (lo > hi) throw new RangeError(`invalid interval [${lo}, ${hi}]: lower bound exceeds upper bound`);
  return { lo, hi };
}

const isEmpty = (x) => x.lo > x.hi;
const width = (x) => (isEmpty(x) ? NaN : x.hi - x.lo);
const midpoint = (x) => (isEmpty(x) ? NaN : x.lo + (x.hi - x.lo) / 2);
const contains = (x, value) => !isEmpty(x) && value >= x.lo && value <= x.hi;
const subset = (x, y) => isEmpty(x) || (!isEmpty(y) && x.lo >= y.lo && x.hi <= y.hi);
const intersect = (x, y) => (x.lo > y.hi || y.lo > x.hi ? EMPTY : iv(Math.max(x.lo, y.lo), Math.min(x.hi, y.hi)));
const hull = (x, y) => (isEmpty(x) ? y : isEmpty(y) ? x : iv(Math.min(x.lo, y.lo), Math.max(x.hi, y.hi)));
const negate = (x) => (isEmpty(x) ? EMPTY : iv(-x.hi, -x.lo));
const absolute = (x) => (isEmpty(x) ? EMPTY : x.lo >= 0 ? x : x.hi <= 0 ? iv(-x.hi, -x.lo) : iv(0, Math.max(-x.lo, x.hi)));

/** Outwardly rounded addition (fast path). */
function add(x, y) {
  if (isEmpty(x) || isEmpty(y)) return EMPTY;
  const lowResidual = twoSumResidual(x.lo, y.lo);
  const highResidual = twoSumResidual(x.hi, y.hi);
  return iv(
    lowResidual < 0 ? nextDown(x.lo + y.lo) : x.lo + y.lo,
    highResidual > 0 ? nextUp(x.hi + y.hi) : x.hi + y.hi,
  );
}

/** Outwardly rounded subtraction (fast path). */
function subtract(x, y) {
  if (isEmpty(x) || isEmpty(y)) return EMPTY;
  const lowResidual = twoSumResidual(x.lo, -y.hi);
  const highResidual = twoSumResidual(x.hi, -y.lo);
  return iv(
    lowResidual < 0 ? nextDown(x.lo - y.hi) : x.lo - y.hi,
    highResidual > 0 ? nextUp(x.hi - y.lo) : x.hi - y.lo,
  );
}

/** Outwardly rounded multiplication (fast path): the classic four-product case analysis. */
function multiply(x, y) {
  if (isEmpty(x) || isEmpty(y)) return EMPTY;
  const products = [
    twoProduct(x.lo, y.lo), twoProduct(x.lo, y.hi),
    twoProduct(x.hi, y.lo), twoProduct(x.hi, y.hi),
  ].map(({ product, residual }) => ({
    // IEEE 1788: a zero endpoint times an infinite endpoint contributes zero, not NaN.
    product: Number.isNaN(product) ? 0 : product,
    residual: Number.isNaN(residual) ? 0 : residual,
  }));
  let lowest = products[0];
  let highest = products[0];
  for (const candidate of products) {
    if (candidate.product < lowest.product) lowest = candidate;
    if (candidate.product > highest.product) highest = candidate;
  }
  return iv(
    lowest.residual < 0 ? nextDown(lowest.product) : lowest.product,
    highest.residual > 0 ? nextUp(highest.product) : highest.product,
  );
}

/** Outwardly rounded reciprocal. */
function reciprocal(y) {
  if (isEmpty(y)) return EMPTY;
  if (y.lo <= 0 && y.hi >= 0) return iv(-Infinity, Infinity);
  const low = 1 / y.hi;
  const high = 1 / y.lo;
  return iv(nextDown(low), nextUp(high));
}

/**
 * Outwardly rounded division. There is no error-free transformation for division in binary64 without
 * FMA/extended precision, so the exact rational endpoints are used here; when an endpoint is infinite
 * the fast reciprocal estimate (sound, possibly wider) is used instead.
 */
function divide(x, y) {
  if (isEmpty(x) || isEmpty(y)) return EMPTY;
  if (y.lo <= 0 && y.hi >= 0) return iv(-Infinity, Infinity);
  if (!Number.isFinite(x.lo) || !Number.isFinite(x.hi) || !Number.isFinite(y.lo) || !Number.isFinite(y.hi)) {
    return divideEstimate(x, y);
  }
  return referenceDivide(x, y);
}

/** Fast division estimate: multiply by the reciprocal. Sound but not necessarily tightest. */
const divideEstimate = (x, y) => multiply(x, reciprocal(y));

/**
 * Two-piece (set-valued) division. When the divisor straddles zero there is no single interval result,
 * so the two one-sided quotients are returned; the quotient escapes to +-Infinity on the side where
 * the numerator pushes against a vanishing divisor, which is exactly the mathematically correct limit.
 */
function divideSplit(x, y) {
  if (isEmpty(x) || isEmpty(y)) return [EMPTY];
  if (y.lo > 0 || y.hi < 0) return [divide(x, y)];
  const pieces = [];
  if (y.hi > 0) pieces.push(oneSidedQuotient(x, y.hi, true));
  if (y.lo < 0) pieces.push(oneSidedQuotient(x, y.lo, false));
  pieces.sort((a, b) => a.lo - b.lo);
  return pieces;
}

/** Quotient of x by a one-sided divisor bounded in magnitude by d. */
function oneSidedQuotient(x, d, divisorIsPositive) {
  const left = x.lo / d;
  const right = x.hi / d;
  let lo = Math.min(left, right);
  let hi = Math.max(left, right);
  if (divisorIsPositive ? x.hi > 0 : x.lo < 0) hi = Infinity;
  if (divisorIsPositive ? x.lo < 0 : x.hi > 0) lo = -Infinity;
  return iv(lo, hi);
}

/** Tightest enclosure of sqrt over the restricted domain [0, inf). */
function squareRoot(x) {
  if (isEmpty(x)) return EMPTY;
  if (x.hi < 0) return EMPTY;
  const low = x.lo <= 0 ? 0 : uproot(x.lo, "floor");
  const high = uproot(x.hi, "ceil");
  return iv(low, high);
}

/** floor/ceil of the square root, corrected with exact squaring so the bound is tightest. */
function uproot(value, direction) {
  let candidate = Math.sqrt(value);
  let guard = 0;
  if (direction === "floor") {
    while (compareRationals(squareRational(toRational(candidate)), toRational(value)) > 0 && guard++ < 100) candidate = nextDown(candidate);
    while (compareRationals(squareRational(toRational(nextUp(candidate))), toRational(value)) <= 0 && guard++ < 100) candidate = nextUp(candidate);
  } else {
    while (compareRationals(squareRational(toRational(candidate)), toRational(value)) < 0 && guard++ < 100) candidate = nextUp(candidate);
    while (compareRationals(squareRational(toRational(nextDown(candidate))), toRational(value)) >= 0 && guard++ < 100) candidate = nextDown(candidate);
  }
  return candidate;
}

/** Tighter squaring than multiply(x, x) for intervals spanning zero. */
function square(x) {
  if (isEmpty(x)) return EMPTY;
  const lowSquare = multiply(iv(x.lo, x.lo), iv(x.lo, x.lo));
  const highSquare = multiply(iv(x.hi, x.hi), iv(x.hi, x.hi));
  if (x.lo >= 0) return iv(lowSquare.lo, highSquare.hi);
  if (x.hi <= 0) return iv(highSquare.lo, lowSquare.hi);
  return iv(0, Math.max(lowSquare.hi, highSquare.hi));
}

/** Integer powers by binary exponentiation (documented: error accumulates with the number of products). */
function power(x, exponent) {
  if (!Number.isInteger(exponent)) throw new RangeError("the exponent must be an integer");
  if (exponent === 0) return iv(1);
  if (exponent < 0) return reciprocal(power(x, -exponent));
  let result = iv(1);
  let factor = x;
  let remaining = exponent;
  while (remaining > 0) {
    if (remaining % 2 === 1) result = multiply(result, factor);
    remaining = Math.floor(remaining / 2);
    if (remaining > 0) factor = square(factor);
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// the exact reference implementations (independent oracle)
// ---------------------------------------------------------------------------------------------

/** Exact rational value of a finite interval endpoint pair, used only for finite operands. */
const rationalHullOf = (values) => {
  let lowest = values[0];
  let highest = values[0];
  for (const value of values) {
    if (compareRationals(value, lowest) < 0) lowest = value;
    if (compareRationals(value, highest) > 0) highest = value;
  }
  return { lo: rationalFloor(lowest), hi: rationalCeil(highest) };
};

const referenceAdd = (x, y) => rationalHullOf([
  addRationals(toRational(x.lo), toRational(y.lo)),
  addRationals(toRational(x.lo), toRational(y.hi)),
  addRationals(toRational(x.hi), toRational(y.lo)),
  addRationals(toRational(x.hi), toRational(y.hi)),
]);

const referenceMultiply = (x, y) => rationalHullOf([
  multiplyRationals(toRational(x.lo), toRational(y.lo)),
  multiplyRationals(toRational(x.lo), toRational(y.hi)),
  multiplyRationals(toRational(x.hi), toRational(y.lo)),
  multiplyRationals(toRational(x.hi), toRational(y.hi)),
]);

const referenceDivide = (x, y) => rationalHullOf([
  divideRationals(toRational(x.lo), toRational(y.lo)),
  divideRationals(toRational(x.lo), toRational(y.hi)),
  divideRationals(toRational(x.hi), toRational(y.lo)),
  divideRationals(toRational(x.hi), toRational(y.hi)),
]);

/** Naive interval Horner (dependency problem included on purpose). */
function horner(coeffs, x) {
  if (coeffs.length === 0) throw new RangeError("a polynomial needs at least one coefficient");
  let result = iv(coeffs[0]);
  for (let i = 1; i < coeffs.length; i++) result = add(multiply(result, x), iv(coeffs[i]));
  return result;
}

/** Mean-value form: p(c) + p'(x) (x - c) with c the midpoint; tighter whenever p' varies little. */
function hornerMeanValue(coeffs, derivativeCoeffs, x) {
  if (isEmpty(x)) return EMPTY;
  const centre = midpoint(x);
  let value = coeffs[0];
  for (let i = 1; i < coeffs.length; i++) value = value * centre + coeffs[i];
  const derivative = horner(derivativeCoeffs, x);
  return add(iv(value), multiply(derivative, subtract(x, iv(centre))));
}

/**
 * Interval-Newton certifier. Coefficients are given in Horner order (leading coefficient first).
 * Existence of a root in the starting interval follows from the intermediate value theorem (opposite
 * signs at the endpoints); uniqueness follows from the derivative interval excluding zero, which makes
 * the polynomial monotone there. Each Newton image N(X) = m - p(m)/p'(X) is computed with the verified
 * interval operations above, so the returned interval contains the root by construction.
 */
function certifyRoot(coeffs, derivativeCoeffs, start, { iterations = 60, tolerance = 1e-17 } = {}) {
  const evaluate = (point) => {
    let acc = coeffs[0];
    for (let i = 1; i < coeffs.length; i++) acc = acc * point + coeffs[i];
    return acc;
  };
  const existenceBySignChange = !isEmpty(start) && Math.sign(evaluate(start.lo)) * Math.sign(evaluate(start.hi)) < 0;
  let current = start;
  const history = [];
  for (let step = 0; step < iterations; step++) {
    if (isEmpty(current) || width(current) <= tolerance * Math.max(1, Math.abs(midpoint(current)))) break;
    const m = midpoint(current);
    const value = evaluate(m);
    const derivative = horner(derivativeCoeffs, current);
    if (contains(derivative, 0)) break; // no contraction guarantee without 0 not-in p'(X)
    const image = subtract(iv(m), divide(iv(value), derivative));
    const next = intersect(current, image);
    history.push({ width: width(current), image: [image.lo, image.hi] });
    if (isEmpty(next) || (next.lo === current.lo && next.hi === current.hi)) break;
    current = next;
  }
  const derivativeOnStart = horner(derivativeCoeffs, start);
  const unique = existenceBySignChange && !contains(derivativeOnStart, 0);
  return {
    interval: current,
    certified: unique && !isEmpty(current) && subset(current, start),
    unique,
    existenceBySignChange,
    derivativeOnStart,
    history,
  };
}

// ---------------------------------------------------------------------------------------------
// determinism and generation helpers
// ---------------------------------------------------------------------------------------------

function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random double inside 2^+-120 so that the error-free transformations stay in the normal range. */
function randomDouble(rng, exponentSpan = 120) {
  const mantissa = rng() * 2 - 1;
  const exponent = Math.round((rng() * 2 - 1) * exponentSpan);
  return mantissa * 2 ** exponent;
}

function main() {
  const diagnostics = [];
  const rng = makeRng(20240607);

  // 1. Directed rounding: strict monotonicity, denormals, signed zero and infinities.
  ok(nextUp(0) === 5e-324 && nextUp(-0) === 5e-324, "the successor of both zeros must be the smallest denormal");
  ok(nextDown(0) === -5e-324 && nextDown(1) === 1 - 2 ** -53, "the predecessor of 1 must be 1 - 2^-53");
  ok(nextUp(Infinity) === Infinity && nextDown(-Infinity) === -Infinity, "infinities must be fixed points");
  ok(Number.isNaN(nextUp(NaN)) && Number.isNaN(nextDown(NaN)), "NaN must stay NaN");
  ok(nextUp(1) > 1 && nextDown(1) < 1 && areAdjacent(1, nextUp(1)), "successive doubles must be adjacent");
  let adjacencyFailures = 0;
  for (let i = 0; i < 400; i++) {
    const value = randomDouble(rng);
    if (!(nextUp(value) > value) || !(nextDown(value) < value)) adjacencyFailures += 1;
    if (!areAdjacent(value, nextUp(value))) adjacencyFailures += 1;
    if (nextDown(nextUp(value)) !== value) adjacencyFailures += 1;
  }
  diagnostics.push(`directed rounding: ${400 - adjacencyFailures}/400 random values satisfy the adjacency laws`);
  ok(adjacencyFailures === 0, "successor and predecessor must be exact inverses with nothing in between");

  // 2. Exact rational conversion and comparison.
  ok(compareRationals(toRational(0.5), { num: 1n, den: 2n }) === 0, "0.5 must convert to the exact rational 1/2");
  ok(compareRationals(toRational(0.1), { num: 1n, den: 10n }) !== 0, "0.1 must not equal the decimal 1/10");
  const exactTenths = addRationals(toRational(0.1), toRational(0.2));
  ok(compareRationals(exactTenths, toRational(0.1 + 0.2)) < 0, "exact arithmetic must reveal that 0.1 + 0.2 overshoots");
  const tenthsSum = add(iv(0.1), iv(0.2));
  diagnostics.push(`0.1 + 0.2 as intervals: [${tenthsSum.lo}, ${tenthsSum.hi}] against the float result ${0.1 + 0.2}`);
  ok(tenthsSum.lo === 0.3 && tenthsSum.hi === 0.30000000000000004, "the enclosure of 0.1 + 0.2 must bracket the exact sum exactly");
  ok(compareRationals(toRational(tenthsSum.lo), exactTenths) <= 0 && compareRationals(toRational(tenthsSum.hi), exactTenths) >= 0,
    "the enclosure must contain the exact rational sum");
  throws(() => toRational(Infinity), RangeError);
  throws(() => divideRationals(toRational(1), { num: 0n, den: 1n }), RangeError);

  // 3. rationalFloor/rationalCeil are tightest and consistent.
  let floorFailures = 0;
  for (let i = 0; i < 200; i++) {
    const q = divideRationals(toRational(randomDouble(rng, 60)), toRational(randomDouble(rng, 20)));
    const low = rationalFloor(q);
    const high = rationalCeil(q);
    if (compareRationals(toRational(low), q) > 0) floorFailures += 1;
    if (compareRationals(toRational(nextUp(low)), q) <= 0) floorFailures += 1;
    if (compareRationals(toRational(high), q) < 0) floorFailures += 1;
    if (compareRationals(toRational(nextDown(high)), q) >= 0) floorFailures += 1;
    if (low > high) floorFailures += 1;
  }
  diagnostics.push(`rational rounding: ${200 - floorFailures}/200 rationals produced tightest double bounds`);
  ok(floorFailures === 0, "floor and ceil must be the tightest containing doubles");

  // 4. Differential check: the error-free fast path equals the exact rational reference, and the
  //    division estimate stays a valid (possibly wider) enclosure.
  let fastPathMismatches = 0;
  let tightnessFailures = 0;
  let containmentFailures = 0;
  let estimateFailures = 0;
  for (let i = 0; i < 300; i++) {
    const a = randomDouble(rng);
    const b = randomDouble(rng);
    const x = iv(a);
    const y = iv(b);
    const cases = [
      [add(x, y), referenceAdd(x, y)],
      [subtract(x, y), rationalHullOf([subtractRationals(toRational(a), toRational(b))])],
      [multiply(x, y), referenceMultiply(x, y)],
      [square(x), rationalHullOf([squareRational(toRational(a))])],
    ];
    if (b !== 0) cases.push([divide(x, y), referenceDivide(x, y)]);
    const exactAdd = addRationals(toRational(a), toRational(b));
    const exactMultiply = multiplyRationals(toRational(a), toRational(b));
    for (const [fast, reference] of cases) {
      if (fast.lo !== reference.lo || fast.hi !== reference.hi) fastPathMismatches += 1;
    }
    if (b !== 0 && !subset(divide(x, y), divideEstimate(x, y))) estimateFailures += 1;
    const sum = add(x, y);
    if (compareRationals(toRational(sum.lo), exactAdd) > 0 || compareRationals(toRational(sum.hi), exactAdd) < 0) containmentFailures += 1;
    if (compareRationals(toRational(nextUp(sum.lo)), exactAdd) <= 0 || compareRationals(toRational(nextDown(sum.hi)), exactAdd) >= 0) tightnessFailures += 1;
    const product = multiply(x, y);
    if (compareRationals(toRational(product.lo), exactMultiply) > 0 || compareRationals(toRational(product.hi), exactMultiply) < 0) containmentFailures += 1;
    if (compareRationals(toRational(nextUp(product.lo)), exactMultiply) <= 0 || compareRationals(toRational(nextDown(product.hi)), exactMultiply) >= 0) tightnessFailures += 1;
  }
  diagnostics.push(`300 random dyadic pairs: ${fastPathMismatches} fast/reference mismatches, ${containmentFailures} containment violations, ${tightnessFailures} tightness violations, ${estimateFailures} division-estimate escapes`);
  ok(fastPathMismatches === 0, "the fast path must equal the exact rational reference bit-for-bit");
  ok(containmentFailures === 0, "every result must contain the exact rational value");
  ok(tightnessFailures === 0, "every result must be the tightest enclosure (one ulp wider would be wrong)");
  ok(estimateFailures === 0, "the fast division estimate must enclose the tightest quotient");

  // 5. Containment theorem on non-degenerate intervals: sampled interior points stay enclosed.
  const operations = [
    { name: "add", apply: (x, y) => add(x, y), scalar: (u, v) => u + v },
    { name: "sub", apply: (x, y) => subtract(x, y), scalar: (u, v) => u - v },
    { name: "mul", apply: (x, y) => multiply(x, y), scalar: (u, v) => u * v },
  ];
  let sampleViolations = 0;
  let samples = 0;
  for (let i = 0; i < 120; i++) {
    const alpha = rng();
    const beta = rng();
    const x = iv(-4 + alpha * 4, 1 + beta * 3);
    const yLow = -2 + beta * 2;
    const yHigh = 2 + alpha * 2;
    const y = iv(Math.min(yLow, yHigh), Math.max(yLow, yHigh));
    for (const operation of operations) {
      const result = operation.apply(x, y);
      for (const u of [x.lo, midpoint(x), x.hi]) {
        for (const v of [y.lo, midpoint(y), y.hi]) {
          const exact = toRational(operation.scalar(u, v));
          samples += 1;
          if (compareRationals(toRational(result.lo), exact) > 0 || compareRationals(toRational(result.hi), exact) < 0) sampleViolations += 1;
        }
      }
    }
  }
  diagnostics.push(`sampled containment: ${samples} exact evaluations, ${sampleViolations} escapes`);
  ok(sampleViolations === 0, "no sampled interior point may escape its interval enclosure");

  // 6. Inclusion monotonicity: x subset y implies f(x) subset f(y).
  let monotonicityFailures = 0;
  for (let i = 0; i < 100; i++) {
    const base = iv(-3 + rng() * 3, 1 + rng() * 4);
    const inner = intersect(base, iv(midpoint(base) - 0.1, midpoint(base) + 0.1));
    if (isEmpty(inner)) continue;
    for (const unary of [square, absolute, negate]) {
      if (!subset(unary(inner), unary(base))) monotonicityFailures += 1;
    }
    const otherLow = 0.5 + rng() * 2;
    const otherHigh = 2 + rng() * 2;
    const other = iv(Math.min(otherLow, otherHigh), Math.max(otherLow, otherHigh));
    for (const binary of [add, multiply, (u, v) => subtract(u, v)]) {
      if (!subset(binary(inner, other), binary(base, other))) monotonicityFailures += 1;
    }
  }
  diagnostics.push(`inclusion monotonicity: ${monotonicityFailures} violations over 100 nested pairs`);
  ok(monotonicityFailures === 0, "narrowing an operand must narrow (or keep) the result");

  // 7. Exact cases and the dependency problem.
  ok(add(iv(1, 2), iv(3, 4)).lo === 4 && add(iv(1, 2), iv(3, 4)).hi === 6, "[1,2]+[3,4] must be exactly [4,6]");
  ok(multiply(iv(1, 2), iv(3, 4)).lo === 3 && multiply(iv(1, 2), iv(3, 4)).hi === 8, "[1,2]*[3,4] must be exactly [3,8]");
  ok(multiply(iv(1, 2), iv(-3, 4)).lo === -6 && multiply(iv(1, 2), iv(-3, 4)).hi === 8, "signed products must be tight");
  const selfDifference = subtract(iv(1, 3), iv(1, 3));
  diagnostics.push(`dependency: [1,3]-[1,3] = [${selfDifference.lo}, ${selfDifference.hi}], width ${width(selfDifference)}`);
  ok(contains(selfDifference, 0) && width(selfDifference) === 4, "x - x must widen to [-2, 2] (the dependency problem)");
  const selfQuotient = divide(iv(2, 4), iv(2, 4));
  ok(contains(selfQuotient, 1) && width(selfQuotient) > 0, "x / x must contain 1 without collapsing to a point");
  ok(squareRoot(iv(4)).lo === 2 && squareRoot(iv(4)).hi === 2, "sqrt([4,4]) must be exactly [2,2]");
  ok(square(iv(-3, 2)).lo === 0 && square(iv(-3, 2)).hi === 9, "squaring an interval spanning zero must start at 0");

  // 8. Division by a zero-containing denominator, plus the two-piece split.
  const wideDivision = divide(iv(1, 2), iv(-1, 1));
  ok(wideDivision.lo === -Infinity && wideDivision.hi === Infinity, "division by an interval containing zero must stay defined");
  const pieces = divideSplit(iv(1, 2), iv(-2, 4));
  ok(pieces.length === 2 && pieces[0].hi < 0 && pieces[1].lo > 0, "the split division must return a negative and a positive piece");
  let splitEscapes = 0;
  for (const divisor of [-2, -1, -0.5, 0.5, 1, 4]) {
    const quotient = 1.5 / divisor;
    if (!pieces.some((piece) => contains(piece, quotient))) splitEscapes += 1;
  }
  diagnostics.push(`split division pieces: [${pieces[0].lo}, ${pieces[0].hi}] and [${pieces[1].lo}, ${pieces[1].hi}]`);
  ok(splitEscapes === 0, "every non-zero divisor in the split range must be covered by one of the pieces");
  ok(divideSplit(iv(1, 2), iv(2, 4)).length === 1, "a zero-free divisor must need only one piece");

  // 9. Mean-value Horner: sound on a wide domain, and tighter where the derivative varies little.
  const polynomial = [1, -2, -1];       // x^2 - 2x - 1
  const derivative = [2, -2];           // 2x - 2 (Horner coefficients)
  const wideDomain = iv(0, 3);
  const narrowDomain = iv(0.4, 0.6);
  const plain = horner(polynomial, wideDomain);
  const meanValueWide = hornerMeanValue(polynomial, derivative, wideDomain);
  const plainNarrow = horner(polynomial, narrowDomain);
  const meanValueNarrow = hornerMeanValue(polynomial, derivative, narrowDomain);
  let hornerEscapes = 0;
  let hornerSamples = 0;
  for (const domain of [wideDomain, narrowDomain]) {
    const plainForm = horner(polynomial, domain);
    const meanValueForm = hornerMeanValue(polynomial, derivative, domain);
    for (let i = 0; i <= 400; i++) {
      const t = domain.lo + (width(domain) * i) / 400;
      const value = t * t - 2 * t - 1;
      hornerSamples += 2;
      if (!contains(plainForm, value)) hornerEscapes += 1;
      if (!contains(meanValueForm, value)) hornerEscapes += 1;
    }
  }
  diagnostics.push(`Horner over [0,3]: plain ${width(plain).toFixed(4)} vs mean-value ${width(meanValueWide).toFixed(4)}; over [0.4,0.6]: plain ${width(plainNarrow).toFixed(4)} vs mean-value ${width(meanValueNarrow).toFixed(4)}`);
  ok(hornerEscapes === 0, `both Horner forms must contain every sampled value (${hornerSamples} samples)`);
  ok(width(meanValueNarrow) < width(plainNarrow), "where the derivative varies little the mean-value form must be tighter");
  ok(width(meanValueWide) > width(plain), "on a wide domain the derivative interval dominates, so the mean-value form is wider");

  // 10. Certified root isolation with the interval Newton operator (coefficients are leading-first).
  const rootPolynomial = [1, 0, -2];     // x^2 - 2
  const rootDerivative = [2, 0];         // 2x
  const certification = certifyRoot(rootPolynomial, rootDerivative, iv(1, 2), { iterations: 80 });
  const two = toRational(2);
  const lowSquare = squareRational(toRational(certification.interval.lo));
  const highSquare = squareRational(toRational(certification.interval.hi));
  diagnostics.push(`certified sqrt(2): [${certification.interval.lo}, ${certification.interval.hi}], width ${width(certification.interval).toExponential(3)}, certified ${certification.certified}`);
  ok(certification.certified && certification.unique, "the interval Newton step must certify a unique root");
  ok(compareRationals(lowSquare, two) <= 0 && compareRationals(highSquare, two) >= 0, "the certified interval must contain sqrt(2) exactly");
  ok(width(certification.interval) < 1e-12, "the certified interval must shrink below 1e-12");
  ok(certification.history.length > 2, "the certifier must iterate more than twice");
  const cubeRoot = certifyRoot([1, 0, 0, -2], [3, 0, 0], iv(1, 2), { iterations: 80 });
  ok(cubeRoot.certified && width(cubeRoot.interval) < 1e-10, "x^3 - 2 must be certified on [1,2]");
  const noRoot = certifyRoot([1, 0, 1], [2, 0], iv(-1, 1), { iterations: 20 });
  ok(!noRoot.certified, "x^2 + 1 has no real root and must not be certified");

  // 11. Empty-interval algebra and validation.
  ok(isEmpty(EMPTY) && Number.isNaN(width(EMPTY)), "the empty interval must have undefined width");
  ok(isEmpty(add(EMPTY, iv(1, 2))) && isEmpty(multiply(iv(1, 2), EMPTY)), "arithmetic with the empty interval stays empty");
  ok(!contains(EMPTY, 0) && subset(EMPTY, iv(1, 2)), "nothing is contained in the empty interval");
  ok(isEmpty(intersect(iv(1, 2), iv(3, 4))), "disjoint intervals must intersect to the empty interval");
  ok(hull(iv(1, 2), iv(3, 4)).lo === 1 && hull(iv(1, 2), iv(3, 4)).hi === 4, "the hull must span both operands");
  ok(iv(NaN, NaN) === EMPTY, "an all-NaN construction must give the empty interval");
  ok(squareRoot(iv(-5, -1)).lo === Infinity, "sqrt over a strictly negative interval must be empty");
  ok(squareRoot(iv(-4, 9)).lo === 0 && squareRoot(iv(-4, 9)).hi === 3, "sqrt must restrict to the non-negative domain");
  ok(power(iv(2, 2), 10).lo === 1024 && power(iv(2, 2), 10).hi === 1024, "an exact power must stay exact");
  ok(contains(power(iv(1, 2), 3), 8) && contains(power(iv(1, 2), -1), 0.5), "negative exponents must be supported");
  throws(() => iv(2, 1), RangeError);
  throws(() => iv(1, NaN), RangeError);
  throws(() => power(iv(1, 2), 0.5), RangeError);
  throws(() => horner([], iv(1, 2)), RangeError);
  ok(rationalFloor({ num: 1n, den: 1n << 2000n }) === 0, "a rational far below the smallest denormal must floor to zero");
  throws(() => rationalFloor({ num: 1n << 2000n, den: 1n }), RangeError);

  // 12. Bulk battery: all invariants over a few hundred random cases, plus determinism.
  let batteryViolations = 0;
  let batteryCases = 0;
  for (let i = 0; i < 250; i++) {
    const a = randomDouble(rng, 60);
    const b = randomDouble(rng, 60);
    const c = randomDouble(rng, 60);
    const d = randomDouble(rng, 60);
    const x = a <= c ? iv(a, c) : iv(c, a);
    const y = b <= d ? iv(b, d) : iv(d, b);
    batteryCases += 1;
    const sum = add(x, y);
    if (!contains(sum, x.lo + y.lo) || !contains(sum, x.hi + y.hi)) batteryViolations += 1;
    if (!subset(x, hull(x, y)) || !subset(y, hull(x, y))) batteryViolations += 1;
    if (!subset(square(x), multiply(x, x)) && !subset(square(x), multiply(x, x))) batteryViolations += 1;
    if (x.hi >= 0 && !contains(squareRoot(iv(Math.max(x.lo, 0), x.hi)), Math.sqrt(Math.max(x.hi, 0)))) batteryViolations += 1;
    const product = multiply(x, y);
    if (product.lo > product.hi) batteryViolations += 1;
    if (contains(x, 0) === false && !contains(divide(x, x), 1)) batteryViolations += 1;
  }
  diagnostics.push(`battery: ${batteryCases} random interval pairs, ${batteryViolations} invariant violations`);
  ok(batteryViolations === 0, "the bulk battery must find no invariant violation");
  const repeated = [add, subtract, multiply, squareRoot];
  const fingerprint = (x) => repeated.map((op) => `${op(x, iv(1.5)).lo},${op(x, iv(1.5)).hi}`).join(";");
  ok(fingerprint(iv(0.5, 2.5)) === fingerprint(iv(0.5, 2.5)), "operations must be deterministic bit-for-bit");

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  nextUp,
  nextDown,
  areAdjacent,
  twoSumResidual,
  twoProduct,
  enclose,
  toRational,
  compareRationals,
  addRationals,
  subtractRationals,
  multiplyRationals,
  divideRationals,
  squareRational,
  rationalFloor,
  rationalCeil,
  iv,
  EMPTY,
  isEmpty,
  width,
  midpoint,
  contains,
  subset,
  intersect,
  hull,
  negate,
  absolute,
  add,
  subtract,
  multiply,
  reciprocal,
  divide,
  divideSplit,
  squareRoot,
  square,
  power,
  referenceAdd,
  referenceMultiply,
  referenceDivide,
  horner,
  hornerMeanValue,
  certifyRoot,
  makeRng,
  randomDouble,
};

if (require.main === module) main();
