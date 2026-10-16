/** Task 944 - Implement high-precision numeric library with arbitrary precision rational arithmetic.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * An exact rational type built on BigInt: canonical form (positive denominator, gcd-reduced),
 * closed arithmetic (add, sub, mul, div, neg, pow, compare), a decimal expander that uses nothing
 * but integer division, a rational Newton square root and a continued-fraction expansion with its
 * inverse. The test battery uses the library to reach results with a known closed form - the
 * twentieth harmonic number 55835135/15519504, the fraction 415/93 = [4; 2, 6, 7], twenty digits of
 * pi from Machin's formula and twenty digits of sqrt(2) from Newton - and cross-checks every one of
 * them against a computation that avoids rationals entirely (fixed-point integer accumulation) or
 * against published constants.
 *
 * Simplified: no automatic precision control (callers choose the iteration count or the digit count),
 * no transcendental functions beyond arctan(1/x) and sqrt, and no irrational type. Verified:
 * canonicalisation, exact identities, division-by-zero and negative-sqrt rejection, the harmonic and
 * Basel-style sums, the continued-fraction round trip, Machin's pi, Newton's sqrt, float agreement
 * and determinism.
 */
"use strict";
const assert = require("node:assert");

function gcd(a, b) {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y) { const t = x % y; x = y; y = t; }
  return x;
}

/** Canonical rational: gcd-reduced with a positive denominator. */
function rat(n, d = 1n) {
  if (d === 0n) throw new Error("zero denominator");
  let num = BigInt(n);
  let den = BigInt(d);
  if (den < 0n) { num = -num; den = -den; }
  const g = gcd(num, den) || 1n;
  return { n: num / g, d: den / g };
}
const NUM = (n) => rat(BigInt(n));
const add = (a, b) => rat(a.n * b.d + b.n * a.d, a.d * b.d);
const sub = (a, b) => rat(a.n * b.d - b.n * a.d, a.d * b.d);
const mul = (a, b) => rat(a.n * b.n, a.d * b.d);
const neg = (a) => ({ n: -a.n, d: a.d });
const div = (a, b) => {
  if (b.n === 0n) throw new Error("division by zero");
  return rat(a.n * b.d, a.d * b.n);
};
const cmp = (a, b) => {
  const left = a.n * b.d;
  const right = b.n * a.d;
  return left < right ? -1 : left > right ? 1 : 0;
};
const abs = (a) => (a.n < 0n ? neg(a) : a);
const toNumber = (a) => Number(a.n) / Number(a.d);
const pow = (a, k) => (k < 0 ? div(NUM(1), pow(a, -k)) : Array.from({ length: k }).reduce((acc) => mul(acc, a), NUM(1)));

/** Decimal expansion: integer division only, so it is independent of float formatting. */
function toDecimal(a, digits) {
  const scale = 10n ** BigInt(digits);
  const sign = a.n < 0n ? "-" : "";
  const magnitude = a.n < 0n ? -a.n : a.n;
  const scaled = (magnitude * scale) / a.d;
  const whole = scaled / scale;
  const frac = (scaled % scale).toString().padStart(digits, "0");
  return `${sign}${whole}.${frac}`;
}

/** Newton's iteration on rationals; the error squares each step. */
function sqrtRat(a, iterations = 5) {
  if (cmp(a, NUM(0)) < 0) throw new Error("square root of a negative rational");
  let x = NUM(1);
  for (let i = 0; i < iterations; i += 1) x = div(add(x, div(a, x)), NUM(2));
  return x;
}

function continuedFraction(a) {
  let n = a.n;
  let d = a.d;
  const terms = [];
  for (;;) {
    const q = n / d;
    terms.push(q);
    const r = n % d;
    if (r === 0n) return terms;
    n = d;
    d = r;
  }
}
const fromContinuedFraction = (terms) => {
  let acc = NUM(terms[terms.length - 1]);
  for (let i = terms.length - 2; i >= 0; i -= 1) acc = add(NUM(terms[i]), div(NUM(1), acc));
  return acc;
};

/** arctan(1/x) truncated after `terms` series terms, exactly. */
function arctanInverse(x, terms) {
  const x2 = BigInt(x) * BigInt(x);
  let power = rat(1n, BigInt(x));
  let sum = NUM(0);
  for (let k = 0; k < terms; k += 1) {
    const term = div(power, NUM(2 * k + 1));
    sum = k % 2 === 0 ? add(sum, term) : sub(sum, term);
    power = div(power, NUM(x2));
  }
  return sum;
}
/** Machin's formula: pi = 16 arctan(1/5) - 4 arctan(1/239). */
const machinPi = (terms = 24) => sub(mul(NUM(16), arctanInverse(5, terms)), mul(NUM(4), arctanInverse(239, terms)));

const harmonic = (n) => Array.from({ length: n }, (_, i) => rat(1n, BigInt(i + 1))).reduce(add, NUM(0));

function main() {
  // 1. Canonical form: signs are pulled out and the fraction is reduced.
  const same = (a, b) => a.n === b.n && a.d === b.d;
  assert(same(rat(2n, 4n), rat(1n, 2n)), "2/4 canonicalises to 1/2");
  assert(same(rat(1n, -2n), { n: -1n, d: 2n }), "negative denominators move to the numerator");
  assert(same(rat(-4n, -8n), rat(1n, 2n)), "double negatives cancel");
  assert(gcd(rat(123456789n, 987654321n).n, rat(123456789n, 987654321n).d) === 1n, "reduced fractions are coprime");

  // 2. Exactness: rational arithmetic never rounds, even with awkward denominators.
  assert(cmp(add(rat(1n, 3n), rat(1n, 6n)), rat(1n, 2n)) === 0, "1/3 + 1/6 is exactly 1/2");
  assert(cmp(mul(rat(1n, 3n), rat(3n)), NUM(1)) === 0, "1/3 * 3 is exactly 1");
  assert(cmp(sub(sub(NUM(1), rat(1n, 3n)), rat(2n, 3n)), NUM(0)) === 0, "1 - 1/3 - 2/3 is exactly 0");
  assert(cmp(pow(rat(2n, 3n), 10), rat(1024n, 59049n)) === 0, "(2/3)^10 is exactly 1024/59049");
  assert(cmp(pow(rat(2n, 3n), -2), rat(9n, 4n)) === 0 && cmp(pow(rat(2n, 3n), 0), NUM(1)) === 0, "negative and zero exponents are exact");

  // 3. Boundary: impossible operations are rejected.
  assert.throws(() => rat(1n, 0n), /zero denominator/);
  assert.throws(() => div(rat(1n, 2n), NUM(0)), /division by zero/);
  assert.throws(() => sqrtRat(NUM(-1)), /negative rational/);

  // 4. Closed form: the twentieth harmonic number.
  const h20 = harmonic(20);
  assert(cmp(h20, rat(55835135n, 15519504n)) === 0, `H20 is ${h20.n}/${h20.d}`);
  assert(Math.abs(toNumber(h20) - 3.597739657143682) < 1e-12, "H20 agrees with its double value");

  // 5. Oracle: the Basel-style sum 1/k^2 is cross-checked with fixed-point integer accumulation.
  let exact = NUM(0);
  let fixedPoint = 0n;
  const unit = 10n ** 40n;
  for (let k = 1; k <= 50; k += 1) {
    exact = add(exact, rat(1n, BigInt(k * k)));
    fixedPoint += unit / BigInt(k * k);
  }
  assert(toDecimal(exact, 20) === toDecimal(rat(fixedPoint, unit), 20), `sum 1/k^2 ${toDecimal(exact, 20)} vs fixed point ${toDecimal(rat(fixedPoint, unit), 20)}`);
  assert(toDecimal(exact, 20).startsWith("1.6251327336"), "the partial sum is close to pi^2/6 minus the tail");

  // 6. Continued fractions: expansion and reconstruction are exact inverses.
  assert(continuedFraction(rat(415n, 93n)).join(",") === "4,2,6,7", `415/93 = ${continuedFraction(rat(415n, 93n))}`);
  for (const value of [rat(1n), rat(3n, 2n), rat(355n, 113n), rat(-3n, 2n), rat(123456789n, 987654321n)]) {
    assert(cmp(fromContinuedFraction(continuedFraction(value)), value) === 0, `round trip failed for ${value.n}/${value.d}`);
  }
  assert(continuedFraction(NUM(7)).length === 1, "an integer has a single continued-fraction term");

  // 7. Machin's formula reaches twenty digits of pi with exact rationals.
  const pi = machinPi(24);
  assert(toDecimal(pi, 20) === "3.14159265358979323846", `pi from Machin: ${toDecimal(pi, 20)}`);
  assert(cmp(pi, NUM(3)) > 0 && cmp(pi, NUM(4)) < 0, "the computed pi sits between 3 and 4");
  assert(toDecimal(machinPi(12), 8) === "3.14159265", "fewer series terms still give the first eight digits");

  // 8. Newton's square root: the residual is below the requested precision and the digits are right.
  const root2 = sqrtRat(NUM(2), 5);
  assert(cmp(abs(sub(mul(root2, root2), NUM(2))), rat(1n, 10n ** 20n)) < 0, "|x^2 - 2| is below 1e-20");
  assert(toDecimal(root2, 20) === "1.41421356237309504880", `sqrt(2) = ${toDecimal(root2, 20)}`);
  assert(toDecimal(sqrtRat(NUM(4), 5), 10) === "2.0000000000", "sqrt(4) converges to exactly two in decimal");

  // 9. Agreement with floating point where double precision is enough.
  assert(toNumber(add(rat(1n, 3n), rat(1n, 6n))) === 0.5, "exact sums convert to the expected double");
  const ordered = [rat(1n, 3n), rat(2n, 5n), rat(7n, 11n), rat(355n, 113n)];
  ordered.slice(1).forEach((value, i) => assert(cmp(ordered[i], value) < 0 === (toNumber(ordered[i]) < toNumber(value)), "ordering agrees with doubles"));

  // 10. Determinism and large exact results: everything is a pure function of its inputs.
  assert(same(harmonic(20), h20), "recomputing H20 gives the same fraction");
  assert(toDecimal(harmonic(30), 10) !== toDecimal(harmonic(20), 10) && cmp(harmonic(30), h20) > 0, "harmonic numbers grow with n");
  assert(cmp(Array.from({ length: 60 }, (_, i) => rat(1n, 2n ** BigInt(i + 1))).reduce(add, NUM(0)), sub(NUM(1), rat(1n, 2n ** 60n))) === 0, "a geometric series sums exactly");

  console.log(`rationals: H20=${h20.n}/${h20.d} pi=${toDecimal(pi, 20)} sqrt2=${toDecimal(root2, 20)} cf415_93=${continuedFraction(rat(415n, 93n)).join(";")}`);
  console.log("PASS");
}
module.exports = { gcd, rat, add, sub, mul, div, neg, cmp, abs, pow, toNumber, toDecimal, sqrtRat,
  continuedFraction, fromContinuedFraction, arctanInverse, machinPi, harmonic };
if (require.main === module) main();
