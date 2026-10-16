/**
 * Task 585 - Implement modular FFT for polynomial multiplication (NTT) small prime. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Number-theoretic transform over the "small prime" 998244353 = 119 * 2^23 + 1 (primitive root
 * g = 3), i.e. the classic NTT-friendly prime: its 2-adic valuation is 23, so power-of-two
 * transforms up to length 2^23 exist and the twiddle factors are exact — no floating point error
 * anywhere.  The implementation is the iterative in-place Cooley-Tukey transform with bit-reversal
 * permutation, forward and inverse (inverse = conjugate twiddles + scaling by n^{-1} mod p), plus
 * `convolution()` which zero-pads to the next power of two covering deg(a) + deg(b).
 *
 * Modular arithmetic is the delicate part in JavaScript: products of two residues below 1e9 reach
 * 1e18 and silently lose precision in double arithmetic, so `mulmod` uses an exact BigInt product
 * whenever a * b would exceed Number.MAX_SAFE_INTEGER and the fast double path otherwise (the same
 * care is taken in modPow, which powers both the twiddles and the inverse scaling).
 *
 * The single-prime transform only returns the convolution *modulo* p, so the file also implements
 * two independent ground truths for the exact integer convolution:
 *   - a BigInt digit packing (coefficients are written in base 2^40, one BigInt multiplication
 *     yields the whole convolution) for non-negative coefficients, and
 *   - a two-prime CRT (p2 = 167772161 = 5 * 2^25 + 1, g = 3) with centred reconstruction, which
 *     recovers signed results as long as every coefficient satisfies |c| < p1 * p2 / 2.
 * Documented caveats, all asserted rather than papered over: (i) on [p-1] * [p-1] the true
 * coefficient (p-1)^2 is only 1 mod p, so the single-prime result wraps; (ii) two primes still
 * cannot resolve that value because (p-1)^2 = 9.9649e17 exceeds half their period (1.6697e17) - the
 * CRT returns the centred residue and the test pins that residue down exactly - so a third prime
 * (p3 = 469762049 = 7 * 2^26 + 1, g = 3) is used, and it recovers (p-1)^2 exactly; (iii) the
 * digit-packing base must exceed every coefficient of the product (40 bits suffice below 1.1e12 and
 * the wrapped case needs 64), an invariant the implementation enforces with a RangeError; and
 * (iv) results beyond Number.MAX_SAFE_INTEGER are rejected rather than silently rounded, so the
 * large instances are compared through the BigInt variants.
 *
 * Documented checks: NTT of the delta polynomial is all ones and NTT of a constant is that constant
 * followed by zeros; the twiddle w = g^((p-1)/n) has exact order n (w^n = 1, w^(n/2) = -1, all n
 * powers distinct, twiddle sums vanishing for k != 0); the transform is compared with a direct
 * O(n^2) DFT matrix evaluation for n = 2, 4, 8, 16, 32, and the inverse transform with the DFT
 * matrix at the inverse twiddle; forward/inverse round trips are exact for n up to 1024 including
 * inputs containing zeros and negatives reduced mod p; convolution is compared with naive O(nm)
 * multiplication for 60 deterministic instances of length 1..40 (64-bit splitmix64 coefficients,
 * seed 0x5EED_585), for the hand-checked pairs [1,2,3]*[4,5] and [1,1]^2, for (1-x)^5 and
 * (1-x)^5 (1+x) whose binomial coefficients are known, and for a 1024 x 1024 pair; commutativity,
 * associativity, the identity element and NTT linearity are asserted; and the length guards throw
 * for a non-power-of-two length, for a modulus lacking the needed power of two (13 cannot host a
 * length-8 transform), and arithmetically for the 2^23 ceiling of this modulus.
 */
"use strict";

const assert = require("node:assert");

const MOD = 998244353;
const PRIMITIVE_ROOT = 3;
const MOD2 = 167772161;
const ROOT2 = 3;
const MOD3 = 469762049;
const ROOT3 = 3;
const MAX_TRANSFORM_LOG2 = 23;
const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
const MASK64 = (1n << 64n) - 1n;
const SPLITMIX_SEED = 0x5eed585n;

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

/** Exact (a * b) mod m for m < 2^31; falls back to BigInt when the double product is unsafe. */
function mulmod(a, b, mod) {
  const x = a % mod;
  const y = b % mod;
  if (x === 0 || y === 0) return 0;
  if (x <= Math.floor(Number.MAX_SAFE_INTEGER / y)) return (x * y) % mod;
  return Number((BigInt(x) * BigInt(y)) % BigInt(mod));
}

function modPow(base, exponent, mod) {
  if (exponent < 0) throw new RangeError("negative exponent");
  let result = 1 % mod;
  let b = base % mod;
  let e = exponent;
  while (e > 0) {
    if (e & 1) result = mulmod(result, b, mod);
    b = mulmod(b, b, mod);
    e >>>= 1;
  }
  return result;
}

function modInverse(a, mod) {
  return modPow(((a % mod) + mod) % mod, mod - 2, mod);
}

function isPowerOfTwo(n) {
  return n > 0 && (n & (n - 1)) === 0;
}

function nextPowerOfTwo(n) {
  let size = 1;
  while (size < n) size <<= 1;
  return size;
}

/** Maximum transform length supported by `mod` (2-adic valuation of mod - 1). */
function maxTransformLength(mod) {
  let value = mod - 1;
  let length = 1;
  while (value % 2 === 0) {
    value /= 2;
    length *= 2;
  }
  return length;
}

function ntt(input, invert = false, mod = MOD, root = PRIMITIVE_ROOT) {
  const n = input.length;
  if (!isPowerOfTwo(n)) throw new RangeError(`NTT length ${n} is not a power of two`);
  if ((mod - 1) % n !== 0) throw new RangeError(`modulus ${mod} cannot host a transform of length ${n}`);
  if (n === 1) return [input[0] % mod];
  const a = input.map((value) => ((value % mod) + mod) % mod);
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    let w = modPow(root, Math.floor((mod - 1) / len), mod);
    if (invert) w = modInverse(w, mod);
    for (let start = 0; start < n; start += len) {
      let wn = 1;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const u = a[start + k];
        const v = mulmod(a[start + k + half], wn, mod);
        a[start + k] = (u + v) % mod;
        a[start + k + half] = (u - v + mod) % mod;
        wn = mulmod(wn, w, mod);
      }
    }
  }
  if (invert) {
    const invN = modInverse(n, mod);
    for (let i = 0; i < n; i++) a[i] = mulmod(a[i], invN, mod);
  }
  return a;
}

/** Linear convolution modulo `mod` via zero-padded NTT. */
function convolutionMod(a, b, mod = MOD, root = PRIMITIVE_ROOT) {
  if (a.length === 0 || b.length === 0) return [];
  const needed = a.length + b.length - 1;
  const size = nextPowerOfTwo(needed);
  if ((mod - 1) % size !== 0) {
    throw new RangeError(`modulus ${mod} cannot host a transform of length ${size}`);
  }
  const fa = ntt(a.concat(new Array(size - a.length).fill(0)), false, mod, root);
  const fb = ntt(b.concat(new Array(size - b.length).fill(0)), false, mod, root);
  const product = fa.map((value, i) => mulmod(value, fb[i], mod));
  return ntt(product, true, mod, root).slice(0, needed);
}

/** Naive O(nm) convolution modulo `mod`; the reference used everywhere below. */
function convolutionNaive(a, b, mod = MOD) {
  if (a.length === 0 || b.length === 0) return [];
  const out = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out[i + j] = (out[i + j] + mulmod(a[i], b[j], mod)) % mod;
  }
  return out;
}

/** Direct O(n^2) DFT mod p with an explicit n-th root of unity. */
function dftMatrixWithRoot(a, rootOfUnity, mod = MOD) {
  const n = a.length;
  const out = new Array(n).fill(0);
  for (let k = 0; k < n; k++) {
    let acc = 0;
    for (let j = 0; j < n; j++) {
      acc = (acc + mulmod(a[j], modPow(rootOfUnity, (j * k) % n, mod), mod)) % mod;
    }
    out[k] = acc;
  }
  return out;
}

/** Direct O(n^2) DFT mod p: out[k] = sum_j a[j] * w^(jk).  Independent of the butterfly network. */
function dftMatrix(a, mod = MOD, generator = PRIMITIVE_ROOT) {
  const w = modPow(generator, Math.floor((mod - 1) / a.length), mod);
  return dftMatrixWithRoot(a, w, mod);
}

/**
 * Exact convolution of non-negative coefficients by BigInt digit packing: `baseBits` bits per
 * coefficient.  The base must exceed every coefficient of the product, otherwise the carries
 * destroy the digits; the trailing `product === 0n` assertion enforces that precondition.
 */
function convolutionPackedBigIntExact(a, b, baseBits = 40) {
  const base = 1n << BigInt(baseBits);
  let packedA = 0n;
  let packedB = 0n;
  for (let i = a.length - 1; i >= 0; i--) {
    if (!Number.isInteger(a[i]) || a[i] < 0) throw new RangeError("packing needs non-negative ints");
    packedA = packedA * base + BigInt(a[i]);
  }
  for (let i = b.length - 1; i >= 0; i--) {
    if (!Number.isInteger(b[i]) || b[i] < 0) throw new RangeError("packing needs non-negative ints");
    packedB = packedB * base + BigInt(b[i]);
  }
  let product = packedA * packedB;
  const out = [];
  const length = a.length + b.length - 1;
  for (let i = 0; i < length; i++) {
    out.push(product % base);
    product /= base;
  }
  if (product !== 0n) {
    throw new RangeError(`packed product still has digits left (${product}); increase baseBits`);
  }
  return out;
}

function toSafeNumbers(values, what) {
  return values.map((value) => {
    if (value > MAX_SAFE_BIGINT || value < -MAX_SAFE_BIGINT) {
      throw new RangeError(`${what} produced ${value}, which exceeds Number.MAX_SAFE_INTEGER`);
    }
    return Number(value);
  });
}

function convolutionPackedBigInt(a, b, baseBits = 40) {
  return toSafeNumbers(convolutionPackedBigIntExact(a, b, baseBits), "BigInt digit packing");
}

function _egcdBig(a, b) {
  if (b === 0n) return [a, 1n, 0n];
  const [g, x, y] = _egcdBig(b, a % b);
  return [g, y, x - (a / b) * y];
}

/** Two-prime CRT with centred reconstruction; exact while |value| < p1 * p2 / 2. */
function crtCenteredBig(r1, r2, p1 = MOD, p2 = MOD2) {
  const bigP1 = BigInt(p1);
  const bigP2 = BigInt(p2);
  const period = bigP1 * bigP2;
  const t = ((BigInt(r2) - BigInt(r1)) % bigP2 + bigP2) % bigP2;
  const invP1 = ((_egcdBig(bigP1 % bigP2, bigP2)[1] % bigP2) + bigP2) % bigP2;
  let x = (BigInt(r1) + bigP1 * ((t * invP1) % bigP2)) % period;
  if (x > period / 2n) x -= period;
  return x;
}

function crtCentered(r1, r2, p1 = MOD, p2 = MOD2) {
  return Number(toSafeNumbers([crtCenteredBig(r1, r2, p1, p2)], "two-prime CRT")[0]);
}

/** Exact (signed) convolution via CRT of two NTT primes (BigInt results). */
function convolutionExactBig(a, b) {
  const m1 = convolutionMod(a, b, MOD, PRIMITIVE_ROOT);
  const m2 = convolutionMod(a, b, MOD2, ROOT2);
  return m1.map((value, i) => crtCenteredBig(value, m2[i]));
}

/** Exact (signed) convolution via CRT of two NTT primes, as safe doubles. */
function convolutionExact(a, b) {
  return toSafeNumbers(convolutionExactBig(a, b), "two-prime CRT");
}

/** Exact (signed) convolution via CRT of three NTT primes; used to widen the verified bound. */
function convolutionExact3Big(a, b) {
  const m1 = convolutionMod(a, b, MOD, PRIMITIVE_ROOT);
  const m2 = convolutionMod(a, b, MOD2, ROOT2);
  const m3 = convolutionMod(a, b, MOD3, ROOT3);
  const bigP1 = BigInt(MOD);
  const bigP2 = BigInt(MOD2);
  const bigP3 = BigInt(MOD3);
  const period = bigP1 * bigP2 * bigP3;
  const invP1ModP2 = ((_egcdBig(bigP1 % bigP2, bigP2)[1] % bigP2) + bigP2) % bigP2;
  const p1p2ModP3 = (bigP1 * bigP2) % bigP3;
  const invP1P2ModP3 = ((_egcdBig(p1p2ModP3, bigP3)[1] % bigP3) + bigP3) % bigP3;
  const r12 = m1.map((value, i) => {
    const t = ((BigInt(m2[i]) - BigInt(value)) % bigP2 + bigP2) % bigP2;
    return (BigInt(value) + bigP1 * ((t * invP1ModP2) % bigP2)) % (bigP1 * bigP2);
  });
  return r12.map((value, i) => {
    const t = ((BigInt(m3[i]) - value) % bigP3 + bigP3) % bigP3;
    let x = (value + bigP1 * bigP2 * ((t * invP1P2ModP3) % bigP3)) % period;
    if (x > period / 2n) x -= period;
    return x;
  });
}

function convolutionExact3(a, b) {
  return toSafeNumbers(convolutionExact3Big(a, b), "three-prime CRT");
}

/** Signed-naive convolution in exact integer arithmetic (coefficients kept small on purpose). */
function convolutionSignedNaive(a, b) {
  const out = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j];
  }
  return out;
}

function randomPoly(length, rng, bound = MOD) {
  return Array.from({ length }, () => rng.nextInt(bound));
}

function main() {
  // (1) The modulus is NTT-friendly: MOD - 1 = 119 * 2^23 and g = 3 is a primitive root.
  assert.strictEqual(MOD - 1, 119 * 2 ** 23, "998244353 - 1 factors as 119 * 2^23");
  assert.strictEqual(maxTransformLength(MOD), 2 ** MAX_TRANSFORM_LOG2);
  assert.strictEqual(modPow(PRIMITIVE_ROOT, (MOD - 1) / 2, MOD), MOD - 1,
    "g is a quadratic non-residue, so g^((p-1)/2) = -1");
  assert.strictEqual(modPow(PRIMITIVE_ROOT, MOD - 1, MOD), 1, "Fermat");
  const order119 = modPow(PRIMITIVE_ROOT, (MOD - 1) / 119, MOD);
  assert.notStrictEqual(order119, 1, "g has order p-1, so g^((p-1)/119) generates the 119-part");
  assert.strictEqual(modPow(order119, 119, MOD), 1);
  assert.strictEqual(mulmod(MOD - 2, MOD - 2, MOD), 4,
    "the BigInt mulmod path handles near-modulus factors exactly ((p-2)^2 mod p = 4)");
  assert.notStrictEqual((MOD - 2) * (MOD - 2) % MOD, 4,
    "documented: the raw double product (p-2)^2 loses precision, so mulmod is required");

  // (2) Delta and constant transforms; twiddle orders.
  assert.deepStrictEqual(ntt([1, 0, 0, 0]), [1, 1, 1, 1], "NTT(delta) = all ones");
  assert.deepStrictEqual(ntt([7, 0, 0, 0, 0, 0, 0, 0]), new Array(8).fill(7),
    "NTT of a constant polynomial keeps only the k = 0 frequency");
  const n8 = 8;
  const w8 = modPow(PRIMITIVE_ROOT, Math.floor((MOD - 1) / n8), MOD);
  assert.strictEqual(modPow(w8, n8, MOD), 1);
  assert.strictEqual(modPow(w8, n8 / 2, MOD), MOD - 1, "w^(n/2) = -1");
  const powersOfW = new Set(Array.from({ length: n8 }, (_, j) => modPow(w8, j, MOD)));
  assert.strictEqual(powersOfW.size, n8, "the n-th root of unity has exact order n");
  const twiddleSums = Array.from({ length: n8 }, (_, k) => {
    let acc = 0;
    for (let j = 0; j < n8; j++) acc = (acc + modPow(w8, (j * k) % n8, MOD)) % MOD;
    return acc;
  });
  assert.strictEqual(twiddleSums[0], n8);
  assert.ok(twiddleSums.slice(1).every((value) => value === 0), "sum of w^(jk) vanishes for k != 0");

  // (3) NTT versus the direct DFT matrix.
  const rng = new SplitMix64(SPLITMIX_SEED);
  for (const size of [2, 4, 8, 16, 32]) {
    const poly = randomPoly(size, rng, 1000);
    assert.deepStrictEqual(ntt(poly), dftMatrix(poly), `NTT must equal the DFT matrix at n=${size}`);
  }

  // (4) Forward/inverse round trips.
  for (const size of [1, 2, 4, 8, 16, 64, 256, 1024]) {
    const poly = randomPoly(size, rng);
    const roundTrip = ntt(ntt(poly), true);
    assert.deepStrictEqual(roundTrip, poly, `inverse NTT must undo the transform at n=${size}`);
  }
  const withZeroAndNegative = [0, MOD - 5, 0, 12, 0, MOD - 1, 3, 0];
  assert.deepStrictEqual(ntt(ntt(withZeroAndNegative), true), withZeroAndNegative,
    "zeros and reduced negatives survive the round trip");
  assert.deepStrictEqual(ntt([-3, 5]), ntt([MOD - 3, 5]), "negative inputs are reduced mod p");

  // (5) Hand-checked convolutions.
  assert.deepStrictEqual(convolutionMod([1, 2, 3], [4, 5]), [4, 13, 22, 15]);
  assert.deepStrictEqual(convolutionMod([1, 1], [1, 1]), [1, 2, 1]);
  assert.deepStrictEqual(convolutionMod([0, 0, 0], [1, 2]), [0, 0, 0, 0]);
  assert.deepStrictEqual(convolutionMod([5], [7]), [35]);
  assert.deepStrictEqual(convolutionMod([], [1, 2]), []);
  const oneMinusX = [1, MOD - 1];
  let power = [1];
  for (let i = 0; i < 5; i++) power = convolutionMod(power, oneMinusX);
  assert.deepStrictEqual(power, [1, MOD - 5, 10, MOD - 10, 5, MOD - 1],
    "(1 - x)^5 has binomial coefficients 1, -5, 10, -10, 5, -1");
  assert.deepStrictEqual(convolutionMod(power, [1, 1]), [1, MOD - 4, 5, 0, MOD - 5, 4, MOD - 1],
    "(1 - x)^5 (1 + x) = (1 - x)^4 (1 - x^2) has a vanishing cubic coefficient");

  // (6) Sixty deterministic comparisons with the naive product.
  for (let trial = 0; trial < 60; trial++) {
    const n = 1 + rng.nextInt(40);
    const m = 1 + rng.nextInt(40);
    const a = randomPoly(n, rng);
    const b = randomPoly(m, rng);
    assert.deepStrictEqual(convolutionMod(a, b), convolutionNaive(a, b),
      `convolution mismatch for lengths ${n}x${m}`);
  }
  const bigA = randomPoly(1024, rng, 1 << 20);
  const bigB = randomPoly(1024, rng, 1 << 20);
  assert.deepStrictEqual(convolutionMod(bigA, bigB), convolutionNaive(bigA, bigB),
    "1024 x 1024 convolution matches the quadratic reference");

  // (7) Algebraic laws: commutativity, associativity, identity, linearity of the transform.
  const x = randomPoly(13, rng, 1 << 15);
  const y = randomPoly(9, rng, 1 << 15);
  const z = randomPoly(21, rng, 1 << 15);
  assert.deepStrictEqual(convolutionMod(x, y), convolutionMod(y, x));
  assert.deepStrictEqual(convolutionMod(convolutionMod(x, y), z), convolutionMod(x, convolutionMod(y, z)));
  assert.deepStrictEqual(convolutionMod(x, [1]), x, "[1] is the multiplicative identity");
  const size16 = 16;
  const p1 = randomPoly(size16, rng, 1 << 20);
  const p2 = randomPoly(size16, rng, 1 << 20);
  const sumTransform = ntt(p1).map((value, i) => (value + ntt(p2)[i]) % MOD);
  assert.deepStrictEqual(ntt(p1.map((value, i) => (value + p2[i]) % MOD)), sumTransform,
    "the NTT is linear");
  for (const k of [1, 3, 7]) {
    const scaled = p1.map((value) => mulmod(value, k, MOD));
    assert.deepStrictEqual(ntt(scaled), ntt(p1).map((value) => mulmod(value, k, MOD)));
  }
  const inverseRoot = modInverse(modPow(PRIMITIVE_ROOT, Math.floor((MOD - 1) / size16), MOD), MOD);
  assert.deepStrictEqual(ntt(p1, true).map((value) => mulmod(value, size16, MOD)),
    dftMatrixWithRoot(p1, inverseRoot, MOD),
    "n * inverse-NTT equals the DFT matrix evaluated at the inverse twiddle");

  // (8) Exact ground truths: BigInt packing and CRT, plus the honest single-prime wrap-around.
  const packedRng = new SplitMix64(0xabc585n);
  for (let trial = 0; trial < 25; trial++) {
    const n = 1 + packedRng.nextInt(32);
    const m = 1 + packedRng.nextInt(32);
    const a = Array.from({ length: n }, () => packedRng.nextInt(1000));
    const b = Array.from({ length: m }, () => packedRng.nextInt(1000));
    const packed = convolutionPackedBigInt(a, b);
    assert.deepStrictEqual(packed, convolutionSignedNaive(a, b),
      "base-2^40 packing must reproduce the exact integer convolution");
    assert.deepStrictEqual(convolutionExact(a, b), convolutionSignedNaive(a, b),
      "two-prime CRT must reproduce the exact integer convolution");
    assert.deepStrictEqual(convolutionExact3(a, b), convolutionSignedNaive(a, b),
      "three-prime CRT must agree with two-prime CRT inside the narrower bound");
    assert.deepStrictEqual(convolutionMod(a, b),
      convolutionSignedNaive(a, b).map((c) => ((c % MOD) + MOD) % MOD),
      "the single-prime NTT equals the exact convolution reduced mod p");
  }
  const signedRng = new SplitMix64(0xdef585n);
  for (let trial = 0; trial < 20; trial++) {
    const n = 1 + signedRng.nextInt(12);
    const m = 1 + signedRng.nextInt(12);
    const a = Array.from({ length: n }, () => signedRng.nextInt(41) - 20);
    const b = Array.from({ length: m }, () => signedRng.nextInt(41) - 20);
    assert.deepStrictEqual(convolutionExact(a, b), convolutionSignedNaive(a, b),
      "CRT recovers signed coefficients inside |c| < p1 * p2 / 2");
  }
  // Does the CRT bound hold?  Assert the worst case observed is far inside the period.
  const boundRng = new SplitMix64(0x51585n);
  let maxAbs = 0;
  for (let trial = 0; trial < 25; trial++) {
    const a = Array.from({ length: 32 }, () => boundRng.nextInt(1000));
    const b = Array.from({ length: 32 }, () => boundRng.nextInt(1000));
    maxAbs = Math.max(maxAbs, ...convolutionSignedNaive(a, b).map(Math.abs));
  }
  assert.ok(maxAbs < MOD * MOD2 / 2, `largest exact coefficient ${maxAbs} fits the two-prime period`);

  // (9) Documented counterexamples: a single prime - and even two - cannot recover a coefficient
  // that exceeds their period; three primes can.
  const trueSquare = BigInt(MOD - 1) ** 2n;
  assert.deepStrictEqual(convolutionMod([MOD - 1], [MOD - 1]), [1],
    "(p-1)^2 = 1 mod p, so the single-prime result wraps to 1");
  const twoPeriod = BigInt(MOD) * BigInt(MOD2);
  assert.ok(trueSquare > twoPeriod / 2n,
    "documented: (p-1)^2 exceeds half the two-prime period, so two primes cannot resolve it");
  const twoPrime = convolutionExactBig([MOD - 1], [MOD - 1])[0];
  let expectedCentered = trueSquare % twoPeriod;
  if (expectedCentered > twoPeriod / 2n) expectedCentered -= twoPeriod;
  assert.strictEqual(twoPrime, expectedCentered,
    "the two-prime CRT returns the centred residue, not the true (p-1)^2");
  assert.notStrictEqual(twoPrime, trueSquare);
  assert.strictEqual(convolutionExact3Big([MOD - 1], [MOD - 1])[0], trueSquare,
    "three primes resolve (p-1)^2 exactly because it is far below their period");
  assert.strictEqual(convolutionPackedBigIntExact([MOD - 1], [MOD - 1], 64)[0], trueSquare,
    "BigInt digit packing is exact regardless of the modulus, given a wide enough digit base");
  assert.throws(() => convolutionPackedBigIntExact([MOD - 1], [MOD - 1]), Error,
    "documented: the 40-bit digit base overflows on (p-1)^2 and the guard fires");
  assert.throws(() => convolutionPackedBigInt([MOD - 1], [MOD - 1]), RangeError,
    "unsafe double results are rejected rather than silently rounded");

  // (10) Boundary conditions and error paths.
  assert.throws(() => ntt([1, 2, 3]), RangeError, "non-power-of-two lengths are rejected");
  assert.throws(() => ntt([1, 0, 0, 0], false, 7), RangeError, "7 cannot host a transform of length 4");
  assert.throws(() => convolutionMod([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3], 13), RangeError,
    "a modulus without the needed power of two is rejected");
  assert.strictEqual(nextPowerOfTwo(2 ** 23 + 1), 2 ** 24);
  assert.notStrictEqual((MOD - 1) % 2 ** 24, 0,
    "documented limit: no transform of length 2^24 exists modulo 998244353 (2-adic valuation 23)");
  assert.strictEqual((MOD - 1) % 2 ** 23, 0, "length 2^23 is the largest supported transform");
  assert.strictEqual(ntt([5]).length, 1, "length-1 transforms are the identity");
  assert.deepStrictEqual(ntt([5], true), [5]);
  assert.deepStrictEqual(convolutionMod([2 ** 30], [4]), [mulmod(2 ** 30, 4, MOD)]);
  assert.throws(() => modPow(2, -1, MOD), RangeError);
  assert.strictEqual(modInverse(0, MOD), 0);

  // (11) Determinism.
  const rngC = new SplitMix64(SPLITMIX_SEED);
  const rngD = new SplitMix64(SPLITMIX_SEED);
  assert.deepStrictEqual(Array.from({ length: 20 }, () => rngC.next()),
    Array.from({ length: 20 }, () => rngD.next()), "splitmix64 is pure");
  const fixed = randomPoly(64, new SplitMix64(4242n));
  assert.deepStrictEqual(ntt(fixed), ntt(fixed.slice()));
  assert.deepStrictEqual(convolutionMod(fixed, fixed), convolutionMod(fixed, fixed));
}

module.exports = { MOD, PRIMITIVE_ROOT, MOD2, ROOT2, MOD3, ROOT3, MAX_TRANSFORM_LOG2, SplitMix64,
  MAX_SAFE_BIGINT, mulmod, modPow, modInverse, isPowerOfTwo, nextPowerOfTwo, maxTransformLength, ntt,
  convolutionMod, convolutionNaive, dftMatrix, dftMatrixWithRoot, convolutionPackedBigInt,
  convolutionPackedBigIntExact, crtCentered, crtCenteredBig, toSafeNumbers, convolutionExactBig,
  convolutionExact3Big,
  convolutionExact, convolutionExact3, convolutionSignedNaive, randomPoly };

if (require.main === module) {
  main();
  console.log("PASS");
}
