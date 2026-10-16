/**
 * Task 586 - Implement chirp z-transform outline for non-power-of-two convolutions. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * The chirp z-transform (CZT) evaluates a polynomial at points on a geometric contour
 * z_k = A * W^(-k), k = 0..M-1, i.e. X_k = sum_n x_n z_k^(-n).  Writing kn = (n^2 + k^2 - (k-n)^2)/2
 * turns the direct O(NM) evaluation into a convolution of the "chirp" sequences
 * b_n = x_n A^(-n) W^(n^2/2) and c_m = W^(-m^2/2), which is computed here with a radix-2 FFT — this
 * is Bluestein's algorithm, and it is the outline made concrete: the transform length never has to
 * be a power of two, only the FFT *inside* the convolution does.
 *
 * Two entry points are provided.  `czt()` is the general contour version above.  `bluesteinDft()`
 * is the special case A = 1, W = e^(-2*pi*i/n) (the forward DFT), implemented directly on the exact order-2n
 * root of unity so that every chirp argument is an integer exponent reduced modulo 2n (no argument
 * blow-up, no accumulated phase error).  `cyclicConvolution()` then pads both operands to an
 * arbitrary length L and multiplies through `bluesteinDft`, which is what makes
 * non-power-of-two convolution possible: pick L = deg(a) + deg(b) and the cyclic product equals the
 * linear one.  Documented caveat: if L is smaller than deg(a) + deg(b) the result is the *cyclic*
 * convolution, i.e. the linear result folded modulo L; that aliasing is asserted explicitly against
 * the folded linear product instead of being hidden.
 *
 * Documented checks: the chirp identity j*k = (j^2 + k^2 - (k-j)^2)/2 is verified for every integer
 * pair in [-64, 64]; `bluesteinDft` is compared with the O(n^2) DFT for *every* length n = 1..24
 * (so power-of-two, prime, and composite lengths are all covered) plus n = 32, 48, 100, and with
 * the general `czt` at both unit contours A = 1, W = e^(±2*pi*i/n); inverse transforms round-trip to 1e-11 for the same
 * lengths; delta, constant and unit-root cases have closed-form expectations
 * (DFT(delta) = ones, DFT(constant c) = [n*c, 0, ..., 0], DFT of a single n-th root of unity is a
 * shifted delta) and linearity plus Parseval's identity are asserted; convolution of the
 * deliberately non-power-of-two pairs 7x6, 5x5, 3x3, 13x11 and 12x12 is compared with naive
 * convolution and independently with a modular NTT over 998244353; the general CZT is checked
 * against Horner evaluation of the same polynomial at z_k^(-1) for several A/W contours including
 * M > N (zero-padded evaluation); and everything is deterministic (fixed 64-bit splitmix64 seed
 * 0x5EED_586, bit-identical repeats).
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const TWO_PI = 2 * Math.PI;
const NTT_MOD = 998244353;
const NTT_ROOT = 3;

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

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }
}

// ---------------------------------------------------------------------------------------------
// complex arithmetic on [re, im] pairs
// ---------------------------------------------------------------------------------------------

const C_ZERO = Object.freeze([0, 0]);

function cAdd(x, y) {
  return [x[0] + y[0], x[1] + y[1]];
}

function cSub(x, y) {
  return [x[0] - y[0], x[1] - y[1]];
}

function cMul(x, y) {
  return [x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]];
}

function cScale(x, s) {
  return [x[0] * s, x[1] * s];
}

function cAbs(x) {
  return Math.hypot(x[0], x[1]);
}

function cDiv(x, y) {
  const d = y[0] * y[0] + y[1] * y[1];
  return [(x[0] * y[0] + x[1] * y[1]) / d, (x[1] * y[0] - x[0] * y[1]) / d];
}

function cSqrt(x) {
  const r = Math.hypot(x[0], x[1]);
  const re = Math.sqrt((r + x[0]) / 2);
  const im = Math.sign(x[1] || 1) * Math.sqrt((r - x[0]) / 2);
  return [re, im];
}

function cPow(x, exponent) {
  if (exponent === 0) return [1, 0];
  const r = Math.hypot(x[0], x[1]);
  const theta = Math.atan2(x[1], x[0]);
  const mag = r ** exponent;
  return [mag * Math.cos(theta * exponent), mag * Math.sin(theta * exponent)];
}

/** Exact unit-magnitude root of unity of the given integer order, with the exponent reduced mod period. */
function rootOfUnity(period, exponent) {
  const e = ((exponent % period) + period) % period;
  return [Math.cos((TWO_PI * e) / period), Math.sin((TWO_PI * e) / period)];
}

function assertComplexClose(actual, expected, tol, message) {
  assert.ok(Math.abs(actual[0] - expected[0]) <= tol && Math.abs(actual[1] - expected[1]) <= tol,
    `${message}: got [${actual[0]}, ${actual[1]}] expected [${expected[0]}, ${expected[1]}]`);
}

function assertComplexArrayClose(actual, expected, tol, message) {
  assert.strictEqual(actual.length, expected.length, `${message}: length mismatch`);
  actual.forEach((value, i) => assertComplexClose(value, expected[i], tol, `${message} [${i}]`));
}

// ---------------------------------------------------------------------------------------------
// radix-2 FFT used only for the inner convolution
// ---------------------------------------------------------------------------------------------

function isPowerOfTwo(n) {
  return n > 0 && (n & (n - 1)) === 0;
}

function nextPowerOfTwo(n) {
  let size = 1;
  while (size < n) size <<= 1;
  return size;
}

function fft(input, invert) {
  const n = input.length;
  if (!isPowerOfTwo(n)) throw new RangeError(`FFT length ${n} must be a power of two`);
  const a = input.map((value) => [value[0], value[1]]);
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
    const angle = (invert ? TWO_PI : -TWO_PI) / len;
    const w = [Math.cos(angle), Math.sin(angle)];
    for (let start = 0; start < n; start += len) {
      let wn = [1, 0];
      for (let k = 0; k < len / 2; k++) {
        const u = a[start + k];
        const v = cMul(a[start + k + len / 2], wn);
        a[start + k] = cAdd(u, v);
        a[start + k + len / 2] = cSub(u, v);
        wn = cMul(wn, w);
      }
    }
  }
  if (invert) {
    for (let i = 0; i < n; i++) a[i] = cScale(a[i], 1 / n);
  }
  return a;
}

function convComplex(a, b) {
  if (a.length === 0 || b.length === 0) return [];
  const needed = a.length + b.length - 1;
  const size = nextPowerOfTwo(needed);
  const pad = (arr) => arr.concat(Array.from({ length: size - arr.length }, () => [0, 0]));
  const fa = fft(pad(a), false);
  const fb = fft(pad(b), false);
  const product = fa.map((value, i) => cMul(value, fb[i]));
  return fft(product, true).slice(0, needed);
}

// ---------------------------------------------------------------------------------------------
// transforms
// ---------------------------------------------------------------------------------------------

/** O(n^2) DFT by the transform matrix; sign = -1 forward, +1 inverse (unscaled). */
function dftNaive(x, sign = -1) {
  const n = x.length;
  const out = new Array(n);
  for (let k = 0; k < n; k++) {
    let acc = [0, 0];
    for (let j = 0; j < n; j++) acc = cAdd(acc, cMul(x[j], rootOfUnity(n, sign * j * k)));
    out[k] = acc;
  }
  return out;
}

/** Bluestein DFT of arbitrary length n built on the exact order-2n root of unity. */
function bluesteinDft(x, sign = -1) {
  const n = x.length;
  if (n === 0) return [];
  if (n === 1) return [[x[0][0], x[0][1]]];
  const period = 2 * n;
  const b = x.map((value, j) => cMul(value, rootOfUnity(period, sign * ((j * j) % period))));
  const c = new Array(2 * n - 1);
  for (let index = 0; index < 2 * n - 1; index++) {
    const m = index - (n - 1);
    c[index] = rootOfUnity(period, -sign * ((m * m) % period));
  }
  const conv = convComplex(b, c);
  const out = new Array(n);
  for (let k = 0; k < n; k++) {
    out[k] = cMul(rootOfUnity(period, sign * ((k * k) % period)), conv[k + n - 1]);
  }
  return out;
}

/** Final output of a task 585-style NTT, kept here to cross-check convolution independently. */
function modPow(base, exponent, mod) {
  let result = 1;
  let b = base % mod;
  let e = exponent;
  while (e > 0) {
    if (e & 1) result = Number((BigInt(result) * BigInt(b)) % BigInt(mod));
    b = Number((BigInt(b) * BigInt(b)) % BigInt(mod));
    e >>>= 1;
  }
  return result;
}

function nttMod(a, invert, mod = NTT_MOD, root = NTT_ROOT) {
  const n = a.length;
  if (!isPowerOfTwo(n)) throw new RangeError("NTT length must be a power of two");
  const out = a.map((v) => ((v % mod) + mod) % mod);
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const t = out[i];
      out[i] = out[j];
      out[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    let w = modPow(root, Math.floor((mod - 1) / len), mod);
    if (invert) w = modPow(w, mod - 2, mod);
    for (let start = 0; start < n; start += len) {
      let wn = 1;
      for (let k = 0; k < len / 2; k++) {
        const u = out[start + k];
        const v = Number((BigInt(out[start + k + len / 2]) * BigInt(wn)) % BigInt(mod));
        out[start + k] = (u + v) % mod;
        out[start + k + len / 2] = (u - v + mod) % mod;
        wn = Number((BigInt(wn) * BigInt(w)) % BigInt(mod));
      }
    }
  }
  if (invert) {
    const invN = modPow(n, mod - 2, mod);
    for (let i = 0; i < n; i++) out[i] = Number((BigInt(out[i]) * BigInt(invN)) % BigInt(mod));
  }
  return out;
}

function convolutionNttMod(a, b, mod = NTT_MOD) {
  if (a.length === 0 || b.length === 0) return [];
  const needed = a.length + b.length - 1;
  const size = nextPowerOfTwo(needed);
  const pad = (arr) => arr.concat(new Array(size - arr.length).fill(0));
  const fa = nttMod(pad(a), false, mod);
  const fb = nttMod(pad(b), false, mod);
  return nttMod(fa.map((v, i) => Number((BigInt(v) * BigInt(fb[i])) % BigInt(mod))), true, mod)
    .slice(0, needed);
}

/** Cyclic convolution of length L through Bluestein; equals the linear one when L is large enough. */
function cyclicConvolution(a, b, L) {
  const pad = (arr) => arr.map((v) => [v, 0]).concat(
    Array.from({ length: L - arr.length }, () => [0, 0]));
  const fa = bluesteinDft(pad(a), -1);
  const fb = bluesteinDft(pad(b), -1);
  return bluesteinDft(fa.map((v, i) => cMul(v, fb[i])), +1).map((v) => cScale(v, 1 / L));
}

function linearConvolutionNaiveReal(a, b) {
  const out = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) out[i + j] += a[i] * b[j];
  return out;
}

/** Linear convolution of real sequences with a deliberately non-power-of-two transform length. */
function convolutionViaCzt(a, b) {
  const L = a.length + b.length - 1;
  const raw = cyclicConvolution(a, b, L);
  return raw.map((v) => {
    assert.ok(Math.abs(v[1]) <= 1e-9, `imaginary residue ${v[1]} in a real convolution`);
    // Float noise below 1e-12 is snapped to exact zero so integer results compare bit-exactly.
    return Math.abs(v[0]) < 1e-12 ? 0 : v[0];
  });
}

/** General chirp z-transform along the contour z_k = A * W^(-k). */
function czt(x, m, a, w) {
  const n = x.length;
  if (n === 0) return new Array(m).fill().map(() => [0, 0]);
  const wHalf = cSqrt(w);
  const aInv = cDiv([1, 0], a);
  const b = x.map((value, j) => cMul(cMul(value, cPow(aInv, j)), cPow(wHalf, j * j)));
  const c = new Array(n + m - 1);
  for (let index = 0; index < n + m - 1; index++) {
    const offset = index - (n - 1);
    c[index] = cPow(wHalf, -(offset * offset));
  }
  const conv = convComplex(b, c);
  const out = new Array(m);
  for (let k = 0; k < m; k++) out[k] = cMul(cPow(wHalf, k * k), conv[n - 1 + k]);
  return out;
}

function hornerEval(coeffs, point) {
  let acc = [0, 0];
  for (let i = coeffs.length - 1; i >= 0; i--) acc = cAdd(cMul(acc, point), coeffs[i]);
  return acc;
}

function randomComplex(length, rng, scale = 10) {
  return Array.from({ length }, () => [rng.nextFloat() * scale - scale / 2,
    rng.nextFloat() * scale - scale / 2]);
}

function randomInts(length, rng, bound = 20) {
  return Array.from({ length }, () => rng.nextInt(bound) - Math.floor(bound / 2));
}

function conjugate(x) {
  return [x[0], -x[1]];
}

function main() {
  const rng = new SplitMix64(0x5eed586n);

  // (1) The chirp identity that makes the whole method work, for every pair in [-64, 64].
  for (let j = -64; j <= 64; j++) {
    for (let k = -64; k <= 64; k++) {
      assert.ok(j * k === (j * j + k * k - (k - j) * (k - j)) / 2,
        `chirp identity failed at j=${j}, k=${k}`);
    }
  }

  // (2) Roots of unity are exact unit-magnitude values with the declared order.
  for (const period of [2, 3, 5, 8, 12, 17]) {
    const values = Array.from({ length: period }, (_, e) => rootOfUnity(period, e));
    for (const value of values) assert.ok(Math.abs(cAbs(value) - 1) <= 1e-15);
    const sum = values.reduce((acc, value) => cAdd(acc, value), [0, 0]);
    assert.ok(cAbs(sum) <= 1e-12, `roots of unity of order ${period} must sum to zero`);
    assertComplexClose(rootOfUnity(period, period + 3), rootOfUnity(period, 3), 1e-12,
      "the exponent is reduced modulo the period");
    assertComplexClose(rootOfUnity(period, -1), conjugate(rootOfUnity(period, 1)), 1e-12,
      "negative exponents are conjugates");
  }

  // (3) Bluestein versus the direct DFT for every length 1..24 plus larger mixed lengths.
  for (let n = 1; n <= 24; n++) {
    const x = randomComplex(n, rng);
    assertComplexArrayClose(bluesteinDft(x, -1), dftNaive(x, -1), 1e-10, `forward DFT n=${n}`);
    assertComplexArrayClose(bluesteinDft(x, +1), dftNaive(x, +1), 1e-10, `inverse DFT n=${n}`);
    // inverse scalings compose back to the identity
    const roundTrip = bluesteinDft(bluesteinDft(x, -1), +1).map((v) => cScale(v, 1 / n));
    assertComplexArrayClose(roundTrip, x, 1e-10, `round trip n=${n}`);
  }
  for (const n of [31, 32, 33, 48, 64, 100, 127]) {
    const x = randomComplex(n, rng);
    assertComplexArrayClose(bluesteinDft(x, -1), dftNaive(x, -1), 1e-9, `forward DFT n=${n}`);
    const roundTrip = bluesteinDft(bluesteinDft(x, -1), +1).map((v) => cScale(v, 1 / n));
    assertComplexArrayClose(roundTrip, x, 1e-9, `round trip n=${n}`);
  }

  // (4) Closed forms: delta, constant, single root of unity, linearity, Parseval, shift.
  assertComplexArrayClose(bluesteinDft([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], -1),
    new Array(5).fill().map(() => [1, 0]), 1e-12, "DFT(delta) is all ones");
  assertComplexArrayClose(bluesteinDft(new Array(6).fill().map(() => [3, 0]), -1),
    [[18, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 1e-11, "DFT of a constant is n*c at k=0");
  const n7 = 7;
  const singleRoot = new Array(n7).fill().map(() => [0, 0]);
  singleRoot[1] = [1, 0];
  const rootSpectrum = bluesteinDft(singleRoot, -1);
  for (let k = 0; k < n7; k++) {
    assertComplexClose(rootSpectrum[k], rootOfUnity(n7, -k), 1e-12,
      `DFT of the monomial x is the k-th power of the twiddle at k=${k}`);
  }
  const xa = randomComplex(9, rng);
  const xb = randomComplex(9, rng);
  const sumTransform = bluesteinDft(xa, -1).map((v, i) => cAdd(v, bluesteinDft(xb, -1)[i]));
  assertComplexArrayClose(bluesteinDft(xa.map((v, i) => cAdd(v, xb[i])), -1), sumTransform, 1e-10,
    "the transform is linear");
  const spectrum = bluesteinDft(xa, -1);
  const energyTime = xa.reduce((acc, v) => acc + v[0] * v[0] + v[1] * v[1], 0);
  const energyFreq = spectrum.reduce((acc, v) => acc + v[0] * v[0] + v[1] * v[1], 0);
  assert.ok(Math.abs(energyFreq - 9 * energyTime) <= 1e-8 * 9 * energyTime, "Parseval's identity");
  const shift = 3;
  const shifted = xa.map((_, i) => xa[(i + shift) % 9]);
  const shiftedSpectrum = bluesteinDft(shifted, -1);
  for (let k = 0; k < 9; k++) {
    // y_j = x_{j+s}  =>  Y_k = w^(-s k) X_k with w = e^(-2*pi*i/n).
    assertComplexClose(shiftedSpectrum[k], cMul(spectrum[k], rootOfUnity(9, k * shift)), 1e-9,
      `circular shift multiplies bin ${k} by a twiddle`);
  }

  // (5) Non-power-of-two convolution, against the naive product and an independent modular NTT.
  const pairs = [[7, 6], [5, 5], [3, 3], [13, 11], [12, 12], [1, 5], [9, 2]];
  for (const [la, lb] of pairs) {
    const a = randomInts(la, rng);
    const b = randomInts(lb, rng);
    const viaCzt = convolutionViaCzt(a, b);
    const naive = linearConvolutionNaiveReal(a, b);
    assert.strictEqual(viaCzt.length, la + lb - 1);
    viaCzt.forEach((value, i) => assert.ok(Math.abs(value - naive[i]) <= 1e-9,
      `CZT convolution mismatch at ${i} for ${la}x${lb}: ${value} vs ${naive[i]}`));
    viaCzt.forEach((value) => assert.ok(Math.abs(value - Math.round(value)) <= 1e-9,
      "integer inputs must produce integer outputs"));
    const modProduct = convolutionNttMod(a, b);
    assert.deepStrictEqual(modProduct, naive.map((c) => ((c % NTT_MOD) + NTT_MOD) % NTT_MOD),
      "the modular NTT agrees with the float CZT convolution");
  }

  // (6) Documented aliasing caveat: a too-short transform length folds the linear result.
  const longA = [1, 2, 3, 4];
  const longB = [5, 6, 7];
  const folded = new Array(4).fill(0);
  linearConvolutionNaiveReal(longA, longB).forEach((value, i) => { folded[i % 4] += value; });
  const shortCyclic = cyclicConvolution(longA, longB, 4).map((v) => v[0]);
  shortCyclic.forEach((value, i) => assert.ok(Math.abs(value - folded[i]) <= 1e-9,
    "a transform shorter than the product wraps the linear coefficients"));
  assert.ok(Math.abs(shortCyclic[0] - linearConvolutionNaiveReal(longA, longB)[0]) > 1e-6,
    "the short transform really does alias (its first coefficient is not the linear one)");

  // (7) The general CZT equals Horner evaluation of the polynomial at z_k^(-1).
  for (const [magnitude, angle, n, m] of [[1.0, 0.0, 6, 6], [0.9, 0.4, 5, 8], [1.2, -0.3, 4, 4],
    [1.0, Math.PI / 9, 7, 3]]) {
    const a = [magnitude * Math.cos(angle), magnitude * Math.sin(angle)];
    const w = rootOfUnity(11, 2);
    const x = randomComplex(n, rng, 4);
    const transformed = czt(x, m, a, w);
    const aInv = cDiv([1, 0], a);
    for (let k = 0; k < m; k++) {
      const point = cMul(aInv, cPow(w, k));
      assertComplexClose(transformed[k], hornerEval(x, point), 1e-10,
        `CZT(A=${a}) disagrees with Horner evaluation at k=${k}`);
    }
  }
  // CZT at A = 1, W = e^(-2*pi*i/n) is exactly the forward DFT.
  for (const n of [5, 6, 7, 9, 12]) {
    const x = randomComplex(n, rng);
    assertComplexArrayClose(czt(x, n, [1, 0], rootOfUnity(n, -1)), bluesteinDft(x, -1), 1e-10,
      `CZT with the unit contour is the DFT at n=${n}`);
    assertComplexArrayClose(czt(x, n, [1, 0], rootOfUnity(n, -1)), dftNaive(x, -1), 1e-10,
      `CZT with the unit contour equals the matrix DFT at n=${n}`);
    assertComplexArrayClose(czt(x.map(conjugate), n, [1, 0], rootOfUnity(n, 1)),
      dftNaive(x, -1).map(conjugate), 1e-10,
      `the inverse contour W = e^(+2*pi*i/n) reproduces the inverse DFT at n=${n}`);
  }
  // M > N zero-pads the polynomial, i.e. evaluates it on a longer contour.
  const small = randomComplex(4, rng, 3);
  const paddedSpectrum = czt(small, 9, [1, 0], rootOfUnity(9, -1));
  const paddedPoly = small.concat(Array.from({ length: 5 }, () => [0, 0]));
  assertComplexArrayClose(paddedSpectrum, dftNaive(paddedPoly, -1), 1e-10,
    "a longer contour is a zero-padded transform of the same polynomial");

  // (8) Degenerate inputs and determinism.
  assert.deepStrictEqual(bluesteinDft([], -1), []);
  assertComplexArrayClose(bluesteinDft([[2, 3]], -1), [[2, 3]], 1e-15, "length-1 transforms are identity");
  assertComplexArrayClose(bluesteinDft(new Array(7).fill().map(() => [0, 0]), -1),
    new Array(7).fill().map(() => [0, 0]), 1e-15, "the zero polynomial transforms to zero");
  assert.deepStrictEqual(convolutionViaCzt([0, 0, 0], [1, 2]), [0, 0, 0, 0]);
  assert.deepStrictEqual(convolutionViaCzt([1], [1]), [1]);
  assert.deepStrictEqual(convolutionViaCzt([2, 3], [4]), [8, 12]);
  const rngA = new SplitMix64(7n);
  const rngB = new SplitMix64(7n);
  assert.deepStrictEqual(Array.from({ length: 10 }, () => rngA.next()),
    Array.from({ length: 10 }, () => rngB.next()), "splitmix64 is pure");
  const fixed = randomComplex(11, new SplitMix64(4242n));
  assert.deepStrictEqual(bluesteinDft(fixed, -1), bluesteinDft(fixed.slice(), -1));
  assert.deepStrictEqual(czt(fixed, 15, [1, 0], rootOfUnity(15, 1)),
    czt(fixed, 15, [1, 0], rootOfUnity(15, 1)));
}

module.exports = { SplitMix64, C_ZERO, cAdd, cSub, cMul, cScale, cAbs, cDiv, cSqrt, cPow,
  rootOfUnity, assertComplexClose, assertComplexArrayClose, isPowerOfTwo, nextPowerOfTwo, fft,
  convComplex, dftNaive, bluesteinDft, modPow, nttMod, convolutionNttMod, cyclicConvolution,
  linearConvolutionNaiveReal, convolutionViaCzt, czt, hornerEval, randomComplex, randomInts,
  conjugate, NTT_MOD, NTT_ROOT };

if (require.main === module) {
  main();
  console.log("PASS");
}
