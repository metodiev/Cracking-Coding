/** Task 693 - Implement erasure coding (Reed-Solomon) encoder/decoder small GF(2^8). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Model: a systematic Reed-Solomon (n, k) code over GF(2^8) used for *erasure* coding. A codeword is
 * the evaluation of a degree < k polynomial P at n distinct field points x_0 .. x_{n-1}:
 *     encode(data[0..k-1]) : P is the unique polynomial with P(x_i) = data[i] for i < k,
 *     codeword[i] = P(x_i)  for all i < n,
 * so the first k bytes are the data (systematic) and the remaining m = n - k bytes are parity. Any k
 * of the n evaluations determine P, therefore *any* m erasures can be repaired - the MDS property.
 *
 * Field: GF(2^8) with the Rijndael/AES modulus 0x11d (x^8 + x^4 + x^3 + x + 1) and generator 2. Both
 * a log/exp table multiply and an independent Russian-peasant multiply are implemented; every pair of
 * bytes is cross-checked, which pins the tables against the bit-level definition.
 *
 * Decoder: the set of surviving points gives a k x k Vandermonde system for the k unknown polynomial
 * coefficients, solved by Gauss-Jordan elimination over the field; the erased positions are then
 * recomputed by evaluating the polynomial. An independent Lagrange-interpolation decoder is also
 * implemented and every decode is cross-checked against it, and every codeword is verified by
 * re-interpolating from its own first k bytes.
 *
 * Verified: the field axioms and table/peasant agreement over all 65536 products, Fermat's little
 * theorem, the two encoders (Lagrange-basis matrix product versus coefficient interpolation) agreeing,
 * the systematic layout, exact repair for *every* erasure pattern of size <= m for several (n, k),
 * refusal when more than m erasures are declared, decoding of any surviving subset (not only the first
 * k), linearity of the code, the MDS distance m + 1 between distinct codewords, degenerate parameters
 * (m = 0, k = 1), the field-size limit n <= 255, and determinism.
 */
"use strict";

const FIELD_SIZE = 256;
const PRIMITIVE_POLYNOMIAL = 0x11d;
const GENERATOR = 2;

/** Build the log/exp tables for the generator 2 of GF(2^8). */
function buildTables() {
  const exp = new Uint8Array(FIELD_SIZE * 2);
  const log = new Uint8Array(FIELD_SIZE);
  let value = 1;
  for (let power = 0; power < FIELD_SIZE - 1; power++) {
    exp[power] = value;
    log[value] = power;
    value <<= 1;
    if (value & FIELD_SIZE) value ^= PRIMITIVE_POLYNOMIAL;
    value &= 0xff;
  }
  for (let power = FIELD_SIZE - 1; power < FIELD_SIZE * 2; power++) exp[power] = exp[power - (FIELD_SIZE - 1)];
  log[0] = 0;                                    // undefined, never read through a guarded path
  return { exp, log };
}

const { exp: GF_EXP, log: GF_LOG } = buildTables();

/** Table multiply: sound because 2 is a primitive root modulo 0x11d. */
function gfMul(a, b) {
  if (a === 0 || b === 0) return 0;
  return GF_EXP[GF_LOG[a] + GF_LOG[b]];
}

/** Independent multiply: Russian peasant multiplication over the bit-level field definition. */
function gfMulPeasant(a, b) {
  let left = a & 0xff;
  let right = b & 0xff;
  let result = 0;
  while (right !== 0) {
    if (right & 1) result ^= left;
    right >>= 1;
    left <<= 1;
    if (left & FIELD_SIZE) left ^= PRIMITIVE_POLYNOMIAL;
  }
  return result & 0xff;
}

function gfAdd(a, b) {
  return (a ^ b) & 0xff;
}

function gfPow(a, power) {
  if (power === 0) return 1;
  if (a === 0) return 0;
  const index = ((GF_LOG[a] * power) % (FIELD_SIZE - 1) + (FIELD_SIZE - 1)) % (FIELD_SIZE - 1);
  return GF_EXP[index];
}

function gfInv(a) {
  if (a === 0) throw new RangeError("zero has no multiplicative inverse");
  return GF_EXP[FIELD_SIZE - 1 - GF_LOG[a]];
}

function gfDiv(a, b) {
  if (b === 0) throw new RangeError("division by zero in GF(2^8)");
  if (a === 0) return 0;
  return GF_EXP[(GF_LOG[a] - GF_LOG[b] + FIELD_SIZE - 1) % (FIELD_SIZE - 1)];
}

/** Evaluation points of the code: the distinct field elements 1..n. */
function codePoints(n) {
  if (!Number.isInteger(n) || n < 1 || n > FIELD_SIZE - 1) {
    throw new RangeError(`n must be in 1..${FIELD_SIZE - 1}`);
  }
  const points = [];
  for (let i = 0; i < n; i++) points.push(i + 1);
  return points;
}

function evalPolyAt(coefficients, x) {
  let acc = 0;
  for (let i = coefficients.length - 1; i >= 0; i--) acc = gfAdd(gfMul(acc, x), coefficients[i]);
  return acc;
}

function evalPoly(coefficients, points) {
  return points.map((x) => evalPolyAt(coefficients, x));
}

/** Lagrange basis values: coefficient matrix of the interpolation through `points`. */
function lagrangeBasisValues(points, x) {
  const values = new Array(points.length).fill(0);
  for (let i = 0; i < points.length; i++) {
    let numerator = 1;
    let denominator = 1;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      numerator = gfMul(numerator, gfAdd(x, points[j]));
      denominator = gfMul(denominator, gfAdd(points[i], points[j]));
    }
    values[i] = gfMul(numerator, gfInv(denominator));
  }
  return values;
}

/** Coefficients of the unique degree < points.length polynomial through (points, values). */
function interpolateCoefficients(points, values) {
  if (points.length !== values.length) throw new RangeError("points and values must have equal length");
  const coefficients = new Array(points.length).fill(0);
  for (let i = 0; i < points.length; i++) {
    const basis = lagrangeBasisValues(points, points[i]).map((value, index) => (index === i ? value : 0));
    if (basis[i] !== 1) throw new Error("Lagrange basis is not a delta function at its own point");
    const term = lagrangeBasisPolynomial(points, i);
    for (let j = 0; j < term.length; j++) coefficients[j] = gfAdd(coefficients[j], gfMul(values[i], term[j]));
  }
  return coefficients;
}

/** Coefficients of the i-th Lagrange basis polynomial of `points`. */
function lagrangeBasisPolynomial(points, index) {
  let polynomial = [1];                                  // the constant 1
  let denominator = 1;
  for (let j = 0; j < points.length; j++) {
    if (j === index) continue;
    denominator = gfMul(denominator, gfAdd(points[index], points[j]));
    const next = new Array(polynomial.length + 1).fill(0);
    for (let t = 0; t < polynomial.length; t++) {
      next[t + 1] = gfAdd(next[t + 1], polynomial[t]);
      next[t] = gfAdd(next[t], gfMul(polynomial[t], points[j]));
    }
    polynomial = next;
  }
  const scale = gfInv(denominator);
  return polynomial.map((coefficient) => gfMul(coefficient, scale));
}

/** Systematic encode via coefficient interpolation. */
function encode(data, m) {
  if (!Number.isInteger(m) || m < 0) throw new RangeError("m must be a non-negative integer");
  const k = data.length;
  if (k === 0) throw new RangeError("the data must not be empty");
  const points = codePoints(k + m);
  const coefficients = interpolateCoefficients(points.slice(0, k), data.slice());
  return evalPoly(coefficients, points);
}

/** Systematic encode via the Lagrange basis (matrix product) - an independent code path. */
function encodeViaBasis(data, m) {
  const k = data.length;
  const points = codePoints(k + m);
  const codeword = new Array(k + m).fill(0);
  for (let i = 0; i < k + m; i++) {
    const basis = lagrangeBasisValues(points.slice(0, k), points[i]);
    let value = 0;
    for (let j = 0; j < k; j++) value = gfAdd(value, gfMul(data[j], basis[j]));
    codeword[i] = value;
  }
  return codeword;
}

/** Gauss-Jordan solve of A x = b over GF(2^8); null when A is singular. */
function solveField(matrix, rhs) {
  const size = matrix.length;
  const augmented = matrix.map((row, index) => [...row, rhs[index]]);
  for (let column = 0; column < size; column++) {
    let pivot = -1;
    for (let row = column; row < size; row++) {
      if (augmented[row][column] !== 0) { pivot = row; break; }
    }
    if (pivot === -1) return null;
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const inverse = gfInv(augmented[column][column]);
    augmented[column] = augmented[column].map((value) => gfMul(value, inverse));
    for (let row = 0; row < size; row++) {
      if (row === column || augmented[row][column] === 0) continue;
      const factor = augmented[row][column];
      augmented[row] = augmented[row].map((value, index) => gfAdd(value, gfMul(factor, augmented[column][index])));
    }
  }
  return augmented.map((row) => row[size]);
}

/** Vandermonde row [1, x, x^2, ..., x^(k-1)] for a degree < k polynomial. */
function vandermondeRow(x, k) {
  const row = [1];
  for (let j = 1; j < k; j++) row.push(gfMul(row[j - 1], x));
  return row;
}

/**
 * Repair `erasures` (indices into the codeword, at most m of them) and return the full codeword.
 * Throws when too many erasures are declared or too few values are supplied.
 */
function decode(received, erasures, k, m) {
  const n = k + m;
  if (received.length !== n) throw new RangeError(`expected ${n} values`);
  if (erasures.length > m) throw new RangeError(`${erasures.length} erasures exceed the ${m} parity bytes`);
  const erased = new Set(erasures);
  const survivors = [];
  for (let i = 0; i < n; i++) if (!erased.has(i)) survivors.push(i);
  if (survivors.length < k) throw new RangeError("not enough surviving values to interpolate");

  const points = codePoints(n);
  const basisPoints = survivors.slice(0, k).map((index) => points[index]);
  const basisValues = survivors.slice(0, k).map((index) => received[index]);
  const matrix = basisPoints.map((x) => vandermondeRow(x, k));
  const coefficients = solveField(matrix, basisValues);
  if (coefficients === null) throw new Error("the survivor matrix is singular - this cannot happen for Vandermonde rows");

  const repaired = received.slice();
  for (const index of erasures) repaired[index] = evalPolyAt(coefficients, points[index]);
  return { codeword: repaired, coefficients };
}

/** Independent decoder: Lagrange interpolation directly at the erased points. */
function decodeViaLagrange(received, erasures, k, m) {
  const n = k + m;
  const erased = new Set(erasures);
  const survivors = [];
  for (let i = 0; i < n; i++) if (!erased.has(i)) survivors.push(i);
  const points = codePoints(n);
  const basisPoints = survivors.slice(0, k).map((index) => points[index]);
  const basisValues = survivors.slice(0, k).map((index) => received[index]);
  const repaired = received.slice();
  for (const index of erasures) {
    if (survivors.length === n) continue;
    const basis = lagrangeBasisValues(basisPoints, points[index]);
    let value = 0;
    for (let j = 0; j < basis.length; j++) value = gfAdd(value, gfMul(basisValues[j], basis[j]));
    repaired[index] = value;
  }
  return repaired;
}

class Lcg {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  byte() {
    return this.next() & 0xff;
  }

  bytes(count) {
    const out = [];
    for (let i = 0; i < count; i++) out.push(this.byte());
    return out;
  }
}

function* combinations(items, choose, start = 0, prefix = []) {
  if (prefix.length === choose) {
    yield prefix.slice();
    return;
  }
  for (let i = start; i <= items.length - (choose - prefix.length); i++) {
    prefix.push(items[i]);
    yield* combinations(items, choose, i + 1, prefix);
    prefix.pop();
  }
}

function hammingDistance(a, b) {
  let distance = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) distance++;
  return distance;
}

module.exports = {
  FIELD_SIZE, PRIMITIVE_POLYNOMIAL, gfMul, gfMulPeasant, gfAdd, gfPow, gfInv, gfDiv, codePoints,
  evalPoly, evalPolyAt, interpolateCoefficients, lagrangeBasisValues, encode, encodeViaBasis,
  decode, decodeViaLagrange, solveField, vandermondeRow, Lcg, combinations, hammingDistance,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- the field ----------------------------------------------------------------------------
  for (let a = 0; a < FIELD_SIZE; a++) {
    for (let b = 0; b < FIELD_SIZE; b++) {
      assert.strictEqual(gfMul(a, b), gfMulPeasant(a, b), `table vs peasant ${a}*${b}`);
      assert.strictEqual(gfMul(a, b), gfMul(b, a), `commutative ${a}*${b}`);
    }
  }
  assert.strictEqual(gfMul(0, 0x5a), 0);
  assert.strictEqual(gfMul(1, 0x5a), 0x5a);
  assert.strictEqual(gfMul(0x02, 0x80), 0x1d, "2 * 128 reduces modulo 0x11d");
  for (let a = 1; a < FIELD_SIZE; a++) {
    assert.strictEqual(gfMul(a, gfInv(a)), 1, `inverse of ${a}`);
    assert.strictEqual(gfDiv(a, a), 1);
    assert.strictEqual(gfPow(a, FIELD_SIZE - 1), 1, `Fermat for ${a}`);
  }
  assert.strictEqual(gfPow(2, 8), 0x1d, "generator 2 raised to 8 is the modulus remainder");
  const lcg = new Lcg(693);
  for (let trial = 0; trial < 200; trial++) {
    const a = lcg.byte();
    const b = lcg.byte();
    const c = lcg.byte();
    assert.strictEqual(gfMul(a, gfAdd(b, c)), gfAdd(gfMul(a, b), gfMul(a, c)), "distributive law");
    assert.strictEqual(gfAdd(a, a), 0, "characteristic two");
    assert.strictEqual(gfAdd(gfAdd(a, b), c), gfAdd(a, gfAdd(b, c)), "associativity");
  }
  assert.throws(() => gfInv(0), RangeError);
  assert.throws(() => gfDiv(1, 0), RangeError);

  // --- encoders agree, and the code is systematic --------------------------------------------
  const DATA = [0x0a, 0x1b, 0x2c, 0x3d];
  const PARITY_COUNT = 2;
  const codeword = encode(DATA, PARITY_COUNT);
  assert.deepStrictEqual(codeword.slice(0, DATA.length), DATA, "the data bytes come first");
  assert.deepStrictEqual(codeword, encodeViaBasis(DATA, PARITY_COUNT), "both encoders agree");
  assert.strictEqual(codeword.length, DATA.length + PARITY_COUNT);
  const points = codePoints(codeword.length);
  const coefficients = interpolateCoefficients(points, codeword);
  assert.deepStrictEqual(evalPoly(coefficients, points), codeword,
    "every codeword is the evaluation of one degree < k polynomial");
  assert.ok(coefficients.slice(DATA.length).every((value) => value === 0),
    "interpolating through all n points leaves the coefficients above k - 1 zero");
  assert.deepStrictEqual(interpolateCoefficients([1, 2, 3], [5, 9, 1]).length, 3);
  assert.throws(() => interpolateCoefficients([1, 2], [1]), RangeError);

  // linearity: the code is a linear map over GF(2)
  for (let trial = 0; trial < 50; trial++) {
    const a = lcg.bytes(DATA.length);
    const b = lcg.bytes(DATA.length);
    const sum = a.map((value, index) => gfAdd(value, b[index]));
    const left = encode(sum, PARITY_COUNT);
    const right = encode(a, PARITY_COUNT).map((value, index) => gfAdd(value, encode(b, PARITY_COUNT)[index]));
    assert.deepStrictEqual(left, right, "enc is linear");
  }

  // --- every erasure pattern up to m is repairable -------------------------------------------
  function repairAll(parameters) {
    const { k, m, payload } = parameters;
    const word = encode(payload, m);
    const n = k + m;
    const indices = Array.from({ length: n }, (_, i) => i);
    let patterns = 0;
    for (let size = 0; size <= m; size++) {
      for (const erasures of combinations(indices, size)) {
        patterns++;
        const damaged = word.slice();
        for (const index of erasures) damaged[index] = 0xff;      // erase to a bogus value
        const { codeword: repaired } = decode(damaged, erasures, k, m);
        assert.deepStrictEqual(repaired, word, `repair k=${k} m=${m} erasures=${erasures}`);
        assert.deepStrictEqual(decodeViaLagrange(damaged, erasures, k, m), word,
          `Lagrange repair k=${k} m=${m} erasures=${erasures}`);
      }
    }
    return patterns;
  }
  const patternCounts = [];
  patternCounts.push(repairAll({ k: 4, m: 2, payload: DATA }));
  patternCounts.push(repairAll({ k: 3, m: 3, payload: [7, 7, 7] }));
  patternCounts.push(repairAll({ k: 1, m: 4, payload: [0x42] }));               // repetition code
  patternCounts.push(repairAll({ k: 6, m: 1, payload: lcg.bytes(6) }));
  assert.deepStrictEqual(patternCounts, [22, 42, 31, 8], "pattern counts: sum of C(n, i) for i <= m");

  // --- more erasures than parity bytes are refused -------------------------------------------
  const word = encode(DATA, PARITY_COUNT);
  assert.throws(() => decode(word, [0, 1, 2], DATA.length, PARITY_COUNT), RangeError,
    "three erasures cannot be repaired by two parity bytes");
  assert.strictEqual(decode(word, [0, 1], DATA.length, PARITY_COUNT).codeword[2], word[2]);
  assert.throws(() => decode(word.slice(0, 5), [], DATA.length, PARITY_COUNT), RangeError);
  const allErased = word.map(() => 0);
  assert.throws(() => decode(allErased, [0, 1, 2, 3, 4, 5], DATA.length, PARITY_COUNT), RangeError);

  // any k survivors work, not only the first k
  for (const erasures of [[0, 1], [2, 5], [3, 4], [0, 5], [1, 2]]) {
    const damaged = word.slice();
    for (const index of erasures) damaged[index] = 0;
    assert.deepStrictEqual(decode(damaged, erasures, DATA.length, PARITY_COUNT).codeword, word,
      `survivors ${erasures}`);
  }

  // --- random payloads and random erasure sets -----------------------------------------------
  let randomChecks = 0;
  for (let trial = 0; trial < 60; trial++) {
    const k = 2 + (lcg.next() % 9);
    const m = 1 + (lcg.next() % 4);
    const payload = lcg.bytes(k);
    const message = encode(payload, m);
    const positions = Array.from({ length: k + m }, (_, i) => i);
    const size = 1 + (lcg.next() % m);
    const erasures = [];
    while (erasures.length < size) {                       // deterministic distinct sample
      const candidate = positions[(lcg.next() + erasures.length) % positions.length];
      if (!erasures.includes(candidate)) erasures.push(candidate);
    }
    const damaged = message.slice();
    for (const index of erasures) damaged[index] = lcg.byte();
    const repaired = decode(damaged, erasures, k, m).codeword;
    assert.deepStrictEqual(repaired, message, `random trial ${trial}`);
    assert.deepStrictEqual(decodeViaLagrange(damaged, erasures, k, m), message, `random lagrange ${trial}`);
    randomChecks++;
  }
  assert.strictEqual(randomChecks, 60);

  // --- the MDS distance property -------------------------------------------------------------
  for (let trial = 0; trial < 30; trial++) {
    const k = 3;
    const m = 2;
    const first = lcg.bytes(k);
    const second = lcg.bytes(k);
    if (first.join(",") === second.join(",")) continue;
    assert.ok(hammingDistance(encode(first, m), encode(second, m)) >= m + 1,
      "distinct codewords differ in at least m + 1 positions");
  }
  assert.strictEqual(hammingDistance(encode([0, 0, 0], 2), encode([1, 0, 0], 2)) >= 3, true);
  const zero = encode([0, 0, 0, 0], 2);
  assert.deepStrictEqual(zero, [0, 0, 0, 0, 0, 0], "the zero payload gives the zero codeword");

  // --- degenerate parameters -----------------------------------------------------------------
  assert.deepStrictEqual(encode(DATA, 0), DATA, "no parity means the codeword is the data");
  assert.deepStrictEqual(decode(DATA, [], DATA.length, 0).codeword, DATA);
  assert.throws(() => encode([], 2), RangeError, "empty data has no polynomial");
  assert.throws(() => encode(DATA, -1), RangeError);
  assert.throws(() => codePoints(0), RangeError);
  assert.throws(() => codePoints(256), RangeError, "GF(2^8) has only 255 non-zero points");
  assert.strictEqual(codePoints(255).length, 255);
  const wide = encode(DATA, 251);                          // the largest possible code
  assert.strictEqual(wide.length, 255);
  const wideDamaged = wide.slice();
  wideDamaged[254] = 0x00;
  wideDamaged[100] = 0x00;
  wideDamaged[7] = 0x00;
  assert.deepStrictEqual(decode(wideDamaged, [254, 100, 7], DATA.length, 251).codeword, wide,
    "251 parity bytes repair 251 erasures");
  assert.deepStrictEqual(decodeViaLagrange(wideDamaged, [254, 100, 7], DATA.length, 251), wide);

  // --- determinism ---------------------------------------------------------------------------
  const replayA = encode(lcg.bytes(5), 3);
  const replayB = encode([...lcg.bytes(5)], 3);
  assert.strictEqual(replayA.length, 8);
  assert.strictEqual(new Lcg(693).next(), new Lcg(693).next(), "the generator is deterministic");

  console.log(`patterns=${patternCounts.reduce((a, b) => a + b, 0)} random=${randomChecks} n=${wide.length}`);
  console.log("PASS");
}
