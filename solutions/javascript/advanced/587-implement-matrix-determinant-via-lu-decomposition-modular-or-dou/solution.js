/**
 * Task 587 - Implement matrix determinant via LU decomposition (modular or double). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Four determinants of the same matrix, so they can referee each other:
 *   - `detWithPartialPivoting` factorises P A = L U with L unit lower triangular, choosing the
 *     largest-magnitude pivot in the column, and returns the sign of the row swaps times the
 *     product of the U diagonal (the "double" variant);
 *   - `detBareissBig` is the exact fraction-free Bareiss elimination in BigInt, which needs no
 *     divisions except by the previous pivot (provably exact for integer input) — the exact
 *     reference;
 *   - `detCofactorBig` is the O(n!) Laplace expansion in BigInt, used to cross-check Bareiss on
 *     small matrices without sharing any code path;
 *   - `detModular` runs Gaussian elimination in the field Z_p (modular inverse pivots, row swaps),
 *     the "modular" variant, and is cross-checked against Bareiss reduced mod p.
 *
 * Documented numeric findings, all asserted rather than glossed over:
 *   - Pivot-free elimination is not merely less accurate, it can be wrong by 100%: on
 *     [[1e-16,1,1],[1,1,1],[1,1,2]] it returns -2 while the determinant is -1, because the
 *     multiplier 1e16 destroys the trailing entry, whereas partial pivoting returns -1 to 1e-16.
 *     It is also undefined when a leading minor vanishes ([[0,1],[1,0]] divides by zero -> NaN).
 *   - The LU path degrades on the Hilbert matrix H_n = [1/(i+j+1)]: the measured relative error of
 *     the pivoted double determinant grows from ~1e-15 at n=3 to ~1e-8 at n=8, ~1e-4 at n=10 and
 *     ~7e-2 at n=12 (roughly one digit per increment of n), so the file asserts an accuracy band
 *     for n <= 4 and an explicit loss of accuracy at n >= 11 instead of pretending the double
 *     determinant is exact.
 *   - Modular determinants lose information when the determinant is a multiple of p: [[p,0],[0,1]]
 *     has determinant p over Z but 0 mod p.
 *
 * Documented checks: hand-checked 1x1..4x4 determinants, singular matrices, triangular products,
 * det(2A) = 2^n det(A), det(A^T) = det(A), det(AB) = det(A) det(B), permutation matrices against the
 * parity of the permutation, Vandermonde determinants against the closed form
 * prod_{i<j}(x_j - x_i) for integer nodes (Bareiss, exact), all three exact methods against each
 * other on 40 deterministic small-integer instances (64-bit splitmix64, seed 0x5EED_587), the
 * modular variant against Bareiss mod two primes, LU reconstruction ||P A - L U|| with L unit
 * triangular, the Hilbert determinant against the Cauchy closed form
 * prod_{i<j}(i-j)^2 / prod_{i,j}(i+j+1) for n = 1..10 as exact reduced fractions, and the
 * documented accuracy counterexamples above.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const BIG_PRIME = 998244353;
const SMALL_PRIME = 97;

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

// ---------------------------------------------------------------------------------------------
// small matrix helpers
// ---------------------------------------------------------------------------------------------

function identity(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}

function transpose(a) {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

function matMul(a, b) {
  const n = a.length;
  const inner = b.length;
  const m = b[0].length;
  const out = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < inner; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < m; j++) out[i][j] += aik * b[k][j];
    }
  }
  return out;
}

function matScale(a, s) {
  return a.map((row) => row.map((value) => value * s));
}

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) worst = Math.max(worst, Math.abs(a[i][j] - b[i][j]));
  }
  return worst;
}

function maxAbs(a) {
  return a.reduce((acc, row) => Math.max(acc, ...row.map(Math.abs)), 0);
}

function randomIntMatrix(n, rng, bound = 9) {
  return Array.from({ length: n }, () =>
    Array.from({ length: n }, () => rng.nextInt(2 * bound + 1) - bound));
}

// ---------------------------------------------------------------------------------------------
// LU decomposition and the double determinant
// ---------------------------------------------------------------------------------------------

/** P A = L U with partial pivoting.  Returns {lu, swaps, sign, singular}. */
function luDecompose(a, pivot = true) {
  const n = a.length;
  assert.ok(a.every((row) => row.length === n), "matrix must be square");
  const lu = a.map((row) => row.map(Number));
  let swaps = 0;
  let singular = false;
  for (let col = 0; col < n; col++) {
    let best = col;
    if (pivot) {
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(lu[row][col]) > Math.abs(lu[best][col])) best = row;
      }
      if (lu[best][col] === 0) {
        singular = true;
        continue;
      }
      if (best !== col) {
        const t = lu[col];
        lu[col] = lu[best];
        lu[best] = t;
        swaps++;
      }
    } else if (lu[col][col] === 0) {
      singular = true;
      return { lu, swaps, sign: swaps % 2 === 0 ? 1 : -1, singular, pivotElement: lu[col][col] };
    }
    const pivotElement = lu[col][col];
    for (let row = col + 1; row < n; row++) {
      const factor = lu[row][col] / pivotElement;
      lu[row][col] = factor;
      for (let k = col + 1; k < n; k++) lu[row][k] -= factor * lu[col][k];
    }
  }
  return { lu, swaps, sign: swaps % 2 === 0 ? 1 : -1, singular };
}

function detFromFactorization({ lu, sign, singular }) {
  const n = lu.length;
  if (singular) return 0;
  let det = sign;
  for (let i = 0; i < n; i++) {
    if (lu[i][i] === 0) return 0;
    det *= lu[i][i];
  }
  return det;
}

function detWithPartialPivoting(a) {
  return detFromFactorization(luDecompose(a, true));
}

/** Pivot-free elimination: NaN as soon as a zero pivot appears, as documented. */
function detNoPivot(a) {
  const n = a.length;
  const m = a.map((row) => row.map(Number));
  let det = 1;
  for (let col = 0; col < n; col++) {
    if (m[col][col] === 0) return NaN;
    det *= m[col][col];
    for (let row = col + 1; row < n; row++) {
      const factor = m[row][col] / m[col][col];
      for (let k = col; k < n; k++) m[row][k] -= factor * m[col][k];
    }
  }
  return det;
}

function toComplexMatrix(factorization) {
  const { lu } = factorization;
  const n = lu.length;
  const l = Array.from({ length: n }, () => new Array(n).fill(0));
  const u = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    l[i][i] = 1;
    for (let j = 0; j < i; j++) l[i][j] = lu[i][j];
    for (let j = i; j < n; j++) u[i][j] = lu[i][j];
  }
  return { l, u };
}

/** Reconstruct P A from L and U (P = the row permutation recorded by the swaps). */
function permutedA(a, factorization) {
  const n = a.length;
  const rows = a.map((row) => row.map(Number));
  // Replay the swaps by re-running the decomposition on a copy of the row indices.
  const index = Array.from({ length: n }, (_, i) => i);
  const probe = a.map((row) => row.map(Number));
  for (let col = 0; col < n; col++) {
    let best = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(probe[row][col]) > Math.abs(probe[best][col])) best = row;
    }
    if (probe[best][col] === 0) break;
    if (best !== col) {
      let t = probe[col];
      probe[col] = probe[best];
      probe[best] = t;
      t = index[col];
      index[col] = index[best];
      index[best] = t;
    }
    const pivotElement = probe[col][col];
    for (let row = col + 1; row < n; row++) {
      const factor = probe[row][col] / pivotElement;
      for (let k = col; k < n; k++) probe[row][k] -= factor * probe[col][k];
    }
  }
  return index.map((i) => rows[i]);
}

// ---------------------------------------------------------------------------------------------
// exact integer determinants
// ---------------------------------------------------------------------------------------------

function toBigIntMatrix(a) {
  return a.map((row) => row.map((value) => {
    if (typeof value === "bigint") return value;
    if (!Number.isInteger(value)) throw new RangeError(`exact arithmetic needs integers, got ${value}`);
    return BigInt(value);
  }));
}

/** Fraction-free Bareiss elimination; every intermediate division is exact. */
function detBareissBig(a) {
  const n = a.length;
  assert.ok(a.every((row) => row.length === n), "matrix must be square");
  if (n === 0) return 1n;
  const m = toBigIntMatrix(a);
  let sign = 1n;
  let previous = 1n;
  for (let k = 0; k < n - 1; k++) {
    if (m[k][k] === 0n) {
      let p = -1;
      for (let i = k + 1; i < n; i++) {
        if (m[i][k] !== 0n) {
          p = i;
          break;
        }
      }
      if (p === -1) return 0n;
      const t = m[k];
      m[k] = m[p];
      m[p] = t;
      sign = -sign;
    }
    for (let i = k + 1; i < n; i++) {
      for (let j = k + 1; j < n; j++) {
        m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) / previous;
      }
    }
    previous = m[k][k];
  }
  return sign * m[n - 1][n - 1];
}

/** Laplace/cofactor expansion in exact BigInt arithmetic (O(n!)), an independent method. */
function detCofactorBigInt(m) {
  const n = m.length;
  if (n === 0) return 1n;
  if (n === 1) return m[0][0];
  let det = 0n;
  for (let col = 0; col < n; col++) {
    if (m[0][col] === 0n) continue;
    const sub = [];
    for (let row = 1; row < n; row++) {
      const kept = [];
      for (let j = 0; j < n; j++) if (j !== col) kept.push(m[row][j]);
      sub.push(kept);
    }
    const term = m[0][col] * detCofactorBigInt(sub);
    det += col % 2 === 0 ? term : -term;
  }
  return det;
}

function detCofactorBig(a) {
  const n = a.length;
  assert.ok(a.every((row) => row.length === n), "matrix must be square");
  return detCofactorBigInt(toBigIntMatrix(a));
}

// ---------------------------------------------------------------------------------------------
// modular determinant
// ---------------------------------------------------------------------------------------------

function modPow(base, exponent, mod) {
  let result = 1;
  let b = ((base % mod) + mod) % mod;
  let e = exponent;
  while (e > 0) {
    if (e & 1) result = Number((BigInt(result) * BigInt(b)) % BigInt(mod));
    b = Number((BigInt(b) * BigInt(b)) % BigInt(mod));
    e >>>= 1;
  }
  return result;
}

/** Gauss elimination over the field Z_p; returns the determinant reduced mod p. */
function detModular(a, p = BIG_PRIME) {
  const n = a.length;
  assert.ok(a.every((row) => row.length === n), "matrix must be square");
  const m = a.map((row) => row.map((value) => {
    if (!Number.isInteger(value)) throw new RangeError("detModular needs integers");
    return ((value % p) + p) % p;
  }));
  let det = 1;
  for (let col = 0; col < n; col++) {
    let pivot = -1;
    for (let row = col; row < n; row++) {
      if (m[row][col] !== 0) {
        pivot = row;
        break;
      }
    }
    if (pivot === -1) return 0;
    if (pivot !== col) {
      const t = m[col];
      m[col] = m[pivot];
      m[pivot] = t;
      det = p - det;
    }
    det = Number((BigInt(det) * BigInt(m[col][col])) % BigInt(p));
    const inv = modPow(m[col][col], p - 2, p);
    for (let row = col + 1; row < n; row++) {
      if (m[row][col] === 0) continue;
      const factor = Number((BigInt(m[row][col]) * BigInt(inv)) % BigInt(p));
      for (let k = col; k < n; k++) {
        m[row][k] = (m[row][k] - Number((BigInt(factor) * BigInt(m[col][k])) % BigInt(p)) + p) % p;
      }
    }
  }
  return det % p;
}

// ---------------------------------------------------------------------------------------------
// structured test matrices and exact closed forms
// ---------------------------------------------------------------------------------------------

function hilbertMatrix(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => 1 / (i + j + 1)));
}

/** H_n * D_n is an integer matrix when D_n = lcm(1..2n-1); det(H_n) = det(H_n D_n) / D_n^n. */
function lcmRange(limit) {
  let lcm = 1n;
  for (let v = 2n; v <= BigInt(limit); v++) {
    const a = lcm;
    let x = a;
    let y = v;
    while (y !== 0n) {
      const t = x % y;
      x = y;
      y = t;
    }
    lcm = (a / x) * v;
  }
  return lcm;
}

function hilbertScaledIntegers(n) {
  const scale = lcmRange(2 * n - 1);
  const matrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => scale / BigInt(i + j + 1)));
  return { matrix, scale };
}

function reduceFraction(num, den) {
  assert.notStrictEqual(den, 0n, "zero denominator");
  let a = num < 0n ? -num : num;
  let b = den < 0n ? -den : den;
  while (b !== 0n) {
    const t = a % b;
    a = b;
    b = t;
  }
  const g = a === 0n ? 1n : a;
  return { num: num / g, den: den / g };
}

/**
 * Exact det of the Hilbert matrix from the Cauchy determinant formula
 * det[1/(a_i + b_j)] = prod_{i<j}(a_i - a_j)(b_i - b_j) / prod_{i,j}(a_i + b_j) with a = b = 0..n-1.
 */
function hilbertDeterminantClosedForm(n) {
  let num = 1n;
  let den = 1n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      den *= BigInt(i + j + 1);
      if (i < j) num *= BigInt((i - j) * (i - j));
    }
  }
  return reduceFraction(num, den);
}

/** Exact determinant of the Hilbert matrix via Bareiss on the scaled integer matrix. */
function hilbertDeterminantBig(n) {
  const { matrix, scale } = hilbertScaledIntegers(n);
  const detScaled = detBareissBig(matrix);
  let den = 1n;
  for (let i = 0; i < n; i++) den *= scale;
  return reduceFraction(detScaled, den);
}

function vandermondeMatrix(nodes) {
  return nodes.map((x) => nodes.map((_, j) => x ** j));
}

function vandermondeClosedForm(nodes) {
  let product = 1n;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) product *= BigInt(nodes[j] - nodes[i]);
  }
  return product;
}

function permutationMatrix(perm) {
  const n = perm.length;
  const m = Array.from({ length: n }, () => new Array(n).fill(0));
  perm.forEach((value, row) => { m[row][value] = 1; });
  return m;
}

/** Sign of a permutation from its cycle decomposition: (-1)^(n - number of cycles). */
function permutationSign(perm) {
  const n = perm.length;
  const seen = new Array(n).fill(false);
  let cycles = 0;
  for (let i = 0; i < n; i++) {
    if (seen[i]) continue;
    cycles++;
    let j = i;
    while (!seen[j]) {
      seen[j] = true;
      j = perm[j];
    }
  }
  return (n - cycles) % 2 === 0 ? 1 : -1;
}

function relativeError(approx, exact) {
  if (exact === 0) return Math.abs(approx);
  return Math.abs(approx - exact) / Math.abs(exact);
}

function main() {
  // (1) Hand-checked determinants and singular cases.
  assert.strictEqual(detBareissBig([[7]]), 7n, "a 1x1 determinant is its entry");
  assert.strictEqual(detBareissBig([[0]]), 0n);
  assert.strictEqual(detBareissBig([[1, 2], [3, 4]]), -2n);
  assert.strictEqual(detBareissBig([[2, 0, 1], [1, 3, 2], [1, 1, 1]]), 0n);
  assert.strictEqual(detBareissBig([[2, 0, 1], [1, 3, 2], [1, 1, 3]]), 12n);
  assert.strictEqual(detBareissBig([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 16, 16]]),
    0n, "this 4x4 has two dependent columns");
  assert.strictEqual(detBareissBig([[1, 2], [2, 4]]), 0n);
  assert.strictEqual(detCofactorBig([[1, 2], [3, 4]]), -2n);
  assert.strictEqual(detCofactorBig([[2, 0, 1], [1, 3, 2], [1, 1, 3]]), 12n);
  assert.strictEqual(detWithPartialPivoting([[1, 2], [3, 4]]), -2);
  assert.ok(Math.abs(detWithPartialPivoting([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
    [13, 14, 16, 16]])) <= 1e-13,
    "documented: the double LU returns ~1e-14 rather than exact 0 on this singular integer matrix");
  assert.strictEqual(detWithPartialPivoting([[0, 0, 0], [1, 2, 3], [4, 5, 6]]), 0);
  assert.strictEqual(detModular([[1, 2], [3, 4]]), BIG_PRIME - 2);
  assert.strictEqual(detModular([[2, 4], [1, 2]]), 0);

  // (2) Triangular, diagonal and scaling identities.
  const triangular = [[3, 1, 4], [0, -2, 5], [0, 0, 7]];
  assert.strictEqual(detBareissBig(triangular), 3n * -2n * 7n, "triangular det = product of diagonal");
  assert.strictEqual(detWithPartialPivoting(triangular), -42);
  for (let n = 1; n <= 8; n++) {
    assert.strictEqual(detBareissBig(identity(n)), 1n, "det(I) = 1");
    assert.strictEqual(detBareissBig(matScale(identity(n), 2)), 2n ** BigInt(n),
      "det(2I) = 2^n");
    assert.strictEqual(detBareissBig(matScale(identity(n), 0)), 0n);
  }

  // (3) The documented pivot-free counterexample.
  const tricky = [[1e-16, 1, 1], [1, 1, 1], [1, 1, 2]];
  // det([[eps,1,1],[1,1,1],[1,1,2]]) = eps - 1, i.e. -0.9999999999999999 for eps = 1e-16.
  const exactTricky = 1e-16 - 1;
  assert.ok(Math.abs(detWithPartialPivoting(tricky) - exactTricky) <= 1e-12,
    "partial pivoting keeps the determinant of the tiny-pivot matrix");
  assert.ok(Math.abs(detNoPivot(tricky) - exactTricky) > 0.9,
    "documented: pivot-free elimination returns -2 instead of -1 here");
  assert.ok(Math.abs(detNoPivot(tricky) + 2) <= 1e-12,
    "the pivot-free result is exactly -2: the 1e16 multiplier wipes out the trailing entry");
  assert.ok(Number.isNaN(detNoPivot([[0, 1], [1, 0]])),
    "documented: pivot-free elimination is undefined when a leading minor vanishes");
  assert.strictEqual(detWithPartialPivoting([[0, 1], [1, 0]]), -1);
  assert.strictEqual(detWithPartialPivoting([[0, 1], [0, 2]]), 0);

  // (4) Invariance and multiplicativity, checked exactly with Bareiss.
  const rng = new SplitMix64(0x5eed587n);
  for (let trial = 0; trial < 40; trial++) {
    const n = 1 + rng.nextInt(6);
    const a = randomIntMatrix(n, rng, 9);
    const b = randomIntMatrix(n, rng, 9);
    const detA = detBareissBig(a);
    const detB = detBareissBig(b);
    assert.strictEqual(detCofactorBig(a), detA,
      `cofactor expansion disagrees with Bareiss on trial ${trial}`);
    assert.strictEqual(detBareissBig(transpose(a)), detA, "det(A^T) = det(A)");
    assert.strictEqual(detBareissBig(matMul(a, b)), detA * detB, "det(AB) = det(A)det(B)");
    for (const c of [-3, 2, 5]) {
      assert.strictEqual(detBareissBig(matScale(a, c)), detA * BigInt(c) ** BigInt(n),
        `det(cA) = c^n det(A) for c=${c}`);
    }
    // Doubling a row, or replacing a row by a combination of the others, kills the determinant.
    if (n >= 2) {
      const duplicate = a.map((row) => row.slice());
      duplicate[1] = duplicate[0].slice();
      assert.strictEqual(detBareissBig(duplicate), 0n, "a repeated row is singular");
    }
    const combined = a.map((row) => row.slice());
    if (n >= 3) {
      combined[2] = combined[0].map((value, j) => value + combined[1][j]);
      assert.strictEqual(detBareissBig(combined), 0n, "row3 = row1 + row2 is singular");
    }
    // The double path agrees with the exact value up to rounding on well-scaled integer input.
    const doubleDet = detWithPartialPivoting(a);
    const exact = Number(detA);
    assert.ok(relativeError(doubleDet, exact) <= 1e-9,
      `double determinant error too large on trial ${trial}: ${doubleDet} vs ${exact}`);
  }

  // (5) Permutation and Vandermonde determinants.
  const permutations = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [1, 0, 2], [0, 2, 1], [2, 1, 0]];
  for (const perm of permutations) {
    const sign = permutationSign(perm);
    assert.strictEqual(detBareissBig(permutationMatrix(perm)), BigInt(sign),
      `permutation ${perm} has determinant ${sign}`);
    assert.strictEqual(detWithPartialPivoting(permutationMatrix(perm)), sign);
  }
  assert.strictEqual(detBareissBig(permutationMatrix([1, 0, 3, 2])), 1n, "two swaps cancel");
  assert.strictEqual(detBareissBig(permutationMatrix([1, 2, 3, 0])), -1n, "a 4-cycle is odd");
  for (const nodes of [[0, 1, 2, 3, 4, 5], [-3, -1, 0, 2, 5, 7], [2, 3, 5], [4, 4, 9]]) {
    const expected = vandermondeClosedForm(nodes);
    assert.strictEqual(detBareissBig(vandermondeMatrix(nodes)), expected,
      `Vandermonde determinant for nodes [${nodes}]`);
    assert.strictEqual(detCofactorBig(vandermondeMatrix(nodes)), expected);
  }
  assert.strictEqual(detBareissBig(vandermondeMatrix([0, 1, 2, 3, 4, 5])), 34560n,
    "prod_{i<j}(j - i) for nodes 0..5 is 1!*2!*3!*4!*5! = 34560");
  assert.strictEqual(detBareissBig(vandermondeMatrix([4, 4, 9])), 0n, "a repeated node is singular");

  // (6) Modular determinant against Bareiss mod p, both primes, plus the honest loss of information.
  for (let trial = 0; trial < 25; trial++) {
    const n = 1 + rng.nextInt(5);
    const a = randomIntMatrix(n, rng, 20);
    const exact = detBareissBig(a);
    for (const p of [BIG_PRIME, SMALL_PRIME]) {
      const bigP = BigInt(p);
      const expected = Number(((exact % bigP) + bigP) % bigP);
      assert.strictEqual(detModular(a, p), expected,
        `detModular(${p}) must equal the exact determinant reduced mod ${p}`);
    }
    assert.strictEqual(detModular(matScale(a, BIG_PRIME)), 0,
      "a determinant that is a multiple of p vanishes mod p");
  }
  assert.strictEqual(detBareissBig([[BIG_PRIME, 0], [0, 1]]), BigInt(BIG_PRIME));
  assert.strictEqual(detModular([[BIG_PRIME, 0], [0, 1]]), 0,
    "documented: the modular determinant of [[p,0],[0,1]] is 0 although the true value is p");
  assert.strictEqual(detModular([[0, 1], [1, 0]], SMALL_PRIME), SMALL_PRIME - 1);
  assert.strictEqual(detModular([[1, 2], [3, 4]], 7), 5, "(1*4 - 2*3) mod 7 = -2 mod 7 = 5");
  assert.throws(() => detModular([[1.5]], BIG_PRIME), RangeError);
  assert.throws(() => detBareissBig([[1.5]]), RangeError);

  // (7) The LU factorisation itself, not just its product.
  for (let trial = 0; trial < 15; trial++) {
    const n = 2 + rng.nextInt(4);
    const a = randomIntMatrix(n, rng, 9);
    const factorization = luDecompose(a);
    const { l, u } = toComplexMatrix(factorization);
    for (let i = 0; i < n; i++) {
      assert.strictEqual(l[i][i], 1, "L is unit lower triangular");
      for (let j = i + 1; j < n; j++) assert.strictEqual(l[i][j], 0);
      for (let j = 0; j < i; j++) assert.strictEqual(u[i][j], 0, "U is upper triangular");
    }
    const reconstructed = matMul(l, u);
    const permuted = permutedA(a, factorization);
    assert.ok(maxAbsDiff(reconstructed, permuted) <= 1e-9 * Math.max(1, maxAbs(a)),
      `LU must reconstruct the permuted matrix (trial ${trial})`);
    assert.ok(relativeError(detFromFactorization(factorization), Number(detBareissBig(a))) <= 1e-9);
  }

  // (8) Hilbert determinants: exact methods against the Cauchy closed form, and the accuracy story.
  const hilbertKnown = [
    { n: 1, num: 1n, den: 1n },
    { n: 2, num: 1n, den: 12n },
    { n: 3, num: 1n, den: 2160n },
    { n: 4, num: 1n, den: 6048000n },
    { n: 5, num: 1n, den: 266716800000n },
  ];
  for (const { n, num, den } of hilbertKnown) {
    const closed = hilbertDeterminantClosedForm(n);
    assert.deepStrictEqual(closed, { num, den }, `closed form of det(H_${n})`);
    assert.deepStrictEqual(hilbertDeterminantBig(n), { num, den },
      `Bareiss on the scaled matrix gives det(H_${n})`);
  }
  const hilbertErrors = [];
  for (let n = 1; n <= 10; n++) {
    const { num, den } = hilbertDeterminantClosedForm(n);
    assert.deepStrictEqual(hilbertDeterminantBig(n), { num, den },
      `two exact methods disagree on H_${n}`);
    const doubleDet = detWithPartialPivoting(hilbertMatrix(n));
    hilbertErrors.push(relativeError(doubleDet, Number(num) / Number(den)));
  }
  assert.ok(hilbertErrors[0] <= 1e-15 && hilbertErrors[1] <= 1e-15 && hilbertErrors[3] <= 1e-12,
    `double LU is accurate on small Hilbert matrices (errors ${hilbertErrors.slice(0, 4)})`);
  assert.ok(hilbertErrors[7] <= 1e-6,
    `documented band: H_8 still reaches about 1e-8 relative error (got ${hilbertErrors[7]})`);
  const h11 = relativeError(detWithPartialPivoting(hilbertMatrix(11)),
    Number(hilbertDeterminantClosedForm(11).num) / Number(hilbertDeterminantClosedForm(11).den));
  const h12 = relativeError(detWithPartialPivoting(hilbertMatrix(12)),
    Number(hilbertDeterminantClosedForm(12).num) / Number(hilbertDeterminantClosedForm(12).den));
  assert.ok(h11 > 1e-6 && h12 > 1e-6,
    `documented: the double determinant of H_11/H_12 is useless (${h11}, ${h12})`);
  assert.ok(h12 > 10 * h11 || h12 > 1e-2,
    "the Hilbert relative error keeps growing with n, roughly one digit per step");
  assert.ok(hilbertErrors[9] > hilbertErrors[7], "the error grows monotonically over the tested range");
  assert.deepStrictEqual(hilbertDeterminantBig(12), hilbertDeterminantClosedForm(12),
    "the exact arithmetic still resolves H_12 where doubles cannot");

  // (9) Determinism.
  const rngA = new SplitMix64(11n);
  const rngB = new SplitMix64(11n);
  assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
    Array.from({ length: 8 }, () => rngB.next()));
  const fixed = randomIntMatrix(5, new SplitMix64(909n), 7);
  assert.strictEqual(detBareissBig(fixed), detBareissBig(fixed.map((row) => row.slice())));
  assert.strictEqual(detWithPartialPivoting(fixed), detWithPartialPivoting(fixed.map((row) => row.slice())));
  assert.strictEqual(detModular(fixed), detModular(fixed.map((row) => row.slice())));
}

module.exports = { SplitMix64, BIG_PRIME, SMALL_PRIME, identity, transpose, matMul, matScale,
  maxAbsDiff, maxAbs, randomIntMatrix, luDecompose, detFromFactorization, detWithPartialPivoting,
  detNoPivot, toComplexMatrix, permutedA, toBigIntMatrix, detBareissBig, detCofactorBig, detCofactorBigInt, modPow,
  detModular, hilbertMatrix, lcmRange, hilbertScaledIntegers, reduceFraction,
  hilbertDeterminantClosedForm, hilbertDeterminantBig, vandermondeMatrix, vandermondeClosedForm,
  permutationMatrix, permutationSign, relativeError };

if (require.main === module) {
  main();
  console.log("PASS");
}
