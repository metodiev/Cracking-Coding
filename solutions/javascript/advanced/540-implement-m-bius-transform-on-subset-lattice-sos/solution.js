/**
 * Task 540 - Implement Möbius transform on subset lattice (SOS). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Sum-over-subsets (zeta) and its exact inverse (Möbius) on the Boolean lattice with k bits:
 * zeta(f)[S] = sum_{T subset of S} f[T] is built by the standard doubling sweep - for each bit b,
 * every mask carrying b absorbs its partner without b - and mobius(f) undoes it by subtracting
 * instead of adding, so the two are exact inverses in either order.  The superset (downward)
 * variants, where F[S] = sum_{T superset of S} f[T], use the mirrored sweep.  On top of those
 * primitives the file implements OR-convolution c[S] = sum_{A or B == S} f[A] * g[B] as
 * mobius(zeta(f) * zeta(g)), i.e. pointwise multiplication in the zeta domain.
 *
 * Documented checks: both transforms agree mask-for-mask with brute-force subset enumeration
 * (2^k * k work, used as the ground truth for k <= 5); zeta and mobius are exact inverses in both
 * orders; the closed forms zeta(all-ones)[S] = 2^popcount(S), superset-zeta(all-ones)[S] =
 * 2^(k - popcount(S)), and zeta(delta_T)[S] = [T subset of S] are asserted; the grand total
 * sum_S zeta(f)[S] = sum_T f[T] * 2^(k - popcount(T)) is checked, as is the lattice monotonicity
 * of the zeta transform for non-negative f; OR-convolution is compared with the O(4^k) naive
 * definition, plus the delta identity, commutativity and associativity; and non power-of-two
 * lengths are rejected.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function isPowerOfTwo(n) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

function popcount(x) {
  let count = 0;
  let v = x;
  while (v) {
    v &= v - 1;
    count += 1;
  }
  return count;
}

function bitsOf(length) {
  let k = 0;
  while (1 << k < length) k += 1;
  return k;
}

/** -0 and 0 compare unequal under deepStrictEqual, so canonicalise zero results. */
function normalizeZeros(values) {
  return values.map((value) => (value === 0 ? 0 : value));
}

function checkLength(values, what) {
  if (!isPowerOfTwo(values.length)) {
    throw new Error(`${what} needs a power-of-two length`);
  }
}

/** zeta(f)[S] = sum of f over subsets of S (SOS DP). */
function subsetZeta(values) {
  checkLength(values, "subset zeta transform");
  const a = values.slice();
  const k = bitsOf(a.length);
  for (let bit = 0; bit < k; bit++) {
    const step = 1 << bit;
    for (let mask = 0; mask < a.length; mask++) {
      if (mask & step) a[mask] += a[mask ^ step];
    }
  }
  return normalizeZeros(a);
}

/** The exact inverse of subsetZeta. */
function subsetMobius(values) {
  checkLength(values, "subset Möbius transform");
  const a = values.slice();
  const k = bitsOf(a.length);
  for (let bit = 0; bit < k; bit++) {
    const step = 1 << bit;
    for (let mask = 0; mask < a.length; mask++) {
      if (mask & step) a[mask] -= a[mask ^ step];
    }
  }
  return normalizeZeros(a);
}

/** zeta(f)[S] = sum of f over supersets of S. */
function supersetZeta(values) {
  checkLength(values, "superset zeta transform");
  const a = values.slice();
  const k = bitsOf(a.length);
  for (let bit = 0; bit < k; bit++) {
    const step = 1 << bit;
    for (let mask = 0; mask < a.length; mask++) {
      if (!(mask & step)) a[mask] += a[mask | step];
    }
  }
  return normalizeZeros(a);
}

function supersetMobius(values) {
  checkLength(values, "superset Möbius transform");
  const a = values.slice();
  const k = bitsOf(a.length);
  for (let bit = 0; bit < k; bit++) {
    const step = 1 << bit;
    for (let mask = 0; mask < a.length; mask++) {
      if (!(mask & step)) a[mask] -= a[mask | step];
    }
  }
  return normalizeZeros(a);
}

/** Brute-force definitions - the independent ground truth. */
function subsetZetaBrute(values) {
  const n = values.length;
  const out = new Array(n).fill(0);
  for (let mask = 0; mask < n; mask++) {
    let total = 0;
    for (let sub = mask; ; sub = (sub - 1) & mask) {
      total += values[sub];
      if (sub === 0) break;
    }
    out[mask] = total;
  }
  return out;
}

function supersetZetaBrute(values) {
  const n = values.length;
  const full = n - 1;
  const out = new Array(n).fill(0);
  for (let mask = 0; mask < n; mask++) {
    let total = 0;
    for (let sup = mask; ; sup = (sup + 1) | mask) {
      total += values[sup];
      if (sup === full) break;
    }
    out[mask] = total;
  }
  return out;
}

/** OR-convolution via the zeta/Möbius pair: zeta turns OR into pointwise product. */
function orConvolution(a, b) {
  if (a.length !== b.length) throw new Error("length mismatch");
  checkLength(a, "OR convolution");
  const za = subsetZeta(a);
  const zb = subsetZeta(b);
  return subsetMobius(za.map((value, i) => value * zb[i]));
}

function orConvolutionNaive(a, b) {
  const n = a.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) out[i | j] += a[i] * b[j];
  }
  return out;
}

function randomVector(n, seed, limit = 13) {
  const values = [];
  let s = seed % 65537;
  for (let i = 0; i < n; i++) {
    s = lcgNext(s);
    values.push((s % limit) - Math.floor(limit / 2));
  }
  return values;
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-checked k = 1 and k = 2 lattices.
  assert.deepStrictEqual(subsetZeta([3, 5]), [3, 8]);
  assert.deepStrictEqual(subsetMobius([3, 8]), [3, 5]);
  assert.deepStrictEqual(subsetZeta([1, 2, 3, 4]), [1, 3, 4, 10]);
  assert.deepStrictEqual(supersetZeta([1, 2, 3, 4]), [10, 6, 7, 4]);
  assert.deepStrictEqual(subsetZetaBrute([1, 2, 3, 4]), [1, 3, 4, 10]);
  assert.deepStrictEqual(supersetZetaBrute([1, 2, 3, 4]), [10, 6, 7, 4]);

  // (2) Transform versus brute-force enumeration, both directions, for k <= 5.
  for (let k = 1; k <= 5; k++) {
    const n = 1 << k;
    for (let seed = 1; seed <= 4; seed++) {
      const f = randomVector(n, seed * 23 + k, 17);
      assert.deepStrictEqual(subsetZeta(f), subsetZetaBrute(f),
        `k=${k} seed ${seed}: subset zeta matches enumeration`);
      assert.deepStrictEqual(supersetZeta(f), supersetZetaBrute(f),
        `k=${k} seed ${seed}: superset zeta matches enumeration`);
      assert.deepStrictEqual(subsetMobius(subsetZeta(f)), f, `k=${k}: mobius undoes zeta`);
      assert.deepStrictEqual(subsetZeta(subsetMobius(f)), f, `k=${k}: zeta undoes mobius`);
      assert.deepStrictEqual(supersetMobius(supersetZeta(f)), f,
        `k=${k}: superset pair is inverse`);
      assert.deepStrictEqual(supersetZeta(supersetMobius(f)), f);
      assert.deepStrictEqual(subsetMobius(subsetZetaBrute(f)), f,
        `k=${k}: inversion works against the brute-force forward transform`);
      assert.deepStrictEqual(supersetMobius(supersetZetaBrute(f)), f);
    }
  }

  // (3) Closed forms.
  for (let k = 1; k <= 5; k++) {
    const n = 1 << k;
    const ones = new Array(n).fill(1);
    const zetaOnes = subsetZeta(ones);
    for (let mask = 0; mask < n; mask++) {
      assert.strictEqual(zetaOnes[mask], 2 ** popcount(mask),
        `all-ones subset sums to 2^popcount(${mask})`);
    }
    const superOnes = supersetZeta(ones);
    for (let mask = 0; mask < n; mask++) {
      assert.strictEqual(superOnes[mask], 2 ** (k - popcount(mask)),
        `all-ones superset sums to 2^(k - popcount(${mask}))`);
    }
    for (let target = 0; target < n; target++) {
      const delta = new Array(n).fill(0);
      delta[target] = 1;
      const zetaDelta = subsetZeta(delta);
      for (let mask = 0; mask < n; mask++) {
        const isSuperset = (mask & target) === target;
        assert.strictEqual(zetaDelta[mask], isSuperset ? 1 : 0,
          `zeta(delta_${target}) must indicate supersets`);
      }
    }
  }

  // (4) Grand total identity and lattice monotonicity.
  for (let k = 1; k <= 5; k++) {
    const n = 1 << k;
    const f = randomVector(n, 500 + k, 17);
    const zeta = subsetZeta(f);
    let lhs = 0;
    for (const value of zeta) lhs += value;
    let rhs = 0;
    for (let mask = 0; mask < n; mask++) rhs += f[mask] * 2 ** (k - popcount(mask));
    assert.strictEqual(lhs, rhs, `k=${k}: sum of zeta = sum f[T] * 2^(k - |T|)`);
    const positive = f.map((value) => value + 20);
    const zetaPositive = subsetZeta(positive);
    for (let sub = 0; sub < n; sub++) {
      for (let sup = 0; sup < n; sup++) {
        if ((sub & sup) === sub) {
          assert.ok(zetaPositive[sub] <= zetaPositive[sup],
            "non-negative f makes the subset sums monotone in the lattice");
        }
      }
    }
    assert.strictEqual(zeta[n - 1], f.reduce((acc, value) => acc + value, 0),
      "the full mask sums everything exactly once");
  }

  // (5) OR-convolution: transform method versus the naive O(4^k) definition.
  for (let k = 1; k <= 5; k++) {
    const n = 1 << k;
    for (let seed = 7; seed <= 10; seed++) {
      const a = randomVector(n, seed * 37 + k, 13);
      const b = randomVector(n, seed * 71 + k, 13);
      const viaTransform = orConvolution(a, b);
      const naive = orConvolutionNaive(a, b);
      assert.deepStrictEqual(viaTransform, naive, `k=${k} seed ${seed}: OR convolution`);
      assert.ok(viaTransform.every(Number.isInteger), "integer inputs give integer outputs");
    }
    const identity = new Array(n).fill(0);
    identity[0] = 1;
    const a = randomVector(n, 999 + k, 13);
    assert.deepStrictEqual(orConvolution(a, identity), a, "delta at the empty set is the identity");
    assert.deepStrictEqual(orConvolution(a, bOrEmpty(k)), orConvolutionNaive(a, bOrEmpty(k)));
  }
  const a4 = randomVector(8, 4, 13);
  const b4 = randomVector(8, 8, 13);
  const c4 = randomVector(8, 12, 13);
  assert.deepStrictEqual(orConvolution(orConvolution(a4, b4), c4),
    orConvolution(a4, orConvolution(b4, c4)), "OR convolution is associative");
  assert.deepStrictEqual(orConvolution(a4, b4), orConvolution(b4, a4), "and commutative");

  // (6) OR-convolution counts unions of explicit sets - a third, index-free code path.
  const setA = [0, 1, 2, 4];
  const setB = [1, 2, 4, 5];
  const indicatorA = new Array(8).fill(0);
  const indicatorB = new Array(8).fill(0);
  for (const v of setA) indicatorA[v] += 1;
  for (const v of setB) indicatorB[v] += 1;
  const unionCounts = new Array(8).fill(0);
  for (const i of setA) for (const j of setB) unionCounts[i | j] += 1;
  assert.deepStrictEqual(orConvolution(indicatorA, indicatorB), unionCounts);

  // (7) Guard rails and determinism.
  assert.throws(() => subsetZeta([1, 2, 3]), /power-of-two/);
  assert.throws(() => subsetMobius([1, 2, 3]), /power-of-two/);
  assert.throws(() => supersetZeta([1, 2, 3]), /power-of-two/);
  assert.throws(() => orConvolution([1, 2], [1, 2, 3, 4]), /length mismatch/);
  const repeat = randomVector(16, 31337, 9);
  assert.deepStrictEqual(subsetZeta(repeat), subsetZeta(repeat));
  assert.deepStrictEqual(subsetMobius(subsetZeta(repeat)), repeat);
  assert.deepStrictEqual(supersetMobius(supersetZeta(repeat)), repeat);
  assert.deepStrictEqual(supersetMobius(supersetZetaBrute(repeat)), repeat);
}

/** Small helper used to keep the identity check readable: an arbitrary second operand. */
function bOrEmpty(k) {
  const n = 1 << k;
  const values = new Array(n).fill(0);
  values[n - 1] = 1;
  return values;
}

module.exports = { lcgNext, isPowerOfTwo, popcount, bitsOf, subsetZeta, subsetMobius,
  supersetZeta, supersetMobius, subsetZetaBrute, supersetZetaBrute, orConvolution,
  orConvolutionNaive, randomVector };

if (require.main === module) {
  main();
  console.log("PASS");
}
