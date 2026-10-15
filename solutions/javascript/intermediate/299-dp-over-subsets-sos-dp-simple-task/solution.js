/** Task 299 - DP over subsets (SOS DP simple task). (Tier: Intermediate, CTCI Beyond — Networking, OS & Real-Time) */
"use strict";

function sumOverSubsets(a) {
  const n = Math.log2(a.length);
  if (!Number.isInteger(n) || a.length === 0) throw new Error("length must be a power of two");
  const f = a.slice();
  for (let i = 0; i < n; i++) {
    const bit = 1 << i;
    for (let mask = 0; mask < f.length; mask++) {
      if (mask & bit) f[mask] += f[mask ^ bit];
    }
  }
  return f;
}

function bruteSos(a) {
  const result = [];
  for (let mask = 0; mask < a.length; mask++) {
    let total = 0;
    let sub = mask;
    for (;;) {
      total += a[sub];
      if (sub === 0) break;
      sub = (sub - 1) & mask;
    }
    result.push(total);
  }
  return result;
}

function lcgArray(size, seed) {
  const values = [];
  for (let i = 0; i < size; i++) {
    seed = (seed * 75 + 74) % 65537;
    values.push(seed % 50);
  }
  return [values, seed];
}

function popcount(value) {
  let count = 0;
  while (value !== 0) {
    count += value & 1;
    value >>>= 1;
  }
  return count;
}

module.exports = { sumOverSubsets, bruteSos, lcgArray, popcount };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(sumOverSubsets([1, 2, 4, 8]), [1, 3, 5, 15]);
  assert.deepStrictEqual(sumOverSubsets([7]), [7]);

  let seed = 606;
  for (const n of [1, 2, 3, 4, 10]) {
    let a;
    [a, seed] = lcgArray(1 << n, seed);
    const f = sumOverSubsets(a);
    assert.deepStrictEqual(f, bruteSos(a), `n = ${n} vs brute force`);
    // each a[sub] lands in exactly the 2^(n - popcount(sub)) supersets of sub
    const expected = a.reduce((acc, value, mask) => acc + value * 2 ** (n - popcount(mask)), 0);
    assert.strictEqual(f.reduce((x, y) => x + y, 0), expected, `n = ${n} superset identity`);
  }
  console.log("PASS");
}
