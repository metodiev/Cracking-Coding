/** Task 174 - Implement basic matrix multiplication (naive). (Tier: Beginner, CTCI Ch 6) */
"use strict";

function matmul(a, b) {
  if (a.length === 0 || b.length === 0) return [];
  const n = a.length, m = a[0].length;
  const m2 = b.length, p = b[0].length;
  if (m !== m2) throw new Error(`dimension mismatch: ${n}x${m} vs ${m2}x${p}`);
  const c = Array.from({ length: n }, () => new Array(p).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      let total = 0;
      for (let k = 0; k < m; k++) total += a[i][k] * b[k][j];
      c[i][j] = total;
    }
  }
  return c;
}

module.exports = { matmul };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]), [[58, 64], [139, 154]]);
  const identity = [[1, 0], [0, 1]];
  const m = [[5, 6], [7, 8]];
  assert.deepStrictEqual(matmul(m, identity), m);
  assert.deepStrictEqual(matmul([[1, 2]], [[3], [4]]), [[11]]);
  assert.throws(() => matmul([[1, 2]], [[1, 2]]), /dimension mismatch/);
  console.log("PASS");
}
