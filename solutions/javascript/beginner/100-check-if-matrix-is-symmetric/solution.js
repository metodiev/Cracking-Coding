/** Task 100 - Check if matrix is symmetric. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function isSymmetric(m) {
  const n = m.length;
  for (const row of m) {
    if (row.length !== n) return false;
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (m[i][j] !== m[j][i]) return false;
    }
  }
  return true;
}

module.exports = { isSymmetric };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isSymmetric([[1, 2], [2, 1]]), true);
  assert.strictEqual(isSymmetric([[1, 2], [3, 4]]), false);
  assert.strictEqual(isSymmetric([[5]]), true);
  assert.strictEqual(isSymmetric([]), true);
  assert.strictEqual(isSymmetric([[]]), false);
  assert.strictEqual(isSymmetric([[1, 2, 3], [2, 4, 5], [3, 5, 6]]), true);
  assert.strictEqual(isSymmetric([[1, 2, 3], [2, 4, 5], [3, 6, 6]]), false);
  assert.strictEqual(isSymmetric([[1, 2, 3], [4, 5, 6]]), false);
  assert.strictEqual(isSymmetric([[1, 2], [2]]), false);
  assert.strictEqual(isSymmetric([[0, -1], [-1, 2]]), true);
  assert.strictEqual(isSymmetric([[0, -1], [1, 2]]), false);
  console.log("PASS");
}
