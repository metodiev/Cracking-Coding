/** Task 099 - Rotate matrix 90 degrees clockwise (in-place for square). (Tier: Beginner, CTCI Ch 1) */
"use strict";

function rotateMatrix90Clockwise(m) {
  const n = m.length;
  for (const row of m) {
    if (row.length !== n) throw new Error("matrix must be square");
  }
  // Transpose across the main diagonal.
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const tmp = m[i][j];
      m[i][j] = m[j][i];
      m[j][i] = tmp;
    }
  }
  // Reverse every row.
  for (const row of m) {
    for (let lo = 0, hi = row.length - 1; lo < hi; lo++, hi--) {
      const tmp = row[lo];
      row[lo] = row[hi];
      row[hi] = tmp;
    }
  }
  return m;
}

module.exports = { rotateMatrix90Clockwise };

if (require.main === module) {
  const assert = require("node:assert");
  const m1 = [[1]];
  assert.deepStrictEqual(rotateMatrix90Clockwise(m1), [[1]]);

  const m2 = [[1, 2], [3, 4]];
  assert.deepStrictEqual(rotateMatrix90Clockwise(m2), [[3, 1], [4, 2]]);

  const m3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
  assert.deepStrictEqual(rotateMatrix90Clockwise(m3), [[7, 4, 1], [8, 5, 2], [9, 6, 3]]);

  const m4 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]];
  assert.deepStrictEqual(rotateMatrix90Clockwise(m4), [
    [13, 9, 5, 1],
    [14, 10, 6, 2],
    [15, 11, 7, 3],
    [16, 12, 8, 4],
  ]);

  const m5 = [[1, 2], [3, 4]];
  assert.strictEqual(rotateMatrix90Clockwise(m5), m5); // same array: in place
  assert.deepStrictEqual(m5, [[3, 1], [4, 2]]);

  assert.throws(() => rotateMatrix90Clockwise([[1, 2, 3], [4, 5, 6]]), /square/);
  console.log("PASS");
}
