/** Task 097 - Flatten a 2D array (row-major). (Tier: Beginner, CTCI Ch 1) */
"use strict";

function flatten2D(matrix) {
  const result = [];
  for (const row of matrix) {
    for (const value of row) {
      result.push(value);
    }
  }
  return result;
}

module.exports = { flatten2D };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(flatten2D([[1, 2], [3, 4], [5]]), [1, 2, 3, 4, 5]);
  assert.deepStrictEqual(flatten2D([[1], [2, 3], [4, 5, 6]]), [1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual(flatten2D([[1, 2, 3], [4, 5, 6]]), [1, 2, 3, 4, 5, 6]);
  assert.deepStrictEqual(flatten2D([[1, 2], [3, 4]]), [1, 2, 3, 4]);
  assert.deepStrictEqual(flatten2D([[], [1, 2], []]), [1, 2]);
  assert.deepStrictEqual(flatten2D([[1]]), [1]);
  assert.deepStrictEqual(flatten2D([]), []);
  assert.deepStrictEqual(flatten2D([[]]), []);
  assert.deepStrictEqual(flatten2D([[], []]), []);
  assert.deepStrictEqual(flatten2D([[-1, 0], [0, -1]]), [-1, 0, 0, -1]);
  console.log("PASS");
}
