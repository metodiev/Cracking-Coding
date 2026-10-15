/** Task 098 - Transpose a 2D matrix. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function transpose(matrix) {
  if (matrix.length === 0) return [];
  const rows = matrix.length;
  const cols = matrix[0].length;
  const result = [];
  for (let c = 0; c < cols; c++) {
    const newRow = [];
    for (let r = 0; r < rows; r++) {
      newRow.push(matrix[r][c]);
    }
    result.push(newRow);
  }
  return result;
}

module.exports = { transpose };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]]);
  assert.deepStrictEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]]);
  assert.deepStrictEqual(transpose([[1, 2, 3]]), [[1], [2], [3]]);
  assert.deepStrictEqual(transpose([[1], [2], [3]]), [[1, 2, 3]]);
  assert.deepStrictEqual(transpose([[1]]), [[1]]);
  assert.deepStrictEqual(transpose([]), []);
  assert.deepStrictEqual(transpose([[]]), []);
  assert.deepStrictEqual(transpose([[1, 2], [3, 4], [5, 6]]), [[1, 3, 5], [2, 4, 6]]);

  const m = [[1, 2], [3, 4]];
  const t = transpose(m);
  assert.notStrictEqual(t, m);
  assert.deepStrictEqual(m, [[1, 2], [3, 4]]);
  console.log("PASS");
}
