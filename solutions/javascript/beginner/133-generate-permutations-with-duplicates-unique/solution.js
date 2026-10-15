/** Task 133 - Generate permutations with duplicates (unique). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function uniquePermutations(items) {
  const sorted = [...items].sort((a, b) => a - b);
  const used = new Array(sorted.length).fill(false);
  const out = [];

  function backtrack(current) {
    if (current.length === sorted.length) {
      out.push([...current]);
      return;
    }
    for (let i = 0; i < sorted.length; i++) {
      if (used[i]) continue;
      if (i > 0 && sorted[i] === sorted[i - 1] && !used[i - 1]) continue;
      used[i] = true;
      current.push(sorted[i]);
      backtrack(current);
      current.pop();
      used[i] = false;
    }
  }

  backtrack([]);
  return out;
}

module.exports = { uniquePermutations };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(uniquePermutations([1, 1, 2]), [[1, 1, 2], [1, 2, 1], [2, 1, 1]]);
  assert.deepStrictEqual(uniquePermutations([1, 1, 1]), [[1, 1, 1]]);
  assert.strictEqual(uniquePermutations([1, 2, 3]).length, 6);
  assert.strictEqual(uniquePermutations([1, 2, 2, 3]).length, 12);
  console.log("PASS");
}
