/** Task 132 - Generate permutations of distinct elements. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function permutations(items) {
  const used = new Array(items.length).fill(false);
  const out = [];

  function backtrack(current) {
    if (current.length === items.length) {
      out.push([...current]);
      return;
    }
    for (let i = 0; i < items.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      current.push(items[i]);
      backtrack(current);
      current.pop();
      used[i] = false;
    }
  }

  backtrack([]);
  return out;
}

module.exports = { permutations };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(permutations([1, 2, 3]), [
    [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1],
  ]);
  assert.deepStrictEqual(permutations([]), [[]]);
  assert.strictEqual(permutations([1, 2, 3, 4]).length, 24);
  console.log("PASS");
}
