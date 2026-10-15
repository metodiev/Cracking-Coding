/** Task 131 - Generate all subsets recursive. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function subsetsRecursive(items) {
  const out = [];

  function backtrack(i, current) {
    if (i === items.length) {
      out.push([...current]);
      return;
    }
    backtrack(i + 1, current);            // skip items[i]
    current.push(items[i]);
    backtrack(i + 1, current);            // take items[i]
    current.pop();
  }

  backtrack(0, []);
  return out;
}

module.exports = { subsetsRecursive };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(subsetsRecursive([1, 2, 3]), [
    [], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3],
  ]);
  assert.deepStrictEqual(subsetsRecursive([]), [[]]);
  assert.strictEqual(subsetsRecursive([1, 2, 3, 4]).length, 16);
  console.log("PASS");
}
