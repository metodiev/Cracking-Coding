/** Task 166 - Boyer-Moore majority vote algorithm (element > n/2). (Tier: Beginner, CTCI Ch 6) */
"use strict";

function majorityElement(nums) {
  let candidate = null, count = 0;
  for (const x of nums) {
    if (count === 0) {
      candidate = x;
      count = 1;
    } else if (x === candidate) {
      count++;
    } else {
      count--;
    }
  }
  if (candidate === null) return null;
  let occurrences = 0;
  for (const x of nums) if (x === candidate) occurrences++;
  return occurrences > nums.length / 2 ? candidate : null;
}

module.exports = { majorityElement };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(majorityElement([3, 2, 3]), 3);
  assert.strictEqual(majorityElement([2, 2, 1, 1, 1, 2, 2]), 2);
  assert.strictEqual(majorityElement([1]), 1);
  assert.strictEqual(majorityElement([]), null);
  assert.strictEqual(majorityElement([1, 2, 3]), null);
  assert.strictEqual(majorityElement([1, 1, 2, 2]), null);
  console.log("PASS");
}
