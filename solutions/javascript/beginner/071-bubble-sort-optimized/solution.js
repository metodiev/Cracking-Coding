/** Task 071 - Bubble sort (optimized). (Tier: Beginner, CTCI Ch 10) */
"use strict";

function bubbleSort(a) {
  let end = a.length;
  while (end > 1) {
    let lastSwap = 0;
    for (let i = 1; i < end; i++) {
      if (a[i - 1] > a[i]) {
        [a[i - 1], a[i]] = [a[i], a[i - 1]];
        lastSwap = i;
      }
    }
    if (lastSwap === 0) break;
    end = lastSwap;
  }
  return a;
}

module.exports = { bubbleSort };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(bubbleSort([5, 3, 8, 1, 9, 2]), [1, 2, 3, 5, 8, 9]);
  assert.deepStrictEqual(bubbleSort([1, 2, 3]), [1, 2, 3]);
  assert.deepStrictEqual(bubbleSort([]), []);
  console.log("PASS");
}
