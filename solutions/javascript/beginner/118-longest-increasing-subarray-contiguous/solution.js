/** Task 118 - Longest increasing subarray (contiguous). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function longestIncreasingSubarray(a) {
  if (a.length === 0) return [0, -1];
  let bestLen = 1, bestStart = 0, curLen = 1, curStart = 0;
  for (let i = 1; i < a.length; i++) {
    if (a[i] > a[i - 1]) {
      curLen++;
    } else {
      curLen = 1;
      curStart = i;
    }
    if (curLen > bestLen) {
      bestLen = curLen;
      bestStart = curStart;
    }
  }
  return [bestLen, bestStart];
}

module.exports = { longestIncreasingSubarray };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(longestIncreasingSubarray([1, 2, 3, 1, 2]), [3, 0]);
  assert.deepStrictEqual(longestIncreasingSubarray([5, 4, 3]), [1, 0]);
  assert.deepStrictEqual(longestIncreasingSubarray([1, 3, 2, 4, 5, 6]), [4, 2]);
  assert.deepStrictEqual(longestIncreasingSubarray([]), [0, -1]);
  console.log("PASS");
}
