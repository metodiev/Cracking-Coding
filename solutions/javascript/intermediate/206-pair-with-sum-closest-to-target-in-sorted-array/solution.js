/** Task 206 - Pair with sum closest to target in sorted array. (Tier: Intermediate, CTCI Ch 16) */
"use strict";

function closestPair(a, target) {
  if (a.length < 2) throw new Error("need at least two values");
  let lo = 0, hi = a.length - 1;
  let bestX = a[lo], bestY = a[hi];
  let bestDiff = Math.abs(bestX + bestY - target);
  while (lo < hi) {
    const total = a[lo] + a[hi];
    const diff = Math.abs(total - target);
    if (diff < bestDiff) {
      bestDiff = diff;
      bestX = a[lo];
      bestY = a[hi];
    }
    if (total < target) lo++;
    else if (total > target) hi--;
    else break;
  }
  return { pair: [bestX, bestY], sum: bestX + bestY };
}

module.exports = { closestPair };

if (require.main === module) {
  const assert = require("node:assert");
  const a = [1, 3, 4, 5, 7];
  assert.deepStrictEqual(closestPair(a, 10), { pair: [3, 7], sum: 10 });
  assert.deepStrictEqual(closestPair(a, 8), { pair: [1, 7], sum: 8 });
  assert.deepStrictEqual(closestPair(a, 0), { pair: [1, 3], sum: 4 });
  assert.throws(() => closestPair([1], 5), /need at least two/);
  console.log("PASS");
}
