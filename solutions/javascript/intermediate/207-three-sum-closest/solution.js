/** Task 207 - Three-sum closest. (Tier: Intermediate, CTCI Ch 16) */
"use strict";

function threeSumClosest(nums, target) {
  if (nums.length < 3) throw new Error("need at least three values");
  const a = [...nums].sort((x, y) => x - y);
  let best = a[0] + a[1] + a[2];
  for (let i = 0; i + 2 < a.length; i++) {
    let lo = i + 1, hi = a.length - 1;
    while (lo < hi) {
      const total = a[i] + a[lo] + a[hi];
      if (Math.abs(total - target) < Math.abs(best - target)) best = total;
      if (total < target) lo++;
      else if (total > target) hi--;
      else return total;
    }
  }
  return best;
}

module.exports = { threeSumClosest };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(threeSumClosest([-1, 2, 1, -4], 1), 2);
  assert.strictEqual(threeSumClosest([0, 0, 0], 1), 0);
  assert.strictEqual(threeSumClosest([1, 1, 1, 1], 5), 3);
  assert.strictEqual(threeSumClosest([-5, -2, -1], -3), -8);
  assert.throws(() => threeSumClosest([1, 2], 3), /need at least three/);
  console.log("PASS");
}
