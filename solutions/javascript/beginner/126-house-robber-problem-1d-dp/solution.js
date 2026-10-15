/** Task 126 - House robber problem (1D DP). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function rob(houses) {
  let take = 0, skip = 0;
  for (const value of houses) {
    [take, skip] = [skip + value, Math.max(skip, take)];
  }
  return Math.max(take, skip);
}

module.exports = { rob };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(rob([2, 7, 9, 3, 1]), 12);
  assert.strictEqual(rob([]), 0);
  assert.strictEqual(rob([5]), 5);
  assert.strictEqual(rob([2, 1, 1, 2]), 4);
  assert.strictEqual(rob([5, 1, 1, 5]), 10);
  console.log("PASS");
}
