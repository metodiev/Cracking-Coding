/** Task 222 - Stock buy/sell single transaction max profit. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function maxProfitSingle(prices) {
  let best = 0, cheapest = Infinity;
  for (const p of prices) {
    cheapest = Math.min(cheapest, p);
    best = Math.max(best, p - cheapest);
  }
  return best;
}

module.exports = { maxProfitSingle };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxProfitSingle([7, 1, 5, 3, 6, 4]), 5);
  assert.strictEqual(maxProfitSingle([7, 6, 4, 3, 1]), 0);
  assert.strictEqual(maxProfitSingle([1]), 0);
  assert.strictEqual(maxProfitSingle([]), 0);
  assert.strictEqual(maxProfitSingle([1, 2]), 1);
  assert.strictEqual(maxProfitSingle([2, 1]), 0);
  assert.strictEqual(maxProfitSingle([3, 3]), 0);
  assert.strictEqual(maxProfitSingle([5, 4, 6, 2, 7]), 5);
  console.log("PASS");
}
