/** Task 223 - Stock buy/sell multiple transactions (greedy). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function maxProfitMultiple(prices) {
  let total = 0;
  for (let i = 1; i < prices.length; i++) {
    if (prices[i] > prices[i - 1]) total += prices[i] - prices[i - 1];
  }
  return total;
}

module.exports = { maxProfitMultiple };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxProfitMultiple([7, 1, 5, 3, 6, 4]), 7);
  assert.strictEqual(maxProfitMultiple([7, 6, 4, 3, 1]), 0);
  assert.strictEqual(maxProfitMultiple([1, 2, 3, 4, 5]), 4);
  assert.strictEqual(maxProfitMultiple([1, 3, 2, 8]), 8);
  assert.strictEqual(maxProfitMultiple([]), 0);
  assert.strictEqual(maxProfitMultiple([5]), 0);
  assert.strictEqual(maxProfitMultiple([3, 3]), 0);
  console.log("PASS");
}
