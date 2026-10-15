/** Task 225 - Stock with transaction fee (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function maxProfitFee(prices, fee) {
  if (fee < 0) throw new Error("fee must be non-negative");
  if (prices.length === 0) return 0;
  let cash = 0, hold = -prices[0];
  for (let i = 1; i < prices.length; i++) {
    const p = prices[i];
    const nextHold = Math.max(hold, cash - p);
    const nextCash = Math.max(cash, hold + p - fee);
    hold = nextHold;
    cash = nextCash;
  }
  return cash;
}

module.exports = { maxProfitFee };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxProfitFee([1, 3, 2, 8, 4, 9], 2), 8);
  assert.strictEqual(maxProfitFee([1, 3, 2, 8, 4, 9], 0), 13);
  assert.strictEqual(maxProfitFee([1, 2], 5), 0);
  assert.strictEqual(maxProfitFee([1, 2], 0), 1);
  assert.strictEqual(maxProfitFee([1, 4], 1), 2);
  assert.strictEqual(maxProfitFee([], 2), 0);
  assert.strictEqual(maxProfitFee([5], 1), 0);
  assert.strictEqual(maxProfitFee([5], 0), 0);
  assert.throws(() => maxProfitFee([1, 2], -1), /non-negative/);
  console.log("PASS");
}
