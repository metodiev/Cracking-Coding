/** Task 224 - Stock with cooldown (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function maxProfitCooldown(prices) {
  if (prices.length === 0) return 0;
  let held = -prices[0];                    // bought on day 0
  let sold = -Infinity;                     // no sale yet
  let rest = 0;                             // idle on day 0
  for (let i = 1; i < prices.length; i++) {
    const p = prices[i];
    const nextHeld = Math.max(held, rest - p);
    const nextSold = held + p;
    const nextRest = Math.max(rest, sold);
    held = nextHeld;
    sold = nextSold;
    rest = nextRest;
  }
  return Math.max(sold, rest);
}

module.exports = { maxProfitCooldown };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(maxProfitCooldown([1, 2, 3, 0, 2]), 3);
  assert.strictEqual(maxProfitCooldown([1]), 0);
  assert.strictEqual(maxProfitCooldown([]), 0);
  assert.strictEqual(maxProfitCooldown([2, 1, 4]), 3);
  assert.strictEqual(maxProfitCooldown([6, 1, 3, 2, 4, 7]), 6);
  assert.strictEqual(maxProfitCooldown([1, 2]), 1);
  assert.strictEqual(maxProfitCooldown([2, 2, 2]), 0);
  console.log("PASS");
}
