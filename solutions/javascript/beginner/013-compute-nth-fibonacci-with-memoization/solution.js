/** Task 013 - Compute nth Fibonacci with memoization. O(n). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function fibMemo(n, cache = new Map([[0, 0n], [1, 1n]])) {
  if (cache.has(n)) return cache.get(n);
  const value = fibMemo(n - 1, cache) + fibMemo(n - 2, cache);
  cache.set(n, value);
  return value;
}

module.exports = { fibMemo };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(fibMemo(50), 12586269025n);
  assert.strictEqual(fibMemo(90), 2880067194370816120n);
  console.log("PASS");
}
