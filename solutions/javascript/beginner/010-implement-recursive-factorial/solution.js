/** Task 010 - Implement recursive factorial. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function factorialRecursive(n) {
  if (n < 0) throw new Error("n must be >= 0");
  return n <= 1 ? 1n : BigInt(n) * factorialRecursive(n - 1);
}

module.exports = { factorialRecursive };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(factorialRecursive(0), 1n);
  assert.strictEqual(factorialRecursive(5), 120n);
  assert.strictEqual(factorialRecursive(10), 3628800n);
  console.log("PASS");
}
