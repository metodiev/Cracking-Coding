/** Task 009 - Implement iterative factorial. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function factorialIterative(n) {
  if (n < 0) throw new Error("n must be >= 0");
  let result = 1n;
  for (let i = 2n; i <= BigInt(n); i++) result *= i;
  return result;   // BigInt: exact for any n (numbers overflow past 20!)
}

module.exports = { factorialIterative };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(factorialIterative(0), 1n);
  assert.strictEqual(factorialIterative(5), 120n);
  assert.strictEqual(factorialIterative(20), 2432902008176640000n);
  console.log("PASS");
}
