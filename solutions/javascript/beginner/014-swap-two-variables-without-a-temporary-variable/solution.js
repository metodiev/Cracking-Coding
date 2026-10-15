/** Task 014 - Swap two variables without a temporary variable. (Tier: Beginner, CTCI Ch 5) */
"use strict";

function swapArithmetic(a, b) {
  a = a + b;
  b = a - b;
  a = a - b;
  return [a, b];
}

function swapXor(a, b) {
  a ^= b;
  b ^= a;
  a ^= b;
  return [a, b];
}

module.exports = { swapArithmetic, swapXor };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(swapArithmetic(5, 3), [3, 5]);
  assert.deepStrictEqual(swapXor(5, 3), [3, 5]);
  assert.deepStrictEqual(swapArithmetic(-1, 7), [7, -1]);
  console.log("PASS");
}
