/** Task 012 - Compute nth Fibonacci using recursion. O(2^n), intentionally naive. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function fib(n) {
  return n <= 1 ? n : fib(n - 1) + fib(n - 2);
}

module.exports = { fib };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(fib(10), 55);
  assert.strictEqual(fib(20), 6765);
  console.log("PASS");
}
