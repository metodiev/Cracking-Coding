/** Task 129 - Implement tail recursion (if language supports). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function factorialTail(n, acc = 1) {
  return n <= 1 ? acc : factorialTail(n - 1, n * acc);
}

function trampoline(fn, ...args) {
  let result = fn(...args);
  while (typeof result === "function") result = result();
  return result;
}

function factorialTrampolined(n) {
  // BigInt arithmetic cannot mix with Number literals: use 1n and BigInt operands
  const step = (remaining, acc) =>
    remaining <= 1n ? acc : () => step(remaining - 1n, acc * remaining);
  return trampoline(step, BigInt(n), 1n);
}

module.exports = { factorialTail, factorialTrampolined, trampoline };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(factorialTail(5), 120);
  assert.strictEqual(factorialTail(20), 2432902008176640000);

  const deep = factorialTrampolined(20000);
  let expected = 1n;
  for (let i = 2n; i <= 20000n; i++) expected *= i;
  assert.strictEqual(deep, expected, "trampoline matches 20000!");

  let raised = false;
  try {
    factorialTail(100000);          // blows node's stack: no TCO in V8
  } catch (e) {
    raised = e instanceof RangeError;
  }
  assert.ok(raised, "plain tail recursion is not stack safe in JS");
  console.log("PASS");
}
