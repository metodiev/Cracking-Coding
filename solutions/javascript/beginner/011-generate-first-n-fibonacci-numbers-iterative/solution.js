/** Task 011 - Generate first N Fibonacci numbers (iterative). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function fibSequence(n) {
  const seq = [];
  let a = 0, b = 1;
  for (let i = 0; i < n; i++) {
    seq.push(a);
    [a, b] = [b, a + b];
  }
  return seq;
}

module.exports = { fibSequence };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(fibSequence(0), []);
  assert.deepStrictEqual(fibSequence(8), [0, 1, 1, 2, 3, 5, 8, 13]);
  console.log("PASS");
}
