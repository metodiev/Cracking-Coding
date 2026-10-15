/** Task 018 - Convert integer to string (manual itoa). (Tier: Beginner, CTCI Ch 1) */
"use strict";

function itoa(n) {
  if (n === 0) return "0";
  const negative = n < 0;
  let value = Math.abs(n);
  const digits = [];
  while (value > 0) {
    digits.push(String.fromCharCode(48 + (value % 10)));
    value = Math.floor(value / 10);
  }
  digits.reverse();
  return (negative ? "-" : "") + digits.join("");
}

module.exports = { itoa };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(itoa(0), "0");
  assert.strictEqual(itoa(123), "123");
  assert.strictEqual(itoa(-456), "-456");
  console.log("PASS");
}
