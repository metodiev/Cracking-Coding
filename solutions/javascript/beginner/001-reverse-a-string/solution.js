/** Task 001 - Reverse a string. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function reverseString(s) {
  return [...s].reverse().join("");
}

module.exports = { reverseString };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(reverseString("hello"), "olleh");
  assert.strictEqual(reverseString(""), "");
  assert.strictEqual(reverseString("ab"), "ba");
  console.log("PASS");
}
