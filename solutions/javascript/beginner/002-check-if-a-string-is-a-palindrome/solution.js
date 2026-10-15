/** Task 002 - Check if a string is a palindrome. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function isPalindrome(s) {
  for (let i = 0, j = s.length - 1; i < j; i++, j--)
    if (s[i] !== s[j]) return false;
  return true;
}

module.exports = { isPalindrome };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isPalindrome("racecar"), true);
  assert.strictEqual(isPalindrome(""), true);
  assert.strictEqual(isPalindrome("hello"), false);
  console.log("PASS");
}
