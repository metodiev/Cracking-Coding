/** Task 170 - Simple palindrome partitioning backtracking (list partitions). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function isPalindrome(s) {
  for (let i = 0, j = s.length - 1; i < j; i++, j--) {
    if (s[i] !== s[j]) return false;
  }
  return true;
}

function palindromePartitions(s) {
  const out = [];

  function backtrack(start, current) {
    if (start === s.length) {
      out.push([...current]);
      return;
    }
    for (let end = start + 1; end <= s.length; end++) {
      const piece = s.slice(start, end);
      if (isPalindrome(piece)) {
        current.push(piece);
        backtrack(end, current);
        current.pop();
      }
    }
  }

  backtrack(0, []);
  return out;
}

module.exports = { isPalindrome, palindromePartitions };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(palindromePartitions("aab"), [["a", "a", "b"], ["aa", "b"]]);
  assert.deepStrictEqual(palindromePartitions("a"), [["a"]]);
  assert.deepStrictEqual(palindromePartitions(""), [[]]);
  assert.deepStrictEqual(palindromePartitions("aaa"), [["a", "a", "a"], ["a", "aa"], ["aa", "a"], ["aaa"]]);
  assert.deepStrictEqual(palindromePartitions("ab"), [["a", "b"]]);
  console.log("PASS");
}
