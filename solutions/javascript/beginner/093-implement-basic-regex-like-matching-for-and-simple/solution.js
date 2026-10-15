/** Task 093 - Implement basic regex-like matching for '.' and '*'. (Tier: Beginner, CTCI Ch 8) */
"use strict";

function isMatch(s, p) {
  const memo = new Map();

  function rec(i, j) {
    const key = i + "," + j;
    if (memo.has(key)) return memo.get(key);
    let res;
    if (j === p.length) {
      res = i === s.length;
    } else {
      const first = i < s.length && (p[j] === s[i] || p[j] === ".");
      if (j + 1 < p.length && p[j + 1] === "*") {
        res = rec(i, j + 2) || (first && rec(i + 1, j));
      } else {
        res = first && rec(i + 1, j + 1);
      }
    }
    memo.set(key, res);
    return res;
  }

  return rec(0, 0);
}

module.exports = { isMatch };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isMatch("", ""), true);
  assert.strictEqual(isMatch("", "a"), false);
  assert.strictEqual(isMatch("", "a*"), true);
  assert.strictEqual(isMatch("", ".*"), true);
  assert.strictEqual(isMatch("a", ""), false);
  assert.strictEqual(isMatch("a", "a"), true);
  assert.strictEqual(isMatch("a", "."), true);
  assert.strictEqual(isMatch("a", "b"), false);
  assert.strictEqual(isMatch("aa", "a"), false);
  assert.strictEqual(isMatch("aa", "a*"), true);
  assert.strictEqual(isMatch("aa", ".*"), true);
  assert.strictEqual(isMatch("ab", ".*"), true);
  assert.strictEqual(isMatch("aab", "c*a*b"), true);
  assert.strictEqual(isMatch("abc", "ab.*c"), true);
  assert.strictEqual(isMatch("abXYZc", "ab.*c"), true);
  assert.strictEqual(isMatch("ab", "ab.*c"), false);
  assert.strictEqual(isMatch("aaa", "a*a"), true);
  assert.strictEqual(isMatch("aaa", "ab*a"), false);
  assert.strictEqual(isMatch("mississippi", "mis*is*p*."), false);
  console.log("PASS");
}
