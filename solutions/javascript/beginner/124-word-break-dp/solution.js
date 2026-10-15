/** Task 124 - Word break (DP). (Tier: Beginner, CTCI Ch 8) */
"use strict";

function wordBreak(s, dictionary) {
  const words = new Set(dictionary);
  const dp = new Array(s.length + 1).fill(false);
  dp[0] = true;
  for (let i = 1; i <= s.length; i++) {
    for (let j = 0; j < i; j++) {
      if (dp[j] && words.has(s.slice(j, i))) {
        dp[i] = true;
        break;
      }
    }
  }
  return dp[s.length];
}

function wordBreakSegmentation(s, dictionary) {
  const words = new Set(dictionary);
  const dp = new Array(s.length + 1).fill(false);
  const prev = new Array(s.length + 1).fill(-1);
  dp[0] = true;
  for (let i = 1; i <= s.length; i++) {
    for (let j = 0; j < i; j++) {
      if (dp[j] && words.has(s.slice(j, i))) {
        dp[i] = true;
        prev[i] = j;
        break;
      }
    }
  }
  if (!dp[s.length]) return null;
  const parts = [];
  for (let i = s.length; i > 0; i = prev[i]) parts.unshift(s.slice(prev[i], i));
  return parts;
}

module.exports = { wordBreak, wordBreakSegmentation };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(wordBreak("leetcode", ["leet", "code"]), true);
  assert.strictEqual(wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"]), false);
  assert.strictEqual(wordBreak("applepenapple", ["apple", "pen"]), true);
  assert.deepStrictEqual(wordBreakSegmentation("leetcode", ["leet", "code"]), ["leet", "code"]);
  assert.strictEqual(wordBreakSegmentation("catsandog", ["cats", "dog"]), null);
  console.log("PASS");
}
